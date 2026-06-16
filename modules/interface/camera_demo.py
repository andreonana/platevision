"""
Module Interface — Démonstrateur Temps Réel PlateVision
MINT/DGI Cameroun — UCAC-ICAM / ULC-ICAM

Pipeline intégré A→B→C en temps réel sur flux caméra :
  1. Module A2 : YOLO détecte la plaque dans le flux vidéo
  2. Module A2 : EasyOCR lit le texte → signal d'alerte CNN
  3. Module B  : CNN extrait embedding 256D → K-Means → cluster_id + conf_level
  4. Module C  : MDP détermine l'état (cluster × conf × alerte) → π*(s) = action optimale

Usage :
  python main.py --demo                          # webcam par défaut (index 0)
  python main.py --demo --camera 1               # caméra index 1
  python main.py --demo --input video.mp4        # fichier vidéo
  python main.py --demo --input plaque.jpg       # image statique unique
  python main.py --demo --input dossier/images/  # galerie d'images (diaporama)

Contrôles pendant la démonstration :
  Q / ESC   : quitter
  S         : sauvegarder la frame courante dans reports/figures/
  ESPACE    : pause / reprendre (vidéo) | auto-avance on/off (galerie)
  I         : afficher / masquer le panneau d'information détaillé
  +/-       : ajuster le seuil de détection YOLO

Contrôles supplémentaires en mode galerie :
  N / →     : image suivante
  P / ←     : image précédente
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import torch.nn.functional as F

from modules.module_a.yolo_ocr_pipeline import deskew_plate_crop, read_plate_ocr
from data.prepare_datasets import _resize_plate, _segment_chars_from_plate

logger = logging.getLogger(__name__)

# ── Constantes couleurs BGR ───────────────────────────────────────────────────
_COLORS = {
    "LAISSER_PASSER":   (34, 177, 76),    # vert
    "CONTROLE_STANDARD": (0, 165, 255),   # orange
    "ARRET_SAISIE":     (0, 50, 255),     # rouge
    "SIGNALEMENT_DGI":  (255, 165, 0),    # bleu clair
    "TRANSFERT_PJ":     (128, 0, 128),    # violet
    "default":          (200, 200, 200),  # gris
}

_CONF_COLORS = {
    0: (34, 177, 76),    # haute  → vert
    1: (0, 165, 255),    # moyenne → orange
    2: (0, 50, 255),     # faible → rouge
}

# Seuils de distance au centroïde (espace normalisé) calculés sur données d'entraînement
# Module B kmeans_fit.py : tertiles de dist_centroid par cluster
_DIST_THRESHOLDS = {
    0: (8.8249, 9.7917),   # cluster 0 : q33, q66
    1: (5.9232, 7.1918),   # cluster 1
    2: (8.6483, 9.6677),   # cluster 2
}

# ── Panel layout ──────────────────────────────────────────────────────────────
_PANEL_W  = 380    # largeur du panneau d'info (pixels)
_FONT     = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SM  = 0.45
_FONT_MD  = 0.60
_FONT_LG  = 0.80
_THICK_N  = 1
_THICK_B  = 2


# ══════════════════════════════════════════════════════════════════════════════
# 1. MOTEUR D'INFÉRENCE
# ══════════════════════════════════════════════════════════════════════════════

class PlateVisionEngine:
    """
    Charge tous les modèles et expose classify_plate() pour le pipeline A→B→C.
    """

    def __init__(
        self,
        data_dir:  Path = Path("data/processed"),
        model_dir: Path = Path("models"),
        conf_yolo: float = 0.25,
        ocr_alert_threshold: float = 0.50,
    ) -> None:
        self.data_dir  = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.conf_yolo = conf_yolo
        self.ocr_alert_threshold = ocr_alert_threshold

        logger.info("Chargement des modèles PlateVision…")
        self._load_models()
        self._load_metadata()
        logger.info("Moteur prêt — %d états MDP, γ=0.95", len(self.mdp_states_list))

    # ── Chargement modèles ────────────────────────────────────────────────────

    def _load_models(self) -> None:
        from ultralytics import YOLO
        import easyocr

        yolo_path = self.model_dir / "weights" / "yolov8_platevision.pt"
        if not yolo_path.exists():
            raise FileNotFoundError(f"Poids YOLO introuvables : {yolo_path}")
        self.yolo = YOLO(str(yolo_path))
        logger.info("YOLO chargé : %s", yolo_path.name)

        self.ocr = easyocr.Reader(["fr", "en"], verbose=False)
        logger.info("EasyOCR initialisé")

        cnn_path = self.model_dir / "char_cnn.pth"
        if not cnn_path.exists():
            raise FileNotFoundError(f"Poids CNN introuvables : {cnn_path}")
        from modules.module_b.cnn_embeddings import CharEmbeddingCNN
        self.cnn = CharEmbeddingCNN()
        checkpoint = torch.load(str(cnn_path), map_location="cpu", weights_only=True)
        # Le checkpoint peut être un state_dict direct ou un dict avec "model_state_dict"
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.cnn.load_state_dict(state_dict)
        self.cnn.eval()
        logger.info("CNN CharEmbedding chargé")

        km_path  = self.model_dir / "kmeans_model.pkl"
        sc_path  = self.model_dir / "kmeans_scaler.pkl"
        if not km_path.exists() or not sc_path.exists():
            raise FileNotFoundError("Modèles K-Means introuvables dans models/")
        self.kmeans = joblib.load(km_path)
        self.scaler = joblib.load(sc_path)
        logger.info("K-Means chargé (k=%d)", self.kmeans.n_clusters)

    def _load_metadata(self) -> None:
        dp = self.data_dir

        self.pi_star = np.load(dp / "vi_pi_star.npy")
        self.V_star  = np.load(dp / "vi_V_star.npy")

        with open(dp / "vi_policy.json", encoding="utf-8") as f:
            vi_policy = json.load(f)
        self.policy_by_state: dict[int, dict] = {
            row["state_id"]: row for row in vi_policy
        }

        with open(dp / "mdp_states.json", encoding="utf-8") as f:
            mdp_json = json.load(f)
        self.mdp_states_list: list[dict] = mdp_json["states"]
        self._state_index: dict[tuple, dict] = {
            (s["cluster_id"], s["conf_level"], s["alerte_cnn"]): s
            for s in self.mdp_states_list
        }

        with open(dp / "mdp_actions.json", encoding="utf-8") as f:
            actions_json = json.load(f)
        self.actions: dict[int, dict] = {
            a["action_id"]: a for a in actions_json["actions"]
        }

        with open(dp / "cluster_mapping.json", encoding="utf-8") as f:
            cm = json.load(f)
        self.cluster_info: dict[int, dict] = {
            int(cid): info for cid, info in cm["clusters"].items()
        }

    # ── Inférence pipeline A→B→C ─────────────────────────────────────────────

    def embed_char(self, char_img_28: np.ndarray) -> np.ndarray:
        """
        Module B : extrait l'embedding 256D d'un caractère 28×28 déjà segmenté
        (même unité que l'entraînement K-Means — voir _segment_chars_from_plate).
        """
        norm   = char_img_28.astype(np.float32) / 255.0
        tensor = torch.tensor(norm[np.newaxis, np.newaxis]).float()  # (1,1,28,28)

        captured: list[np.ndarray] = []

        def _hook(module, inp, out):
            captured.append(F.relu(out).detach().cpu().numpy())

        handle = self.cnn.fc1.register_forward_hook(_hook)
        with torch.no_grad():
            self.cnn(tensor)
        handle.remove()

        return captured[0][0]  # (256,)

    def _dist_to_conf_level(self, dist: float, cluster_id: int) -> int:
        """Tertiles de distance → niveau de confiance 0=haute / 1=moy / 2=faible."""
        q33, q66 = _DIST_THRESHOLDS.get(cluster_id, (9.0, 10.5))
        if dist <= q33:
            return 0
        if dist <= q66:
            return 1
        return 2

    def classify_plate(self, plate_bgr: np.ndarray) -> dict:
        """
        Pipeline complet A→B→C sur une image de plaque.
        Retourne un dict avec OCR, cluster, état MDP, action optimale.
        """
        # ── A2 : OCR ─────────────────────────────────────────────────────────
        # Même chaîne que le pipeline officiel Module A : deskew + allowlist alphanumérique
        # + post-traitement (évite de lire le décor/texte parasite autour de la plaque)
        plate_deskewed = deskew_plate_crop(plate_bgr)
        ocr_res  = read_plate_ocr(plate_deskewed, reader=self.ocr)
        ocr_text = ocr_res["plate_text"] or "—"
        ocr_conf = ocr_res["confidence"]
        alerte_cnn = 1 if ocr_conf < self.ocr_alert_threshold else 0

        # ── B : segmentation caractères → CNN → K-Means → cluster + conf_level ──
        # Le K-Means a été entraîné sur des CARACTÈRES individuels 28×28 (cf. Module B,
        # segment_characters() dans data/prepare_datasets.py), pas sur la plaque entière.
        # On reproduit donc la même segmentation ici, puis on agrège le vote des
        # caractères détectés — sinon l'embedding d'une plaque entière écrasée en 28×28
        # est hors distribution et retombe quasi toujours dans le même cluster majoritaire.
        plate_for_seg = _resize_plate(plate_deskewed)
        chars = _segment_chars_from_plate(plate_for_seg)

        if chars:
            cluster_ids, dists = [], []
            for (_x, _y, _w, _h, char_img_28) in chars:
                embedding  = self.embed_char(char_img_28)
                emb_scaled = self.scaler.transform(embedding.reshape(1, -1))
                cid        = int(self.kmeans.predict(emb_scaled)[0])
                centroid   = self.kmeans.cluster_centers_[cid]
                cluster_ids.append(cid)
                dists.append(float(np.linalg.norm(emb_scaled[0] - centroid)))

            # Cluster majoritaire parmi les caractères détectés sur cette plaque
            cluster_id = int(np.bincount(cluster_ids).argmax())
            dist       = float(np.mean([d for d, c in zip(dists, cluster_ids) if c == cluster_id]))
            conf_level = self._dist_to_conf_level(dist, cluster_id)
        else:
            # Aucun caractère segmentable → plaque illisible : confiance faible + alerte,
            # on ne peut pas prétendre à une lecture fiable du cluster.
            cluster_id = 0
            dist       = 0.0
            conf_level = 2
            alerte_cnn = 1

        # ── C : MDP état → π* ────────────────────────────────────────────────
        key   = (cluster_id, conf_level, alerte_cnn)
        state = self._state_index.get(key)
        if state is None:
            key   = (cluster_id, conf_level, 0)
            state = self._state_index.get(key)
        if state is None:
            key   = (cluster_id, 0, 0)
            state = self._state_index.get(key, self.mdp_states_list[0])

        state_id  = state["state_id"]
        action_id = int(self.pi_star[state_id])
        action    = self.actions[action_id]
        v_star    = float(self.V_star[state_id])
        cluster_nm = self.cluster_info[cluster_id]["label"]

        return {
            "ocr_text":        ocr_text or "—",
            "ocr_conf":        ocr_conf,
            "alerte_cnn":      alerte_cnn,
            "cluster_id":      cluster_id,
            "cluster_name":    cluster_nm,
            "conf_level":      conf_level,
            "conf_label":      ["haute", "moyenne", "faible"][conf_level],
            "dist_centroid":   dist,
            "state_id":        state_id,
            "state_name":      state["label"],
            "action_id":       action_id,
            "action_code":     action["code"],
            "action_label":    action["label"],
            "action_procedure":action["procedure"],
            "V_star":          v_star,
        }

    def set_yolo_conf(self, delta: float) -> None:
        self.conf_yolo = float(np.clip(self.conf_yolo + delta, 0.05, 0.95))

    def detect_plates(self, frame: np.ndarray) -> list[dict]:
        """
        Module A2 : YOLO détecte les plaques dans un frame BGR.
        Retourne liste de {bbox: (x1,y1,x2,y2), yolo_conf: float, crop: ndarray}.
        """
        results = self.yolo.predict(
            frame,
            conf=self.conf_yolo,
            verbose=False,
            device="cpu",
        )
        detections: list[dict] = []
        h, w = frame.shape[:2]
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if (x2 - x1) < 20 or (y2 - y1) < 8:
                    continue
                crop = frame[y1:y2, x1:x2]
                detections.append({
                    "bbox":      (x1, y1, x2, y2),
                    "yolo_conf": float(box.conf[0]),
                    "crop":      crop,
                })
        return detections


# ══════════════════════════════════════════════════════════════════════════════
# 2. RENDU VISUEL
# ══════════════════════════════════════════════════════════════════════════════

def _put_text(img, text, pos, scale, color, thick=_THICK_N, bg=True):
    """Affiche du texte avec fond semi-transparent optionnel."""
    (tw, th), bl = cv2.getTextSize(text, _FONT, scale, thick)
    x, y = pos
    if bg:
        cv2.rectangle(img, (x - 2, y - th - 2), (x + tw + 2, y + bl + 2),
                      (20, 20, 20), -1)
    cv2.putText(img, text, (x, y), _FONT, scale, color, thick, cv2.LINE_AA)


def _draw_detection(frame: np.ndarray, det: dict, result: dict) -> None:
    """Dessine la bbox et les informations de classification sur le frame."""
    x1, y1, x2, y2 = det["bbox"]
    action_code = result["action_code"]
    color = _COLORS.get(action_code, _COLORS["default"])
    conf_color = _CONF_COLORS.get(result["conf_level"], (200, 200, 200))

    # Rectangle plaque
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, _THICK_B)

    # Texte OCR + conf YOLO au-dessus
    yolo_pct = int(det["yolo_conf"] * 100)
    header = f"OCR: {result['ocr_text']}  YOLO:{yolo_pct}%"
    _put_text(frame, header, (x1, max(y1 - 8, 12)), _FONT_SM, (255, 255, 255))

    # Cluster sous la bbox
    cluster_txt = f"Cluster {result['cluster_id']}: {result['cluster_name']}"
    _put_text(frame, cluster_txt, (x1, y2 + 16), _FONT_SM, color)

    # Action (ligne en dessous)
    action_short = result["action_label"][:35]
    _put_text(frame, action_short, (x1, y2 + 34), _FONT_SM, color)

    # Pastille confiance
    conf_label = f"Conf: {result['conf_label']}"
    _put_text(frame, conf_label, (x1, y2 + 52), _FONT_SM, conf_color)

    # Alerte CNN
    if result["alerte_cnn"]:
        _put_text(frame, "! ALERTE OCR", (x2 - 100, y1 + 16), _FONT_SM,
                  (0, 50, 255), thick=_THICK_B)


def _draw_panel(canvas: np.ndarray, result: dict | None, fps: float,
                conf_yolo: float, paused: bool, show_detail: bool,
                gallery_info: dict | None = None) -> None:
    """
    Dessine le panneau d'information latéral (droite du canvas).
    canvas shape : (H, W_frame + _PANEL_W, 3)
    gallery_info : {'idx': int, 'total': int, 'filename': str} en mode galerie
    """
    h, full_w = canvas.shape[:2]
    panel_x = full_w - _PANEL_W

    # Fond panneau
    cv2.rectangle(canvas, (panel_x, 0), (full_w, h), (25, 25, 35), -1)
    cv2.line(canvas, (panel_x, 0), (panel_x, h), (80, 80, 80), 1)

    y = 22
    step = 22

    # ── En-tête ───────────────────────────────────────────────────────────────
    cv2.putText(canvas, "PlateVision", (panel_x + 8, y),
                _FONT, _FONT_LG, (255, 200, 0), _THICK_B, cv2.LINE_AA)
    y += 26
    cv2.putText(canvas, "MINT / DGI Cameroun", (panel_x + 8, y),
                _FONT, _FONT_SM, (180, 180, 180), _THICK_N, cv2.LINE_AA)
    y += 20
    cv2.line(canvas, (panel_x + 5, y), (full_w - 5, y), (80, 80, 80), 1)
    y += step

    # ── Statut système ────────────────────────────────────────────────────────
    if gallery_info:
        status = f"IMG {gallery_info['idx'] + 1}/{gallery_info['total']}"
        st_col = (0, 165, 255) if paused else (34, 177, 76)
    else:
        status = "PAUSE" if paused else "EN DIRECT"
        st_col = (0, 165, 255) if paused else (34, 177, 76)
    cv2.putText(canvas, status, (panel_x + 8, y),
                _FONT, _FONT_MD, st_col, _THICK_B, cv2.LINE_AA)
    if not gallery_info:
        fps_txt = f"  {fps:.1f} FPS"
        cv2.putText(canvas, fps_txt, (panel_x + 100, y),
                    _FONT, _FONT_SM, (180, 180, 180), _THICK_N, cv2.LINE_AA)
    y += step

    if gallery_info:
        fname = gallery_info["filename"][:28]
        cv2.putText(canvas, fname, (panel_x + 8, y),
                    _FONT, _FONT_SM - 0.05, (160, 160, 160), _THICK_N, cv2.LINE_AA)
        y += step - 4
        auto_txt = "Auto: OFF (ESPACE)" if paused else "Auto: ON  (ESPACE)"
        auto_col = (100, 100, 100) if paused else (80, 180, 80)
        cv2.putText(canvas, auto_txt, (panel_x + 8, y),
                    _FONT, _FONT_SM - 0.05, auto_col, _THICK_N, cv2.LINE_AA)
        y += step - 4

    conf_txt = f"Seuil YOLO : {conf_yolo:.2f}  (+/- pour ajuster)"
    cv2.putText(canvas, conf_txt, (panel_x + 8, y),
                _FONT, _FONT_SM - 0.05, (150, 150, 150), _THICK_N, cv2.LINE_AA)
    y += 6
    cv2.line(canvas, (panel_x + 5, y), (full_w - 5, y), (60, 60, 60), 1)
    y += step

    if result is None:
        cv2.putText(canvas, "Aucune plaque", (panel_x + 8, y),
                    _FONT, _FONT_SM, (100, 100, 100), _THICK_N, cv2.LINE_AA)
        y += step
        cv2.putText(canvas, "detectee", (panel_x + 8, y),
                    _FONT, _FONT_SM, (100, 100, 100), _THICK_N, cv2.LINE_AA)
    else:
        _draw_panel_result(canvas, result, panel_x, y, full_w, show_detail)

    # ── Commandes ─────────────────────────────────────────────────────────────
    if gallery_info:
        keys = [
            ("Q/ESC",  "Quitter"),
            ("N / →",  "Suivante"),
            ("P / ←",  "Precedente"),
            ("ESPACE", "Auto on/off"),
            ("S",      "Sauvegarder"),
            ("I",      "Detail on/off"),
            ("+/-",    "Seuil YOLO"),
        ]
    else:
        keys = [
            ("Q/ESC", "Quitter"),
            ("ESPACE", "Pause"),
            ("S",     "Sauvegarder"),
            ("I",     "Detail on/off"),
            ("+/-",   "Seuil YOLO"),
        ]
    y_k = h - len(keys) * 18 - 10
    cv2.line(canvas, (panel_x + 5, y_k - 8), (full_w - 5, y_k - 8), (60, 60, 60), 1)
    for key, desc in keys:
        cv2.putText(canvas, f"{key:<8} {desc}", (panel_x + 8, y_k),
                    _FONT, _FONT_SM - 0.05, (120, 120, 120), _THICK_N, cv2.LINE_AA)
        y_k += 18


def _draw_panel_result(canvas, result, panel_x, y, full_w, show_detail):
    """Affiche les détails du résultat de classification dans le panneau."""
    step = 22

    def _row(label, value, col=(220, 220, 220)):
        nonlocal y
        cv2.putText(canvas, label, (panel_x + 8, y),
                    _FONT, _FONT_SM - 0.05, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(value), (panel_x + 120, y),
                    _FONT, _FONT_SM - 0.05, col, 1, cv2.LINE_AA)
        y += step

    # ── Résultat OCR ─────────────────────────────────────────────────────────
    cv2.putText(canvas, "Module A2 — OCR", (panel_x + 8, y),
                _FONT, _FONT_SM, (255, 200, 0), _THICK_N, cv2.LINE_AA)
    y += step
    _row("Texte:",    result["ocr_text"][:16])
    ocr_pct = int(result["ocr_conf"] * 100)
    ocr_col = (34, 177, 76) if ocr_pct >= 50 else (0, 50, 255)
    _row("Confiance:", f"{ocr_pct}%", ocr_col)
    alerte_col = (0, 50, 255) if result["alerte_cnn"] else (34, 177, 76)
    _row("Alerte CNN:", "OUI" if result["alerte_cnn"] else "NON", alerte_col)
    y += 6
    cv2.line(canvas, (panel_x + 5, y), (full_w - 5, y), (60, 60, 60), 1)
    y += 12

    # ── Module B ─────────────────────────────────────────────────────────────
    cv2.putText(canvas, "Module B — Clustering", (panel_x + 8, y),
                _FONT, _FONT_SM, (255, 200, 0), _THICK_N, cv2.LINE_AA)
    y += step
    c_col = _COLORS.get(result["action_code"], _COLORS["default"])
    _row("Cluster:", f"{result['cluster_id']} – {result['cluster_name'][:16]}", c_col)
    conf_col = _CONF_COLORS[result["conf_level"]]
    _row("Confiance:", result["conf_label"], conf_col)
    if show_detail:
        _row("Dist. centr.:", f"{result['dist_centroid']:.3f}")
    y += 6
    cv2.line(canvas, (panel_x + 5, y), (full_w - 5, y), (60, 60, 60), 1)
    y += 12

    # ── Module C ─────────────────────────────────────────────────────────────
    cv2.putText(canvas, "Module C — MDP", (panel_x + 8, y),
                _FONT, _FONT_SM, (255, 200, 0), _THICK_N, cv2.LINE_AA)
    y += step
    _row("Etat:",     f"S{result['state_id']}")
    if show_detail:
        state_short = result["state_name"].split("—")[-1].strip()[:20]
        _row("", state_short, (180, 180, 180))

    action_col = _COLORS.get(result["action_code"], _COLORS["default"])
    action_short = result["action_label"][:20]
    _row("Action pi*:", action_short, action_col)

    v = result["V_star"]
    v_col = (34, 177, 76) if v >= 0 else (0, 50, 255)
    v_str = f"{v:+,.0f} FCFA"
    _row("V*(s):", v_str, v_col)

    if show_detail:
        y += 6
        cv2.line(canvas, (panel_x + 5, y), (full_w - 5, y), (60, 60, 60), 1)
        y += 10
        cv2.putText(canvas, "Procedure:", (panel_x + 8, y),
                    _FONT, _FONT_SM - 0.05, (150, 150, 150), 1, cv2.LINE_AA)
        y += step - 4
        # Wrap procedure text
        proc = result["action_procedure"]
        words = proc.split()
        line, max_chars = "", 32
        for w in words:
            if len(line) + len(w) + 1 > max_chars:
                cv2.putText(canvas, line, (panel_x + 8, y),
                            _FONT, _FONT_SM - 0.08, (180, 180, 180), 1, cv2.LINE_AA)
                y += 16
                line = w
            else:
                line += (" " if line else "") + w
        if line:
            cv2.putText(canvas, line, (panel_x + 8, y),
                        _FONT, _FONT_SM - 0.08, (180, 180, 180), 1, cv2.LINE_AA)


def _overlay_timestamp(frame: np.ndarray) -> None:
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, ts, (8, frame.shape[0] - 8),
                _FONT, _FONT_SM - 0.05, (200, 200, 200), 1, cv2.LINE_AA)


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".JPG", ".JPEG", ".PNG"}


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODE GALERIE (diaporama d'images)
# ══════════════════════════════════════════════════════════════════════════════

def run_gallery_demo(
    image_dir:   Path,
    data_dir:    Path  = Path("data/processed"),
    model_dir:   Path  = Path("models"),
    figures_dir: Path  = Path("reports/figures"),
    conf_yolo:   float = 0.25,
    auto_delay:  int   = 3000,
) -> None:
    """
    Diaporama : parcourt toutes les images d'un dossier et applique le pipeline A→B→C.

    Args:
        image_dir  : dossier contenant les images
        auto_delay : délai auto-avance en ms (mode auto, touche ESPACE)
    """
    image_dir   = Path(image_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in image_dir.iterdir() if p.suffix in _IMG_EXTS)
    if not images:
        raise FileNotFoundError(f"Aucune image trouvée dans : {image_dir}")

    engine = PlateVisionEngine(data_dir=data_dir, model_dir=model_dir, conf_yolo=conf_yolo)

    idx         = 0
    auto_mode   = False   # ESPACE bascule l'avance automatique
    show_detail = True
    last_result: dict | None = None
    last_dets:   list[dict]  = []
    needs_refresh = True

    window_name = "PlateVision — Galerie MINT/DGI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1060, 520)

    print("\n" + "═" * 60)
    print("  PlateVision — Mode Galerie")
    print(f"  {len(images)} image(s) dans : {image_dir}")
    print("  MINT / DGI Cameroun — UCAC-ICAM / ULC-ICAM")
    print("═" * 60)
    print("  Commandes : Q=quitter  N/→=suivante  P/←=précédente")
    print("              ESPACE=auto  S=screenshot  I=détails  +/-=YOLO")
    print("═" * 60 + "\n")

    while True:
        # ── Chargement et inférence si nécessaire ────────────────────────────
        if needs_refresh:
            img_path = images[idx]
            frame_orig = cv2.imread(str(img_path))
            if frame_orig is None:
                logger.warning("Image illisible : %s", img_path.name)
                idx = (idx + 1) % len(images)
                continue

            detections = engine.detect_plates(frame_orig)
            last_dets  = detections
            if detections:
                best = max(detections, key=lambda d: d["yolo_conf"])
                last_result = engine.classify_plate(best["crop"])
                logger.info(
                    "[%d/%d] %s | OCR='%s' | Cluster=%d | Action=%s | V*=%.0f FCFA",
                    idx + 1, len(images), img_path.name,
                    last_result["ocr_text"],
                    last_result["cluster_id"],
                    last_result["action_code"],
                    last_result["V_star"],
                )
            else:
                last_result = None
                logger.info("[%d/%d] %s — aucune plaque détectée", idx + 1, len(images), img_path.name)
            needs_refresh = False

        # ── Rendu ────────────────────────────────────────────────────────────
        frame = frame_orig.copy()
        for det in last_dets:
            if last_result is not None:
                _draw_detection(frame, det, last_result)
        _overlay_timestamp(frame)

        h, w = frame.shape[:2]
        canvas = np.zeros((h, w + _PANEL_W, 3), np.uint8)
        canvas[:, :w] = frame
        gallery_info = {
            "idx":      idx,
            "total":    len(images),
            "filename": images[idx].name,
        }
        _draw_panel(canvas, last_result, 0.0, engine.conf_yolo,
                    paused=not auto_mode, show_detail=show_detail,
                    gallery_info=gallery_info)
        cv2.imshow(window_name, canvas)

        # ── Gestion touches ───────────────────────────────────────────────────
        delay = auto_delay if auto_mode else 0
        key   = cv2.waitKey(delay) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("n"), ord("N"), 83):   # N ou flèche droite
            idx = (idx + 1) % len(images)
            needs_refresh = True
        elif key in (ord("p"), ord("P"), 81):   # P ou flèche gauche
            idx = (idx - 1) % len(images)
            needs_refresh = True
        elif key == ord(" "):
            auto_mode = not auto_mode
            logger.info("Mode auto : %s", "ON" if auto_mode else "OFF")
        elif key in (ord("s"), ord("S")):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = figures_dir / f"platevision_galerie_{ts}.png"
            cv2.imwrite(str(path), canvas)
            logger.info("Screenshot → %s", path)
            print(f"  Screenshot → {path}")
        elif key in (ord("i"), ord("I")):
            show_detail = not show_detail
        elif key == ord("+"):
            engine.set_yolo_conf(+0.05)
            needs_refresh = True
        elif key == ord("-"):
            engine.set_yolo_conf(-0.05)
            needs_refresh = True
        elif auto_mode and key == 255:
            # délai auto-avance expiré → image suivante
            idx = (idx + 1) % len(images)
            needs_refresh = True

    cv2.destroyAllWindows()
    print("\nGalerie terminée.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. BOUCLE PRINCIPALE (caméra / vidéo / image statique)
# ══════════════════════════════════════════════════════════════════════════════

def run_camera_demo(
    source: int | str = 0,
    data_dir:  Path  = Path("data/processed"),
    model_dir: Path  = Path("models"),
    figures_dir: Path = Path("reports/figures"),
    conf_yolo: float = 0.25,
) -> None:
    """
    Lance la démonstration temps réel PlateVision.

    Args:
        source      : index caméra (int) ou chemin fichier vidéo/image (str)
        data_dir    : répertoire data/processed/ contenant les .npy/.json MDP
        model_dir   : répertoire models/ contenant les poids (YOLO, CNN, KMeans)
        figures_dir : répertoire de sauvegarde des screenshots (S)
        conf_yolo   : seuil de confiance YOLO initial
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Chargement moteur ─────────────────────────────────────────────────────
    engine = PlateVisionEngine(
        data_dir=data_dir,
        model_dir=model_dir,
        conf_yolo=conf_yolo,
    )

    # ── Source vidéo / image ─────────────────────────────────────────────────
    static_image = False
    if isinstance(source, str) and Path(source).suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff"
    }:
        static_image = True
        frame_orig = cv2.imread(source)
        if frame_orig is None:
            raise FileNotFoundError(f"Image introuvable : {source}")
        cap = None
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la source : {source}")

    # ── État UI ───────────────────────────────────────────────────────────────
    paused       = False
    show_detail  = True
    last_result: dict | None = None
    last_dets:   list[dict]  = []
    fps_history: list[float] = []
    frame_count  = 0

    window_name = "PlateVision — Demo MINT/DGI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1060, 520)

    print("\n" + "═" * 60)
    print("  PlateVision — Démonstrateur Temps Réel")
    print("  MINT / DGI Cameroun — UCAC-ICAM / ULC-ICAM")
    print("═" * 60)
    print("  Commandes : Q=quitter  ESPACE=pause  S=screenshot")
    print("              I=détails  +/-=seuil YOLO")
    print("═" * 60 + "\n")

    t_prev = time.perf_counter()

    while True:
        # ── Acquisition frame ─────────────────────────────────────────────────
        if static_image:
            frame = frame_orig.copy()
        elif paused:
            frame = last_frame.copy() if "last_frame" in dir() else np.zeros((480, 640, 3), np.uint8)
        else:
            ret, frame = cap.read()
            if not ret:
                logger.info("Fin du flux vidéo.")
                break
            last_frame = frame.copy()

        frame_count += 1

        # ── Inférence (sauf pause) ────────────────────────────────────────────
        if not paused or frame_count == 1:
            detections = engine.detect_plates(frame)
            last_dets = detections

            if detections:
                # Traite la détection la plus confiante (pas la plus grande bbox,
                # qui peut englober du bruit autour de la plaque)
                best = max(detections, key=lambda d: d["yolo_conf"])
                last_result = engine.classify_plate(best["crop"])

                # Log console
                logger.info(
                    "[Frame %d] OCR='%s' | Cluster=%d | Action=%s | V*=%.0f FCFA",
                    frame_count,
                    last_result["ocr_text"],
                    last_result["cluster_id"],
                    last_result["action_code"],
                    last_result["V_star"],
                )
            else:
                last_result = None

        # ── Rendu frame ───────────────────────────────────────────────────────
        for det in last_dets:
            if last_result is not None:
                _draw_detection(frame, det, last_result)

        _overlay_timestamp(frame)

        # ── Calcul FPS ────────────────────────────────────────────────────────
        t_now = time.perf_counter()
        fps = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = float(np.mean(fps_history))

        # ── Canvas final (frame + panneau) ────────────────────────────────────
        h, w = frame.shape[:2]
        canvas = np.zeros((h, w + _PANEL_W, 3), np.uint8)
        canvas[:, :w] = frame
        _draw_panel(canvas, last_result, avg_fps,
                    engine.conf_yolo, paused, show_detail)

        cv2.imshow(window_name, canvas)

        # ── Gestion touches ───────────────────────────────────────────────────
        delay = 0 if (static_image or paused) else 1
        key   = cv2.waitKey(delay) & 0xFF

        if key in (ord("q"), ord("Q"), 27):   # Q ou ESC
            break
        elif key == ord(" "):
            paused = not paused
            logger.info("Pause : %s", paused)
        elif key in (ord("s"), ord("S")):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = figures_dir / f"platevision_demo_{ts}.png"
            cv2.imwrite(str(path), canvas)
            logger.info("Screenshot sauvegardé : %s", path)
            print(f"  Screenshot → {path}")
        elif key in (ord("i"), ord("I")):
            show_detail = not show_detail
        elif key == ord("+"):
            engine.set_yolo_conf(+0.05)
            logger.info("Seuil YOLO : %.2f", engine.conf_yolo)
        elif key == ord("-"):
            engine.set_yolo_conf(-0.05)
            logger.info("Seuil YOLO : %.2f", engine.conf_yolo)

        if static_image and key == 255:
            # Image statique : on attend la touche sans boucle active
            continue

    # ── Nettoyage ─────────────────────────────────────────────────────────────
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("\nDémo terminée.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. ENTRÉE DIRECTE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="PlateVision — Démonstrateur Temps Réel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python camera_demo.py                          # webcam\n"
            "  python camera_demo.py --input video.mp4        # vidéo\n"
            "  python camera_demo.py --input plaque.jpg       # image unique\n"
            "  python camera_demo.py --input dossier/images/  # galerie\n"
            "  python camera_demo.py --input dossier/ --delay 2000  # galerie 2s/img\n"
        ),
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Index de la caméra (défaut: 0)"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Fichier vidéo, image unique, ou dossier d'images (galerie)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Seuil de confiance YOLO initial (défaut: 0.25)"
    )
    parser.add_argument(
        "--delay", type=int, default=3000,
        help="Délai auto-avance en mode galerie, millisecondes (défaut: 3000)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/processed",
        help="Répertoire data/processed/"
    )
    parser.add_argument(
        "--model-dir", type=str, default="models",
        help="Répertoire models/"
    )
    parser.add_argument(
        "--figures-dir", type=str, default="reports/figures",
        help="Répertoire de sauvegarde screenshots"
    )
    args = parser.parse_args()

    common = dict(
        data_dir    = Path(args.data_dir),
        model_dir   = Path(args.model_dir),
        figures_dir = Path(args.figures_dir),
        conf_yolo   = args.conf,
    )

    if args.input and Path(args.input).is_dir():
        run_gallery_demo(image_dir=Path(args.input), auto_delay=args.delay, **common)
    else:
        source = args.input if args.input else args.camera
        run_camera_demo(source=source, **common)


if __name__ == "__main__":
    import logging as _lg
    _lg.basicConfig(level=_lg.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
