"""
Module A2 — Pipeline YOLO + OCR (Partie 1 : Détection YOLOv8n)
PlateVision / MINT-DGI Cameroun

Justification architecture (cahier des charges §4.1 A2) :
  YOLOv8n (nano) choisi pour :
    - Vitesse d'inférence < 20ms/image sur GPU entrée de gamme
      → compatible contrainte temps réel postes de contrôle MINT
    - Transfer learning depuis COCO (80 classes objets) :
      les poids pré-entraînés capturent formes, bords et rectangles
      → plaque d'immatriculation = rectangle texturé, bien représenté
    - mAP SOTA sur datasets véhicules avec modèle 3.2M paramètres seulement
    - Alternative Detectron2 écartée : trop lourd pour déploiement embarqué

Comparaison NB vs YOLO+OCR (obligatoire §4.1) :
  Naïves Bayes (A1) : classification de caractères PRÉ-SEGMENTÉS (28×28)
    → suppose que la plaque est déjà trouvée et les caractères isolés
    → inutilisable en conditions réelles (pas de détection dans l'image globale)
  YOLO (A2) : détection bout-en-bout dans l'image brute
    → localise la plaque sans segmentation préalable
    → robuste aux variations d'angle, d'échelle, d'occultation partielle
"""

from pathlib import Path
import shutil
import json
import time
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
YAML_PATH   = Path("models/configs/yolov8_platevision.yaml")
WEIGHTS_DIR = Path("models/weights")
RUNS_DIR    = Path("runs/detect")
REPORT_DIR  = Path("reports/rapport_technique/figures")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PRÉPARATION DU YAML DE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_yaml(yaml_path: Path = YAML_PATH) -> Path:
    """
    Vérifie que le YAML de configuration existe et est valide.
    Si absent, le recrée avec la structure correcte (chemin absolu).
    Retourne le Path vers le fichier YAML.
    """
    import yaml  # pyyaml

    yaml_path = Path(yaml_path)

    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # S'assurer que path est absolu
        if not Path(str(cfg.get("path", ""))).is_absolute():
            cfg["path"] = str(Path.cwd() / "data/raw")
            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
            logger.info("YAML mis à jour avec chemin absolu.")
    else:
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        # Pointe directement sur le dataset Roboflow brut (scènes complètes + bbox
        # réelles), pas sur data/processed/images/* — voir prepare_yaml docstring
        # et le commentaire en tête de models/configs/yolov8_platevision.yaml.
        cfg = {
            "path":  str(Path.cwd() / "data/raw/roboflow/Nigerian License Plate.v1i.yolov5pytorch"),
            "train": "train/images",
            "val":   "valid/images",
            "test":  "test/images",
            "nc":    1,
            "names": {0: "plaque_immatriculation"},
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        logger.info("YAML créé : %s", yaml_path)

    # Affichage de vérification
    with open(yaml_path, "r", encoding="utf-8") as f:
        content = f.read()
    print("─" * 60)
    print(f"[prepare_yaml] Contenu de {yaml_path} :")
    print(content)
    print("─" * 60)

    return yaml_path


# ══════════════════════════════════════════════════════════════════════════════
# 2. ENTRAÎNEMENT YOLOV8n
# ══════════════════════════════════════════════════════════════════════════════

def train_yolo(yaml_path: Path = YAML_PATH,
               epochs: int = 50,
               imgsz: int = 640,
               batch: int = 16,
               device: str = "auto") -> Path:
    """
    Lance l'entraînement YOLOv8n avec transfer learning depuis COCO.

    Justification hyperparamètres :
      epochs=50  : suffisant pour fine-tuning sur dataset < 5000 images
                   (convergence observée entre 30-40 epochs sur datasets ALPR)
      imgsz=640  : résolution standard YOLOv8, compatible contrainte temps réel
      batch=16   : compromis mémoire GPU / stabilité des gradients
      device     : "auto" → GPU si disponible (Colab/Kaggle), sinon CPU

    Retourne : Path vers models/weights/yolov8_platevision.pt
    Lève FileNotFoundError si best.pt introuvable après entraînement.
    """
    from ultralytics import YOLO

    yaml_path = Path(yaml_path)
    logger.info("Démarrage entraînement YOLOv8n — epochs=%d, batch=%d, device=%s",
                epochs, batch, device)

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(yaml_path.absolute()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(RUNS_DIR),
        name="platevision",
        exist_ok=True,
        verbose=True,
    )

    best_pt = RUNS_DIR / "platevision" / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt introuvable après entraînement : {best_pt}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = WEIGHTS_DIR / "yolov8_platevision.pt"
    shutil.copy2(best_pt, dest)
    logger.info("Poids sauvegardés → %s", dest)
    print(f"[train_yolo] Poids finaux : {dest}")

    return dest


# ══════════════════════════════════════════════════════════════════════════════
# 3. ÉVALUATION — MÉTRIQUES OBLIGATOIRES CAHIER DES CHARGES
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_yolo(weights_path: Path = None,
                  yaml_path: Path = YAML_PATH,
                  report_dir: Path = REPORT_DIR) -> dict:
    """
    Évalue le modèle sur le jeu de test.
    Calcule les métriques obligatoires §4.1 A2 + temps d'inférence.
    Sauvegarde dans report_dir/yolo_metrics.json.
    """
    from ultralytics import YOLO

    weights_path = Path(weights_path) if weights_path else WEIGHTS_DIR / "yolov8_platevision.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Poids introuvables : {weights_path}")

    model = YOLO(str(weights_path))

    # ── Métriques mAP ──────────────────────────────────────────────────────
    val_results = model.val(data=str(Path(yaml_path).absolute()), split="test")
    map50     = float(val_results.box.map50)
    map50_95  = float(val_results.box.map)
    precision = float(val_results.box.mp)
    recall    = float(val_results.box.mr)

    # ── Temps d'inférence sur 10 images de test ────────────────────────────
    test_dir = Path(yaml_path).parent.parent.parent / "data/processed/yolo/images/test"
    test_images = sorted(test_dir.glob("*.jpg"))[:10] if test_dir.exists() else []

    inference_times = []
    for img_path in test_images:
        t0 = time.perf_counter()
        model.predict(str(img_path), conf=0.25, verbose=False)
        inference_times.append((time.perf_counter() - t0) * 1000)

    # Fallback : mesure sur image synthétique si pas d'images de test
    if not inference_times:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(10):
            t0 = time.perf_counter()
            model.predict(dummy, conf=0.25, verbose=False)
            inference_times.append((time.perf_counter() - t0) * 1000)

    inf_mean = float(np.mean(inference_times))
    inf_std  = float(np.std(inference_times))

    metrics = {
        "map50":             map50,
        "map50_95":          map50_95,
        "precision":         precision,
        "recall":            recall,
        "inference_ms_mean": inf_mean,
        "inference_ms_std":  inf_std,
        "realtime_ok":       inf_mean < 100.0,
    }

    # ── Sauvegarde JSON ────────────────────────────────────────────────────
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "yolo_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ── Affichage terminal ─────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  MÉTRIQUES YOLOv8n — PlateVision MINT/DGI")
    print("═" * 55)
    print(f"  mAP@0.5          : {map50:.4f}")
    print(f"  mAP@0.5:0.95     : {map50_95:.4f}")
    print(f"  Précision        : {precision:.4f}")
    print(f"  Rappel           : {recall:.4f}")
    print(f"  Inférence moy.   : {inf_mean:.1f} ms  (σ={inf_std:.1f} ms)")
    print(f"  Temps réel OK    : {'OUI' if metrics['realtime_ok'] else 'NON'}")
    print("═" * 55 + "\n")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 4. VISUALISATION DES COURBES D'ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(run_dir: Path = None,
                         report_dir: Path = REPORT_DIR) -> None:
    """
    Relit results.csv généré par Ultralytics et produit une figure 2×2.
    Sauvegarde dans report_dir/yolo_training_curves.png (dpi=150).
    Si results.csv absent : log warning, retourne sans exception.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    run_dir  = Path(run_dir) if run_dir else RUNS_DIR / "platevision"
    csv_path = run_dir / "results.csv"

    if not csv_path.exists():
        logger.warning("results.csv introuvable : %s — courbes non générées.", csv_path)
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    col_map = {
        "train_box": "train/box_loss",
        "val_box":   "val/box_loss",
        "train_cls": "train/cls_loss",
        "val_cls":   "val/cls_loss",
        "map50":     "metrics/mAP50(B)",
        "map50_95":  "metrics/mAP50-95(B)",
    }

    epochs     = range(1, len(df) + 1)
    best_epoch = (df[col_map["map50"]].idxmax() + 1
                  if col_map["map50"] in df.columns else None)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Courbes d'entraînement YOLOv8n — PlateVision MINT/DGI", fontsize=14)

    def _plot(ax, train_col, val_col, title):
        if train_col in df.columns:
            ax.plot(epochs, df[train_col], color="blue",   label="train", linewidth=1.5)
        if val_col in df.columns:
            ax.plot(epochs, df[val_col],   color="orange", label="val",
                    linestyle="--", linewidth=1.5)
        if best_epoch:
            ax.axvline(best_epoch, color="green", linestyle=":", alpha=0.7,
                       label=f"best ep.{best_epoch}")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], col_map["train_box"], col_map["val_box"], "Box Loss")
    _plot(axes[0, 1], col_map["train_cls"], col_map["val_cls"], "Cls Loss")

    if col_map["map50"] in df.columns:
        axes[1, 0].plot(epochs, df[col_map["map50"]], color="blue", linewidth=1.5)
        if best_epoch:
            axes[1, 0].axvline(best_epoch, color="green", linestyle=":", alpha=0.7)
    axes[1, 0].set_title("mAP@0.5")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True, alpha=0.3)

    if col_map["map50_95"] in df.columns:
        axes[1, 1].plot(epochs, df[col_map["map50_95"]], color="blue", linewidth=1.5)
        if best_epoch:
            axes[1, 1].axvline(best_epoch, color="green", linestyle=":", alpha=0.7)
    axes[1, 1].set_title("mAP@0.5:0.95")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / "yolo_training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Courbes sauvegardées → %s", out)
    print(f"[plot_training_curves] Courbes → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. INFÉRENCE SUR IMAGE UNIQUE (démo jury §5)
# ══════════════════════════════════════════════════════════════════════════════

def detect_plate(image_path: str,
                 weights_path: Path = None,
                 conf_threshold: float = 0.45,
                 save_annotated: bool = True) -> dict:
    """
    Détecte la plaque dans une image et retourne le crop localisé.
    Utilisé pour la démonstration live devant le jury MINT/DGI (§5).
    """
    from ultralytics import YOLO

    weights_path = Path(weights_path) if weights_path else WEIGHTS_DIR / "yolov8_platevision.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Poids introuvables : {weights_path}")

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    model = YOLO(str(weights_path))

    t0 = time.perf_counter()
    results = model.predict(str(image_path), conf=conf_threshold, verbose=False)
    inference_ms = (time.perf_counter() - t0) * 1000

    h, w = img_bgr.shape[:2]
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        conf = float(box.conf[0])

        # Padding 5%
        pad_x = int((x2 - x1) * 0.05)
        pad_y = int((y2 - y1) * 0.05)
        x1c = max(0, x1 - pad_x)
        y1c = max(0, y1 - pad_y)
        x2c = min(w, x2 + pad_x)
        y2c = min(h, y2 + pad_y)

        crop = img_bgr[y1c:y2c, x1c:x2c].copy()
        detections.append({
            "bbox":       [x1, y1, x2, y2],
            "confidence": conf,
            "crop":       crop,
        })

    if save_annotated:
        annotated = results[0].plot()
        out_path = REPORT_DIR / "detection_demo.png"
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"[detect_plate] Image annotée → {out_path}")

    result = {
        "n_detections": len(detections),
        "detections":   detections,
        "inference_ms": inference_ms,
    }
    print(f"[detect_plate] {len(detections)} plaque(s) détectée(s) en {inference_ms:.1f} ms")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 6. PIPELINE COMPLET PARTIE 1
# ══════════════════════════════════════════════════════════════════════════════

def run_yolo_training_pipeline(epochs: int = 50,
                               batch: int = 16,
                               device: str = "auto") -> dict:
    """
    Orchestre : prepare_yaml → train_yolo → evaluate_yolo → plot_training_curves.
    Affiche un résumé final. Retourne le dict de métriques d'évaluation.
    """
    print("\n" + "═" * 60)
    print("  PlateVision — Pipeline YOLOv8n (Partie 1)")
    print("═" * 60)

    yaml_path    = prepare_yaml()
    weights_path = train_yolo(yaml_path=yaml_path, epochs=epochs,
                              batch=batch, device=device)
    metrics      = evaluate_yolo(weights_path=weights_path, yaml_path=yaml_path)
    plot_training_curves()

    print("\n[RÉSUMÉ PIPELINE]")
    print(f"  Poids        : {weights_path}")
    print(f"  mAP@0.5      : {metrics['map50']:.4f}")
    print(f"  mAP@0.5:0.95 : {metrics['map50_95']:.4f}")
    print(f"  Inférence    : {metrics['inference_ms_mean']:.1f} ms")
    print(f"  Temps réel   : {'OUI' if metrics['realtime_ok'] else 'NON'}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 7. REDRESSEMENT DU CROP
# ══════════════════════════════════════════════════════════════════════════════

def deskew_plate_crop(crop: np.ndarray) -> np.ndarray:
    """
    Redresse le crop de plaque pour compenser l'angle oblique de la caméra.
    Retourne le crop original si aucun contour valide ou angle < 2°.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return crop

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return crop

    rect = cv2.minAreaRect(largest)
    angle = rect[2]

    # cv2.minAreaRect renvoie angle dans [-90, 0) — normaliser
    if angle < -45:
        angle += 90

    if abs(angle) <= 2 or abs(angle) >= 45:
        return crop

    h, w = crop.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(crop, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ══════════════════════════════════════════════════════════════════════════════
# 8. POST-TRAITEMENT OCR
# ══════════════════════════════════════════════════════════════════════════════

def postprocess_ocr_text(raw_text: str) -> str:
    """
    Nettoie et normalise le texte brut EasyOCR.
    Étapes : majuscules → suppression espaces/tirets/points/underscores
    → filtre [A-Z0-9]. Retourne "" si résultat vide.
    """
    import re
    text = raw_text.upper()
    text = re.sub(r"[ \-._]", "", text)
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# 9. LECTURE OCR D'UN CROP
# ══════════════════════════════════════════════════════════════════════════════

def read_plate_ocr(crop: np.ndarray,
                   reader=None,
                   lang_list: "list | None" = None) -> dict:
    """
    Applique EasyOCR sur un crop de plaque.
    Justification §3.2.2 : EasyOCR gère mieux les polices alphanumériques
    des plaques camerounaises que Tesseract, supporte le fine-tuning.
    """
    if reader is None:
        import easyocr
        try:
            import torch
            gpu = torch.cuda.is_available()
        except ImportError:
            gpu = False
        reader = easyocr.Reader(lang_list or ["en"], gpu=gpu)

    # Allowlist restreint EasyOCR aux caractères alphanumériques — évite les artefacts OCR (accents, ponctuation)
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    raw_results = reader.readtext(
        crop,
        detail=1,
        paragraph=False,
        allowlist=allowlist,
    )

    # Trier par position x (gauche → droite) : les plaques camerounaises se lisent gauche→droite
    # easyocr retourne list[tuple[bbox, text, conf]]
    raw_results.sort(key=lambda r: r[0][0][0])  # type: ignore[index]

    raw_text = "".join(str(r[1]) for r in raw_results)  # type: ignore[index]
    confidences: list[float] = [float(r[2]) for r in raw_results]  # type: ignore[index]
    confidence = float(np.mean(confidences)) if confidences else 0.0
    plate_text = postprocess_ocr_text(raw_text)

    return {
        "plate_text": plate_text,
        "raw_text":   raw_text,
        "confidence": confidence,
        "n_boxes":    len(raw_results),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 10. MÉTRIQUES CER ET WER
# ══════════════════════════════════════════════════════════════════════════════

def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate = edit_distance(ref, hyp) / len(ref).
    Implémentation Levenshtein sans bibliothèque externe.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0

    r, h = list(reference), list(hypothesis)
    n, m = len(r), len(h)
    # Programmation dynamique O(n×m) — dp[j] = distance d'édition(r[:i], h[:j])
    # Implémentation from scratch : pas de dépendance python-Levenshtein ou nltk
    dp = list(range(m + 1))

    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if r[i - 1] == h[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])

    return dp[m] / n


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate sur plaques : chaque plaque entière = 1 mot.
    ref_words=[reference], hyp_words=[hypothesis].
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return 0.0 if reference == hypothesis else 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 11. ÉVALUATION PIPELINE COMPLET YOLO + OCR
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_full_pipeline(weights_path: "Path | None" = None,
                            yaml_path: Path = YAML_PATH,
                            ocr_results_path: Path = Path("data/processed/ocr_results.json"),
                            report_dir: Path = REPORT_DIR) -> dict:
    """
    Évalue le pipeline bout-en-bout YOLO+OCR sur le jeu de test.
    Calcule CER, WER (métriques obligatoires §4.1 A2).
    Sauvegarde dans report_dir/yolo_ocr_evaluation.json.
    """
    weights_path = Path(weights_path) if weights_path else WEIGHTS_DIR / "yolov8_platevision.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Poids introuvables : {weights_path}")

    # ── Références OCR ────────────────────────────────────────────────────
    ocr_results_path = Path(ocr_results_path)
    references: dict[str, str] = {}
    if ocr_results_path.exists():
        with open(ocr_results_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        entries = raw if isinstance(raw, list) else raw.get("results", [])
        for entry in entries:
            pt = entry.get("plate_text") or ""
            conf = entry.get("ocr_confidence", 0.0) or 0.0
            # crop_path: e.g. data/processed/plate_crops/roboflow_00001.jpg
            # maps to YOLO test image: 000001.jpg (strip source prefix, zero-pad)
            crop = entry.get("crop_path") or entry.get("image_path") or entry.get("filename") or ""
            if pt and conf >= 0.45 and crop:
                stem = Path(crop).stem  # e.g. "roboflow_00001"
                parts = stem.split("_", 1)
                img_name = parts[-1].zfill(6) + ".jpg"  # "000001.jpg"
                references[img_name] = postprocess_ocr_text(pt)

    if len(references) < 10:
        logger.warning("Seulement %d références OCR disponibles (< 10).", len(references))

    # ── Images de test ────────────────────────────────────────────────────
    test_dir = Path("data/processed/images/test")
    test_images = sorted(test_dir.glob("*.jpg")) if test_dir.exists() else []

    # Filtrer celles qui ont une référence
    pairs = [(p, references[p.name]) for p in test_images if p.name in references]
    if not pairs:
        logger.warning("Aucune image de test avec référence OCR disponible.")
        pairs = [(p, "") for p in test_images[:20]]

    import easyocr
    try:
        import torch
        gpu = torch.cuda.is_available()
    except ImportError:
        gpu = False
    reader = easyocr.Reader(["en"], gpu=gpu)

    cer_list, wer_list, ocr_conf_list = [], [], []
    n_detections_ok = 0

    for img_path, ref_text in pairs:
        det = detect_plate(str(img_path), weights_path=weights_path, save_annotated=False)
        if det["n_detections"] == 0:
            cer_list.append(1.0)
            wer_list.append(1.0)
            continue

        n_detections_ok += 1
        crop = det["detections"][0]["crop"]
        crop = deskew_plate_crop(crop)
        ocr_res = read_plate_ocr(crop, reader=reader)
        cer_list.append(compute_cer(ref_text, ocr_res["plate_text"]))
        wer_list.append(compute_wer(ref_text, ocr_res["plate_text"]))
        ocr_conf_list.append(ocr_res["confidence"])

    n_tested = len(pairs)
    perfect = sum(1 for c in cer_list if c == 0.0)

    metrics: dict = {
        "n_images_tested":   n_tested,
        "n_detections_ok":   n_detections_ok,
        "detection_rate":    n_detections_ok / n_tested if n_tested else 0.0,
        "mean_cer":          float(np.mean(cer_list)) if cer_list else 0.0,
        "mean_wer":          float(np.mean(wer_list)) if wer_list else 0.0,
        "mean_ocr_conf":     float(np.mean(ocr_conf_list)) if ocr_conf_list else 0.0,
        "perfect_reads":     perfect,
        "perfect_read_rate": perfect / n_tested if n_tested else 0.0,
    }

    # ── Comparaison NB vs YOLO+OCR ────────────────────────────────────────
    nb_metrics_path = WEIGHTS_DIR / "naive_bayes_metrics.json"
    if not nb_metrics_path.exists():
        nb_metrics_path = Path("models/weights/naive_bayes_metrics.json")
    if nb_metrics_path.exists():
        with open(nb_metrics_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        nb_test = nb.get("test", nb)
        nb_acc = float(nb_test.get("accuracy", nb.get("accuracy", 0.0)))
        better = "YOLO+OCR meilleur" if metrics["mean_wer"] < (1 - nb_acc) else "NB meilleur"
        metrics["comparison_nb_vs_yolo_ocr"] = {
            "nb_accuracy":   nb_acc,
            "yolo_ocr_wer":  metrics["mean_wer"],
            "improvement":   better,
            "nb_limitation": (
                "NB classe des caractères pré-segmentés, "
                "inutilisable sans détection préalable"
            ),
        }

    # ── Sauvegarde + affichage ────────────────────────────────────────────
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / "yolo_ocr_evaluation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "═" * 58)
    print("  ÉVALUATION PIPELINE YOLO+OCR — PlateVision MINT/DGI")
    print("═" * 58)
    print(f"  Images testées     : {metrics['n_images_tested']}")
    print(f"  Détections OK      : {metrics['n_detections_ok']}  "
          f"({metrics['detection_rate']:.1%})")
    print(f"  CER moyen          : {metrics['mean_cer']:.4f}")
    print(f"  WER moyen          : {metrics['mean_wer']:.4f}")
    print(f"  Conf. OCR moyenne  : {metrics['mean_ocr_conf']:.4f}")
    print(f"  Lectures parfaites : {metrics['perfect_reads']}  "
          f"({metrics['perfect_read_rate']:.1%})")
    print("═" * 58 + "\n")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 12. PIPELINE COMPLET PARTIE 2 + DÉMO JURY
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(image_path: "str | None" = None,
                      weights_path: "Path | None" = None) -> dict:
    """
    Pipeline bout-en-bout pour une image unique (démo jury §5) :
    detect_plate → deskew → read_plate_ocr → affichage formaté.
    Si image_path=None, utilise la première image de test disponible.
    """
    if image_path is None:
        test_dir = Path("data/processed/yolo/images/test")
        candidates = sorted(test_dir.glob("*.jpg")) if test_dir.exists() else []
        if not candidates:
            raise FileNotFoundError("Aucune image de test trouvée dans data/processed/yolo/images/test/")
        image_path = str(candidates[0])

    import easyocr
    try:
        import torch
        gpu = torch.cuda.is_available()
    except ImportError:
        gpu = False
    reader = easyocr.Reader(["en"], gpu=gpu)

    det = detect_plate(image_path, weights_path=weights_path, save_annotated=True)  # type: ignore[arg-type]
    output_detections = []

    for d in det["detections"]:
        crop = deskew_plate_crop(d["crop"])
        ocr_res = read_plate_ocr(crop, reader=reader)
        entry = {
            "bbox":       d["bbox"],
            "yolo_conf":  d["confidence"],
            "plate_text": ocr_res["plate_text"],
            "ocr_conf":   ocr_res["confidence"],
        }
        output_detections.append(entry)
        print(f"Plaque détectée : {ocr_res['plate_text']} "
              f"(conf YOLO: {d['confidence']:.2f}, conf OCR: {ocr_res['confidence']:.2f})")

    if not output_detections:
        print("Aucune plaque détectée.")

    return {"image_path": image_path, "detections": output_detections}


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="PlateVision A2 — Détection YOLOv8n + OCR")
    parser.add_argument("--train",    action="store_true", help="Lance le pipeline d'entraînement")
    parser.add_argument("--epochs",   type=int, default=50, help="Nombre d'epochs (défaut 50)")
    parser.add_argument("--batch",    type=int, default=16, help="Taille du batch (défaut 16)")
    parser.add_argument("--device",   type=str, default="auto", help="Device (auto/cpu/0)")
    parser.add_argument("--evaluate", action="store_true", help="Évalue le modèle sauvegardé")
    parser.add_argument("--detect",   type=str, default=None, metavar="IMAGE_PATH",
                        help="Inférence sur une image")
    parser.add_argument("--curves",   action="store_true", help="Régénère les courbes")
    parser.add_argument("--ocr",      type=str, default=None, metavar="IMAGE_PATH",
                        help="Pipeline complet YOLO+OCR sur une image (démo jury)")
    parser.add_argument("--eval-ocr", action="store_true",
                        help="Évalue le pipeline complet YOLO+OCR (CER, WER) sur le jeu de test")
    args = parser.parse_args()

    if args.train:
        run_yolo_training_pipeline(epochs=args.epochs, batch=args.batch, device=args.device)

    elif args.evaluate:
        prepare_yaml()
        metrics = evaluate_yolo()
        print(json.dumps(metrics, indent=2))

    elif args.detect:
        result = detect_plate(args.detect)
        print(f"Détections : {result['n_detections']}")
        for i, d in enumerate(result["detections"]):
            print(f"  [{i}] bbox={d['bbox']}  conf={d['confidence']:.3f}")

    elif args.curves:
        plot_training_curves()

    elif args.ocr:
        result = run_full_pipeline(image_path=args.ocr)
        for d in result["detections"]:
            print(f"Plaque : {d['plate_text']}  "
                  f"(YOLO: {d['yolo_conf']:.2f}, OCR: {d['ocr_conf']:.2f})")

    elif args.eval_ocr:
        metrics = evaluate_full_pipeline()
        print(json.dumps(
            {k: v for k, v in metrics.items() if k != "comparison_nb_vs_yolo_ocr"},
            indent=2))
        if "comparison_nb_vs_yolo_ocr" in metrics:
            print("\n[Comparaison NB vs YOLO+OCR]")
            print(json.dumps(metrics["comparison_nb_vs_yolo_ocr"], indent=2,
                             ensure_ascii=False))

    else:
        parser.print_help()
