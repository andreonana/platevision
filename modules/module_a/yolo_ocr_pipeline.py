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
        cfg = {
            "path":  str(Path.cwd() / "data/raw"),
            "train": "images/train",
            "val":   "images/val",
            "test":  "images/test",
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
# POINT D'ENTRÉE CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="PlateVision A2 — Détection YOLOv8n")
    parser.add_argument("--train",    action="store_true", help="Lance le pipeline d'entraînement")
    parser.add_argument("--epochs",   type=int, default=50, help="Nombre d'epochs (défaut 50)")
    parser.add_argument("--batch",    type=int, default=16, help="Taille du batch (défaut 16)")
    parser.add_argument("--device",   type=str, default="auto", help="Device (auto/cpu/0)")
    parser.add_argument("--evaluate", action="store_true", help="Évalue le modèle sauvegardé")
    parser.add_argument("--detect",   type=str, default=None, metavar="IMAGE_PATH",
                        help="Inférence sur une image")
    parser.add_argument("--curves",   action="store_true", help="Régénère les courbes")
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

    else:
        parser.print_help()
