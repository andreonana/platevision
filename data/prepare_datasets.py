"""
PlateVision — Pipeline de prétraitement des datasets
MINT/DGI Cameroun — UCAC-ICAM / ULC-ICAM

Ce script prépare les données brutes (Roboflow + Mendeley) pour alimenter :
  - Module A1 : Naïves Bayes Gaussien (features manuelles 120D sur caractères 28×28)
  - Module A2 : YOLOv8 + EasyOCR (images plaques normalisées 640×640)
  - Module B  : K-Means sur embeddings CNN
  - Module C  : MDP (états = cluster_id, niveau OCR, alerte)

Usage :
  python data/prepare_datasets.py                  → phases 1 à 7 complètes
  python data/prepare_datasets.py --from-phase 4   → reprendre à la phase 4
  python data/prepare_datasets.py --phase-only 6   → phase 6 uniquement
"""

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Suppression des avertissements non critiques
warnings.filterwarnings("ignore")

# ─── Configuration du logger ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/processed/pipeline.log", mode="a"),
    ],
)
logger = logging.getLogger("platevision")

# ─── Constantes globales ──────────────────────────────────────────────────────
VALID_CHARS = sorted("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
LABEL_TO_IDX = {c: i for i, c in enumerate(VALID_CHARS)}
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
PLATE_TARGET_SIZE = (200, 60)   # largeur × hauteur pour les crops
CHAR_TARGET_SIZE = (28, 28)     # taille standard des caractères
YOLO_TARGET_SIZE = (640, 640)   # entrée YOLOv8


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — ACQUISITION
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_datasets(roboflow_path: Path, mendeley_path: Path) -> dict:
    """
    Charge et unifie les deux datasets bruts.

    Entrée  : chemins vers les dossiers raw Roboflow et Mendeley
    Sortie  : dictionnaire unifié {roboflow: {...}, mendeley: {...}}
    Alimente : tous les modules en aval
    """
    roboflow_path = Path(roboflow_path)
    mendeley_path = Path(mendeley_path)

    roboflow_data = _load_roboflow(roboflow_path)
    mendeley_data = _load_mendeley(mendeley_path)

    datasets = {
        "roboflow": roboflow_data,
        "mendeley": mendeley_data,
    }

    # Rapport d'acquisition
    n_rf = roboflow_data["n_images"]
    n_md = mendeley_data["n_images"]
    n_rf_ann = len([l for l in roboflow_data["labels"] if l is not None])

    print("\n╔══════════════════════════════════════════╗")
    print("║     RAPPORT D'ACQUISITION PlateVision   ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  Roboflow  : {n_rf:4d} images, {n_rf_ann:4d} annotations   ║")
    print(f"║  Mendeley  : {n_md:4d} images, {n_md:4d} annotations   ║")
    print(f"║  Total     : {n_rf + n_md:4d} images                  ║")
    print("╚══════════════════════════════════════════╝\n")

    return datasets


def _load_roboflow(roboflow_path: Path) -> dict:
    """Charge le dataset Roboflow au format YOLO natif."""
    # Recherche du sous-dossier principal
    candidates = [d for d in roboflow_path.iterdir() if d.is_dir()]
    root = candidates[0] if candidates else roboflow_path

    # Lecture de data.yaml si présent
    yaml_path = root / "data.yaml"
    class_names = ["plaque_immatriculation"]
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            content = f.read()
        # Extraction simple du champ names sans dépendance PyYAML
        match = re.search(r"names:\s*\[([^\]]+)\]", content)
        if match:
            class_names = [n.strip().strip("'\"") for n in match.group(1).split(",")]
        logger.info(f"Roboflow data.yaml : nc={len(class_names)}, names={class_names}")

    images, labels = [], []
    splits = ["train", "valid", "test"]

    for split in splits:
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            images.append(img_path)
            if lbl_path.exists():
                labels.append(lbl_path)
            else:
                labels.append(None)
                logger.warning(f"[Roboflow] Image sans annotation : {img_path.name}")

    n_no_label = sum(1 for l in labels if l is None)
    if n_no_label > 0:
        logger.warning(f"[Roboflow] {n_no_label} images sans annotation détectées")

    return {
        "images": images,
        "labels": labels,
        "n_images": len(images),
        "has_text": False,
        "class_names": class_names,
        "root": root,
    }


def _load_mendeley(mendeley_path: Path) -> dict:
    """
    Charge le dataset Mendeley.
    Format détecté automatiquement — dans ce dataset, images uniquement (pas d'annotations).
    Les images sont des gros plans de plaques → image entière = région de la plaque.
    """
    # Détection du format d'annotation
    annotation_format = _detect_annotation_format(mendeley_path)
    logger.info(f"[Mendeley] Format d'annotation détecté : {annotation_format}")

    # Collecte de toutes les images
    all_images = []
    for img_path in sorted(mendeley_path.rglob("*")):
        if img_path.suffix in IMG_EXTENSIONS:
            all_images.append(img_path)

    if annotation_format == "none":
        # Images directes de plaques sans annotation → bbox = image entière
        records = []
        for img_path in all_images:
            records.append({
                "image_path": str(img_path),
                "plate_text": None,
                "x1": None,   # sera calculé à la lecture de l'image
                "y1": None,
                "x2": None,
                "y2": None,
                "img_width": None,
                "img_height": None,
                "source": "mendeley",
                "region": img_path.parent.name,
            })
        df = pd.DataFrame(records)

    elif annotation_format == "csv":
        df = _parse_csv_annotations(mendeley_path, all_images)
    elif annotation_format == "xml":
        df = _parse_xml_annotations(mendeley_path, all_images)
    elif annotation_format == "json":
        df = _parse_json_annotations(mendeley_path, all_images)
    elif annotation_format == "yolo_txt":
        df = _parse_yolo_txt_annotations(mendeley_path, all_images)
    else:
        df = pd.DataFrame()

    return {
        "dataframe": df,
        "n_images": len(all_images),
        "annotation_format": annotation_format,
        "images": all_images,
    }


def _detect_annotation_format(path: Path) -> str:
    """Détecte automatiquement le format d'annotation dans un dossier."""
    csv_files = list(path.rglob("*.csv"))
    xml_files = list(path.rglob("*.xml"))
    json_files = list(path.rglob("*.json"))
    # Fichiers .txt non-README
    txt_files = [f for f in path.rglob("*.txt") if "readme" not in f.name.lower()]

    if csv_files:
        return "csv"
    if xml_files:
        return "xml"
    if json_files:
        return "json"
    if txt_files:
        return "yolo_txt"
    return "none"


def _parse_csv_annotations(base_path: Path, images: list) -> pd.DataFrame:
    """Parse les annotations CSV en DataFrame unifié."""
    csv_files = list(base_path.rglob("*.csv"))
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Erreur lecture CSV {csv_file} : {e}")
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    # Normalisation des colonnes si possible
    col_map = {
        "filename": "image_path", "file": "image_path",
        "label": "plate_text", "text": "plate_text",
        "xmin": "x1", "x_min": "x1",
        "ymin": "y1", "y_min": "y1",
        "xmax": "x2", "x_max": "x2",
        "ymax": "y2", "y_max": "y2",
    }
    combined.rename(columns={k: v for k, v in col_map.items() if k in combined.columns}, inplace=True)
    for col in ["image_path", "plate_text", "x1", "y1", "x2", "y2", "img_width", "img_height"]:
        if col not in combined.columns:
            combined[col] = None
    combined["source"] = "mendeley"
    return combined


def _parse_xml_annotations(base_path: Path, images: list) -> pd.DataFrame:
    """Parse les annotations XML (format Pascal VOC) en DataFrame unifié."""
    records = []
    xml_files = list(base_path.rglob("*.xml"))
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            filename = root.findtext("filename", default="")
            size = root.find("size")
            img_w = int(size.findtext("width", 0)) if size else None
            img_h = int(size.findtext("height", 0)) if size else None
            for obj in root.findall("object"):
                bbox = obj.find("bndbox")
                if bbox is None:
                    continue
                records.append({
                    "image_path": str(base_path / filename),
                    "plate_text": obj.findtext("name"),
                    "x1": float(bbox.findtext("xmin", 0)),
                    "y1": float(bbox.findtext("ymin", 0)),
                    "x2": float(bbox.findtext("xmax", 0)),
                    "y2": float(bbox.findtext("ymax", 0)),
                    "img_width": img_w,
                    "img_height": img_h,
                    "source": "mendeley",
                })
        except Exception as e:
            logger.error(f"Erreur lecture XML {xml_file} : {e}")
    return pd.DataFrame(records)


def _parse_json_annotations(base_path: Path, images: list) -> pd.DataFrame:
    """Parse les annotations JSON (format COCO ou autre) en DataFrame unifié."""
    records = []
    json_files = list(base_path.rglob("*.json"))
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            # Format COCO
            if "images" in data and "annotations" in data:
                id_to_img = {img["id"]: img for img in data["images"]}
                for ann in data["annotations"]:
                    img_info = id_to_img.get(ann["image_id"], {})
                    bbox = ann.get("bbox", [0, 0, 0, 0])  # COCO: x, y, w, h
                    records.append({
                        "image_path": str(base_path / img_info.get("file_name", "")),
                        "plate_text": ann.get("text"),
                        "x1": bbox[0],
                        "y1": bbox[1],
                        "x2": bbox[0] + bbox[2],
                        "y2": bbox[1] + bbox[3],
                        "img_width": img_info.get("width"),
                        "img_height": img_info.get("height"),
                        "source": "mendeley",
                    })
        except Exception as e:
            logger.error(f"Erreur lecture JSON {json_file} : {e}")
    return pd.DataFrame(records)


def _parse_yolo_txt_annotations(base_path: Path, images: list) -> pd.DataFrame:
    """Parse les annotations YOLO .txt en DataFrame unifié."""
    records = []
    for img_path in images:
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    records.append({
                        "image_path": str(img_path),
                        "plate_text": None,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "img_width": w, "img_height": h,
                        "source": "mendeley",
                    })
        except Exception as e:
            logger.error(f"Erreur lecture YOLO txt {txt_path} : {e}")
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — EXPLORATION RAPIDE (EDA MINIMALE)
# ══════════════════════════════════════════════════════════════════════════════

def quick_eda(datasets_dict: dict, output_dir: Path) -> dict:
    """
    EDA orientée prétraitement — uniquement ce qui impacte les phases suivantes.

    Entrée  : dictionnaire datasets, répertoire de sortie
    Sortie  : dictionnaire eda_report + fichier eda_report.json
    Alimente : phases 3-7, aide à la normalisation et au filtrage
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_images_paths = []
    all_bboxes = []  # (img_path, x1, y1, x2, y2, img_w, img_h)

    # Collecte Roboflow
    rf = datasets_dict["roboflow"]
    for img_path, lbl_path in zip(rf["images"], rf["labels"]):
        all_images_paths.append((img_path, "roboflow"))
        if lbl_path is not None and lbl_path.exists():
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                H, W = img.shape[:2]
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cx, cy, bw, bh = map(float, parts[1:5])
                            x1 = int((cx - bw / 2) * W)
                            y1 = int((cy - bh / 2) * H)
                            x2 = int((cx + bw / 2) * W)
                            y2 = int((cy + bh / 2) * H)
                            all_bboxes.append((img_path, x1, y1, x2, y2, W, H))
            except Exception as e:
                logger.error(f"EDA Roboflow bbox {img_path} : {e}")

    # Collecte Mendeley
    md = datasets_dict["mendeley"]
    for img_path in md["images"]:
        all_images_paths.append((img_path, "mendeley"))

    report = {
        "total_images": len(all_images_paths),
        "blurry_images": [],
        "dark_images": [],
        "corrupted_images": [],
        "resolution_stats": {},
        "plate_ratio_stats": {},
        "character_frequency": {},
        "recommended_target_size": [640, 640],
    }

    widths, heights = [], []
    plate_ratios = []
    sample_images_for_grid = []

    logger.info("EDA : analyse des images en cours...")
    for img_path, source in tqdm(all_images_paths, desc="EDA images"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                report["corrupted_images"].append(str(img_path))
                continue

            H, W = img.shape[:2]
            widths.append(W)
            heights.append(H)

            # Signalement résolutions non standards
            if W < 100 or H < 100 or W > 4000 or H > 4000:
                logger.warning(f"Résolution non standard ({W}×{H}) : {img_path.name}")

            # Flou (variance du Laplacien)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if blur_score < 100:
                report["blurry_images"].append(str(img_path))

            # Luminosité (canal V de HSV)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mean_v = float(hsv[:, :, 2].mean())
            if mean_v < 50:
                report["dark_images"].append(str(img_path))

            if len(sample_images_for_grid) < 12:
                sample_images_for_grid.append((img.copy(), img_path, source))

        except Exception as e:
            logger.error(f"EDA erreur {img_path} : {e}")
            report["corrupted_images"].append(str(img_path))

    # Statistiques bounding boxes Roboflow
    for img_path, x1, y1, x2, y2, img_w, img_h in all_bboxes:
        bw = x2 - x1
        bh = y2 - y1
        if bh > 0 and img_w > 0 and img_h > 0:
            ratio = bw / bh
            plate_ratios.append(ratio)
            area_ratio = (bw * bh) / (img_w * img_h)
            if ratio < 2.5 or ratio > 6.0:
                logger.warning(
                    f"Plaque outlier ratio={ratio:.2f} : {Path(img_path).name}"
                )

    # Stats résolutions
    if widths:
        report["resolution_stats"] = {
            "min_width": int(np.min(widths)),
            "max_width": int(np.max(widths)),
            "median_width": int(np.median(widths)),
            "min_height": int(np.min(heights)),
            "max_height": int(np.max(heights)),
            "median_height": int(np.median(heights)),
        }

    if plate_ratios:
        report["plate_ratio_stats"] = {
            "min": round(float(np.min(plate_ratios)), 3),
            "max": round(float(np.max(plate_ratios)), 3),
            "median": round(float(np.median(plate_ratios)), 3),
        }

    # Fréquence des caractères (si transcriptions disponibles dans Mendeley)
    md_df = datasets_dict["mendeley"]["dataframe"]
    char_freq = {}
    if not md_df.empty and "plate_text" in md_df.columns:
        texts = md_df["plate_text"].dropna()
        for text in texts:
            for ch in str(text).upper():
                if ch in VALID_CHARS:
                    char_freq[ch] = char_freq.get(ch, 0) + 1
    report["character_frequency"] = char_freq

    # Sauvegarde du rapport JSON
    report_path = output_dir / "eda_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"EDA rapport sauvegardé : {report_path}")

    # Génération de la grille d'images EDA
    _generate_eda_grid(sample_images_for_grid, all_bboxes, figures_dir)

    logger.info(
        f"EDA terminée : {report['total_images']} images, "
        f"{len(report['blurry_images'])} floues, "
        f"{len(report['dark_images'])} sombres, "
        f"{len(report['corrupted_images'])} corrompues"
    )
    return report


def _generate_eda_grid(samples: list, bboxes: list, figures_dir: Path):
    """Génère une grille 4×3 d'images aléatoires avec bboxes vertes."""
    bbox_map = {str(b[0]): b[1:] for b in bboxes}
    grid_rows, grid_cols = 3, 4
    cell_w, cell_h = 320, 240
    canvas = np.ones((grid_rows * cell_h, grid_cols * cell_w, 3), dtype=np.uint8) * 200

    for idx, (img, img_path, source) in enumerate(samples[:grid_rows * grid_cols]):
        row = idx // grid_cols
        col = idx % grid_cols
        # Dessin bbox si disponible
        vis = img.copy()
        H, W = vis.shape[:2]
        if str(img_path) in bbox_map:
            x1, y1, x2, y2, iw, ih = bbox_map[str(img_path)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Redimensionnement pour la grille
        cell = cv2.resize(vis, (cell_w, cell_h))
        label = f"{source[:3].upper()} {Path(img_path).name[:20]}"
        cv2.putText(cell, label, (4, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 0), 1)
        y_off = row * cell_h
        x_off = col * cell_w
        canvas[y_off:y_off + cell_h, x_off:x_off + cell_w] = cell

    out_path = figures_dir / "eda_sample_grid.png"
    cv2.imwrite(str(out_path), canvas)
    logger.info(f"Grille EDA sauvegardée : {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — EXTRACTION DES RÉGIONS DE PLAQUES
# ══════════════════════════════════════════════════════════════════════════════

def extract_plate_regions(datasets_dict: dict, output_dir: Path) -> tuple:
    """
    Extrait et sauvegarde les recadrages de plaques depuis chaque image.
    Sépare les deux sources dès cette phase.

    Entrée  : dictionnaire datasets, répertoire de sortie
    Sortie  : (crops_rf, crops_md) — deux listes de dicts indépendantes
      - crops_rf : crops Roboflow → A1 segmentation/labellisation + A2 YOLO
      - crops_md : crops Mendeley → A2 YOLO uniquement, sans labellisation chars
    Alimente : Phase 4 OCR (rf seulement) · Phase 5 segmentation (rf) · A2 YOLO (rf+md)
    """
    crops_dir = Path(output_dir) / "plate_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    plate_crops_rf = []   # Roboflow → pipeline A1 complet + YOLO A2
    plate_crops_md = []   # Mendeley → YOLO A2 uniquement
    n_rf = 0
    n_md = 0
    n_heuristic = 0
    n_rejected = 0
    idx = 0

    # ── Roboflow (annotations YOLO) ──────────────────────────────────────────
    rf = datasets_dict["roboflow"]
    logger.info(f"Phase 3 : extraction Roboflow ({rf['n_images']} images)...")
    for img_path, lbl_path in tqdm(
        zip(rf["images"], rf["labels"]),
        total=rf["n_images"],
        desc="Crops Roboflow"
    ):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Image illisible : {img_path.name}")
                continue
            H, W = img.shape[:2]

            if lbl_path is None or not lbl_path.exists():
                n_rejected += 1
                continue

            with open(lbl_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw / 2) * W)
                y1 = int((cy - bh / 2) * H)
                x2 = int((cx + bw / 2) * W)
                y2 = int((cy + bh / 2) * H)

                crop = _extract_crop(img, x1, y1, x2, y2, padding_pct=0.05)
                if crop is None:
                    n_rejected += 1
                    continue

                crop_resized = _resize_plate(crop)
                blur_score = float(cv2.Laplacian(
                    cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY), cv2.CV_64F
                ).var())

                crop_filename = f"roboflow_{idx:05d}.jpg"
                crop_path = crops_dir / crop_filename
                cv2.imwrite(str(crop_path), crop_resized)

                plate_crops_rf.append({
                    "crop_path": str(crop_path),
                    "plate_text": None,
                    "source": "roboflow",
                    "detection_method": "annotation",
                    "original_image": str(img_path),
                    "blur_score": blur_score,
                })
                idx += 1
                n_rf += 1

        except Exception as e:
            logger.error(f"Phase 3 Roboflow {img_path.name} : {e}")
            n_rejected += 1

    # ── Mendeley (image entière = plaque) ────────────────────────────────────
    md = datasets_dict["mendeley"]
    logger.info(f"Phase 3 : extraction Mendeley ({md['n_images']} images)...")
    for img_path in tqdm(md["images"], desc="Crops Mendeley"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Image illisible : {img_path.name}")
                continue
            H, W = img.shape[:2]

            # Image entière = plaque (gros plan direct)
            crop = img.copy()
            crop_resized = _resize_plate(crop)

            blur_score = float(cv2.Laplacian(
                cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY), cv2.CV_64F
            ).var())

            crop_filename = f"mendeley_{idx:05d}.jpg"
            crop_path = crops_dir / crop_filename
            cv2.imwrite(str(crop_path), crop_resized)

            plate_crops_md.append({
                "crop_path": str(crop_path),
                "plate_text": None,
                "source": "mendeley",
                "detection_method": "full_image_crop",
                "original_image": str(img_path),
                "blur_score": blur_score,
                "region": img_path.parent.name,
            })
            idx += 1
            n_md += 1

        except Exception as e:
            logger.error(f"Phase 3 Mendeley {img_path.name} : {e}")
            n_rejected += 1

    logger.info(
        f"Phase 3 : {n_rf} crops Roboflow (→ A1 chars + A2 YOLO) + "
        f"{n_md} crops Mendeley (→ A2 YOLO uniquement) | "
        f"{n_rejected} rejetés"
    )
    return plate_crops_rf, plate_crops_md


def _extract_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                  padding_pct: float = 0.05) -> np.ndarray | None:
    """
    Recadre une région de plaque avec padding, validation de taille minimale.
    Retourne None si le crop est trop petit.
    """
    H, W = img.shape[:2]
    bw = x2 - x1
    bh = y2 - y1

    if bw < 20 or bh < 10:
        return None

    pad_x = int(bw * padding_pct)
    pad_y = int(bh * padding_pct)
    x1p = max(0, x1 - pad_x)
    y1p = max(0, y1 - pad_y)
    x2p = min(W, x2 + pad_x)
    y2p = min(H, y2 + pad_y)

    crop = img[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        return None
    return crop


def _resize_plate(img: np.ndarray) -> np.ndarray:
    """
    Redimensionne un crop de plaque vers PLATE_TARGET_SIZE (200×60).
    Utilise INTER_CUBIC si upscaling, INTER_AREA si downscaling.
    """
    tw, th = PLATE_TARGET_SIZE
    h, w = img.shape[:2]
    if w < tw or h < th:
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_AREA
    return cv2.resize(img, (tw, th), interpolation=interpolation)


def _detect_plates_heuristic(img: np.ndarray, img_path: Path,
                              crops_dir: Path, idx: int) -> list:
    """
    Détection heuristique de plaques par couleur (blanc/jaune) et forme rectangulaire.
    Utilisé quand aucune annotation n'est disponible.
    """
    results = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Masque plaque blanche
    mask_white = cv2.inRange(
        hsv,
        np.array([0, 0, 180]),
        np.array([30, 60, 255])
    )
    # Masque plaque jaune
    mask_yellow = cv2.inRange(
        hsv,
        np.array([20, 100, 100]),
        np.array([35, 255, 255])
    )
    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)

    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    H, W = img.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) < 4:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        ratio = w / h
        if not (2.0 < ratio < 6.0):
            continue

        crop = _extract_crop(img, x, y, x + w, y + h, padding_pct=0.05)
        if crop is None:
            continue
        crop_resized = _resize_plate(crop)
        blur_score = float(cv2.Laplacian(
            cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY), cv2.CV_64F
        ).var())

        crop_filename = f"heuristic_{idx:05d}.jpg"
        crop_path = crops_dir / crop_filename
        cv2.imwrite(str(crop_path), crop_resized)

        results.append({
            "crop_path": str(crop_path),
            "plate_text": None,
            "source": "mendeley",
            "detection_method": "heuristique",
            "original_image": str(img_path),
            "blur_score": blur_score,
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3BIS — OCR SUR LES CROPS DE PLAQUES
# ══════════════════════════════════════════════════════════════════════════════

# Longueur plausible d'une transcription de plaque (caractères alphanumériques)
_PLATE_MIN_CHARS = 2
_PLATE_MAX_CHARS = 8


def _ocr_mendeley_crop(reader, img: np.ndarray) -> tuple:
    """
    OCR strict dédié aux crops Mendeley.

    Logique :
      1. Lecture EasyOCR → liste de (bbox, texte, confiance)
      2. Nettoyage de chaque fragment : MAJUSCULES + alphanum uniquement
      3. Confiance moyenne pondérée par la longueur de chaque fragment nettoyé
         (évite qu'un long fragment peu fiable dilue un fragment court bien lu)
      4. Filtre confiance : avg_conf >= 0.45
      5. Filtre plausibilité : longueur totale ∈ [_PLATE_MIN_CHARS, _PLATE_MAX_CHARS]

    Retourne : (plate_text: str | None, avg_conf: float)
      - plate_text est None si l'une des conditions ci-dessus n'est pas remplie.
      - avg_conf est toujours retourné (même quand plate_text est None) pour
        permettre la journalisation des raisons du rejet.
    """
    try:
        detections = reader.readtext(img)
    except Exception as exc:
        logger.debug(f"EasyOCR exception : {exc}")
        return None, 0.0

    if not detections:
        return None, 0.0

    # Nettoyage et pondération par longueur de fragment
    cleaned_parts: list[str] = []
    total_weight = 0.0
    weighted_conf = 0.0

    for (_bbox, text, conf) in detections:
        fragment = re.sub(r"[^A-Z0-9]", "", text.upper())
        if not fragment:
            continue
        w = len(fragment)
        weighted_conf += conf * w
        total_weight += w
        cleaned_parts.append(fragment)

    if total_weight == 0.0:
        return None, 0.0

    avg_conf = weighted_conf / total_weight
    plate_text = "".join(cleaned_parts)

    # Filtre confiance
    if avg_conf < 0.45:
        return None, float(avg_conf)

    # Filtre plausibilité de longueur
    if not (_PLATE_MIN_CHARS <= len(plate_text) <= _PLATE_MAX_CHARS):
        logger.debug(
            f"OCR Mendeley rejeté — longueur hors plage "
            f"({len(plate_text)} chars, texte='{plate_text}', conf={avg_conf:.2f})"
        )
        return None, float(avg_conf)

    return plate_text, float(avg_conf)


def run_ocr_on_crops(plate_crops: list, output_dir: Path) -> list:
    """
    Phase 3bis — Reconnaissance OCR sur les crops de plaques.

    Parcourt chaque crop produit par extract_plate_regions() et tente
    de lire le texte de la plaque via EasyOCR.

    Entrée  : liste de dicts produits par extract_plate_regions()
              (chaque dict a les clés : crop_path, plate_text, source,
               detection_method, original_image, blur_score)

    Sortie  : même liste enrichie avec deux champs supplémentaires :
              - plate_text      : str nettoyé (MAJUSCULES, alphanum only)
                                  ou None si OCR échoue / confiance trop basse /
                                  longueur hors plage [2, 8] (Mendeley uniquement)
              - ocr_confidence  : float [0.0 – 1.0]
                                  • Mendeley : moyenne pondérée par longueur de fragment
                                  • Roboflow : moyenne simple des scores EasyOCR

    Deux chemins distincts selon la source :
      - source == "mendeley" → _ocr_mendeley_crop() : confiance pondérée +
        validation plausibilité longueur ; plate_text est None si l'un des
        deux critères échoue. Seuls les crops retenus alimenteront le
        labellisage automatique dans la Phase 5.
      - source != "mendeley" → logique standard (moyenne simple, pas de filtre
        longueur) ; ces crops sont labelisés via le modèle YOLO+OCR (Module A2).

    Alimente : Phase 5 (segmentation caractères) — labellisage des caractères
               segmentés depuis les crops Mendeley à transcription fiable
    """
    if not EASYOCR_AVAILABLE:
        logger.warning(
            "EasyOCR non installé (pip install easyocr). "
            "Phase 3bis ignorée — plate_text restera None pour tous les crops."
        )
        for crop_info in plate_crops:
            crop_info.setdefault("ocr_confidence", 0.0)
        return plate_crops

    output_dir = Path(output_dir)

    # Initialisation du reader UNE SEULE fois avant la boucle
    use_gpu = False
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except ImportError:
        pass

    logger.info(f"Phase 3bis : initialisation EasyOCR (gpu={use_gpu})...")
    reader = easyocr.Reader(["en"], gpu=use_gpu)

    ocr_results_log = []
    n_success        = 0   # texte retenu (confiance + plausibilité OK)
    n_low_conf       = 0   # rejeté : confiance < 0.45
    n_implausible    = 0   # rejeté : longueur hors [2, 8]
    n_failed         = 0   # image illisible ou exception

    for crop_info in tqdm(plate_crops, desc="OCR plaques"):
        crop_path = crop_info.get("crop_path", "")
        source    = crop_info.get("source", "")

        try:
            img = cv2.imread(crop_path)
            if img is None:
                crop_info["plate_text"]     = crop_info.get("plate_text")
                crop_info["ocr_confidence"] = 0.0
                n_failed += 1
                ocr_results_log.append({
                    "crop_path":      crop_path,
                    "plate_text":     crop_info.get("plate_text"),
                    "ocr_confidence": 0.0,
                    "source":         source,
                })
                continue

            if source == "mendeley":
                # ── Chemin strict : confiance pondérée + plausibilité longueur ──
                plate_text, avg_conf = _ocr_mendeley_crop(reader, img)
                crop_info["plate_text"]     = plate_text
                crop_info["ocr_confidence"] = avg_conf

                if plate_text is not None:
                    n_success += 1
                elif avg_conf >= 0.45:
                    # Confiance suffisante mais longueur hors plage
                    n_implausible += 1
                else:
                    n_low_conf += 1

            else:
                # ── Chemin standard : Roboflow (ou source inconnue) ──────────
                # Moyenne simple des confidences — pas de filtre plausibilité
                # car ces crops seront relabellisés par le modèle YOLO+OCR (A2)
                detections = reader.readtext(img)

                if not detections:
                    crop_info["plate_text"]     = None
                    crop_info["ocr_confidence"] = 0.0
                    n_failed += 1
                else:
                    texts       = []
                    confidences = []
                    for (_bbox, text, conf) in detections:
                        texts.append(text)
                        confidences.append(conf)

                    raw_text = "".join(texts)
                    cleaned  = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
                    avg_conf = float(sum(confidences) / len(confidences))

                    if avg_conf >= 0.45 and cleaned:
                        crop_info["plate_text"]     = cleaned
                        crop_info["ocr_confidence"] = avg_conf
                        n_success += 1
                    else:
                        crop_info["plate_text"]     = None
                        crop_info["ocr_confidence"] = avg_conf
                        n_low_conf += 1

        except Exception as exc:
            logger.error(f"OCR erreur sur {crop_path} : {exc}")
            crop_info["ocr_confidence"] = 0.0
            crop_info.setdefault("plate_text", None)
            n_failed += 1

        ocr_results_log.append({
            "crop_path":      crop_path,
            "plate_text":     crop_info.get("plate_text"),
            "ocr_confidence": crop_info.get("ocr_confidence", 0.0),
            "source":         source,
        })

    # Sauvegarde du rapport OCR
    ocr_report_path = output_dir / "ocr_results.json"
    with open(ocr_report_path, "w", encoding="utf-8") as f:
        json.dump(ocr_results_log, f, indent=2, ensure_ascii=False)

    n_mendeley_retained = sum(
        1 for c in plate_crops
        if c.get("source") == "mendeley" and c.get("plate_text") is not None
    )
    n_mendeley_total = sum(1 for c in plate_crops if c.get("source") == "mendeley")

    logger.info(
        f"Phase 3bis OCR terminée : {n_success} textes retenus, "
        f"{n_low_conf} conf<0.45, {n_implausible} longueur hors plage, "
        f"{n_failed} échecs"
    )
    logger.info(
        f"  Mendeley : {n_mendeley_retained}/{n_mendeley_total} crops "
        f"avec transcription fiable (→ labellisation Phase 5)"
    )
    logger.info(f"  Résultats OCR → {ocr_report_path}")
    return plate_crops


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — SEGMENTATION DES CARACTÈRES
# ══════════════════════════════════════════════════════════════════════════════

def segment_characters(plate_crops_list: list, output_dir: Path) -> list:
    """
    Segmente les caractères individuels depuis les crops de plaques 200×60.
    Produit des images 28×28 — unité de base du Naïves Bayes (Module A1).

    Entrée  : liste de dicts crops de plaques
    Sortie  : liste de dicts caractères segmentés
    Alimente : Module A1 (Naïves Bayes Gaussien)
    """
    chars_dir = Path(output_dir) / "characters"
    chars_dir.mkdir(parents=True, exist_ok=True)
    (chars_dir / "unknown").mkdir(exist_ok=True)

    all_chars = []
    char_counter = 0
    label_distribution = {}

    logger.info(f"Phase 4 : segmentation de {len(plate_crops_list)} crops...")
    for crop_info in tqdm(plate_crops_list, desc="Segmentation caractères"):
        try:
            img = cv2.imread(crop_info["crop_path"])
            if img is None:
                continue

            plate_text = crop_info.get("plate_text")
            chars = _segment_chars_from_plate(img)

            for pos, (x, y, w, h, char_img_28) in enumerate(chars):
                # Attribution du label
                label_certain = False
                if (plate_text and
                        plate_text.upper().replace(" ", "").isalnum() and
                        pos < len(plate_text.replace(" ", ""))):
                    label = plate_text.replace(" ", "").upper()[pos]
                    if label in VALID_CHARS:
                        label_certain = True
                    else:
                        label = "unknown"
                else:
                    label = "unknown"

                # Création du répertoire pour ce label
                label_dir = chars_dir / label
                label_dir.mkdir(exist_ok=True)

                char_filename = f"img_{char_counter:06d}.png"
                char_path = label_dir / char_filename
                cv2.imwrite(str(char_path), char_img_28)

                all_chars.append({
                    "char_path": str(char_path),
                    "label": label,
                    "plate_source": crop_info["source"],
                    "position": pos,
                    "bbox": (x, y, w, h),
                    "label_certain": label_certain,
                })
                char_counter += 1

                if label != "unknown":
                    label_distribution[label] = label_distribution.get(label, 0) + 1

        except Exception as e:
            logger.error(f"Phase 4 crop {crop_info.get('crop_path', '?')} : {e}")

    # Affichage de la distribution
    _display_char_distribution(label_distribution)
    logger.info(f"Phase 4 : {char_counter} caractères segmentés")
    return all_chars


def _segment_chars_from_plate(img: np.ndarray) -> list:
    """
    Segmente les caractères d'un crop de plaque 200×60.
    Retourne une liste de (x, y, w, h, char_img_28).
    """
    crop_H, crop_W = img.shape[:2]

    # Prétraitement
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Détection des contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_chars = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filtres de validation pour un caractère de plaque africaine
        if not (0.30 * crop_H < h < 0.90 * crop_H):
            continue
        if not (5 < w < 0.15 * crop_W):
            continue
        if h == 0 or not (1.0 < h / w < 4.0):
            continue
        if w * h <= 100:
            continue

        char_img = thresh[y:y + h, x:x + w]
        char_img_28 = cv2.resize(char_img, CHAR_TARGET_SIZE)
        valid_chars.append((x, y, w, h, char_img_28))

    # Tri gauche → droite
    valid_chars.sort(key=lambda c: c[0])
    return valid_chars


def _display_char_distribution(distribution: dict):
    """Affiche la distribution des classes labellisées, signale les < 50."""
    if not distribution:
        logger.info("Aucune transcription disponible — distribution vide.")
        return

    sorted_dist = sorted(distribution.items())
    parts = [f"{c}: {n}" for c, n in sorted_dist]
    logger.info("Distribution caractères : " + " | ".join(parts))

    low_classes = [c for c, n in sorted_dist if n < 50]
    if low_classes:
        logger.warning(
            f"Classes avec < 50 échantillons : {low_classes} "
            f"(augmentation recommandée en phase 5)"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — NETTOYAGE ET VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def clean_and_validate(chars_list: list, crops_list: list) -> tuple:
    """
    Nettoie et valide les images de caractères et de plaques.
    Équilibre les classes pour le Naïves Bayes (Module A1).

    Entrée  : liste chars, liste crops
    Sortie  : (chars_valides, crops_valides)
    Alimente : Module A1 (Naïves Bayes Gaussien)
    """
    n_avant = len(chars_list)
    n_noirs_blancs = 0
    n_doublons = 0
    n_label_invalide = 0
    n_flous = 0
    n_corrompus = 0

    valides = []
    seen_hashes = set()

    logger.info(f"Phase 5 : nettoyage de {n_avant} caractères...")
    for char_info in tqdm(chars_list, desc="Validation caractères"):
        try:
            img = cv2.imread(char_info["char_path"], cv2.IMREAD_GRAYSCALE)

            # Vérification forme
            if img is None or img.shape != (28, 28):
                n_corrompus += 1
                continue

            # Rejet noir/blanc total
            total_sum = int(img.sum())
            if total_sum < 10:
                n_noirs_blancs += 1
                continue
            if total_sum > 28 * 28 * 250:
                n_noirs_blancs += 1
                continue

            # Rejet flous
            blur_score = float(cv2.Laplacian(img, cv2.CV_64F).var())
            if blur_score < 5.0:
                n_flous += 1
                continue

            # Rejet doublons MD5
            img_hash = hashlib.md5(img.tobytes()).hexdigest()
            if img_hash in seen_hashes:
                n_doublons += 1
                continue
            seen_hashes.add(img_hash)

            # Validation du label
            label = str(char_info.get("label", "unknown")).strip().upper()
            if label != "UNKNOWN" and label not in VALID_CHARS:
                n_label_invalide += 1
                continue
            if label == "UNKNOWN":
                char_info["label"] = "unknown"
            else:
                char_info["label"] = label

            valides.append(char_info)

        except Exception as e:
            logger.error(f"Phase 5 validation {char_info.get('char_path', '?')} : {e}")
            n_corrompus += 1

    # Équilibrage des classes
    valides, n_augmented = _balance_classes(valides)

    n_apres = len(valides)
    distribution_finale = {}
    for c_info in valides:
        lbl = c_info["label"]
        if lbl != "unknown":
            distribution_finale[lbl] = distribution_finale.get(lbl, 0) + 1

    # Sauvegarde du rapport
    validation_report = {
        "chars_avant_nettoyage": n_avant,
        "chars_apres_nettoyage": n_apres,
        "rejetes_noirs_blancs": n_noirs_blancs,
        "rejetes_doublons": n_doublons,
        "rejetes_label_invalide": n_label_invalide,
        "rejetes_flous": n_flous,
        "rejetes_corrompus": n_corrompus,
        "chars_augmentes": n_augmented,
        "distribution_finale": distribution_finale,
    }

    report_path = Path("data/processed/validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2, ensure_ascii=False)
    logger.info(f"Rapport validation sauvegardé : {report_path}")
    logger.info(
        f"Phase 5 : {n_avant} → {n_apres} chars valides "
        f"({n_noirs_blancs} noirs/blancs, {n_doublons} doublons, "
        f"{n_flous} flous, {n_label_invalide} labels invalides)"
    )

    # Validation des crops (rejet corrompus uniquement)
    crops_valides = []
    for crop in crops_list:
        img = cv2.imread(crop["crop_path"])
        if img is not None:
            crops_valides.append(crop)

    return valides, crops_valides


def _balance_classes(chars_list: list) -> tuple:
    """
    Équilibre les classes par sous-échantillonnage (> 500) et augmentation (< 50).
    Retourne (liste_équilibrée, n_augmented).
    """
    from collections import defaultdict

    MAX_SAMPLES = 500
    MIN_SAMPLES = 50

    by_label = defaultdict(list)
    for c in chars_list:
        by_label[c["label"]].append(c)

    n_augmented = 0
    result = []

    for label, samples in by_label.items():
        if label == "unknown":
            result.extend(samples)
            continue

        # Sous-échantillonnage
        if len(samples) > MAX_SAMPLES:
            rng = random.Random(42)
            samples = rng.sample(samples, MAX_SAMPLES)

        # Augmentation légère
        if len(samples) < MIN_SAMPLES:
            needed = MIN_SAMPLES - len(samples)
            augmented = _augment_chars(samples, needed, label)
            samples = samples + augmented
            n_augmented += len(augmented)

        result.extend(samples)

    return result, n_augmented


def _augment_chars(samples: list, n_needed: int, label: str) -> list:
    """
    Augmentation légère : rotations ±5°, bruit gaussien σ=2, distorsion affine.
    Implémentée directement sans dépendance externe.
    """
    augmented = []
    rng = np.random.RandomState(42)

    chars_dir = Path(samples[0]["char_path"]).parent
    n_saved = int(1e6)  # compteur global approximatif

    for i in range(n_needed):
        src = samples[i % len(samples)]
        img = cv2.imread(src["char_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        aug_img = img.copy().astype(np.float32)

        # Rotation ±5°
        angle = rng.uniform(-5, 5)
        M_rot = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
        aug_img = cv2.warpAffine(aug_img, M_rot, (28, 28),
                                  borderValue=0).astype(np.float32)

        # Bruit gaussien σ=2
        noise = rng.normal(0, 2, aug_img.shape).astype(np.float32)
        aug_img = np.clip(aug_img + noise, 0, 255)

        # Distorsion affine légère
        pts1 = np.float32([[0, 0], [27, 0], [0, 27]])
        dx = rng.uniform(-1, 1, 3)
        dy = rng.uniform(-1, 1, 3)
        pts2 = pts1 + np.column_stack([dx, dy]).astype(np.float32)
        M_aff = cv2.getAffineTransform(pts1, pts2)
        aug_img = cv2.warpAffine(aug_img, M_aff, (28, 28),
                                  borderValue=0)

        aug_img = aug_img.astype(np.uint8)
        aug_filename = f"aug_{n_saved + i:07d}.png"
        aug_path = chars_dir / aug_filename
        cv2.imwrite(str(aug_path), aug_img)

        new_info = dict(src)
        new_info["char_path"] = str(aug_path)
        new_info["label_certain"] = src.get("label_certain", False)
        augmented.append(new_info)

    return augmented


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — EXTRACTION DES FEATURES MANUELLES
# ══════════════════════════════════════════════════════════════════════════════

def extract_all_features(chars_valides: list, crops_valides: list) -> tuple:
    """
    Extrait les features manuelles pour le Naïves Bayes Gaussien (Module A1).
    Features : histogramme HSV, gradient, ratio pixels, cadre métallique,
               dimensions, densité — vecteur 20D par caractère (pipeline interne).
    Note     : preprocessing/feature_extraction.py expose la version 175D
               alignée sur le cahier des charges (75D char + 100D plaque).

    Entrée  : chars_valides (liste dicts), crops_valides (liste dicts)
    Sortie  : (X: ndarray (N,20) float32, y: ndarray (N,) int32)
    Alimente : Module A1 (Naïves Bayes Gaussien)
    """
    # Cache des features de plaques (coûteux à calculer)
    crop_features_cache = {}
    logger.info(f"Phase 6 : extraction features plaques ({len(crops_valides)} crops)...")

    for crop_info in tqdm(crops_valides, desc="Features plaques"):
        crop_path = crop_info["crop_path"]
        if crop_path in crop_features_cache:
            continue
        try:
            img = cv2.imread(crop_path)
            if img is None:
                crop_features_cache[crop_path] = np.zeros(100, dtype=np.float32)
                continue
            feat = _extract_plate_features(img)
            crop_features_cache[crop_path] = feat
        except Exception as e:
            logger.error(f"Features plaque {crop_path} : {e}")
            crop_features_cache[crop_path] = np.zeros(100, dtype=np.float32)

    # Construction de X et y
    labeled_chars = [c for c in chars_valides if c["label"] in LABEL_TO_IDX]
    logger.info(
        f"Phase 6 : extraction features caractères "
        f"({len(labeled_chars)} chars labellisés)..."
    )

    X_list = []
    y_list = []

    for char_info in tqdm(labeled_chars, desc="Features caractères"):
        try:
            img = cv2.imread(char_info["char_path"], cv2.IMREAD_GRAYSCALE)
            if img is None or img.shape != (28, 28):
                continue

            feat_char = _extract_char_features(img)

            # Récupération des features de la plaque parente
            original_img = char_info.get("original_image", "")
            # Chercher le crop correspondant
            parent_crop_feat = np.zeros(100, dtype=np.float32)
            for crop_info in crops_valides:
                if crop_info.get("original_image") == original_img:
                    parent_crop_feat = crop_features_cache.get(
                        crop_info["crop_path"],
                        np.zeros(100, dtype=np.float32)
                    )
                    break

            feat_final = np.concatenate([feat_char, parent_crop_feat])
            X_list.append(feat_final.astype(np.float32))
            y_list.append(LABEL_TO_IDX[char_info["label"]])

        except Exception as e:
            logger.error(f"Features char {char_info.get('char_path', '?')} : {e}")

    if not X_list:
        logger.warning("Phase 6 : aucun caractère labellisé — X vide.")
        return np.zeros((0, 120), dtype=np.float32), np.zeros(0, dtype=np.int32)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    logger.info(f"Phase 6 terminée : X.shape={X.shape}, y.shape={y.shape}")
    return X, y


def _extract_char_features(img: np.ndarray) -> np.ndarray:
    """
    Extrait 20 features manuelles d'une image de caractère 28×28.

    Groupe A (4) : ratio h/w, densité, centroïde cx/28, cy/28
    Groupe B (8) : densité par zone dans grille 4×2
    Groupe C (4) : transitions 0→1 par lignes et colonnes
    Groupe D (4) : gradients Sobel X et Y
    """
    img_f = img.astype(np.float32) / 255.0
    H, W = img.shape

    # ── Groupe A : dimensions et densité ────────────────────────────────────
    # Ratio h/w réel (ici toujours 1 car 28×28, mais on utilise la valeur brute)
    ratio_hw = H / W

    # Densité = proportion pixels > 128
    density = float(np.count_nonzero(img > 128)) / (H * W)

    # Centroïde relatif
    binary = (img > 128).astype(np.float32)
    total = binary.sum() + 1e-6
    ys, xs = np.mgrid[0:H, 0:W]
    cx = float((xs * binary).sum() / total) / W
    cy = float((ys * binary).sum() / total) / H

    feat_A = np.array([ratio_hw, density, cx, cy], dtype=np.float32)

    # ── Groupe B : densité par zone (grille 4 colonnes × 2 lignes) ──────────
    feat_B = np.zeros(8, dtype=np.float32)
    zone_h = H // 2
    zone_w = W // 4
    for row in range(2):
        for col in range(4):
            y0 = row * zone_h
            x0 = col * zone_w
            zone = binary[y0:y0 + zone_h, x0:x0 + zone_w]
            feat_B[row * 4 + col] = float(zone.mean()) if zone.size > 0 else 0.0

    # ── Groupe C : transitions 0→1 ──────────────────────────────────────────
    bin_uint8 = binary.astype(np.uint8)
    row_trans = []
    for r in range(H):
        diff = np.diff(bin_uint8[r])
        row_trans.append(int(np.count_nonzero(diff == 1)))

    col_trans = []
    for c in range(W):
        diff = np.diff(bin_uint8[:, c])
        col_trans.append(int(np.count_nonzero(diff == 1)))

    feat_C = np.array([
        float(np.mean(row_trans)),
        float(np.std(row_trans)),
        float(np.mean(col_trans)),
        float(np.std(col_trans)),
    ], dtype=np.float32)

    # ── Groupe D : gradients Sobel ───────────────────────────────────────────
    img_u8 = img
    sobel_x = cv2.Sobel(img_u8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_u8, cv2.CV_64F, 0, 1, ksize=3)

    feat_D = np.array([
        float(sobel_x.mean()),
        float(sobel_x.std()),
        float(sobel_y.mean()),
        float(sobel_y.std()),
    ], dtype=np.float32)

    return np.concatenate([feat_A, feat_B, feat_C, feat_D])  # 4+8+4+4 = 20


def _extract_plate_features(img: np.ndarray) -> np.ndarray:
    """
    Extrait 100 features manuelles d'un crop de plaque 200×60.

    a) Histogramme HSV (96) : H(32) + S(32) + V(32) normalisés
    b) Gradient moyen (2)   : magnitude Sobel moyenne + écart-type
    c) Ratio pixels foncés  (1) : count(pixel<128) / total en gris
    d) Présence cadre métal (1) : binaire sur bordure 5px
    """
    # ── a) Histogramme HSV ───────────────────────────────────────────────────
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()

    # Normalisation à 1.0
    def _norm(h):
        s = h.sum()
        return h / s if s > 0 else h

    feat_hsv = np.concatenate([_norm(hist_h), _norm(hist_s), _norm(hist_v)])

    # ── b) Gradient moyen ────────────────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    feat_grad = np.array([
        float(magnitude.mean()),
        float(magnitude.std()),
    ], dtype=np.float32)

    # ── c) Ratio pixels foncés ───────────────────────────────────────────────
    total_px = gray.size
    dark_ratio = float(np.count_nonzero(gray < 128)) / total_px
    feat_dark = np.array([dark_ratio], dtype=np.float32)

    # ── d) Présence cadre métallique ─────────────────────────────────────────
    H, W = img.shape[:2]
    BORDER = 5
    borders = np.concatenate([
        hsv[:BORDER, :, :].reshape(-1, 3),           # haut
        hsv[-BORDER:, :, :].reshape(-1, 3),          # bas
        hsv[:, :BORDER, :].reshape(-1, 3),           # gauche
        hsv[:, -BORDER:, :].reshape(-1, 3),          # droite
    ])
    # Métal argenté : S < 30 et V > 150
    metal_mask = (borders[:, 1] < 30) & (borders[:, 2] > 150)
    metal_ratio = float(metal_mask.sum()) / len(metal_mask) if len(borders) > 0 else 0.0
    feat_metal = np.array([1.0 if metal_ratio > 0.60 else 0.0], dtype=np.float32)

    return np.concatenate([feat_hsv, feat_grad, feat_dark, feat_metal])  # 96+2+1+1 = 100


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 8 — SAUVEGARDE ET SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def save_and_split(X: np.ndarray, y: np.ndarray,
                   crops_valides: list, datasets_dict: dict,
                   output_dir: Path) -> dict:
    """
    Effectue les splits train/val/test, sauvegarde tous les artefacts.

    Entrée  : X (N,20), y (N,), crops validés, répertoire de sortie
    Note    : le module preprocessing/feature_extraction.py produit les
              vecteurs 175D (cahier des charges) à utiliser pour le NB final.
    Sortie  : dictionnaire récapitulatif
    Alimente : Module A1 (features.npy), Module A2 (images YOLO)
    """
    from sklearn.model_selection import train_test_split

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── 7.1 Split des caractères ─────────────────────────────────────────────
    if len(X) > 0 and len(np.unique(y)) > 1:
        # Premier split : 70% train / 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=42
        )
        # Second split : 15% val / 15% test depuis le temp
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
        )

        np.save(str(output_dir / "features.npy"), X_train)
        np.save(str(output_dir / "labels.npy"), y_train)
        np.save(str(output_dir / "features_val.npy"), X_val)
        np.save(str(output_dir / "labels_val.npy"), y_val)
        np.save(str(output_dir / "features_test.npy"), X_test)
        np.save(str(output_dir / "labels_test.npy"), y_test)

        # class_names.txt : une classe par ligne, position = index dans LABEL_TO_IDX
        # Doit correspondre exactement à VALID_CHARS (sorted alphanum)
        # → naive_bayes.py lit ce fichier pour reconstruire les noms de classes
        with open(output_dir / "class_names.txt", "w") as f:
            f.write("\n".join(VALID_CHARS))

        results["chars"] = {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
            "total": len(X),
        }
        logger.info(
            f"Phase 7.1 : chars train={len(X_train)}, "
            f"val={len(X_val)}, test={len(X_test)}"
        )
    else:
        logger.warning("Phase 7.1 : pas assez de données labellisées pour le split.")
        results["chars"] = {"train": 0, "val": 0, "test": 0, "total": 0}

    # ── 7.2 Split des images YOLO ────────────────────────────────────────────
    results["images"] = _split_yolo_images(crops_valides, output_dir)

    # ── 7.3 Fichier YAML YOLOv8 ─────────────────────────────────────────────
    _create_yolo_yaml(output_dir)

    # ── 7.4 Métadonnées du dataset ───────────────────────────────────────────
    metadata = {
        "sources": ["roboflow", "mendeley"],
        "urls": {
            "roboflow": "https://universe.roboflow.com/nigerianlpd/nigerian-license-plate",
            "mendeley": "https://data.mendeley.com/datasets/74zz6dj9vn/2",
        },
        "total_images_plaques": len(crops_valides),
        "total_chars_labellises": int(results["chars"]["total"]),
        "split": {
            "train": results["chars"].get("train", 0),
            "val": results["chars"].get("val", 0),
            "test": results["chars"].get("test", 0),
        },
        "feature_dim": 20,   # pipeline interne ; cahier des charges : 175D (preprocessing/feature_extraction.py)
        "n_classes_chars": 36,
        "label_to_idx": LABEL_TO_IDX,
        "biais_identifies": [
            "Dataset principalement nigérian — biais géographique vs plaques camerounaises",
            "Sous-représentation conditions nocturnes",
            "Format CEMAC potentiellement absent",
            "Dataset Mendeley ghanéen — variété régionale limitée au Ghana",
        ],
        "date_creation": str(date.today()),
        "roboflow_stats": {
            "n_images": datasets_dict["roboflow"]["n_images"],
            "splits": ["train", "valid", "test"],
        },
        "mendeley_stats": {
            "n_images": datasets_dict["mendeley"]["n_images"],
            "annotation_format": datasets_dict["mendeley"]["annotation_format"],
            "regions": _get_mendeley_regions(datasets_dict["mendeley"]["images"]),
        },
    }

    metadata_path = output_dir / "dataset_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Métadonnées sauvegardées : {metadata_path}")

    # ── 7.5 Rapport final ────────────────────────────────────────────────────
    _print_final_report(results)

    return results


def _split_yolo_images(crops_valides: list, output_dir: Path) -> dict:
    """
    Effectue le split 70/15/15 des images YOLO et les copie avec letterbox.
    """
    import shutil

    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    n = len(crops_valides)
    if n == 0:
        return {"train": 0, "val": 0, "test": 0}

    indices = list(range(n))
    rng = random.Random(42)
    rng.shuffle(indices)

    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    splits_idx = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }

    counts = {}
    for split_name, idxs in splits_idx.items():
        counts[split_name] = 0
        for i, idx in enumerate(tqdm(idxs, desc=f"YOLO split {split_name}")):
            crop_info = crops_valides[idx]
            src_path = Path(crop_info["crop_path"])

            img = cv2.imread(str(src_path))
            if img is None:
                continue

            # Letterbox 640×640
            img_lb = _letterbox(img, YOLO_TARGET_SIZE)

            dst_img = output_dir / "images" / split_name / f"{i:06d}.jpg"
            cv2.imwrite(str(dst_img), img_lb)

            # Annotation YOLO : classe 0, plaque = image entière normalisée
            dst_lbl = output_dir / "labels" / split_name / f"{i:06d}.txt"
            with open(dst_lbl, "w") as f:
                f.write("0 0.5 0.5 1.0 1.0\n")

            counts[split_name] += 1

    return counts


def _letterbox(img: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Redimensionne une image en conservant le ratio, avec padding gris (114,114,114).
    """
    tw, th = target_size
    H, W = img.shape[:2]
    scale = min(tw / W, th / H)
    new_W = int(W * scale)
    new_H = int(H * scale)

    resized = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_AREA)
    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)

    pad_x = (tw - new_W) // 2
    pad_y = (th - new_H) // 2
    canvas[pad_y:pad_y + new_H, pad_x:pad_x + new_W] = resized
    return canvas


def _create_yolo_yaml(output_dir: Path):
    """Crée le fichier de configuration YOLOv8."""
    models_configs = Path("models/configs")
    models_configs.mkdir(parents=True, exist_ok=True)

    yaml_content = f"""# PlateVision — Configuration YOLOv8
# Généré automatiquement par data/prepare_datasets.py
path: {str(output_dir.resolve())}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: plaque_immatriculation
"""
    yaml_path = models_configs / "yolov8_platevision.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    logger.info(f"YAML YOLOv8 créé : {yaml_path}")


def _get_mendeley_regions(images: list) -> dict:
    """Retourne le décompte d'images par région Mendeley."""
    regions = {}
    for img_path in images:
        region = img_path.parent.name
        regions[region] = regions.get(region, 0) + 1
    return regions


def _print_final_report(results: dict):
    """Affiche le rapport final du pipeline."""
    c = results.get("chars", {})
    i = results.get("images", {})

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║       PLATEVISION — PRÉTRAITEMENT TERMINÉ                   ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  CARACTÈRES (→ Module A1 Naïves Bayes)                     ║")
    print(f"║    Train : {c.get('train',0):<6d}  |  Val : {c.get('val',0):<6d}  |  Test : {c.get('test',0):<6d}  ║")
    print("║    Classes : 36  |  Features : 120D                        ║")
    print("║    → data/processed/features.npy                           ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  IMAGES PLAQUES (→ Module A2 YOLO+OCR)                     ║")
    print(f"║    Train : {i.get('train',0):<6d}  |  Val : {i.get('val',0):<6d}  |  Test : {i.get('test',0):<6d}  ║")
    print("║    → data/processed/images/                                ║")
    print("║    → models/configs/yolov8_platevision.yaml                ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  PROCHAINE ÉTAPE                                            ║")
    print("║    python main.py --module A --input data/processed/        ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")


# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES INTER-PHASES
# ══════════════════════════════════════════════════════════════════════════════

def _source_from_crop_path(p: Path) -> str:
    """Infère la source d'un crop depuis le préfixe de son nom de fichier."""
    name = p.name
    if name.startswith("roboflow_"):
        return "roboflow"
    if name.startswith("mendeley_"):
        return "mendeley"
    return "unknown"


def _load_crops_from_disk(crops_dir: Path) -> tuple:
    """
    Charge tous les crops d'un répertoire et les sépare par source
    (inférée depuis le préfixe du nom de fichier : roboflow_* / mendeley_*).
    Retourne : (plate_crops_rf, plate_crops_md)
    """
    if not crops_dir.exists():
        return [], []
    all_crops = [
        {
            "crop_path":        str(p),
            "plate_text":       None,
            "source":           _source_from_crop_path(p),
            "detection_method": "loaded",
            "original_image":   "",
            "blur_score":       0.0,
        }
        for p in sorted(crops_dir.glob("*.jpg"))
    ]
    rf = [c for c in all_crops if c["source"] == "roboflow"]
    md = [c for c in all_crops if c["source"] != "roboflow"]
    return rf, md


# ══════════════════════════════════════════════════════════════════════════════
# GESTION DE L'ÉTAT DU PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

PIPELINE_STATE_PATH = Path("data/processed/.pipeline_state.json")


def _save_pipeline_state(phase: int, data: dict):
    """Sauvegarde l'état du pipeline après chaque phase pour reprise possible."""
    PIPELINE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    if PIPELINE_STATE_PATH.exists():
        with open(PIPELINE_STATE_PATH, "r") as f:
            state = json.load(f)

    state[f"phase_{phase}"] = {
        "completed": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **data,
    }
    with open(PIPELINE_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _load_pipeline_state() -> dict:
    """Charge l'état du pipeline depuis le fichier de persistance."""
    if not PIPELINE_STATE_PATH.exists():
        return {}
    with open(PIPELINE_STATE_PATH, "r") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Point d'entrée du pipeline PlateVision.
    Exécute les 8 phases de préparation des données.
    Supporte la reprise à partir d'une phase donnée (--from-phase) ou
    l'exécution d'une phase unique (--phase-only).

    Numérotation des phases :
      1 — Acquisition
      2 — EDA
      3 — Extraction régions de plaques
      4 — OCR sur les crops (Phase 3bis)
      5 — Segmentation des caractères
      6 — Nettoyage et validation
      7 — Extraction des features manuelles
      8 — Sauvegarde et split
    """
    parser = argparse.ArgumentParser(
        description="PlateVision — Pipeline de préparation des datasets"
    )
    parser.add_argument(
        "--from-phase",
        type=int,
        default=1,
        choices=range(1, 9),
        metavar="N",
        help="Reprendre à partir de la phase N (1-8)"
    )
    parser.add_argument(
        "--phase-only",
        type=int,
        default=None,
        choices=range(1, 9),
        metavar="N",
        help="Exécuter uniquement la phase N (1-8)"
    )
    args = parser.parse_args()

    start_time = time.time()
    random.seed(42)
    np.random.seed(42)

    # Chemins des données brutes
    ROBOFLOW_PATH = Path("data/raw/roboflow")
    MENDELEY_PATH = Path("data/raw/mendeley")
    OUTPUT_DIR = Path("data/processed")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Création du répertoire de logs si absent
    (OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    state = _load_pipeline_state()

    # Détermination des phases à exécuter
    if args.phase_only is not None:
        phases_to_run = {args.phase_only}
    else:
        phases_to_run = set(range(args.from_phase, 9))

    logger.info(f"PlateVision — Phases à exécuter : {sorted(phases_to_run)}")

    # Variables inter-phases
    datasets_dict    = None
    eda_report       = None
    plate_crops_rf   = None          # Roboflow → A1 caractères (Phases 4-7)
    plate_crops_md   = None          # Mendeley  → A2 YOLO uniquement
    chars_list       = None
    chars_valides    = None
    crops_valides_rf = None          # crops Roboflow validés → extract_all_features
    crops_valides_md = None          # crops Mendeley validés → split YOLO
    crops_valides    = None          # union rf + md → Phase 8 YOLO split
    X = None
    y = None

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if 1 in phases_to_run:
        logger.info("═" * 60)
        logger.info("PHASE 1 — ACQUISITION")
        logger.info("═" * 60)
        datasets_dict = load_raw_datasets(ROBOFLOW_PATH, MENDELEY_PATH)
        _save_pipeline_state(1, {
            "n_roboflow": datasets_dict["roboflow"]["n_images"],
            "n_mendeley": datasets_dict["mendeley"]["n_images"],
        })

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    if 2 in phases_to_run:
        logger.info("═" * 60)
        logger.info("PHASE 2 — EDA")
        logger.info("═" * 60)
        if datasets_dict is None:
            logger.info("Rechargement des données pour phase 2...")
            datasets_dict = load_raw_datasets(ROBOFLOW_PATH, MENDELEY_PATH)
        eda_report = quick_eda(datasets_dict, OUTPUT_DIR)
        _save_pipeline_state(2, {
            "total_images": eda_report["total_images"],
            "n_blurry": len(eda_report["blurry_images"]),
        })

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    if 3 in phases_to_run:
        logger.info("═" * 60)
        logger.info("PHASE 3 — EXTRACTION RÉGIONS PLAQUES")
        logger.info("═" * 60)
        if datasets_dict is None:
            datasets_dict = load_raw_datasets(ROBOFLOW_PATH, MENDELEY_PATH)
        # extract_plate_regions() retourne deux listes séparées dès la source
        plate_crops_rf, plate_crops_md = extract_plate_regions(datasets_dict, OUTPUT_DIR)
        _save_pipeline_state(3, {
            "n_crops":    len(plate_crops_rf) + len(plate_crops_md),
            "n_crops_rf": len(plate_crops_rf),
            "n_crops_md": len(plate_crops_md),
            "crops_file": str(OUTPUT_DIR / "plate_crops"),
        })

    # ── Phase 4 — OCR ─────────────────────────────────────────────────────────
    if 4 in phases_to_run:
        logger.info("═" * 60)
        logger.info("PHASE 4 — OCR SUR LES CROPS ROBOFLOW (labellisation A1)")
        logger.info("═" * 60)
        if plate_crops_rf is None:
            plate_crops_rf, plate_crops_md = _load_crops_from_disk(
                OUTPUT_DIR / "plate_crops"
            )
            logger.info(
                f"Phase 4 : {len(plate_crops_rf)} crops Roboflow + "
                f"{len(plate_crops_md)} crops Mendeley chargés depuis le disque."
            )

        # OCR uniquement sur les crops Roboflow — Mendeley n'alimente pas A1
        plate_crops_rf = run_ocr_on_crops(plate_crops_rf, OUTPUT_DIR)
        n_with_text = sum(1 for c in plate_crops_rf if c.get("plate_text"))
        _save_pipeline_state(4, {
            "n_crops_rf":  len(plate_crops_rf),
            "n_with_text": n_with_text,
            "ocr_results": str(OUTPUT_DIR / "ocr_results.json"),
        })

    # ── Phase 5 ───────────────────────────────────────────────────────────────
    if 5 in phases_to_run:
        logger.info("═" * 60)
        logger.info("PHASE 5 — SEGMENTATION CARACTÈRES (Roboflow uniquement)")
        logger.info("═" * 60)
        if plate_crops_rf is None:
            plate_crops_rf, plate_crops_md = _load_crops_from_disk(
                OUTPUT_DIR / "plate_crops"
            )
            logger.info(
                f"Phase 5 : {len(plate_crops_rf)} crops Roboflow chargés "
                f"({len(plate_crops_md)} Mendeley ignorés pour la segmentation)."
            )

        # Seuls les crops Roboflow produisent des caractères labellisés pour A1
        chars_list = segment_characters(plate_crops_rf, OUTPUT_DIR)
        _save_pipeline_state(5, {
            "n_chars":      len(chars_list),
            "source_chars": "roboflow",
        })

    # ── Phase 6 ───────────────────────────────────────────────────────────────
    if 6 in phases_to_run:
        logger.info("═" * 60)
        logger.info("PHASE 6 — NETTOYAGE ET VALIDATION")
        logger.info("═" * 60)
        if chars_list is None:
            # Reconstituer depuis le dossier characters (issus de Roboflow uniquement)
            chars_dir = OUTPUT_DIR / "characters"
            chars_list = []
            if chars_dir.exists():
                for label_dir in chars_dir.iterdir():
                    if label_dir.is_dir():
                        for img_path in label_dir.glob("*.png"):
                            chars_list.append({
                                "char_path": str(img_path),
                                "label": label_dir.name,
                                "plate_source": "roboflow",
                                "position": 0,
                                "bbox": (0, 0, 0, 0),
                                "label_certain": label_dir.name in VALID_CHARS,
                            })
            logger.info(f"Phase 6 : {len(chars_list)} chars Roboflow chargés depuis le disque.")

        if plate_crops_rf is None:
            plate_crops_rf, plate_crops_md = _load_crops_from_disk(
                OUTPUT_DIR / "plate_crops"
            )

        # Validation complète (déduplication, flou, labels) sur chars + crops rf
        chars_valides, crops_valides_rf = clean_and_validate(chars_list, plate_crops_rf)

        # Mendeley EXCLU du split YOLO (Phase 8) : ces images sont des photos de scène
        # complète (véhicule + décor), pas des gros plans de plaque, et le dataset ne
        # fournit aucune annotation de localisation réelle. Le label "image entière =
        # plaque" qui leur était appliqué était donc faux et a entraîné YOLO à détecter
        # la scène entière au lieu de la plaque (bbox ≈ 99% de l'image en pratique).
        # Mendeley reste cependant valide et chargé pour analyse exploratoire (Phase 2).
        crops_valides_md = []
        logger.warning(
            f"Mendeley : {len(plate_crops_md or [])} images exclues du split YOLO "
            f"(Phase 8) — aucune annotation de localisation réelle disponible, "
            f"'image entière = plaque' est faux pour ce sous-ensemble (photos de scène)."
        )

        # Phase 8 : YOLO entraîné uniquement sur Roboflow (seule source avec bbox réelles)
        crops_valides = crops_valides_rf

        _save_pipeline_state(6, {
            "n_chars_valides":    len(chars_valides),
            "n_crops_rf_valides": len(crops_valides_rf),
            "n_crops_md_valides": len(crops_valides_md),
            "source_chars":       "roboflow",
        })

    # ── Phase 7 ───────────────────────────────────────────────────────────────
    if 7 in phases_to_run:
        logger.info("═" * 60)
        logger.info("PHASE 7 — EXTRACTION FEATURES MANUELLES")
        logger.info("═" * 60)
        if chars_valides is None or crops_valides_rf is None:
            logger.error("Phase 7 nécessite les résultats de la phase 6. "
                         "Relancer avec --from-phase 6")
            sys.exit(1)
        # Extraction sur crops Roboflow uniquement — les features Mendeley
        # ne doivent pas contaminer features.npy / labels.npy (A1)
        X, y = extract_all_features(chars_valides, crops_valides_rf)
        _save_pipeline_state(7, {
            "X_shape": list(X.shape) if X is not None else [],
            "y_shape": list(y.shape) if y is not None else [],
        })

    # ── Phase 8 ───────────────────────────────────────────────────────────────
    if 8 in phases_to_run:
        logger.info("═" * 60)
        logger.info("PHASE 8 — SAUVEGARDE ET SPLIT")
        logger.info("═" * 60)
        if X is None or y is None:
            logger.error("Phase 8 nécessite les features de la phase 7. "
                         "Relancer avec --from-phase 7")
            sys.exit(1)
        if datasets_dict is None:
            datasets_dict = load_raw_datasets(ROBOFLOW_PATH, MENDELEY_PATH)
        results = save_and_split(X, y, crops_valides, datasets_dict, OUTPUT_DIR)
        _save_pipeline_state(8, results)

    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    logger.info(f"Pipeline terminé en {minutes}m {seconds}s")


if __name__ == "__main__":
    main()
