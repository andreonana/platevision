"""
PlateVision — Module A2 : Pipeline YOLO + OCR.

Ce module implémente le pipeline de détection de plaques d'immatriculation
avec YOLOv8 (ultralytics) suivi de la reconnaissance des caractères avec
EasyOCR. Il gère les images statiques et les flux vidéo.

Rôle dans le pipeline : détection + lecture des plaques (approche principale).
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Seuils de confiance depuis les variables d'environnement
CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", 0.5))
IOU_THRESHOLD = float(os.getenv("YOLO_IOU_THRESHOLD", 0.45))
IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", 640))
OCR_CONFIDENCE = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", 0.3))


def load_yolo_model(weights_path: Optional[str] = None):
    """
    Charge le modèle YOLOv8 depuis les poids spécifiés ou le modèle de base.

    Args:
        weights_path: Chemin vers les poids .pt entraînés (None = YOLOv8n de base)

    Returns:
        Modèle YOLO chargé
    """
    from ultralytics import YOLO

    if weights_path and Path(weights_path).exists():
        logger.info(f"Chargement des poids YOLO : {weights_path}")
        model = YOLO(weights_path)
    else:
        # Modèle nano préentraîné sur COCO (base de départ pour fine-tuning)
        logger.warning("Poids personnalisés non trouvés. Utilisation de YOLOv8n (base).")
        model = YOLO("yolov8n.pt")

    return model


def load_ocr_reader(languages: Optional[List[str]] = None):
    """
    Initialise le lecteur EasyOCR pour les langues spécifiées.

    Args:
        languages: Liste des codes de langue (ex: ['fr', 'en'])

    Returns:
        Lecteur EasyOCR initialisé
    """
    import easyocr

    if languages is None:
        lang_str = os.getenv("OCR_LANGUAGES", "en")
        languages = [l.strip() for l in lang_str.split(",")]

    logger.info(f"Initialisation EasyOCR — langues : {languages}")
    # use_gpu=True si GPU disponible selon la variable d'environnement
    use_gpu = os.getenv("USE_GPU", "True").lower() == "true"
    reader = easyocr.Reader(languages, gpu=use_gpu)
    return reader


def detect_plates(model, image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Détecte les plaques d'immatriculation dans une image avec YOLO.

    Args:
        model: Modèle YOLOv8 chargé
        image: Image BGR (numpy array)

    Returns:
        Liste de détections avec bbox, confiance et classe
    """
    results = model(
        image,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
                "class_id": class_id,
            })

    return detections


def recognize_text(reader, image_crop: np.ndarray) -> str:
    """
    Reconnaît le texte sur un recadrage de plaque avec EasyOCR.

    Args:
        reader: Lecteur EasyOCR initialisé
        image_crop: Recadrage de la région de la plaque (BGR)

    Returns:
        Texte reconnu (chaîne vide si aucune lecture fiable)
    """
    # Conversion BGR → RGB pour EasyOCR
    image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
    ocr_results = reader.readtext(image_rgb)

    # Concaténation des textes au-dessus du seuil de confiance OCR
    texts = [
        text
        for (_, text, prob) in ocr_results
        if prob >= OCR_CONFIDENCE
    ]

    plate_text = " ".join(texts).strip().upper()
    return plate_text


def process_image(
    model,
    reader,
    image_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Traite une image : détection des plaques et reconnaissance des textes.

    Args:
        model: Modèle YOLO
        reader: Lecteur EasyOCR
        image_path: Chemin vers l'image à traiter
        output_path: Dossier de sauvegarde des images annotées

    Returns:
        Dictionnaire avec les plaques détectées et leurs textes
    """
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Impossible de charger l'image : {image_path}")
        return {"image": str(image_path), "plates": []}

    detections = detect_plates(model, image)
    plates = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        # Recadrage de la région de la plaque
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        text = recognize_text(reader, crop)
        plates.append({
            "bbox": det["bbox"],
            "confidence": det["confidence"],
            "text": text,
        })

        # Annotation de l'image originale
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{text} ({det['confidence']:.2f})"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Sauvegarde de l'image annotée
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"annotated_{image_path.name}"
    cv2.imwrite(str(output_file), image)

    logger.debug(f"{image_path.name} : {len(plates)} plaque(s) détectée(s)")
    return {"image": str(image_path), "plates": plates}


def run_yolo_ocr(input_path: Path, output_path: Path) -> List[Dict[str, Any]]:
    """
    Fonction principale : traite toutes les images d'un dossier avec YOLO+OCR.

    Args:
        input_path: Dossier d'images ou fichier vidéo
        output_path: Dossier de sauvegarde des résultats

    Returns:
        Liste des résultats par image
    """
    from tqdm import tqdm

    weights_path = os.getenv("MODELS_WEIGHTS_PATH", "models/weights/")
    model_file = Path(weights_path) / "best.pt"

    model = load_yolo_model(str(model_file) if model_file.exists() else None)
    reader = load_ocr_reader()

    # Traitement d'un dossier d'images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        logger.warning(f"Aucune image trouvée dans {input_path}")
        return []

    logger.info(f"Traitement de {len(image_files)} image(s)...")
    all_results = []

    for image_path in tqdm(image_files, desc="YOLO+OCR"):
        result = process_image(model, reader, image_path, output_path / "annotated")
        all_results.append(result)

    # Affichage d'un résumé
    total_plates = sum(len(r["plates"]) for r in all_results)
    logger.info(f"Pipeline terminé : {total_plates} plaque(s) détectée(s) sur {len(image_files)} images")

    return all_results
