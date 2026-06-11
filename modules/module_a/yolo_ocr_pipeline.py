"""
Module A2 — Pipeline de détection YOLOv8 et reconnaissance OCR (EasyOCR).

Rôle dans le pipeline PlateVision :
    Détecte les plaques d'immatriculation dans des images ou flux vidéo via
    YOLOv8, puis extrait le texte de chaque plaque détectée avec EasyOCR.
    Fournit les embeddings CNN pour le Module B.

Responsable : Étudiant 2
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", 0.5))
IOU_THRESHOLD = float(os.getenv("YOLO_IOU_THRESHOLD", 0.45))
OCR_CONFIDENCE = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", 0.7))
WEIGHTS_PATH = os.getenv("YOLO_WEIGHTS_PATH", "models/weights/yolov8_platevision.pt")
DEVICE = os.getenv("DEVICE", "cpu")


def load_yolo_model(weights_path: str):
    """Charge le modèle YOLOv8 depuis les poids entraînés."""
    from ultralytics import YOLO
    model = YOLO(weights_path)
    logger.info("Modèle YOLOv8 chargé : %s", weights_path)
    return model


def load_ocr_reader(languages: list[str] | None = None):
    """Initialise le lecteur EasyOCR pour les langues spécifiées."""
    import easyocr
    if languages is None:
        languages = ["en"]
    reader = easyocr.Reader(languages, gpu=(DEVICE == "cuda"))
    logger.info("EasyOCR initialisé pour les langues : %s", languages)
    return reader


def detect_plates(model, image: np.ndarray) -> list[dict]:
    """
    Détecte les plaques dans une image avec YOLOv8.

    Args:
        model: Modèle YOLOv8 chargé
        image: Image BGR (numpy array)

    Returns:
        Liste de dicts avec 'bbox' (x1,y1,x2,y2), 'confidence', 'crop'
    """
    results = model.predict(
        source=image,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=DEVICE,
        verbose=False,
    )

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            crop = image[y1:y2, x1:x2]
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
                "crop": crop,
            })

    logger.debug("%d plaque(s) détectée(s)", len(detections))
    return detections


def recognize_text(reader, plate_crop: np.ndarray) -> str:
    """
    Extrait le texte d'un crop de plaque via EasyOCR.

    Args:
        reader: Instance EasyOCR
        plate_crop: Image recadrée de la plaque (numpy array)

    Returns:
        Texte de la plaque (chaîne concaténée des segments fiables).
    """
    ocr_results = reader.readtext(plate_crop)
    text_parts = [
        text for (_, text, conf) in ocr_results if conf >= OCR_CONFIDENCE
    ]
    return " ".join(text_parts).strip().upper()


def extract_embeddings(model, image: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    Extrait l'embedding CNN de la backbone YOLOv8 pour une région de plaque.
    Ces embeddings sont utilisés comme entrée du Module B (clustering).
    """
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(512)

    # Redimensionnement au format attendu par la backbone
    crop_resized = cv2.resize(crop, (64, 64))
    # Normalisation [0, 1]
    crop_norm = crop_resized.astype(np.float32) / 255.0

    # L'embedding réel nécessite un forward pass sur la backbone — placeholder
    embedding = crop_norm.flatten()[:512]
    return embedding


def run_yolo_ocr_pipeline(
    input_path: Path,
    output_path: Path,
    save_embeddings: bool = True,
) -> list[dict]:
    """
    Exécute le pipeline complet YOLO + OCR sur un dossier d'images.

    Args:
        input_path: Dossier d'images ou fichier vidéo
        output_path: Dossier de sortie
        save_embeddings: Si True, sauvegarde les embeddings dans models/

    Returns:
        Liste des détections avec texte OCR reconnu.
    """
    logger.info("=== Pipeline YOLO + OCR démarré ===")

    model = load_yolo_model(WEIGHTS_PATH)
    reader = load_ocr_reader()

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    if input_path.is_dir():
        image_files = [
            f for f in input_path.iterdir() if f.suffix.lower() in image_extensions
        ]
    else:
        image_files = [input_path]

    all_results = []
    all_embeddings = []

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Impossible de lire l'image : %s", img_path)
            continue

        detections = detect_plates(model, image)

        for det in detections:
            text = recognize_text(reader, det["crop"])
            embedding = extract_embeddings(model, image, det["bbox"])

            result = {
                "image": img_path.name,
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "plate_text": text,
            }
            all_results.append(result)
            all_embeddings.append(embedding)

            logger.info("Plaque détectée : '%s' (confiance : %.2f)", text, det["confidence"])

    if save_embeddings and all_embeddings:
        embeddings_array = np.array(all_embeddings)
        emb_path = Path("models/embeddings.npy")
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, embeddings_array)
        logger.info("Embeddings sauvegardés : %s (%d vecteurs)", emb_path, len(all_embeddings))

    # Sauvegarde JSON des résultats
    import json
    results_path = output_path / "module_a2_yolo_ocr_results.json"
    serializable = [
        {**r, "bbox": list(r["bbox"])} for r in all_results
    ]
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    logger.info("Résultats OCR sauvegardés : %s", results_path)
    logger.info("=== Pipeline YOLO + OCR terminé — %d détections ===", len(all_results))

    return all_results
