"""
Preprocessing — Normalisation des images pour le pipeline PlateVision.

Rôle dans le pipeline PlateVision :
    Standardise les images brutes (redimensionnement, normalisation des pixels,
    conversion d'espace colorimétrique) avant leur passage dans les modèles.
    Garantit la cohérence des entrées entre l'entraînement et l'inférence.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Taille cible pour le modèle YOLOv8
YOLO_INPUT_SIZE = (640, 640)
# Taille pour le classifieur Naïves Bayes
NB_INPUT_SIZE = (128, 64)


def resize_image(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Redimensionne une image en conservant les proportions avec padding."""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)

    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Padding centré pour atteindre la taille cible
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded


def normalize_pixel_values(image: np.ndarray) -> np.ndarray:
    """Normalise les valeurs de pixels de [0, 255] vers [0.0, 1.0]."""
    return image.astype(np.float32) / 255.0


def normalize_images(input_path: Path, target_size: tuple = YOLO_INPUT_SIZE) -> Path:
    """
    Normalise toutes les images d'un dossier ou d'un fichier unique.

    Args:
        input_path: Dossier ou fichier image source
        target_size: Dimensions cibles (largeur, hauteur)

    Returns:
        Chemin vers le dossier des images normalisées.
    """
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    if input_path.is_dir():
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
    else:
        image_files = [input_path]

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Impossible de lire : %s", img_path)
            continue

        normalized = resize_image(image, target_size)
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), normalized)

    logger.info("%d images normalisées → %s", len(image_files), output_dir)
    return output_dir
