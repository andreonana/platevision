"""
PlateVision — Prétraitement : Normalisation des images.

Ce module prépare les images brutes pour l'inférence et l'entraînement :
redimensionnement, normalisation des valeurs de pixels, conversion des
espaces colorimétriques. Toutes les transformations sont reproductibles.

Rôle dans le pipeline : standardisation des entrées avant YOLO et extraction features.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Dimensions cibles pour YOLO v8
TARGET_SIZE = (640, 640)
# Moyennes et écarts-types ImageNet (pour fine-tuning de CNN)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def resize_with_padding(
    image: np.ndarray,
    target_size: Tuple[int, int] = TARGET_SIZE,
    pad_color: Tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """
    Redimensionne une image en conservant le ratio avec padding (letterbox).

    Cette approche est utilisée par YOLO pour éviter la déformation des plaques.

    Args:
        image: Image BGR source
        target_size: Dimensions cibles (largeur, hauteur)
        pad_color: Couleur de remplissage pour le padding (gris par défaut)

    Returns:
        Image redimensionnée et paddée au format target_size
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calcul du ratio de redimensionnement en conservant le rapport d'aspect
    ratio = min(target_w / w, target_h / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Calcul du padding pour centrer l'image
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=pad_color,
    )
    return padded


def normalize_pixel_values(
    image: np.ndarray,
    method: str = "imagenet",
) -> np.ndarray:
    """
    Normalise les valeurs de pixels d'une image.

    Args:
        image: Image BGR uint8 (0-255)
        method: Méthode de normalisation :
                'imagenet' : normalisation ImageNet (pour CNN pré-entraînés)
                'minmax'   : normalisation [0, 1]
                'standard' : z-score (moyenne=0, std=1)

    Returns:
        Image normalisée en float32
    """
    image_float = image.astype(np.float32) / 255.0
    # Conversion BGR → RGB pour les normes ImageNet
    image_rgb = cv2.cvtColor(image_float, cv2.COLOR_BGR2RGB)

    if method == "imagenet":
        normalized = (image_rgb - IMAGENET_MEAN) / IMAGENET_STD
    elif method == "minmax":
        normalized = image_rgb  # déjà dans [0, 1]
    elif method == "standard":
        mean = image_rgb.mean()
        std = image_rgb.std() + 1e-8
        normalized = (image_rgb - mean) / std
    else:
        raise ValueError(f"Méthode de normalisation inconnue : {method}")

    return normalized.astype(np.float32)


def preprocess_single_image(
    image_path: Path,
    target_size: Tuple[int, int] = TARGET_SIZE,
    normalize: bool = True,
) -> Optional[np.ndarray]:
    """
    Prétraite une image complète : lecture, redimensionnement, normalisation.

    Args:
        image_path: Chemin vers le fichier image
        target_size: Dimensions cibles
        normalize: Appliquer la normalisation ImageNet

    Returns:
        Image prétraitée ou None en cas d'erreur
    """
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Impossible de lire : {image_path}")
        return None

    image_resized = resize_with_padding(image, target_size)

    if normalize:
        image_norm = normalize_pixel_values(image_resized, method="imagenet")
        return image_norm

    return image_resized


def normalize_images(
    input_path: Path,
    output_path: Path,
    target_size: Tuple[int, int] = TARGET_SIZE,
) -> int:
    """
    Prétraite toutes les images d'un dossier ou d'un fichier vidéo.

    Args:
        input_path: Dossier d'images ou chemin vers une vidéo
        output_path: Dossier de sortie pour les images prétraitées
        target_size: Dimensions cibles pour le redimensionnement

    Returns:
        Nombre d'images traitées
    """
    from tqdm import tqdm

    output_path.mkdir(parents=True, exist_ok=True)
    processed_count = 0

    # Traitement d'un fichier vidéo
    if input_path.is_file() and input_path.suffix.lower() in {".mp4", ".avi", ".mov"}:
        logger.info(f"Extraction des frames depuis la vidéo : {input_path}")
        cap = cv2.VideoCapture(str(input_path))
        frame_idx = 0
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # Échantillonnage : 1 frame par seconde
        sample_rate = max(1, fps)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_rate == 0:
                resized = resize_with_padding(frame, target_size)
                out_file = output_path / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_file), resized)
                processed_count += 1
            frame_idx += 1
        cap.release()

    # Traitement d'un dossier d'images
    elif input_path.is_dir():
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
        logger.info(f"Prétraitement de {len(image_files)} images...")

        for img_path in tqdm(image_files, desc="Normalisation"):
            resized = resize_with_padding(cv2.imread(str(img_path)), target_size)
            if resized is not None:
                out_file = output_path / img_path.name
                cv2.imwrite(str(out_file), resized)
                processed_count += 1

    logger.info(f"Prétraitement terminé : {processed_count} images dans {output_path}")
    return processed_count
