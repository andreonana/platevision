"""
Preprocessing — Extraction de features manuelles pour le Module A1 (Naïves Bayes).

Rôle dans le pipeline PlateVision :
    Extrait des descripteurs d'image de bas niveau (histogrammes HSV, gradients
    de Sobel, moments de Hu) depuis les régions de plaques. Ces features sont
    utilisées comme entrée du classifieur Naïves Bayes, offrant une baseline
    interprétable face au pipeline deep learning (Module A2).
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

N_HSV_BINS = 32
GRADIENT_SIZE = (32, 32)


def extract_hsv_histogram(image: np.ndarray) -> np.ndarray:
    """
    Extrait un histogramme HSV normalisé de l'image.

    Args:
        image: Image BGR (numpy array)

    Returns:
        Vecteur de features 1D de taille 3 * N_HSV_BINS
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = []
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [N_HSV_BINS], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-9)   # normalisation L1
        features.extend(hist)
    return np.array(features, dtype=np.float32)


def extract_sobel_gradients(image: np.ndarray) -> np.ndarray:
    """
    Extrait les gradients de Sobel (magnitude) redimensionnés et aplatis.

    Les gradients capturent les contours des caractères sur la plaque.

    Returns:
        Vecteur de features 1D de taille GRADIENT_SIZE[0] * GRADIENT_SIZE[1]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    resized = cv2.resize(magnitude, GRADIENT_SIZE)
    normalized = resized / (resized.max() + 1e-9)
    return normalized.flatten().astype(np.float32)


def extract_hu_moments(image: np.ndarray) -> np.ndarray:
    """
    Extrait les 7 moments de Hu invariants à la rotation et à l'échelle.

    Returns:
        Vecteur de 7 features (log-transformé pour stabilité numérique)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    # Log-transformation pour réduire la variance des valeurs
    log_hu = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    return log_hu.astype(np.float32)


def extract_all_features(image: np.ndarray) -> np.ndarray:
    """
    Concatène tous les descripteurs en un seul vecteur de features.

    Returns:
        Vecteur 1D de taille : 3*N_HSV_BINS + GRADIENT_SIZE^2 + 7
    """
    hsv_feats = extract_hsv_histogram(image)
    grad_feats = extract_sobel_gradients(image)
    hu_feats = extract_hu_moments(image)
    return np.concatenate([hsv_feats, grad_feats, hu_feats])


def extract_features_from_dataset(
    images_dir: Path,
    labels_csv: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Extrait les features de toutes les images d'un dossier.

    Args:
        images_dir: Dossier contenant les images preprocessées
        labels_csv: Fichier CSV avec colonnes 'filename' et 'label' (optionnel)

    Returns:
        Tuple (features_matrix, filenames)
    """
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_files = sorted([
        f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions
    ])

    features_list = []
    filenames = []

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Image illisible : %s", img_path)
            continue

        # Redimensionnement uniforme pour la cohérence des features
        image = cv2.resize(image, (128, 64))
        features = extract_all_features(image)
        features_list.append(features)
        filenames.append(img_path.name)

    if not features_list:
        raise ValueError(f"Aucune image valide trouvée dans : {images_dir}")

    X = np.array(features_list, dtype=np.float32)

    # Sauvegarde des features
    output_dir = images_dir.parent
    np.save(output_dir / "features.npy", X)
    logger.info("Features sauvegardées : %s (%d × %d)", output_dir / "features.npy", *X.shape)

    return X, filenames
