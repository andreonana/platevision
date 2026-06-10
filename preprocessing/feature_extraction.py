"""
PlateVision — Prétraitement : Extraction de features manuelles.

Ce module extrait des descripteurs visuels manuels des images de plaques
pour alimenter le classifieur Naïves Bayes (Module A1) :
- Histogrammes HSV (teinte, saturation, valeur)
- Gradients Sobel (détection de contours)
- Descripteurs HOG (Histogram of Oriented Gradients)

Ces features capturent les propriétés visuelles distinctives des plaques
camerounaises sans nécessiter de deep learning.

Rôle dans le pipeline : extraction features → classifieur Naïves Bayes (A1).
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_hsv_histogram(image: np.ndarray, bins: int = 32) -> np.ndarray:
    """
    Calcule l'histogramme HSV normalisé d'une image.

    L'espace HSV est plus robuste aux variations de luminosité que BGR
    pour distinguer les couleurs de fond des plaques (blanc/jaune).

    Args:
        image: Image BGR (uint8)
        bins: Nombre de bins par canal de l'histogramme

    Returns:
        Vecteur de features de taille 3 * bins (normalisé)
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = []

    for channel in range(3):
        hist = cv2.calcHist([image_hsv], [channel], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist.tolist())

    return np.array(features, dtype=np.float32)


def extract_sobel_features(image: np.ndarray) -> np.ndarray:
    """
    Extrait des features de gradient via le filtre de Sobel.

    Les gradients horizontaux et verticaux capturent les contours des
    caractères sur la plaque.

    Args:
        image: Image BGR (uint8)

    Returns:
        Vecteur de 4 features statistiques (moyenne, écart-type par direction)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gradients Sobel dans les directions X et Y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    features = np.array([
        magnitude.mean(),
        magnitude.std(),
        np.abs(sobel_x).mean(),
        np.abs(sobel_y).mean(),
    ], dtype=np.float32)

    return features


def extract_hog_features(
    image: np.ndarray,
    target_size: tuple = (64, 128),
) -> np.ndarray:
    """
    Extrait les descripteurs HOG (Histogram of Oriented Gradients).

    HOG est particulièrement efficace pour la détection de formes et
    de patterns récurrents dans les plaques.

    Args:
        image: Image BGR (uint8)
        target_size: Taille de redimensionnement avant HOG (largeur, hauteur)

    Returns:
        Vecteur de features HOG
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)

    hog = cv2.HOGDescriptor(
        _winSize=(target_size[0], target_size[1]),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    features = hog.compute(resized).flatten()
    return features.astype(np.float32)


def extract_all_features(image: np.ndarray) -> np.ndarray:
    """
    Concatène tous les descripteurs en un seul vecteur de features.

    Args:
        image: Image BGR de la plaque

    Returns:
        Vecteur de features complet (HSV + Sobel + HOG)
    """
    hsv_features = extract_hsv_histogram(image, bins=32)     # 96 features
    sobel_features = extract_sobel_features(image)            # 4 features
    hog_features = extract_hog_features(image)                # ~3780 features

    return np.concatenate([hsv_features, sobel_features, hog_features])


def extract_features_from_dataset(
    input_path: Path,
    output_path: Path,
    labels_file: Optional[Path] = None,
) -> tuple:
    """
    Extrait les features de toutes les images d'un dossier et les sauvegarde.

    Les images doivent être organisées en sous-dossiers par classe :
        input_path/
          classe_0/image1.jpg, image2.jpg, ...
          classe_1/image1.jpg, ...

    Args:
        input_path: Dossier racine du dataset organisé par classe
        output_path: Dossier de sauvegarde des fichiers features.npy et labels.npy
        labels_file: Fichier CSV de labels alternatif (optionnel)

    Returns:
        Tuple (X, y) : matrice de features et vecteur de labels
    """
    output_path.mkdir(parents=True, exist_ok=True)
    all_features = []
    all_labels = []
    class_names = sorted([d.name for d in input_path.iterdir() if d.is_dir()])

    if not class_names:
        logger.error(f"Aucun sous-dossier de classes trouvé dans {input_path}")
        return None, None

    logger.info(f"Classes détectées : {class_names}")
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for class_idx, class_name in enumerate(class_names):
        class_path = input_path / class_name
        image_files = [f for f in class_path.iterdir() if f.suffix.lower() in image_extensions]
        logger.info(f"  Classe '{class_name}' ({class_idx}) : {len(image_files)} images")

        for img_path in tqdm(image_files, desc=f"Features {class_name}"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            # Redimensionnement uniforme avant extraction
            image_resized = cv2.resize(image, (128, 64))
            features = extract_all_features(image_resized)
            all_features.append(features)
            all_labels.append(class_idx)

    if not all_features:
        logger.error("Aucune feature extraite.")
        return None, None

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    # Sauvegarde des features et des labels
    np.save(str(output_path / "features.npy"), X)
    np.save(str(output_path / "labels.npy"), y)

    # Sauvegarde du mapping classes → indices
    with open(output_path / "class_names.txt", "w") as f:
        for idx, name in enumerate(class_names):
            f.write(f"{idx},{name}\n")

    logger.info(f"Features extraites : {X.shape[0]} échantillons, {X.shape[1]} features")
    logger.info(f"Fichiers sauvegardés dans {output_path}")
    return X, y
