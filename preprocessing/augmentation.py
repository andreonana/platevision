"""
PlateVision — Prétraitement : Augmentation de données.

Ce module implémente des techniques d'augmentation de données pour
enrichir le dataset d'entraînement. L'augmentation simule les conditions
réelles de prise de vue sur les routes camerounaises : variations de
luminosité, flou de mouvement, rotations, pluie, etc.

Rôle dans le pipeline : amélioration de la robustesse du modèle YOLO.
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def apply_augmentations(
    image: np.ndarray,
    use_albumentations: bool = True,
) -> List[np.ndarray]:
    """
    Applique un ensemble d'augmentations à une image.

    Args:
        image: Image BGR (numpy array, uint8)
        use_albumentations: Utiliser albumentations si disponible

    Returns:
        Liste d'images augmentées (incluant l'image originale)
    """
    augmented = [image.copy()]  # l'image originale est toujours incluse

    if use_albumentations:
        try:
            import albumentations as A
            transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.GaussianBlur(blur_limit=(3, 7), p=0.4),
                A.MotionBlur(blur_limit=5, p=0.3),
                A.Rotate(limit=15, p=0.5),
                A.HorizontalFlip(p=0.3),
                A.RandomRain(rain_type="heavy", p=0.2),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
            ])
            result = transform(image=image)
            augmented.append(result["image"])
            return augmented
        except ImportError:
            logger.warning("albumentations non disponible, utilisation des augmentations manuelles")

    # Augmentations manuelles (fallback sans albumentations)
    # Variation de luminosité
    bright = cv2.convertScaleAbs(image, alpha=1.3, beta=30)
    dark = cv2.convertScaleAbs(image, alpha=0.7, beta=-20)
    augmented.extend([bright, dark])

    # Flou gaussien
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    augmented.append(blurred)

    # Rotation légère
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle=10, scale=1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    augmented.append(rotated)

    return augmented


def augment_dataset(
    input_path: Path,
    output_path: Path,
    multiplier: int = 3,
) -> int:
    """
    Augmente toutes les images d'un dossier et sauvegarde les résultats.

    Args:
        input_path: Dossier d'images originales
        output_path: Dossier de destination pour les images augmentées
        multiplier: Facteur multiplicateur du dataset (défaut : 3x)

    Returns:
        Nombre total d'images générées
    """
    from tqdm import tqdm

    output_path.mkdir(parents=True, exist_ok=True)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        logger.warning(f"Aucune image trouvée dans {input_path}")
        return 0

    logger.info(f"Augmentation de {len(image_files)} images (×{multiplier})...")
    total_generated = 0

    for img_path in tqdm(image_files, desc="Augmentation"):
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"Impossible de lire : {img_path}")
            continue

        # Sauvegarde de l'image originale
        cv2.imwrite(str(output_path / img_path.name), image)
        total_generated += 1

        # Génération des versions augmentées
        for i in range(multiplier - 1):
            augmented_list = apply_augmentations(image)
            for j, aug_img in enumerate(augmented_list[1:], 1):
                aug_name = f"{img_path.stem}_aug_{i}_{j}{img_path.suffix}"
                cv2.imwrite(str(output_path / aug_name), aug_img)
                total_generated += 1

    logger.info(f"Augmentation terminée : {total_generated} images générées dans {output_path}")
    return total_generated
