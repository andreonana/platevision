"""
Preprocessing — Augmentation de données pour le dataset PlateVision.

Rôle dans le pipeline PlateVision :
    Augmente artificiellement la diversité du dataset d'entraînement en appliquant
    des transformations géométriques et photométriques sur les images de plaques.
    Crucial pour la robustesse du modèle face aux conditions réelles camerounaises
    (pluie, poussière, luminosité variable, angles de prise de vue).
"""

import logging
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Pipeline d'augmentation calibré pour les plaques d'immatriculation camerounaises
AUGMENTATION_PIPELINE = A.Compose(
    [
        # Variations géométriques légères (plaques fixées sur véhicules)
        A.HorizontalFlip(p=0.0),   # Pas de flip horizontal (texte inversé)
        A.Rotate(limit=10, p=0.5),
        A.Perspective(scale=(0.02, 0.08), p=0.4),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=8, p=0.4),

        # Variations photométriques (conditions lumineuses camerounaises)
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),

        # Dégradations simulant les conditions de terrain
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.MotionBlur(blur_limit=(3, 9), p=0.2),   # Véhicules en mouvement
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),

        # Simulation de conditions météo (pluie, brume)
        A.RandomRain(p=0.1),
        A.RandomFog(p=0.1),
        A.RandomShadow(p=0.2),

        # Compression JPEG (caméras bas de gamme)
        A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)


def augment_image(
    image: np.ndarray,
    bboxes: list[list[float]] | None = None,
    class_labels: list[int] | None = None,
    n_augmentations: int = 5,
) -> list[dict]:
    """
    Génère n_augmentations variantes augmentées d'une image.

    Args:
        image: Image source (BGR numpy array)
        bboxes: Bounding boxes YOLO format [[x_c, y_c, w, h], ...]
        class_labels: Labels de classe associés aux bboxes
        n_augmentations: Nombre de variantes à générer

    Returns:
        Liste de dicts {'image', 'bboxes', 'class_labels'}
    """
    if bboxes is None:
        bboxes = []
    if class_labels is None:
        class_labels = []

    augmented_samples = []

    for _ in range(n_augmentations):
        try:
            result = AUGMENTATION_PIPELINE(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels,
            )
            augmented_samples.append({
                "image": result["image"],
                "bboxes": result["bboxes"],
                "class_labels": result["class_labels"],
            })
        except Exception as e:
            logger.warning("Échec augmentation : %s", e)

    return augmented_samples


def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    n_augmentations: int = 5,
) -> int:
    """
    Augmente toutes les images d'un dossier et sauvegarde dans output_dir.

    Args:
        input_dir: Dossier source contenant images + annotations YOLO (.txt)
        output_dir: Dossier de sortie
        n_augmentations: Nombre de variantes par image

    Returns:
        Nombre total d'images générées.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]

    total_generated = 0

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Impossible de lire : %s", img_path)
            continue

        # Chargement des annotations YOLO correspondantes
        annotation_path = img_path.with_suffix(".txt")
        bboxes, class_labels = [], []
        if annotation_path.exists():
            with open(annotation_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_labels.append(int(parts[0]))
                        bboxes.append([float(x) for x in parts[1:]])

        augmented = augment_image(image, bboxes, class_labels, n_augmentations)

        for i, sample in enumerate(augmented):
            out_img_path = output_dir / f"{img_path.stem}_aug{i:03d}{img_path.suffix}"
            cv2.imwrite(str(out_img_path), sample["image"])

            out_ann_path = output_dir / f"{img_path.stem}_aug{i:03d}.txt"
            with open(out_ann_path, "w") as f:
                for cls, bbox in zip(sample["class_labels"], sample["bboxes"]):
                    f.write(f"{cls} {' '.join(f'{v:.6f}' for v in bbox)}\n")

            total_generated += 1

    logger.info("Augmentation terminée : %d images générées dans %s", total_generated, output_dir)
    return total_generated
