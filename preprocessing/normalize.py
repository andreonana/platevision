"""
Normalisation et standardisation des images PlateVision.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
PLATE_SIZE = (200, 60)    # largeur × hauteur crops plaques
CHAR_SIZE  = (28, 28)     # caractères
YOLO_SIZE  = (640, 640)   # entrée YOLOv8


def normalize_plate_crop(img: np.ndarray) -> np.ndarray:
    """
    Normalisation d'un crop de plaque 200×60 pour le pipeline A1/B/C.

    Étapes :
      1. Convertir en BGR (3 canaux) si niveaux de gris
      2. Redimensionner à PLATE_SIZE (INTER_CUBIC si agrandissement, INTER_AREA sinon)
      3. CLAHE sur canal V (HSV) : clipLimit=2.0, tileGridSize=(4, 2)
      4. Normalisation [0, 255] → [0.0, 1.0] float32

    Retourne : np.ndarray float32 shape (60, 200, 3)
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    tw, th = PLATE_SIZE
    if (w, h) != (tw, th):
        interpolation = cv2.INTER_CUBIC if (w < tw or h < th) else cv2.INTER_AREA
        img = cv2.resize(img, (tw, th), interpolation=interpolation)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 2))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return (img.astype(np.float32) / 255.0)


def normalize_char_image(img: np.ndarray,
                          apply_threshold: bool = False) -> np.ndarray:
    """
    Normalisation d'un caractère 28×28 pour le classificateur Naïves Bayes.

    Étapes :
      1. Convertir en niveaux de gris si 3 canaux
      2. Redimensionner à CHAR_SIZE si nécessaire (INTER_AREA)
      3. CLAHE : clipLimit=3.0, tileGridSize=(4, 4) sur image grise
      4. Normalisation [0, 255] → [0.0, 1.0] float32
      5. Binarisation adaptative (si apply_threshold=True) :
           ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, blockSize=11, C=2

    Retourne : np.ndarray float32 shape (28, 28)
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape
    cw, ch = CHAR_SIZE
    if (w, h) != (cw, ch):
        img = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    img = clahe.apply(img)

    out = img.astype(np.float32) / 255.0

    if apply_threshold:
        img_u8 = (out * 255).astype(np.uint8)
        img_u8 = cv2.adaptiveThreshold(
            img_u8, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11, C=2,
        )
        out = img_u8.astype(np.float32) / 255.0

    return out


def normalize_yolo_image(img: np.ndarray) -> np.ndarray:
    """
    Préparation d'une image pour YOLOv8 (640×640 letterbox).

    Étapes :
      1. Letterbox : ratio conservé, padding gris (114, 114, 114)
      2. BGR → RGB
      3. HWC → CHW (transpose (2, 0, 1))
      4. Normalisation [0, 255] → [0.0, 1.0] float32

    Retourne : np.ndarray float32 shape (3, 640, 640)
    """
    tw, th = YOLO_SIZE
    H, W = img.shape[:2]
    scale   = min(tw / W, th / H)
    new_w   = int(W * scale)
    new_h   = int(H * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas  = np.full((th, tw, 3), 114, dtype=np.uint8)
    pad_x   = (tw - new_w) // 2
    pad_y   = (th - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1)
    return chw.astype(np.float32) / 255.0


def batch_normalize_crops(crops_dir: Path,
                           output_dir: Path,
                           mode: str = "plate") -> dict:
    """
    Normalise en batch tous les crops d'un dossier.

    mode : "plate" → normalize_plate_crop()
           "char"  → normalize_char_image()
           "yolo"  → normalize_yolo_image()

    Sauvegarde dans output_dir/ avec le même nom de fichier (float32 → PNG uint8).
    Retourne : {"n_processed": int, "n_errors": int, "output_dir": str}
    """
    crops_dir  = Path(crops_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _fn_map = {
        "plate": normalize_plate_crop,
        "char":  normalize_char_image,
        "yolo":  normalize_yolo_image,
    }
    if mode not in _fn_map:
        raise ValueError(f"mode doit être 'plate', 'char' ou 'yolo', reçu : '{mode}'")
    normalize_fn = _fn_map[mode]

    extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    paths = [p for p in sorted(crops_dir.rglob("*")) if p.suffix in extensions]

    n_processed = 0
    n_errors    = 0

    for img_path in tqdm(paths, desc=f"Normalize [{mode}]"):
        try:
            flag = cv2.IMREAD_GRAYSCALE if mode == "char" else cv2.IMREAD_COLOR
            img  = cv2.imread(str(img_path), flag)
            if img is None:
                n_errors += 1
                continue

            norm = normalize_fn(img)

            # Reconversion en uint8 pour la sauvegarde PNG
            if mode == "yolo":
                # CHW → HWC pour imwrite
                save_img = (norm.transpose(1, 2, 0) * 255).astype(np.uint8)
                save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            else:
                save_img = (norm * 255).astype(np.uint8)

            dst = output_dir / img_path.name
            cv2.imwrite(str(dst), save_img)
            n_processed += 1

        except Exception as exc:
            logger.error(f"normalize_batch erreur {img_path.name} : {exc}")
            n_errors += 1

    logger.info(
        f"batch_normalize_crops [{mode}] — {n_processed} OK, {n_errors} erreurs"
    )
    return {
        "n_processed": n_processed,
        "n_errors":    n_errors,
        "output_dir":  str(output_dir),
    }
