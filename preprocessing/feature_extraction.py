"""
Extraction des features manuelles 120D pour le Module A1 (Naïves Bayes).
Miroir fidèle de _extract_char_features / _extract_plate_features de
prepare_datasets.py — exposé ici en fonctions unitaires testables.
"""

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── Dimensions ────────────────────────────────────────────────────────────────
CHAR_FEAT_DIM  = 20
PLATE_FEAT_DIM = 100
TOTAL_FEAT_DIM = 120   # CHAR_FEAT_DIM + PLATE_FEAT_DIM


def extract_char_features(img: np.ndarray) -> np.ndarray:
    """
    Extrait le vecteur 20D d'un caractère 28×28 (niveaux de gris uint8).
    Miroir exact de _extract_char_features() dans prepare_datasets.py.

    Groupe A — forme globale (4D) :
      [0]  ratio h/w
      [1]  densité : nb pixels > 127 / (28×28)
      [2]  centroïde cx normalisé (cx_pixels / W)
      [3]  centroïde cy normalisé (cy_pixels / H)

    Groupe B — densités zonales (8D) :
      Grille 4 colonnes × 2 lignes → 8 zones
      [4:12] densité moyenne de chaque zone (pixels > 128 / taille zone)

    Groupe C — transitions 0→1 (4D) :
      [12] mean transitions horizontales (diff==1 par ligne)
      [13] std  transitions horizontales
      [14] mean transitions verticales   (diff==1 par colonne)
      [15] std  transitions verticales

    Groupe D — gradient Sobel (4D) :
      Sobel X et Y sur image uint8, ksize=3
      [16] mean Sobel X · [17] std Sobel X
      [18] mean Sobel Y · [19] std Sobel Y

    Robustesse : img entièrement noire → np.zeros(20)
    Retourne : np.ndarray float32 shape (20,)
    """
    if img.max() == 0:
        return np.zeros(CHAR_FEAT_DIM, dtype=np.float32)

    H, W = img.shape

    # ── Groupe A ──────────────────────────────────────────────────────────────
    ratio_hw = H / W
    density  = float(np.count_nonzero(img > 128)) / (H * W)
    binary   = (img > 128).astype(np.float32)
    total    = binary.sum() + 1e-6
    ys, xs   = np.mgrid[0:H, 0:W]
    cx = float((xs * binary).sum() / total) / W
    cy = float((ys * binary).sum() / total) / H
    feat_A = np.array([ratio_hw, density, cx, cy], dtype=np.float32)

    # ── Groupe B ──────────────────────────────────────────────────────────────
    feat_B  = np.zeros(8, dtype=np.float32)
    zone_h  = H // 2
    zone_w  = W // 4
    for row in range(2):
        for col in range(4):
            y0   = row * zone_h
            x0   = col * zone_w
            zone = binary[y0:y0 + zone_h, x0:x0 + zone_w]
            feat_B[row * 4 + col] = float(zone.mean()) if zone.size > 0 else 0.0

    # ── Groupe C ──────────────────────────────────────────────────────────────
    bin_u8    = binary.astype(np.uint8)
    row_trans = [int(np.count_nonzero(np.diff(bin_u8[r]) == 1)) for r in range(H)]
    col_trans = [int(np.count_nonzero(np.diff(bin_u8[:, c]) == 1)) for c in range(W)]
    feat_C = np.array([
        float(np.mean(row_trans)), float(np.std(row_trans)),
        float(np.mean(col_trans)), float(np.std(col_trans)),
    ], dtype=np.float32)

    # ── Groupe D ──────────────────────────────────────────────────────────────
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    feat_D  = np.array([
        float(sobel_x.mean()), float(sobel_x.std()),
        float(sobel_y.mean()), float(sobel_y.std()),
    ], dtype=np.float32)

    return np.concatenate([feat_A, feat_B, feat_C, feat_D])


def extract_plate_features(img: np.ndarray) -> np.ndarray:
    """
    Extrait le vecteur 100D d'un crop de plaque 200×60 (BGR uint8).

    Histogramme HSV (96D) :
      H : 32 bins [0, 180] · S : 32 bins [0, 256] · V : 32 bins [0, 256]
      Normalisation indépendante : h / (h.sum() + 1e-7)
      [0:96] = concat(H_hist, S_hist, V_hist)

    Gradient magnitude (2D) :
      Gris float32/255 → Sobel X + Y → magnitude = √(Gx²+Gy²)
      [96] mean · [97] std

    Ratio pixels sombres (1D) :
      Pixels avec V < 50 (canal HSV) / total pixels
      [98] dark_ratio

    Cadre métallique (1D) :
      Bordure 10% de chaque côté (top/bottom/left/right)
      Pixels avec S < 30 ET V > 150 / total pixels bordure
      [99] = 1.0 si ratio > 0.60, sinon 0.0

    Retourne : np.ndarray float32 shape (100,)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ── Histogramme HSV ───────────────────────────────────────────────────────
    def _norm_hist(h):
        s = h.sum() + 1e-7
        return (h / s).astype(np.float32)

    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    feat_hsv = np.concatenate([_norm_hist(hist_h), _norm_hist(hist_s), _norm_hist(hist_v)])

    # ── Gradient magnitude ────────────────────────────────────────────────────
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    sobel_x   = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y   = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    feat_grad = np.array([float(magnitude.mean()), float(magnitude.std())],
                         dtype=np.float32)

    # ── Ratio pixels sombres (V < 50) ─────────────────────────────────────────
    v_channel = hsv[:, :, 2].astype(np.float32)
    dark_ratio = float((v_channel < 50).sum()) / v_channel.size
    feat_dark  = np.array([dark_ratio], dtype=np.float32)

    # ── Présence cadre métallique ─────────────────────────────────────────────
    H, W   = img.shape[:2]
    bh     = max(1, H // 10)   # 10% hauteur
    bw     = max(1, W // 10)   # 10% largeur
    border = np.concatenate([
        hsv[:bh, :, :].reshape(-1, 3),
        hsv[-bh:, :, :].reshape(-1, 3),
        hsv[:, :bw, :].reshape(-1, 3),
        hsv[:, -bw:, :].reshape(-1, 3),
    ])
    metal_mask  = (border[:, 1] < 30) & (border[:, 2] > 150)
    metal_ratio = float(metal_mask.sum()) / len(border) if len(border) > 0 else 0.0
    feat_metal  = np.array([1.0 if metal_ratio > 0.60 else 0.0], dtype=np.float32)

    return np.concatenate([feat_hsv, feat_grad, feat_dark, feat_metal])


def extract_feature_vector(char_img: np.ndarray,
                            plate_img: np.ndarray) -> np.ndarray:
    """
    Combine char + plate en vecteur 120D.

    char_img  : np.ndarray (28, 28) uint8 niveaux de gris
    plate_img : np.ndarray (60, 200, 3) uint8 BGR

    Retourne : np.ndarray float32 shape (120,)
    """
    return np.concatenate([
        extract_char_features(char_img),
        extract_plate_features(plate_img),
    ]).astype(np.float32)


def extract_features_batch(char_paths: list,
                             plate_paths: list) -> np.ndarray:
    """
    Extraction batch pour N paires (char_path, plate_path).

    char_paths  : liste de N chemins vers images 28×28 (grayscale)
    plate_paths : liste de N chemins vers crops 200×60 (BGR)

    Retourne : np.ndarray float32 shape (N, 120)
    """
    N      = len(char_paths)
    X      = np.zeros((N, TOTAL_FEAT_DIM), dtype=np.float32)
    errors = 0

    for i, (cp, pp) in enumerate(tqdm(
        zip(char_paths, plate_paths), total=N, desc="Features batch"
    )):
        try:
            char_img  = cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE)
            plate_img = cv2.imread(str(pp), cv2.IMREAD_COLOR)
            if char_img is None or plate_img is None:
                errors += 1
                continue
            X[i] = extract_feature_vector(char_img, plate_img)
        except Exception:
            errors += 1

    if errors:
        import logging
        logging.getLogger(__name__).warning(
            f"extract_features_batch : {errors}/{N} paires en erreur"
        )
    return X
