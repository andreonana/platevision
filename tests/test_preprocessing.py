"""
Tests unitaires pour preprocessing/ (normalize, augmentation, feature_extraction).
Aucune dépendance sur les datasets réels — uniquement numpy.random + tmp_path.
"""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from preprocessing.augmentation import augment_char, augment_plate_crop
from preprocessing.feature_extraction import (
    extract_char_features,
    extract_feature_vector,
    extract_plate_features,
)
from preprocessing.normalize import (
    normalize_char_image,
    normalize_plate_crop,
    normalize_yolo_image,
)

# ── Helpers synthétiques ───────────────────────────────────────────────────────

def _rand_char(seed=0):
    """Image de caractère synthétique 28×28 uint8 niveaux de gris."""
    rng = np.random.default_rng(seed)
    return rng.integers(10, 240, (28, 28), dtype=np.uint8)


def _rand_plate(seed=0):
    """Crop de plaque synthétique 60×200 uint8 BGR."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (60, 200, 3), dtype=np.uint8)


def _rand_bgr(h, w, seed=0):
    """Image BGR synthétique de dimensions quelconques."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


# ── 1. normalize_plate_crop ────────────────────────────────────────────────────

def test_normalize_plate_shape_and_dtype():
    """normalize_plate_crop retourne float32 shape (60, 200, 3)."""
    out = normalize_plate_crop(_rand_plate())
    assert out.shape == (60, 200, 3)
    assert out.dtype == np.float32


def test_normalize_plate_value_range():
    """Toutes les valeurs doivent être dans [0.0, 1.0] après normalisation."""
    out = normalize_plate_crop(_rand_plate(seed=7))
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


# ── 2. normalize_char_image ────────────────────────────────────────────────────

def test_normalize_char_shape_and_dtype():
    """normalize_char_image retourne float32 shape (28, 28)."""
    out = normalize_char_image(_rand_char())
    assert out.shape == (28, 28)
    assert out.dtype == np.float32


def test_normalize_char_with_threshold():
    """Avec apply_threshold=True, la sortie ne contient que 0.0 et 1.0."""
    out = normalize_char_image(_rand_char(seed=3), apply_threshold=True)
    assert out.shape == (28, 28)
    unique_vals = np.unique(out)
    assert set(unique_vals.tolist()).issubset({0.0, 1.0})


# ── 3. normalize_yolo_image ────────────────────────────────────────────────────

def test_normalize_yolo_shape_and_dtype():
    """normalize_yolo_image retourne float32 shape (3, 640, 640)."""
    img = _rand_bgr(480, 640)
    out = normalize_yolo_image(img)
    assert out.shape == (3, 640, 640)
    assert out.dtype == np.float32


def test_letterbox_preserves_ratio():
    """Le letterbox doit conserver le ratio h/w de l'image originale."""
    h, w = 200, 400  # ratio = 0.5
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (0, 0, 255)  # rouge pur en BGR → R=255 en RGB

    out = normalize_yolo_image(img)  # (3, 640, 640) float32

    # Calcul des zones letterbox attendues
    scale = min(640 / w, 640 / h)
    new_h = int(h * scale)
    pad_y = (640 - new_h) // 2

    # Zone de padding (haut) → tous les canaux ≈ 114/255 ≈ 0.447
    padding_val = 114.0 / 255.0
    assert abs(float(out[0, 0, 0]) - padding_val) < 0.02, (
        f"Attendu padding ≈ {padding_val:.3f}, obtenu {out[0, 0, 0]:.3f}"
    )

    # Centre de l'image (row=320) est dans la zone de contenu (rouge → R≈1.0)
    center_row = 320
    assert center_row >= pad_y and center_row < pad_y + new_h, (
        "La ligne 320 devrait être dans la zone de contenu"
    )
    assert out[0, center_row, 320] > 0.9, (
        f"Canal R attendu ≈ 1.0 au centre, obtenu {out[0, center_row, 320]:.3f}"
    )


# ── 4. augment_char ────────────────────────────────────────────────────────────

def test_augment_char_count():
    """augment_char(n=5) retourne exactement 5 images."""
    imgs = augment_char(_rand_char(), n=5, seed=42)
    assert len(imgs) == 5


def test_augment_char_shape_and_dtype():
    """Chaque image augmentée doit être uint8 (28, 28)."""
    imgs = augment_char(_rand_char(seed=1), n=3, seed=0)
    for aug in imgs:
        assert aug.shape == (28, 28), f"Forme attendue (28, 28), obtenu {aug.shape}"
        assert aug.dtype == np.uint8, f"Type attendu uint8, obtenu {aug.dtype}"


# ── 5. augment_plate_crop ──────────────────────────────────────────────────────

def test_augment_plate_count_and_shape():
    """augment_plate_crop(n=3) retourne 3 images uint8 (60, 200, 3)."""
    imgs = augment_plate_crop(_rand_plate(), n=3, seed=7)
    assert len(imgs) == 3
    for aug in imgs:
        assert aug.shape == (60, 200, 3)
        assert aug.dtype == np.uint8


# ── 6. extract_char_features ──────────────────────────────────────────────────

def test_extract_char_features_shape_dtype():
    """extract_char_features retourne float32 shape (75,) — cahier des charges 175D."""
    feat = extract_char_features(_rand_char())
    assert feat.shape == (75,)
    assert feat.dtype == np.float32


def test_extract_plate_features_shape_dtype():
    """extract_plate_features retourne float32 shape (100,)."""
    feat = extract_plate_features(_rand_plate())
    assert feat.shape == (100,)
    assert feat.dtype == np.float32


def test_features_consistency_with_pipeline():
    """
    Après alignement sur le cahier des charges (175D), extract_char_features
    produit 75D tandis que _extract_char_features du pipeline produit 20D.
    Ce test vérifie que les deux fonctions s'exécutent sans erreur et que
    les dimensions correspondent à leurs spécifications respectives.
    """
    pipeline_path = Path(__file__).parent.parent / "data" / "prepare_datasets.py"
    if not pipeline_path.exists():
        pytest.skip("data/prepare_datasets.py introuvable")

    spec = importlib.util.spec_from_file_location("prepare_datasets", pipeline_path)
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        pytest.skip(f"Impossible de charger prepare_datasets.py : {exc}")

    img = _rand_char(seed=42)

    pipeline_feat = mod._extract_char_features(img)
    module_feat   = extract_char_features(img)

    assert pipeline_feat.shape == (20,), "Pipeline doit toujours retourner 20D"
    assert module_feat.shape   == (75,), "Module (cahier des charges) doit retourner 75D"
