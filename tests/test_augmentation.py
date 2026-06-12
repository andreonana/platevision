"""
Tests unitaires — preprocessing/augmentation.py
Aucune dépendance sur les datasets réels : images générées via numpy.random.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from preprocessing.augmentation import (
    add_char_occlusion,
    add_gaussian_noise,
    adjust_brightness,
    adjust_contrast,
    affine_distort_char,
    augment_char,
    augment_dataset_chars,
    augment_plate_crop,
    erode_or_dilate,
    flip_plate,
    perspective_plate,
    rotate_char,
    rotate_plate,
    simulate_motion_blur,
    simulate_night_conditions,
    simulate_plate_degradation,
    translate_char,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _char(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(20, 220, (28, 28), dtype=np.uint8)


def _plate(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (60, 200, 3), dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINES
# ══════════════════════════════════════════════════════════════════════════════

def test_augment_char_count():
    """augment_char(n=5) retourne exactement 5 éléments."""
    results = augment_char(_char(), n=5, seed=0)
    assert len(results) == 5


def test_augment_char_shapes():
    """Chaque image augmentée est uint8 (28, 28)."""
    for aug in augment_char(_char(), n=4, seed=1):
        assert aug.shape == (28, 28), f"Forme inattendue : {aug.shape}"
        assert aug.dtype == np.uint8


def test_augment_char_seed_reproducible():
    """Même seed → sorties identiques sur deux appels consécutifs."""
    img = _char(seed=99)
    r1 = augment_char(img, n=3, seed=42)
    r2 = augment_char(img, n=3, seed=42)
    for a, b in zip(r1, r2):
        np.testing.assert_array_equal(a, b)


def test_augment_char_not_identical():
    """Sans seed fixe, au moins 1 résultat sur 5 doit différer de l'original."""
    img = _char(seed=7)
    results = augment_char(img, n=5)
    assert any(not np.array_equal(r, img) for r in results), (
        "Toutes les images augmentées sont identiques à l'original"
    )


def test_augment_plate_shapes():
    """augment_plate_crop(n=4) → 4 images uint8 (60, 200, 3)."""
    for aug in augment_plate_crop(_plate(), n=4, seed=0):
        assert aug.shape == (60, 200, 3)
        assert aug.dtype == np.uint8


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS PLAQUES
# ══════════════════════════════════════════════════════════════════════════════

def test_motion_blur_shape():
    """simulate_motion_blur retourne (60, 200, 3)."""
    out = simulate_motion_blur(_plate())
    assert out.shape == (60, 200, 3)


def test_motion_blur_differs_from_original():
    """Le flou de mouvement doit modifier une image non uniforme."""
    img = _plate(seed=3)
    out = simulate_motion_blur(img)
    assert not np.array_equal(img, out), "Le flou n'a pas modifié l'image"


def test_night_conditions_darker():
    """simulate_night_conditions doit réduire la luminosité moyenne (canal V)."""
    img = _plate(seed=5)
    hsv_orig   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out        = simulate_night_conditions(img)
    hsv_result = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    assert hsv_result[:, :, 2].mean() < hsv_orig[:, :, 2].mean(), (
        "La luminosité n'a pas diminué après simulate_night_conditions"
    )


def test_degradation_preserves_shape():
    """simulate_plate_degradation → (60, 200, 3) uint8."""
    out = simulate_plate_degradation(_plate())
    assert out.shape == (60, 200, 3)
    assert out.dtype == np.uint8


def test_perspective_preserves_dims():
    """perspective_plate → (60, 200, 3) sans erreur."""
    out = perspective_plate(_plate())
    assert out.shape == (60, 200, 3)


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS CARACTÈRES
# ══════════════════════════════════════════════════════════════════════════════

def test_char_occlusion_shape():
    """add_char_occlusion → (28, 28) uint8."""
    out = add_char_occlusion(_char())
    assert out.shape == (28, 28)
    assert out.dtype == np.uint8


# ══════════════════════════════════════════════════════════════════════════════
# ROBUSTESSE / CLIPS
# ══════════════════════════════════════════════════════════════════════════════

def test_brightness_clip():
    """adjust_brightness avec facteurs extrêmes → valeurs toujours dans [0, 255]."""
    white = np.full((60, 200, 3), 255, dtype=np.uint8)
    out_high = adjust_brightness(white, factor_range=(3.0, 3.0))
    assert out_high.min() >= 0 and out_high.max() <= 255

    dark = np.full((60, 200, 3), 10, dtype=np.uint8)
    out_low = adjust_brightness(dark, factor_range=(0.0, 0.0))
    assert out_low.min() >= 0 and out_low.max() <= 255


# ══════════════════════════════════════════════════════════════════════════════
# augment_dataset_chars
# ══════════════════════════════════════════════════════════════════════════════

def test_augment_dataset_chars_adds_files(tmp_path):
    """
    3 classes × 2 images → après augmentation target=10,
    chaque dossier contient ≥ 10 fichiers PNG.
    """
    rng = np.random.default_rng(0)
    for label in ["A", "B", "C"]:
        label_dir = tmp_path / label
        label_dir.mkdir()
        for i in range(2):
            img = rng.integers(0, 255, (28, 28), dtype=np.uint8)
            cv2.imwrite(str(label_dir / f"img_{i:06d}.png"), img)

    result = augment_dataset_chars(tmp_path, target_per_class=10, seed=42)

    assert result["n_classes"] == 3
    assert result["n_added"] > 0  # des images ont bien été créées

    for label in ["A", "B", "C"]:
        files = list((tmp_path / label).glob("*.png"))
        assert len(files) >= 10, (
            f"Classe {label} : {len(files)} fichiers, attendu ≥ 10"
        )


def test_augment_dataset_chars_missing_dir():
    """augment_dataset_chars sur un dossier inexistant → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        augment_dataset_chars(Path("/nonexistent/chars_dir"), target_per_class=10)
