"""
PlateVision — Tests unitaires : Classifieur Naïves Bayes (Module A1).
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestComputeCER:
    """Tests pour la fonction compute_cer (distance de Levenshtein)."""

    def test_cer_perfect_match(self):
        from modules.module_a.evaluate import compute_cer
        assert compute_cer("AB123CD", "AB123CD") == 0.0

    def test_cer_one_substitution(self):
        from modules.module_a.evaluate import compute_cer
        # 1 substitution sur 7 caractères = 1/7 ≈ 0.143
        result = compute_cer("AB123CD", "AB123CE")
        assert pytest.approx(result, abs=0.01) == 1 / 7

    def test_cer_empty_reference(self):
        from modules.module_a.evaluate import compute_cer
        assert compute_cer("", "ABC") == 1.0

    def test_cer_both_empty(self):
        from modules.module_a.evaluate import compute_cer
        assert compute_cer("", "") == 0.0

    def test_cer_case_insensitive(self):
        from modules.module_a.evaluate import compute_cer
        assert compute_cer("ab123cd", "AB123CD") == 0.0

    def test_cer_with_spaces_ignored(self):
        from modules.module_a.evaluate import compute_cer
        assert compute_cer("AB 123 CD", "AB123CD") == 0.0


class TestComputeWER:
    """Tests pour la fonction compute_wer."""

    def test_wer_exact_match(self):
        from modules.module_a.evaluate import compute_wer
        assert compute_wer("AB123CD", "AB123CD") == 0.0

    def test_wer_different_plates(self):
        from modules.module_a.evaluate import compute_wer
        assert compute_wer("AB123CD", "XY456ZZ") == 1.0

    def test_wer_case_insensitive(self):
        from modules.module_a.evaluate import compute_wer
        assert compute_wer("ab123cd", "AB123CD") == 0.0


class TestComputeIoU:
    """Tests pour la fonction compute_iou (bounding box overlap)."""

    def test_iou_perfect_overlap(self):
        from modules.module_a.evaluate import compute_iou
        box = (0, 0, 100, 100)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        from modules.module_a.evaluate import compute_iou
        box1 = (0, 0, 50, 50)
        box2 = (100, 100, 150, 150)
        assert compute_iou(box1, box2) == 0.0

    def test_iou_partial_overlap(self):
        from modules.module_a.evaluate import compute_iou
        box1 = (0, 0, 100, 100)
        box2 = (50, 0, 150, 100)
        # Intersection : 50x100 = 5000, Union : 15000
        expected = 5000 / 15000
        assert compute_iou(box1, box2) == pytest.approx(expected, abs=0.001)

    def test_iou_zero_area(self):
        from modules.module_a.evaluate import compute_iou
        box1 = (0, 0, 0, 0)
        box2 = (0, 0, 100, 100)
        assert compute_iou(box1, box2) == 0.0


class TestNaiveBayesTraining:
    """Tests d'intégration pour l'entraînement Naïves Bayes."""

    def test_train_naive_bayes_returns_model(self, tmp_path):
        from modules.module_a.naive_bayes import train_naive_bayes
        X_train = np.random.randn(100, 20).astype(np.float32)
        y_train = np.random.randint(0, 2, 100)
        classifier, scaler = train_naive_bayes(X_train, y_train, tmp_path)
        assert classifier is not None
        assert scaler is not None

    def test_trained_model_predicts_correct_shape(self, tmp_path):
        from modules.module_a.naive_bayes import train_naive_bayes, evaluate_naive_bayes
        X_train = np.random.randn(100, 20).astype(np.float32)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 20).astype(np.float32)
        y_test = np.random.randint(0, 2, 20)

        classifier, scaler = train_naive_bayes(X_train, y_train, tmp_path)
        results = evaluate_naive_bayes(classifier, scaler, X_test, y_test)

        assert "accuracy" in results
        assert 0.0 <= results["accuracy"] <= 1.0
        assert len(results["predictions"]) == len(y_test)
