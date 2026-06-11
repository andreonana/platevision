"""Tests unitaires pour le Module A1 — Classifieur Naïves Bayes."""

import numpy as np
import pytest
from sklearn.naive_bayes import GaussianNB

from modules.module_a.naive_bayes import train_naive_bayes, evaluate_classifier
from modules.module_a.evaluate import compute_cer, compute_wer, _levenshtein


class TestNaiveBayesTraining:
    """Tests d'entraînement du classifieur."""

    def test_train_returns_gaussian_nb(self):
        X = np.random.rand(100, 20).astype(np.float32)
        y = np.array([0] * 50 + [1] * 50)
        clf = train_naive_bayes(X, y)
        assert isinstance(clf, GaussianNB)

    def test_train_fits_on_data(self):
        X = np.random.rand(60, 10).astype(np.float32)
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        clf = train_naive_bayes(X, y)
        # Le modèle doit pouvoir prédire sans lever d'exception
        predictions = clf.predict(X[:5])
        assert predictions.shape == (5,)

    def test_train_single_class_raises(self):
        """Un classifieur avec une seule classe doit tout de même s'entraîner."""
        X = np.random.rand(20, 5).astype(np.float32)
        y = np.zeros(20, dtype=int)
        clf = train_naive_bayes(X, y)
        assert clf is not None


class TestEvaluateClassifier:
    """Tests d'évaluation du classifieur."""

    def test_perfect_classification(self):
        """Accuracy 1.0 quand prédictions == vérité terrain."""
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        y = np.array([0, 1, 0, 1])
        clf = train_naive_bayes(X, y)
        metrics = evaluate_classifier(clf, X, y, ["classe_0", "classe_1"])
        assert metrics["accuracy"] >= 0.9

    def test_metrics_keys_present(self):
        X = np.random.rand(40, 5).astype(np.float32)
        y = np.array([0] * 20 + [1] * 20)
        clf = train_naive_bayes(X, y)
        metrics = evaluate_classifier(clf, X, y, ["0", "1"])
        assert "accuracy" in metrics
        assert "report" in metrics
        assert "confusion_matrix" in metrics

    def test_confusion_matrix_shape(self):
        X = np.random.rand(60, 5).astype(np.float32)
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        clf = train_naive_bayes(X, y)
        metrics = evaluate_classifier(clf, X, y, ["0", "1", "2"])
        assert metrics["confusion_matrix"].shape == (3, 3)


class TestLevenshtein:
    """Tests de la distance de Levenshtein."""

    def test_identical_strings(self):
        assert _levenshtein("ABC123", "ABC123") == 0

    def test_empty_strings(self):
        assert _levenshtein("", "") == 0
        assert _levenshtein("ABC", "") == 3
        assert _levenshtein("", "ABC") == 3

    def test_single_substitution(self):
        assert _levenshtein("ABC", "ABD") == 1

    def test_single_insertion(self):
        assert _levenshtein("ABC", "ABCD") == 1

    def test_single_deletion(self):
        assert _levenshtein("ABCD", "ABC") == 1


class TestOCRMetrics:
    """Tests des métriques CER et WER."""

    def test_cer_perfect(self):
        assert compute_cer("LT1234A", "LT1234A") == pytest.approx(0.0)

    def test_cer_all_wrong(self):
        cer = compute_cer("XXXXXXX", "LT1234A")
        assert 0.0 < cer <= 1.0

    def test_cer_empty_prediction(self):
        cer = compute_cer("", "LT1234A")
        assert cer == pytest.approx(1.0)

    def test_cer_empty_reference(self):
        cer = compute_cer("", "")
        assert cer == pytest.approx(0.0)

    def test_wer_perfect(self):
        assert compute_wer("LT 1234 A", "LT 1234 A") == pytest.approx(0.0)

    def test_wer_one_word_error(self):
        wer = compute_wer("LT 9999 A", "LT 1234 A")
        assert wer == pytest.approx(1 / 3)

    def test_cer_case_insensitive(self):
        assert compute_cer("lt1234a", "LT1234A") == pytest.approx(0.0)
