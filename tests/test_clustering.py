"""Tests unitaires pour le Module B — Clustering K-Means."""

import numpy as np
import pytest

from modules.module_b.clustering import normalize_embeddings, train_kmeans
from modules.module_b.interpret_clusters import analyze_cluster_texts, CAMEROON_PLATE_CATEGORIES


class TestNormalizeEmbeddings:
    """Tests de normalisation des embeddings."""

    def test_output_shape_preserved(self):
        X = np.random.rand(50, 128).astype(np.float32)
        X_norm, scaler = normalize_embeddings(X)
        assert X_norm.shape == X.shape

    def test_zero_mean_after_normalization(self):
        X = np.random.rand(100, 32).astype(np.float32) * 100
        X_norm, _ = normalize_embeddings(X)
        assert np.abs(X_norm.mean()) < 0.1

    def test_unit_variance_after_normalization(self):
        X = np.random.rand(200, 16).astype(np.float32) * 50 + 10
        X_norm, _ = normalize_embeddings(X)
        assert np.abs(X_norm.std() - 1.0) < 0.1


class TestKMeansClustering:
    """Tests du clustering K-Means."""

    def test_labels_count_matches_samples(self):
        X = np.random.rand(80, 64).astype(np.float32)
        _, labels, _ = train_kmeans(X, n_clusters=4)
        assert len(labels) == 80

    def test_labels_in_valid_range(self):
        X = np.random.rand(60, 32).astype(np.float32)
        _, labels, _ = train_kmeans(X, n_clusters=3)
        assert labels.min() >= 0
        assert labels.max() <= 2

    def test_metrics_present(self):
        X = np.random.rand(50, 16).astype(np.float32)
        _, _, metrics = train_kmeans(X, n_clusters=3)
        assert "silhouette_score" in metrics
        assert "inertia" in metrics
        assert "davies_bouldin_score" in metrics

    def test_silhouette_score_range(self):
        X = np.random.rand(60, 8).astype(np.float32)
        _, _, metrics = train_kmeans(X, n_clusters=3)
        assert -1.0 <= metrics["silhouette_score"] <= 1.0

    def test_inertia_positive(self):
        X = np.random.rand(40, 8).astype(np.float32)
        _, _, metrics = train_kmeans(X, n_clusters=2)
        assert metrics["inertia"] >= 0.0


class TestClusterInterpretation:
    """Tests de l'interprétation sémantique des clusters."""

    def test_known_prefix_recognized(self):
        texts = ["LT 1234 A", "LT 5678 B", "LT 9012 C"] * 5
        labels = np.zeros(15, dtype=int)
        result = analyze_cluster_texts(texts, labels, n_clusters=1)
        assert result[0]["dominant_prefix"] == "LT"
        assert result[0]["dominant_category"] == "Littoral (Douala)"

    def test_cd_prefix_recognized(self):
        texts = ["CD 001 A", "CD 002 B", "CD 003 C"] * 5
        labels = np.zeros(15, dtype=int)
        result = analyze_cluster_texts(texts, labels, n_clusters=1)
        assert result[0]["dominant_category"] == "Corps Diplomatique"

    def test_unknown_prefix_handled(self):
        texts = ["XX 9999 Z"] * 5
        labels = np.zeros(5, dtype=int)
        result = analyze_cluster_texts(texts, labels, n_clusters=1)
        assert result[0]["dominant_category"] == "Catégorie inconnue"

    def test_cluster_size_correct(self):
        texts = ["LT 1234 A"] * 10 + ["CE 5678 B"] * 5
        labels = np.array([0] * 10 + [1] * 5)
        result = analyze_cluster_texts(texts, labels, n_clusters=2)
        assert result[0]["size"] == 10
        assert result[1]["size"] == 5

    def test_all_cameroon_prefixes_defined(self):
        """Vérifie que les préfixes de la carte administrative sont tous définis."""
        required_prefixes = {"LT", "CE", "NO", "EN", "AD", "NW", "SW", "OU", "ES", "SU", "CD"}
        for prefix in required_prefixes:
            assert prefix in CAMEROON_PLATE_CATEGORIES, f"Préfixe manquant : {prefix}"
