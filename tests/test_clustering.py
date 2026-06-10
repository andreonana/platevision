"""
PlateVision — Tests unitaires : Clustering K-Means (Module B).
"""

import numpy as np
import pytest
from pathlib import Path


class TestKMeansClustering:
    """Tests pour le clustering K-Means sur embeddings."""

    def test_run_kmeans_returns_correct_shapes(self, tmp_path):
        from modules.module_b.clustering import run_kmeans

        # Création d'un fichier d'embeddings factice
        embeddings = np.random.randn(100, 128).astype(np.float32)
        embeddings_path = tmp_path / "embeddings.npy"
        np.save(str(embeddings_path), embeddings)

        labels, centroids = run_kmeans(embeddings_path, n_clusters=5)

        assert len(labels) == 100
        assert centroids.shape[0] == 5

    def test_run_kmeans_labels_in_valid_range(self, tmp_path):
        from modules.module_b.clustering import run_kmeans

        embeddings = np.random.randn(50, 64).astype(np.float32)
        path = tmp_path / "emb.npy"
        np.save(str(path), embeddings)

        labels, _ = run_kmeans(path, n_clusters=3)

        assert set(labels).issubset({0, 1, 2})

    def test_load_embeddings_raises_on_missing_file(self, tmp_path):
        from modules.module_b.clustering import load_embeddings

        with pytest.raises(FileNotFoundError):
            load_embeddings(tmp_path / "nonexistent.npy")

    def test_cluster_stats_keys(self, tmp_path):
        from modules.module_b.clustering import run_kmeans, compute_cluster_stats

        embeddings = np.random.randn(60, 32).astype(np.float32)
        path = tmp_path / "emb.npy"
        np.save(str(path), embeddings)

        labels, _ = run_kmeans(path, n_clusters=4)
        stats = compute_cluster_stats(embeddings, labels)

        for cluster_id in range(4):
            assert cluster_id in stats
            assert "n_samples" in stats[cluster_id]
            assert "intra_variance" in stats[cluster_id]


class TestInterpretClusters:
    """Tests pour l'interprétation des clusters."""

    def test_assign_cluster_returns_string(self):
        from modules.module_b.interpret_clusters import assign_cluster_to_category

        result = assign_cluster_to_category(0, ["AB123CD", "XY456ZZ"])
        assert isinstance(result, str)

    def test_interpret_returns_all_clusters(self):
        from modules.module_b.interpret_clusters import interpret

        labels = np.array([0, 1, 0, 2, 1, 2])
        centroids = np.random.randn(3, 10)

        result = interpret(labels, centroids)

        assert 0 in result
        assert 1 in result
        assert 2 in result

    def test_interpret_proportion_sums_to_100(self):
        from modules.module_b.interpret_clusters import interpret

        labels = np.array([0, 0, 1, 1, 2, 2])
        centroids = np.random.randn(3, 10)

        result = interpret(labels, centroids)

        total = sum(r["n_samples"] for r in result.values())
        assert total == len(labels)
