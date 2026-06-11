"""
Module B — Clustering K-Means sur embeddings CNN des plaques détectées.

Rôle dans le pipeline PlateVision :
    Regroupe les plaques détectées en clusters homogènes à partir des
    embeddings extraits par la backbone YOLOv8 (Module A2). Les clusters
    sont ensuite interprétés pour correspondre aux catégories de procédures
    administratives MINT/DGI (immatriculations diplomatiques, commerciales,
    particulières, transit, etc.).

Responsable : Étudiant 3
"""

import logging
import os
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

N_CLUSTERS = int(os.getenv("KMEANS_N_CLUSTERS", 8))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    """
    Charge les embeddings CNN depuis un fichier .npy.

    Args:
        embeddings_path: Chemin vers le fichier .npy généré par le Module A2

    Returns:
        Tableau numpy de forme (n_samples, embedding_dim)
    """
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Fichier embeddings introuvable : {embeddings_path}. "
            "Exécuter d'abord le Module A2 pour générer les embeddings."
        )
    embeddings = np.load(embeddings_path)
    logger.info("Embeddings chargés : %d vecteurs de dimension %d", *embeddings.shape)
    return embeddings


def normalize_embeddings(embeddings: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """Normalise les embeddings (moyenne 0, variance 1) avant clustering."""
    scaler = StandardScaler()
    embeddings_norm = scaler.fit_transform(embeddings)
    logger.info("Embeddings normalisés (StandardScaler)")
    return embeddings_norm, scaler


def find_optimal_k(
    embeddings: np.ndarray,
    k_range: range | None = None,
) -> tuple[list[float], list[float]]:
    """
    Calcule les scores d'inertie et silhouette pour différentes valeurs de k.
    Utile pour la méthode du coude et le choix optimal de k.

    Args:
        embeddings: Embeddings normalisés
        k_range: Plage de valeurs k à tester [défaut: range(2, 15)]

    Returns:
        Tuple (inertia_scores, silhouette_scores)
    """
    if k_range is None:
        k_range = range(2, 15)

    inertia_scores = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        inertia_scores.append(kmeans.inertia_)

        if k > 1:
            sil = silhouette_score(embeddings, labels, sample_size=min(1000, len(embeddings)))
            silhouette_scores.append(sil)
            logger.info("k=%d | Inertie=%.2f | Silhouette=%.4f", k, kmeans.inertia_, sil)

    return inertia_scores, silhouette_scores


def train_kmeans(embeddings: np.ndarray, n_clusters: int = N_CLUSTERS) -> tuple:
    """
    Entraîne le modèle K-Means et retourne le modèle et les labels.

    Args:
        embeddings: Embeddings normalisés
        n_clusters: Nombre de clusters

    Returns:
        Tuple (kmeans_model, labels, métriques)
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_SEED,
        n_init=10,
        max_iter=300,
    )
    labels = kmeans.fit_predict(embeddings)

    silhouette = silhouette_score(embeddings, labels, sample_size=min(1000, len(embeddings)))
    davies_bouldin = davies_bouldin_score(embeddings, labels)

    metrics = {
        "n_clusters": n_clusters,
        "inertia": float(kmeans.inertia_),
        "silhouette_score": float(silhouette),
        "davies_bouldin_score": float(davies_bouldin),
    }

    logger.info("K-Means entraîné : k=%d, Inertie=%.2f, Silhouette=%.4f, DB=%.4f",
                n_clusters, kmeans.inertia_, silhouette, davies_bouldin)

    return kmeans, labels, metrics


def run_clustering_pipeline(embeddings_path: Path) -> dict:
    """
    Exécute le pipeline complet de clustering : chargement → normalisation → K-Means.

    Args:
        embeddings_path: Chemin vers le fichier .npy des embeddings

    Returns:
        Dictionnaire des résultats et métriques.
    """
    logger.info("=== Pipeline Clustering démarré ===")

    embeddings = load_embeddings(embeddings_path)
    embeddings_norm, scaler = normalize_embeddings(embeddings)

    kmeans, labels, metrics = train_kmeans(embeddings_norm, n_clusters=N_CLUSTERS)

    # Sauvegarde des labels et du modèle
    labels_path = embeddings_path.parent / "cluster_labels.npy"
    np.save(labels_path, labels)
    logger.info("Labels de clusters sauvegardés : %s", labels_path)

    import joblib
    model_path = embeddings_path.parent / "kmeans_model.pkl"
    joblib.dump(kmeans, model_path)
    logger.info("Modèle K-Means sauvegardé : %s", model_path)

    logger.info("=== Pipeline Clustering terminé ===")
    return {"labels": labels, "metrics": metrics, "model": kmeans}
