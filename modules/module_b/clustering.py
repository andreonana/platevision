"""
PlateVision — Module B : Clustering K-Means sur embeddings CNN.

Ce module applique l'algorithme K-Means sur les embeddings extraits par
un réseau de neurones convolutif (CNN) pour regrouper automatiquement
les plaques en clusters homogènes. Ces clusters sont ensuite interprétés
pour correspondre aux catégories de véhicules et procédures MINT/DGI.

Rôle dans le pipeline : analyse non supervisée post-détection.
"""

import logging
import os
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    """
    Charge les embeddings CNN depuis un fichier .npy.

    Args:
        embeddings_path: Chemin vers le fichier d'embeddings

    Returns:
        Matrice d'embeddings de forme (n_samples, embedding_dim)
    """
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Fichier d'embeddings introuvable : {embeddings_path}")

    embeddings = np.load(str(embeddings_path))
    logger.info(f"Embeddings chargés : {embeddings.shape[0]} échantillons, dimension {embeddings.shape[1]}")
    return embeddings


def find_optimal_k(embeddings: np.ndarray, k_range: range = range(2, 15)) -> int:
    """
    Trouve le nombre optimal de clusters via la méthode du coude (inertie).

    Args:
        embeddings: Matrice d'embeddings normalisés
        k_range: Plage de valeurs K à tester

    Returns:
        Valeur K optimale estimée
    """
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        inertias.append(km.inertia_)
        sil = silhouette_score(embeddings, labels) if k > 1 else 0
        silhouettes.append(sil)
        logger.debug(f"K={k} | Inertie={km.inertia_:.2f} | Silhouette={sil:.4f}")

    # Méthode du coude : point de plus forte décroissance de l'inertie
    diffs = np.diff(inertias)
    second_diffs = np.diff(diffs)
    optimal_k = list(k_range)[np.argmax(second_diffs) + 2]

    logger.info(f"K optimal estimé (méthode du coude) : {optimal_k}")
    logger.info(f"K optimal (meilleur silhouette) : {list(k_range)[np.argmax(silhouettes)]}")

    return optimal_k


def run_kmeans(
    embeddings_path: Path,
    n_clusters: int = 8,
    output_path: Path = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applique K-Means sur les embeddings et retourne les labels et centroïdes.

    Args:
        embeddings_path: Chemin vers le fichier .npy des embeddings
        n_clusters: Nombre de clusters K
        output_path: Dossier de sauvegarde du modèle (optionnel)

    Returns:
        Tuple (labels, centroids) : labels de clusters et centroïdes
    """
    embeddings = load_embeddings(embeddings_path)

    # Normalisation L2 des embeddings avant clustering
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    logger.info(f"Démarrage K-Means avec K={n_clusters}...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=int(os.getenv("RANDOM_SEED", 42)),
        n_init=10,
        max_iter=300,
    )
    labels = kmeans.fit_predict(embeddings_scaled)
    centroids = kmeans.cluster_centers_

    # Métriques de qualité du clustering
    sil_score = silhouette_score(embeddings_scaled, labels)
    db_score = davies_bouldin_score(embeddings_scaled, labels)
    logger.info(f"Score Silhouette : {sil_score:.4f} (plus proche de 1 = meilleur)")
    logger.info(f"Score Davies-Bouldin : {db_score:.4f} (plus proche de 0 = meilleur)")

    # Distribution des clusters
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        logger.info(f"  Cluster {cluster_id} : {count} échantillons ({count/len(labels)*100:.1f}%)")

    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(kmeans, output_path / "kmeans_model.pkl")
        joblib.dump(scaler, output_path / "kmeans_scaler.pkl")
        np.save(output_path / "cluster_labels.npy", labels)
        logger.info(f"Modèle K-Means sauvegardé dans {output_path}")

    return labels, centroids


def compute_cluster_stats(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """
    Calcule les statistiques descriptives pour chaque cluster.

    Args:
        embeddings: Matrice d'embeddings
        labels: Labels de clusters correspondants

    Returns:
        Dictionnaire {cluster_id: {stats}}
    """
    stats = {}
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        cluster_embeddings = embeddings[mask]
        stats[int(cluster_id)] = {
            "n_samples": int(mask.sum()),
            "mean_norm": float(np.linalg.norm(cluster_embeddings, axis=1).mean()),
            "std_norm": float(np.linalg.norm(cluster_embeddings, axis=1).std()),
            "intra_variance": float(np.var(cluster_embeddings)),
        }
    return stats
