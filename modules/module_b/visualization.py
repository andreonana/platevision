"""
Module B — Visualisation des clusters par réduction de dimensionnalité.

Rôle dans le pipeline PlateVision :
    Réduit la dimension des embeddings à 2D via PCA ou t-SNE pour permettre
    une inspection visuelle de la qualité du clustering. Les graphiques
    produits illustrent la séparation des groupes de plaques MINT/DGI.

Responsable : Étudiant 3
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


def reduce_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Réduit les embeddings à n_components dimensions via PCA."""
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_
    logger.info("PCA : variance expliquée = %.2f%% + %.2f%%",
                explained[0] * 100, explained[1] * 100)
    return reduced, pca


def reduce_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
) -> np.ndarray:
    """Réduit les embeddings à 2D via t-SNE."""
    sample_size = min(5000, len(embeddings))
    if len(embeddings) > sample_size:
        logger.info("t-SNE : sous-échantillonnage à %d points", sample_size)
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[indices]

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, n_iter=1000)
    reduced = tsne.fit_transform(embeddings)
    return reduced


def plot_clusters_2d(
    reduced: np.ndarray,
    labels: np.ndarray,
    method_name: str,
    output_path: Path,
    title: str | None = None,
) -> None:
    """
    Génère un scatter plot 2D coloré par cluster.

    Args:
        reduced: Embeddings réduits en 2D
        labels: Labels de cluster pour chaque point
        method_name: 'PCA' ou 't-SNE' (utilisé dans le titre et nom de fichier)
        output_path: Dossier de sortie pour la figure
        title: Titre personnalisé (optionnel)
    """
    n_clusters = len(np.unique(labels))
    palette = sns.color_palette("tab10", n_clusters)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=[palette[l] for l in labels],
        alpha=0.7,
        s=20,
        edgecolors="none",
    )

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[i],
                   markersize=8, label=f"Cluster {i}")
        for i in range(n_clusters)
    ]
    ax.legend(handles=handles, title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_title(title or f"Visualisation {method_name} — {n_clusters} clusters PlateVision")
    ax.set_xlabel(f"{method_name} Dimension 1")
    ax.set_ylabel(f"{method_name} Dimension 2")
    plt.tight_layout()

    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / f"clusters_{method_name.lower()}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Graphique sauvegardé : %s", fig_path)


def plot_elbow_curve(
    k_values: list[int],
    inertia_scores: list[float],
    output_path: Path,
) -> None:
    """Génère la courbe du coude pour le choix optimal de k."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, inertia_scores, "o-", color="steelblue", linewidth=2)
    ax.set_title("Méthode du coude — Choix du nombre de clusters K")
    ax.set_xlabel("Nombre de clusters K")
    ax.set_ylabel("Inertie (Within-Cluster Sum of Squares)")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    fig_path = output_path / "elbow_curve.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info("Courbe du coude sauvegardée : %s", fig_path)


def generate_all_visualizations(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    """Génère PCA et t-SNE en un seul appel."""
    logger.info("Génération des visualisations PCA et t-SNE...")

    reduced_pca, _ = reduce_pca(embeddings)
    plot_clusters_2d(reduced_pca, labels, "PCA", output_path)

    reduced_tsne = reduce_tsne(embeddings)
    plot_clusters_2d(reduced_tsne, labels, "t-SNE", output_path)

    logger.info("Visualisations générées dans : %s", output_path)
