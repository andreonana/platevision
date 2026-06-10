"""
PlateVision — Module B : Visualisation des clusters.

Ce module produit des visualisations 2D des clusters d'embeddings CNN
via réduction de dimensionnalité PCA et t-SNE. Ces visualisations aident
à interpréter la structure des groupes de plaques détectées.

Rôle dans le pipeline : analyse exploratoire et présentation des résultats Module B.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_pca_2d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str = "PCA 2D — Clusters d'embeddings CNN",
) -> None:
    """
    Visualise les clusters en 2D via PCA.

    Args:
        embeddings: Matrice d'embeddings (n_samples, dim)
        labels: Labels de clusters
        output_path: Dossier de sauvegarde de la figure
        title: Titre du graphique
    """
    from sklearn.decomposition import PCA

    logger.info("Réduction PCA vers 2 dimensions...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)

    variance_ratio = pca.explained_variance_ratio_
    logger.info(f"Variance expliquée : PC1={variance_ratio[0]:.3f}, PC2={variance_ratio[1]:.3f}")

    n_clusters = len(np.unique(labels))
    colors = cm.tab10(np.linspace(0, 1, n_clusters))

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, cluster_id in enumerate(np.unique(labels)):
        mask = labels == cluster_id
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[idx]],
            label=f"Cluster {cluster_id} (n={mask.sum()})",
            alpha=0.7,
            s=30,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"PC1 ({variance_ratio[0]:.1%} variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({variance_ratio[1]:.1%} variance)", fontsize=11)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / "pca_clusters.png"
    plt.tight_layout()
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure PCA sauvegardée : {fig_path}")


def plot_tsne_2d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    perplexity: int = 30,
    title: str = "t-SNE 2D — Clusters d'embeddings CNN",
) -> None:
    """
    Visualise les clusters en 2D via t-SNE.

    t-SNE préserve mieux la structure locale que PCA mais est plus lent.

    Args:
        embeddings: Matrice d'embeddings
        labels: Labels de clusters
        output_path: Dossier de sauvegarde
        perplexity: Paramètre de perplexité t-SNE (défaut : 30)
        title: Titre du graphique
    """
    from sklearn.manifold import TSNE

    logger.info(f"Calcul t-SNE (perplexity={perplexity})... (peut prendre quelques minutes)")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=int(os.getenv("RANDOM_SEED", 42)),
        n_iter=1000,
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    n_clusters = len(np.unique(labels))
    colors = cm.tab10(np.linspace(0, 1, n_clusters))

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, cluster_id in enumerate(np.unique(labels)):
        mask = labels == cluster_id
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[idx]],
            label=f"Cluster {cluster_id} (n={mask.sum()})",
            alpha=0.7,
            s=30,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE dimension 2", fontsize=11)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / "tsne_clusters.png"
    plt.tight_layout()
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure t-SNE sauvegardée : {fig_path}")


def plot_elbow_curve(inertias: list, k_range: range, output_path: Path) -> None:
    """
    Trace la courbe du coude pour déterminer le K optimal.

    Args:
        inertias: Liste des inerties pour chaque K
        k_range: Plage de valeurs K testées
        output_path: Dossier de sauvegarde
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), inertias, "bo-", markersize=8, linewidth=2)
    ax.set_title("Méthode du Coude — Sélection du K optimal", fontsize=13, fontweight="bold")
    ax.set_xlabel("Nombre de clusters K", fontsize=11)
    ax.set_ylabel("Inertie (distorsion)", fontsize=11)
    ax.grid(True, alpha=0.3)

    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / "elbow_curve.png"
    plt.tight_layout()
    plt.savefig(str(fig_path), dpi=150)
    plt.close()
    logger.info(f"Courbe du coude sauvegardée : {fig_path}")


def plot_clusters(embeddings_path: Path, labels: np.ndarray) -> None:
    """
    Fonction principale : génère toutes les visualisations des clusters.

    Args:
        embeddings_path: Chemin vers le fichier .npy des embeddings
        labels: Labels de clusters
    """
    embeddings = np.load(str(embeddings_path))
    output_path = Path(os.getenv("OUTPUT_FIGURES_PATH", "reports/figures/"))

    plot_pca_2d(embeddings, labels, output_path)
    plot_tsne_2d(embeddings, labels, output_path)
    logger.info("Toutes les visualisations générées.")
