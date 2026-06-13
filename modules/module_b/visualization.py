"""
Module B — Étape 2 : Réduction de dimension et visualisation des embeddings CNN
PlateVision / MINT-DGI Cameroun

PCA et t-SNE projettent les embeddings 256D en 2D pour observer visuellement
la structure des données avant et après le clustering K-Means (Étape 3).
"""

import logging
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

_DEFAULT_DATA    = Path("data/processed")
_DEFAULT_FIGURES = Path("reports/rapport_technique/figures")

# Mapping entier → caractère : 0-9 → '0'-'9' ; 10-35 → 'A'-'Z'
_LABEL_NAMES: list[str] = (
    [str(i) for i in range(10)] + [chr(ord("A") + i) for i in range(26)]
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def load_embeddings(
    data_dir: Path = _DEFAULT_DATA,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Charge embeddings.npy et metadata.csv depuis data_dir.
    Vérifie la cohérence des longueurs.
    Retourne (embeddings, metadata).
    """
    data_dir = Path(data_dir)
    emb_path  = data_dir / "embeddings.npy"
    meta_path = data_dir / "metadata.csv"

    if not emb_path.exists():
        raise RuntimeError(
            "Embeddings absents. Exécute d'abord : python main.py --module B1"
        )
    if not meta_path.exists():
        raise RuntimeError(
            "metadata.csv absent. Exécute d'abord : python main.py --module B1"
        )

    embeddings = np.load(emb_path).astype(np.float32)
    metadata   = pd.read_csv(meta_path)

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Incohérence : {len(embeddings)} embeddings mais "
            f"{len(metadata)} lignes dans metadata.csv"
        )

    logger.info(
        "Embeddings chargés : %d vecteurs de dimension %d",
        embeddings.shape[0], embeddings.shape[1],
    )
    return embeddings, metadata


# ══════════════════════════════════════════════════════════════════════════════
# 2. RÉDUCTION PCA
# ══════════════════════════════════════════════════════════════════════════════

def reduce_pca(
    embeddings: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[np.ndarray, PCA]:
    """
    Réduit les embeddings en n_components dimensions par PCA.
    Log la variance expliquée par composante.
    Retourne (coords_2d, pca_model).
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(embeddings).astype(np.float32)

    var = pca.explained_variance_ratio_ * 100
    total = var.sum()
    if n_components >= 2:
        logger.info(
            "PCA — variance expliquée : PC1=%.1f%%, PC2=%.1f%%, total=%.1f%%",
            var[0], var[1], total,
        )
    else:
        logger.info("PCA — variance expliquée : PC1=%.1f%%", var[0])

    return coords, pca


# ══════════════════════════════════════════════════════════════════════════════
# 3. RÉDUCTION t-SNE
# ══════════════════════════════════════════════════════════════════════════════

def reduce_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
    verbose: bool = True,
) -> np.ndarray:
    """
    Réduit les embeddings en n_components dimensions par t-SNE.
    Log le temps de calcul. Avertit si N > 5000.
    Retourne coords_2d (N, n_components).
    """
    n = len(embeddings)
    if n > 5000:
        logger.warning(
            "t-SNE sur %d points — calcul long (~%d min estimé)",
            n, n // 500,
        )

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        verbose=int(verbose),
    )

    t0 = time.perf_counter()
    coords = tsne.fit_transform(embeddings).astype(np.float32)
    elapsed = time.perf_counter() - t0

    logger.info("t-SNE calculé en %.1fs sur %d points", elapsed, n)
    return coords


# ══════════════════════════════════════════════════════════════════════════════
# 4. SCATTER SANS COLORATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_embeddings_raw(
    coords_2d: np.ndarray,
    method_name: str,
    out_path: Path,
    title: "str | None" = None,
) -> None:
    """
    Scatter plot des embeddings projetés en 2D, sans coloration par label.
    Couleur unique "#4C72B0", alpha=0.4, taille=8, marker=".".
    Sauvegarde dans out_path (PNG, dpi=150).
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c="#4C72B0", alpha=0.4, s=8, marker=".",
    )

    ax.set_title(title or f"Embeddings CNN — {method_name} (sans labels)", fontsize=13)
    ax.set_xlabel(f"{method_name} dim 1", fontsize=11)
    ax.set_ylabel(f"{method_name} dim 2", fontsize=11)

    ax.annotate(
        f"N={len(coords_2d)} embeddings × 256D",
        xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=9, color="gray",
    )

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure sauvegardée : %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# 5. SCATTER COLORIÉ PAR LABEL
# ══════════════════════════════════════════════════════════════════════════════

def plot_embeddings_by_label(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    method_name: str,
    out_path: Path,
    label_names: "list[str] | None" = None,
) -> None:
    """
    Scatter plot des embeddings projetés en 2D, colorié par label (0-35 classes).
    Utilise matplotlib.cm.tab20, cyclé sur 36 classes.
    Légende limitée aux classes effectivement présentes dans labels.
    Sauvegarde dans out_path (PNG, dpi=150).
    """
    if label_names is None:
        label_names = _LABEL_NAMES

    unique_labels = np.unique(labels)
    cmap          = plt.colormaps["tab20"]
    n_colors      = 20  # tab20 a 20 couleurs — on cycle

    fig, ax = plt.subplots(figsize=(14, 10))

    for lbl in unique_labels:
        mask = labels == lbl
        color = cmap(int(lbl) % n_colors)
        name  = label_names[int(lbl)] if int(lbl) < len(label_names) else str(lbl)
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=[color], alpha=0.5, s=10, marker=".",
            label=name,
        )

    ax.set_title(
        f"Embeddings CNN — {method_name} (colorié par classe)", fontsize=13
    )
    ax.set_xlabel(f"{method_name} dim 1", fontsize=11)
    ax.set_ylabel(f"{method_name} dim 2", fontsize=11)

    # Légende compacte — colonnes multiples si beaucoup de classes
    n_cols = max(1, len(unique_labels) // 10)
    ax.legend(
        loc="upper right", fontsize=7,
        markerscale=2, ncol=n_cols,
        title="Classe", title_fontsize=8,
        framealpha=0.7,
    )

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure sauvegardée : %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# 6. COURBE DE VARIANCE EXPLIQUÉE
# ══════════════════════════════════════════════════════════════════════════════

def plot_variance_explained(
    pca_model: PCA,
    out_path: Path,
    n_components_show: int = 50,
) -> None:
    """
    Produit une figure 2 panneaux : variance par composante (bar) et variance
    cumulée (courbe) avec seuil 90%. Gère le cas où pca_model a moins de
    n_components_show composantes disponibles.
    Sauvegarde dans out_path (PNG, dpi=150).
    """
    var_ratio = pca_model.explained_variance_ratio_
    n_avail   = len(var_ratio)

    if n_avail < n_components_show:
        logger.warning(
            "pca_model n'a que %d composantes (< %d demandées) — "
            "tracé limité aux composantes disponibles.",
            n_avail, n_components_show,
        )
        n_show = n_avail
    else:
        n_show = n_components_show

    var_show  = var_ratio[:n_show] * 100
    cum_show  = np.cumsum(var_ratio[:n_show]) * 100
    indices   = np.arange(1, n_show + 1)

    # Trouver le composant où la variance cumulée dépasse 90 %
    cum_all   = np.cumsum(var_ratio) * 100
    idx_90    = int(np.searchsorted(cum_all, 90.0)) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PCA — Variance expliquée par composante", fontsize=13)

    # ── Panneau gauche : variance par composante ───────────────────────────
    ax1.bar(indices, var_show, color="#4C72B0", alpha=0.8)
    ax1.set_xlabel("Composante principale", fontsize=11)
    ax1.set_ylabel("Variance expliquée (%)", fontsize=11)
    ax1.set_title(f"Variance par composante (top {n_show})", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)

    # ── Panneau droit : variance cumulée ──────────────────────────────────
    ax2.plot(indices, cum_show, color="#C44E52", linewidth=2, marker="o",
             markersize=3)
    ax2.axhline(90, color="green", linestyle="--", linewidth=1.2,
                label="Seuil 90%")
    if idx_90 <= n_show:
        ax2.axvline(idx_90, color="orange", linestyle=":", linewidth=1.2,
                    label=f"90% atteint à CP{idx_90}")
    ax2.set_xlabel("Nombre de composantes", fontsize=11)
    ax2.set_ylabel("Variance cumulée (%)", fontsize=11)
    ax2.set_title("Variance cumulée", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure sauvegardée : %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# 7. PIPELINE COMPLET
# ══════════════════════════════════════════════════════════════════════════════

def run_visualization_pipeline(
    data_dir: Path = _DEFAULT_DATA,
    figures_dir: Path = _DEFAULT_FIGURES,
    tsne_perplexity: float = 30.0,
    tsne_n_iter: int = 1000,
    run_tsne: bool = True,
    random_state: int = 42,
) -> dict:
    """
    Orchestre la réduction de dimension et la génération des figures pour
    le Module B Étape 2. Produit PCA et (optionnellement) t-SNE.

    Retourne un dict contenant pca_coords, pca_model, tsne_coords,
    embeddings et metadata pour réutilisation par le clustering (Étape 3).
    """
    data_dir    = Path(data_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Chargement ─────────────────────────────────────────────────────────
    embeddings, metadata = load_embeddings(data_dir)

    # Labels numériques depuis metadata
    if "label_int" in metadata.columns:
        labels = metadata["label_int"].to_numpy(dtype=np.int32)
    else:
        labels = np.zeros(len(metadata), dtype=np.int32)
        logger.warning("Colonne 'label_int' absente — labels mis à 0.")

    # ── 2. PCA 2D ─────────────────────────────────────────────────────────────
    pca_coords, pca_model = reduce_pca(embeddings, n_components=2,
                                        random_state=random_state)

    # ── 3. Variance expliquée (PCA refittée sur min(50, D) composantes) ───────
    d = embeddings.shape[1]
    n_var = min(50, d, len(embeddings))
    pca_var = PCA(n_components=n_var, random_state=random_state)
    pca_var.fit(embeddings)
    plot_variance_explained(pca_var, figures_dir / "variance_explained.png")

    # ── 4 & 5. Figures PCA ────────────────────────────────────────────────────
    plot_embeddings_raw(pca_coords, "PCA", figures_dir / "pca_raw.png")
    plot_embeddings_by_label(
        pca_coords, labels, "PCA",
        figures_dir / "pca_by_label.png",
    )

    # ── 6. t-SNE (optionnel) ──────────────────────────────────────────────────
    tsne_coords: "np.ndarray | None" = None
    if run_tsne:
        tsne_coords = reduce_tsne(
            embeddings,
            perplexity=tsne_perplexity,
            n_iter=tsne_n_iter,
            random_state=random_state,
        )
        plot_embeddings_raw(tsne_coords, "t-SNE", figures_dir / "tsne_raw.png")
        plot_embeddings_by_label(
            tsne_coords, labels, "t-SNE",
            figures_dir / "tsne_by_label.png",
        )

    # ── Résumé terminal ───────────────────────────────────────────────────────
    print("\n=== Module B — Étape 2 terminée ===")
    print(f"Figures générées dans : {figures_dir}")
    print("  pca_raw.png, pca_by_label.png, variance_explained.png")
    if run_tsne:
        print("  tsne_raw.png, tsne_by_label.png")
    print("Prêt pour le clustering K-Means (Étape 3)")

    return {
        "pca_coords":  pca_coords,
        "pca_model":   pca_model,
        "tsne_coords": tsne_coords,
        "embeddings":  embeddings,
        "metadata":    metadata,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Module B Étape 2 — Réduction de dimension et visualisation"
    )
    parser.add_argument("--data-dir",        type=Path,  default=_DEFAULT_DATA)
    parser.add_argument("--figures-dir",     type=Path,  default=_DEFAULT_FIGURES)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-n-iter",     type=int,   default=1000)
    parser.add_argument("--no-tsne",         action="store_true")
    parser.add_argument("--random-state",    type=int,   default=42)
    args = parser.parse_args()

    run_visualization_pipeline(
        data_dir        = args.data_dir,
        figures_dir     = args.figures_dir,
        tsne_perplexity = args.tsne_perplexity,
        tsne_n_iter     = args.tsne_n_iter,
        run_tsne        = not args.no_tsne,
        random_state    = args.random_state,
    )
