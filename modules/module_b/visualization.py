"""
Module B — Étape 5 : Visualisation des clusters K-Means (§4.2 cahier des charges)
PlateVision / MINT-DGI Cameroun

Exigence §4.2 : "Visualiser via PCA 2D ou t-SNE avec axes labellisés et titre
explicatif" + "Comparer les clusters avec les annotations du dataset."

Critère jury §7.1 : "K-Means : justification k, interprétation procédures MINT/DGI"
"""

import logging
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_DEFAULT_DATA    = Path("data/processed")
_DEFAULT_MODELS  = Path("models")
_DEFAULT_FIGURES = Path("reports/rapport_technique/figures")

# ── Import CLUSTER_PROCEDURES ─────────────────────────────────────────────────
try:
    from modules.module_b.clustering import CLUSTER_PROCEDURES
except Exception as _e:
    logger.warning("Import CLUSTER_PROCEDURES échoué (%s) — noms génériques utilisés", _e)
    CLUSTER_PROCEDURES = {}


# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def load_data(
    data_dir: Path = _DEFAULT_DATA,
    models_dir: Path = _DEFAULT_MODELS,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, object]:
    data_dir   = Path(data_dir)
    models_dir = Path(models_dir)

    emb_path    = data_dir   / "embeddings.npy"
    meta_path   = data_dir   / "metadata.csv"
    scaler_path = models_dir / "kmeans_scaler.pkl"
    km_path     = models_dir / "kmeans_model.pkl"

    if not emb_path.exists():
        raise RuntimeError(
            "embeddings.npy absent. Exécute : python main.py --module B1"
        )
    if not meta_path.exists():
        raise RuntimeError(
            "metadata.csv absent. Exécute : python main.py --module B1"
        )
    if not km_path.exists():
        raise RuntimeError(
            "kmeans_model.pkl absent. Exécute d'abord : python main.py --module B4"
        )

    metadata = pd.read_csv(meta_path)
    if "cluster_id" not in metadata.columns:
        raise RuntimeError(
            "cluster_id absent de metadata.csv. Exécute d'abord l'Étape 4 :\n"
            "python main.py --module B4"
        )

    embeddings = np.load(emb_path).astype(np.float32)

    if scaler_path.exists():
        scaler            = joblib.load(scaler_path)
        embeddings_scaled = scaler.transform(embeddings).astype(np.float32)
    else:
        logger.warning("kmeans_scaler.pkl absent — StandardScaler refitté sur embeddings")
        scaler            = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings).astype(np.float32)

    cluster_labels = metadata["cluster_id"].values.astype(np.int32)
    logger.info(
        "Données chargées : %d embeddings | k=%d clusters",
        len(embeddings_scaled), len(np.unique(cluster_labels)),
    )
    return embeddings_scaled, cluster_labels, metadata, scaler


# ══════════════════════════════════════════════════════════════════════════════
# 2. RÉDUCTION PCA
# ══════════════════════════════════════════════════════════════════════════════

def reduce_pca(
    embeddings_scaled: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[np.ndarray, PCA]:
    pca    = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(embeddings_scaled).astype(np.float32)
    var    = pca.explained_variance_ratio_ * 100
    logger.info(
        "PCA variance expliquée : PC1=%.1f%%, PC2=%.1f%%, total=%.1f%%",
        var[0], var[1], var.sum(),
    )
    return coords, pca


# ══════════════════════════════════════════════════════════════════════════════
# 3. RÉDUCTION t-SNE
# ══════════════════════════════════════════════════════════════════════════════

def reduce_tsne(
    embeddings_scaled: np.ndarray,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    n = len(embeddings_scaled)
    if n > 5000:
        logger.warning("t-SNE sur %d points — calcul long", n)

    tsne   = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                  random_state=random_state, verbose=0)
    t0     = time.perf_counter()
    coords = tsne.fit_transform(embeddings_scaled).astype(np.float32)
    logger.info("t-SNE calculé en %.1fs", time.perf_counter() - t0)
    return coords


# ══════════════════════════════════════════════════════════════════════════════
# 4. PALETTE DE COULEURS
# ══════════════════════════════════════════════════════════════════════════════

def get_cluster_colors(k: int) -> list:
    cmap = plt.colormaps["tab20"] if k > 10 else plt.colormaps["tab10"]
    return [cmap(i / max(k - 1, 1)) for i in range(k)]


# ══════════════════════════════════════════════════════════════════════════════
# 5. SCATTER PCA PAR CLUSTER
# ══════════════════════════════════════════════════════════════════════════════

def plot_clusters_pca(
    coords_2d: np.ndarray,
    cluster_labels: np.ndarray,
    k: int,
    metadata: pd.DataFrame,
    out_path: Path,
    cluster_names: "dict[int, str] | None" = None,
) -> None:
    colors = get_cluster_colors(k)
    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)

    for i in range(k):
        mask = cluster_labels == i
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=[colors[i]], alpha=0.5, s=15, marker=".",
        )

    ax.set_xlabel("Composante principale 1", fontsize=11)
    ax.set_ylabel("Composante principale 2", fontsize=11)
    ax.set_title(
        f"Projection PCA des embeddings CNN — clusters K-Means (k={k})\n"
        f"Module B — Familles de plaques MINT/DGI",
        fontsize=12,
    )

    patches = _build_legend_patches(k, colors, cluster_names)
    ax.legend(handles=patches, loc="upper right", fontsize=7,
              framealpha=0.85, title="Clusters / Procédures MINT/DGI",
              title_fontsize=8)

    ax.annotate(
        f"N={len(coords_2d)} embeddings | embeddings CNN 256D (§4.2)",
        xy=(0.01, 0.01), xycoords="axes fraction", fontsize=8, color="gray",
    )

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure PCA clusters sauvegardée : %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# 6. SCATTER t-SNE PAR CLUSTER
# ══════════════════════════════════════════════════════════════════════════════

def plot_clusters_tsne(
    coords_2d: np.ndarray,
    cluster_labels: np.ndarray,
    k: int,
    metadata: pd.DataFrame,
    out_path: Path,
    cluster_names: "dict[int, str] | None" = None,
) -> None:
    """
    Note : t-SNE n'a pas d'interprétation directe des axes —
    la structure globale seule est significative.
    """
    colors = get_cluster_colors(k)
    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)

    for i in range(k):
        mask = cluster_labels == i
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=[colors[i]], alpha=0.5, s=15, marker=".",
        )

    ax.set_xlabel("Dimension t-SNE 1", fontsize=11)
    ax.set_ylabel("Dimension t-SNE 2", fontsize=11)
    ax.set_title(
        f"Projection t-SNE des embeddings CNN — clusters K-Means (k={k})\n"
        f"Module B — Familles de plaques MINT/DGI",
        fontsize=12,
    )

    patches = _build_legend_patches(k, colors, cluster_names)
    ax.legend(handles=patches, loc="upper right", fontsize=7,
              framealpha=0.85, title="Clusters / Procédures MINT/DGI",
              title_fontsize=8)

    ax.annotate(
        f"N={len(coords_2d)} embeddings | embeddings CNN 256D (§4.2)",
        xy=(0.01, 0.01), xycoords="axes fraction", fontsize=8, color="gray",
    )

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure t-SNE clusters sauvegardée : %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# 7. GRILLE D'IMAGES REPRÉSENTATIVES
# ══════════════════════════════════════════════════════════════════════════════

def plot_representative_images(
    embeddings_scaled: np.ndarray,
    cluster_labels: np.ndarray,
    km_centroids: np.ndarray,
    data_dir: Path,
    metadata: pd.DataFrame,
    k: int,
    n_samples: int = 8,
    out_path: "Path | None" = None,
    cluster_names: "dict[int, str] | None" = None,
) -> None:
    import cv2

    data_dir = Path(data_dir)
    chars_dir = data_dir / "characters"

    # Pré-construction index→chemin image depuis characters/
    label_to_files: dict[str, list] = {}
    if chars_dir.exists():
        for cls_dir in sorted(chars_dir.iterdir()):
            if cls_dir.is_dir() and cls_dir.name != "unknown":
                files = sorted(f for f in cls_dir.iterdir()
                               if f.suffix.lower() in {".png", ".jpg", ".jpeg"})
                if files:
                    label_to_files[cls_dir.name] = files

    fig_h = max(k * 2.5, 4)
    fig_w = n_samples * 1.8
    fig, axes = plt.subplots(k, n_samples, figsize=(fig_w, fig_h), dpi=120)
    if k == 1:
        axes = axes[np.newaxis, :]

    for i in range(k):
        mask      = cluster_labels == i
        indices_i = np.where(mask)[0]

        # Top n_samples les plus proches du centroïde
        if len(indices_i) == 0:
            top_idx = np.array([], dtype=int)
        else:
            dists_i = np.linalg.norm(
                embeddings_scaled[indices_i] - km_centroids[i], axis=1
            )
            top_n   = min(n_samples, len(indices_i))
            top_idx = indices_i[np.argsort(dists_i)[:top_n]]

        # Label de la ligne
        info       = cluster_names.get(i, {}) if isinstance(cluster_names, dict) \
                     else (CLUSTER_PROCEDURES.get(i, {}) if cluster_names is None else {})
        nom        = info.get("label",     f"Cluster {i}") if isinstance(info, dict) \
                     else str(info)
        proc       = CLUSTER_PROCEDURES.get(i, {}).get("procedure", "—")
        row_title  = f"C{i}\n{nom}\n{proc}"

        for j, ax in enumerate(axes[i]):
            ax.axis("off")
            img_shown = False

            if j < len(top_idx):
                idx = top_idx[j]

                # Priorité 1 — colonne "filename" dans metadata
                if "filename" in metadata.columns and idx < len(metadata):
                    fpath = Path(metadata.iloc[idx]["filename"])
                    if fpath.exists():
                        img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                            img_shown = True

                # Priorité 2 — characters/{label_char}/
                if not img_shown and "label_char" in metadata.columns \
                        and idx < len(metadata):
                    lchar = str(metadata.iloc[idx]["label_char"])
                    files = label_to_files.get(lchar, [])
                    if files:
                        # On prend le fichier dont l'index dans la liste
                        # est cohérent avec la position dans le cluster
                        file_idx = j % len(files)
                        img = cv2.imread(str(files[file_idx]), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                            img_shown = True

                # Priorité 3 — placeholder
                if not img_shown:
                    placeholder = np.full((28, 28), 180, dtype=np.uint8)
                    ax.imshow(placeholder, cmap="gray", vmin=0, vmax=255)
                    ax.text(0.5, 0.5, f"#{idx}", ha="center", va="center",
                            transform=ax.transAxes, fontsize=6, color="black")

        # Titre de la ligne (à gauche)
        axes[i][0].set_ylabel(row_title, fontsize=7, rotation=0,
                               labelpad=55, va="center")

    fig.suptitle(
        "Images représentatives par cluster (proches du centroïde)\n"
        "Inspection qualitative — §4.2 PlateVision",
        fontsize=11,
    )
    plt.tight_layout(rect=[0.12, 0, 1, 1])

    if out_path is None:
        out_path = _DEFAULT_FIGURES / "clusters_representative_images.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Grille images représentatives sauvegardée : %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# 8. DISTRIBUTION DE LA CONFIANCE PAR CLUSTER
# ══════════════════════════════════════════════════════════════════════════════

def plot_confidence_distribution(
    metadata: pd.DataFrame,
    k: int,
    out_path: Path,
) -> None:
    if "confidence_level" not in metadata.columns:
        logger.warning("confidence_level absent de metadata — plot_confidence_distribution ignoré")
        return

    conf_colors = ["#2ca02c", "#ff7f0e", "#d62728"]  # vert / orange / rouge
    conf_labels = ["Confiance haute", "Confiance moyenne", "Confiance faible"]

    counts = np.zeros((k, 3), dtype=int)
    for i in range(k):
        mask = metadata["cluster_id"] == i
        for lvl in range(3):
            counts[i, lvl] = int((metadata.loc[mask, "confidence_level"] == lvl).sum())

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    xs    = np.arange(k)
    bottoms = np.zeros(k, dtype=int)

    for lvl in range(3):
        ax.bar(xs, counts[:, lvl], bottom=bottoms,
               color=conf_colors[lvl], label=conf_labels[lvl], edgecolor="white")
        bottoms += counts[:, lvl]

    ax.set_xticks(xs)
    ax.set_xticklabels([f"Cluster {i}" for i in range(k)], rotation=30, ha="right")
    ax.set_xlabel("Cluster ID", fontsize=10)
    ax.set_ylabel("Nombre de points", fontsize=10)
    ax.set_title(
        "Distribution de la confiance du clustering par groupe\n"
        "(distance au centroïde discrétisée — utile pour états MDP §4.3)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure confiance sauvegardée : %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRE INTERNE
# ══════════════════════════════════════════════════════════════════════════════

def _build_legend_patches(
    k: int,
    colors: list,
    cluster_names: "dict[int, str] | None",
) -> list:
    patches = []
    for i in range(k):
        if cluster_names and i in cluster_names:
            info = cluster_names[i]
            nom  = info.get("label", f"Cluster {i}") if isinstance(info, dict) else str(info)
            proc = info.get("procedure", "—") if isinstance(info, dict) else "—"
            lbl  = f"Cluster {i} — {nom} | {proc}"
        elif CLUSTER_PROCEDURES and i in CLUSTER_PROCEDURES:
            nom  = CLUSTER_PROCEDURES[i].get("label", f"Cluster {i}")
            proc = CLUSTER_PROCEDURES[i].get("procedure", "—")
            lbl  = f"Cluster {i} — {nom} | {proc}"
        else:
            lbl = f"Cluster {i}"
        patches.append(Patch(color=colors[i], label=lbl))
    return patches


# ══════════════════════════════════════════════════════════════════════════════
# 9. PIPELINE COMPLET
# ══════════════════════════════════════════════════════════════════════════════

def run_visualization_pipeline(
    data_dir: Path = _DEFAULT_DATA,
    models_dir: Path = _DEFAULT_MODELS,
    figures_dir: Path = _DEFAULT_FIGURES,
    run_tsne: bool = True,
    tsne_perplexity: float = 30.0,
    tsne_n_iter: int = 1000,
    n_representative: int = 8,
    random_state: int = 42,
    cluster_names: "dict[int, str] | None" = None,
) -> dict:
    data_dir    = Path(data_dir)
    models_dir  = Path(models_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Chargement
    embeddings_scaled, cluster_labels, metadata, scaler = load_data(data_dir, models_dir)
    k = int(cluster_labels.max()) + 1

    # 2. Centroïdes
    centroids_path = data_dir / "kmeans_centroids.npy"
    if centroids_path.exists():
        km_centroids = np.load(centroids_path).astype(np.float32)
    else:
        logger.warning("kmeans_centroids.npy absent — recalcul depuis kmeans_model.pkl")
        km = joblib.load(models_dir / "kmeans_model.pkl")
        km_centroids = km.cluster_centers_.astype(np.float32)

    # 3. PCA
    pca_coords, pca_model = reduce_pca(embeddings_scaled, random_state=random_state)

    # 4. Scatter PCA
    plot_clusters_pca(
        pca_coords, cluster_labels, k, metadata,
        figures_dir / "clusters_pca_final.png",
        cluster_names=cluster_names,
    )

    # 5. t-SNE (optionnel)
    tsne_coords = None
    if run_tsne:
        tsne_coords = reduce_tsne(
            embeddings_scaled,
            perplexity=tsne_perplexity,
            n_iter=tsne_n_iter,
            random_state=random_state,
        )
        plot_clusters_tsne(
            tsne_coords, cluster_labels, k, metadata,
            figures_dir / "clusters_tsne_final.png",
            cluster_names=cluster_names,
        )

    # 6. Grille images représentatives
    plot_representative_images(
        embeddings_scaled, cluster_labels, km_centroids,
        data_dir, metadata, k,
        n_samples=n_representative,
        out_path=figures_dir / "clusters_representative_images.png",
        cluster_names=cluster_names,
    )

    # 7. Distribution confiance
    plot_confidence_distribution(
        metadata, k,
        figures_dir / "clusters_confidence_dist.png",
    )

    print("\n=== Module B — Étape 5 terminée ===")
    print(f"Figures générées dans : {figures_dir}")
    print("  clusters_pca_final.png        (§4.2 — PCA colorée par cluster)")
    if run_tsne:
        print("  clusters_tsne_final.png       (t-SNE si run_tsne=True)")
    print("  clusters_representative.png   (inspection visuelle qualitative)")
    print("  clusters_confidence_dist.png  (distribution confiance → états MDP)")
    print()
    print("Ces figures sont prêtes pour le rapport LaTeX (§L3) et la soutenance (§5).")
    print("Prêt pour : modules/module_b/interpret_clusters.py")

    return {
        "pca_coords":     pca_coords,
        "tsne_coords":    tsne_coords,
        "cluster_labels": cluster_labels,
        "k":              k,
        "metadata":       metadata,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI — SOUTENANCE §5 : jury peut demander PCA ou t-SNE en direct
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Module B Étape 5 — Visualisation clusters (§4.2 PlateVision)"
    )
    parser.add_argument("--data-dir",        type=Path,  default=_DEFAULT_DATA)
    parser.add_argument("--models-dir",      type=Path,  default=_DEFAULT_MODELS)
    parser.add_argument("--figures-dir",     type=Path,  default=_DEFAULT_FIGURES)
    parser.add_argument("--no-tsne",         action="store_true",
                        help="Skip t-SNE (long si N > 5000)")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-n-iter",     type=int,   default=1000)
    parser.add_argument("--n-representative",type=int,   default=8)
    parser.add_argument("--random-state",    type=int,   default=42)
    args = parser.parse_args()

    run_visualization_pipeline(
        data_dir         = args.data_dir,
        models_dir       = args.models_dir,
        figures_dir      = args.figures_dir,
        run_tsne         = not args.no_tsne,
        tsne_perplexity  = args.tsne_perplexity,
        tsne_n_iter      = args.tsne_n_iter,
        n_representative = args.n_representative,
        random_state     = args.random_state,
    )
