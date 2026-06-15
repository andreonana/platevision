"""
Module B — Étape 4 : K-Means final + enrichissement metadata pour Module C
PlateVision / MINT-DGI Cameroun — §4.2 + §4.3 cahier des charges

La distance au centroïde discrétisée en 3 niveaux alimente le Module C §4.3 :
  États S = cluster_id × niveau_confiance × signal_alerte_CNN
  → k×3×2 états MDP (§4.3 recommande 9-16 états)
"""

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_CONFIDENCE_LABELS = {0: "haute", 1: "moyenne", 2: "faible"}

# ── Import CLUSTER_PROCEDURES depuis clustering.py ────────────────────────────
try:
    from modules.module_b.clustering import CLUSTER_PROCEDURES
except Exception as _e:
    logger.warning("Import CLUSTER_PROCEDURES échoué (%s) — mint_dgi_procedure = 'à_définir'", _e)
    CLUSTER_PROCEDURES = {}


# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT + NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def load_and_scale(
    data_dir: Path = Path("data/processed"),
) -> tuple[np.ndarray, pd.DataFrame, StandardScaler]:
    data_dir  = Path(data_dir)
    emb_path  = data_dir / "embeddings.npy"
    meta_path = data_dir / "metadata.csv"

    if not emb_path.exists():
        raise RuntimeError(
            "Embeddings absents. Exécute : python main.py --module B1"
        )
    if not meta_path.exists():
        raise RuntimeError(
            "metadata.csv absent. Exécute : python main.py --module B1"
        )

    k_txt = Path("reports/rapport_technique/k_analysis.txt")
    if not k_txt.exists():
        logger.warning(
            "Attention : k_analysis.txt absent — as-tu exécuté l'Étape 3 ?"
        )

    embeddings = np.load(emb_path).astype(np.float32)
    metadata   = pd.read_csv(meta_path)

    scaler            = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings).astype(np.float32)

    logger.info(
        "Embeddings chargés : %d × %dD — normalisés pour K-Means",
        embeddings.shape[0], embeddings.shape[1],
    )
    return embeddings_scaled, metadata, scaler


# ══════════════════════════════════════════════════════════════════════════════
# 2. ENTRAÎNEMENT K-MEANS FINAL
# ══════════════════════════════════════════════════════════════════════════════

def fit_kmeans(
    embeddings_scaled: np.ndarray,
    k: int,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> KMeans:
    # n_init=10 : lance 10 initialisations aléatoires et garde la meilleure (inertie minimale)
    km = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    km.fit(embeddings_scaled)
    logger.info("KMeans entraîné : k=%d clusters", k)
    logger.info("Inertie finale  : %.4f", km.inertia_)
    logger.info("Itérations      : %d", km.n_iter_)
    return km


# ══════════════════════════════════════════════════════════════════════════════
# 3. DISTANCES AUX CENTROÏDES
# ══════════════════════════════════════════════════════════════════════════════

def compute_distances_to_centroids(
    embeddings_scaled: np.ndarray,
    km: KMeans,
) -> np.ndarray:
    """Vectorisé : pas de boucle Python."""
    dists = np.linalg.norm(
        embeddings_scaled - km.cluster_centers_[km.labels_], axis=1
    ).astype(np.float32)
    return dists


# ══════════════════════════════════════════════════════════════════════════════
# 4. DISCRÉTISATION EN NIVEAUX DE CONFIANCE
# ══════════════════════════════════════════════════════════════════════════════

def discretize_distance(
    distances: np.ndarray,
    n_levels: int = 3,
) -> np.ndarray:
    """
    Terciles → 3 niveaux de confiance pour les états MDP du Module C §4.3.
      0 = haute   (distance ≤ q33 — proche centroïde)
      1 = moyenne  (q33 < dist ≤ q66)
      2 = faible   (distance > q66 — cas ambigu)
    """
    # Terciles empiriques sur dist_centroid réelles — les seuils s'adaptent à chaque dataset
    q33 = float(np.percentile(distances, 33))
    q66 = float(np.percentile(distances, 66))

    # 0=haute (≤q33, proche centroïde), 1=moyenne, 2=faible (>q66, cas ambigu pour le MDP)
    levels = np.where(distances <= q33, 0,
             np.where(distances <= q66, 1, 2)).astype(np.int32)

    n0 = int((levels == 0).sum())
    n1 = int((levels == 1).sum())
    n2 = int((levels == 2).sum())

    logger.info("Seuils distance→confiance : q33=%.4f, q66=%.4f", q33, q66)
    logger.info(
        "Distribution : haute=%d | moyenne=%d | faible=%d", n0, n1, n2
    )
    return levels


# ══════════════════════════════════════════════════════════════════════════════
# 5. ENRICHISSEMENT METADATA
# ══════════════════════════════════════════════════════════════════════════════

def enrich_metadata(
    metadata: pd.DataFrame,
    km: KMeans,
    distances: np.ndarray,
    confidence_levels: np.ndarray,
) -> pd.DataFrame:
    metadata = metadata.copy()
    metadata["cluster_id"]        = km.labels_.astype(np.int32)
    metadata["dist_centroid"]     = distances.astype(np.float32)
    metadata["confidence_level"]  = confidence_levels.astype(np.int32)
    metadata["confidence_label"]  = [_CONFIDENCE_LABELS[l] for l in confidence_levels]
    metadata["mint_dgi_procedure"] = [
        CLUSTER_PROCEDURES.get(int(c), {}).get("procedure", "à_définir")
        for c in km.labels_
    ]

    for i in range(km.n_clusters):
        mask = km.labels_ == i
        n    = int(mask.sum())
        d    = float(distances[mask].mean())
        n0   = int((confidence_levels[mask] == 0).sum())
        proc = CLUSTER_PROCEDURES.get(i, {}).get("procedure", "à_définir")
        logger.info(
            "Cluster %d: %d images | dist_moy=%.4f | confiance_haute=%d | procédure=%s",
            i, n, d, n0, proc,
        )
    return metadata


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAUVEGARDE
# ══════════════════════════════════════════════════════════════════════════════

def save_results(
    metadata: pd.DataFrame,
    km: KMeans,
    scaler: StandardScaler,
    data_dir: Path = Path("data/processed"),
    models_dir: Path = Path("models"),
) -> None:
    data_dir   = Path(data_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    meta_path      = data_dir / "metadata.csv"
    km_path        = models_dir / "kmeans_model.pkl"
    scaler_path    = models_dir / "kmeans_scaler.pkl"
    centroids_path = data_dir / "kmeans_centroids.npy"

    metadata.to_csv(meta_path, index=False, encoding="utf-8")
    joblib.dump(km,     km_path)
    joblib.dump(scaler, scaler_path)
    np.save(centroids_path, km.cluster_centers_)

    logger.info("metadata.csv         → %s", meta_path)
    logger.info("kmeans_model.pkl     → %s", km_path)
    logger.info("kmeans_scaler.pkl    → %s", scaler_path)
    logger.info("kmeans_centroids.npy → %s  shape=%s",
                centroids_path, km.cluster_centers_.shape)


# ══════════════════════════════════════════════════════════════════════════════
# 7. ÉVALUATION DU CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_clustering(
    embeddings_scaled: np.ndarray,
    km: KMeans,
    figures_dir: Path,
) -> dict:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    labels = km.labels_
    k      = km.n_clusters

    sil = silhouette_score(embeddings_scaled, labels)
    db  = davies_bouldin_score(embeddings_scaled, labels)
    ch  = calinski_harabasz_score(embeddings_scaled, labels)

    logger.info("Métriques clustering final :")
    logger.info("  Silhouette        : %.4f", sil)
    logger.info("  Davies-Bouldin    : %.4f (↓ mieux)", db)
    logger.info("  Calinski-Harabasz : %.1f (↑ mieux)", ch)

    # ── Figure centroïdes superposés aux embeddings en PCA 2D ─────────────
    pca      = PCA(n_components=2, random_state=42)
    all_pts  = pca.fit_transform(embeddings_scaled)
    centroids_2d = pca.transform(km.cluster_centers_)

    cmap = plt.colormaps["tab10"]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Scatter de fond — tous les embeddings en gris
    ax.scatter(all_pts[:, 0], all_pts[:, 1],
               c="lightgray", alpha=0.1, s=6, marker=".", zorder=1)

    # Centroïdes
    for i, (cx, cy) in enumerate(centroids_2d):
        ax.scatter(cx, cy, marker="*", s=200, color=cmap(i % 10),
                   edgecolors="black", linewidths=0.5, zorder=3)
        label_short = CLUSTER_PROCEDURES.get(i, {}).get("label", f"C{i}")
        ax.annotate(
            f"C{i} — {label_short}",
            xy=(cx, cy), xytext=(cx + 0.3, cy + 0.3),
            fontsize=8, color=cmap(i % 10),
            arrowprops=dict(arrowstyle="-", color=cmap(i % 10), lw=0.8),
        )

    ax.set_title(f"Centroïdes K-Means — k={k} (espace PCA 2D)", fontsize=12)
    ax.set_xlabel("PCA composante 1", fontsize=10)
    ax.set_ylabel("PCA composante 2", fontsize=10)
    plt.tight_layout()
    fig.savefig(figures_dir / "centroids_pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure sauvegardée : %s", figures_dir / "centroids_pca.png")

    return {
        "silhouette":         float(sil),
        "davies_bouldin":     float(db),
        "calinski_harabasz":  float(ch),
        "k":                  k,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8. PIPELINE COMPLET
# ══════════════════════════════════════════════════════════════════════════════

def _read_k_from_analysis() -> int | None:
    """Tente de lire k_recommended depuis k_analysis.txt (ligne 'k_recommended=N')."""
    txt_path = Path("reports/rapport_technique/k_analysis.txt")
    if not txt_path.exists():
        return None
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("k_recommended="):
            try:
                return int(line.split("=", 1)[1])
            except ValueError:
                pass
    return None


def _read_k_from_model() -> int | None:
    """Lit k depuis un kmeans_model.pkl existant."""
    pkl = Path("models/kmeans_model.pkl")
    if not pkl.exists():
        return None
    try:
        km = joblib.load(pkl)
        return int(km.n_clusters)
    except Exception:
        return None


def run_kmeans_pipeline(
    data_dir: Path = Path("data/processed"),
    models_dir: Path = Path("models"),
    figures_dir: Path = Path("reports/rapport_technique/figures"),
    k: "int | None" = None,
    random_state: int = 42,
    n_init: int = 10,
) -> dict:
    data_dir    = Path(data_dir)
    models_dir  = Path(models_dir)
    figures_dir = Path(figures_dir)

    # ── Résolution de k ───────────────────────────────────────────────────────
    if k is None:
        k = _read_k_from_analysis()
    if k is None:
        k = _read_k_from_model()
    if k is None:
        raise ValueError(
            "k non spécifié. Exécute d'abord l'Étape 3 ou passe --k explicitement."
        )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    embeddings_scaled, metadata, scaler = load_and_scale(data_dir)
    km          = fit_kmeans(embeddings_scaled, k, random_state=random_state, n_init=n_init)
    distances   = compute_distances_to_centroids(embeddings_scaled, km)
    conf_levels = discretize_distance(distances)
    metadata    = enrich_metadata(metadata, km, distances, conf_levels)
    save_results(metadata, km, scaler, data_dir, models_dir)
    metrics     = evaluate_clustering(embeddings_scaled, km, figures_dir)

    print("\n=== Module B — Étape 4 terminée ===")
    print(f"K-Means final : k={k} clusters")
    print(f"Inertie       : {km.inertia_:.4f}")
    print(f"Silhouette    : {metrics['silhouette']:.4f}")
    print("metadata.csv enrichi : cluster_id + dist_centroid + confidence_level")
    print(f"Modèle sauvegardé    : {models_dir}/kmeans_model.pkl")
    print(f"Centroïdes           : {data_dir}/kmeans_centroids.npy")
    print()
    print("→ Colonnes ajoutées à metadata.csv pour Module C (§4.3) :")
    print(f"  cluster_id × confidence_level = {k}×3 = {k*3} états possibles")
    print(f"  (avec signal alerte : {k*3*2} états MDP total — §4.3 recommande 9-16)")
    print()
    print("Prêt pour : modules/module_b/interpret_clusters.py")

    return {
        "k":                  k,
        "km":                 km,
        "scaler":             scaler,
        "embeddings_scaled":  embeddings_scaled,
        "distances":          distances,
        "confidence_levels":  conf_levels,
        "metadata":           metadata,
        "metrics":            metrics,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Module B Étape 4 — K-Means final (§4.2 PlateVision)"
    )
    parser.add_argument("--data-dir",     type=Path, default=Path("data/processed"))
    parser.add_argument("--models-dir",   type=Path, default=Path("models"))
    parser.add_argument("--figures-dir",  type=Path,
                        default=Path("reports/rapport_technique/figures"))
    parser.add_argument("--k",            type=int,  default=None,
                        help="Nombre de clusters. Si absent : lu depuis k_analysis.txt")
    parser.add_argument("--random-state", type=int,  default=42)
    parser.add_argument("--n-init",       type=int,  default=10)
    args = parser.parse_args()

    run_kmeans_pipeline(
        data_dir     = args.data_dir,
        models_dir   = args.models_dir,
        figures_dir  = args.figures_dir,
        k            = args.k,
        random_state = args.random_state,
        n_init       = args.n_init,
    )
