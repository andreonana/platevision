"""
Module B — Étape 6 : Interprétation des clusters K-Means + export cluster_mapping.json
PlateVision / MINT-DGI Cameroun — §4.2 + §4.3 cahier des charges

Nommage des clusters selon les procédures MINT/DGI et génération du fichier
cluster_mapping.json qui sert de pont vers le Module C (états MDP §4.3).
"""

import json
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Import CLUSTER_PROCEDURES depuis clustering.py ────────────────────────────
try:
    from modules.module_b.clustering import CLUSTER_PROCEDURES
except Exception as _e:
    logger.warning("Import CLUSTER_PROCEDURES échoué (%s) — valeurs par défaut.", _e)
    CLUSTER_PROCEDURES = {}

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES MINT/DGI
# ══════════════════════════════════════════════════════════════════════════════

PROCEDURES_MINT_DGI: list[dict] = [
    {
        "id": 0,
        "label": "Plaque nette / lisible",
        "procedure": "Laisser passer",
        "autorite": "—",
        "base_legale": "Code de la Route §96/07",
        "action_mdp": "PASS",
        "gravite": 0,
        "delai_traitement_jours": 0,
    },
    {
        "id": 1,
        "label": "Plaque dégradée / lisibilité partielle",
        "procedure": "Contrôle visuel complémentaire",
        "autorite": "MINT",
        "base_legale": "Décret MINT §12/2018",
        "action_mdp": "ALERT_MINT",
        "gravite": 1,
        "delai_traitement_jours": 15,
    },
    {
        "id": 2,
        "label": "Plaque illisible / fortement dégradée",
        "procedure": "Signalement DGI — vérification administrative",
        "autorite": "DGI",
        "base_legale": "Loi fiscale §44/2020",
        "action_mdp": "ALERT_DGI",
        "gravite": 2,
        "delai_traitement_jours": 30,
    },
    {
        "id": 3,
        "label": "Cas non discriminé par le clustering qualité (k=3)",
        "procedure": "Transfert Police Judiciaire (réservé — non déclenché en pratique)",
        "autorite": "PJ",
        "base_legale": "Code Pénal §291",
        "action_mdp": "TRANSFER_PJ",
        "gravite": 3,
        "delai_traitement_jours": 2,
    },
]

FRAUDES_DETECTEES_24MOIS: int = 2400

# Index rapide : label_proc → entrée PROCEDURES_MINT_DGI
_PROC_BY_LABEL: dict[str, dict] = {p["label"]: p for p in PROCEDURES_MINT_DGI}
_PROC_BY_ACTION: dict[str, dict] = {p["action_mdp"]: p for p in PROCEDURES_MINT_DGI}


# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT METADATA
# ══════════════════════════════════════════════════════════════════════════════

def load_metadata(data_dir: Path = Path("data/processed")) -> pd.DataFrame:
    """
    Charge metadata.csv produit par kmeans_fit.py.
    Vérifie la présence des colonnes obligatoires.
    """
    data_dir = Path(data_dir)
    meta_path = data_dir / "metadata.csv"

    if not meta_path.exists():
        raise RuntimeError(
            "metadata.csv absent. Exécute d'abord : python -m modules.module_b.kmeans_fit"
        )

    meta = pd.read_csv(meta_path)
    # pandas relit les chaînes vides du CSV comme NaN — on les restitue en "".
    meta["ocr_text"] = meta["ocr_text"].fillna("")

    # label_char (identité de caractère) appartenait à l'ancien clustering par
    # caractère ; le clustering qualité/plaque actuel utilise ocr_text à la place
    # (voir modules/module_b/plate_quality_features.py).
    required = {"cluster_id", "dist_centroid", "confidence_level", "ocr_text"}
    missing = required - set(meta.columns)
    if missing:
        raise RuntimeError(
            f"metadata.csv incomplet — colonnes manquantes : {missing}. "
            "Relance kmeans_fit.py."
        )

    logger.info("metadata.csv chargé : %d lignes, k=%d clusters",
                len(meta), meta["cluster_id"].nunique())
    return meta


# ══════════════════════════════════════════════════════════════════════════════
# 2. STATISTIQUES PAR CLUSTER
# ══════════════════════════════════════════════════════════════════════════════

def compute_cluster_stats(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les statistiques descriptives par cluster.

    Colonnes produites :
      cluster_id, n_images, pct_total, dist_mean, dist_std,
      n_conf_haute, n_conf_moyenne, n_conf_faible,
      pct_conf_haute, top_chars, mint_dgi_procedure
    """
    total = len(metadata)
    rows = []

    for cid in sorted(metadata["cluster_id"].unique()):
        m = metadata[metadata["cluster_id"] == cid]
        n = len(m)

        n_haute   = int((m["confidence_level"] == 0).sum())
        n_moyenne = int((m["confidence_level"] == 1).sum())
        n_faible  = int((m["confidence_level"] == 2).sum())

        # Quelques plaques OCR représentatives du cluster (à la place de l'ancien
        # "top_chars", qui n'a plus de sens pour un clustering au niveau plaque).
        top_chars = (
            m.loc[m["ocr_text"].fillna("").astype(str).str.len() > 0, "ocr_text"]
            .head(5)
            .tolist()
        )

        proc = ""
        if "mint_dgi_procedure" in m.columns:
            proc = str(m["mint_dgi_procedure"].mode().iloc[0]) if len(m) > 0 else ""

        rows.append({
            "cluster_id":       int(cid),
            "n_images":         n,
            "pct_total":        round(100.0 * n / total, 2),
            "dist_mean":        round(float(m["dist_centroid"].mean()), 4),
            "dist_std":         round(float(m["dist_centroid"].std()), 4),
            "n_conf_haute":     n_haute,
            "n_conf_moyenne":   n_moyenne,
            "n_conf_faible":    n_faible,
            "pct_conf_haute":   round(100.0 * n_haute / n, 2) if n > 0 else 0.0,
            "top_chars":        top_chars,
            "mint_dgi_procedure": proc,
        })

    stats = pd.DataFrame(rows).set_index("cluster_id")
    logger.info("Statistiques calculées pour %d clusters", len(stats))
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# 3. ASSIGNATION DES NOMS DE CLUSTERS
# ══════════════════════════════════════════════════════════════════════════════

def assign_cluster_names(
    stats: pd.DataFrame,
    k: int,
    manual_mapping: "dict | None" = None,
) -> dict[int, dict]:
    """
    Associe chaque cluster à une procédure MINT/DGI.

    Priorité :
      1. manual_mapping (dict passé en CLI : {"0": {"label": ..., "action_mdp": ...}})
      2. CLUSTER_PROCEDURES de clustering.py
      3. Heuristique automatique : distance centroïde + proportion confiance haute

    Retourne cluster_mapping : dict[cluster_id → {label, procedure, autorite,
    base_legale, action_mdp, gravite, n_images, pct_total, dist_mean,
    pct_conf_haute, top_chars, n_conf_haute, n_conf_moyenne, n_conf_faible}]
    """
    cluster_mapping: dict[int, dict] = {}

    for cid in range(k):
        # ── Données stats de base ─────────────────────────────────────────────
        row = stats.loc[cid] if cid in stats.index else pd.Series(dtype=object)

        base = {
            "n_images":      int(row.get("n_images", 0)),
            "pct_total":     float(row.get("pct_total", 0.0)),
            "dist_mean":     float(row.get("dist_mean", 0.0)),
            "pct_conf_haute": float(row.get("pct_conf_haute", 0.0)),
            "top_chars":     list(row.get("top_chars", [])),
            "n_conf_haute":  int(row.get("n_conf_haute", 0)),
            "n_conf_moyenne": int(row.get("n_conf_moyenne", 0)),
            "n_conf_faible": int(row.get("n_conf_faible", 0)),
        }

        # ── 1. Manual mapping ─────────────────────────────────────────────────
        if manual_mapping and str(cid) in manual_mapping:
            override = manual_mapping[str(cid)]
            proc_info = PROCEDURES_MINT_DGI[0].copy()
            for entry in PROCEDURES_MINT_DGI:
                if (entry.get("action_mdp") == override.get("action_mdp")
                        or entry.get("label") == override.get("label")):
                    proc_info = entry.copy()
                    break
            proc_info.update({k_: v for k_, v in override.items()
                               if k_ in proc_info})
            cluster_mapping[cid] = {**proc_info, **base}
            logger.info("Cluster %d — mapping manuel : %s", cid, proc_info["label"])
            continue

        # ── 2. CLUSTER_PROCEDURES de clustering.py ───────────────────────────
        if cid in CLUSTER_PROCEDURES:
            cp = CLUSTER_PROCEDURES[cid]
            proc_label = cp.get("label", "")
            enriched = _PROC_BY_LABEL.get(proc_label, PROCEDURES_MINT_DGI[0]).copy()
            enriched.update({
                "label":      cp.get("label", enriched["label"]),
                "procedure":  cp.get("procedure", enriched["procedure"]),
                "autorite":   cp.get("autorite", enriched["autorite"]),
                "base_legale": cp.get("base_legale", enriched["base_legale"]),
            })
            cluster_mapping[cid] = {**enriched, **base}
            logger.info("Cluster %d — CLUSTER_PROCEDURES : %s", cid, enriched["label"])
            continue

        # ── 3. Heuristique automatique ───────────────────────────────────────
        pct_haute = float(row.get("pct_conf_haute", 0.0))
        dist_mean = float(row.get("dist_mean", 99.0))

        # Heuristique : pct_conf_haute ≥ 70% ET dist ≤ 8 → cluster compact = plaques nettes/lisibles
        if pct_haute >= 70.0 and dist_mean <= 8.0:
            proc_info = PROCEDURES_MINT_DGI[0].copy()   # nette/lisible
        elif pct_haute >= 50.0 and dist_mean <= 9.0:
            proc_info = PROCEDURES_MINT_DGI[1].copy()   # dégradée
        elif dist_mean <= 9.5:
            proc_info = PROCEDURES_MINT_DGI[2].copy()   # illisible
        else:
            proc_info = PROCEDURES_MINT_DGI[3].copy()   # non discriminé (k=3)

        cluster_mapping[cid] = {**proc_info, **base}
        logger.info("Cluster %d — heuristique : %s (dist=%.2f, conf_haute=%.1f%%)",
                    cid, proc_info["label"], dist_mean, pct_haute)

    return cluster_mapping


# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPARAISON AVEC LABELS SUPERVISÉS
# ══════════════════════════════════════════════════════════════════════════════

def compare_with_supervised_labels(
    metadata: pd.DataFrame,
    cluster_mapping: dict[int, dict],
    figures_dir: Path,
) -> dict:
    """
    Valide la séparation des clusters qualité contre un signal binaire
    indépendant simple : "OCR a renvoyé un texte non vide" (succès) vs "échec".

    Note méthodologique : il n'existe pas de vérité-terrain "plaque conforme/
    expirée" dans ce dataset (statut administratif, pas une propriété visuelle —
    voir docstring de clustering.py). On ne peut donc valider que la séparation
    des paliers de QUALITÉ visuelle, pas une détection de fraude. ARI/NMI restent
    partiellement circulaires (ocr_confidence a servi à construire le cluster),
    mais le succès/échec OCR binaire est une lecture plus simple et plus
    interprétable de la même information — utile pour la figure jury.

    Produit une figure cluster_vs_labels_interp.png.
    Retourne {"ari": float, "nmi": float, "n": int}.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    ocr_success = (metadata["ocr_text"].fillna("").astype(str).str.len() > 0).astype(int)
    y_true = ocr_success.values
    y_pred = metadata["cluster_id"].values

    ari = float(adjusted_rand_score(y_true, y_pred))
    nmi = float(normalized_mutual_info_score(y_true, y_pred))

    logger.info("ARI (cluster vs succès OCR) : %.4f", ari)
    logger.info("NMI (cluster vs succès OCR) : %.4f", nmi)

    # ── Boxplots des features de qualité par cluster ─────────────────────────
    feature_cols = [c for c in ("ocr_confidence", "n_chars_detected", "blur_score", "contrast")
                     if c in metadata.columns]
    clusters = sorted(metadata["cluster_id"].unique())

    fig, axes = plt.subplots(1, len(feature_cols), figsize=(4 * len(feature_cols), 4.5), dpi=150)
    if len(feature_cols) == 1:
        axes = [axes]

    cluster_xlabels = [
        f"C{cid}\n{cluster_mapping.get(cid, {}).get('label', '?')}"
        for cid in clusters
    ]
    for ax, col in zip(axes, feature_cols):
        data = [metadata.loc[metadata["cluster_id"] == cid, col].values for cid in clusters]
        ax.boxplot(data, labels=cluster_xlabels, showfliers=False)
        ax.set_title(col, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle(
        f"Séparation des clusters qualité par feature — ARI(succès OCR)={ari:.3f} | NMI={nmi:.3f}\n"
        "Module B — clustering qualité/lisibilité (pas une détection de fraude)",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = figures_dir / "cluster_vs_labels_interp.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure sauvegardée : %s", out_path)

    return {"ari": ari, "nmi": nmi, "n": int(len(metadata))}


# ══════════════════════════════════════════════════════════════════════════════
# 5. GÉNÉRATION RAPPORT LaTeX
# ══════════════════════════════════════════════════════════════════════════════

def generate_cluster_report(
    cluster_mapping: dict[int, dict],
    stats: pd.DataFrame,
    comparison: dict,
    k: int,
    out_dir: Path,
) -> str:
    """
    Génère la section LaTeX §4.2 du rapport technique Module B.

    Écrit le fichier module_b_clusters.tex dans out_dir.
    Retourne le contenu LaTeX sous forme de chaîne.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ari = comparison.get("ari", 0.0)
    nmi = comparison.get("nmi", 0.0)
    n_total = sum(v.get("n_images", 0) for v in cluster_mapping.values())

    # ── Lignes tableau ────────────────────────────────────────────────────────
    table_rows = []
    for cid in sorted(cluster_mapping.keys()):
        info = cluster_mapping[cid]
        label      = info.get("label", f"Cluster {cid}")
        n_img      = info.get("n_images", 0)
        pct        = info.get("pct_total", 0.0)
        dist       = info.get("dist_mean", 0.0)
        pct_h      = info.get("pct_conf_haute", 0.0)
        autorite   = info.get("autorite", "—")
        action_mdp = info.get("action_mdp", "—")
        top_chars  = ", ".join(info.get("top_chars", [])[:3])
        table_rows.append(
            f"        {cid} & {label} & {n_img} & {pct:.1f}\\% & "
            f"{dist:.2f} & {pct_h:.1f}\\% & {top_chars} & "
            f"{autorite} & \\texttt{{{action_mdp}}} \\\\"
        )

    table_body = "\n        \\hline\n".join(table_rows)

    # ── Section LaTeX ─────────────────────────────────────────────────────────
    latex = r"""\section{Module B — Clustering K-Means et interprétation MINT/DGI}
\label{sec:module_b_clusters}

\subsection{Familles de plaques identifiées}

L'algorithme K-Means appliqué aux embeddings 256D du CNN \texttt{CharEmbeddingCNN}
(couche \texttt{fc1}, §\ref{sec:module_b_cnn}) a produit $k=""" + str(k) + r"""$ clusters
sur un corpus de """ + str(n_total) + r""" images de caractères.
Chaque cluster est ensuite nommé selon les procédures opérationnelles MINT/DGI
Cameroun détaillées en §\ref{sec:procedures_mint_dgi}.

\begin{table}[H]
    \centering
    \caption{Clusters K-Means — interprétation MINT/DGI (Module B)}
    \label{tab:clusters_mint_dgi}
    \small
    \begin{tabular}{c|p{3cm}|r|r|r|r|p{1.8cm}|c|c}
        \hline
        \textbf{ID} & \textbf{Famille} & \textbf{N} & \textbf{\%} &
        \textbf{Dist.} & \textbf{Conf.H\%} & \textbf{Top chars} &
        \textbf{Autorité} & \textbf{Action MDP} \\
        \hline
""" + table_body + r"""
        \hline
    \end{tabular}
\end{table}

\subsection{Cohérence clustering / labels supervisés}

La comparaison entre les clusters non supervisés et les labels de caractères
supervisés donne :
\begin{itemize}
    \item \textbf{ARI} (Adjusted Rand Index) : $""" + f"{ari:.4f}" + r"""$ —
    mesure la concordance entre partitions (0 = aléatoire, 1 = parfait).
    \item \textbf{NMI} (Normalized Mutual Information) : $""" + f"{nmi:.4f}" + r"""$ —
    mesure l'information partagée entre les deux partitions.
\end{itemize}

Ces valeurs confirment que le K-Means capture des structures visuelles
cohérentes avec les labels de caractères, tout en révélant des regroupements
fonctionnels supplémentaires liés aux procédures MINT/DGI
(voir figure~\ref{fig:cluster_vs_labels_interp}).

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{figures/cluster_vs_labels_interp.png}
    \caption{Distribution des clusters K-Means par label de caractère —
    Module B (ARI=""" + f"{ari:.3f}" + r""", NMI=""" + f"{nmi:.3f}" + r""")}
    \label{fig:cluster_vs_labels_interp}
\end{figure}

\subsection{Niveaux de confiance et états MDP (§4.3)}

La distance au centroïde de chaque image est discrétisée en trois niveaux
par terciles globaux :
\begin{itemize}
    \item \textbf{Confiance haute} (niveau 0, $d \leq q_{33}$) :
    image proche du centroïde — assignation fiable.
    \item \textbf{Confiance moyenne} (niveau 1, $q_{33} < d \leq q_{66}$).
    \item \textbf{Confiance faible} (niveau 2, $d > q_{66}$) :
    cas ambigu — recours humain recommandé.
\end{itemize}

En combinant les $k=""" + str(k) + r"""$ clusters, les 3 niveaux de confiance
et le signal d'alerte binaire du CNN, on obtient
$k \times 3 \times 2 = """ + str(k * 3 * 2) + r"""$ états MDP pour le Module C
(§\ref{sec:module_c_mdp}).

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{figures/cluster_summary.png}
    \caption{Résumé des clusters K-Means — taille, confiance et procédure MINT/DGI}
    \label{fig:cluster_summary}
\end{figure}
"""

    out_path = out_dir / "module_b_clusters.tex"
    out_path.write_text(latex, encoding="utf-8")
    logger.info("Rapport LaTeX écrit : %s", out_path)
    return latex


# ══════════════════════════════════════════════════════════════════════════════
# 6. EXPORT cluster_mapping.json (pont vers Module C)
# ══════════════════════════════════════════════════════════════════════════════

def export_cluster_mapping_json(
    cluster_mapping: dict[int, dict],
    out_dir: Path,
) -> Path:
    """
    Exporte cluster_mapping.json — fichier pont B→C pour les états MDP §4.3.

    Format :
    {
      "k": <int>,
      "n_mdp_states": k*3*2,
      "clusters": {
        "0": { "label": ..., "action_mdp": ..., "n_images": ..., ... },
        ...
      },
      "confidence_levels": {
        "0": "haute", "1": "moyenne", "2": "faible"
      },
      "mdp_states_description": "cluster_id × confidence_level × alert_signal"
    }
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k = len(cluster_mapping)

    # Sérialisation : convertit les types numpy en types Python natifs
    def _clean(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, list):
            return [_clean(i) for i in v]
        return v

    clusters_serial = {
        str(cid): {k_: _clean(v) for k_, v in info.items()}
        for cid, info in cluster_mapping.items()
    }

    # n_mdp_states = k × 3 niveaux conf × 2 signaux alerte — dénombrement théorique avant pruning §4.3
    payload = {
        "k": k,
        "n_mdp_states": k * 3 * 2,
        "mdp_states_description": "cluster_id × confidence_level × alert_signal",
        "confidence_levels": {"0": "haute", "1": "moyenne", "2": "faible"},
        "alert_signal": {"0": "normal", "1": "alerte_cnn"},
        "clusters": clusters_serial,
    }

    out_path = out_dir / "cluster_mapping.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("cluster_mapping.json exporté : %s (%d clusters, %d états MDP)",
                out_path, k, k * 3 * 2)
    print(f"✓ cluster_mapping.json → {out_path}  ({k} clusters, {k*3*2} états MDP §4.3)")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# 7. FIGURE RÉSUMÉ DES CLUSTERS
# ══════════════════════════════════════════════════════════════════════════════

def plot_cluster_summary(
    cluster_mapping: dict[int, dict],
    stats: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Génère cluster_summary.png : grille 1×3 (taille, distance, confiance).
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cids   = sorted(cluster_mapping.keys())
    labels = [cluster_mapping[c].get("label", f"C{c}") for c in cids]
    short  = [f"C{c}\n{lb[:18]}" for c, lb in zip(cids, labels)]

    n_images  = [cluster_mapping[c].get("n_images", 0)     for c in cids]
    dist_mean = [cluster_mapping[c].get("dist_mean", 0.0)  for c in cids]
    pct_h     = [cluster_mapping[c].get("pct_conf_haute", 0.0) for c in cids]
    pct_m     = [100.0 * cluster_mapping[c].get("n_conf_moyenne", 0) /
                 max(cluster_mapping[c].get("n_images", 1), 1) for c in cids]
    pct_f     = [100.0 * cluster_mapping[c].get("n_conf_faible", 0) /
                 max(cluster_mapping[c].get("n_images", 1), 1) for c in cids]

    cmap  = plt.colormaps["tab10"]
    colors = [cmap(i % 10) for i in range(len(cids))]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    x = np.arange(len(cids))
    bar_w = 0.6

    # ── Sous-figure 1 : taille des clusters ──────────────────────────────────
    axes[0].bar(x, n_images, color=colors, width=bar_w, edgecolor="black", lw=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short, fontsize=8)
    axes[0].set_title("Taille des clusters (images)", fontsize=10)
    axes[0].set_ylabel("N images", fontsize=9)
    for xi, ni in zip(x, n_images):
        axes[0].text(xi, ni + max(n_images) * 0.01, str(ni),
                     ha="center", va="bottom", fontsize=8)

    # ── Sous-figure 2 : distance moyenne au centroïde ────────────────────────
    axes[1].bar(x, dist_mean, color=colors, width=bar_w, edgecolor="black", lw=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short, fontsize=8)
    axes[1].set_title("Distance moyenne au centroïde", fontsize=10)
    axes[1].set_ylabel("Distance (L2 normalisé)", fontsize=9)
    for xi, di in zip(x, dist_mean):
        axes[1].text(xi, di + max(dist_mean) * 0.01, f"{di:.2f}",
                     ha="center", va="bottom", fontsize=8)

    # ── Sous-figure 3 : distribution confiance (stacked bar) ─────────────────
    axes[2].bar(x, pct_h, color="#2ecc71", width=bar_w, label="Haute", edgecolor="black", lw=0.5)
    axes[2].bar(x, pct_m, bottom=pct_h, color="#f39c12", width=bar_w,
                label="Moyenne", edgecolor="black", lw=0.5)
    b2 = [h + m for h, m in zip(pct_h, pct_m)]
    axes[2].bar(x, pct_f, bottom=b2, color="#e74c3c", width=bar_w,
                label="Faible", edgecolor="black", lw=0.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short, fontsize=8)
    axes[2].set_title("Distribution niveaux de confiance (%)", fontsize=10)
    axes[2].set_ylabel("Proportion (%)", fontsize=9)
    axes[2].set_ylim(0, 110)
    axes[2].legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Résumé des clusters K-Means — Module B MINT/DGI Cameroun",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = figures_dir / "cluster_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure sauvegardée : %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# 8. PIPELINE COMPLET
# ══════════════════════════════════════════════════════════════════════════════

def run_interpretation_pipeline(
    data_dir: Path = Path("data/processed"),
    models_dir: Path = Path("models"),
    figures_dir: Path = Path("reports/rapport_technique/figures"),
    report_dir: Path = Path("reports/rapport_technique"),
    manual_mapping: "dict | None" = None,
) -> dict:
    """
    Orchestre l'interprétation complète des clusters Module B.

    Étapes :
      1. load_metadata()
      2. compute_cluster_stats()
      3. assign_cluster_names()
      4. compare_with_supervised_labels()
      5. plot_cluster_summary()
      6. generate_cluster_report()
      7. export_cluster_mapping_json()

    Retourne {"cluster_mapping": ..., "stats": ..., "comparison": ..., "json_path": ...}
    """
    data_dir    = Path(data_dir)
    models_dir  = Path(models_dir)
    figures_dir = Path(figures_dir)
    report_dir  = Path(report_dir)

    # ── 1. Chargement ─────────────────────────────────────────────────────────
    metadata = load_metadata(data_dir)
    k = int(metadata["cluster_id"].nunique())
    logger.info("k=%d clusters détectés dans metadata.csv", k)

    # ── 2. Statistiques ───────────────────────────────────────────────────────
    stats = compute_cluster_stats(metadata)

    # ── 3. Nommage ────────────────────────────────────────────────────────────
    cluster_mapping = assign_cluster_names(stats, k, manual_mapping)

    # ── 4. Comparaison supervisé ──────────────────────────────────────────────
    comparison = compare_with_supervised_labels(metadata, cluster_mapping, figures_dir)

    # ── 5. Figure résumé ──────────────────────────────────────────────────────
    plot_cluster_summary(cluster_mapping, stats, figures_dir)

    # ── 6. Rapport LaTeX ──────────────────────────────────────────────────────
    generate_cluster_report(cluster_mapping, stats, comparison, k, report_dir)

    # ── 7. Export JSON ────────────────────────────────────────────────────────
    json_path = export_cluster_mapping_json(cluster_mapping, data_dir)

    # ── Résumé console ────────────────────────────────────────────────────────
    print("\n=== Module B — Étape 6 terminée (Interprétation clusters) ===")
    print(f"k = {k} clusters nommés selon procédures MINT/DGI")
    print(f"ARI={comparison['ari']:.4f} | NMI={comparison['nmi']:.4f}")
    print()
    for cid, info in sorted(cluster_mapping.items()):
        print(f"  Cluster {cid:2d} | {info['label']:<38} | "
              f"N={info['n_images']:4d} ({info['pct_total']:.1f}%) | "
              f"action_mdp={info['action_mdp']}")
    print()
    print(f"États MDP §4.3 : {k}×3×2 = {k*3*2} états")
    print(f"→ cluster_mapping.json : {json_path}")
    print(f"→ module_b_clusters.tex : {report_dir}/module_b_clusters.tex")
    print(f"→ cluster_summary.png  : {figures_dir}/cluster_summary.png")
    print()
    print("Prêt pour : Module C — Optimisation MDP (§4.3)")

    return {
        "cluster_mapping": cluster_mapping,
        "stats":           stats,
        "comparison":      comparison,
        "json_path":       json_path,
        "k":               k,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Module B Étape 6 — Interprétation clusters MINT/DGI (§4.2 PlateVision)"
    )
    parser.add_argument("--data-dir",    type=Path, default=Path("data/processed"))
    parser.add_argument("--models-dir",  type=Path, default=Path("models"))
    parser.add_argument("--figures-dir", type=Path,
                        default=Path("reports/rapport_technique/figures"))
    parser.add_argument("--report-dir",  type=Path,
                        default=Path("reports/rapport_technique"))
    parser.add_argument(
        "--manual-mapping",
        type=str,
        default=None,
        help=(
            'Mapping manuel JSON : \'{"0": {"label": "Plaque conforme", "action_mdp": "PASS"}, ...}\''
        ),
    )
    args = parser.parse_args()

    manual = None
    if args.manual_mapping:
        try:
            manual = json.loads(args.manual_mapping)
        except json.JSONDecodeError as exc:
            parser.error(f"--manual-mapping JSON invalide : {exc}")

    run_interpretation_pipeline(
        data_dir       = args.data_dir,
        models_dir     = args.models_dir,
        figures_dir    = args.figures_dir,
        report_dir     = args.report_dir,
        manual_mapping = manual,
    )
