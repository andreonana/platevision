"""
Module C — Étape 3 : Matrice de transition P(s'|s,a)
======================================================
PlateVision — MINT/DGI Cameroun (§4.3 cahier des charges)

Matrice de transition P(s'|s,a) — Hypothèses de construction
=============================================================

HYPOTHÈSE 1 — Absence de trajectoires réelles
  Aucune donnée de trajectoire de contrôle routier MINT n'est disponible.
  La matrice est construite par modélisation documentée (méthode hybride
  données + expertise domaine) — pratique standard en MDP appliqué quand
  les données de trajectoire sont absentes (Sutton & Barto §1.6).

HYPOTHÈSE 2 — Reclassification inter-clusters (Source : Module B)
  La distance au centroïde dist_centroid (Étape B4) est utilisée comme
  proxy de l'incertitude de classification. Un point proche de son
  centroïde (confiance haute) a une faible probabilité d'être reclassé
  si la plaque est recontrôlée. Un point à la frontière entre clusters
  (confiance faible) a une probabilité plus élevée de migrer.
  Méthode : softmax des distances inverses aux k centroïdes.

HYPOTHÈSE 3 — Bruit opérationnel MINT (Source : §1.2)
  Taux d'erreur agent : 7–15% (note interne MINT/DGI, §1.2).
  Interprétation : à chaque lecture/recontrôle, avec probabilité ε ∈ [0.07, 0.15],
  l'état perçu change d'une dimension (cluster, confiance OCR, ou alerte).
  On utilise ε = 0.10 (valeur centrale — justifié dans le rapport).
  Cette erreur s'applique uniquement aux dimensions observées (conf_ocr, alerte),
  pas au cluster_id (qui reste stable entre deux recontrôles).

HYPOTHÈSE 4 — Sémantique des actions sur les transitions
  Les 5 actions ont des effets différents sur la dynamique des états :

  A0 — Laisser passer : état suivant = état actuel (véhicule repart).
    P(s'=s | s, A0) = 1 - ε_base
    P(s'≠s | s, A0) = ε_base   (bruit résiduel capteurs)
    où ε_base = 0.05

  A1 — Contrôle standard : révèle partiellement l'état vrai.
    Réduit l'incertitude sur conf_ocr : augmente la probabilité d'être
    dans un état à confiance haute si la plaque est réellement lisible.
    P(conf_ocr monte d'un niveau | s, A1) = 0.3 × (conf_level / n_conf_max)
    [formule : conf_level 0=haute → pas d'amélioration possible ;
               conf_level 2=faible → p=0.30 de monter à moyenne]
    P(alerte résolue | s, A1, alerte=1) = 0.4

  A2 — Arrêt + saisie : révèle l'état vrai avec haute probabilité.
    P(état vrai révélé | s, A2) = 0.85
    Après saisie, la plaque est neutralisée → état terminal simulé
    par un état absorbant (cluster conforme, confiance haute, sans alerte).

  A3 — Signalement DGI : transition administrative, pas physique.
    L'état du véhicule ne change pas (il n'est pas arrêté).
    P(s'=s | s, A3) = 1 - ε_base  (même comportement que A0)
    Mais dans le module Récompenses, R(s, A3) > R(s, A0) si cluster=expiré.

  A4 — Transfert PJ : similaire à A2 mais avec délai d'enquête.
    P(état vrai révélé | s, A4) = 0.90 (enquête plus approfondie)
    État absorbant après : même que A2.

HYPOTHÈSE 5 — Normalisation
  Chaque ligne P(s'|s,a) est normalisée (somme = 1.0).
  Les états éliminés lors du pruning (Étape C1) reçoivent P=0.
"""

import json
import logging
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    _SEABORN = True
except ImportError:
    _SEABORN = False

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Chargement des composantes MDP
# ──────────────────────────────────────────────────────────────────────────────

def load_mdp_components(
    data_dir: Path = Path("data/processed"),
) -> tuple:
    """Charge mdp_states.json, mdp_actions.json, metadata.csv, kmeans_centroids.npy."""
    data_dir = Path(data_dir)

    states_path = data_dir / "mdp_states.json"
    actions_path = data_dir / "mdp_actions.json"
    meta_path = data_dir / "metadata.csv"
    centroids_path = data_dir / "kmeans_centroids.npy"

    for p in (states_path, actions_path, meta_path, centroids_path):
        if not p.exists():
            raise RuntimeError(f"Fichier requis introuvable : {p}")

    with open(states_path, encoding="utf-8") as f:
        state_space = json.load(f)

    with open(actions_path, encoding="utf-8") as f:
        actions_dict = json.load(f)

    metadata = pd.read_csv(meta_path)
    for col in ("cluster_id", "dist_centroid", "confidence_level"):
        if col not in metadata.columns:
            raise RuntimeError(
                f"Colonne obligatoire absente dans metadata.csv : {col}"
            )

    centroids = np.load(centroids_path)

    N = state_space["n_states"]
    A = actions_dict["n_actions"]

    # Convertit les clés "[0,0,0]" → tuple (0,0,0)
    encoding = {}
    for k, v in state_space["encoding"].items():
        key = tuple(json.loads(k))
        encoding[key] = v

    state_space["_encoding_tuples"] = encoding

    logger.info("Composantes MDP chargées : N=%d états, A=%d actions", N, A)
    return state_space, actions_dict, metadata, centroids


# ──────────────────────────────────────────────────────────────────────────────
# 2. Matrice de reclassification inter-clusters (Hypothèse 2)
# ──────────────────────────────────────────────────────────────────────────────

def compute_reclassification_matrix(
    metadata: pd.DataFrame,
    centroids: np.ndarray,
    embeddings: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Matrice (k, k) : reclass_matrix[c][j] = probabilité qu'une plaque du
    cluster c soit reclassée dans j lors d'un recontrôle.
    Méthode : softmax des distances inverses aux centroïdes (Hypothèse 2).
    """
    n_samples = embeddings.shape[0]

    # Calcule distances euclidiennes à tous les centroïdes pour chaque point
    # shape (n_samples, k)
    diffs = embeddings[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (N, k, 256)
    dists = np.linalg.norm(diffs, axis=2)  # (N, k)

    # Softmax des distances inverses
    scores = 1.0 / (dists + 1e-8)  # (N, k)
    row_sums = scores.sum(axis=1, keepdims=True)
    probs = scores / row_sums  # (N, k) — chaque ligne somme à 1

    # Agrège par cluster actuel
    reclass = np.zeros((k, k), dtype=np.float64)
    for c in range(k):
        mask = metadata["cluster_id"].values == c
        if mask.sum() == 0:
            reclass[c, c] = 1.0
            continue
        reclass[c] = probs[mask].mean(axis=0)
        # Renormalise pour corriger l'arrondi flottant
        reclass[c] /= reclass[c].sum()

    logger.info("Matrice de reclassification (k=%d):", k)
    for c in range(k):
        row_str = "  ".join(f"{reclass[c, j]:.4f}" for j in range(k))
        logger.info("  cluster %d → [%s]", c, row_str)

    return reclass


# ──────────────────────────────────────────────────────────────────────────────
# 3. Matrice de bruit opérationnel MINT §1.2 (Hypothèse 3)
# ──────────────────────────────────────────────────────────────────────────────

def build_noise_matrix(
    N: int,
    state_space: dict,
    epsilon: float = 0.10,
) -> np.ndarray:
    """
    Matrice (N, N) : bruit opérationnel ε=10% (valeur centrale §1.2).
    Voisins d'une seule dimension : conf±1, alerte flippé.
    États non retenus (hors espace d'états pruné) sont ignorés.
    """
    encoding = state_space["_encoding_tuples"]
    # Inverse : state_id → (cluster, conf, alerte)
    decoding = {v: k for k, v in encoding.items()}

    noise = np.zeros((N, N), dtype=np.float64)

    for s in range(N):
        cluster, conf, alerte = decoding[s]
        n_conf = state_space["n_conf_levels"]

        neighbors = []

        # Voisins sur la dimension conf_ocr (±1)
        if conf - 1 >= 0:
            nb = (cluster, conf - 1, alerte)
            if nb in encoding:
                neighbors.append(encoding[nb])
        if conf + 1 < n_conf:
            nb = (cluster, conf + 1, alerte)
            if nb in encoding:
                neighbors.append(encoding[nb])

        # Voisin sur la dimension alerte (flip 0↔1)
        nb_alerte = (cluster, conf, 1 - alerte)
        if nb_alerte in encoding:
            neighbors.append(encoding[nb_alerte])

        # Déduplique (précaution)
        neighbors = list(dict.fromkeys(neighbors))
        n_nb = len(neighbors)

        noise[s, s] = 1.0 - epsilon
        if n_nb > 0:
            p_per_nb = epsilon / n_nb
            for nb_id in neighbors:
                noise[s, nb_id] += p_per_nb
        else:
            # Aucun voisin retenu → tout le bruit revient sur s
            noise[s, s] = 1.0

    # Renormalise chaque ligne (précaution)
    row_sums = noise.sum(axis=1, keepdims=True)
    noise /= row_sums

    return noise


# ──────────────────────────────────────────────────────────────────────────────
# 4. A0 — Laisser passer
# ──────────────────────────────────────────────────────────────────────────────

def build_transition_action0(
    N: int,
    noise_matrix: np.ndarray,
    epsilon_base: float = 0.05,
) -> np.ndarray:
    """
    LAISSER PASSER — état suivant ≈ état actuel (Hypothèse 4-A0).
    Combinaison d'une matrice quasi-identité et du bruit opérationnel.
    """
    T = np.zeros((N, N), dtype=np.float64)

    for s in range(N):
        # Bruit résiduel capteurs via noise_matrix pondéré par epsilon_base
        # Masse principale sur s, redistribution via voisins du bruit
        base = np.zeros(N)
        base[s] = 1.0 - epsilon_base
        # Ajoute epsilon_base de bruit résiduel via la ligne noise
        noise_row = noise_matrix[s].copy()
        noise_row[s] = 0.0
        noise_sum = noise_row.sum()
        if noise_sum > 0:
            noise_row /= noise_sum  # redistribue sur les voisins uniquement
            base += epsilon_base * noise_row
        else:
            base[s] += epsilon_base  # aucun voisin → revient sur s

        T[s] = base

    # Renormalise
    T /= T.sum(axis=1, keepdims=True)
    return T


# ──────────────────────────────────────────────────────────────────────────────
# 5. A1 — Contrôle standard
# ──────────────────────────────────────────────────────────────────────────────

def build_transition_action1(
    N: int,
    state_space: dict,
    noise_matrix: np.ndarray,
) -> np.ndarray:
    """
    CONTRÔLE STANDARD — révèle partiellement l'état vrai (Hypothèse 4-A1).
    P(conf améliore d'un niveau | s) = 0.3 × (conf_level / n_conf_max)
    [conf_level=0 (haute) → aucune amélioration possible ;
     conf_level=2 (faible) → p=0.30 de passer à moyenne]
    """
    encoding = state_space["_encoding_tuples"]
    decoding = {v: k for k, v in encoding.items()}
    n_conf = state_space["n_conf_levels"]  # 3
    n_conf_max = n_conf - 1  # 2

    T = np.zeros((N, N), dtype=np.float64)

    for s in range(N):
        cluster, conf, alerte = decoding[s]
        row = np.zeros(N)

        # Probabilité d'amélioration de conf (conf_level diminue = meilleure confiance)
        p_improve = 0.3 * (conf / n_conf_max) if n_conf_max > 0 else 0.0
        conf_better = conf - 1

        if conf_better >= 0:
            s_improved = encoding.get((cluster, conf_better, alerte))
            if s_improved is not None:
                row[s_improved] += p_improve
            else:
                p_improve = 0.0
        else:
            p_improve = 0.0

        # Probabilité de résolution d'alerte
        p_alerte_resolve = 0.4 * alerte  # 0 si alerte=0
        if alerte == 1:
            s_no_alerte = encoding.get((cluster, conf, 0))
            if s_no_alerte is not None:
                row[s_no_alerte] += p_alerte_resolve
            else:
                p_alerte_resolve = 0.0

        # Masse résiduelle sur état actuel
        p_stay = 1.0 - p_improve - p_alerte_resolve
        if p_stay < 0:
            p_stay = 0.0
        row[s] += p_stay

        # Mélange avec bruit MINT (Hypothèse 3)
        alpha_noise = 0.10
        row = (1.0 - alpha_noise) * row + alpha_noise * noise_matrix[s]

        T[s] = row

    # Renormalise
    T /= T.sum(axis=1, keepdims=True)
    return T


# ──────────────────────────────────────────────────────────────────────────────
# 6. A2 — Arrêt + saisie
# ──────────────────────────────────────────────────────────────────────────────

def build_transition_action2(
    N: int,
    state_space: dict,
    absorbing_state_id: int,
    p_reveal: float = 0.85,
) -> np.ndarray:
    """
    ARRÊT + SAISIE — révèle l'état vrai et neutralise la plaque (Hypothèse 4-A2).
    P(absorbing | s, A2) = p_reveal ; P(s | s, A2) = 1 - p_reveal.
    """
    T = np.zeros((N, N), dtype=np.float64)

    for s in range(N):
        T[s, absorbing_state_id] += p_reveal
        T[s, s] += 1.0 - p_reveal

    # Si s == absorbing, corrige pour éviter doublon
    T[absorbing_state_id, absorbing_state_id] = 1.0

    # Renormalise
    row_sums = T.sum(axis=1, keepdims=True)
    T /= row_sums
    return T


# ──────────────────────────────────────────────────────────────────────────────
# 7. A3 — Signalement DGI
# ──────────────────────────────────────────────────────────────────────────────

def build_transition_action3(
    N: int,
    noise_matrix: np.ndarray,
    epsilon_base: float = 0.05,
) -> np.ndarray:
    """
    SIGNALEMENT DGI — transition administrative, état physique inchangé
    (Hypothèse 4-A3). Identique à A0 (l'impact est capturé par les récompenses).
    """
    return build_transition_action0(N, noise_matrix, epsilon_base)


# ──────────────────────────────────────────────────────────────────────────────
# 8. A4 — Transfert PJ
# ──────────────────────────────────────────────────────────────────────────────

def build_transition_action4(
    N: int,
    state_space: dict,
    absorbing_state_id: int,
    p_reveal: float = 0.90,
) -> np.ndarray:
    """
    TRANSFERT PJ — similaire à A2 mais p_reveal=0.90 (enquête plus approfondie).
    """
    return build_transition_action2(N, state_space, absorbing_state_id, p_reveal)


# ──────────────────────────────────────────────────────────────────────────────
# 9. Assemblage du tenseur (N, A, N)
# ──────────────────────────────────────────────────────────────────────────────

def assemble_transition_tensor(T_list: list) -> np.ndarray:
    """
    Assemble les A matrices (N×N) en tenseur (N, A, N).
    T[s, a, s'] = P(s'|s,a).
    """
    N = T_list[0].shape[0]
    A = len(T_list)
    T = np.zeros((N, A, N), dtype=np.float64)

    for a, Ta in enumerate(T_list):
        T[:, a, :] = Ta

    # Vérification
    row_sums = T.sum(axis=2)  # (N, A)
    ok = np.allclose(row_sums, 1.0, atol=1e-6)
    if not ok:
        bad = np.argwhere(np.abs(row_sums - 1.0) > 1e-6)
        logger.warning("Lignes non normalisées (s, a) : %s", bad)

    logger.info("Tenseur assemblé : shape=%s, stochastique=%s", T.shape, ok)
    return T


# ──────────────────────────────────────────────────────────────────────────────
# 10. Validation du tenseur
# ──────────────────────────────────────────────────────────────────────────────

def validate_transition_tensor(T: np.ndarray, tol: float = 1e-6) -> bool:
    """Valide que T est un tenseur stochastique (N, A, N) sans NaN."""
    if T.ndim != 3 or T.shape[0] != T.shape[2]:
        raise ValueError(f"Shape invalide : {T.shape} — attendu (N, A, N)")

    if np.isnan(T).any():
        raise ValueError("Tenseur contient des NaN.")

    if (T < 0).any():
        raise ValueError("Tenseur contient des valeurs négatives.")

    row_sums = T.sum(axis=2)  # (N, A)
    if not np.allclose(row_sums, 1.0, atol=tol):
        bad = np.argwhere(np.abs(row_sums - 1.0) > tol)
        raise ValueError(
            f"Lignes non normalisées à {tol} près — paires (s,a) : {bad}"
        )

    logger.info("Tenseur P(s'|s,a) valide : shape=%s", T.shape)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 11. Visualisation — heatmaps des matrices de transition
# ──────────────────────────────────────────────────────────────────────────────

def _make_state_short_labels(state_space: dict) -> list:
    """Fabrique des labels courts pour les axes des heatmaps."""
    decoding = {v: k for k, v in state_space["_encoding_tuples"].items()}
    n_conf = state_space["n_conf_levels"]
    conf_sym = {0: "H", 1: "M", 2: "F"}  # Haute / Moyenne / Faible
    labels = []
    for s in sorted(decoding.keys()):
        cluster, conf, alerte = decoding[s]
        alerte_sym = "A" if alerte == 1 else "¬A"
        labels.append(f"C{cluster}·{conf_sym.get(conf, str(conf))}·{alerte_sym}")
    return labels


def plot_transition_matrices(
    T: np.ndarray,
    state_labels: list,
    action_labels: list,
    figures_dir: Path,
) -> None:
    """
    Figure A=5 heatmaps de P(s'|s,a) — disposition (1×5).
    Sauvegarde → figures_dir/transition_matrices.png
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    N, A, _ = T.shape
    annotate = N <= 12

    fig, axes = plt.subplots(1, A, figsize=(22, 5))
    if A == 1:
        axes = [axes]

    # Actions qui ont une dynamique quasi-identité (A0, A3)
    identity_actions = {0, 3}

    for a, ax in enumerate(axes):
        mat = T[:, a, :]

        if _SEABORN:
            sns.heatmap(
                mat,
                ax=ax,
                cmap="Blues",
                vmin=0,
                vmax=1,
                annot=annotate,
                fmt=".2f" if annotate else "",
                xticklabels=state_labels,
                yticklabels=state_labels,
                cbar=(a == A - 1),
                linewidths=0.3,
                linecolor="white",
            )
        else:
            im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            if annotate:
                for i in range(N):
                    for j in range(N):
                        ax.text(j, i, f"{mat[i, j]:.2f}",
                                ha="center", va="center", fontsize=7,
                                color="white" if mat[i, j] > 0.6 else "black")
            ax.set_xticks(range(N))
            ax.set_yticks(range(N))
            ax.set_xticklabels(state_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(state_labels, fontsize=8)
            if a == A - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xlabel("État suivant s'", fontsize=9)
        ax.set_ylabel("État courant s", fontsize=9)
        ax.set_title(
            f"P(s'|s, {action_labels[a]})\n(A{a})",
            fontsize=10,
            fontweight="bold",
        )

        # Encadrement de la diagonale pour A0 et A3
        if a in identity_actions:
            for i in range(N):
                ax.add_patch(
                    plt.Rectangle(
                        (i - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="orange",
                        lw=1.5,
                    )
                )

        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    fig.suptitle(
        f"Tenseur de transition P(s'|s,a) — MDP PlateVision\n"
        f"N={N} états × A=5 actions (§4.3 — méthode hybride dataset+MINT§1.2)",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )

    plt.tight_layout()
    out_path = figures_dir / "transition_matrices.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Heatmaps sauvegardées : %s", out_path)


# ──────────────────────────────────────────────────────────────────────────────
# 12. Export du tenseur
# ──────────────────────────────────────────────────────────────────────────────

def export_transition_tensor(
    T: np.ndarray,
    out_dir: Path = Path("data/processed"),
) -> Path:
    """Sauvegarde T → out_dir/mdp_transitions.npy"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mdp_transitions.npy"
    np.save(out_path, T)
    logger.info("mdp_transitions.npy sauvegardé : shape=%s", T.shape)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# 13. Pipeline principal
# ──────────────────────────────────────────────────────────────────────────────

def run_transitions_pipeline(
    data_dir: Path = Path("data/processed"),
    figures_dir: Path = Path("reports/rapport_technique/figures"),
    epsilon_mint: float = 0.10,
    epsilon_base: float = 0.05,
    p_reveal_saisie: float = 0.85,
    p_reveal_pj: float = 0.90,
) -> np.ndarray:
    """
    Orchestre la construction complète du tenseur P(s'|s,a).
    Retourne T de shape (N, A, N).
    """
    data_dir = Path(data_dir)
    figures_dir = Path(figures_dir)

    # 1. Chargement des composantes
    state_space, actions_dict, metadata, centroids = load_mdp_components(data_dir)
    N = state_space["n_states"]
    k = state_space["k_clusters"]
    A = actions_dict["n_actions"]

    # 2. Chargement des embeddings
    emb_path = data_dir / "embeddings.npy"
    if not emb_path.exists():
        raise RuntimeError(f"Embeddings introuvables : {emb_path}")
    embeddings = np.load(emb_path)
    logger.info("Embeddings chargés : shape=%s", embeddings.shape)

    # 3. Matrice de reclassification inter-clusters (Hypothèse 2)
    reclass = compute_reclassification_matrix(metadata, centroids, embeddings, k)

    # 4. Matrice de bruit opérationnel MINT §1.2 (Hypothèse 3, ε=10%)
    noise = build_noise_matrix(N, state_space, epsilon=epsilon_mint)

    # 5. Identification de l'état absorbant : (cluster=0, conf=0 [haute], alerte=0)
    encoding = state_space["_encoding_tuples"]
    absorbing_key = (0, 0, 0)
    if absorbing_key not in encoding:
        # Fallback : premier état du cluster 0
        absorbing_key = min(
            (k for k in encoding if k[0] == 0),
            key=lambda x: (x[1], x[2]),
        )
    absorbing_state_id = encoding[absorbing_key]
    logger.info(
        "État absorbant identifié : ID=%d %s", absorbing_state_id, absorbing_key
    )

    # 6–10. Construction des 5 matrices d'action
    T0 = build_transition_action0(N, noise, epsilon_base)
    T1 = build_transition_action1(N, state_space, noise)
    T2 = build_transition_action2(N, state_space, absorbing_state_id, p_reveal_saisie)
    T3 = build_transition_action3(N, noise, epsilon_base)
    T4 = build_transition_action4(N, state_space, absorbing_state_id, p_reveal_pj)

    # 11. Assemblage du tenseur (N, A, N)
    T = assemble_transition_tensor([T0, T1, T2, T3, T4])

    # 12. Validation
    validate_transition_tensor(T)

    # Labels pour la visualisation
    action_labels = [a["code"] for a in actions_dict["actions"]]
    state_labels = _make_state_short_labels(state_space)

    # 13. Heatmaps
    plot_transition_matrices(T, state_labels, action_labels, figures_dir)

    # 14. Export
    export_transition_tensor(T, data_dir)

    # Affichage récapitulatif
    print("=== Module C — Matrice de transition P(s'|s,a) estimée ===")
    print(f"Shape tenseur   : {T.shape}  (N_états × N_actions × N_états)")
    print(f"ε MINT §1.2     : {epsilon_mint:.0%} (bruit opérationnel)")
    print(f"p_reveal saisie : {p_reveal_saisie:.0%} (A2 — arrêt+saisie)")
    print(f"p_reveal PJ     : {p_reveal_pj:.0%} (A4 — transfert PJ)")
    print(f"État absorbant  : ID={absorbing_state_id} (conforme·haute·sans alerte)")
    print("Validation      : OK — toutes les lignes somment à 1.0")
    print(f"Fichier         : {data_dir}/mdp_transitions.npy")
    print("Prêt pour       : modules/module_c/mdp_rewards.py")

    return T


# ──────────────────────────────────────────────────────────────────────────────
# CLI — jury peut modifier ε et p_reveal en direct (§5 soutenance)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Module C — Matrice transition P(s'|s,a) (§4.3 PlateVision)"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed")
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("reports/rapport_technique/figures"),
    )
    parser.add_argument(
        "--epsilon-mint",
        type=float,
        default=0.10,
        help="Taux d'erreur MINT §1.2 (0.07–0.15)",
    )
    parser.add_argument(
        "--epsilon-base",
        type=float,
        default=0.05,
        help="Bruit résiduel capteurs (A0, A3)",
    )
    parser.add_argument(
        "--p-reveal-saisie",
        type=float,
        default=0.85,
        help="Prob. révélation état vrai — arrêt+saisie",
    )
    parser.add_argument(
        "--p-reveal-pj",
        type=float,
        default=0.90,
        help="Prob. révélation état vrai — transfert PJ",
    )
    args = parser.parse_args()

    run_transitions_pipeline(
        data_dir=args.data_dir,
        figures_dir=args.figures_dir,
        epsilon_mint=args.epsilon_mint,
        epsilon_base=args.epsilon_base,
        p_reveal_saisie=args.p_reveal_saisie,
        p_reveal_pj=args.p_reveal_pj,
    )
