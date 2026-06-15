"""
Module C — Étape 1 : Définition de l'espace d'états S du MDP PlateVision
MINT/DGI Cameroun — §4.3 cahier des charges

L'espace d'états est construit à partir des sorties conjointes du Module A
(signal d'alerte CNN / YOLO) et du Module B (cluster K-Means + confiance OCR) :

    S = {cluster_id} × {confiance_ocr} × {alerte_cnn}
      = k × 3 × 2 états (produit cartésien théorique)

§4.3 recommande 9 à 16 états. Le pruning par fréquence élimine les états
jamais ou rarement observés dans metadata.csv.
"""

import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MDPState:
    """Un état discret de l'espace S du MDP PlateVision."""
    state_id:       int
    cluster_id:     int
    conf_level:     int       # 0=faible 1=moyen 2=haute
    alerte_cnn:     int       # 0=absent 1=présent
    label:          str
    cluster_name:   str
    procedure:      str
    n_observations: int   = 0
    frequency:      float = 0.0


@dataclass
class MDPStateSpace:
    """L'espace d'états complet du MDP."""
    states:         list[MDPState]
    n_states:       int
    k_clusters:     int
    n_conf_levels:  int
    n_alerte:       int  = 2
    ocr_thresholds: dict = field(default_factory=dict)
    encoding:       dict = field(default_factory=dict)
    decoding:       dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT DES SORTIES MODULE B
# ══════════════════════════════════════════════════════════════════════════════

def load_module_b_outputs(
    data_dir: Path = Path("data/processed"),
) -> tuple[pd.DataFrame, dict]:
    """
    Charge metadata.csv et cluster_mapping.json produits par le Module B.
    Retourne (metadata, cluster_mapping).
    """
    data_dir = Path(data_dir)
    meta_path    = data_dir / "metadata.csv"
    mapping_path = data_dir / "cluster_mapping.json"

    if not meta_path.exists():
        raise RuntimeError(
            "metadata.csv absent. Exécute d'abord Module B (Étapes B4 + B6) : "
            "python main.py --module B"
        )
    required_cols = {"cluster_id", "confidence_level"}
    metadata = pd.read_csv(meta_path)
    missing = required_cols - set(metadata.columns)
    if missing:
        raise RuntimeError(
            f"metadata.csv incomplet — colonnes manquantes : {missing}. "
            "Exécute d'abord Module B (Étapes B4 + B6)"
        )

    if not mapping_path.exists():
        raise RuntimeError(
            "cluster_mapping.json absent. Exécute d'abord : "
            "python main.py --module B6"
        )
    with mapping_path.open(encoding="utf-8") as f:
        cluster_mapping = json.load(f)

    k = int(cluster_mapping["k"])
    n = len(metadata)
    logger.info("Module B chargé : k=%d clusters, N=%d observations", k, n)
    logger.info("Colonnes disponibles : %s", list(metadata.columns))
    return metadata, cluster_mapping


# ══════════════════════════════════════════════════════════════════════════════
# 2. SEUILS DE DISCRÉTISATION OCR
# ══════════════════════════════════════════════════════════════════════════════

def compute_ocr_thresholds(
    metadata: pd.DataFrame,
    n_levels: int = 3,
    method: str = "terciles",
) -> dict:
    """
    Calcule les seuils de discrétisation de la confiance OCR sur données réelles.

    Si ocr_conf absent ou tout NaN, utilise confidence_level issu du Module B4.
    """
    has_ocr = ("ocr_conf" in metadata.columns
               and not metadata["ocr_conf"].isna().all())

    if not has_ocr:
        logger.warning(
            "ocr_conf non disponible — confidence_level utilisé à la place "
            "(produit par Module B4 — terciles sur dist_centroid)"
        )
        return {
            "method": "from_b4",
            "q33": float("nan"),
            "q66": float("nan"),
            "levels": {0: "faible", 1: "moyen", 2: "haute"},
        }

    ocr = metadata["ocr_conf"].dropna().values

    # Terciles : seuils calculés sur les données réelles pour équilibrer les classes
    if method == "terciles":
        q33 = float(np.nanpercentile(ocr, 33))  # ≈ frontière bas/moyen
        q66 = float(np.nanpercentile(ocr, 66))  # ≈ frontière moyen/haute
    elif method == "fixed":
        # Seuils fixes calibrés MINT (utiles pour reproductibilité inter-sessions)
        q33 = 0.5
        q66 = 0.8
    else:
        raise ValueError(f"method inconnu : {method!r}. Choix : 'terciles', 'fixed'")

    n0 = int((ocr <= q33).sum())
    n1 = int(((ocr > q33) & (ocr <= q66)).sum())
    n2 = int((ocr > q66).sum())
    total = len(ocr)

    logger.info("Seuils OCR (method=%s) : q33=%.4f, q66=%.4f", method, q33, q66)
    logger.info(
        "Distribution : faible=%d (%.1f%%) | moy=%d (%.1f%%) | haute=%d (%.1f%%)",
        n0, 100 * n0 / total,
        n1, 100 * n1 / total,
        n2, 100 * n2 / total,
    )
    return {
        "method": method,
        "q33": q33,
        "q66": q66,
        "levels": {0: "faible", 1: "moyen", 2: "haute"},
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. DISCRÉTISATION OCR
# ══════════════════════════════════════════════════════════════════════════════

def discretize_ocr(
    metadata: pd.DataFrame,
    thresholds: dict,
) -> np.ndarray:
    """
    Applique les seuils pour produire un vecteur de niveaux 0/1/2.
    Utilise confidence_level directement si ocr_conf absent.
    """
    # Si ocr_conf absent, on réutilise directement confidence_level du Module B4
    if thresholds.get("method") == "from_b4":
        return metadata["confidence_level"].values.astype(np.int32)

    ocr = metadata["ocr_conf"].values.astype(np.float32)
    q33 = thresholds["q33"]
    q66 = thresholds["q66"]
    # Application des seuils : 0=faible (≤q33), 1=moyen (q33<·≤q66), 2=haute (>q66)
    levels = np.where(ocr <= q33, 0, np.where(ocr <= q66, 1, 2)).astype(np.int32)
    return levels


# ══════════════════════════════════════════════════════════════════════════════
# 4. SIGNAL D'ALERTE CNN
# ══════════════════════════════════════════════════════════════════════════════

def infer_alerte_cnn(
    metadata: pd.DataFrame,
    yolo_conf_threshold: float = 0.5,
) -> np.ndarray:
    """
    Infère le signal d'alerte CNN (dimension 3 de S) par ordre de priorité :
      1. Colonne alerte_cnn existante
      2. ocr_conf < 0.3 → alerte
      3. conformite != "conforme" → alerte
      4. Fallback : tout à 0 (log warning)
    """
    n = len(metadata)

    if "alerte_cnn" in metadata.columns:
        alerte = metadata["alerte_cnn"].values.astype(np.int32)
        source = "alerte_cnn"
    elif "ocr_conf" in metadata.columns and not metadata["ocr_conf"].isna().all():
        alerte = (metadata["ocr_conf"].fillna(1.0) < 0.3).astype(np.int32)
        source = "ocr_conf<0.3"
    elif "conformite" in metadata.columns:
        alerte = (metadata["conformite"] != "conforme").astype(np.int32)
        source = "conformite"
    else:
        logger.warning(
            "Aucune colonne alerte disponible (alerte_cnn / ocr_conf / conformite). "
            "Signal alerte CNN forcé à 0 (absence alerte) pour tous les états."
        )
        alerte = np.zeros(n, dtype=np.int32)
        source = "fallback_zero"

    n_alertes = int(alerte.sum())
    pct = 100.0 * n_alertes / n if n > 0 else 0.0
    logger.info(
        "Signal alerte CNN : %d alertes (%.1f%%) — source=%s",
        n_alertes, pct, source,
    )
    return alerte


# ══════════════════════════════════════════════════════════════════════════════
# 5. PRODUIT CARTÉSIEN DES ÉTATS THÉORIQUES
# ══════════════════════════════════════════════════════════════════════════════

def build_cartesian_states(
    k: int,
    n_conf: int = 3,
    cluster_mapping: dict = None,
) -> list[tuple]:
    """
    Génère les k × n_conf × 2 combinaisons théoriques (cluster, conf, alerte).
    """
    # Produit cartésien complet : tous les triplets (cluster, conf, alerte) théoriques
    # Certains seront éliminés par prune_states si jamais observés dans les données
    states = list(itertools.product(range(k), range(n_conf), range(2)))
    logger.info(
        "Produit cartésien : %d clusters × %d niveaux conf × 2 alertes = %d états",
        k, n_conf, len(states),
    )
    return states


# ══════════════════════════════════════════════════════════════════════════════
# 6. COMPTAGE DES OBSERVATIONS
# ══════════════════════════════════════════════════════════════════════════════

def count_state_observations(
    metadata: pd.DataFrame,
    cluster_labels: np.ndarray,
    conf_labels: np.ndarray,
    alerte_labels: np.ndarray,
    all_states: list[tuple],
) -> dict[tuple, int]:
    """
    Compte les occurrences de chaque triplet (cluster, conf, alerte) dans les données.
    """
    counts: dict[tuple, int] = {t: 0 for t in all_states}
    for c, l, a in zip(
        cluster_labels.astype(int),
        conf_labels.astype(int),
        alerte_labels.astype(int),
    ):
        key = (c, l, a)
        if key in counts:
            counts[key] += 1
    observed = sum(1 for v in counts.values() if v > 0)
    logger.info(
        "Observations comptées : %d/%d triplets observés au moins une fois",
        observed, len(all_states),
    )
    return counts


# ══════════════════════════════════════════════════════════════════════════════
# 7. PRUNING
# ══════════════════════════════════════════════════════════════════════════════

def prune_states(
    all_states: list[tuple],
    observations: dict[tuple, int],
    n_total: int,
    target_n_states: tuple[int, int] = (9, 16),
    min_frequency: float = 0.01,
) -> tuple[list[tuple], list[tuple]]:
    """
    Réduit l'espace d'états pour respecter §4.3 (9-16 états).

    Étapes dans l'ordre :
      1. Élimine états jamais observés (count == 0)
      2. Élimine états trop rares (freq < min_frequency)
      3. Si encore > 16 : fusionne niveaux de confiance 1+2 → 1
      4. Si encore > 16 : fusionne clusters proches
    """
    n_max = target_n_states[1]
    retained = list(all_states)
    eliminated: list[tuple] = []

    # ── Étape 1 : jamais observés — états non représentés dans metadata.csv
    # Ces états sont théoriquement possibles mais absents des données d'entraînement
    n_avant = len(retained)
    retained_new = [t for t in retained if observations.get(t, 0) > 0]
    elim_new = [t for t in retained if observations.get(t, 0) == 0]
    eliminated.extend(elim_new)
    retained = retained_new
    logger.info("Pruning étape 1 (jamais observés) : %d → %d états", n_avant, len(retained))

    # ── Étape 2 : trop rares — fréq < min_frequency (1%) : bruit, pas de signal
    # Garder des états très rares rendrait les matrices P et R non fiables
    n_avant = len(retained)
    retained_new = [
        t for t in retained
        if observations.get(t, 0) / n_total >= min_frequency
    ]
    elim_new = [
        t for t in retained
        if observations.get(t, 0) / n_total < min_frequency
    ]
    eliminated.extend(elim_new)
    retained = retained_new
    logger.info(
        "Pruning étape 2 (freq < %.1f%%) : %d → %d états",
        min_frequency * 100, n_avant, len(retained),
    )

    # ── Étape 3 : fusion niveaux de confiance si encore > n_max ──────────────
    if len(retained) > n_max:
        n_avant = len(retained)
        # Fusionne conf 1 et 2 → 1 (binaire : 0=faible, 1=élevé)
        merged: dict[tuple, int] = {}
        for (c, l, a), cnt in observations.items():
            l_bin = 0 if l == 0 else 1
            key = (c, l_bin, a)
            merged[key] = merged.get(key, 0) + cnt
        retained_merged = list(set((c, 0 if l == 0 else 1, a) for (c, l, a) in retained))
        retained_merged = [
            t for t in retained_merged
            if merged.get(t, 0) / n_total >= min_frequency
        ]
        elim_merged = [t for t in retained if (t[0], 0 if t[1] == 0 else 1, t[2]) not in retained_merged]
        eliminated.extend(elim_merged)
        retained = sorted(set(retained_merged))
        logger.info(
            "Pruning étape 3 (fusion conf 1+2→1) : %d → %d états",
            n_avant, len(retained),
        )

    # ── Étape 4 : fusion clusters proches si encore > n_max ───────────────────
    if len(retained) > n_max:
        n_avant = len(retained)
        clusters_present = sorted(set(t[0] for t in retained))
        if len(clusters_present) >= 2:
            # Fusionne les deux plus petits clusters (par taille d'observation)
            cluster_sizes = {
                c: sum(observations.get(t, 0) for t in retained if t[0] == c)
                for c in clusters_present
            }
            sorted_by_size = sorted(clusters_present, key=lambda c: cluster_sizes[c])
            c_merge_from = sorted_by_size[0]
            c_merge_into = sorted_by_size[1]
            retained = [
                (c_merge_into if t[0] == c_merge_from else t[0], t[1], t[2])
                for t in retained
            ]
            retained = sorted(set(retained))
            logger.info(
                "Pruning étape 4 (fusion clusters %d→%d) : %d → %d états",
                c_merge_from, c_merge_into, n_avant, len(retained),
            )

    n_elim = len(eliminated)
    in_range = target_n_states[0] <= len(retained) <= target_n_states[1]
    logger.info(
        "Espace d'états retenu : %d états (§4.3 : %d-%d recommandés) — %s",
        len(retained), target_n_states[0], target_n_states[1],
        "OK" if in_range else "HORS RECOMMANDATION",
    )
    logger.info("États éliminés : %d (jamais/rarement observés)", n_elim)
    return retained, eliminated


# ══════════════════════════════════════════════════════════════════════════════
# 8. CONSTRUCTION DE L'ESPACE D'ÉTATS
# ══════════════════════════════════════════════════════════════════════════════

# Convention Module B (kmeans_fit.py — discretize_distance) :
#   confidence_level 0 = haute  (dist ≤ q33, proche centroïde)
#   confidence_level 1 = moyenne
#   confidence_level 2 = faible (dist > q66, cas ambigu)
_CONF_LABELS  = {0: "Conf.Haute", 1: "Conf.Moy",  2: "Conf.Faible"}
_ALERTE_LABELS = {0: "SansAlerte", 1: "AvecAlerte"}
_CONF_NAMES   = {0: "haute",      1: "moyen",     2: "faible"}


def _get_cluster_info(cluster_mapping: dict, cluster_id: int) -> tuple[str, str]:
    """Retourne (nom_cluster, procedure) depuis cluster_mapping."""
    cid_str = str(cluster_id)
    clusters = cluster_mapping.get("clusters", {})
    if cid_str in clusters:
        c = clusters[cid_str]
        return c.get("label", f"Cluster {cluster_id}"), c.get("procedure", "—")
    # Compatibilité avec format alternatif {"mapping": {...}}
    mapping = cluster_mapping.get("mapping", {})
    if cid_str in mapping:
        m = mapping[cid_str]
        return m.get("nom", f"Cluster {cluster_id}"), m.get("procedure", "—")
    return f"Cluster {cluster_id}", "—"


def build_state_space(
    retained_states: list[tuple],
    observations: dict[tuple, int],
    n_total: int,
    cluster_mapping: dict,
    thresholds: dict,
    n_conf_levels: int,
) -> MDPStateSpace:
    """
    Construit l'objet MDPStateSpace à partir des triplets retenus.
    """
    k = int(cluster_mapping["k"])
    states: list[MDPState] = []
    encoding: dict[tuple, int] = {}
    decoding: dict[int, tuple] = {}

    for sid, triplet in enumerate(sorted(retained_states)):
        c, l, a = triplet
        cluster_name, procedure = _get_cluster_info(cluster_mapping, c)
        conf_label   = _CONF_LABELS.get(l, f"Conf.{l}")
        alerte_label = _ALERTE_LABELS.get(a, f"Alerte.{a}")
        label = f"État {sid} — {cluster_name}·{conf_label}·{alerte_label}"

        n_obs = int(observations.get(triplet, 0))
        freq  = float(n_obs / n_total) if n_total > 0 else 0.0

        state = MDPState(
            state_id       = sid,
            cluster_id     = c,
            conf_level     = l,
            alerte_cnn     = a,
            label          = label,
            cluster_name   = cluster_name,
            procedure      = procedure,
            n_observations = n_obs,
            frequency      = freq,
        )
        states.append(state)
        encoding[triplet] = sid
        decoding[sid] = triplet

    return MDPStateSpace(
        states         = states,
        n_states       = len(states),
        k_clusters     = k,
        n_conf_levels  = n_conf_levels,
        n_alerte       = 2,
        ocr_thresholds = thresholds,
        encoding       = encoding,
        decoding       = decoding,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 9. EXPORT JSON
# ══════════════════════════════════════════════════════════════════════════════

def export_state_space(
    space: MDPStateSpace,
    out_dir: Path = Path("data/processed"),
) -> Path:
    """
    Sauvegarde mdp_states.json — chargeable par mdp_solver.py.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sérialise encoding avec clé string "[c,l,a]"
    encoding_serial = {
        f"[{c},{l},{a}]": sid
        for (c, l, a), sid in space.encoding.items()
    }

    # Seuils : convertit NaN en null pour JSON
    thresholds_serial = {}
    for k, v in space.ocr_thresholds.items():
        if k == "levels":
            thresholds_serial[k] = {str(kk): vv for kk, vv in v.items()}
        elif isinstance(v, float) and (v != v):  # NaN check
            thresholds_serial[k] = None
        else:
            thresholds_serial[k] = v

    payload = {
        "n_states":        space.n_states,
        "k_clusters":      space.k_clusters,
        "n_conf_levels":   space.n_conf_levels,
        "ocr_thresholds":  thresholds_serial,
        "states": [
            {
                "state_id":       s.state_id,
                "cluster_id":     s.cluster_id,
                "conf_level":     s.conf_level,
                "alerte_cnn":     s.alerte_cnn,
                "label":          s.label,
                "cluster_name":   s.cluster_name,
                "procedure":      s.procedure,
                "n_observations": s.n_observations,
                "frequency":      round(s.frequency, 6),
            }
            for s in space.states
        ],
        "encoding":        encoding_serial,
        "mint_dgi_source": "§4.3 PlateVision — MINT/DGI Cameroun",
    }

    out_path = out_dir / "mdp_states.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "mdp_states.json exporté — %d états → chargeable par mdp_solver.py",
        space.n_states,
    )
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# 10. AFFICHAGE RÉCAPITULATIF
# ══════════════════════════════════════════════════════════════════════════════

def print_state_space_summary(space: MDPStateSpace) -> None:
    """Affiche le tableau récapitulatif des états pour rapport et soutenance §5."""
    k_th = space.k_clusters * space.n_conf_levels * space.n_alerte
    n_elim = k_th - space.n_states

    header = (
        f"{'ID':^4} │ {'Cluster':^5} │ {'Conf.OCR':^9} │ {'Alerte':^8} │ "
        f"{'Procédure MINT/DGI':<22} │ {'Freq (%)':>8}"
    )
    sep = "─" * len(header)
    print()
    print("┌" + sep + "┐")
    print("│" + header + "│")
    print("├" + sep + "┤")
    for s in space.states:
        conf_str  = _CONF_NAMES.get(s.conf_level, str(s.conf_level))
        alert_str = "présent" if s.alerte_cnn else "absent"
        proc_short = s.procedure[:22] if s.procedure else "—"
        row = (
            f"{s.state_id:^4} │ {s.cluster_id:^5} │ {conf_str:^9} │ "
            f"{alert_str:^8} │ {proc_short:<22} │ {s.frequency*100:>7.1f}%"
        )
        print("│" + row + "│")
    print("└" + sep + "┘")
    print()
    print(f"N états retenus : {space.n_states} (§4.3 recommande 9-16)")
    print(f"États théoriques (produit cartésien) : {k_th}")
    print(f"États éliminés : {n_elim} (fréquence < seuil ou jamais observés)")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 11. PIPELINE COMPLET
# ══════════════════════════════════════════════════════════════════════════════

def run_state_space_pipeline(
    data_dir: Path = Path("data/processed"),
    out_dir: Path = Path("data/processed"),
    ocr_method: str = "terciles",
    yolo_conf_threshold: float = 0.5,
    min_frequency: float = 0.01,
    target_n_states: tuple[int, int] = (9, 16),
) -> MDPStateSpace:
    """
    Orchestre la construction complète de l'espace d'états MDP Module C.
    """
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)

    # ── 1. Chargement sorties Module B ────────────────────────────────────────
    metadata, cluster_mapping = load_module_b_outputs(data_dir)
    k = int(cluster_mapping["k"])
    n_conf = 3

    # ── 2. Seuils OCR ─────────────────────────────────────────────────────────
    thresholds = compute_ocr_thresholds(metadata, n_levels=n_conf, method=ocr_method)

    # ── 3. Discrétisation OCR ─────────────────────────────────────────────────
    conf_labels = discretize_ocr(metadata, thresholds)

    # ── 4. Signal alerte CNN ──────────────────────────────────────────────────
    alerte_labels = infer_alerte_cnn(metadata, yolo_conf_threshold)

    # ── 5. Produit cartésien ──────────────────────────────────────────────────
    cluster_labels = metadata["cluster_id"].values.astype(np.int32)
    all_states = build_cartesian_states(k, n_conf, cluster_mapping)

    # ── 6. Comptage observations ──────────────────────────────────────────────
    observations = count_state_observations(
        metadata, cluster_labels, conf_labels, alerte_labels, all_states
    )

    # ── 7. Pruning ────────────────────────────────────────────────────────────
    retained, eliminated = prune_states(
        all_states, observations, len(metadata),
        target_n_states=target_n_states,
        min_frequency=min_frequency,
    )

    # ── 8. Construction espace d'états ────────────────────────────────────────
    space = build_state_space(
        retained, observations, len(metadata),
        cluster_mapping, thresholds, n_conf,
    )

    # ── 9. Affichage ──────────────────────────────────────────────────────────
    print_state_space_summary(space)

    # ── 10. Export JSON ───────────────────────────────────────────────────────
    json_path = export_state_space(space, out_dir)

    # ── Résumé console ────────────────────────────────────────────────────────
    n_th = k * n_conf * 2
    in_range = target_n_states[0] <= space.n_states <= target_n_states[1]
    print("=== Module C — Espace d'états MDP défini ===")
    print(f"Dimensions : {k} clusters × {n_conf} niveaux OCR × 2 alertes")
    print(f"États théoriques : {n_th}")
    print(f"États retenus    : {space.n_states} "
          f"(après pruning fréquence < {min_frequency*100:.0f}%)")
    print(f"Respect §4.3     : {'OUI' if in_range else 'HORS RECOMMANDATION — justifier'}")
    if not in_range:
        print(
            "  Note : le signal alerte_cnn est absent de metadata.csv (données d'entraînement).\n"
            "  En inférence réelle (YOLO live), les états alerte=1 s'activent → espace complet.\n"
            "  Pour forcer l'espace complet : --min-frequency 0.0"
        )
    print(f"Fichier          : {json_path}")
    print(f"Prêt pour        : modules/module_c/mdp_transitions.py")

    return space


# ══════════════════════════════════════════════════════════════════════════════
# CLI — §5 soutenance : jury peut modifier seuils et n_states en direct
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Module C — Définition espace d'états MDP (§4.3 PlateVision)"
    )
    parser.add_argument("--data-dir",       type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir",        type=Path, default=Path("data/processed"))
    parser.add_argument("--ocr-method",     choices=["terciles", "fixed"],
                        default="terciles",
                        help="Méthode de discrétisation confiance OCR")
    parser.add_argument("--yolo-threshold", type=float, default=0.5,
                        help="Seuil confiance YOLO pour signal alerte CNN")
    parser.add_argument("--min-frequency",  type=float, default=0.01,
                        help="Fréquence minimale pour conserver un état")
    parser.add_argument("--n-states-min",   type=int,   default=9)
    parser.add_argument("--n-states-max",   type=int,   default=16)
    args = parser.parse_args()

    run_state_space_pipeline(
        data_dir           = args.data_dir,
        out_dir            = args.out_dir,
        ocr_method         = args.ocr_method,
        yolo_conf_threshold= args.yolo_threshold,
        min_frequency      = args.min_frequency,
        target_n_states    = (args.n_states_min, args.n_states_max),
    )
