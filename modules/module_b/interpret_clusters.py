"""
Module B — Interprétation sémantique des clusters selon les procédures MINT/DGI.

Rôle dans le pipeline PlateVision :
    Associe chaque cluster K-Means à une catégorie administrative camerounaise
    (immatriculation particulière, commerciale, diplomatique, transit, etc.)
    en analysant les caractéristiques statistiques des textes OCR regroupés.

Responsable : Étudiant 3
"""

import logging
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mapping des préfixes de plaques camerounaises vers leurs catégories MINT/DGI
CAMEROON_PLATE_CATEGORIES = {
    "LT": "Littoral (Douala)",
    "CE": "Centre (Yaoundé)",
    "NO": "Nord",
    "EN": "Extrême-Nord",
    "AD": "Adamaoua",
    "NW": "Nord-Ouest",
    "SW": "Sud-Ouest",
    "OU": "Ouest",
    "ES": "Est",
    "SU": "Sud",
    "CD": "Corps Diplomatique",
    "OF": "Officiel (Gouvernement)",
    "TT": "Transit temporaire",
    "MC": "Moto-cyclette commerciale",
}


def analyze_cluster_texts(
    plate_texts: list[str],
    labels: np.ndarray,
    n_clusters: int,
) -> dict[int, dict]:
    """
    Analyse les textes OCR de chaque cluster pour en déduire la catégorie dominante.

    Args:
        plate_texts: Textes OCR des plaques (un par embedding)
        labels: Labels de cluster assignés par K-Means
        n_clusters: Nombre total de clusters

    Returns:
        Dictionnaire {cluster_id: {'dominant_category', 'plate_samples', 'size'}}
    """
    cluster_analysis = {}

    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_texts = [plate_texts[i] for i, m in enumerate(cluster_mask) if m]

        # Extraction des préfixes (2 premières lettres)
        prefixes = []
        for text in cluster_texts:
            clean = text.replace(" ", "").upper()
            if len(clean) >= 2 and clean[:2].isalpha():
                prefixes.append(clean[:2])

        prefix_counts = Counter(prefixes)
        dominant_prefix = prefix_counts.most_common(1)[0][0] if prefix_counts else "INCONNU"
        dominant_category = CAMEROON_PLATE_CATEGORIES.get(dominant_prefix, "Catégorie inconnue")

        cluster_analysis[cluster_id] = {
            "size": int(cluster_mask.sum()),
            "dominant_prefix": dominant_prefix,
            "dominant_category": dominant_category,
            "prefix_distribution": dict(prefix_counts.most_common(5)),
            "plate_samples": cluster_texts[:5],
        }

        logger.info(
            "Cluster %d : %d plaques | Préfixe dominant '%s' → %s",
            cluster_id, cluster_mask.sum(), dominant_prefix, dominant_category,
        )

    return cluster_analysis


def build_cluster_summary_table(cluster_analysis: dict) -> pd.DataFrame:
    """
    Construit un tableau récapitulatif des clusters pour le rapport.

    Returns:
        DataFrame avec colonnes : Cluster, Taille, Catégorie, Préfixe dominant
    """
    rows = []
    for cluster_id, info in cluster_analysis.items():
        rows.append({
            "Cluster": cluster_id,
            "Taille": info["size"],
            "Catégorie MINT/DGI": info["dominant_category"],
            "Préfixe dominant": info["dominant_prefix"],
            "Exemples": ", ".join(info["plate_samples"][:3]),
        })

    df = pd.DataFrame(rows).sort_values("Cluster")
    return df


def map_cluster_to_dgi_procedure(cluster_id: int, cluster_analysis: dict) -> str:
    """
    Retourne la procédure DGI recommandée pour un cluster donné.
    Entrée pour le Module C (MDP) : définit les états et leurs procédures associées.

    Args:
        cluster_id: Identifiant du cluster
        cluster_analysis: Résultat de analyze_cluster_texts()

    Returns:
        Description de la procédure administrative DGI applicable.
    """
    dgi_procedures = {
        "Corps Diplomatique": "Exemption de taxes — vérification ambassade",
        "Officiel (Gouvernement)": "Contrôle immatriculation gouvernementale",
        "Transit temporaire": "Vérification délai de transit + taxe de passage",
        "Littoral (Douala)": "Vérification vignette fiscale + assurance DGI",
        "Centre (Yaoundé)": "Contrôle standard + carte grise",
    }

    category = cluster_analysis.get(cluster_id, {}).get("dominant_category", "")
    return dgi_procedures.get(category, "Procédure standard de contrôle véhicule")
