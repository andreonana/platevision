"""
PlateVision — Module B : Interprétation des clusters.

Ce module fait la correspondance entre les clusters K-Means et les
catégories de véhicules identifiables par leurs plaques au Cameroun.
Chaque cluster est associé à des procédures administratives spécifiques
relevant du MINT (contrôle routier) ou de la DGI (fiscalité).

Rôle dans le pipeline : donner du sens aux clusters non supervisés.
"""

import logging
from typing import Dict, List, Any

import numpy as np

logger = logging.getLogger(__name__)

# Correspondance entre catégories de plaques camerounaises et procédures MINT/DGI
PLATE_CATEGORIES_CAMEROON = {
    "vehicule_particulier": {
        "description": "Véhicule particulier (fond blanc, lettres noires)",
        "format_regex": r"[A-Z]{2}\s?\d{3,4}\s?[A-Z]{2}",
        "mint_procedure": "Contrôle technique, assurance RC obligatoire",
        "dgi_procedure": "Vignette auto annuelle, déclaration IR si usage professionnel",
    },
    "vehicule_commercial": {
        "description": "Véhicule de transport public / taxi (fond jaune)",
        "format_regex": r"[A-Z]{1,2}\s?\d{3,4}\s?[A-Z]{1,2}",
        "mint_procedure": "Carte de transport, visite technique semestrielle",
        "dgi_procedure": "Patente transport, TVA sur recettes",
    },
    "vehicule_officiel": {
        "description": "Véhicule de l'État ou collectivité",
        "format_regex": r"CD\s?\d+|OF\s?\d+",
        "mint_procedure": "Immatriculation gouvernementale, contrôle préfectoral",
        "dgi_procedure": "Exonéré de vignette — suivi budgétaire",
    },
    "moto": {
        "description": "Motocyclette / cyclomoteur",
        "format_regex": r"\d{1,3}\s?[A-Z]\s?\d{2,4}",
        "mint_procedure": "Contrôle technique moto, port casque obligatoire",
        "dgi_procedure": "Taxe de roulage moto",
    },
    "plaque_temporaire": {
        "description": "Plaque provisoire (véhicule en transit)",
        "format_regex": r"T\s?\d{4,6}",
        "mint_procedure": "Autorisation temporaire de circulation (30 jours max)",
        "dgi_procedure": "Exonération temporaire, contrôle à l'importation",
    },
}


def assign_cluster_to_category(
    cluster_id: int,
    cluster_samples: List[str],
    categories: Dict = PLATE_CATEGORIES_CAMEROON,
) -> str:
    """
    Assigne un cluster à une catégorie de plaque selon les textes OCR dominants.

    Cette fonction utilise des heuristiques de matching regex pour identifier
    la catégorie la plus probable d'un cluster.

    Args:
        cluster_id: Identifiant du cluster
        cluster_samples: Liste des textes OCR des plaques dans ce cluster
        categories: Dictionnaire des catégories de plaques

    Returns:
        Nom de la catégorie assignée
    """
    import re

    category_scores = {cat: 0 for cat in categories}

    for text in cluster_samples:
        for cat_name, cat_info in categories.items():
            pattern = cat_info.get("format_regex", "")
            if pattern and re.search(pattern, text.upper()):
                category_scores[cat_name] += 1

    best_category = max(category_scores, key=category_scores.get)
    best_score = category_scores[best_category]

    if best_score == 0:
        logger.warning(f"Cluster {cluster_id} : aucun pattern correspondant, catégorie inconnue")
        return "inconnu"

    logger.info(
        f"Cluster {cluster_id} → Catégorie '{best_category}' "
        f"({best_score}/{len(cluster_samples)} correspondances)"
    )
    return best_category


def interpret(
    labels: np.ndarray,
    centroids: np.ndarray,
    ocr_texts: List[str] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Interprète chaque cluster en termes de catégorie de plaque et procédures.

    Args:
        labels: Labels K-Means (un label par échantillon)
        centroids: Centroïdes des clusters
        ocr_texts: Liste des textes OCR correspondants (optionnel)

    Returns:
        Dictionnaire d'interprétation par cluster
    """
    n_clusters = len(np.unique(labels))
    interpretation = {}

    logger.info(f"\n{'='*60}")
    logger.info("INTERPRÉTATION DES CLUSTERS — PlateVision")
    logger.info(f"{'='*60}")

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        n_samples = int(mask.sum())

        # Textes OCR du cluster (si disponibles)
        cluster_texts = []
        if ocr_texts is not None:
            cluster_texts = [
                text for i, text in enumerate(ocr_texts)
                if i < len(labels) and labels[i] == cluster_id
            ]

        # Tentative d'assignation à une catégorie
        category_name = assign_cluster_to_category(cluster_id, cluster_texts)
        category_info = PLATE_CATEGORIES_CAMEROON.get(category_name, {})

        interpretation[cluster_id] = {
            "n_samples": n_samples,
            "proportion": f"{n_samples/len(labels)*100:.1f}%",
            "category": category_name,
            "description": category_info.get("description", "Catégorie non identifiée"),
            "mint_procedure": category_info.get("mint_procedure", "À déterminer"),
            "dgi_procedure": category_info.get("dgi_procedure", "À déterminer"),
            "centroid_norm": float(np.linalg.norm(centroids[cluster_id])),
        }

        logger.info(f"\nCluster {cluster_id} ({n_samples} plaques, {interpretation[cluster_id]['proportion']}):")
        logger.info(f"  Catégorie     : {category_name}")
        logger.info(f"  Description   : {category_info.get('description', 'N/A')}")
        logger.info(f"  MINT          : {category_info.get('mint_procedure', 'N/A')}")
        logger.info(f"  DGI           : {category_info.get('dgi_procedure', 'N/A')}")

    logger.info(f"\n{'='*60}")
    return interpretation
