"""
PlateVision — Point d'entrée CLI du pipeline complet.

Ce module expose une interface en ligne de commande permettant d'exécuter
les modules A (détection/OCR), B (clustering), C (MDP) ou le pipeline
complet sur une vidéo. Il orchestre les appels aux sous-modules et gère
la configuration via les variables d'environnement.

Usage :
    python main.py --module A --input data/processed/ --output reports/
    python main.py --module B --embeddings models/embeddings.npy
    python main.py --module C --gamma 0.9 --algorithm VI
    python main.py --pipeline full --input data/raw/video.mp4
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Chargement des variables d'environnement depuis .env
load_dotenv()

# Configuration du logger principal
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("platevision")


def run_module_a(args: argparse.Namespace) -> None:
    """Exécute le Module A : Naïves Bayes et/ou pipeline YOLO+OCR."""
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Dossier d'entrée introuvable : {input_path}")
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Module A — Entrée : {input_path} | Sortie : {output_path}")

    # Import différé pour éviter les erreurs si dépendances absentes
    from modules.module_a.naive_bayes import run_naive_bayes
    from modules.module_a.yolo_ocr_pipeline import run_yolo_ocr
    from modules.module_a.evaluate import compute_metrics

    logger.info("Étape A1 : Classifieur Naïves Bayes...")
    nb_results = run_naive_bayes(input_path, output_path)

    logger.info("Étape A2 : Pipeline YOLO + OCR...")
    ocr_results = run_yolo_ocr(input_path, output_path)

    logger.info("Calcul des métriques (mAP, CER, WER)...")
    compute_metrics(nb_results, ocr_results, output_path)

    logger.info("Module A terminé.")


def run_module_b(args: argparse.Namespace) -> None:
    """Exécute le Module B : Clustering K-Means sur embeddings CNN."""
    embeddings_path = Path(args.embeddings)

    if not embeddings_path.exists():
        logger.error(f"Fichier d'embeddings introuvable : {embeddings_path}")
        sys.exit(1)

    logger.info(f"Module B — Embeddings : {embeddings_path}")

    from modules.module_b.clustering import run_kmeans
    from modules.module_b.visualization import plot_clusters
    from modules.module_b.interpret_clusters import interpret

    n_clusters = int(os.getenv("KMEANS_N_CLUSTERS", 8))
    logger.info(f"Clustering K-Means avec K={n_clusters}...")
    labels, centroids = run_kmeans(embeddings_path, n_clusters=n_clusters)

    logger.info("Visualisation PCA 2D / t-SNE...")
    plot_clusters(embeddings_path, labels)

    logger.info("Interprétation des clusters → procédures MINT/DGI...")
    interpret(labels, centroids)

    logger.info("Module B terminé.")


def run_module_c(args: argparse.Namespace) -> None:
    """Exécute le Module C : MDP via Value Iteration ou Policy Iteration."""
    gamma = args.gamma
    algorithm = args.algorithm.upper()

    if algorithm not in ("VI", "PI"):
        logger.error("Algorithme invalide. Utiliser VI (Value Iteration) ou PI (Policy Iteration).")
        sys.exit(1)

    if not (0.0 < gamma <= 1.0):
        logger.error("Le facteur gamma doit être entre 0 (exclus) et 1.0.")
        sys.exit(1)

    logger.info(f"Module C — Algorithme : {algorithm} | Gamma : {gamma}")

    from modules.module_c.mdp_definition import build_mdp
    from modules.module_c.value_iteration import value_iteration
    from modules.module_c.policy_iteration import policy_iteration
    from modules.module_c.compare_vi_pi import compare_algorithms

    mdp = build_mdp()

    if algorithm == "VI":
        logger.info("Exécution de Value Iteration...")
        policy, values = value_iteration(mdp, gamma=gamma)
    else:
        logger.info("Exécution de Policy Iteration...")
        policy, values = policy_iteration(mdp, gamma=gamma)

    logger.info("Comparaison VI vs PI et analyse de sensibilité...")
    compare_algorithms(mdp, gamma=gamma)

    logger.info(f"Module C terminé. Politique optimale calculée en {algorithm}.")


def run_full_pipeline(args: argparse.Namespace) -> None:
    """Exécute le pipeline complet PlateVision sur une vidéo ou dossier d'images."""
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error(f"Fichier ou dossier d'entrée introuvable : {input_path}")
        sys.exit(1)

    output_path = Path(os.getenv("OUTPUT_REPORTS_PATH", "reports/"))
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLET PLATEVISION")
    logger.info("=" * 60)

    # Prétraitement
    logger.info("[1/4] Prétraitement des images...")
    from preprocessing.normalization import normalize_images
    from preprocessing.augmentation import augment_dataset
    processed_path = Path(os.getenv("DATA_PROCESSED_PATH", "data/processed/"))
    normalize_images(input_path, processed_path)

    # Module A
    logger.info("[2/4] Module A — Détection et reconnaissance...")
    args.input = str(processed_path)
    args.output = str(output_path)
    run_module_a(args)

    # Module B
    logger.info("[3/4] Module B — Clustering...")
    embeddings_path = Path(os.getenv("EMBEDDINGS_PATH", "models/embeddings.npy"))
    if embeddings_path.exists():
        args.embeddings = str(embeddings_path)
        run_module_b(args)
    else:
        logger.warning("Embeddings non disponibles, Module B ignoré.")

    # Module C
    logger.info("[4/4] Module C — Décision MDP...")
    args.gamma = float(os.getenv("MDP_GAMMA", 0.9))
    args.algorithm = "VI"
    run_module_c(args)

    logger.info("=" * 60)
    logger.info(f"Pipeline terminé. Résultats disponibles dans : {output_path}")
    logger.info("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    """Construit et retourne le parseur d'arguments CLI."""
    parser = argparse.ArgumentParser(
        prog="platevision",
        description="PlateVision — Reconnaissance automatique de plaques d'immatriculation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
  python main.py --module A --input data/processed/ --output reports/
  python main.py --module B --embeddings models/embeddings.npy
  python main.py --module C --gamma 0.9 --algorithm VI
  python main.py --pipeline full --input data/raw/video.mp4
        """,
    )

    # Groupe exclusif : module spécifique OU pipeline complet
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--module",
        choices=["A", "B", "C", "D"],
        help="Module à exécuter : A (détection+OCR), B (clustering), C (MDP)",
    )
    group.add_argument(
        "--pipeline",
        choices=["full"],
        help="Exécuter le pipeline complet end-to-end",
    )

    # Arguments communs
    parser.add_argument(
        "--input",
        type=str,
        help="Chemin vers le dossier d'images ou fichier vidéo d'entrée",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.getenv("OUTPUT_REPORTS_PATH", "reports/"),
        help="Dossier de sortie pour les résultats (défaut : reports/)",
    )

    # Arguments Module B
    parser.add_argument(
        "--embeddings",
        type=str,
        default=os.getenv("EMBEDDINGS_PATH", "models/embeddings.npy"),
        help="Chemin vers le fichier d'embeddings .npy (Module B)",
    )

    # Arguments Module C
    parser.add_argument(
        "--gamma",
        type=float,
        default=float(os.getenv("MDP_GAMMA", 0.9)),
        help="Facteur d'actualisation gamma pour le MDP (défaut : 0.9)",
    )
    parser.add_argument(
        "--algorithm",
        choices=["VI", "PI"],
        default="VI",
        help="Algorithme MDP : VI (Value Iteration) ou PI (Policy Iteration)",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    logger.info("PlateVision démarré.")

    if args.pipeline == "full":
        if not args.input:
            parser.error("--input est requis pour le pipeline complet.")
        run_full_pipeline(args)
    elif args.module == "A":
        if not args.input:
            parser.error("--input est requis pour le Module A.")
        run_module_a(args)
    elif args.module == "B":
        run_module_b(args)
    elif args.module == "C":
        run_module_c(args)
    elif args.module == "D":
        logger.info("Module D — Éthique : voir modules/module_d/ethical_analysis.md")
        logger.info("Module D — Carnet de bord : voir modules/module_d/logbook.md")
