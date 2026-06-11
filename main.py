"""
PlateVision — Point d'entrée principal du pipeline CLI.

Ce module orchestre l'exécution des modules A, B, C et du pipeline complet
via une interface en ligne de commande. Il charge la configuration depuis
le fichier .env et délègue l'exécution aux modules correspondants.

Usage :
    python main.py --module A --input data/processed/ --output reports/
    python main.py --module B --embeddings models/embeddings.npy
    python main.py --module C --gamma 0.9 --algorithm VI
    python main.py --pipeline full --input data/raw/video.mp4
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Chargement des variables d'environnement depuis .env
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("platevision")


def run_module_a(args: argparse.Namespace) -> None:
    """Exécute le Module A : détection et reconnaissance de plaques."""
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("Chemin d'entrée introuvable : %s", input_path)
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    submodule = getattr(args, "submodule", "A1").upper()

    if submodule == "A1":
        logger.info("Lancement du Module A1 — Classifieur Naïves Bayes")
        from modules.module_a.naive_bayes import run_naive_bayes_pipeline
        run_naive_bayes_pipeline(input_path, output_path)

    elif submodule == "A2":
        logger.info("Lancement du Module A2 — Pipeline YOLO + OCR")
        from modules.module_a.yolo_ocr_pipeline import run_yolo_ocr_pipeline
        run_yolo_ocr_pipeline(input_path, output_path)

    else:
        logger.error("Sous-module A inconnu : %s (options : A1, A2)", submodule)
        sys.exit(1)

    logger.info("Module A terminé. Résultats dans : %s", output_path)


def run_module_b(args: argparse.Namespace) -> None:
    """Exécute le Module B : clustering non supervisé sur embeddings CNN."""
    embeddings_path = Path(args.embeddings)

    if not embeddings_path.exists():
        logger.error("Fichier embeddings introuvable : %s", embeddings_path)
        sys.exit(1)

    logger.info("Lancement du Module B — K-Means sur embeddings CNN")
    from modules.module_b.clustering import run_clustering_pipeline
    run_clustering_pipeline(embeddings_path)

    logger.info("Module B terminé.")


def run_module_c(args: argparse.Namespace) -> None:
    """Exécute le Module C : résolution MDP par Value Iteration ou Policy Iteration."""
    gamma = float(args.gamma)
    algorithm = args.algorithm.upper()

    if not (0.0 < gamma < 1.0):
        logger.error("Le facteur gamma doit être dans ]0, 1[. Valeur reçue : %s", gamma)
        sys.exit(1)

    if algorithm == "VI":
        logger.info("Lancement Module C — Value Iteration (γ=%.2f)", gamma)
        from modules.module_c.value_iteration import run_value_iteration
        run_value_iteration(gamma=gamma)

    elif algorithm == "PI":
        logger.info("Lancement Module C — Policy Iteration (γ=%.2f)", gamma)
        from modules.module_c.policy_iteration import run_policy_iteration
        run_policy_iteration(gamma=gamma)

    elif algorithm == "COMPARE":
        logger.info("Lancement Module C — Comparaison VI vs PI (γ=%.2f)", gamma)
        from modules.module_c.compare_vi_pi import run_comparison
        run_comparison(gamma=gamma)

    else:
        logger.error("Algorithme inconnu : %s (options : VI, PI, COMPARE)", algorithm)
        sys.exit(1)

    logger.info("Module C terminé.")


def run_full_pipeline(args: argparse.Namespace) -> None:
    """Exécute le pipeline complet : prétraitement → détection → OCR → clustering → décision."""
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error("Fichier/dossier d'entrée introuvable : %s", input_path)
        sys.exit(1)

    logger.info("=== Démarrage du pipeline complet PlateVision ===")
    logger.info("Source : %s", input_path)

    # Étape 1 — Prétraitement
    logger.info("[1/4] Prétraitement des images...")
    from preprocessing.normalization import normalize_images
    processed_images = normalize_images(input_path)

    # Étape 2 — Détection et OCR
    logger.info("[2/4] Détection des plaques et OCR...")
    from modules.module_a.yolo_ocr_pipeline import run_yolo_ocr_pipeline
    detections = run_yolo_ocr_pipeline(processed_images, output_path=Path("reports/"))

    # Étape 3 — Clustering
    logger.info("[3/4] Analyse par clustering...")
    from modules.module_b.clustering import run_clustering_pipeline
    run_clustering_pipeline(Path("models/embeddings.npy"))

    # Étape 4 — Décision MDP
    logger.info("[4/4] Prise de décision MDP...")
    from modules.module_c.value_iteration import run_value_iteration
    run_value_iteration(gamma=0.9)

    logger.info("=== Pipeline complet terminé ===")


def build_parser() -> argparse.ArgumentParser:
    """Construit le parseur d'arguments CLI."""
    parser = argparse.ArgumentParser(
        prog="platevision",
        description="PlateVision — Système de reconnaissance automatique de plaques (MINT/DGI Cameroun)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
  python main.py --module A --submodule A1 --input data/processed/ --output reports/
  python main.py --module A --submodule A2 --input data/processed/ --output reports/
  python main.py --module B --embeddings models/embeddings.npy
  python main.py --module C --gamma 0.9 --algorithm VI
  python main.py --module C --gamma 0.95 --algorithm COMPARE
  python main.py --pipeline full --input data/raw/video.mp4
        """,
    )

    parser.add_argument(
        "--module",
        choices=["A", "B", "C"],
        help="Module IA à exécuter (A, B ou C)",
    )
    parser.add_argument(
        "--submodule",
        choices=["A1", "A2"],
        default="A2",
        help="Sous-module A à exécuter : A1 (Naïves Bayes) ou A2 (YOLO+OCR) [défaut: A2]",
    )
    parser.add_argument(
        "--pipeline",
        choices=["full"],
        help="Exécuter le pipeline complet de bout en bout",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Chemin vers le dossier d'images ou fichier vidéo source",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/",
        help="Dossier de sortie pour les résultats [défaut: reports/]",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default="models/embeddings.npy",
        help="Chemin vers le fichier d'embeddings .npy pour le Module B",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        help="Facteur d'actualisation γ pour le MDP [défaut: 0.9]",
    )
    parser.add_argument(
        "--algorithm",
        choices=["VI", "PI", "COMPARE"],
        default="VI",
        help="Algorithme MDP : VI (Value Iteration), PI (Policy Iteration), COMPARE [défaut: VI]",
    )

    return parser


def main() -> None:
    """Point d'entrée principal."""
    parser = build_parser()
    args = parser.parse_args()

    if args.pipeline == "full":
        run_full_pipeline(args)
    elif args.module == "A":
        run_module_a(args)
    elif args.module == "B":
        run_module_b(args)
    elif args.module == "C":
        run_module_c(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
