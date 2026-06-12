"""
PlateVision — Point d'entrée principal du pipeline
MINT/DGI Cameroun — UCAC-ICAM / ULC-ICAM

Pipeline modulaire exécutable en ligne de commande (exigence §3.2.1).
Chaque module est démontrable individuellement devant le jury (exigence §5).

Usage rapide :
  python main.py --module A1 --input data/processed/ --output reports/
  python main.py --module A2 --input data/raw/ --weights models/weights/yolov8_platevision.pt
  python main.py --module A2 --detect image.jpg
  python main.py --module A  --compare
  python main.py --pipeline full
  python main.py --prepare-data --from-phase 1

Documentation complète : python main.py --help
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("platevision")


# ══════════════════════════════════════════════════════════════════════════════
# HANDLERS PAR MODULE
# ══════════════════════════════════════════════════════════════════════════════

def run_module_a1(args) -> None:
    """
    Lance le classificateur Naïves Bayes Gaussien (Module A1).

    Rôle dans PlateVision :
      Module baseline probabiliste — classifie des caractères alphanumériques
      isolés (28×28 px) à partir de features manuelles 120D extraites des
      plaques et caractères. Sert de référence pour justifier le passage
      au pipeline YOLO+OCR (A2) devant le jury MINT/DGI.

    Modes disponibles selon args :
      --input dir   : répertoire contenant features.npy / labels.npy
                      (produits par data/prepare_datasets.py phases 1-8)
      --output dir  : répertoire de sortie pour figures et métriques
                      (défaut : reports/rapport_technique/figures/)
      --predict N   : prédit le sample d'index N dans X_test, affiche top3
      --evaluate    : charge le modèle sauvegardé et réévalue sur test
    """
    from modules.module_a.naive_bayes import (
        run_full_pipeline, predict_character, load_features, load_model,
    )
    from modules.module_a import naive_bayes as nb_mod

    input_dir  = Path(args.input)  if args.input  else Path("data/processed")
    output_dir = Path(args.output) if args.output else Path("reports/rapport_technique/figures")

    nb_mod.DATA_DIR   = input_dir
    nb_mod.REPORT_DIR = output_dir

    if args.predict is not None:
        _, _, _, _, X_test, y_test, class_names = load_features(input_dir)
        model, scaler, _ = load_model()
        idx = args.predict
        if idx >= len(X_test):
            logger.error("Index %d hors bornes (taille test : %d)", idx, len(X_test))
            sys.exit(1)
        result = predict_character(X_test[idx], model=model, scaler=scaler)
        print(f"\nSample index    : {idx}")
        print(f"Vrai label      : {class_names[y_test[idx]]}")
        print(f"Prédit          : {result['predicted_class']}")
        print(f"Confiance       : {result['confidence']:.4f}")
        print("Top 3 :")
        for item in result["top3"]:
            print(f"  {item['class']} — {item['probability']:.4f}")

    elif args.evaluate:
        from modules.module_a.naive_bayes import evaluate_model
        import json
        model, scaler, _ = load_model()
        X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_features(input_dir)
        result = evaluate_model(model, scaler, X_val, y_val, X_test, y_test,
                                class_names, output_dir)
        print(json.dumps(result, indent=2))

    else:
        run_full_pipeline(data_dir=input_dir, report_dir=output_dir)


def run_module_a2(args) -> None:
    """
    Lance le pipeline YOLO + EasyOCR (Module A2).

    Rôle dans PlateVision :
      Pipeline de détection et reconnaissance bout-en-bout. Détecte la plaque
      dans l'image brute via YOLOv8n (transfer learning COCO), puis lit les
      caractères avec EasyOCR. Dépasse les limites du Naïves Bayes (A1) en
      opérant sur des images non pré-segmentées — condition obligatoire pour
      un déploiement réel aux postes de contrôle MINT/DGI.

    Modes disponibles selon args :
      --train             : entraîne YOLOv8n (epochs, batch, device)
      --detect image.jpg  : pipeline complet YOLO+OCR sur une image (démo jury)
      --evaluate          : évalue mAP@0.5, mAP@0.5:0.95, CER, WER sur test
    """
    from modules.module_a import yolo_ocr_pipeline as yolo_mod

    weights    = Path(args.weights) if args.weights else Path("models/weights/yolov8_platevision.pt")
    output_dir = Path(args.output)  if args.output  else Path("reports/rapport_technique/figures")
    yolo_mod.REPORT_DIR = output_dir

    if args.detect:
        from modules.module_a.yolo_ocr_pipeline import run_full_pipeline as ocr_pipeline
        result = ocr_pipeline(image_path=args.detect, weights_path=weights)
        print(f"\nImage analysée : {args.detect}")
        print(f"Plaques détectées : {len(result['detections'])}")
        for i, d in enumerate(result["detections"]):
            print(f"  [{i+1}] Texte : {d['plate_text'] or '(non lu)':12s} "
                  f"| YOLO: {d['yolo_conf']:.2f} "
                  f"| OCR: {d['ocr_conf']:.2f}")

    elif args.train:
        from modules.module_a.yolo_ocr_pipeline import run_yolo_training_pipeline
        run_yolo_training_pipeline(
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
        )

    elif args.evaluate:
        import json
        from modules.module_a.yolo_ocr_pipeline import evaluate_yolo, evaluate_full_pipeline
        yolo_metrics = evaluate_yolo(weights_path=weights, report_dir=output_dir)
        ocr_metrics  = evaluate_full_pipeline(weights_path=weights, report_dir=output_dir)
        print("\n[YOLO — mAP + inférence]")
        print(json.dumps(yolo_metrics, indent=2))
        print("\n[OCR — CER / WER]")
        print(json.dumps(
            {k: v for k, v in ocr_metrics.items() if k != "comparison_nb_vs_yolo_ocr"},
            indent=2,
        ))

    else:
        logger.error("Module A2 : spécifier --train, --detect IMAGE ou --evaluate")
        logger.error("Exemple : python main.py --module A2 --detect image.jpg")
        sys.exit(1)


def run_module_a_compare(args) -> None:
    """
    Lance l'évaluation comparative obligatoire A1 vs A2 (exigence §4.1).

    Rôle dans PlateVision :
      Produit le tableau comparatif exigé par le jury MINT/DGI :
      Naïves Bayes (accuracy, F1-macro, temps inférence) vs
      YOLO+OCR (mAP@0.5, mAP@0.5:0.95, CER, WER, temps bout-en-bout).
      Ce tableau apparaît dans le rapport technique (L3) et la soutenance (L5).
    """
    import modules.module_a.evaluate as eval_mod
    from modules.module_a.evaluate import run_comparative_evaluation

    output_dir = Path(args.output) if args.output else Path("reports/rapport_technique/figures")
    eval_mod.REPORT_DIR             = output_dir
    eval_mod.FIGURE_OUTPUT_PATH     = output_dir / "comparative_figure.png"
    eval_mod.REPORT_TABLE_PATH      = output_dir / "comparative_table.txt"
    eval_mod.COMPARISON_OUTPUT_PATH = output_dir / "comparative_evaluation.json"

    run_comparative_evaluation()


def run_full_pipeline(args) -> None:
    """
    Lance le pipeline complet PlateVision bout-en-bout.

    Rôle dans PlateVision :
      Enchaîne dans l'ordre :
        1. Préparation des données    (data/prepare_datasets.py phases 1-8)
        2. Module A1 — Naïves Bayes   (entraînement + évaluation)
        3. Module A2 — YOLO+OCR       (entraînement + évaluation)
        4. Évaluation comparative     (tableau jury §4.1)
      Produit tous les livrables techniques (L4) en une seule commande.
      Durée estimée : 20-40 min selon GPU disponible.
    """
    logger.info("=== Pipeline PlateVision complet — démarrage ===")

    # Étape 1 — Données
    logger.info("--- Étape 1/4 : Préparation des données ---")
    import subprocess
    result = subprocess.run(
        [sys.executable, "data/prepare_datasets.py"],
        check=False,
    )
    if result.returncode != 0:
        logger.warning("prepare_datasets.py a retourné code %d — continuation.",
                       result.returncode)

    # Étape 2 — Module A1
    logger.info("--- Étape 2/4 : Module A1 — Naïves Bayes ---")
    from modules.module_a.naive_bayes import run_full_pipeline as nb_pipeline
    nb_pipeline()

    # Étape 3 — Module A2
    logger.info("--- Étape 3/4 : Module A2 — YOLO+OCR ---")
    from modules.module_a.yolo_ocr_pipeline import run_yolo_training_pipeline
    run_yolo_training_pipeline(epochs=50, batch=16, device="auto")

    # Étape 4 — Comparaison
    logger.info("--- Étape 4/4 : Évaluation comparative ---")
    from modules.module_a.evaluate import run_comparative_evaluation
    run_comparative_evaluation()

    logger.info("=== Pipeline complet terminé. Résultats dans reports/ ===")


def run_prepare_data(args) -> None:
    """
    Lance uniquement la préparation des données (phases 1-8).

    Rôle dans PlateVision :
      Exécute data/prepare_datasets.py avec support de reprise par phase.
      Produit features.npy, labels.npy, plate_crops/, characters/,
      ocr_results.json et les images YOLO 640×640.
    """
    import subprocess
    cmd = [sys.executable, "data/prepare_datasets.py"]
    if getattr(args, "from_phase", None):
        cmd += ["--from-phase", str(args.from_phase)]
    if getattr(args, "phase_only", None):
        cmd += ["--phase-only", str(args.phase_only)]
    logger.info("Lancement : %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION DU PARSER ARGPARSE
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Construit le parser argparse principal de PlateVision."""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "PlateVision — Système de reconnaissance de plaques MINT/DGI Cameroun\n"
            "UCAC-ICAM / ULC-ICAM — PSTNAC 2023-2028"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python main.py --module A1 --input data/processed/ --output reports/
  python main.py --module A1 --predict 42
  python main.py --module A1 --evaluate
  python main.py --module A2 --train --epochs 50
  python main.py --module A2 --detect image.jpg --weights models/weights/yolov8_platevision.pt
  python main.py --module A2 --evaluate
  python main.py --module A  --compare
  python main.py --pipeline full
  python main.py --prepare-data --from-phase 5
  python main.py --prepare-data --phase-only 8
        """,
    )

    # ── Sélection du module ───────────────────────────────────────────────────
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--module",
        choices=["A1", "A2", "A"],
        metavar="MODULE",
        help="Module à exécuter : A1 (Naïves Bayes) | A2 (YOLO+OCR) | A (comparaison)",
    )
    group.add_argument(
        "--pipeline",
        choices=["full"],
        metavar="PIPELINE",
        help="Pipeline complet bout-en-bout : full",
    )
    group.add_argument(
        "--prepare-data",
        action="store_true",
        dest="prepare_data",
        help="Prépare les données (phases 1-8 de prepare_datasets.py)",
    )

    # ── Arguments partagés ────────────────────────────────────────────────────
    parser.add_argument("--input",   type=str, default=None,
                        help="Répertoire d'entrée (données)")
    parser.add_argument("--output",  type=str, default=None,
                        help="Répertoire de sortie (rapports, figures)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Chemin vers les poids YOLO (.pt)")

    # ── Module A1 ─────────────────────────────────────────────────────────────
    parser.add_argument("--predict",  type=int, default=None, metavar="N",
                        help="[A1] Prédit le sample d'index N dans X_test")
    parser.add_argument("--evaluate", action="store_true",
                        help="[A1/A2] Évalue le modèle sauvegardé sur test")

    # ── Module A2 ─────────────────────────────────────────────────────────────
    parser.add_argument("--train",  action="store_true",
                        help="[A2] Entraîne YOLOv8n")
    parser.add_argument("--epochs", type=int, default=50,
                        help="[A2] Nombre d'epochs (défaut : 50)")
    parser.add_argument("--batch",  type=int, default=16,
                        help="[A2] Taille du batch (défaut : 16)")
    parser.add_argument("--device", type=str, default="auto",
                        help="[A2] Device : auto / cpu / 0 (défaut : auto)")
    parser.add_argument("--detect", type=str, default=None, metavar="IMAGE",
                        help="[A2] Pipeline YOLO+OCR sur image (démo jury §5)")
    parser.add_argument("--conf",   type=float, default=0.45,
                        help="[A2] Seuil confiance YOLO (défaut : 0.45)")

    # ── Comparaison ───────────────────────────────────────────────────────────
    parser.add_argument("--compare", action="store_true",
                        help="[A] Tableau comparatif A1 vs A2 (exigence §4.1)")

    # ── Préparation données ───────────────────────────────────────────────────
    parser.add_argument("--from-phase", type=int, default=None, metavar="N",
                        dest="from_phase",
                        help="[prepare-data] Reprendre à la phase N (1-8)")
    parser.add_argument("--phase-only", type=int, default=None, metavar="N",
                        dest="phase_only",
                        help="[prepare-data] Exécuter uniquement la phase N (1-8)")

    return parser


# ══════════════════════════════════════════════════════════════════════════════
# ROUTAGE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Point d'entrée principal — route vers le handler du module sélectionné."""
    parser = build_parser()
    args   = parser.parse_args()

    if args.prepare_data:
        run_prepare_data(args)

    elif args.pipeline == "full":
        run_full_pipeline(args)

    elif args.module == "A1":
        run_module_a1(args)

    elif args.module == "A2":
        run_module_a2(args)

    elif args.module == "A":
        if args.compare:
            run_module_a_compare(args)
        else:
            logger.error("--module A nécessite --compare")
            logger.error("Exemple : python main.py --module A --compare")
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
