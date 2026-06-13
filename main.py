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
  python main.py --module B  --k 3
  python main.py --module B  --k 4 --force-rerun
  python main.py --module B4 --k 3
  python main.py --module B8 --seeds 0 1 42 --n-init-values 5 10 20
  python main.py --pipeline full
  python main.py --prepare-data --from-phase 1

Documentation complète : python main.py --help
"""

import argparse
import logging
import sys
from pathlib import Path


def _resolve_yolo_device(device: str) -> str:
    """
    Ultralytics n'accepte pas "auto" — traduit vers "0" (GPU) ou "cpu".
    PyTorch accepte "auto" mais pas l'argument --device de YOLO CLI/API.
    """
    if device != "auto":
        return device
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

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
            device=_resolve_yolo_device(args.device),
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
    run_yolo_training_pipeline(epochs=50, batch=16,
                               device=_resolve_yolo_device("auto"))

    # Étape 4 — Comparaison
    logger.info("--- Étape 4/5 : Évaluation comparative ---")
    from modules.module_a.evaluate import run_comparative_evaluation
    run_comparative_evaluation()

    # Étape 5 — Module B
    logger.info("--- Étape 5/5 : Module B — Clustering K-Means ---")
    run_module_b(args)

    logger.info("=== Pipeline complet terminé. Résultats dans reports/ ===")


def run_module_b(args) -> None:
    """
    Pipeline complet Module B — Clustering non supervisé (§4.2 PlateVision).

    Étapes enchaînées :
      B0 — CNN CharEmbeddingCNN (si embeddings absents)
      B1 — Extraction embeddings 256D + metadata.csv
      B3 — Justification k (elbow + silhouette)
      B4 — K-Means final + dist_centroid + confidence_level
      B5 — Visualisation PCA/t-SNE colorée par cluster
      B6 — Nommage métier + cluster_mapping.json → Module C
      B8 — Robustesse (optionnel, skip si --fast)

    §5 — Soutenance : passer --k N --force-rerun pour régénérer
    avec un k différent à la demande du jury.
    """
    import json as json_lib

    data_dir    = Path("data/processed")
    models_dir  = Path("models")
    output_base = Path(args.output) if args.output else Path("reports")
    figures_dir = output_base / "rapport_technique" / "figures"
    report_dir  = output_base / "rapport_technique"
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("MODULE B — Clustering Non Supervisé (§4.2 PlateVision)")
    print("=" * 60)

    # ── Étape B0 : CNN (seulement si embeddings absents) ─────────────────────
    emb_path = Path(args.embeddings) if args.embeddings else data_dir / "embeddings.npy"
    if not emb_path.exists() or args.force_rerun:
        print("\n[B0] Entraînement CNN CharEmbeddingCNN...")
        from modules.module_b.cnn_embeddings import run_cnn_embedding_pipeline
        run_cnn_embedding_pipeline(
            data_dir=data_dir,
            model_path=models_dir / "char_cnn.pth",
            out_dir=data_dir,
            force_retrain=args.force_rerun,
        )
    else:
        print(f"[B0] Embeddings existants détectés : {emb_path} — skip CNN")

    # ── Étape B1 : Extraction embeddings ─────────────────────────────────────
    meta_path = data_dir / "metadata.csv"
    if not meta_path.exists() or args.force_rerun:
        print("\n[B1] Extraction embeddings + metadata.csv...")
        from modules.module_b.extract_embeddings import run_embedding_pipeline
        run_embedding_pipeline(
            data_dir=data_dir,
            model_path=models_dir / "char_cnn.pth",
            out_dir=data_dir,
        )
    else:
        print(f"[B1] metadata.csv existant — skip extraction")

    # ── Étape B3 : Justification k ────────────────────────────────────────────
    k_analysis_path = report_dir / "k_analysis.txt"
    k_auto = None
    if not k_analysis_path.exists() or args.force_rerun or args.k:
        print("\n[B3] Calcul du k optimal (elbow + silhouette)...")
        from modules.module_b.clustering import run_optimal_k_pipeline
        k_result = run_optimal_k_pipeline(
            data_dir=data_dir,
            figures_dir=figures_dir,
        )
        k_auto = k_result["recommended"]
    else:
        print(f"[B3] k_analysis.txt existant — skip calcul k")

    # Résolution de k : --k explicite > cluster_mapping.json > k_auto
    k_final = args.k
    if k_final is None:
        mapping_path = data_dir / "cluster_mapping.json"
        if mapping_path.exists():
            with open(mapping_path) as f:
                k_final = json_lib.load(f).get("k")
            print(f"[B3] k lu depuis cluster_mapping.json : k={k_final}")
        elif k_auto is not None:
            k_final = k_auto
        else:
            raise ValueError(
                "k non déterminé. Passe --k explicitement ou exécute B3 d'abord."
            )

    print(f"\n→ k utilisé : {k_final}")
    if args.k and args.k != k_final:
        print(f"  [Soutenance §5] k forcé à {args.k} (jury override)")

    # ── Étape B4 : K-Means final ──────────────────────────────────────────────
    kmeans_path = models_dir / "kmeans_model.pkl"
    if not kmeans_path.exists() or args.force_rerun:
        print(f"\n[B4] K-Means final k={k_final}...")
        from modules.module_b.kmeans_fit import run_kmeans_pipeline
        run_kmeans_pipeline(
            data_dir=data_dir,
            models_dir=models_dir,
            figures_dir=figures_dir,
            k=k_final,
        )
    else:
        print(f"[B4] kmeans_model.pkl existant — skip (--force-rerun pour relancer)")

    # ── Étape B5 : Visualisation ──────────────────────────────────────────────
    print(f"\n[B5] Visualisation PCA/t-SNE (no_tsne={args.no_tsne})...")
    from modules.module_b.visualization import run_visualization_pipeline
    run_visualization_pipeline(
        data_dir=data_dir,
        models_dir=models_dir,
        figures_dir=figures_dir,
        run_tsne=not args.no_tsne,
    )

    # ── Étape B6 : Interprétation métier ─────────────────────────────────────
    print(f"\n[B6] Interprétation métier des clusters...")
    manual = None
    if args.manual_mapping:
        manual = json_lib.loads(args.manual_mapping)
    from modules.module_b.interpret_clusters import run_interpretation_pipeline
    run_interpretation_pipeline(
        data_dir=data_dir,
        models_dir=models_dir,
        figures_dir=figures_dir,
        report_dir=report_dir,
        manual_mapping=manual,
    )

    print("\n" + "=" * 60)
    print("MODULE B TERMINÉ")
    print("=" * 60)
    print(f"k utilisé         : {k_final}")
    print(f"Figures           : {figures_dir}/")
    print(f"Rapport LaTeX     : {report_dir}/module_b_clusters.tex")
    print(f"Mapping → MDP     : {data_dir}/cluster_mapping.json")
    print(f"\nCommande soutenance §5 (changer k en direct) :")
    print(f"  python main.py --module B --k 4 --force-rerun")
    print(f"\nProchain module :")
    print(f"  python main.py --module C")


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
Exemples Module A :
  python main.py --module A1 --input data/processed/ --output reports/
  python main.py --module A1 --predict 42
  python main.py --module A1 --evaluate
  python main.py --module A2 --train --epochs 50
  python main.py --module A2 --detect image.jpg --weights models/weights/yolov8_platevision.pt
  python main.py --module A2 --evaluate
  python main.py --module A  --compare

Exemples Module B (§4.2) :
  python main.py --module B  --k 3
  python main.py --module B  --embeddings data/processed/embeddings.npy --k 3 --output reports/
  python main.py --module B  --k 4 --force-rerun
  python main.py --module B4 --k 3
  python main.py --module B8 --seeds 0 1 42 --n-init-values 5 10 20

Pipelines :
  python main.py --pipeline full
  python main.py --prepare-data --from-phase 5
  python main.py --prepare-data --phase-only 8
        """,
    )

    # ── Sélection du module ───────────────────────────────────────────────────
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--module",
        choices=["A1", "A2", "A",
                 "B", "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B8"],
        metavar="MODULE",
        help=(
            "Module à exécuter : "
            "A1 (Naïves Bayes) | A2 (YOLO+OCR) | A (comparaison) | "
            "B (pipeline complet clustering) | B0-B8 (étapes isolées)"
        ),
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

    # ── Module B ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--embeddings", type=str, default=None,
        help="[B] Chemin vers embeddings.npy (défaut : data/processed/embeddings.npy)",
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="[B] Nombre de clusters K-Means. Si absent : déterminé automatiquement.",
    )
    parser.add_argument(
        "--force-rerun", action="store_true", dest="force_rerun",
        help="[B] Force la régénération même si les fichiers existent déjà "
             "(utile soutenance §5 pour changer k en direct)",
    )
    parser.add_argument(
        "--no-tsne", action="store_true", dest="no_tsne",
        help="[B5] Skip t-SNE dans la visualisation (long si N > 5000)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2, 7, 13, 21, 42],
        help="[B8] Graines aléatoires pour test robustesse",
    )
    parser.add_argument(
        "--n-init-values", nargs="+", type=int, default=[1, 5, 10, 20, 50],
        dest="n_init_values",
        help="[B8] Valeurs n_init à tester pour robustesse",
    )
    parser.add_argument(
        "--manual-mapping", type=str, default=None, dest="manual_mapping",
        help='[B6] Mapping cluster→procédure JSON ex: \'{"0":{"action_mdp":"PASS"}}\' '
             "(soutenance §5)",
    )

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

    elif args.module == "B":
        run_module_b(args)

    elif args.module == "B0":
        from modules.module_b.cnn_embeddings import run_cnn_embedding_pipeline
        run_cnn_embedding_pipeline(
            force_retrain=args.force_rerun,
        )

    elif args.module == "B1":
        from modules.module_b.extract_embeddings import run_embedding_pipeline
        run_embedding_pipeline()

    elif args.module in ("B2", "B3"):
        from modules.module_b.clustering import run_optimal_k_pipeline
        run_optimal_k_pipeline(
            k_min=2,
            k_max=10,
            figures_dir=Path(args.output or "reports") / "rapport_technique" / "figures",
        )

    elif args.module == "B4":
        from modules.module_b.kmeans_fit import run_kmeans_pipeline
        run_kmeans_pipeline(
            k=args.k,
            figures_dir=Path(args.output or "reports") / "rapport_technique" / "figures",
        )

    elif args.module == "B5":
        from modules.module_b.visualization import run_visualization_pipeline
        run_visualization_pipeline(
            run_tsne=not args.no_tsne,
            figures_dir=Path(args.output or "reports") / "rapport_technique" / "figures",
        )

    elif args.module == "B6":
        import json as _json
        _manual = None
        if args.manual_mapping:
            _manual = _json.loads(args.manual_mapping)
        from modules.module_b.interpret_clusters import run_interpretation_pipeline
        run_interpretation_pipeline(
            figures_dir=Path(args.output or "reports") / "rapport_technique" / "figures",
            report_dir=Path(args.output or "reports") / "rapport_technique",
            manual_mapping=_manual,
        )

    elif args.module == "B8":
        from modules.module_b.robustness import run_robustness_pipeline
        run_robustness_pipeline(
            k=args.k,
            seeds=args.seeds,
            n_init_values=args.n_init_values,
            figures_dir=Path(args.output or "reports") / "rapport_technique" / "figures",
            report_dir=Path(args.output or "reports") / "rapport_technique",
        )


if __name__ == "__main__":
    main()
