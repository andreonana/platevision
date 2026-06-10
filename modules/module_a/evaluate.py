"""
PlateVision — Module A : Métriques d'évaluation.

Ce module calcule les métriques d'évaluation du système de reconnaissance :
- mAP (mean Average Precision) pour la détection YOLO
- CER (Character Error Rate) pour la précision OCR au niveau caractère
- WER (Word Error Rate) pour la précision OCR au niveau mot (plaque entière)

Rôle dans le pipeline : évaluation quantitative des performances du Module A.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Calcule le Character Error Rate (CER) entre la vérité terrain et la prédiction.

    CER = (Insertions + Suppressions + Substitutions) / Longueur de la référence

    Args:
        reference: Texte de référence (vérité terrain)
        hypothesis: Texte prédit par l'OCR

    Returns:
        CER entre 0.0 (parfait) et 1.0+ (erreurs nombreuses)
    """
    ref = list(reference.upper().replace(" ", ""))
    hyp = list(hypothesis.upper().replace(" ", ""))

    if len(ref) == 0:
        return 1.0 if len(hyp) > 0 else 0.0

    # Algorithme de distance de Levenshtein
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,       # suppression
                d[i][j - 1] + 1,       # insertion
                d[i - 1][j - 1] + cost, # substitution
            )

    return d[len(ref)][len(hyp)] / len(ref)


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Calcule le Word Error Rate (WER) pour une plaque complète.

    Pour les plaques, chaque plaque est traitée comme un seul "mot".
    WER = 0 si la plaque est lue correctement, 1 sinon.

    Args:
        reference: Plaque de référence
        hypothesis: Plaque prédite

    Returns:
        0.0 si la lecture est exacte, 1.0 sinon
    """
    ref_clean = reference.upper().replace(" ", "")
    hyp_clean = hypothesis.upper().replace(" ", "")
    return 0.0 if ref_clean == hyp_clean else 1.0


def compute_iou(box_pred: tuple, box_gt: tuple) -> float:
    """
    Calcule l'Intersection over Union entre deux bounding boxes.

    Args:
        box_pred: Boîte prédite (x1, y1, x2, y2)
        box_gt: Boîte vérité terrain (x1, y1, x2, y2)

    Returns:
        Valeur IoU entre 0.0 et 1.0
    """
    x1 = max(box_pred[0], box_gt[0])
    y1 = max(box_pred[1], box_gt[1])
    x2 = min(box_pred[2], box_gt[2])
    y2 = min(box_pred[3], box_gt[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_pred = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
    area_gt = (box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1])
    union = area_pred + area_gt - intersection

    return intersection / union if union > 0 else 0.0


def compute_average_precision(
    detections: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
) -> float:
    """
    Calcule l'Average Precision à un seuil IoU donné.

    Args:
        detections: Liste de détections triées par confiance décroissante
        ground_truths: Liste des vérités terrain
        iou_threshold: Seuil IoU pour considérer une détection correcte

    Returns:
        Valeur de l'Average Precision
    """
    if not ground_truths:
        return 0.0

    matched_gt = set()
    tp = []
    fp = []

    # Tri par confiance décroissante
    detections_sorted = sorted(detections, key=lambda x: x["confidence"], reverse=True)

    for det in detections_sorted:
        best_iou = 0.0
        best_gt_idx = -1

        for idx, gt in enumerate(ground_truths):
            if idx in matched_gt:
                continue
            iou = compute_iou(det["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_threshold:
            tp.append(1)
            fp.append(0)
            matched_gt.add(best_gt_idx)
        else:
            tp.append(0)
            fp.append(1)

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    n_gt = len(ground_truths)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recall = tp_cumsum / (n_gt + 1e-8)

    # Interpolation de la courbe precision-recall
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p_interp = precision[recall >= t].max() if (recall >= t).any() else 0.0
        ap += p_interp / 11.0

    return float(ap)


def compute_metrics(
    nb_results: Dict[str, Any],
    ocr_results: List[Dict[str, Any]],
    output_path: Path,
) -> Dict[str, float]:
    """
    Calcule et affiche l'ensemble des métriques d'évaluation du Module A.

    Args:
        nb_results: Résultats du classifieur Naïves Bayes
        ocr_results: Résultats du pipeline YOLO+OCR
        output_path: Dossier de sauvegarde du rapport de métriques

    Returns:
        Dictionnaire des métriques globales
    """
    logger.info("=== Métriques Module A ===")

    # --- Métriques Naïves Bayes ---
    nb_accuracy = nb_results.get("accuracy", 0.0)
    logger.info(f"Accuracy Naïves Bayes : {nb_accuracy:.4f}")

    # --- Métriques OCR (CER et WER moyens) ---
    cer_list = []
    wer_list = []

    for result in ocr_results:
        for plate in result.get("plates", []):
            # ground_truth doit être fourni dans les annotations
            gt_text = plate.get("ground_truth", "")
            pred_text = plate.get("text", "")

            if gt_text:
                cer_list.append(compute_cer(gt_text, pred_text))
                wer_list.append(compute_wer(gt_text, pred_text))

    mean_cer = float(np.mean(cer_list)) if cer_list else float("nan")
    mean_wer = float(np.mean(wer_list)) if wer_list else float("nan")

    logger.info(f"CER moyen (OCR) : {mean_cer:.4f}")
    logger.info(f"WER moyen (OCR) : {mean_wer:.4f}")

    metrics = {
        "nb_accuracy": nb_accuracy,
        "mean_cer": mean_cer,
        "mean_wer": mean_wer,
    }

    # Sauvegarde des métriques dans un fichier texte
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_file = output_path / "module_a_metrics.txt"
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write("PlateVision — Métriques Module A\n")
        f.write("=" * 40 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    logger.info(f"Métriques sauvegardées dans {metrics_file}")
    return metrics
