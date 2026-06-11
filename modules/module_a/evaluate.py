"""
Module A — Évaluation des performances de détection et de reconnaissance.

Rôle dans le pipeline PlateVision :
    Calcule les métriques standard de détection (mAP) et de reconnaissance
    de caractères (CER, WER) pour comparer les approches A1 (Naïves Bayes)
    et A2 (YOLO + OCR).

Responsables : Étudiants 1 & 2
"""

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


def compute_iou(box_pred: tuple, box_gt: tuple) -> float:
    """
    Calcule l'Intersection over Union (IoU) entre deux bounding boxes.

    Args:
        box_pred: Boîte prédite (x1, y1, x2, y2)
        box_gt:   Boîte vérité terrain (x1, y1, x2, y2)

    Returns:
        Score IoU entre 0.0 et 1.0
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
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
) -> float:
    """
    Calcule l'Average Precision (AP) pour une classe donnée.

    Args:
        predictions: Liste de dicts {'bbox', 'confidence', 'image_id'}
        ground_truths: Liste de dicts {'bbox', 'image_id'}
        iou_threshold: Seuil IoU pour considérer une détection comme correcte

    Returns:
        Score AP entre 0.0 et 1.0
    """
    # Tri par confiance décroissante
    predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)

    gt_by_image = defaultdict(list)
    for gt in ground_truths:
        gt_by_image[gt["image_id"]].append({"bbox": gt["bbox"], "matched": False})

    tp_list = []
    fp_list = []

    for pred in predictions:
        img_id = pred["image_id"]
        gts_for_image = gt_by_image[img_id]
        best_iou = 0.0
        best_gt_idx = -1

        for i, gt in enumerate(gts_for_image):
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            if not gts_for_image[best_gt_idx]["matched"]:
                tp_list.append(1)
                fp_list.append(0)
                gts_for_image[best_gt_idx]["matched"] = True
            else:
                tp_list.append(0)
                fp_list.append(1)
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)
    total_gt = len(ground_truths)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)
    recall = tp_cumsum / (total_gt + 1e-9)

    # Calcul de l'aire sous la courbe precision-recall (interpolation)
    ap = float(np.trapz(precision, recall))
    return ap


def compute_map(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
) -> float:
    """
    Calcule le mean Average Precision (mAP) sur toutes les classes.

    Returns:
        Score mAP entre 0.0 et 1.0
    """
    classes = set(p["class"] for p in predictions) | set(g["class"] for g in ground_truths)
    ap_scores = []

    for cls in classes:
        cls_preds = [p for p in predictions if p["class"] == cls]
        cls_gts = [g for g in ground_truths if g["class"] == cls]
        ap = compute_average_precision(cls_preds, cls_gts, iou_threshold)
        ap_scores.append(ap)
        logger.info("AP classe '%s' : %.4f", cls, ap)

    map_score = float(np.mean(ap_scores)) if ap_scores else 0.0
    logger.info("mAP@%.1f : %.4f", iou_threshold, map_score)
    return map_score


def compute_cer(predicted_text: str, ground_truth_text: str) -> float:
    """
    Calcule le Character Error Rate (CER).

    CER = (insertions + suppressions + substitutions) / nb_caractères_référence

    Args:
        predicted_text: Texte prédit par OCR
        ground_truth_text: Texte de référence (annotation)

    Returns:
        Score CER (0.0 = parfait, 1.0 = complètement faux)
    """
    pred = predicted_text.upper().replace(" ", "")
    ref = ground_truth_text.upper().replace(" ", "")

    if len(ref) == 0:
        return 0.0 if len(pred) == 0 else 1.0

    # Distance de Levenshtein sur les caractères
    distance = _levenshtein(pred, ref)
    return distance / len(ref)


def compute_wer(predicted_text: str, ground_truth_text: str) -> float:
    """
    Calcule le Word Error Rate (WER).

    WER = (insertions + suppressions + substitutions) / nb_mots_référence

    Returns:
        Score WER (0.0 = parfait)
    """
    pred_words = predicted_text.upper().split()
    ref_words = ground_truth_text.upper().split()

    if len(ref_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0

    distance = _levenshtein(pred_words, ref_words)
    return distance / len(ref_words)


def _levenshtein(seq1: list | str, seq2: list | str) -> int:
    """Calcule la distance de Levenshtein entre deux séquences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def evaluate_ocr_batch(predictions: list[dict], ground_truths: list[dict]) -> dict:
    """
    Évalue les performances OCR sur un ensemble d'annotations.

    Args:
        predictions: Liste de dicts {'image_id', 'plate_text'}
        ground_truths: Liste de dicts {'image_id', 'plate_text'}

    Returns:
        Dictionnaire avec CER moyen, WER moyen, et exact match rate.
    """
    gt_dict = {g["image_id"]: g["plate_text"] for g in ground_truths}

    cer_scores, wer_scores, exact_matches = [], [], []

    for pred in predictions:
        img_id = pred["image_id"]
        if img_id not in gt_dict:
            continue
        ref = gt_dict[img_id]
        prd = pred["plate_text"]

        cer_scores.append(compute_cer(prd, ref))
        wer_scores.append(compute_wer(prd, ref))
        exact_matches.append(1 if prd.upper() == ref.upper() else 0)

    results = {
        "mean_cer": float(np.mean(cer_scores)) if cer_scores else 0.0,
        "mean_wer": float(np.mean(wer_scores)) if wer_scores else 0.0,
        "exact_match_rate": float(np.mean(exact_matches)) if exact_matches else 0.0,
        "n_samples": len(cer_scores),
    }

    logger.info("CER moyen : %.4f", results["mean_cer"])
    logger.info("WER moyen : %.4f", results["mean_wer"])
    logger.info("Exact match rate : %.4f", results["exact_match_rate"])

    return results
