"""Tests unitaires pour le Module A2 — Pipeline YOLO + OCR."""

import numpy as np
import pytest

from modules.module_a.evaluate import compute_iou, compute_average_precision, evaluate_ocr_batch


class TestIoU:
    """Tests du calcul IoU."""

    def test_perfect_overlap(self):
        box = (10, 10, 50, 50)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        box_a = (0, 0, 10, 10)
        box_b = (20, 20, 30, 30)
        assert compute_iou(box_a, box_b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        box_a = (0, 0, 10, 10)
        box_b = (5, 5, 15, 15)
        # Intersection : (5,5,10,10) = 25, union = 100+100-25=175
        iou = compute_iou(box_a, box_b)
        assert pytest.approx(iou, abs=1e-4) == 25 / 175

    def test_contained_box(self):
        outer = (0, 0, 20, 20)
        inner = (5, 5, 15, 15)
        iou = compute_iou(inner, outer)
        # Intersection = inner area = 100, union = outer area = 400
        assert pytest.approx(iou, abs=1e-4) == 100 / 400

    def test_zero_area_box(self):
        box_a = (5, 5, 5, 5)
        box_b = (0, 0, 10, 10)
        assert compute_iou(box_a, box_b) == pytest.approx(0.0)


class TestAveragePrecision:
    """Tests du calcul AP."""

    def test_perfect_detector(self):
        """Toutes les prédictions sont correctes → AP proche de 1."""
        preds = [
            {"image_id": "img1", "bbox": (10, 10, 50, 50), "confidence": 0.99},
            {"image_id": "img2", "bbox": (10, 10, 50, 50), "confidence": 0.95},
        ]
        gts = [
            {"image_id": "img1", "bbox": (10, 10, 50, 50)},
            {"image_id": "img2", "bbox": (10, 10, 50, 50)},
        ]
        ap = compute_average_precision(preds, gts, iou_threshold=0.5)
        assert ap == pytest.approx(1.0, abs=0.01)

    def test_no_predictions(self):
        gts = [{"image_id": "img1", "bbox": (10, 10, 50, 50)}]
        ap = compute_average_precision([], gts)
        assert ap == pytest.approx(0.0)

    def test_no_ground_truth(self):
        preds = [{"image_id": "img1", "bbox": (10, 10, 50, 50), "confidence": 0.9}]
        ap = compute_average_precision(preds, [])
        assert ap == pytest.approx(0.0)


class TestOCRBatchEvaluation:
    """Tests de l'évaluation OCR en batch."""

    def test_perfect_batch(self):
        preds = [
            {"image_id": "img1", "plate_text": "LT 1234 A"},
            {"image_id": "img2", "plate_text": "CE 5678 B"},
        ]
        gts = [
            {"image_id": "img1", "plate_text": "LT 1234 A"},
            {"image_id": "img2", "plate_text": "CE 5678 B"},
        ]
        results = evaluate_ocr_batch(preds, gts)
        assert results["mean_cer"] == pytest.approx(0.0)
        assert results["exact_match_rate"] == pytest.approx(1.0)

    def test_empty_batch(self):
        results = evaluate_ocr_batch([], [])
        assert results["n_samples"] == 0

    def test_missing_prediction(self):
        """Si l'image n'est pas dans les prédictions, elle est ignorée."""
        preds = [{"image_id": "img1", "plate_text": "LT 1234 A"}]
        gts = [
            {"image_id": "img1", "plate_text": "LT 1234 A"},
            {"image_id": "img2", "plate_text": "CE 5678 B"},
        ]
        results = evaluate_ocr_batch(preds, gts)
        assert results["n_samples"] == 1

    def test_results_keys(self):
        preds = [{"image_id": "img1", "plate_text": "LT 1234 A"}]
        gts = [{"image_id": "img1", "plate_text": "LT 0000 A"}]
        results = evaluate_ocr_batch(preds, gts)
        assert all(k in results for k in ["mean_cer", "mean_wer", "exact_match_rate", "n_samples"])
