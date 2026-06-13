"""
Tests unitaires — modules/module_a/yolo_ocr_pipeline.py (Partie 1)
Aucune dépendance sur Ultralytics réel ni GPU — mocks ciblés.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from modules.module_a.yolo_ocr_pipeline import (
    compute_cer,
    compute_wer,
    detect_plate,
    deskew_plate_crop,
    evaluate_full_pipeline,
    evaluate_yolo,
    plot_training_curves,
    postprocess_ocr_text,
    prepare_yaml,
    read_plate_ocr,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. prepare_yaml
# ══════════════════════════════════════════════════════════════════════════════

def test_prepare_yaml_creates_file(tmp_path):
    """YAML absent → fichier créé avec nc: 1 et plaque_immatriculation."""
    yaml_path = tmp_path / "yolov8_test.yaml"
    result = prepare_yaml(yaml_path=yaml_path)

    assert result == yaml_path
    assert yaml_path.exists()
    content = yaml_path.read_text(encoding="utf-8")
    assert "nc: 1" in content
    assert "plaque_immatriculation" in content


def test_prepare_yaml_path_is_absolute(tmp_path):
    """Le champ 'path' dans le YAML généré est un chemin absolu."""
    import yaml

    yaml_path = tmp_path / "yolov8_abs.yaml"
    prepare_yaml(yaml_path=yaml_path)

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert Path(cfg["path"]).is_absolute(), (
        f"'path' devrait être absolu, obtenu : {cfg['path']}"
    )


def test_prepare_yaml_idempotent(tmp_path):
    """Appeler prepare_yaml deux fois ne corrompt pas le fichier."""
    import yaml

    yaml_path = tmp_path / "yolov8_idem.yaml"
    prepare_yaml(yaml_path=yaml_path)
    prepare_yaml(yaml_path=yaml_path)  # deuxième appel

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert cfg["nc"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# 2. evaluate_yolo — mock Ultralytics
# ══════════════════════════════════════════════════════════════════════════════

def _make_mock_val_results():
    """Construit un objet simulant ultralytics DetMetrics."""
    box = MagicMock()
    box.map50 = 0.87
    box.map   = 0.62
    box.mp    = 0.91
    box.mr    = 0.83
    val_res = MagicMock()
    val_res.box = box
    return val_res


def test_evaluate_yolo_output_keys(tmp_path):
    """evaluate_yolo retourne les 7 clés obligatoires du cahier des charges."""
    fake_weights = tmp_path / "best.pt"
    fake_weights.write_bytes(b"fake")
    yaml_path = tmp_path / "cfg.yaml"

    # YAML minimal valide
    yaml_path.write_text(
        "path: /tmp\ntrain: t\nval: v\ntest: t\nnc: 1\nnames: {0: plaque_immatriculation}\n",
        encoding="utf-8",
    )

    mock_model = MagicMock()
    mock_model.val.return_value = _make_mock_val_results()
    # predict retourne immédiatement (temps d'inférence ~0)
    mock_model.predict.return_value = [MagicMock(boxes=[])]

    with patch("ultralytics.YOLO", return_value=mock_model):
        metrics = evaluate_yolo(
            weights_path=fake_weights,
            yaml_path=yaml_path,
            report_dir=tmp_path,
        )

    expected_keys = {
        "map50", "map50_95", "precision", "recall",
        "inference_ms_mean", "inference_ms_std", "realtime_ok",
    }
    assert set(metrics.keys()) == expected_keys
    assert isinstance(metrics["realtime_ok"], bool)
    assert 0.0 <= metrics["map50"] <= 1.0


def test_evaluate_yolo_saves_json(tmp_path):
    """evaluate_yolo crée yolo_metrics.json dans report_dir."""
    fake_weights = tmp_path / "best.pt"
    fake_weights.write_bytes(b"fake")
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "path: /tmp\ntrain: t\nval: v\ntest: t\nnc: 1\nnames: {0: p}\n",
        encoding="utf-8",
    )

    mock_model = MagicMock()
    mock_model.val.return_value = _make_mock_val_results()
    mock_model.predict.return_value = [MagicMock(boxes=[])]

    with patch("ultralytics.YOLO", return_value=mock_model):
        evaluate_yolo(weights_path=fake_weights, yaml_path=yaml_path, report_dir=tmp_path)

    assert (tmp_path / "yolo_metrics.json").exists()


# ══════════════════════════════════════════════════════════════════════════════
# 3. detect_plate
# ══════════════════════════════════════════════════════════════════════════════

def test_detect_plate_missing_weights(tmp_path):
    """Poids inexistants → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        detect_plate("irrelevant.jpg", weights_path=tmp_path / "noexist.pt")


def test_detect_plate_output_structure(tmp_path):
    """Image synthétique 640×640 → résultat contient n_detections et detections."""
    # Créer une fausse image sur disque
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img_path = tmp_path / "test_plate.jpg"
    cv2.imwrite(str(img_path), img)

    fake_weights = tmp_path / "best.pt"
    fake_weights.write_bytes(b"fake")

    # Simuler une détection avec une boîte
    fake_box = MagicMock()
    fake_box.xyxy = [MagicMock()]
    fake_box.xyxy[0].tolist.return_value = [100.0, 200.0, 400.0, 350.0]
    fake_box.conf = [MagicMock()]
    fake_box.conf[0].__float__ = lambda s: 0.92

    fake_result = MagicMock()
    fake_result.boxes = [fake_box]
    fake_result.plot.return_value = img.copy()

    mock_model = MagicMock()
    mock_model.predict.return_value = [fake_result]

    with patch("ultralytics.YOLO", return_value=mock_model):
        result = detect_plate(
            str(img_path),
            weights_path=fake_weights,
            save_annotated=False,
        )

    assert "n_detections" in result
    assert "detections" in result
    assert "inference_ms" in result
    assert isinstance(result["detections"], list)


# ══════════════════════════════════════════════════════════════════════════════
# 4. plot_training_curves
# ══════════════════════════════════════════════════════════════════════════════

def test_plot_training_curves_missing_csv(tmp_path, caplog):
    """results.csv absent → warning loggé, aucune exception."""
    import logging

    with caplog.at_level(logging.WARNING, logger="modules.module_a.yolo_ocr_pipeline"):
        plot_training_curves(run_dir=tmp_path / "norun", report_dir=tmp_path)

    assert any("results.csv" in rec.message for rec in caplog.records)


def test_plot_training_curves_creates_png(tmp_path):
    """results.csv présent → yolo_training_curves.png créé."""
    import pandas as pd

    # CSV minimal mimant le format Ultralytics
    df = pd.DataFrame({
        "epoch":                   list(range(1, 6)),
        "train/box_loss":          [0.9, 0.7, 0.6, 0.5, 0.4],
        "val/box_loss":            [1.0, 0.8, 0.7, 0.6, 0.5],
        "train/cls_loss":          [0.5, 0.4, 0.3, 0.25, 0.2],
        "val/cls_loss":            [0.6, 0.5, 0.4, 0.35, 0.3],
        "metrics/mAP50(B)":        [0.5, 0.6, 0.7, 0.75, 0.8],
        "metrics/mAP50-95(B)":     [0.3, 0.4, 0.5, 0.55, 0.6],
    })
    run_dir = tmp_path / "platevision"
    run_dir.mkdir()
    df.to_csv(run_dir / "results.csv", index=False)

    plot_training_curves(run_dir=run_dir, report_dir=tmp_path)

    assert (tmp_path / "yolo_training_curves.png").exists()


# ══════════════════════════════════════════════════════════════════════════════
# Partie 2 — OCR
# ══════════════════════════════════════════════════════════════════════════════

# ── postprocess_ocr_text ─────────────────────────────────────────────────────

def test_postprocess_removes_spaces():
    assert postprocess_ocr_text(" AB 123 cd ") == "AB123CD"


def test_postprocess_removes_special_chars():
    assert postprocess_ocr_text("AB-12.3_CD") == "AB123CD"


def test_postprocess_empty_input():
    assert postprocess_ocr_text("") == ""


# ── compute_cer ──────────────────────────────────────────────────────────────

def test_compute_cer_perfect():
    assert compute_cer("AB123", "AB123") == 0.0


def test_compute_cer_one_error():
    result = compute_cer("AB123", "AB124")
    assert abs(result - 0.2) < 1e-9


def test_compute_cer_empty_ref():
    assert compute_cer("", "") == 0.0
    assert compute_cer("", "ABC") == 1.0


# ── compute_wer ──────────────────────────────────────────────────────────────

def test_compute_wer_perfect():
    assert compute_wer("AB123CD", "AB123CD") == 0.0


def test_compute_wer_wrong():
    assert compute_wer("AB123CD", "XY456ZZ") == 1.0


# ── deskew_plate_crop ────────────────────────────────────────────────────────

def test_deskew_preserves_shape():
    crop = np.zeros((60, 200, 3), dtype=np.uint8)
    out = deskew_plate_crop(crop)
    assert out.shape == (60, 200, 3)


def test_deskew_flat_image():
    """Image entièrement noire (aucun contour) → retourne sans exception."""
    crop = np.zeros((60, 200, 3), dtype=np.uint8)
    out = deskew_plate_crop(crop)
    assert out.shape == crop.shape


# ── read_plate_ocr ───────────────────────────────────────────────────────────

def test_read_plate_ocr_output_keys(tmp_path):
    """Mock easyocr.Reader → résultat contient les 4 clés attendues."""
    crop = np.zeros((60, 200, 3), dtype=np.uint8)

    # easyocr readtext retourne list[(bbox, text, conf)]
    fake_bbox = [[0, 0], [100, 0], [100, 30], [0, 30]]
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [(fake_bbox, "AB123CD", 0.92)]

    with patch("easyocr.Reader", return_value=mock_reader):
        result = read_plate_ocr(crop, reader=mock_reader)

    assert set(result.keys()) == {"plate_text", "raw_text", "confidence", "n_boxes"}
    assert isinstance(result["plate_text"], str)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["n_boxes"], int)


# ── evaluate_full_pipeline ───────────────────────────────────────────────────

def test_evaluate_full_pipeline_missing_weights(tmp_path):
    """Poids absents → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        evaluate_full_pipeline(weights_path=tmp_path / "noexist.pt")


def test_evaluate_full_pipeline_uses_nested_test_accuracy(tmp_path, monkeypatch):
    """La comparaison NB vs YOLO doit lire accuracy depuis la clef test."""
    import json
    import sys

    monkeypatch.chdir(tmp_path)

    # Poids et images de test minimaux
    fake_weights = tmp_path / "best.pt"
    fake_weights.write_bytes(b"fake")
    test_img_dir = tmp_path / "data/processed/images/test"
    test_img_dir.mkdir(parents=True)
    img_path = test_img_dir / "000001.jpg"
    import numpy as np
    import cv2
    cv2.imwrite(str(img_path), np.zeros((640, 640, 3), dtype=np.uint8))

    # Références OCR pour la même image
    ocr_results_path = tmp_path / "data/processed/ocr_results.json"
    ocr_results_path.parent.mkdir(parents=True, exist_ok=True)
    ocr_results_path.write_text(
        json.dumps([
            {
                "plate_text": "ABC123",
                "ocr_confidence": 0.5,
                "crop_path": "data/processed/plate_crops/000001.jpg",
            }
        ]),
        encoding="utf-8",
    )

    # NB metrics avec champ nested test/accuracy
    nb_metrics_path = tmp_path / "models/weights/naive_bayes_metrics.json"
    nb_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    nb_metrics_path.write_text(
        json.dumps({"test": {"accuracy": 0.95}, "accuracy": 0.1}),
        encoding="utf-8",
    )

    # Mocks du pipeline YOLO+OCR
    monkeypatch.setitem(sys.modules, "easyocr", MagicMock(Reader=lambda langs, gpu: MagicMock()))

    def fake_detect_plate(image_path, weights_path=None, save_annotated=False):
        return {
            "n_detections": 1,
            "detections": [{"crop": np.zeros((224, 224, 3), dtype=np.uint8),
                            "bbox": [0, 0, 10, 10],
                            "confidence": 0.9}],
        }

    def fake_read_plate_ocr(crop, reader=None):
        return {"plate_text": "ABC123", "confidence": 0.9}

    monkeypatch.setattr("modules.module_a.yolo_ocr_pipeline.detect_plate", fake_detect_plate)
    monkeypatch.setattr("modules.module_a.yolo_ocr_pipeline.read_plate_ocr", fake_read_plate_ocr)

    metrics = evaluate_full_pipeline(
        weights_path=fake_weights,
        ocr_results_path=ocr_results_path,
        report_dir=tmp_path / "reports",
    )

    assert metrics["comparison_nb_vs_yolo_ocr"]["nb_accuracy"] == 0.95
    assert metrics["comparison_nb_vs_yolo_ocr"]["improvement"] == "YOLO+OCR meilleur"
