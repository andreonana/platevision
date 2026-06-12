"""
Tests unitaires — modules/module_a/evaluate.py
Utilise des dicts synthétiques — aucun fichier réel requis.
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from modules.module_a.evaluate import (
    compute_gains,
    generate_comparative_figure,
    generate_comparison_table,
    load_nb_metrics,
    load_yolo_metrics,
)

# ── Fixtures synthétiques ──────────────────────────────────────────────────────

NB_MOCK = {
    "accuracy":          0.72,
    "f1_macro":          0.68,
    "precision_macro":   0.70,
    "recall_macro":      0.66,
    "top3_accuracy":     0.91,
    "inference_ms_mean": 0.8,
    "top_confusions":    [],
}

YOLO_MOCK = {
    "map50":             0.88,
    "map50_95":          0.71,
    "precision":         0.85,
    "recall":            0.82,
    "inference_ms_mean": 18.0,
    "realtime_ok":       True,
    "mean_cer":          0.12,
    "mean_wer":          0.25,
    "detection_rate":    0.94,
    "perfect_read_rate": 0.61,
    "mean_ocr_conf":     0.78,
    "inference_total_ms": 45.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# compute_gains
# ══════════════════════════════════════════════════════════════════════════════

def test_compute_gains_keys():
    gains = compute_gains(NB_MOCK, YOLO_MOCK)
    expected = {
        "accuracy_vs_perfect_read",
        "f1_vs_1_minus_cer",
        "inference_ratio",
        "overall_winner",
        "jury_conclusion",
    }
    assert set(gains.keys()) == expected


def test_compute_gains_winner_logic():
    # perfect_read_rate (0.61) > accuracy (0.72) → NB meilleur ici
    gains = compute_gains(NB_MOCK, YOLO_MOCK)
    acc_w = gains["accuracy_vs_perfect_read"]["winner"]
    assert acc_w in ("YOLO+OCR", "NB", "équivalent")

    # Cas forcé : perfect_read_rate > accuracy
    nb_low  = {**NB_MOCK, "accuracy": 0.50}
    yo_high = {**YOLO_MOCK, "perfect_read_rate": 0.80}
    gains2  = compute_gains(nb_low, yo_high)
    assert gains2["accuracy_vs_perfect_read"]["winner"] == "YOLO+OCR"

    # Cas forcé : perfect_read_rate < accuracy
    nb_high = {**NB_MOCK, "accuracy": 0.90}
    yo_low  = {**YOLO_MOCK, "perfect_read_rate": 0.40}
    gains3  = compute_gains(nb_high, yo_low)
    assert gains3["accuracy_vs_perfect_read"]["winner"] == "NB"


def test_compute_gains_jury_conclusion_not_empty():
    gains = compute_gains(NB_MOCK, YOLO_MOCK)
    conclusion = gains["jury_conclusion"]
    assert isinstance(conclusion, str)
    assert len(conclusion) > 0
    assert "MINT" in conclusion


def test_compute_gains_inference_ratio():
    gains = compute_gains(NB_MOCK, YOLO_MOCK)
    inf = gains["inference_ratio"]
    assert inf["nb_ms"] == pytest.approx(0.8, abs=0.01)
    assert inf["yolo_ocr_ms"] == pytest.approx(45.0, abs=0.01)
    assert inf["ratio"] == pytest.approx(45.0 / 0.8, abs=0.1)
    assert isinstance(inf["comment"], str)


# ══════════════════════════════════════════════════════════════════════════════
# generate_comparison_table
# ══════════════════════════════════════════════════════════════════════════════

def _make_gains():
    return compute_gains(NB_MOCK, YOLO_MOCK)


def test_generate_comparison_table_contains_metrics(tmp_path):
    gains = _make_gains()
    table = generate_comparison_table(NB_MOCK, YOLO_MOCK, gains,
                                      output_path=tmp_path / "tbl.txt")
    assert "CER" in table
    assert "mAP" in table
    assert "MINT" in table


def test_generate_comparison_table_saves_file(tmp_path):
    gains = _make_gains()
    out = tmp_path / "comparative_table.txt"
    generate_comparison_table(NB_MOCK, YOLO_MOCK, gains, output_path=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_generate_comparison_table_returns_str(tmp_path):
    gains = _make_gains()
    result = generate_comparison_table(NB_MOCK, YOLO_MOCK, gains,
                                       output_path=tmp_path / "t.txt")
    assert isinstance(result, str)
    assert len(result) > 100


# ══════════════════════════════════════════════════════════════════════════════
# generate_comparative_figure
# ══════════════════════════════════════════════════════════════════════════════

def test_generate_comparative_figure_creates_png(tmp_path):
    gains = _make_gains()
    out = tmp_path / "fig.png"
    generate_comparative_figure(NB_MOCK, YOLO_MOCK, gains, output_path=out)
    assert out.exists()
    assert out.stat().st_size > 10_000  # > 10 KB


# ══════════════════════════════════════════════════════════════════════════════
# load_nb_metrics
# ══════════════════════════════════════════════════════════════════════════════

def test_load_nb_metrics_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_nb_metrics(path=tmp_path / "noexist.json")


def test_load_nb_metrics_flat_structure(tmp_path):
    """JSON à plat (sans clé 'test') → métriques chargées correctement."""
    data = {
        "accuracy": 0.75, "f1_macro": 0.70,
        "precision_macro": 0.72, "recall_macro": 0.68,
        "top3_accuracy": 0.90, "inference_ms_mean": 1.2,
        "top_confusions": [],
    }
    p = tmp_path / "nb.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    result = load_nb_metrics(path=p)
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["inference_ms_mean"] == pytest.approx(1.2)


def test_load_nb_metrics_nested_structure(tmp_path):
    """JSON avec clé 'test' → métriques extraites correctement."""
    data = {
        "test": {
            "accuracy": 0.80, "f1_macro": 0.75,
            "precision_macro": 0.77, "recall_macro": 0.73,
            "top3_accuracy": 0.93, "inference_ms_mean": 0.9,
        }
    }
    p = tmp_path / "nb_nested.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    result = load_nb_metrics(path=p)
    assert result["accuracy"] == pytest.approx(0.80)


# ══════════════════════════════════════════════════════════════════════════════
# load_yolo_metrics
# ══════════════════════════════════════════════════════════════════════════════

def test_load_yolo_metrics_missing_yolo(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yolo_metrics(yolo_path=tmp_path / "noexist.json",
                          ocr_path=tmp_path / "ocr.json")


def test_load_yolo_metrics_missing_ocr(tmp_path):
    """yolo_metrics.json présent, ocr_evaluation absent → clés OCR = None."""
    yolo_data = {
        "map50": 0.88, "map50_95": 0.71,
        "precision": 0.85, "recall": 0.82,
        "inference_ms_mean": 18.0, "realtime_ok": True,
    }
    yolo_path = tmp_path / "yolo_metrics.json"
    yolo_path.write_text(json.dumps(yolo_data), encoding="utf-8")

    result = load_yolo_metrics(yolo_path=yolo_path,
                               ocr_path=tmp_path / "missing_ocr.json")
    assert result["mean_cer"] is None
    assert result["mean_wer"] is None
    assert result["map50"] == pytest.approx(0.88)


def test_load_yolo_metrics_with_ocr(tmp_path):
    """Les deux JSON présents → toutes les clés remplies."""
    yolo_data = {
        "map50": 0.88, "map50_95": 0.71,
        "precision": 0.85, "recall": 0.82,
        "inference_ms_mean": 18.0, "realtime_ok": True,
    }
    ocr_data = {
        "mean_cer": 0.12, "mean_wer": 0.25,
        "detection_rate": 0.94, "perfect_read_rate": 0.61,
        "mean_ocr_conf": 0.78, "inference_total_ms": 45.0,
    }
    yp = tmp_path / "yolo.json"
    op = tmp_path / "ocr.json"
    yp.write_text(json.dumps(yolo_data), encoding="utf-8")
    op.write_text(json.dumps(ocr_data), encoding="utf-8")

    result = load_yolo_metrics(yolo_path=yp, ocr_path=op)
    assert result["mean_cer"] == pytest.approx(0.12)
    assert result["inference_total_ms"] == pytest.approx(45.0)
