"""
Tests unitaires — preprocessing/feature_extraction.py
Analyse discriminabilité, importance Fisher, rapport jury.
Aucun fichier réel requis — tout en numpy.random.
"""

import numpy as np
import pytest

from preprocessing.feature_extraction import (
    analyze_feature_discriminability,
    compute_feature_importance,
    generate_analysis_report,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

# VALID_CHARS en ordre ASCII : chiffres d'abord, puis lettres
VALID_CHARS = sorted("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
N_CLASSES   = 36
N_PER_CLASS = 20     # ≥ 5 pour que toutes les paires soient analysées


def _synthetic(seed=0):
    """Retourne (X, y) synthétiques couvrant les 36 classes."""
    rng = np.random.default_rng(seed)
    X = rng.random((N_CLASSES * N_PER_CLASS, 120)).astype(np.float32)
    y = np.repeat(np.arange(N_CLASSES), N_PER_CLASS)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# analyze_feature_discriminability
# ══════════════════════════════════════════════════════════════════════════════

def test_discriminability_output_structure():
    """Le résultat contient 'critical_pairs' et 'global_verdict'."""
    X, y = _synthetic()
    result = analyze_feature_discriminability(X, y, VALID_CHARS)

    assert "critical_pairs" in result
    assert "global_verdict" in result
    assert isinstance(result["global_verdict"], str)

    expected_pairs = ["0_vs_O", "1_vs_I", "B_vs_8", "5_vs_S", "6_vs_G"]
    for pair in expected_pairs:
        assert pair in result["critical_pairs"], f"Paire manquante : {pair}"
        data = result["critical_pairs"][pair]
        assert "mean_overlap"      in data
        assert "verdict"           in data
        assert "top5_discriminant" in data
        assert "top5_confondant"   in data
        assert len(data["top5_discriminant"]) == 5
        assert len(data["top5_confondant"])   == 5
        # Chaque entrée top5 a les clés attendues
        for entry in data["top5_discriminant"]:
            assert "feature" in entry
            assert "overlap"  in entry


def test_discriminability_verdict_range():
    """mean_overlap est toujours dans [0.0, 1.0] pour toutes les paires."""
    X, y = _synthetic(seed=7)
    result = analyze_feature_discriminability(X, y, VALID_CHARS)
    for pair_key, data in result["critical_pairs"].items():
        overlap = data["mean_overlap"]
        assert 0.0 <= overlap <= 1.0, (
            f"Paire {pair_key} : mean_overlap={overlap} hors [0, 1]"
        )


def test_discriminability_skips_insufficient_data():
    """Une classe avec < 5 samples est ignorée (pas d'exception)."""
    rng = np.random.default_rng(42)
    # Seulement 3 samples pour les classes '0' et 'O' (indices 0 et 24)
    X = rng.random((N_CLASSES * N_PER_CLASS, 120)).astype(np.float32)
    y = np.repeat(np.arange(N_CLASSES), N_PER_CLASS)
    # On retire presque tous les échantillons de '0' (idx 0)
    mask = ~np.isin(np.arange(len(y)), np.where(y == 0)[0][3:])
    X_small, y_small = X[mask], y[mask]

    # Ne doit pas lever d'exception
    result = analyze_feature_discriminability(X_small, y_small, VALID_CHARS)
    # La paire 0_vs_O peut être absente (< 5 samples) — c'est le comportement attendu
    assert "critical_pairs" in result


# ══════════════════════════════════════════════════════════════════════════════
# compute_feature_importance
# ══════════════════════════════════════════════════════════════════════════════

def test_feature_importance_all_groups_present():
    """group_scores a exactement 8 clés ; ranking en a aussi 8."""
    X, y = _synthetic()
    result = compute_feature_importance(X, y, VALID_CHARS)

    expected_groups = {
        "A_forme", "B_zones", "C_transitions", "D_gradients",
        "HSV_hist", "grad_plaque", "dark_ratio", "metal_frame",
    }
    assert set(result["group_scores"].keys()) == expected_groups
    assert len(result["ranking"]) == 8
    assert isinstance(result["conclusion"], str)
    assert len(result["conclusion"]) > 20


def test_feature_importance_scores_positive():
    """Tous les scores Fisher sont ≥ 0."""
    X, y = _synthetic(seed=3)
    result = compute_feature_importance(X, y, VALID_CHARS)
    for grp, score in result["group_scores"].items():
        assert score >= 0.0, f"Score négatif pour le groupe '{grp}' : {score}"


def test_feature_importance_ranking_consistent():
    """Le ranking est bien trié du score le plus élevé au plus bas."""
    X, y = _synthetic(seed=5)
    result = compute_feature_importance(X, y, VALID_CHARS)
    scores  = [result["group_scores"][g] for g in result["ranking"]]
    assert scores == sorted(scores, reverse=True), "Ranking non trié"


# ══════════════════════════════════════════════════════════════════════════════
# generate_analysis_report
# ══════════════════════════════════════════════════════════════════════════════

def test_generate_report_creates_file(tmp_path):
    """generate_analysis_report crée feature_analysis_report.txt sans exception."""
    X, y = _synthetic()
    generate_analysis_report(X, y, VALID_CHARS, tmp_path)

    report = tmp_path / "feature_analysis_report.txt"
    assert report.exists(), "Le fichier de rapport n'a pas été créé"
    content = report.read_text(encoding="utf-8")
    # Vérifications minimales du contenu structurel
    assert "ANALYSE FEATURES 120D" in content
    assert "IMPORTANCE PAR GROUPE"  in content
    assert "PAIRES CRITIQUES"       in content
    assert "QUESTIONS JURY"         in content


def test_generate_report_also_saves_json(tmp_path):
    """generate_analysis_report produit aussi feature_discriminability.json."""
    X, y = _synthetic(seed=9)
    generate_analysis_report(X, y, VALID_CHARS, tmp_path)
    assert (tmp_path / "feature_discriminability.json").exists()
