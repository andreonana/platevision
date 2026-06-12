"""
Tests unitaires — Module A1 : Naïves Bayes Gaussien
Données synthétiques uniquement (100 samples × 120D, 10 classes).
"""

import numpy as np
import pytest
from sklearn.naive_bayes import GaussianNB

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

N_SAMPLES  = 100
N_FEATURES = 120
N_CLASSES  = 10  # sous-ensemble synthétique (réel = 36)


def _make_data(n=N_SAMPLES, n_feat=N_FEATURES, n_cls=N_CLASSES, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.random((n, n_feat)).astype(np.float32)
    y = rng.integers(0, n_cls, size=n)
    return X, y


def _make_full_data(tmp_path, n_cls=36):
    """Crée un répertoire data/processed synthétique pour les tests d'I/O."""
    rng = np.random.default_rng(0)
    n_train, n_val, n_test = 200, 50, 50

    X_train = rng.random((n_train, N_FEATURES)).astype(np.float32)
    y_train = rng.integers(0, n_cls, n_train)
    X_val   = rng.random((n_val,   N_FEATURES)).astype(np.float32)
    y_val   = rng.integers(0, n_cls, n_val)
    X_test  = rng.random((n_test,  N_FEATURES)).astype(np.float32)
    y_test  = rng.integers(0, n_cls, n_test)

    np.save(tmp_path / "features.npy",      X_train)
    np.save(tmp_path / "labels.npy",        y_train)
    np.save(tmp_path / "features_val.npy",  X_val)
    np.save(tmp_path / "labels_val.npy",    y_val)
    np.save(tmp_path / "features_test.npy", X_test)
    np.save(tmp_path / "labels_test.npy",   y_test)

    class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")  # 36 entrées
    (tmp_path / "class_names.txt").write_text("\n".join(class_names))

    return X_train, y_train, X_val, y_val, X_test, y_test, class_names


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_features_correct_shapes(tmp_path):
    """X_train.shape[1] == 120 et len(class_names) == 36."""
    from modules.module_a.naive_bayes import load_features

    _make_full_data(tmp_path)
    X_train, _, X_val, _, X_test, _, class_names = load_features(tmp_path)

    assert X_train.shape[1] == N_FEATURES, f"Attendu 120 features, obtenu {X_train.shape[1]}"
    assert len(class_names) == 36, f"Attendu 36 classes, obtenu {len(class_names)}"
    assert X_val.shape[1]  == N_FEATURES
    assert X_test.shape[1] == N_FEATURES


def test_load_features_missing_file(tmp_path):
    """load_features lève FileNotFoundError si features.npy absent."""
    from modules.module_a.naive_bayes import load_features

    with pytest.raises(FileNotFoundError):
        load_features(tmp_path)


def test_preprocess_no_data_leakage():
    """
    Le scaler est fitté sur X_train uniquement.
    X_train_scaled.mean() ≈ 0 ; X_val_scaled.mean() pas forcément ≈ 0.
    """
    from modules.module_a.naive_bayes import preprocess_features

    rng = np.random.default_rng(7)
    X_train = rng.random((200, N_FEATURES)).astype(np.float32)
    # X_val a une distribution différente (shift + scale)
    X_val   = rng.random((50, N_FEATURES)).astype(np.float32) * 10 + 5
    X_test  = rng.random((50, N_FEATURES)).astype(np.float32)

    X_train_s, X_val_s, _, scaler = preprocess_features(X_train, X_val, X_test)

    # Après StandardScaler fitté sur X_train, la moyenne de X_train_scaled ≈ 0
    assert abs(X_train_s.mean()) < 1e-5, (
        f"X_train_scaled.mean() devrait être ≈ 0, obtenu {X_train_s.mean():.6f}"
    )

    # X_val a été décalé → sa moyenne scalée ne doit PAS être proche de 0
    assert abs(X_val_s.mean()) > 0.1, (
        "X_val_scaled.mean() devrait refléter la distribution décalée de X_val"
    )

    # Vérifier que le scaler est bien un StandardScaler
    from sklearn.preprocessing import StandardScaler
    assert isinstance(scaler, StandardScaler)


def test_train_returns_gnb():
    """train_naive_bayes retourne bien une instance GaussianNB avec var_smoothing=1e-9."""
    from modules.module_a.naive_bayes import train_naive_bayes

    X, y = _make_data()
    model = train_naive_bayes(X, y)

    assert isinstance(model, GaussianNB)
    # Vérification via get_params() pour compatibilité Pylance
    assert model.get_params()["var_smoothing"] == 1e-9


def test_predict_output_structure(tmp_path):
    """
    predict_character retourne les clés attendues.
    confidence ∈ [0, 1] et len(top3) == 3.
    """
    from modules.module_a.naive_bayes import train_naive_bayes, preprocess_features, predict_character, save_model

    rng = np.random.default_rng(1)
    n_cls = 36
    X_train = rng.random((360, N_FEATURES)).astype(np.float32)
    y_train = np.repeat(np.arange(n_cls), 10)

    X_train_s, _, _, scaler = preprocess_features(X_train, X_train, X_train)
    model = train_naive_bayes(X_train_s, y_train)

    weights_dir = tmp_path / "weights"
    save_model(model, scaler, {}, weights_dir)

    vec = rng.random(N_FEATURES).astype(np.float32)
    result = predict_character(vec, model=model, scaler=scaler)

    assert "predicted_class" in result
    assert "confidence"      in result
    assert "top3"            in result
    assert "predicted_index" in result
    assert 0.0 <= result["confidence"] <= 1.0, "confidence doit être dans [0, 1]"
    assert len(result["top3"]) == 3, f"top3 doit contenir 3 éléments, obtenu {len(result['top3'])}"
    for item in result["top3"]:
        assert "class"       in item
        assert "probability" in item
        assert 0.0 <= item["probability"] <= 1.0


def test_save_load_roundtrip(tmp_path):
    """Sauvegarde → rechargement → prédictions identiques."""
    from modules.module_a.naive_bayes import (
        train_naive_bayes, preprocess_features, save_model, load_model,
    )

    rng = np.random.default_rng(2)
    n_cls = 36
    X_train = rng.random((360, N_FEATURES)).astype(np.float32)
    y_train = np.repeat(np.arange(n_cls), 10)

    X_s, _, _, scaler = preprocess_features(X_train, X_train, X_train)
    model = train_naive_bayes(X_s, y_train)

    metrics = {"test": {"accuracy": 0.5}}
    model_dir = tmp_path / "weights"
    save_model(model, scaler, metrics, model_dir)

    model2, scaler2, metrics2 = load_model(model_dir)

    assert metrics2["test"]["accuracy"] == pytest.approx(0.5)

    X_sample = rng.random((10, N_FEATURES)).astype(np.float32)
    preds1 = model.predict(scaler.transform(X_sample))
    preds2 = model2.predict(scaler2.transform(X_sample))
    np.testing.assert_array_equal(preds1, preds2)


def test_ocr_alignment_missing_file(tmp_path):
    """Si ocr_results.json absent → retourne {} sans exception."""
    from modules.module_a.naive_bayes import (
        train_naive_bayes, preprocess_features, analyze_ocr_alignment,
    )

    rng = np.random.default_rng(3)
    X = rng.random((100, N_FEATURES)).astype(np.float32)
    y = rng.integers(0, N_CLASSES, 100)

    X_s, _, _, scaler = preprocess_features(X, X, X)
    model = train_naive_bayes(X_s, y)

    # tmp_path ne contient pas ocr_results.json
    result = analyze_ocr_alignment(model, scaler, data_dir=tmp_path)

    assert result == {}, f"Attendu dict vide, obtenu : {result}"
