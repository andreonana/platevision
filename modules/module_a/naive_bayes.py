"""
Module A1 — Classificateur Naïves Bayes Gaussien
Reconnaissance de caractères alphanumériques (36 classes) sur features 120D.
"""

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR   = Path("data/processed")
MODEL_DIR  = Path("models/weights")
REPORT_DIR = Path("reports/rapport_technique/figures")
N_CLASSES  = 36
N_FEATURES = 120


def load_features(data_dir: Path = DATA_DIR) -> tuple:
    """
    Charge X_train, y_train, X_val, y_val, X_test, y_test, class_names.
    Lève FileNotFoundError si features.npy absent.
    Affiche shapes + distribution des classes.
    Charge aussi ocr_results.json si présent (pour info, pas utilisé en training).
    Retourne : (X_train, y_train, X_val, y_val, X_test, y_test, class_names)
    """
    features_path = data_dir / "features.npy"
    if not features_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {features_path}")

    X_train = np.load(data_dir / "features.npy")
    y_train = np.load(data_dir / "labels.npy")
    X_val   = np.load(data_dir / "features_val.npy")
    y_val   = np.load(data_dir / "labels_val.npy")
    X_test  = np.load(data_dir / "features_test.npy")
    y_test  = np.load(data_dir / "labels_test.npy")

    class_names_path = data_dir / "class_names.txt"
    if class_names_path.exists():
        class_names = class_names_path.read_text().strip().splitlines()
    else:
        class_names = [str(i) for i in range(N_CLASSES)]

    logger.info("Shapes — train: %s, val: %s, test: %s", X_train.shape, X_val.shape, X_test.shape)
    _, counts = np.unique(y_train, return_counts=True)
    logger.info("Distribution classes (train) — min: %d, max: %d, moy: %.1f",
                counts.min(), counts.max(), counts.mean())

    ocr_path = data_dir / "ocr_results.json"
    if ocr_path.exists():
        with ocr_path.open() as f:
            ocr_data = json.load(f)
        logger.info("ocr_results.json chargé — %d entrées", len(ocr_data))
    else:
        logger.info("ocr_results.json absent (ignoré pour l'entraînement)")

    return X_train, y_train, X_val, y_val, X_test, y_test, class_names


def preprocess_features(X_train, X_val, X_test) -> tuple:
    """
    Normalise avec StandardScaler fitté sur X_train uniquement.
    Retourne : (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    # fit uniquement sur X_train : évite la fuite de données (data leakage) vers val/test
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)
    logger.info("Normalisation appliquée — moyenne X_train_scaled ≈ %.4f", X_train_scaled.mean())
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_naive_bayes(X_train, y_train) -> GaussianNB:
    """
    Entraîne GaussianNB(var_smoothing=1e-9).
    Affiche : temps d'entraînement, accuracy sur train.
    Retourne : modèle entraîné
    """
    # var_smoothing=1e-9 : ajoute ε×max(var) à chaque variance — empêche P(x|c)=0 sur features à variance nulle
    model = GaussianNB(var_smoothing=1e-9)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    train_acc = accuracy_score(y_train, model.predict(X_train))
    logger.info("Entraînement terminé en %.3f s — accuracy train : %.4f", elapsed, train_acc)
    return model


def evaluate_model(model, scaler, X_val, y_val, X_test, y_test,
                   class_names, report_dir: Path = REPORT_DIR) -> dict:
    """
    Évalue sur val et test.
    Métriques : accuracy, precision_macro, recall_macro, f1_macro, top3_accuracy.
    Génère dans report_dir :
      - confusion_matrix_nb.png  (seaborn heatmap 36×36, figsize=(14,12))
      - classification_report_nb.txt
    Retourne dict avec toutes les métriques val + test.
    """
    report_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        X_scaled = scaler.transform(X)
        y_pred   = model.predict(X_scaled)
        y_proba  = model.predict_proba(X_scaled)

        n_classes_present = len(np.unique(y))
        k = min(3, n_classes_present)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            top3_acc = top_k_accuracy_score(y, y_proba, k=k, labels=np.arange(len(class_names)))

        results[split_name] = {
            "accuracy":         accuracy_score(y, y_pred),
            "precision_macro":  precision_score(y, y_pred, average="macro", zero_division=0),
            "recall_macro":     recall_score(y, y_pred, average="macro", zero_division=0),
            "f1_macro":         f1_score(y, y_pred, average="macro", zero_division=0),
            "top3_accuracy":    top3_acc,
        }
        logger.info("[%s] accuracy=%.4f  f1_macro=%.4f  top3=%.4f",
                    split_name,
                    results[split_name]["accuracy"],
                    results[split_name]["f1_macro"],
                    results[split_name]["top3_accuracy"])

    # Matrice de confusion sur le jeu de test
    X_test_scaled = scaler.transform(X_test)
    y_pred_test   = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_test, labels=np.arange(len(class_names)))

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title("Matrice de confusion — Naïves Bayes (test)")
    plt.tight_layout()
    cm_path = report_dir / "confusion_matrix_nb.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    logger.info("Matrice de confusion sauvegardée : %s", cm_path)

    # Rapport de classification textuel
    report_txt = classification_report(
        y_test, y_pred_test,
        target_names=class_names,
        zero_division=0,
    )
    report_path = report_dir / "classification_report_nb.txt"
    report_path.write_text(str(report_txt))
    logger.info("Rapport de classification sauvegardé : %s", report_path)

    return results


def analyze_errors(model, scaler, X_test, y_test, class_names) -> dict:
    """
    Identifie les 10 paires de classes les plus confondues (ex : O↔0, I↔1).
    Retourne : {"top_confusions": [(vrai, prédit, count), ...]}
    Affiche dans le terminal.
    """
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))
    np.fill_diagonal(cm, 0)

    # Indices des 10 valeurs les plus élevées hors diagonale
    flat_indices = np.argsort(cm.ravel())[::-1][:10]
    top_confusions = []
    for idx in flat_indices:
        true_idx = idx // len(class_names)
        pred_idx = idx % len(class_names)
        count    = cm[true_idx, pred_idx]
        if count == 0:
            break
        true_name = class_names[true_idx]
        pred_name = class_names[pred_idx]
        top_confusions.append((true_name, pred_name, int(count)))
        logger.info("Confusion — vrai: %s, prédit: %s, occurrences: %d",
                    true_name, pred_name, count)

    return {"top_confusions": top_confusions}


def generate_pedagogical_report(model, scaler, X_test, y_test,
                                  class_names,
                                  report_dir: Path = REPORT_DIR) -> None:
    """
    Génère le rapport pédagogique obligatoire du Module A1.
    Répond aux 3 questions jury §4.1 par du code, pas seulement par du texte.
    Sauvegarde : report_dir/naive_bayes_pedagogical_report.txt
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    char_to_idx = {c: i for i, c in enumerate(class_names)}
    n_feats = X_test.shape[1]

    X_scaled = scaler.transform(X_test)
    y_pred   = model.predict(X_scaled)
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))

    sep   = "═" * 54
    lines = [sep, "RAPPORT PÉDAGOGIQUE — Module A1 Naïves Bayes", sep, ""]

    # ── Section 1 : paires les plus confondues ────────────────────────────────
    lines += [f"{'═'*54}", "SECTION 1 — Paires de caractères les plus confondues",
              f"{'═'*54}", ""]

    # Paires MINT/DGI critiques : confusion cause des faux signalements ou laissez-passer erronés
    CRITICAL = [("0", "O"), ("1", "I"), ("8", "B"), ("5", "S"), ("6", "G")]
    critical_set = {frozenset(p) for p in CRITICAL}

    cm_nodiag = cm.copy()
    np.fill_diagonal(cm_nodiag, 0)

    # Toutes les paires non-nulles, triées par total décroissant
    pair_counts = {}
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            total = int(cm_nodiag[i, j]) + int(cm_nodiag[j, i])
            if total > 0:
                pair_counts[(i, j)] = total

    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)

    lines.append("Top-15 paires les plus confondues :")
    for (i, j), total in sorted_pairs[:15]:
        c1, c2   = class_names[i], class_names[j]
        n12      = int(cm_nodiag[i, j])
        n21      = int(cm_nodiag[j, i])
        critical = "  [CRITIQUE MINT/DGI]" if frozenset((c1, c2)) in critical_set else ""
        lines.append(f"  • '{c1}' → '{c2}' : {n12} fois | '{c2}' → '{c1}' : {n21} fois{critical}")
    lines.append("")

    lines.append("Détail paires critiques MINT/DGI :")
    for c1, c2 in CRITICAL:
        if c1 not in char_to_idx or c2 not in char_to_idx:
            lines.append(f"  • '{c1}' / '{c2}' : données insuffisantes")
            continue
        i1, i2 = char_to_idx[c1], char_to_idx[c2]
        n12    = int(cm[i1, i2])
        n21    = int(cm[i2, i1])
        lines.append(f"  • '{c1}' → '{c2}' : {n12} fois | '{c2}' → '{c1}' : {n21} fois")
    lines.append("")

    # ── Section 2 : violation indépendance conditionnelle ─────────────────────
    lines += [f"{'═'*54}",
              "SECTION 2 — Pourquoi l'indépendance conditionnelle est violée",
              f"{'═'*54}", ""]

    corr    = np.corrcoef(X_scaled.T)
    mask    = np.abs(corr) > 0.7
    np.fill_diagonal(mask, False)
    n_high  = int(mask.sum()) // 2

    lines += [
        f"Corrélation de Pearson calculée sur X_test ({X_scaled.shape[0]} samples, "
        f"{n_feats} features).",
        f"Paires de features avec |r| > 0.7 : {n_high} paires sur "
        f"{n_feats * (n_feats - 1) // 2} possibles.",
        "",
        f"L'hypothèse Naïves Bayes suppose P(x1,...,x{n_feats}|y) = ∏P(xi|y).",
        "Cette hypothèse est violée pour des images car :",
        "  1. Les pixels adjacents d'une même zone partagent la même encre/couleur",
        f"     → forte corrélation spatiale locale (mesurée : {n_high} paires |r|>0.7).",
        "  2. Les 96 bins HSV sont contraints à sommer à 1.0 par canal",
        "     → corrélation négative systématique entre bins voisins.",
        "  3. Les densités zonales (groupe B) sont corrélées entre zones adjacentes.",
        "Conséquence : GaussianNB sous-estime la probabilité des classes similaires",
        "et surestime sa confiance, ce qui nuit à la discrimination 0/O et 1/I.",
        "",
    ]

    # ── Section 3 : impact opérationnel MINT/DGI ─────────────────────────────
    lines += [f"{'═'*54}", "SECTION 3 — Impact opérationnel MINT/DGI",
              f"{'═'*54}", ""]

    # Taux de confusion 0↔O
    c_zero, c_O = "0", "O"
    if c_zero in char_to_idx and c_O in char_to_idx:
        i0, iO   = char_to_idx[c_zero], char_to_idx[c_O]
        count_0O = int(cm[i0, iO]) + int(cm[iO, i0])
        n_0O     = int((y_test == i0).sum()) + int((y_test == iO).sum())
        taux     = count_0O / n_0O if n_0O > 0 else 0.0
    else:
        count_0O, taux = 0, 0.0

    # Taux de confusion 1↔I
    c_one, c_I = "1", "I"
    if c_one in char_to_idx and c_I in char_to_idx:
        i1, iI   = char_to_idx[c_one], char_to_idx[c_I]
        count_1I = int(cm[i1, iI]) + int(cm[iI, i1])
    else:
        count_1I = 0

    lines += [
        f"Impact opérationnel — confusion '0'/'O' (MINT/DGI) :",
        f"  Sur le jeu de test, le modèle a confondu '0' et 'O' {count_0O} fois.",
        "  Dans le contexte MINT : une plaque lue 'AB123OC' au lieu de 'AB1230C'",
        "  génère une requête sur une plaque inexistante dans la base nationale",
        "  → le véhicule frauduleux n'est PAS détecté (faux négatif sécuritaire).",
        "  Dans le contexte DGI : une plaque valide lue incorrectement déclenche",
        "  une alerte injustifiée → contrôle abusif, contestation juridique,",
        "  coût opérationnel (estimé à 22% des charges selon diagnostic MINT/DGI).",
        f"  Taux de confusion 0↔O observé : {taux:.2%}",
        "  Conclusion : ce taux est inacceptable pour un déploiement opérationnel",
        "  → justifie le passage au pipeline A2 (YOLO + EasyOCR).",
        "",
        f"Confusion '1'/'I' observée : {count_1I} fois.",
        "",
    ]

    lines.append(sep)

    report_txt  = "\n".join(lines)
    report_path = report_dir / "naive_bayes_pedagogical_report.txt"
    report_path.write_text(report_txt, encoding="utf-8")
    logger.info("Rapport pédagogique sauvegardé : %s", report_path)
    print(report_txt)


def analyze_ocr_alignment(model, scaler, data_dir: Path = DATA_DIR,
                           class_names=None) -> dict:
    """
    Compare les prédictions GaussianNB avec les textes OCR de ocr_results.json.
    Pour chaque crop ayant ocr_confidence >= 0.45 et plate_text non None :
      - Découpe plate_text en caractères individuels
      - Compare chaque caractère prédit par NB vs lu par OCR
      - Calcule le taux d'accord NB–OCR (Character Agreement Rate)
    Retourne :
      {
        "n_crops_analysed": int,
        "n_chars_compared": int,
        "nb_ocr_agreement_rate": float,
        "disagreements": [{"nb_pred": str, "ocr_text": str, "crop_path": str}, ...]
      }
    Si ocr_results.json absent : log warning, retourne dict vide.
    """
    ocr_path = data_dir / "ocr_results.json"
    if not ocr_path.exists():
        logger.warning("ocr_results.json introuvable — analyse OCR ignorée.")
        return {}

    with ocr_path.open() as f:
        ocr_data = json.load(f)

    features_path = data_dir / "features.npy"
    labels_path   = data_dir / "labels.npy"
    if not features_path.exists() or not labels_path.exists():
        logger.warning("features.npy / labels.npy introuvables — analyse OCR impossible.")
        return {}

    X_all    = np.load(features_path)
    np.load(labels_path)  # chargé mais non utilisé directement (prédictions via X_all)
    X_scaled = scaler.transform(X_all)
    y_pred   = model.predict(X_scaled)

    if class_names is None:
        class_names_path = data_dir / "class_names.txt"
        if class_names_path.exists():
            class_names = class_names_path.read_text().strip().splitlines()
        else:
            class_names = [str(i) for i in range(N_CLASSES)]

    # Indexer les features par crop_path si disponible dans ocr_results
    n_crops_analysed = 0
    n_chars_compared = 0
    n_agreements     = 0
    disagreements    = []

    for entry in ocr_data:
        confidence = entry.get("ocr_confidence", 0.0)
        plate_text = entry.get("plate_text", None)
        crop_path  = entry.get("crop_path", "")

        if confidence < 0.45 or plate_text is None:
            continue

        plate_chars = [c.upper() for c in plate_text if c.isalnum()]
        if not plate_chars:
            continue

        n_crops_analysed += 1

        # Trouver les features correspondant à ce crop (matching par index)
        # On suppose que les features sont ordonnées dans le même ordre que ocr_results
        crop_idx = ocr_data.index(entry)
        chars_per_crop = len(plate_chars)
        start = crop_idx * chars_per_crop
        end   = start + chars_per_crop

        if end > len(X_all):
            continue

        for i, ocr_char in enumerate(plate_chars):
            sample_idx = start + i
            if sample_idx >= len(y_pred):
                break
            nb_pred_idx = y_pred[sample_idx]
            nb_pred_char = class_names[nb_pred_idx] if nb_pred_idx < len(class_names) else "?"

            n_chars_compared += 1
            if nb_pred_char == ocr_char:
                n_agreements += 1
            elif len(disagreements) < 20:
                disagreements.append({
                    "nb_pred":   nb_pred_char,
                    "ocr_text":  ocr_char,
                    "crop_path": crop_path,
                })

    agreement_rate = n_agreements / n_chars_compared if n_chars_compared > 0 else 0.0
    logger.info(
        "Alignement OCR — crops: %d, chars comparés: %d, accord NB–OCR: %.4f",
        n_crops_analysed, n_chars_compared, agreement_rate,
    )

    return {
        "n_crops_analysed":      n_crops_analysed,
        "n_chars_compared":      n_chars_compared,
        "nb_ocr_agreement_rate": agreement_rate,
        "disagreements":         disagreements,
    }


def save_model(model, scaler, metrics: dict,
               model_dir: Path = MODEL_DIR) -> None:
    """
    Sauvegarde :
      models/weights/naive_bayes_model.pkl
      models/weights/naive_bayes_scaler.pkl
      models/weights/naive_bayes_metrics.json
    Crée model_dir si absent.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,  model_dir / "naive_bayes_model.pkl")
    joblib.dump(scaler, model_dir / "naive_bayes_scaler.pkl")
    (model_dir / "naive_bayes_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False)
    )
    logger.info("Modèle sauvegardé dans %s", model_dir)


def load_model(model_dir: Path = MODEL_DIR) -> tuple:
    """
    Charge et retourne (model, scaler, metrics).
    Lève FileNotFoundError si absent.
    """
    model_path  = model_dir / "naive_bayes_model.pkl"
    scaler_path = model_dir / "naive_bayes_scaler.pkl"
    metrics_path = model_dir / "naive_bayes_metrics.json"

    for p in (model_path, scaler_path, metrics_path):
        if not p.exists():
            raise FileNotFoundError(f"Fichier modèle introuvable : {p}")

    model   = joblib.load(model_path)
    scaler  = joblib.load(scaler_path)
    metrics = json.loads(metrics_path.read_text())
    logger.info("Modèle chargé depuis %s", model_dir)
    return model, scaler, metrics


def predict_character(feature_vector: np.ndarray,
                      model=None, scaler=None,
                      model_dir: Path = MODEL_DIR) -> dict:
    """
    Prédit un vecteur 120D.
    Charge le modèle depuis disque si model/scaler non fournis.
    Retourne :
      {
        "predicted_class": str,
        "predicted_index": int,
        "confidence": float,
        "top3": [{"class": str, "probability": float}, ...]
      }
    """
    if model is None or scaler is None:
        model, scaler, _ = load_model(model_dir)

    class_names_path = DATA_DIR / "class_names.txt"
    if class_names_path.exists():
        class_names = class_names_path.read_text().strip().splitlines()
    else:
        class_names = [str(i) for i in range(N_CLASSES)]

    vec = feature_vector.reshape(1, -1)
    vec_scaled = scaler.transform(vec)
    proba      = model.predict_proba(vec_scaled)[0]

    pred_idx  = int(np.argmax(proba))
    top3_idx  = np.argsort(proba)[::-1][:3]

    return {
        "predicted_class": class_names[pred_idx],
        "predicted_index": pred_idx,
        "confidence":      float(proba[pred_idx]),
        "top3": [
            {"class": class_names[i], "probability": float(proba[i])}
            for i in top3_idx
        ],
    }


def run_full_pipeline(data_dir: Path = DATA_DIR,
                      model_dir: Path = MODEL_DIR,
                      report_dir: Path = REPORT_DIR) -> dict:
    """
    Orchestre : load → preprocess → train → evaluate → analyze_errors
                → analyze_ocr_alignment → save.
    Affiche un résumé final avec toutes les métriques.
    Retourne le dict de métriques.
    """
    logger.info("=== Pipeline Naïves Bayes — démarrage ===")

    X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_features(data_dir)
    # preprocess_features normalise X_train_s pour l'entraînement ;
    # evaluate_model et analyze_errors appellent scaler.transform() en interne
    # → on leur passe X_val et X_test bruts pour éviter la double normalisation.
    X_train_s, _, _, scaler = preprocess_features(X_train, X_val, X_test)
    model   = train_naive_bayes(X_train_s, y_train)
    metrics = evaluate_model(model, scaler, X_val, y_val, X_test, y_test,
                             class_names, report_dir)

    errors = analyze_errors(model, scaler, X_test, y_test, class_names)
    generate_pedagogical_report(model, scaler, X_test, y_test, class_names, report_dir)
    ocr    = analyze_ocr_alignment(model, scaler, data_dir, class_names)

    metrics["top_confusions"]    = errors.get("top_confusions", [])
    metrics["ocr_alignment"]     = ocr

    save_model(model, scaler, metrics, model_dir)

    logger.info("=== Résumé final ===")
    for split in ("val", "test"):
        if split in metrics:
            m = metrics[split]
            logger.info(
                "[%s] accuracy=%.4f  precision=%.4f  recall=%.4f  f1=%.4f  top3=%.4f",
                split, m["accuracy"], m["precision_macro"],
                m["recall_macro"], m["f1_macro"], m["top3_accuracy"],
            )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module A1 — Naïves Bayes Gaussien")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train",     action="store_true",
                       help="Lance le pipeline complet d'entraînement")
    group.add_argument("--predict",   type=int, metavar="X",
                       help="Prédit le sample d'index X dans X_test (affiche top3)")
    group.add_argument("--evaluate",  action="store_true",
                       help="Charge le modèle sauvegardé et réévalue sur test")
    group.add_argument("--ocr-align", action="store_true",
                       help="Lance uniquement analyze_ocr_alignment()")
    args = parser.parse_args()

    if args.train:
        run_full_pipeline()

    elif args.predict is not None:
        _, _, _, _, X_test, y_test, class_names = load_features()
        _, X_test_s, _, scaler = preprocess_features(X_test, X_test, X_test)
        model, scaler, _ = load_model()
        idx = args.predict
        if idx >= len(X_test):
            logger.error("Index %d hors bornes (taille test : %d)", idx, len(X_test))
        else:
            result = predict_character(X_test[idx], model=model, scaler=scaler)
            print(f"Vrai label      : {class_names[y_test[idx]]}")
            print(f"Prédit          : {result['predicted_class']}")
            print(f"Confiance       : {result['confidence']:.4f}")
            print("Top 3 :")
            for item in result["top3"]:
                print(f"  {item['class']} — {item['probability']:.4f}")

    elif args.evaluate:
        model, scaler, _ = load_model()
        X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_features()
        # evaluate_model appelle scaler.transform() en interne → données brutes
        metrics = evaluate_model(model, scaler, X_val, y_val, X_test, y_test, class_names)
        print(json.dumps(metrics, indent=2))

    elif args.ocr_align:
        model, scaler, _ = load_model()
        result = analyze_ocr_alignment(model, scaler)
        print(json.dumps(result, indent=2, ensure_ascii=False))
