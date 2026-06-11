"""
Module A1 — Classifieur Naïves Bayes pour la classification de plaques.

Rôle dans le pipeline PlateVision :
    Fournit une baseline de classification supervisée à partir de features
    manuelles (histogrammes HSV, gradients de Sobel) extraites des régions
    candidates de plaques. Sert de référence pour évaluer le gain apporté
    par le pipeline YOLO + OCR du Module A2.

Responsable : Étudiant 1
"""

import logging
from pathlib import Path

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def load_features_and_labels(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge les features extraites et les étiquettes depuis le dossier traité.

    Args:
        data_path: Chemin vers le dossier contenant les features .npy et labels.csv

    Returns:
        Tuple (X, y) avec les features et étiquettes encodées.
    """
    features_file = data_path / "features.npy"
    labels_file = data_path / "labels.csv"

    if not features_file.exists() or not labels_file.exists():
        raise FileNotFoundError(
            f"Fichiers features.npy ou labels.csv introuvables dans {data_path}. "
            "Lancer d'abord preprocessing/feature_extraction.py."
        )

    X = np.load(features_file)

    import pandas as pd
    labels_df = pd.read_csv(labels_file)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels_df["label"].values)

    logger.info("Features chargées : %d échantillons, %d features", X.shape[0], X.shape[1])
    return X, y, encoder


def train_naive_bayes(X_train: np.ndarray, y_train: np.ndarray) -> GaussianNB:
    """Entraîne un classifieur Gaussien Naïves Bayes."""
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    logger.info("Modèle Naïves Bayes entraîné sur %d exemples.", len(y_train))
    return clf


def evaluate_classifier(
    clf: GaussianNB,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
) -> dict:
    """
    Évalue le classifieur et retourne les métriques.

    Returns:
        Dictionnaire contenant accuracy, rapport de classification et matrice de confusion.
    """
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    logger.info("Accuracy : %.4f", report["accuracy"])
    logger.info("Rapport de classification :\n%s",
                classification_report(y_test, y_pred, target_names=label_names))

    return {"accuracy": report["accuracy"], "report": report, "confusion_matrix": cm}


def run_naive_bayes_pipeline(input_path: Path, output_path: Path) -> dict:
    """
    Exécute le pipeline complet Naïves Bayes : chargement → entraînement → évaluation.

    Args:
        input_path: Dossier contenant features.npy et labels.csv
        output_path: Dossier de sortie pour les résultats

    Returns:
        Dictionnaire des métriques d'évaluation.
    """
    logger.info("=== Pipeline Naïves Bayes démarré ===")

    X, y, encoder = load_features_and_labels(input_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = train_naive_bayes(X_train, y_train)

    metrics = evaluate_classifier(clf, X_test, y_test, list(encoder.classes_))

    # Sauvegarde du rapport
    import json
    report_path = output_path / "module_a1_naive_bayes_results.json"
    serializable = {
        "accuracy": metrics["accuracy"],
        "report": metrics["report"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    logger.info("Résultats sauvegardés dans : %s", report_path)
    logger.info("=== Pipeline Naïves Bayes terminé ===")

    return metrics
