"""
PlateVision — Module A1 : Classifieur Naïves Bayes.

Ce module implémente un classifieur Naïves Bayes Gaussien pour la
détection et la classification des régions candidates aux plaques
d'immatriculation. Il utilise des features manuelles extraites par
preprocessing/feature_extraction.py (histogrammes HSV, gradients Sobel).

Rôle dans le pipeline : baseline de classification avant le modèle YOLO.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


def load_features(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge les features prétraitées depuis le dossier de données.

    Args:
        data_path: Chemin vers le dossier contenant les features .npy

    Returns:
        Tuple (X, y) : matrice de features et vecteur de labels
    """
    features_file = data_path / "features.npy"
    labels_file = data_path / "labels.npy"

    if not features_file.exists() or not labels_file.exists():
        raise FileNotFoundError(
            f"Fichiers features.npy ou labels.npy introuvables dans {data_path}. "
            "Exécuter d'abord preprocessing/feature_extraction.py"
        )

    X = np.load(features_file)
    y = np.load(labels_file)
    logger.info(f"Features chargées : {X.shape[0]} échantillons, {X.shape[1]} features")
    return X, y


def train_naive_bayes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    output_path: Path,
) -> GaussianNB:
    """
    Entraîne le classifieur Naïves Bayes Gaussien.

    Args:
        X_train: Matrice de features d'entraînement
        y_train: Labels d'entraînement
        output_path: Dossier de sauvegarde du modèle

    Returns:
        Classifieur entraîné
    """
    # Normalisation des features avant classification
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    classifier = GaussianNB()
    classifier.fit(X_train_scaled, y_train)

    # Sauvegarde du modèle et du scaler pour réutilisation
    output_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, output_path / "naive_bayes_model.pkl")
    joblib.dump(scaler, output_path / "scaler.pkl")
    logger.info(f"Modèle Naïves Bayes sauvegardé dans {output_path}")

    return classifier, scaler


def evaluate_naive_bayes(
    classifier: GaussianNB,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Évalue le classifieur sur les données de test.

    Args:
        classifier: Modèle Naïves Bayes entraîné
        scaler: Scaler ajusté sur les données d'entraînement
        X_test: Features de test
        y_test: Labels de test

    Returns:
        Dictionnaire contenant les métriques d'évaluation
    """
    X_test_scaled = scaler.transform(X_test)
    y_pred = classifier.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"Accuracy Naïves Bayes : {accuracy:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": y_pred,
    }


def run_naive_bayes(input_path: Path, output_path: Path) -> Dict[str, Any]:
    """
    Fonction principale : entraîne et évalue le classifieur Naïves Bayes.

    Args:
        input_path: Dossier contenant les données prétraitées
        output_path: Dossier de sauvegarde des résultats

    Returns:
        Dictionnaire des résultats d'évaluation
    """
    logger.info("Démarrage du classifieur Naïves Bayes...")

    X, y = load_features(input_path)

    # Division train/test (80% / 20%) avec stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Entraînement : {len(X_train)} | Test : {len(X_test)}")

    classifier, scaler = train_naive_bayes(X_train, y_train, output_path)
    results = evaluate_naive_bayes(classifier, scaler, X_test, y_test)

    return results
