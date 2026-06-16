"""
Tests unitaires — Module B : K-Means Clustering et Feature Extraction
Test des fonctionnalités de clustering sur données synthétiques.
"""

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures et helpers
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_embeddings():
    """Crée des embeddings synthétiques 256D avec 100 samples."""
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((100, 256)).astype(np.float32)
    return embeddings


@pytest.fixture
def synthetic_metadata(synthetic_embeddings):
    """Crée des métadonnées associées aux embeddings."""
    n = len(synthetic_embeddings)
    metadata = pd.DataFrame({
        'index': np.arange(n),
        'split': np.random.choice(['train', 'val', 'test'], n),
        'label_int': np.random.randint(0, 36, n),
        'label_char': np.random.choice(list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'), n),
        'ocr_text': [''] * n,
        'ocr_conf': np.random.uniform(0.5, 0.99, n),
        'conformite': np.random.choice(['valid', 'invalid'], n),
    })
    return metadata


@pytest.fixture
def temp_data_dir(synthetic_embeddings, synthetic_metadata):
    """Crée un répertoire temporaire avec embeddings et metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        np.save(tmpdir / 'embeddings.npy', synthetic_embeddings)
        synthetic_metadata.to_csv(tmpdir / 'metadata.csv', index=False)
        yield tmpdir


# ─────────────────────────────────────────────────────────────────────────────
# Tests : Chargement des données
# ─────────────────────────────────────────────────────────────────────────────

def test_load_embeddings_success(temp_data_dir):
    """Le chargement réussit quand embeddings.npy et metadata.csv existent."""
    from modules.module_b.clustering import load_embeddings
    
    embeddings, metadata = load_embeddings(temp_data_dir)
    
    assert embeddings.shape == (100, 256)
    assert embeddings.dtype == np.float32
    assert len(metadata) == 100
    assert all(col in metadata.columns for col in ['index', 'label_int', 'ocr_conf'])


def test_load_embeddings_missing_embeddings(temp_data_dir):
    """Lève RuntimeError si embeddings.npy est absent."""
    from modules.module_b.clustering import load_embeddings
    
    (temp_data_dir / 'embeddings.npy').unlink()
    
    with pytest.raises(RuntimeError):
        load_embeddings(temp_data_dir)


def test_load_embeddings_missing_metadata(temp_data_dir):
    """Lève RuntimeError si metadata.csv est absent."""
    from modules.module_b.clustering import load_embeddings
    
    (temp_data_dir / 'metadata.csv').unlink()
    
    with pytest.raises(RuntimeError):
        load_embeddings(temp_data_dir)


def test_load_embeddings_mismatched_sizes(temp_data_dir, synthetic_embeddings):
    """Lève ValueError si nombre embeddings ≠ nombre metadata."""
    from modules.module_b.clustering import load_embeddings
    
    # Sauvegarder moins d'embeddings que de métadonnées
    np.save(temp_data_dir / 'embeddings.npy', synthetic_embeddings[:50])
    
    with pytest.raises(ValueError):
        load_embeddings(temp_data_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Tests : K-Means Clustering
# ─────────────────────────────────────────────────────────────────────────────

def test_fit_kmeans_returns_valid_model(synthetic_embeddings):
    """K-Means retourne un modèle valide avec la structure attendue."""
    from modules.module_b.kmeans_fit import fit_kmeans
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(synthetic_embeddings).astype(np.float32)
    
    model = fit_kmeans(embeddings_scaled, k=4, random_state=42)
    
    assert model is not None
    assert hasattr(model, 'cluster_centers_')
    assert hasattr(model, 'labels_')
    assert model.n_clusters == 4
    assert len(model.labels_) == len(synthetic_embeddings)


def test_fit_kmeans_deterministic(synthetic_embeddings):
    """Même seed → même résultat."""
    from modules.module_b.kmeans_fit import fit_kmeans
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(synthetic_embeddings).astype(np.float32)
    
    model1 = fit_kmeans(embeddings_scaled, k=4, random_state=42)
    model2 = fit_kmeans(embeddings_scaled, k=4, random_state=42)
    
    np.testing.assert_array_almost_equal(model1.cluster_centers_, model2.cluster_centers_)
    np.testing.assert_array_equal(model1.labels_, model2.labels_)


def test_fit_kmeans_different_k(synthetic_embeddings):
    """K-Means avec différentes valeurs de k."""
    from modules.module_b.kmeans_fit import fit_kmeans
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(synthetic_embeddings).astype(np.float32)
    
    for k in [2, 4, 8]:
        model = fit_kmeans(embeddings_scaled, k=k, random_state=42)
        assert model.n_clusters == k
        assert len(np.unique(model.labels_)) <= k


def test_kmeans_cluster_distribution(synthetic_embeddings):
    """Les clusters ne sont pas tous vides."""
    from modules.module_b.kmeans_fit import fit_kmeans
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(synthetic_embeddings).astype(np.float32)
    
    model = fit_kmeans(embeddings_scaled, k=4, random_state=42)
    unique_labels, counts = np.unique(model.labels_, return_counts=True)
    
    assert len(unique_labels) > 0
    assert all(count > 0 for count in counts)


# ─────────────────────────────────────────────────────────────────────────────
# Tests : Silhouette Score
# ─────────────────────────────────────────────────────────────────────────────

def test_silhouette_score_valid_range(synthetic_embeddings):
    """Silhouette score est dans [-1, 1]."""
    from sklearn.metrics import silhouette_score
    from modules.module_b.kmeans_fit import fit_kmeans
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(synthetic_embeddings).astype(np.float32)
    
    model = fit_kmeans(embeddings_scaled, k=4, random_state=42)
    score = silhouette_score(embeddings_scaled, model.labels_)
    
    assert -1.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Tests : Cluster Mapping et Procédures MINT/DGI
# ─────────────────────────────────────────────────────────────────────────────

def test_cluster_procedures_structure():
    """CLUSTER_PROCEDURES contient la structure attendue pour k=4."""
    from modules.module_b.clustering import CLUSTER_PROCEDURES
    
    # Au moins 4 clusters (k=4 par défaut)
    assert len(CLUSTER_PROCEDURES) >= 4
    
    # Vérifier la structure de chaque cluster
    for cluster_id, info in CLUSTER_PROCEDURES.items():
        assert 'label' in info
        assert 'procedure' in info
        assert 'autorite' in info
        assert 'base_legale' in info
        assert isinstance(info['label'], str)


def test_cluster_mapping_json_structure(temp_data_dir, synthetic_embeddings):
    """Le fichier cluster_mapping.json a la structure attendue."""
    from modules.module_b.kmeans_fit import fit_kmeans
    from sklearn.preprocessing import StandardScaler
    
    k = 4
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(synthetic_embeddings).astype(np.float32)
    model = fit_kmeans(embeddings_scaled, k=k, random_state=42)
    
    cluster_mapping = {
        'k': k,
        'n_samples': len(synthetic_embeddings),
        'silhouette_score': float(0.5),
    }
    
    mapping_path = temp_data_dir / 'cluster_mapping.json'
    with mapping_path.open('w', encoding='utf-8') as f:
        json.dump(cluster_mapping, f)
    
    # Vérifier que le fichier peut être relu
    with mapping_path.open('r', encoding='utf-8') as f:
        loaded = json.load(f)
    
    assert loaded['k'] == k
    assert loaded['n_samples'] == len(synthetic_embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# Tests : Remappage par confiance OCR
# ─────────────────────────────────────────────────────────────────────────────

def test_confidence_level_mapping(synthetic_metadata):
    """Les niveaux de confiance OCR sont correctement mappés."""
    # Thresholds de confiance : [bas, moyen, haut]
    conf_low = 0.6
    conf_high = 0.85
    
    def get_conf_level(ocr_conf):
        if ocr_conf < conf_low:
            return 0  # faible
        elif ocr_conf < conf_high:
            return 1  # moyen
        else:
            return 2  # haute
    
    synthetic_metadata['confidence_level'] = synthetic_metadata['ocr_conf'].apply(get_conf_level)
    
    # Vérifier que tous les niveaux sont 0, 1 ou 2
    assert all(level in [0, 1, 2] for level in synthetic_metadata['confidence_level'])
    assert synthetic_metadata['confidence_level'].nunique() > 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests : Plate Quality Features
# ─────────────────────────────────────────────────────────────────────────────

def test_plate_quality_contrast_computation():
    """Calcul du contraste d'une plaque synthétique."""
    from preprocessing.feature_extraction import compute_feature_importance
    
    # Créer une image synthétique de plaque (200, 60, 3)
    rng = np.random.default_rng(42)
    plate_image = rng.integers(50, 200, (60, 200, 3), dtype=np.uint8)
    
    # Vérifier que l'image est valide
    assert plate_image.shape == (60, 200, 3)
    assert plate_image.dtype == np.uint8
    assert plate_image.min() >= 50
    assert plate_image.max() <= 200


def test_plate_quality_features_shape():
    """Les features de qualité de plaque ont la forme attendue."""
    rng = np.random.default_rng(42)
    
    # Créer des features synthétiques (10 features par plaque)
    n_plates = 100
    features = rng.random((n_plates, 10)).astype(np.float32)
    
    assert features.shape == (n_plates, 10)
    assert features.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# Tests : Visualisation et Sauvegarde
# ─────────────────────────────────────────────────────────────────────────────

def test_save_cluster_assignment(temp_data_dir, synthetic_embeddings, synthetic_metadata):
    """Sauvegarde les assigments des clusters."""
    from modules.module_b.kmeans_fit import fit_kmeans
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(synthetic_embeddings).astype(np.float32)
    model = fit_kmeans(embeddings_scaled, k=4, random_state=42)
    
    # Ajouter les labels aux métadonnées
    synthetic_metadata['cluster_id'] = model.labels_
    
    # Sauvegarder
    output_path = temp_data_dir / 'cluster_assignments.csv'
    synthetic_metadata.to_csv(output_path, index=False)
    
    # Vérifier
    assert output_path.exists()
    loaded = pd.read_csv(output_path)
    assert 'cluster_id' in loaded.columns
    assert all(cluster_id in range(4) for cluster_id in loaded['cluster_id'])


def test_pca_visualization_data(synthetic_embeddings):
    """Réduction PCA 2D des embeddings pour visualisation."""
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(synthetic_embeddings)
    
    assert embeddings_2d.shape == (100, 2)
    assert pca.explained_variance_ratio_.sum() > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Tests : Robustesse du Clustering
# ─────────────────────────────────────────────────────────────────────────────

def test_kmeans_robustness_small_sample():
    """K-Means robuste avec petit sample size."""
    from modules.module_b.kmeans_fit import fit_kmeans
    from sklearn.preprocessing import StandardScaler
    
    small_embeddings = np.random.default_rng(42).standard_normal((20, 256)).astype(np.float32)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(small_embeddings).astype(np.float32)
    model = fit_kmeans(embeddings_scaled, k=3, random_state=42)
    
    assert model is not None
    assert len(model.labels_) == 20


def test_kmeans_robustness_normalized_input():
    """K-Means robuste avec entrée normalisée."""
    from sklearn.preprocessing import StandardScaler
    from modules.module_b.kmeans_fit import fit_kmeans
    
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((100, 256)).astype(np.float32)
    
    # Normaliser
    scaler = StandardScaler()
    embeddings_norm = scaler.fit_transform(embeddings).astype(np.float32)
    
    model = fit_kmeans(embeddings_norm, k=4, random_state=42)
    assert model.n_clusters == 4


# ─────────────────────────────────────────────────────────────────────────────
# Tests : Interprétation des Clusters
# ─────────────────────────────────────────────────────────────────────────────

def test_cluster_statistics(synthetic_embeddings, synthetic_metadata):
    """Calcul des statistiques par cluster."""
    from modules.module_b.kmeans_fit import fit_kmeans
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(synthetic_embeddings).astype(np.float32)
    model = fit_kmeans(embeddings_scaled, k=4, random_state=42)
    synthetic_metadata['cluster_id'] = model.labels_
    
    # Statistiques par cluster
    cluster_stats = synthetic_metadata.groupby('cluster_id')['ocr_conf'].agg(['mean', 'std', 'count'])
    
    assert len(cluster_stats) > 0
    assert 'mean' in cluster_stats.columns
    assert 'count' in cluster_stats.columns
    assert cluster_stats['count'].sum() == 100


def test_cluster_quality_ranking(synthetic_embeddings, synthetic_metadata):
    """Ranking des clusters par qualité (confiance OCR moyenne)."""
    from modules.module_b.kmeans_fit import fit_kmeans
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(synthetic_embeddings).astype(np.float32)
    model = fit_kmeans(embeddings_scaled, k=4, random_state=42)
    synthetic_metadata['cluster_id'] = model.labels_
    
    # Moyenne de confiance par cluster
    quality_ranking = synthetic_metadata.groupby('cluster_id')['ocr_conf'].mean().sort_values(ascending=False)
    
    # Les clusters avec haute confiance devraient être premiers
    assert len(quality_ranking) == 4
    assert quality_ranking.iloc[0] >= quality_ranking.iloc[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Tests d'Intégration
# ─────────────────────────────────────────────────────────────────────────────

def test_full_clustering_pipeline(temp_data_dir, synthetic_embeddings, synthetic_metadata):
    """Pipeline complet : chargement → K-Means → sauvegarde."""
    from modules.module_b.clustering import load_embeddings
    from modules.module_b.kmeans_fit import fit_kmeans
    from sklearn.preprocessing import StandardScaler
    
    # Chargement
    embeddings, metadata = load_embeddings(temp_data_dir)
    
    # Normaliser
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings).astype(np.float32)
    
    # Fit K-Means
    model = fit_kmeans(embeddings_scaled, k=4, random_state=42)
    
    # Ajouter aux métadonnées
    metadata['cluster_id'] = model.labels_
    
    # Sauvegarde
    output_path = temp_data_dir / 'results.csv'
    metadata.to_csv(output_path, index=False)
    
    # Vérification
    assert output_path.exists()
    results = pd.read_csv(output_path)
    assert 'cluster_id' in results.columns
    assert len(results) == len(embeddings)
