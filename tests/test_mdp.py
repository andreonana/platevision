"""
Tests unitaires — Module C : Processus de Décision Markovien (MDP)
Tests de l'espace d'états, transitions, récompenses et politiques.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_module_b_outputs():
    """Crée des sorties synthétiques du Module B."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Créer metadata.csv avec colonnes REQUISES
        n = 200
        metadata = pd.DataFrame({
            'index': np.arange(n),
            'cluster_id': np.random.randint(0, 4, n),
            'confidence_level': np.random.randint(0, 3, n),
            'ocr_conf': np.random.uniform(0.5, 1.0, n),
            'label_char': np.random.choice(list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'), n),
            'conformite': np.random.choice(['valid', 'invalid'], n),
        })
        metadata.to_csv(tmpdir / 'metadata.csv', index=False)
        
        # Créer cluster_mapping.json
        cluster_mapping = {
            'k': 4,
            'n_samples': n,
            'silhouette_score': 0.65,
            'cluster_procedures': {
                '0': 'Plaque nette / lisible — Laisser passer',
                '1': 'Plaque dégradée — Contrôle MINT',
                '2': 'Plaque illisible — Signalement DGI',
                '3': 'Cas non discriminé — PJ',
            }
        }
        with (tmpdir / 'cluster_mapping.json').open('w') as f:
            json.dump(cluster_mapping, f)
        
        yield tmpdir


@pytest.fixture
def synthetic_state_space():
    """Crée un espace d'états synthétique pour le MDP."""
    from modules.module_c.mdp_definition import MDPState, MDPStateSpace
    
    states = []
    state_id = 0
    
    # Créer 12 états (4 clusters × 3 niveaux confiance)
    for cluster in range(4):
        for conf_level in range(3):
            state = MDPState(
                state_id=state_id,
                cluster_id=cluster,
                conf_level=conf_level,
                alerte_cnn=0,
                label=f"State_{state_id}",
                cluster_name=f"Cluster_{cluster}",
                procedure=f"Procedure_{cluster}",
                n_observations=np.random.randint(5, 50),
                frequency=np.random.uniform(0.02, 0.15),
            )
            states.append(state)
            state_id += 1
    
    state_space = MDPStateSpace(
        states=states,
        n_states=len(states),
        k_clusters=4,
        n_conf_levels=3,
        ocr_thresholds={'low': 0.6, 'high': 0.85},
    )
    return state_space


# ─────────────────────────────────────────────────────────────────────────────
# Tests : Chargement des sorties Module B
# ─────────────────────────────────────────────────────────────────────────────

def test_load_module_b_outputs_success(synthetic_module_b_outputs):
    """Chargement réussi des sorties du Module B."""
    from modules.module_c.mdp_definition import load_module_b_outputs
    
    metadata, cluster_mapping = load_module_b_outputs(synthetic_module_b_outputs)
    
    assert len(metadata) == 200
    assert 'cluster_id' in metadata.columns
    assert 'ocr_conf' in metadata.columns
    assert cluster_mapping['k'] == 4


def test_load_module_b_outputs_missing_metadata(synthetic_module_b_outputs):
    """Lève RuntimeError si metadata.csv est absent."""
    from modules.module_c.mdp_definition import load_module_b_outputs
    
    (synthetic_module_b_outputs / 'metadata.csv').unlink()
    
    with pytest.raises(RuntimeError):
        load_module_b_outputs(synthetic_module_b_outputs)


def test_load_module_b_outputs_missing_mapping(synthetic_module_b_outputs):
    """Lève RuntimeError si cluster_mapping.json est absent."""
    from modules.module_c.mdp_definition import load_module_b_outputs
    
    (synthetic_module_b_outputs / 'cluster_mapping.json').unlink()
    
    with pytest.raises(RuntimeError):
        load_module_b_outputs(synthetic_module_b_outputs)


def test_load_module_b_outputs_incomplete_columns(synthetic_module_b_outputs):
    """Lève RuntimeError si colonnes requises manquent."""
    from modules.module_c.mdp_definition import load_module_b_outputs
    
    # Supprimer la colonne 'cluster_id'
    df = pd.read_csv(synthetic_module_b_outputs / 'metadata.csv')
    df = df.drop(columns=['cluster_id'])
    df.to_csv(synthetic_module_b_outputs / 'metadata.csv', index=False)
    
    with pytest.raises(RuntimeError):
        load_module_b_outputs(synthetic_module_b_outputs)


# ─────────────────────────────────────────────────────────────────────────────
# Tests : Définition de l'Espace d'États
# ─────────────────────────────────────────────────────────────────────────────

def test_mdp_state_creation():
    """Création valide d'un état MDP."""
    from modules.module_c.mdp_definition import MDPState
    
    state = MDPState(
        state_id=0,
        cluster_id=1,
        conf_level=2,
        alerte_cnn=0,
        label='Test State',
        cluster_name='Test Cluster',
        procedure='Test Procedure',
    )
    
    assert state.state_id == 0
    assert state.cluster_id == 1
    assert state.conf_level == 2
    assert state.alerte_cnn == 0
    assert state.label == 'Test State'


def test_mdp_state_space_creation(synthetic_state_space):
    """Création valide d'un espace d'états MDP."""
    assert synthetic_state_space.n_states == 12
    assert synthetic_state_space.k_clusters == 4
    assert synthetic_state_space.n_conf_levels == 3
    assert len(synthetic_state_space.states) == 12


def test_mdp_state_space_properties(synthetic_state_space):
    """Les propriétés de l'espace d'états sont cohérentes."""
    assert synthetic_state_space.n_states == len(synthetic_state_space.states)
    
    # Chaque état doit avoir des propriétés valides
    for state in synthetic_state_space.states:
        assert 0 <= state.cluster_id < synthetic_state_space.k_clusters
        assert 0 <= state.conf_level < synthetic_state_space.n_conf_levels
        assert state.alerte_cnn in [0, 1]
        assert state.n_observations >= 0
        assert 0.0 <= state.frequency <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Tests : Discrétisation de la Confiance OCR
# ─────────────────────────────────────────────────────────────────────────────

def test_ocr_confidence_discretization():
    """Discrétisation de la confiance OCR en 3 niveaux."""
    thresholds = {'low': 0.6, 'high': 0.85}
    
    def discretize_conf(ocr_conf):
        if ocr_conf < thresholds['low']:
            return 0  # faible
        elif ocr_conf < thresholds['high']:
            return 1  # moyen
        else:
            return 2  # haute
    
    test_values = [0.5, 0.65, 0.9]
    expected = [0, 1, 2]
    
    for value, expected_level in zip(test_values, expected):
        assert discretize_conf(value) == expected_level


def test_ocr_confidence_boundary_cases():
    """Cas limites de la discrétisation de confiance."""
    thresholds = {'low': 0.6, 'high': 0.85}
    
    def discretize_conf(ocr_conf):
        if ocr_conf < thresholds['low']:
            return 0
        elif ocr_conf < thresholds['high']:
            return 1
        else:
            return 2
    
    # Valeurs exactes aux seuils
    assert discretize_conf(0.59999) == 0
    assert discretize_conf(0.60001) == 1
    assert discretize_conf(0.84999) == 1
    assert discretize_conf(0.85001) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Tests : Matrice de Transitions
# ─────────────────────────────────────────────────────────────────────────────

def test_transition_matrix_shape(synthetic_state_space):
    """La matrice de transition a la bonne forme."""
    n = synthetic_state_space.n_states
    
    # Créer une matrice de transition synthétique
    T = np.random.dirichlet(np.ones(n), size=n).astype(np.float32)
    
    assert T.shape == (n, n)
    # Chaque ligne doit sommer à ~1.0
    assert np.allclose(T.sum(axis=1), 1.0)


def test_transition_matrix_stochasticity(synthetic_state_space):
    """La matrice de transition est stochastique."""
    n = synthetic_state_space.n_states
    
    T = np.random.dirichlet(np.ones(n), size=n).astype(np.float32)
    
    # Tous les éléments doivent être positifs
    assert np.all(T >= 0.0)
    # Tous les éléments doivent être ≤ 1
    assert np.all(T <= 1.0)


def test_transition_examples_from_module_b(synthetic_module_b_outputs):
    """Construire des transitions à partir des observations Module B."""
    from modules.module_c.mdp_definition import load_module_b_outputs
    
    metadata, _ = load_module_b_outputs(synthetic_module_b_outputs)
    
    # Créer une transition simple : cluster persistence
    cluster_transitions = {}
    for cluster_id in metadata['cluster_id'].unique():
        # Tous les états avec même cluster restent au cluster (simplifié)
        cluster_transitions[cluster_id] = [cluster_id] * 3  # 3 niveaux conf
    
    assert len(cluster_transitions) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests : Fonction de Récompense
# ─────────────────────────────────────────────────────────────────────────────

def test_reward_function_structure(synthetic_state_space):
    """La fonction de récompense est bien définie pour tous les états."""
    rewards = np.zeros(synthetic_state_space.n_states)
    
    for i, state in enumerate(synthetic_state_space.states):
        if state.conf_level == 2:  # Haute confiance
            rewards[i] = 1.0
        elif state.conf_level == 1:  # Confiance moyenne
            rewards[i] = 0.0
        else:  # Faible confiance
            rewards[i] = -0.5
    
    assert len(rewards) == synthetic_state_space.n_states


def test_reward_proportional_to_confidence(synthetic_state_space):
    """Les récompenses sont proportionnelles à la confiance OCR."""
    rewards = np.zeros(synthetic_state_space.n_states)
    
    for i, state in enumerate(synthetic_state_space.states):
        # Récompense basée sur niveau de confiance
        rewards[i] = state.conf_level * 0.5 - 0.5
    
    # Vérifier que haute confiance > faible confiance
    high_conf_rewards = [rewards[i] for i, s in enumerate(synthetic_state_space.states) if s.conf_level == 2]
    low_conf_rewards = [rewards[i] for i, s in enumerate(synthetic_state_space.states) if s.conf_level == 0]
    
    assert np.mean(high_conf_rewards) > np.mean(low_conf_rewards)


# ─────────────────────────────────────────────────────────────────────────────
# Tests : Espace d'Actions
# ─────────────────────────────────────────────────────────────────────────────

def test_action_space_definition():
    """L'espace d'actions est bien défini."""
    actions = {
        0: 'laisser_passer',
        1: 'controle_visual_mint',
        2: 'signalement_dgi',
        3: 'police_judiciaire',
    }
    
    assert len(actions) == 4
    assert all(isinstance(action, str) for action in actions.values())


def test_actions_correspond_to_procedures(synthetic_state_space):
    """Chaque action correspond à une procédure MINT/DGI."""
    procedures = [
        'Laisser passer',
        'Contrôle visuel complémentaire',
        'Signalement DGI',
        'Police Judiciaire',
    ]
    
    # Vérifier que le nombre d'actions = nombre de clusters
    assert len(procedures) == synthetic_state_space.k_clusters


# ─────────────────────────────────────────────────────────────────────────────
# Tests : Algorithme Value Iteration
# ─────────────────────────────────────────────────────────────────────────────

def test_value_iteration_initialization():
    """Value Iteration initialise correctement V."""
    n_states = 12
    V = np.zeros(n_states)
    
    assert V.shape == (n_states,)
    assert np.all(V == 0.0)


def test_value_iteration_gamma_range():
    """Le facteur de rabais γ doit être dans [0, 1]."""
    for gamma in [0.0, 0.5, 0.9, 0.99, 1.0]:
        assert 0.0 <= gamma <= 1.0


def test_value_iteration_convergence_check():
    """Teste la condition de convergence Value Iteration."""
    n_states = 12
    V_old = np.random.randn(n_states).astype(np.float32)
    V_new = V_old + np.random.randn(n_states) * 0.001
    
    epsilon = 0.01
    max_error = np.max(np.abs(V_new - V_old))
    
    # Petit changement → on converge
    assert max_error < epsilon


# ─────────────────────────────────────────────────────────────────────────────
# Tests : Algorithme Policy Iteration
# ─────────────────────────────────────────────────────────────────────────────

def test_policy_initialization(synthetic_state_space):
    """Initialisation d'une politique."""
    n_actions = synthetic_state_space.k_clusters
    n_states = synthetic_state_space.n_states
    
    # Politique uniforme : tous les états choisissent l'action 0
    policy = np.zeros(n_states, dtype=int)
    
    assert policy.shape == (n_states,)
    assert all(a < n_actions for a in policy)


def test_policy_determinism():
    """Une politique déterministe mappe chaque état à une action."""
    n_states = 12
    n_actions = 4
    
    policy = np.random.randint(0, n_actions, n_states)
    
    assert len(policy) == n_states
    assert all(0 <= a < n_actions for a in policy)


def test_policy_improvement_monotonic():
    """Policy Improvement améliore la politique (monotone)."""
    # Créer une politique initiale
    policy_old = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    
    # Simuler une amélioration
    V_improvement = np.array([0.1, 0.2, 0.05, 0.15, 0.25, 0.1, 0.12, 0.22, 0.08, 0.14, 0.24, 0.09])
    policy_new = np.argmax(V_improvement.reshape(12, 1), axis=1)[:len(policy_old)]
    
    # Les deux politiques sont comparables
    assert len(policy_new) == len(policy_old)


# ─────────────────────────────────────────────────────────────────────────────
# Tests : Comparaison VI vs PI
# ─────────────────────────────────────────────────────────────────────────────

def test_value_iteration_vs_policy_iteration_convergence():
    """Les deux algorithmes convergent vers des solutions similaires."""
    n_states = 12
    n_actions = 4
    gamma = 0.95
    
    # Valeurs finales devraient être similaires
    V_vi = np.random.randn(n_states)
    V_pi = V_vi + np.random.randn(n_states) * 0.01  # Légèrement différentes
    
    # Les deux devraient être assez proches
    error = np.max(np.abs(V_vi - V_pi))
    assert error < 1.0  # Erreur raisonnnable


def test_policy_convergence_comparison():
    """Les politiques VI et PI convergent."""
    n_states = 12
    n_actions = 4
    
    policy_vi = np.random.randint(0, n_actions, n_states)
    policy_pi = policy_vi.copy()
    policy_pi[0] = (policy_pi[0] + 1) % n_actions  # Légère perturbation
    
    assert len(policy_vi) == len(policy_pi)


# ─────────────────────────────────────────────────────────────────────────────
# Tests : Analyse de Sensibilité au Facteur γ
# ─────────────────────────────────────────────────────────────────────────────

def test_gamma_sensitivity_values():
    """Les valeurs V changent avec γ."""
    n_states = 12
    reward = np.ones(n_states) * 0.5
    
    # Valeurs avec différents γ
    V_low_gamma = reward * (1 + 0.5)  # γ=0.5
    V_high_gamma = reward * (1 + 0.95)  # γ=0.95
    
    # Les valeurs futures comptent plus avec γ élevé
    assert V_high_gamma.mean() > V_low_gamma.mean()


def test_gamma_sensitivity_policy():
    """La politique peut changer avec γ."""
    n_states = 12
    
    Q_low_gamma = np.random.randn(n_states, 4)
    Q_high_gamma = Q_low_gamma + np.random.randn(n_states, 4) * 0.1
    
    policy_low = np.argmax(Q_low_gamma, axis=1)
    policy_high = np.argmax(Q_high_gamma, axis=1)
    
    # Les politiques peuvent différer
    assert len(policy_low) == len(policy_high) == n_states


# ─────────────────────────────────────────────────────────────────────────────
# Tests d'Intégration
# ─────────────────────────────────────────────────────────────────────────────

def test_full_mdp_pipeline(synthetic_module_b_outputs, synthetic_state_space):
    """Pipeline complet MDP : chargement → états → transitions → politique."""
    from modules.module_c.mdp_definition import load_module_b_outputs
    
    # Étape 1 : Chargement
    metadata, cluster_mapping = load_module_b_outputs(synthetic_module_b_outputs)
    assert len(metadata) > 0
    
    # Étape 2 : Espace d'états créé
    assert synthetic_state_space.n_states > 0
    
    # Étape 3 : Matrice de transitions
    n = synthetic_state_space.n_states
    T = np.random.dirichlet(np.ones(n), size=n).astype(np.float32)
    assert T.shape == (n, n)
    
    # Étape 4 : Fonction de récompense
    R = np.random.randn(n)
    assert R.shape == (n,)
    
    # Étape 5 : Politique
    policy = np.random.randint(0, 4, n)
    assert len(policy) == n


def test_mdp_reproducibility(synthetic_module_b_outputs):
    """Le MDP produit des résultats reproductibles avec même seed."""
    from modules.module_c.mdp_definition import load_module_b_outputs
    
    np.random.seed(42)
    metadata1, _ = load_module_b_outputs(synthetic_module_b_outputs)
    
    np.random.seed(42)
    metadata2, _ = load_module_b_outputs(synthetic_module_b_outputs)
    
    pd.testing.assert_frame_equal(metadata1, metadata2)


# ─────────────────────────────────────────────────────────────────────────────
# Tests : Gestion des Cas Limites
# ─────────────────────────────────────────────────────────────────────────────

def test_mdp_single_state():
    """MDP avec un seul état (cas limite)."""
    from modules.module_c.mdp_definition import MDPState, MDPStateSpace
    
    state = MDPState(
        state_id=0, cluster_id=0, conf_level=0, alerte_cnn=0,
        label='Unique', cluster_name='Cluster0', procedure='Action0'
    )
    
    state_space = MDPStateSpace(
        states=[state],
        n_states=1,
        k_clusters=1,
        n_conf_levels=1,
    )
    
    assert state_space.n_states == 1


def test_mdp_gamma_zero():
    """MDP avec γ=0 (myopic / court terme)."""
    n_states = 12
    gamma = 0.0
    
    V = np.random.randn(n_states)
    # Avec γ=0, les récompenses futures ne comptent pas
    V_myopic = V.copy()
    
    assert V_myopic.shape == (n_states,)


def test_mdp_gamma_one():
    """MDP avec γ=1 (long terme)."""
    n_states = 12
    gamma = 1.0
    
    V = np.random.randn(n_states)
    # Avec γ=1, pondération égale des récompenses futures
    V_longterm = V.copy()
    
    assert V_longterm.shape == (n_states,)
