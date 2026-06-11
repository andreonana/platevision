"""Tests unitaires pour le Module C — MDP, Value Iteration, Policy Iteration."""

import numpy as np
import pytest

from modules.module_c.mdp_definition import (
    build_transition_matrix,
    build_reward_matrix,
    get_mdp,
    N_STATES,
    N_ACTIONS,
    STATES,
    ACTIONS,
)
from modules.module_c.value_iteration import value_iteration
from modules.module_c.policy_iteration import policy_evaluation, policy_improvement, policy_iteration


class TestMDPDefinition:
    """Tests de la définition du MDP."""

    def test_transition_matrix_shape(self):
        T = build_transition_matrix()
        assert T.shape == (N_STATES, N_ACTIONS, N_STATES)

    def test_transition_matrix_sums_to_one(self):
        T = build_transition_matrix()
        for s in range(N_STATES):
            for a in range(N_ACTIONS):
                total = T[s, a, :].sum()
                assert abs(total - 1.0) < 1e-6, f"T[{s},{a},:] somme à {total}"

    def test_transition_matrix_non_negative(self):
        T = build_transition_matrix()
        assert (T >= 0).all()

    def test_reward_matrix_shape(self):
        R = build_reward_matrix()
        assert R.shape == (N_STATES, N_ACTIONS)

    def test_states_count(self):
        assert len(STATES) == N_STATES

    def test_actions_count(self):
        assert len(ACTIONS) == N_ACTIONS

    def test_state_ids_unique(self):
        ids = [s.id for s in STATES]
        assert len(ids) == len(set(ids))

    def test_action_ids_unique(self):
        ids = [a.id for a in ACTIONS]
        assert len(ids) == len(set(ids))

    def test_get_mdp_keys(self):
        mdp = get_mdp()
        required_keys = {"states", "actions", "n_states", "n_actions", "T", "R"}
        assert required_keys.issubset(mdp.keys())

    def test_valid_plate_state_laisser_passer_reward(self):
        """Laisser passer une plaque valide doit avoir une récompense positive."""
        R = build_reward_matrix()
        # État 0 = PLAQUE_VALIDE, Action 0 = LAISSER_PASSER
        assert R[0, 0] > 0, "Laisser passer une plaque valide doit être récompensé"

    def test_verbaliser_plaque_valide_negative_reward(self):
        """Verbaliser une plaque valide doit être très pénalisé."""
        R = build_reward_matrix()
        # État 0 = PLAQUE_VALIDE, Action 3 = VERBALISER
        assert R[0, 3] < -10, "Verbaliser abusivement doit avoir une forte pénalité"


class TestValueIteration:
    """Tests de l'algorithme Value Iteration."""

    def setup_method(self):
        mdp = get_mdp()
        self.T = mdp["T"]
        self.R = mdp["R"]

    def test_output_shapes(self):
        V, policy, history = value_iteration(self.T, self.R, gamma=0.9)
        assert V.shape == (N_STATES,)
        assert policy.shape == (N_STATES,)
        assert len(history) > 0

    def test_policy_valid_actions(self):
        _, policy, _ = value_iteration(self.T, self.R, gamma=0.9)
        assert policy.min() >= 0
        assert policy.max() < N_ACTIONS

    def test_convergence_history_decreasing(self):
        _, _, history = value_iteration(self.T, self.R, gamma=0.9)
        # La convergence doit être globalement décroissante
        assert history[0] >= history[-1]

    def test_gamma_effect(self):
        """Un γ plus élevé donne des valeurs plus grandes (horizon plus long)."""
        V_low, _, _ = value_iteration(self.T, self.R, gamma=0.5)
        V_high, _, _ = value_iteration(self.T, self.R, gamma=0.95)
        # Les valeurs absolues tendent à être plus grandes avec un γ élevé
        assert np.abs(V_high).mean() >= np.abs(V_low).mean()

    def test_deterministic_results(self):
        """Mêmes entrées → mêmes résultats."""
        V1, policy1, _ = value_iteration(self.T, self.R, gamma=0.9)
        V2, policy2, _ = value_iteration(self.T, self.R, gamma=0.9)
        np.testing.assert_array_equal(policy1, policy2)
        np.testing.assert_allclose(V1, V2, atol=1e-8)


class TestPolicyIteration:
    """Tests de l'algorithme Policy Iteration."""

    def setup_method(self):
        mdp = get_mdp()
        self.T = mdp["T"]
        self.R = mdp["R"]

    def test_output_shapes(self):
        V, policy, n_iter = policy_iteration(self.T, self.R, gamma=0.9)
        assert V.shape == (N_STATES,)
        assert policy.shape == (N_STATES,)
        assert n_iter >= 1

    def test_policy_valid_actions(self):
        _, policy, _ = policy_iteration(self.T, self.R, gamma=0.9)
        assert policy.min() >= 0
        assert policy.max() < N_ACTIONS

    def test_policy_evaluation_shape(self):
        policy = np.zeros(N_STATES, dtype=int)
        V = policy_evaluation(policy, self.T, self.R, gamma=0.9)
        assert V.shape == (N_STATES,)


class TestVIvsPIComparison:
    """Tests de cohérence entre Value Iteration et Policy Iteration."""

    def test_policies_agree(self):
        """VI et PI doivent converger vers la même politique optimale."""
        mdp = get_mdp()
        T, R = mdp["T"], mdp["R"]

        V_vi, policy_vi, _ = value_iteration(T, R, gamma=0.9)
        V_pi, policy_pi, _ = policy_iteration(T, R, gamma=0.9)

        np.testing.assert_array_equal(policy_vi, policy_pi)

    def test_value_functions_close(self):
        """Les fonctions de valeur de VI et PI doivent être proches."""
        mdp = get_mdp()
        T, R = mdp["T"], mdp["R"]

        V_vi, _, _ = value_iteration(T, R, gamma=0.9)
        V_pi, _, _ = policy_iteration(T, R, gamma=0.9)

        np.testing.assert_allclose(V_vi, V_pi, atol=1e-4)
