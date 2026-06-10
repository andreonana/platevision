"""
PlateVision — Tests unitaires : MDP Value Iteration & Policy Iteration (Module C).
"""

import numpy as np
import pytest


class TestMDPDefinition:
    """Tests pour la définition du MDP PlateVision."""

    def test_build_mdp_returns_mdp(self):
        from modules.module_c.mdp_definition import build_mdp
        mdp = build_mdp()
        assert mdp is not None
        assert len(mdp.states) > 0
        assert len(mdp.actions) > 0

    def test_transition_matrix_shape(self):
        from modules.module_c.mdp_definition import build_mdp
        mdp = build_mdp()
        n_s = len(mdp.states)
        n_a = len(mdp.actions)
        assert mdp.transitions.shape == (n_s, n_a, n_s)

    def test_reward_matrix_shape(self):
        from modules.module_c.mdp_definition import build_mdp
        mdp = build_mdp()
        n_s = len(mdp.states)
        n_a = len(mdp.actions)
        assert mdp.rewards.shape == (n_s, n_a, n_s)

    def test_transition_probabilities_valid(self):
        """Toute distribution de transition non nulle doit sommer à 1."""
        from modules.module_c.mdp_definition import build_mdp
        mdp = build_mdp()
        P = mdp.transitions
        n_s, n_a, _ = P.shape

        for s in range(n_s):
            for a in range(n_a):
                row_sum = P[s, a, :].sum()
                if row_sum > 0:
                    assert pytest.approx(row_sum, abs=1e-6) == 1.0, \
                        f"P[{s}][{a}] ne somme pas à 1 : {row_sum}"

    def test_gamma_default_value(self):
        from modules.module_c.mdp_definition import build_mdp
        mdp = build_mdp()
        assert 0 < mdp.gamma <= 1.0


class TestValueIteration:
    """Tests pour l'algorithme Value Iteration."""

    def test_vi_returns_correct_shapes(self):
        from modules.module_c.mdp_definition import build_mdp
        from modules.module_c.value_iteration import value_iteration

        mdp = build_mdp()
        policy, values = value_iteration(mdp, gamma=0.9)

        assert len(policy) == len(mdp.states)
        assert len(values) == len(mdp.states)

    def test_vi_policy_actions_are_valid(self):
        from modules.module_c.mdp_definition import build_mdp
        from modules.module_c.value_iteration import value_iteration

        mdp = build_mdp()
        policy, _ = value_iteration(mdp, gamma=0.9)
        n_actions = len(mdp.actions)

        for s in range(len(mdp.states)):
            assert 0 <= policy[s] < n_actions

    def test_vi_terminal_states_value_zero(self):
        from modules.module_c.mdp_definition import build_mdp
        from modules.module_c.value_iteration import value_iteration

        mdp = build_mdp()
        _, values = value_iteration(mdp, gamma=0.9)

        for s, state in enumerate(mdp.states):
            if state.is_terminal:
                assert values[s] == pytest.approx(0.0, abs=1e-4), \
                    f"État terminal {state.name} devrait avoir V=0, got {values[s]}"

    def test_vi_convergence_for_different_gammas(self):
        from modules.module_c.mdp_definition import build_mdp
        from modules.module_c.value_iteration import value_iteration

        for gamma in [0.5, 0.9, 0.99]:
            mdp = build_mdp(gamma=gamma)
            policy, values = value_iteration(mdp, gamma=gamma, theta=0.001)
            assert values is not None, f"VI n'a pas convergé pour gamma={gamma}"


class TestPolicyIteration:
    """Tests pour l'algorithme Policy Iteration."""

    def test_pi_returns_correct_shapes(self):
        from modules.module_c.mdp_definition import build_mdp
        from modules.module_c.policy_iteration import policy_iteration

        mdp = build_mdp()
        policy, values = policy_iteration(mdp, gamma=0.9)

        assert len(policy) == len(mdp.states)
        assert len(values) == len(mdp.states)

    def test_vi_and_pi_produce_same_policy(self):
        """VI et PI doivent converger vers la même politique optimale."""
        from modules.module_c.mdp_definition import build_mdp
        from modules.module_c.value_iteration import value_iteration
        from modules.module_c.policy_iteration import policy_iteration

        mdp = build_mdp(gamma=0.9)
        policy_vi, _ = value_iteration(mdp, gamma=0.9)
        policy_pi, _ = policy_iteration(mdp, gamma=0.9)

        assert np.array_equal(policy_vi, policy_pi), \
            "VI et PI produisent des politiques différentes !"

    def test_vi_and_pi_values_close(self):
        """Les valeurs d'état de VI et PI doivent être très proches."""
        from modules.module_c.mdp_definition import build_mdp
        from modules.module_c.value_iteration import value_iteration
        from modules.module_c.policy_iteration import policy_iteration

        mdp = build_mdp(gamma=0.9)
        _, values_vi = value_iteration(mdp, gamma=0.9)
        _, values_pi = policy_iteration(mdp, gamma=0.9)

        max_diff = np.max(np.abs(values_vi - values_pi))
        assert max_diff < 0.01, f"Différence VI/PI trop grande : {max_diff}"
