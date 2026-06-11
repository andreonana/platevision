"""
Module C — Algorithme Policy Iteration pour la résolution du MDP PlateVision.

Rôle dans le pipeline PlateVision :
    Résout le même MDP que Value Iteration, mais par une approche alternée :
    évaluation de la politique courante (Policy Evaluation) puis amélioration
    (Policy Improvement), jusqu'à convergence vers la politique optimale π*.
    Utilisé en comparaison avec VI dans compare_vi_pi.py.

Responsable : Étudiant 4
"""

import logging
import os
from pathlib import Path

import numpy as np

from modules.module_c.mdp_definition import get_mdp, STATES, ACTIONS

logger = logging.getLogger(__name__)

MAX_ITERATIONS = int(os.getenv("MDP_MAX_ITERATIONS", 1000))
CONVERGENCE_THRESHOLD = float(os.getenv("MDP_CONVERGENCE_THRESHOLD", 1e-6))


def policy_evaluation(
    policy: np.ndarray,
    T: np.ndarray,
    R: np.ndarray,
    gamma: float,
    theta: float = CONVERGENCE_THRESHOLD,
) -> np.ndarray:
    """
    Évalue la fonction de valeur V^π pour une politique fixée.

    Résout itérativement l'équation de Bellman d'évaluation :
    V^π(s) = R(s, π(s)) + γ * Σ T(s, π(s), s') * V^π(s')

    Args:
        policy: Politique fixée (N_STATES,) — indices des actions
        T: Matrice de transition
        R: Matrice de récompenses
        gamma: Facteur d'actualisation
        theta: Seuil de convergence

    Returns:
        Fonction de valeur V^π de forme (N_STATES,)
    """
    n_states = len(policy)
    V = np.zeros(n_states)

    for _ in range(MAX_ITERATIONS):
        V_new = np.zeros(n_states)
        for s in range(n_states):
            a = policy[s]
            V_new[s] = R[s, a] + gamma * np.dot(T[s, a, :], V)

        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < theta:
            break

    return V


def policy_improvement(
    V: np.ndarray,
    T: np.ndarray,
    R: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, bool]:
    """
    Améliore la politique en sélectionnant greedy-ment la meilleure action.

    Args:
        V: Fonction de valeur courante
        T, R, gamma: Composantes du MDP

    Returns:
        Tuple (nouvelle_politique, is_stable)
        - is_stable : True si la politique n'a pas changé (convergence)
    """
    n_states, n_actions = R.shape
    new_policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        q_values = np.array([
            R[s, a] + gamma * np.dot(T[s, a, :], V) for a in range(n_actions)
        ])
        new_policy[s] = np.argmax(q_values)

    return new_policy, True


def policy_iteration(
    T: np.ndarray,
    R: np.ndarray,
    gamma: float = 0.9,
    max_iter: int = MAX_ITERATIONS,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Exécute l'algorithme Policy Iteration complet.

    Args:
        T: Matrice de transition
        R: Matrice de récompenses
        gamma: Facteur d'actualisation
        max_iter: Nombre maximum d'itérations d'amélioration

    Returns:
        Tuple (V_star, policy_star, n_iterations)
    """
    n_states = R.shape[0]

    # Politique initiale aléatoire (action 0 pour tous les états)
    policy = np.zeros(n_states, dtype=int)
    n_iterations = 0

    logger.info("Policy Iteration démarré : γ=%.2f", gamma)

    for iteration in range(max_iter):
        # Étape 1 : Évaluation de la politique courante
        V = policy_evaluation(policy, T, R, gamma)

        # Étape 2 : Amélioration de la politique
        new_policy, _ = policy_improvement(V, T, R, gamma)
        n_iterations = iteration + 1

        if np.array_equal(new_policy, policy):
            logger.info("Policy Iteration convergée à l'itération %d", n_iterations)
            break

        policy = new_policy
    else:
        logger.warning("Policy Iteration n'a pas convergé après %d itérations", max_iter)

    # Évaluation finale
    V_star = policy_evaluation(policy, T, R, gamma)
    return V_star, policy, n_iterations


def display_policy(V: np.ndarray, policy: np.ndarray) -> None:
    """Affiche la politique optimale obtenue par Policy Iteration."""
    logger.info("\n=== Politique Optimale (Policy Iteration) ===")
    logger.info("%-5s %-25s %-25s %-10s", "ID", "État", "Action optimale", "V*(s)")
    logger.info("-" * 70)
    for s, state in enumerate(STATES):
        action = ACTIONS[policy[s]]
        logger.info("%-5d %-25s %-25s %-10.4f", s, state.name, action.name, V[s])


def run_policy_iteration(gamma: float = 0.9) -> dict:
    """
    Exécute le pipeline complet Policy Iteration sur le MDP PlateVision.

    Returns:
        Dictionnaire avec V_star, policy, iterations_count
    """
    mdp = get_mdp()
    T, R = mdp["T"], mdp["R"]

    V, policy, n_iter = policy_iteration(T, R, gamma=gamma)
    display_policy(V, policy)

    results = {
        "algorithm": "Policy Iteration",
        "gamma": gamma,
        "V_star": V.tolist(),
        "policy": policy.tolist(),
        "iterations_count": n_iter,
    }

    import json
    output_dir = Path("reports/rapport_technique")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"module_c_pi_gamma{gamma:.2f}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Résultats PI sauvegardés : %s", results_path)
    return results
