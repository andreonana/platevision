"""
Module C — Algorithme Value Iteration pour la résolution du MDP PlateVision.

Rôle dans le pipeline PlateVision :
    Résout le MDP défini dans mdp_definition.py en calculant la fonction de valeur
    optimale V*(s) par itérations successives de l'équation de Bellman, puis en
    dérivant la politique optimale π*(s) = argmax_a Q(s,a).

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


def value_iteration(
    T: np.ndarray,
    R: np.ndarray,
    gamma: float = 0.9,
    max_iter: int = MAX_ITERATIONS,
    theta: float = CONVERGENCE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Exécute l'algorithme Value Iteration.

    Args:
        T: Matrice de transition (N_STATES, N_ACTIONS, N_STATES)
        R: Matrice de récompenses (N_STATES, N_ACTIONS)
        gamma: Facteur d'actualisation ]0, 1[
        max_iter: Nombre maximum d'itérations
        theta: Seuil de convergence (arrêt si max|V_new - V_old| < theta)

    Returns:
        Tuple (V_star, policy, convergence_history)
        - V_star : fonction de valeur optimale (N_STATES,)
        - policy : politique optimale (N_STATES,) — index des actions
        - convergence_history : évolution du delta max à chaque itération
    """
    n_states, n_actions = R.shape
    V = np.zeros(n_states)
    convergence_history = []

    logger.info("Value Iteration démarré : γ=%.2f, θ=%e, max_iter=%d", gamma, theta, max_iter)

    for iteration in range(max_iter):
        V_new = np.zeros(n_states)

        for s in range(n_states):
            q_values = np.zeros(n_actions)
            for a in range(n_actions):
                # Équation de Bellman : Q(s,a) = R(s,a) + γ * Σ T(s,a,s') * V(s')
                q_values[a] = R[s, a] + gamma * np.dot(T[s, a, :], V)
            V_new[s] = np.max(q_values)

        delta = np.max(np.abs(V_new - V))
        convergence_history.append(float(delta))
        V = V_new

        if delta < theta:
            logger.info("Value Iteration convergée à l'itération %d (delta=%.2e)", iteration + 1, delta)
            break
    else:
        logger.warning("Value Iteration n'a pas convergé après %d itérations", max_iter)

    # Extraction de la politique optimale
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        q_values = np.array([
            R[s, a] + gamma * np.dot(T[s, a, :], V) for a in range(n_actions)
        ])
        policy[s] = np.argmax(q_values)

    return V, policy, convergence_history


def display_policy(V: np.ndarray, policy: np.ndarray) -> None:
    """Affiche la politique optimale de manière lisible."""
    logger.info("\n=== Politique Optimale (Value Iteration) ===")
    logger.info("%-5s %-25s %-25s %-10s", "ID", "État", "Action optimale", "V*(s)")
    logger.info("-" * 70)
    for s, state in enumerate(STATES):
        action = ACTIONS[policy[s]]
        logger.info("%-5d %-25s %-25s %-10.4f", s, state.name, action.name, V[s])


def run_value_iteration(gamma: float = 0.9) -> dict:
    """
    Exécute le pipeline complet Value Iteration sur le MDP PlateVision.

    Args:
        gamma: Facteur d'actualisation

    Returns:
        Dictionnaire avec V_star, policy, convergence_history, iterations_count
    """
    mdp = get_mdp()
    T, R = mdp["T"], mdp["R"]

    V, policy, history = value_iteration(T, R, gamma=gamma)
    display_policy(V, policy)

    results = {
        "algorithm": "Value Iteration",
        "gamma": gamma,
        "V_star": V.tolist(),
        "policy": policy.tolist(),
        "convergence_history": history,
        "iterations_count": len(history),
        "converged": len(history) < MAX_ITERATIONS,
    }

    # Sauvegarde des résultats
    import json
    output_dir = Path("reports/rapport_technique")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"module_c_vi_gamma{gamma:.2f}.json"
    serializable = {k: v for k, v in results.items()}
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    logger.info("Résultats VI sauvegardés : %s", results_path)
    return results
