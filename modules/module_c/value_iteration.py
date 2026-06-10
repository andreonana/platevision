"""
PlateVision — Module C : Value Iteration.

Implémentation de l'algorithme Value Iteration (VI) pour résoudre le MDP
PlateVision. VI calcule itérativement la fonction de valeur optimale V*(s)
puis en déduit la politique optimale π*(s).

Convergence garantie pour γ < 1. La vitesse de convergence dépend de γ
et du seuil θ choisi.

Rôle dans le pipeline : résolution exacte du MDP PlateVision (méthode 1).
"""

import logging
import os
import time
from typing import Tuple, List

import numpy as np

from modules.module_c.mdp_definition import MDP

logger = logging.getLogger(__name__)


def value_iteration(
    mdp: MDP,
    gamma: float = None,
    theta: float = None,
    max_iter: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Algorithme Value Iteration pour résoudre le MDP.

    Équation de Bellman optimale :
        V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V_k(s')]

    Args:
        mdp: Instance MDP à résoudre
        gamma: Facteur d'actualisation (remplace mdp.gamma si fourni)
        theta: Seuil de convergence (défaut : variable d'environnement)
        max_iter: Nombre maximum d'itérations

    Returns:
        Tuple (policy, values) :
          - policy[s] = indice de l'action optimale dans l'état s
          - values[s] = valeur optimale V*(s)
    """
    if gamma is None:
        gamma = mdp.gamma
    if theta is None:
        theta = float(os.getenv("MDP_CONVERGENCE_THRESHOLD", 0.001))
    if max_iter is None:
        max_iter = int(os.getenv("MDP_MAX_ITERATIONS", 1000))

    n_states = len(mdp.states)
    n_actions = len(mdp.actions)
    P = mdp.transitions   # P[s, a, s']
    R = mdp.rewards       # R[s, a, s']

    # Initialisation de la fonction de valeur à zéro
    V = np.zeros(n_states)
    convergence_history: List[float] = []

    logger.info(f"Value Iteration démarrée (γ={gamma}, θ={theta}, max_iter={max_iter})")
    start_time = time.time()

    for iteration in range(max_iter):
        delta = 0.0
        V_new = np.zeros(n_states)

        for s in range(n_states):
            if mdp.states[s].is_terminal:
                V_new[s] = 0.0
                continue

            # Q(s, a) = Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * V(s')]
            Q = np.zeros(n_actions)
            for a in range(n_actions):
                Q[a] = np.sum(P[s, a, :] * (R[s, a, :] + gamma * V))

            V_new[s] = np.max(Q)
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new.copy()
        convergence_history.append(delta)

        if iteration % 50 == 0:
            logger.debug(f"VI itération {iteration:4d} | Δ = {delta:.6f}")

        # Critère d'arrêt
        if delta < theta:
            elapsed = time.time() - start_time
            logger.info(
                f"Value Iteration convergée en {iteration + 1} itérations "
                f"(Δ={delta:.2e}, durée={elapsed:.3f}s)"
            )
            break
    else:
        logger.warning(f"Value Iteration non convergée après {max_iter} itérations (Δ={delta:.2e})")

    # Extraction de la politique optimale greedy
    policy = extract_policy(mdp, V, gamma)

    # Affichage de la politique
    log_policy(mdp, policy, V)

    return policy, V


def extract_policy(mdp: MDP, V: np.ndarray, gamma: float) -> np.ndarray:
    """
    Extrait la politique optimale greedy à partir de la fonction de valeur.

    π*(s) = argmax_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V(s')]

    Args:
        mdp: Instance MDP
        V: Fonction de valeur V*(s)
        gamma: Facteur d'actualisation

    Returns:
        Vecteur policy[s] = indice de l'action optimale
    """
    n_states = len(mdp.states)
    n_actions = len(mdp.actions)
    P = mdp.transitions
    R = mdp.rewards
    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        if mdp.states[s].is_terminal:
            policy[s] = 0
            continue

        Q = np.array([
            np.sum(P[s, a, :] * (R[s, a, :] + gamma * V))
            for a in range(n_actions)
        ])
        policy[s] = int(np.argmax(Q))

    return policy


def log_policy(mdp: MDP, policy: np.ndarray, V: np.ndarray) -> None:
    """Affiche la politique optimale et les valeurs d'état de manière lisible."""
    logger.info("\n" + "=" * 60)
    logger.info("POLITIQUE OPTIMALE — Value Iteration")
    logger.info("=" * 60)
    logger.info(f"{'État':<30} {'Action optimale':<25} {'V*(s)':>8}")
    logger.info("-" * 65)
    for s, state in enumerate(mdp.states):
        action_name = mdp.actions[policy[s]].name if not state.is_terminal else "(terminal)"
        logger.info(f"{state.name:<30} {action_name:<25} {V[s]:>8.4f}")
    logger.info("=" * 60)
