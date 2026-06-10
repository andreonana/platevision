"""
PlateVision — Module C : Policy Iteration.

Implémentation de l'algorithme Policy Iteration (PI) pour résoudre le MDP
PlateVision. PI alterne entre l'évaluation d'une politique (résolution d'un
système linéaire) et l'amélioration de la politique jusqu'à stabilisation.

PI converge généralement en moins d'itérations que VI mais chaque itération
est plus coûteuse (résolution de système linéaire).

Rôle dans le pipeline : résolution exacte du MDP PlateVision (méthode 2).
"""

import logging
import os
import time
from typing import Tuple

import numpy as np

from modules.module_c.mdp_definition import MDP

logger = logging.getLogger(__name__)


def policy_evaluation(
    mdp: MDP,
    policy: np.ndarray,
    gamma: float,
    theta: float = 1e-6,
    max_iter: int = 500,
) -> np.ndarray:
    """
    Évalue la fonction de valeur V^π pour une politique donnée.

    Résout itérativement :
        V^π(s) = Σ_{s'} P(s'|s,π(s)) [R(s,π(s),s') + γ V^π(s')]

    Args:
        mdp: Instance MDP
        policy: Politique à évaluer (policy[s] = indice de l'action)
        gamma: Facteur d'actualisation
        theta: Seuil de convergence pour l'évaluation
        max_iter: Nombre maximum d'itérations d'évaluation

    Returns:
        Vecteur V^π(s) pour tous les états
    """
    n_states = len(mdp.states)
    P = mdp.transitions
    R = mdp.rewards
    V = np.zeros(n_states)

    for _ in range(max_iter):
        delta = 0.0
        V_new = np.zeros(n_states)

        for s in range(n_states):
            if mdp.states[s].is_terminal:
                V_new[s] = 0.0
                continue

            a = policy[s]
            V_new[s] = np.sum(P[s, a, :] * (R[s, a, :] + gamma * V))
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new.copy()
        if delta < theta:
            break

    return V


def policy_improvement(
    mdp: MDP,
    V: np.ndarray,
    gamma: float,
) -> Tuple[np.ndarray, bool]:
    """
    Améliore la politique en sélectionnant les actions greedy vis-à-vis de V.

    Args:
        mdp: Instance MDP
        V: Fonction de valeur courante
        gamma: Facteur d'actualisation

    Returns:
        Tuple (new_policy, is_stable) :
          - new_policy : politique améliorée
          - is_stable : True si la politique n'a pas changé (convergence)
    """
    n_states = len(mdp.states)
    n_actions = len(mdp.actions)
    P = mdp.transitions
    R = mdp.rewards
    new_policy = np.zeros(n_states, dtype=int)
    is_stable = True

    for s in range(n_states):
        if mdp.states[s].is_terminal:
            new_policy[s] = 0
            continue

        Q = np.array([
            np.sum(P[s, a, :] * (R[s, a, :] + gamma * V))
            for a in range(n_actions)
        ])
        best_action = int(np.argmax(Q))
        new_policy[s] = best_action

        # Vérifie si la politique a changé pour cet état
        # (la politique initiale est passée via la fonction de valeur courante)

    return new_policy, is_stable


def policy_iteration(
    mdp: MDP,
    gamma: float = None,
    max_iter: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Algorithme Policy Iteration complet.

    Alterne entre :
    1. Évaluation de la politique courante → V^π
    2. Amélioration greedy de la politique → π' meilleur que π

    Args:
        mdp: Instance MDP à résoudre
        gamma: Facteur d'actualisation (remplace mdp.gamma si fourni)
        max_iter: Nombre maximum de cycles amélioration/évaluation

    Returns:
        Tuple (policy, values) : politique optimale et fonction de valeur
    """
    if gamma is None:
        gamma = mdp.gamma
    if max_iter is None:
        max_iter = int(os.getenv("MDP_MAX_ITERATIONS", 1000))

    n_states = len(mdp.states)
    n_actions = len(mdp.actions)

    # Initialisation : politique aléatoire (action 0 pour tous les états)
    policy = np.zeros(n_states, dtype=int)

    logger.info(f"Policy Iteration démarrée (γ={gamma}, max_iter={max_iter})")
    start_time = time.time()

    for iteration in range(max_iter):
        # Étape 1 : Évaluation de la politique courante
        V = policy_evaluation(mdp, policy, gamma)

        # Étape 2 : Amélioration de la politique
        new_policy = np.zeros(n_states, dtype=int)
        P = mdp.transitions
        R = mdp.rewards

        for s in range(n_states):
            if mdp.states[s].is_terminal:
                new_policy[s] = 0
                continue
            Q = np.array([
                np.sum(P[s, a, :] * (R[s, a, :] + gamma * V))
                for a in range(n_actions)
            ])
            new_policy[s] = int(np.argmax(Q))

        # Vérification de la stabilité
        if np.array_equal(new_policy, policy):
            elapsed = time.time() - start_time
            logger.info(
                f"Policy Iteration convergée en {iteration + 1} cycles "
                f"(durée={elapsed:.3f}s)"
            )
            break

        policy = new_policy.copy()
        logger.debug(f"PI cycle {iteration + 1} : politique mise à jour")
    else:
        logger.warning(f"Policy Iteration non convergée après {max_iter} cycles")

    # Évaluation finale de la politique optimale
    V_final = policy_evaluation(mdp, policy, gamma)

    # Affichage de la politique
    from modules.module_c.value_iteration import log_policy
    log_policy(mdp, policy, V_final)

    return policy, V_final
