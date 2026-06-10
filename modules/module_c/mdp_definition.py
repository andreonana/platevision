"""
PlateVision — Module C : Définition du MDP (Markov Decision Process).

Ce module définit le processus de décision markovien pour le système
PlateVision. Le MDP modélise les décisions à prendre lorsqu'une plaque
est détectée : vérification de conformité, alerte aux agents, archivage.

États, actions, transitions et récompenses sont calibrés sur les
procédures réelles du MINT et de la DGI Cameroun.

Rôle dans le pipeline : formalisation du problème de décision séquentielle.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class State:
    """Représente un état du système PlateVision."""
    id: int
    name: str
    description: str
    is_terminal: bool = False


@dataclass
class Action:
    """Représente une action disponible dans un état."""
    id: int
    name: str
    description: str


@dataclass
class MDP:
    """
    Structure du Processus de Décision Markovien PlateVision.

    Attributes:
        states: Liste des états du système
        actions: Liste des actions disponibles
        transitions: P[s][a][s'] = probabilité de transition
        rewards: R[s][a][s'] = récompense immédiate
        gamma: Facteur d'actualisation
    """
    states: List[State]
    actions: List[Action]
    transitions: np.ndarray   # forme (n_states, n_actions, n_states)
    rewards: np.ndarray       # forme (n_states, n_actions, n_states)
    gamma: float = 0.9


# --- Définition des états du système ---
STATES = [
    State(0, "DETECTION_EN_COURS",    "Plaque en cours de détection/lecture"),
    State(1, "PLAQUE_LISE_CLAIRE",    "Plaque lue avec haute confiance (OCR > 90%)"),
    State(2, "PLAQUE_AMBIGUE",        "Plaque lue avec confiance moyenne (60-90%)"),
    State(3, "PLAQUE_ILLISIBLE",      "Plaque illisible ou confiance faible (< 60%)"),
    State(4, "VERIFICATION_MINT",     "Vérification des données MINT (contrôle routier)"),
    State(5, "VERIFICATION_DGI",      "Vérification des données DGI (vignette/patente)"),
    State(6, "VEHICULE_CONFORME",     "Véhicule conforme à toutes les réglementations", is_terminal=True),
    State(7, "ALERTE_INFRACTION",     "Infraction détectée, alerte émise aux agents", is_terminal=True),
    State(8, "ARCHIVAGE",             "Données archivées dans le système", is_terminal=True),
    State(9, "ECHEC_SYSTEME",         "Échec technique, intervention manuelle requise", is_terminal=True),
]

# --- Définition des actions disponibles ---
ACTIONS = [
    Action(0, "LIRE_PLAQUE",          "Tenter la reconnaissance OCR de la plaque"),
    Action(1, "AMELIORER_IMAGE",      "Prétraitement image (débruitage, contraste)"),
    Action(2, "VERIFIER_MINT",        "Interroger la base de données MINT"),
    Action(3, "VERIFIER_DGI",         "Interroger la base de données DGI"),
    Action(4, "EMETTRE_ALERTE",       "Émettre une alerte d'infraction vers les agents"),
    Action(5, "ARCHIVER",             "Archiver les données de passage"),
    Action(6, "ESCALADER_MANUEL",     "Transmettre à un opérateur humain"),
]


def build_transition_matrix(n_states: int, n_actions: int) -> np.ndarray:
    """
    Construit la matrice de transitions P[s][a][s'].

    Les probabilités sont estimées d'après les performances du système
    PlateVision sur le dataset de validation.

    Args:
        n_states: Nombre total d'états
        n_actions: Nombre total d'actions

    Returns:
        Matrice de transitions de forme (n_states, n_actions, n_states)
    """
    P = np.zeros((n_states, n_actions, n_states))

    # Depuis DETECTION_EN_COURS (0)
    # Action LIRE_PLAQUE (0) : 70% → claire, 20% → ambiguë, 10% → illisible
    P[0, 0, 1] = 0.70  # → PLAQUE_LISE_CLAIRE
    P[0, 0, 2] = 0.20  # → PLAQUE_AMBIGUE
    P[0, 0, 3] = 0.10  # → PLAQUE_ILLISIBLE

    # Depuis DETECTION_EN_COURS : Action AMELIORER_IMAGE (1)
    P[0, 1, 0] = 0.30  # reste en DETECTION_EN_COURS (amélioration insuffisante)
    P[0, 1, 1] = 0.55  # → PLAQUE_LISE_CLAIRE
    P[0, 1, 2] = 0.15  # → PLAQUE_AMBIGUE

    # Depuis PLAQUE_LISE_CLAIRE (1)
    P[1, 2, 4] = 0.90  # Action VERIFIER_MINT → VERIFICATION_MINT
    P[1, 2, 9] = 0.10  # → ECHEC_SYSTEME (base indisponible)
    P[1, 3, 5] = 0.90  # Action VERIFIER_DGI → VERIFICATION_DGI
    P[1, 3, 9] = 0.10  # → ECHEC_SYSTEME
    P[1, 5, 8] = 1.00  # Action ARCHIVER → ARCHIVAGE

    # Depuis PLAQUE_AMBIGUE (2)
    P[2, 0, 1] = 0.50  # Nouvelle tentative lecture → CLAIRE
    P[2, 0, 2] = 0.35  # Reste AMBIGUE
    P[2, 0, 3] = 0.15  # → ILLISIBLE
    P[2, 1, 1] = 0.75  # AMELIORER_IMAGE → CLAIRE
    P[2, 1, 2] = 0.25  # Reste AMBIGUE
    P[2, 6, 9] = 1.00  # ESCALADER → ECHEC_SYSTEME (clôture manuelle)

    # Depuis PLAQUE_ILLISIBLE (3)
    P[3, 1, 2] = 0.60  # AMELIORER_IMAGE → AMBIGUE
    P[3, 1, 3] = 0.40  # Reste ILLISIBLE
    P[3, 6, 9] = 1.00  # ESCALADER → intervention manuelle

    # Depuis VERIFICATION_MINT (4)
    P[4, 3, 5] = 0.85  # VERIFIER_DGI → VERIFICATION_DGI
    P[4, 3, 9] = 0.15  # → ECHEC_SYSTEME
    P[4, 4, 7] = 0.30  # EMETTRE_ALERTE si infraction MINT détectée
    P[4, 4, 6] = 0.70  # Pas d'infraction → CONFORME
    P[4, 5, 8] = 1.00  # ARCHIVER → ARCHIVAGE

    # Depuis VERIFICATION_DGI (5)
    P[5, 4, 7] = 0.25  # EMETTRE_ALERTE si défaut fiscal détecté
    P[5, 4, 6] = 0.75  # Conforme → VEHICULE_CONFORME
    P[5, 5, 8] = 1.00  # ARCHIVER → ARCHIVAGE

    # États terminaux : restent en place
    for terminal_state in [6, 7, 8, 9]:
        for action_id in range(n_actions):
            P[terminal_state, action_id, terminal_state] = 1.0

    # Vérification : toutes les lignes non nulles doivent sommer à 1
    for s in range(n_states):
        for a in range(n_actions):
            row_sum = P[s, a, :].sum()
            if row_sum > 0 and not np.isclose(row_sum, 1.0):
                logger.warning(f"Transition P[{s}][{a}] ne somme pas à 1 : {row_sum:.4f}")

    return P


def build_reward_matrix(n_states: int, n_actions: int) -> np.ndarray:
    """
    Construit la matrice de récompenses R[s][a][s'].

    Les récompenses reflètent les objectifs du système :
    - Maximiser les véhicules conformes identifiés rapidement
    - Pénaliser les échecs et les interventions manuelles
    - Récompenser la détection d'infractions (objectif DGI/MINT)

    Args:
        n_states: Nombre total d'états
        n_actions: Nombre total d'actions

    Returns:
        Matrice de récompenses de forme (n_states, n_actions, n_states)
    """
    R = np.zeros((n_states, n_actions, n_states))

    # Récompenses pour atteindre VEHICULE_CONFORME (6)
    R[:, :, 6] = +10.0

    # Récompenses pour atteindre ALERTE_INFRACTION (7) — objectif MINT/DGI
    R[:, :, 7] = +15.0

    # Récompenses pour atteindre ARCHIVAGE (8)
    R[:, :, 8] = +5.0

    # Pénalité pour ECHEC_SYSTEME (9)
    R[:, :, 9] = -10.0

    # Pénalité pour actions coûteuses (traitement image, escalade)
    R[:, 1, :] -= 2.0   # AMELIORER_IMAGE a un coût calculatoire
    R[:, 6, :] -= 5.0   # ESCALADER est coûteux (mobilise un agent humain)

    # Récompense modeste pour une lecture réussie
    R[:, 0, 1] = +2.0   # Lecture claire réussie

    return R


def build_mdp(gamma: float = 0.9) -> MDP:
    """
    Construit et retourne le MDP complet PlateVision.

    Args:
        gamma: Facteur d'actualisation (défaut : 0.9)

    Returns:
        Instance MDP entièrement configurée
    """
    n_states = len(STATES)
    n_actions = len(ACTIONS)

    P = build_transition_matrix(n_states, n_actions)
    R = build_reward_matrix(n_states, n_actions)

    mdp = MDP(
        states=STATES,
        actions=ACTIONS,
        transitions=P,
        rewards=R,
        gamma=gamma,
    )

    logger.info(f"MDP PlateVision construit : {n_states} états, {n_actions} actions, γ={gamma}")
    return mdp
