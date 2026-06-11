"""
Module C — Définition formelle du MDP PlateVision.

Rôle dans le pipeline PlateVision :
    Modélise le problème de contrôle de véhicules comme un Processus de Décision
    Markovien (MDP) : états = situations de contrôle, actions = décisions de l'agent
    (laisser passer, contrôler, verbaliser, orienter), transitions = probabilités
    de changement d'état, récompenses = utilité de chaque décision pour la DGI/MINT.

Responsable : Étudiant 4
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class State:
    """État du MDP : situation d'un véhicule au point de contrôle."""
    id: int
    name: str
    description: str


@dataclass(frozen=True)
class Action:
    """Action disponible pour l'agent au point de contrôle."""
    id: int
    name: str
    description: str


# --- Définition des états ---
STATES = [
    State(0, "PLAQUE_VALIDE",    "Plaque reconnue, confiance haute, véhicule conforme"),
    State(1, "PLAQUE_FLOUE",     "Plaque détectée mais OCR incertain (confiance < seuil)"),
    State(2, "PLAQUE_ABSENTE",   "Aucune plaque détectée sur l'image"),
    State(3, "PLAQUE_ETRANGERE", "Plaque de format non camerounais"),
    State(4, "PLAQUE_CD",        "Corps Diplomatique — procédure d'exemption"),
    State(5, "PLAQUE_OFFICIEL",  "Véhicule gouvernemental — procédure spéciale"),
    State(6, "TRANSIT",          "Immatriculation de transit temporaire"),
    State(7, "VERBALISE",        "Infractions constatées — procès-verbal en cours"),
]

# --- Définition des actions ---
ACTIONS = [
    Action(0, "LAISSER_PASSER",  "Lever la barrière — le véhicule est conforme"),
    Action(1, "CONTROLE_RAPIDE", "Vérification rapide de carte grise et vignette"),
    Action(2, "CONTROLE_APPROFONDI", "Contrôle complet : identité, assurance, amendes DGI"),
    Action(3, "VERBALISER",      "Émettre un procès-verbal d'infraction"),
    Action(4, "ORIENTER_DOUANES","Orienter vers service des douanes"),
]

N_STATES = len(STATES)
N_ACTIONS = len(ACTIONS)


def build_transition_matrix() -> np.ndarray:
    """
    Construit la matrice de transition T[s, a, s'] = P(s' | s, a).

    La matrice est de forme (N_STATES, N_ACTIONS, N_STATES).
    Chaque ligne T[s, a, :] doit sommer à 1.

    Returns:
        Tableau numpy de forme (N_STATES, N_ACTIONS, N_STATES)
    """
    T = np.zeros((N_STATES, N_ACTIONS, N_STATES))

    # État 0 : PLAQUE_VALIDE
    T[0, 0, 0] = 1.0   # LAISSER_PASSER → reste conforme
    T[0, 1, 0] = 0.9   # CONTROLE_RAPIDE → reste conforme avec haute proba
    T[0, 1, 7] = 0.1   # faible chance de découvrir une infraction cachée
    T[0, 2, 0] = 0.8
    T[0, 2, 7] = 0.2
    T[0, 3, 7] = 1.0   # VERBALISER sur plaque valide → état verbalisé
    T[0, 4, 0] = 1.0

    # État 1 : PLAQUE_FLOUE
    T[1, 0, 1] = 1.0
    T[1, 1, 0] = 0.5   # le contrôle rapide peut clarifier
    T[1, 1, 1] = 0.3
    T[1, 1, 7] = 0.2
    T[1, 2, 0] = 0.4
    T[1, 2, 7] = 0.4
    T[1, 2, 2] = 0.2
    T[1, 3, 7] = 1.0
    T[1, 4, 1] = 1.0

    # État 2 : PLAQUE_ABSENTE
    T[2, 0, 2] = 1.0
    T[2, 1, 2] = 0.6
    T[2, 1, 7] = 0.4
    T[2, 2, 7] = 0.7
    T[2, 2, 2] = 0.3
    T[2, 3, 7] = 1.0
    T[2, 4, 2] = 1.0

    # État 3 : PLAQUE_ETRANGERE
    T[3, 0, 3] = 1.0
    T[3, 1, 3] = 0.5
    T[3, 1, 7] = 0.5
    T[3, 2, 7] = 0.4
    T[3, 2, 3] = 0.2
    T[3, 2, 6] = 0.4
    T[3, 3, 7] = 1.0
    T[3, 4, 3] = 1.0

    # État 4 : PLAQUE_CD (Diplomatique)
    T[4, 0, 0] = 1.0   # Laisser passer → conforme
    T[4, 1, 4] = 0.8
    T[4, 1, 0] = 0.2
    T[4, 2, 4] = 0.5
    T[4, 2, 0] = 0.5
    T[4, 3, 4] = 1.0   # Verbaliser un CD reste dans état CD (procédure spéciale)
    T[4, 4, 4] = 1.0

    # État 5 : PLAQUE_OFFICIEL
    T[5, 0, 0] = 1.0
    T[5, 1, 5] = 0.7
    T[5, 1, 0] = 0.3
    T[5, 2, 5] = 0.5
    T[5, 2, 0] = 0.5
    T[5, 3, 7] = 1.0
    T[5, 4, 5] = 1.0

    # État 6 : TRANSIT
    T[6, 0, 6] = 1.0
    T[6, 1, 6] = 0.4
    T[6, 1, 0] = 0.3
    T[6, 1, 7] = 0.3
    T[6, 2, 7] = 0.6
    T[6, 2, 0] = 0.4
    T[6, 3, 7] = 1.0
    T[6, 4, 6] = 1.0

    # État 7 : VERBALISE (état terminal)
    T[7, 0, 7] = 1.0
    T[7, 1, 7] = 1.0
    T[7, 2, 7] = 1.0
    T[7, 3, 7] = 1.0
    T[7, 4, 7] = 1.0

    # Vérification : chaque distribution de transition doit sommer à 1
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            total = T[s, a, :].sum()
            assert abs(total - 1.0) < 1e-6, f"T[{s},{a},:] somme à {total} ≠ 1"

    logger.info("Matrice de transition construite : shape %s", T.shape)
    return T


def build_reward_matrix() -> np.ndarray:
    """
    Construit la matrice de récompenses R[s, a].

    Les récompenses modélisent l'utilité pour la DGI/MINT :
    - Récompenses positives : actions correctes, amendes perçues, fluidité du trafic
    - Récompenses négatives : verbalisations abusives, laisser-passer illicites

    Returns:
        Tableau numpy de forme (N_STATES, N_ACTIONS)
    """
    R = np.full((N_STATES, N_ACTIONS), -1.0)  # coût de base par action

    # PLAQUE_VALIDE : laisser passer est optimal
    R[0, 0] = +10.0   # LAISSER_PASSER → fluidité + confiance citoyen
    R[0, 1] = +2.0    # CONTROLE_RAPIDE acceptable
    R[0, 2] = -3.0    # CONTROLE_APPROFONDI inutile → pénalité
    R[0, 3] = -15.0   # VERBALISER sans raison → très pénalisé (violation droits)
    R[0, 4] = -5.0

    # PLAQUE_FLOUE : contrôle requis
    R[1, 0] = -8.0    # LAISSER_PASSER risqué
    R[1, 1] = +5.0    # CONTROLE_RAPIDE approprié
    R[1, 2] = +8.0    # CONTROLE_APPROFONDI recommandé
    R[1, 3] = -5.0
    R[1, 4] = 0.0

    # PLAQUE_ABSENTE : contrôle obligatoire
    R[2, 0] = -20.0   # LAISSER_PASSER très pénalisé
    R[2, 1] = +3.0
    R[2, 2] = +15.0   # CONTROLE_APPROFONDI optimal
    R[2, 3] = +12.0   # VERBALISER justifié
    R[2, 4] = -2.0

    # PLAQUE_ETRANGERE
    R[3, 0] = -5.0
    R[3, 1] = +3.0
    R[3, 2] = +10.0
    R[3, 3] = +5.0
    R[3, 4] = +8.0    # ORIENTER_DOUANES le plus adapté

    # PLAQUE_CD (Diplomatique) : protocole strict
    R[4, 0] = +8.0    # LAISSER_PASSER : procédure diplomatique normale
    R[4, 1] = +2.0
    R[4, 2] = -10.0   # CONTROLE_APPROFONDI d'un CD : incident diplomatique
    R[4, 3] = -20.0   # VERBALISER un CD : incident diplomatique grave
    R[4, 4] = -5.0

    # PLAQUE_OFFICIEL
    R[5, 0] = +7.0
    R[5, 1] = +3.0
    R[5, 2] = -5.0
    R[5, 3] = -15.0
    R[5, 4] = -3.0

    # TRANSIT
    R[6, 0] = -10.0
    R[6, 1] = +5.0
    R[6, 2] = +12.0
    R[6, 3] = +8.0
    R[6, 4] = +10.0   # ORIENTER_DOUANES optimal pour transit

    # VERBALISE : état terminal
    R[7, 0] = 0.0
    R[7, 1] = 0.0
    R[7, 2] = 0.0
    R[7, 3] = 0.0
    R[7, 4] = 0.0

    logger.info("Matrice de récompenses construite : shape %s", R.shape)
    return R


def get_mdp() -> dict:
    """
    Retourne le MDP complet sous forme de dictionnaire.

    Returns:
        Dict avec clés : states, actions, T (transitions), R (récompenses)
    """
    T = build_transition_matrix()
    R = build_reward_matrix()

    return {
        "states": STATES,
        "actions": ACTIONS,
        "n_states": N_STATES,
        "n_actions": N_ACTIONS,
        "T": T,
        "R": R,
    }
