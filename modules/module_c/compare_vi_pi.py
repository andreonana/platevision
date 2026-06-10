"""
PlateVision — Module C : Comparaison Value Iteration vs Policy Iteration.

Ce module compare les deux algorithmes de résolution du MDP PlateVision :
- Vitesse de convergence (nombre d'itérations, temps CPU)
- Qualité de la politique résultante
- Sensibilité au facteur d'actualisation γ

Rôle dans le pipeline : analyse comparative pour le rapport technique (L3).
"""

import logging
import time
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

from modules.module_c.mdp_definition import MDP, build_mdp
from modules.module_c.value_iteration import value_iteration
from modules.module_c.policy_iteration import policy_iteration

logger = logging.getLogger(__name__)


def compare_algorithms(
    mdp: MDP,
    gamma: float = 0.9,
    output_path: str = "reports/figures/",
) -> Dict[str, Any]:
    """
    Compare VI et PI sur le même MDP et génère des graphiques.

    Args:
        mdp: Instance MDP à résoudre
        gamma: Facteur d'actualisation pour cette comparaison
        output_path: Dossier de sauvegarde des figures

    Returns:
        Dictionnaire des résultats de comparaison
    """
    from pathlib import Path
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"COMPARAISON VI vs PI — γ={gamma}")
    logger.info(f"{'='*60}")

    # --- Value Iteration ---
    start = time.time()
    policy_vi, values_vi = value_iteration(mdp, gamma=gamma)
    time_vi = time.time() - start

    # --- Policy Iteration ---
    start = time.time()
    policy_pi, values_pi = policy_iteration(mdp, gamma=gamma)
    time_pi = time.time() - start

    # Vérification que les politiques sont identiques
    policies_identical = np.array_equal(policy_vi, policy_pi)

    logger.info(f"\nRésultats (γ={gamma}) :")
    logger.info(f"  VI — Temps : {time_vi:.4f}s")
    logger.info(f"  PI — Temps : {time_pi:.4f}s")
    logger.info(f"  Politiques identiques : {policies_identical}")
    logger.info(f"  Différence max valeurs VI-PI : {np.max(np.abs(values_vi - values_pi)):.6f}")

    # Graphique de comparaison des valeurs d'état
    _plot_values_comparison(values_vi, values_pi, mdp, gamma, output_dir)

    results = {
        "gamma": gamma,
        "time_vi": time_vi,
        "time_pi": time_pi,
        "policies_identical": bool(policies_identical),
        "max_value_diff": float(np.max(np.abs(values_vi - values_pi))),
        "values_vi": values_vi,
        "values_pi": values_pi,
    }
    return results


def sensitivity_analysis(
    gamma_values: List[float] = None,
    output_path: str = "reports/figures/",
) -> Dict[float, Dict[str, Any]]:
    """
    Analyse la sensibilité des algorithmes au facteur d'actualisation γ.

    Pour chaque valeur de γ, compare VI et PI en termes de :
    - Nombre d'itérations jusqu'à convergence
    - Valeurs d'état optimales
    - Stabilité de la politique

    Args:
        gamma_values: Liste des valeurs de γ à tester
        output_path: Dossier de sauvegarde

    Returns:
        Dictionnaire des résultats par valeur de γ
    """
    from pathlib import Path
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if gamma_values is None:
        gamma_values = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

    all_results = {}
    times_vi = []
    times_pi = []

    logger.info(f"\nAnalyse de sensibilité — {len(gamma_values)} valeurs de γ")

    for gamma in gamma_values:
        mdp = build_mdp(gamma=gamma)
        results = compare_algorithms(mdp, gamma=gamma, output_path=output_path)
        all_results[gamma] = results
        times_vi.append(results["time_vi"])
        times_pi.append(results["time_pi"])

    # Graphique de sensibilité γ vs temps de convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(gamma_values, times_vi, "b-o", label="Value Iteration", markersize=8)
    ax1.plot(gamma_values, times_pi, "r-s", label="Policy Iteration", markersize=8)
    ax1.set_title("Temps de convergence vs γ", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Facteur d'actualisation γ", fontsize=11)
    ax1.set_ylabel("Temps (secondes)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Valeur de l'état initial pour chaque γ
    v0_vi = [all_results[g]["values_vi"][0] for g in gamma_values]
    v0_pi = [all_results[g]["values_pi"][0] for g in gamma_values]
    ax2.plot(gamma_values, v0_vi, "b-o", label="V*(s0) — VI", markersize=8)
    ax2.plot(gamma_values, v0_pi, "r-s", label="V*(s0) — PI", markersize=8)
    ax2.set_title("Valeur V*(état initial) vs γ", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Facteur d'actualisation γ", fontsize=11)
    ax2.set_ylabel("V*(DETECTION_EN_COURS)", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Analyse de Sensibilité — PlateVision MDP", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = output_dir / "sensitivity_gamma.png"
    plt.savefig(str(fig_path), dpi=150)
    plt.close()
    logger.info(f"Graphique de sensibilité sauvegardé : {fig_path}")

    return all_results


def _plot_values_comparison(
    values_vi: np.ndarray,
    values_pi: np.ndarray,
    mdp: MDP,
    gamma: float,
    output_dir,
) -> None:
    """Génère un graphique en barres comparant les valeurs d'état VI vs PI."""
    state_names = [s.name.replace("_", "\n") for s in mdp.states]
    x = np.arange(len(state_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width / 2, values_vi, width, label="Value Iteration", color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width / 2, values_pi, width, label="Policy Iteration", color="tomato", alpha=0.8)

    ax.set_title(f"Valeurs d'état V*(s) — VI vs PI (γ={gamma})", fontsize=13, fontweight="bold")
    ax.set_xlabel("États du MDP", fontsize=11)
    ax.set_ylabel("Valeur optimale V*(s)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(state_names, fontsize=7, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / f"vi_pi_comparison_gamma_{gamma}.png"
    plt.savefig(str(fig_path), dpi=150)
    plt.close()
    logger.info(f"Comparaison VI/PI sauvegardée : {fig_path}")
