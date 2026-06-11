"""
Module C — Comparaison Value Iteration vs Policy Iteration.

Rôle dans le pipeline PlateVision :
    Compare les deux algorithmes de résolution du MDP sur les critères :
    vitesse de convergence, nombre d'itérations, sensibilité au facteur γ,
    et politique obtenue. Génère des graphiques pour le rapport technique.

Responsable : Étudiant 4
"""

import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from modules.module_c.mdp_definition import get_mdp
from modules.module_c.value_iteration import value_iteration
from modules.module_c.policy_iteration import policy_iteration

logger = logging.getLogger(__name__)


def compare_algorithms(gamma: float = 0.9) -> dict:
    """
    Compare VI et PI sur un même MDP avec le même gamma.

    Returns:
        Dictionnaire de comparaison des deux algorithmes.
    """
    mdp = get_mdp()
    T, R = mdp["T"], mdp["R"]

    logger.info("=== Comparaison VI vs PI (γ=%.2f) ===", gamma)

    # Exécution Value Iteration
    t_start = time.perf_counter()
    V_vi, policy_vi, history_vi = value_iteration(T, R, gamma=gamma)
    t_vi = time.perf_counter() - t_start

    # Exécution Policy Iteration
    t_start = time.perf_counter()
    V_pi, policy_pi, n_iter_pi = policy_iteration(T, R, gamma=gamma)
    t_pi = time.perf_counter() - t_start

    policies_agree = np.array_equal(policy_vi, policy_pi)
    max_v_diff = float(np.max(np.abs(V_vi - V_pi)))

    comparison = {
        "gamma": gamma,
        "vi": {
            "iterations": len(history_vi),
            "time_seconds": t_vi,
            "V_star": V_vi.tolist(),
            "policy": policy_vi.tolist(),
            "convergence_history": history_vi,
        },
        "pi": {
            "iterations": n_iter_pi,
            "time_seconds": t_pi,
            "V_star": V_pi.tolist(),
            "policy": policy_pi.tolist(),
        },
        "policies_identical": bool(policies_agree),
        "max_V_difference": max_v_diff,
    }

    logger.info("VI : %d itérations, %.4fs", len(history_vi), t_vi)
    logger.info("PI : %d itérations, %.4fs", n_iter_pi, t_pi)
    logger.info("Politiques identiques : %s", policies_agree)
    logger.info("Différence max |V_VI - V_PI| : %.6f", max_v_diff)

    return comparison


def sensitivity_analysis(
    gamma_values: list[float] | None = None,
) -> dict:
    """
    Analyse la sensibilité des deux algorithmes en fonction de γ.

    Args:
        gamma_values: Liste de valeurs γ à tester [défaut: 0.1 à 0.99]

    Returns:
        Dictionnaire des résultats par valeur de γ.
    """
    if gamma_values is None:
        gamma_values = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

    sensitivity_results = {}

    for gamma in gamma_values:
        logger.info("Analyse sensibilité γ=%.2f", gamma)
        comparison = compare_algorithms(gamma)
        sensitivity_results[gamma] = comparison

    return sensitivity_results


def plot_convergence_comparison(
    history_vi: list[float],
    output_path: Path,
    gamma: float,
) -> None:
    """Génère le graphique de convergence de Value Iteration."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.semilogy(range(1, len(history_vi) + 1), history_vi, "b-o", markersize=4,
                label=f"Value Iteration (γ={gamma:.2f})")
    ax.axhline(y=1e-6, color="red", linestyle="--", alpha=0.7, label="Seuil θ=1e-6")

    ax.set_xlabel("Itération")
    ax.set_ylabel("Δ max (échelle logarithmique)")
    ax.set_title(f"Convergence Value Iteration — MDP PlateVision (γ={gamma:.2f})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / f"convergence_vi_gamma{gamma:.2f}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info("Graphique convergence sauvegardé : %s", fig_path)


def plot_sensitivity_gamma(
    sensitivity_results: dict,
    output_path: Path,
) -> None:
    """Génère le graphique de sensibilité au facteur γ."""
    gammas = sorted(sensitivity_results.keys())
    vi_iters = [sensitivity_results[g]["vi"]["iterations"] for g in gammas]
    pi_iters = [sensitivity_results[g]["pi"]["iterations"] for g in gammas]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(gammas, vi_iters, "b-o", label="Value Iteration")
    ax.plot(gammas, pi_iters, "r-s", label="Policy Iteration")
    ax.set_xlabel("Facteur d'actualisation γ")
    ax.set_ylabel("Nombre d'itérations jusqu'à convergence")
    ax.set_title("Sensibilité au facteur γ — VI vs PI (MDP PlateVision)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / "sensitivity_gamma.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info("Graphique sensibilité γ sauvegardé : %s", fig_path)


def run_comparison(gamma: float = 0.9) -> dict:
    """
    Exécute la comparaison complète VI vs PI et génère les graphiques.

    Returns:
        Dictionnaire de comparaison avec résultats et chemins des figures.
    """
    output_path = Path("reports/rapport_technique/module_c_figures")

    comparison = compare_algorithms(gamma)
    plot_convergence_comparison(
        comparison["vi"]["convergence_history"],
        output_path,
        gamma,
    )

    sensitivity = sensitivity_analysis()
    plot_sensitivity_gamma(sensitivity, output_path)

    import json
    results_path = output_path.parent / f"module_c_comparison_gamma{gamma:.2f}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    logger.info("Comparaison VI vs PI terminée. Figures dans : %s", output_path)
    return comparison
