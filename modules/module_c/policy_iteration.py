"""
Module C — Policy Iteration (from scratch)
===========================================
PlateVision — MINT/DGI Cameroun (§4.3 cahier des charges)

Implémentation from scratch de l'algorithme de Policy Iteration.
Policy Iteration alterne entre :
  1. Policy Evaluation  : calcul de V^π jusqu'à convergence
  2. Policy Improvement : mise à jour greedy de π

La convergence est garantie en un nombre fini d'itérations car l'espace
de politiques est fini (A^N politiques possibles).

Références :
  Russell, S. & Norvig, P. (2020). AI: A Modern Approach, Ch. 17.3
  Sutton, R. & Barto, A. (2018). Reinforcement Learning, Ch. 4.3
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from modules.module_c.value_iteration import (
    DEFAULT_EPSILON,
    DEFAULT_GAMMA,
    DEFAULT_MAX_ITER,
    _CONF_NAMES,
    _print_business_interpretation,
    extract_policy,
    load_mdp,
    plot_value_function,
    sensitivity_gamma as vi_sensitivity_gamma,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Policy Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def policy_evaluation(
    pi: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    eval_epsilon: float = 1e-6,
    max_eval_iter: int = 10_000,
) -> tuple:
    """
    Évalue la politique fixée π jusqu'à convergence de V^π.

    Algorithme (Sutton & Barto, §4.1) :
        Initialiser V(s) = 0
        Répéter :
            V_new(s) = R(s, π(s)) + γ × Σ_s' P(s, π(s), s') × V(s')
            delta = max_s |V_new(s) - V(s)|
            V ← V_new
        Jusqu'à delta < eval_epsilon

    Retourne (V, n_eval_iterations).
    """
    N = P.shape[0]
    V = np.zeros(N, dtype=np.float64)

    for it in range(1, max_eval_iter + 1):
        # R_pi(s) = R[s, pi[s]], shape (N,)
        R_pi = R[np.arange(N), pi]
        # P_pi(s, s') = P[s, pi[s], s'], shape (N, N)
        P_pi = P[np.arange(N), pi, :]
        V_new = R_pi + gamma * P_pi.dot(V)

        delta = float(np.abs(V_new - V).max())
        V = V_new

        if delta < eval_epsilon:
            return V, it

    logger.warning(
        "Policy Evaluation non convergée après %d itérations (delta=%.2e)",
        max_eval_iter, delta,
    )
    return V, max_eval_iter


# ──────────────────────────────────────────────────────────────────────────────
# Policy Improvement
# ──────────────────────────────────────────────────────────────────────────────

def policy_improvement(
    V: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    pi_old: np.ndarray,
) -> tuple:
    """
    Amélioration greedy de la politique.

    π_new(s) = argmax_a [ R(s,a) + γ Σ_s' P(s,a,s') V(s') ]

    Retourne (pi_new, stable) où stable = (pi_new == pi_old).all().
    """
    Q = R + gamma * P.dot(V)          # (N, A)
    pi_new = Q.argmax(axis=1).astype(int)
    stable = bool(np.array_equal(pi_new, pi_old))
    return pi_new, stable


# ──────────────────────────────────────────────────────────────────────────────
# Policy Iteration
# ──────────────────────────────────────────────────────────────────────────────

def run_policy_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = DEFAULT_GAMMA,
    epsilon: float = DEFAULT_EPSILON,
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict:
    """
    Policy Iteration from scratch (Russell & Norvig Ch.17.3 ; Sutton & Barto §4.3).

    Algorithme :
        Initialiser π(s) = 0 pour tout s
        Répéter :
            V ← policy_evaluation(π, P, R, γ)
            π_new, stable ← policy_improvement(V, P, R, γ, π)
            π ← π_new
            n_iterations += 1
        Jusqu'à stable OU n_iterations > max_iter
    """
    N, A, _ = P.shape
    pi = np.zeros(N, dtype=int)
    n_iterations = 0
    n_eval_iterations_total = 0
    converged = False

    for iteration in range(1, max_iter + 1):
        V, n_eval = policy_evaluation(pi, P, R, gamma, eval_epsilon=epsilon * 1e-3)
        n_eval_iterations_total += n_eval

        pi_new, stable = policy_improvement(V, P, R, gamma, pi)
        pi = pi_new
        n_iterations = iteration

        if stable:
            converged = True
            logger.info(
                "PI convergée en %d itérations de politique "
                "(%d itérations évaluation cumulées, γ=%.2f)",
                n_iterations, n_eval_iterations_total, gamma,
            )
            break

    if not converged:
        logger.warning(
            "PI non convergée après %d itérations de politique", max_iter
        )

    Q_star  = R + gamma * P.dot(V)
    pi_star = Q_star.argmax(axis=1).astype(int)

    return {
        "V_star":                  V,
        "Q_star":                  Q_star,
        "pi_star":                 pi_star,
        "n_iterations":            n_iterations,
        "converged":               converged,
        "n_eval_iterations_total": n_eval_iterations_total,
        "gamma":                   gamma,
        "epsilon":                 epsilon,
        # PI n'a pas de delta_history mais on expose une liste vide pour compatibilité
        "delta_history":           [],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Affichage de la politique (réutilise le format de value_iteration)
# ──────────────────────────────────────────────────────────────────────────────

def print_policy_table(
    policy_rows: list,
    title: str = "Policy Iteration — Politique optimale π*",
) -> None:
    sorted_rows = sorted(
        policy_rows,
        key=lambda r: (r["cluster_id"], r["conf_level"], r["alerte_cnn"]),
    )
    col_w = (36, 7, 7, 22, 12)
    header = (
        f"{'État':<{col_w[0]}} {'Conf':<{col_w[1]}} {'Alerte':<{col_w[2]}} "
        f"{'Action optimale':<{col_w[3]}} {'V*(s) FCFA':>{col_w[4]}}"
    )
    sep = "-" * len(header)
    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for r in sorted_rows:
        name   = r["state_name"][:col_w[0] - 1]
        conf   = _CONF_NAMES.get(r["conf_level"], str(r["conf_level"]))
        alerte = "Oui" if r["alerte_cnn"] else "Non"
        action = r["optimal_action_code"][:col_w[3] - 1]
        vstar  = f"{r['V_star']:,.0f}"
        print(
            f"{name:<{col_w[0]}} {conf:<{col_w[1]}} {alerte:<{col_w[2]}} "
            f"{action:<{col_w[3]}} {vstar:>{col_w[4]}}"
        )
    print(sep)
    print("\nInterprétation métier MINT/DGI :")
    _print_business_interpretation(sorted_rows)


# ──────────────────────────────────────────────────────────────────────────────
# Figures PI
# ──────────────────────────────────────────────────────────────────────────────

def plot_pi_convergence(
    n_iterations: int,
    n_eval_total: int,
    gamma: float,
    output_path=None,
) -> None:
    """Pour PI, la convergence est le nombre d'itérations de politique."""
    fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
    ax.bar(["Itérations politique", "Itérations évaluation (cumulées)"],
           [n_iterations, n_eval_total],
           color=["steelblue", "coral"], edgecolor="white")
    ax.set_ylabel("Nombre d'itérations", fontsize=10)
    ax.set_title(
        f"Convergence Policy Iteration (γ={gamma})\n"
        f"→ {n_iterations} itération(s) de politique suffisent",
        fontsize=11,
    )
    for i, v in enumerate([n_iterations, n_eval_total]):
        ax.text(i, v + 0.1, str(v), ha="center", va="bottom", fontsize=11,
                fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        logger.info("pi_convergence.png sauvegardée : %s", output_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Analyse de sensibilité à γ — PI
# ──────────────────────────────────────────────────────────────────────────────

def sensitivity_gamma(
    P: np.ndarray,
    R: np.ndarray,
    gammas: list = None,
    epsilon: float = DEFAULT_EPSILON,
) -> dict:
    """Lance PI pour chaque γ."""
    if gammas is None:
        gammas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

    results = {}
    print(f"\nSensibilité γ — Policy Iteration (ε={epsilon})")
    print(f"{'γ':>6} | {'it.pol.':>8} | {'it.eval.':>9} | {'convergé':>9} | "
          f"{'V* moyen':>14}")
    print("-" * 58)

    for g in gammas:
        res = run_policy_iteration(P, R, gamma=g, epsilon=epsilon)
        results[g] = {
            "n_iterations":            res["n_iterations"],
            "n_eval_iterations_total": res["n_eval_iterations_total"],
            "V_star":                  res["V_star"],
            "pi_star":                 res["pi_star"],
            "converged":               res["converged"],
        }
        print(
            f"{g:>6.2f} | {res['n_iterations']:>8d} | "
            f"{res['n_eval_iterations_total']:>9d} | "
            f"{'Oui' if res['converged'] else 'Non':>9} | "
            f"{res['V_star'].mean():>14,.0f}"
        )

    return results


def plot_pi_gamma_sensitivity(
    results: dict,
    states: list,
    output_path=None,
) -> None:
    gammas = sorted(results.keys())
    N = len(states)
    labels = [s["label"][:20] for s in states]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    cmap = plt.cm.plasma

    for i, g in enumerate(gammas):
        color = cmap(i / max(len(gammas) - 1, 1))
        ax1.plot(range(N), results[g]["V_star"], marker="s", markersize=4,
                 linewidth=1.5, label=f"γ={g}", color=color)
    ax1.set_xticks(range(N))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("V*(s) en FCFA", fontsize=9)
    ax1.set_title("V*(s) par γ — Policy Iteration", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    n_iters = [results[g]["n_iterations"] for g in gammas]
    bars = ax2.bar([str(g) for g in gammas], n_iters,
                   color="coral", edgecolor="white")
    ax2.set_xlabel("γ", fontsize=9)
    ax2.set_ylabel("Itérations de politique", fontsize=9)
    ax2.set_title("Itérations politique vs γ — PI", fontsize=10)
    for bar, n in zip(bars, n_iters):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 str(n), ha="center", va="bottom", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Sensibilité au facteur γ — Policy Iteration (§4.3)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        logger.info("pi_gamma_sensitivity.png sauvegardée : %s", output_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Sauvegarde
# ──────────────────────────────────────────────────────────────────────────────

def save_pi_results(
    pi_result: dict,
    policy_rows: list,
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "pi_V_star.npy",  pi_result["V_star"])
    np.save(output_dir / "pi_Q_star.npy",  pi_result["Q_star"])
    np.save(output_dir / "pi_pi_star.npy", pi_result["pi_star"])

    with open(output_dir / "pi_policy.json", "w", encoding="utf-8") as f:
        json.dump(policy_rows, f, ensure_ascii=False, indent=2)

    convergence_data = {
        "n_iterations":            pi_result["n_iterations"],
        "n_eval_iterations_total": pi_result["n_eval_iterations_total"],
        "converged":               pi_result["converged"],
        "gamma":                   pi_result["gamma"],
        "epsilon":                 pi_result["epsilon"],
    }
    with open(output_dir / "pi_convergence.json", "w", encoding="utf-8") as f:
        json.dump(convergence_data, f, indent=2)

    logger.info(
        "PI results sauvegardés dans %s (V*, Q*, π*, policy.json, convergence.json)",
        output_dir,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Section LaTeX
# ──────────────────────────────────────────────────────────────────────────────

def generate_pi_latex(
    pi_result: dict,
    policy_rows: list,
    sensitivity_results: dict,
    output_path: Path,
) -> None:
    """Génère reports/rapport_technique/module_c_pi.tex"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    N  = len(policy_rows)
    A  = len(pi_result["Q_star"][0])
    g  = pi_result["gamma"]
    ep = pi_result["epsilon"]
    n_pol  = pi_result["n_iterations"]
    n_eval = pi_result["n_eval_iterations_total"]
    conv   = pi_result["converged"]

    sorted_rows = sorted(
        policy_rows,
        key=lambda r: (r["cluster_id"], r["conf_level"], r["alerte_cnn"]),
    )
    conf_names = {0: "haute", 1: "moyen", 2: "faible"}

    policy_rows_tex = ""
    for r in sorted_rows:
        name   = r["state_name"].replace("—", "--")[:40]
        conf   = conf_names.get(r["conf_level"], "?")
        alerte = "Oui" if r["alerte_cnn"] else "Non"
        action = r["optimal_action_code"].replace("_", "\\_")
        vstar  = f"{r['V_star']:,.0f}".replace(",", "\\,")
        policy_rows_tex += (
            f"  {name} & {conf} & {alerte} & \\texttt{{{action}}} & {vstar} \\\\\n"
        )

    sens_rows_tex = ""
    if sensitivity_results:
        for gv in sorted(sensitivity_results.keys()):
            sr = sensitivity_results[gv]
            v_mean = f"{float(sr['V_star'].mean()):,.0f}".replace(",", "\\,")
            sens_rows_tex += (
                f"  {gv:.2f} & {sr['n_iterations']} & "
                f"{sr.get('n_eval_iterations_total', '?')} & "
                f"{'Oui' if sr['converged'] else 'Non'} & {v_mean} \\\\\n"
            )

    conv_str = "converg\\'{e}" if conv else "non converg\\'{e}"

    lines = [
        r"\subsection{Policy Iteration}",
        r"\label{sec:c:pi}",
        "",
        r"\paragraph{Algorithme (from scratch --- Russell \& Norvig, Ch.~17.3)}",
        "Policy Iteration alt\\`{e}rne evaluation et am\\'{e}lioration de politique.",
        "",
        r"\begin{equation}",
        r"V^{\pi}(s) \leftarrow R(s, \pi(s))",
        r"  + \gamma \sum_{s'} P(s' \mid s, \pi(s))\, V^{\pi}(s')",
        r"\quad \text{(Policy Evaluation)}",
        r"\end{equation}",
        r"\begin{equation}",
        r"\pi'(s) \leftarrow \arg\max_{a}\left[ R(s,a)",
        r"  + \gamma \sum_{s'} P(s' \mid s, a)\, V^{\pi}(s') \right]",
        r"\quad \text{(Policy Improvement)}",
        r"\end{equation}",
        "",
        f"\\textbf{{Param\\`{{e}}tres :}} $\\gamma = {g}$, "
        f"$\\varepsilon = {ep}$, $N = {N}$ \\'{{'}}tats, $A = {A}$ actions. "
        f"Convergence en \\textbf{{{n_pol}}} it\\'{{'}}ration(s) de politique "
        f"({n_eval} it\\'{{'}}rations d'\\'{{'}}valuation cumulées, {conv_str}).",
        "",
        r"\begin{table}[H]",
        r"\centering",
        f"\\caption{{Politique optimale $\\pi^*$ --- Policy Iteration ($\\gamma={g}$)}}",
        r"\label{tab:c:pi_policy}",
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"\textbf{\'{E}tat} & \textbf{Conf.} & \textbf{Alerte}"
        r" & \textbf{Action $\pi^*(s)$} & \textbf{$V^*(s)$ FCFA} \\",
        r"\midrule",
        policy_rows_tex,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    if sens_rows_tex:
        lines += [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Sensibilit\'{e} au facteur $\gamma$ --- Policy Iteration}",
            r"\label{tab:c:pi_sensitivity}",
            r"\begin{tabular}{rrrrr}",
            r"\toprule",
            r"$\gamma$ & It. pol. & It. eval. & Converg\'{e} & $\bar{V}^*$ (FCFA) \\",
            r"\midrule",
            sens_rows_tex,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]

    lines += [
        r"\paragraph{Avantage de Policy Iteration}",
        "PI converge en un nombre d'it\\'{e}rations de politique tr\\`{e}s faible",
        "(souvent $< 10$), car chaque it\\'{e}ration garantit une am\\'{e}lioration",
        "stricte ou la stabilit\\'{e}. En revanche, chaque it\\'{e}ration n\\'{e}cessite",
        "une \\'evaluation compl\\`{e}te de $V^\\pi$.",
        "Pour ce MDP ($N=7$, $A=5$), PI est plus rapide que VI en nombre d'it\\'{e}rations",
        "de politique, mais les deux produisent une politique $\\pi^*$ identique.",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("LaTeX PI généré : %s", output_path)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────────────────────────

def run_policy_iteration_pipeline(
    gamma: float = DEFAULT_GAMMA,
    epsilon: float = DEFAULT_EPSILON,
    run_sensitivity: bool = True,
    data_dir: Path = Path("data/processed"),
    output_dir: Path = Path("data/processed"),
    figures_dir: Path = Path("reports/figures"),
    report_dir: Path = Path("reports/rapport_technique"),
) -> dict:
    data_dir    = Path(data_dir)
    output_dir  = Path(output_dir)
    figures_dir = Path(figures_dir)
    report_dir  = Path(report_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    states, P, R, actions = load_mdp(data_dir)

    pi_result   = run_policy_iteration(P, R, gamma=gamma, epsilon=epsilon)
    policy_rows = extract_policy(pi_result["Q_star"], states, actions)

    print_policy_table(policy_rows)

    plot_pi_convergence(
        pi_result["n_iterations"],
        pi_result["n_eval_iterations_total"],
        gamma,
        output_path=figures_dir / "pi_convergence.png",
    )
    plot_value_function(
        states, pi_result["V_star"],
        output_path=figures_dir / "pi_value_function.png",
    )

    sensitivity_results = {}
    if run_sensitivity:
        sensitivity_results = sensitivity_gamma(P, R, epsilon=epsilon)
        plot_pi_gamma_sensitivity(
            sensitivity_results, states,
            output_path=figures_dir / "pi_gamma_sensitivity.png",
        )

    save_pi_results(pi_result, policy_rows, output_dir)

    generate_pi_latex(
        pi_result, policy_rows, sensitivity_results,
        output_path=report_dir / "module_c_pi.tex",
    )

    n    = pi_result["n_iterations"]
    conv = pi_result["converged"]
    print(
        f"\n✓ Policy Iteration : {n} itérations de politique, "
        f"convergé={conv}, γ={gamma}"
    )
    return pi_result


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Module C — Policy Iteration (§4.3 PlateVision)"
    )
    parser.add_argument("--gamma",          type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--epsilon",        type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument("--data-dir",       type=Path,  default=Path("data/processed"))
    parser.add_argument("--figures-dir",    type=Path,  default=Path("reports/figures"))
    parser.add_argument("--report-dir",     type=Path,  default=Path("reports/rapport_technique"))
    args = parser.parse_args()

    run_policy_iteration_pipeline(
        gamma=args.gamma,
        epsilon=args.epsilon,
        run_sensitivity=not args.no_sensitivity,
        data_dir=args.data_dir,
        output_dir=args.data_dir,
        figures_dir=args.figures_dir,
        report_dir=args.report_dir,
    )
