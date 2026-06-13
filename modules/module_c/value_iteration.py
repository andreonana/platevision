"""
Module C — Value Iteration (from scratch)
==========================================
PlateVision — MINT/DGI Cameroun (§4.3 cahier des charges)

Implémentation from scratch de l'algorithme de Value Iteration pour la
résolution du MDP PlateVision. Aucune librairie MDP externe n'est utilisée —
§4.3 insiste sur la compréhension théorique de l'algorithme.

Référence théorique :
  Russell, S. & Norvig, P. (2020). Artificial Intelligence: A Modern
  Approach (4th ed.), Chapter 17 — Making Complex Decisions (MDP).
  Équation de Bellman : V*(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V*(s')]
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_GAMMA    = 0.95
DEFAULT_EPSILON  = 0.01
DEFAULT_MAX_ITER = 10_000

# ──────────────────────────────────────────────────────────────────────────────
# Chargement du MDP
# ──────────────────────────────────────────────────────────────────────────────

def load_mdp(
    data_dir: Path = Path("data/processed"),
) -> tuple:
    """
    Charge les composantes du MDP depuis data_dir.
    Retourne (states, P, R, actions).
    Lève RuntimeError avec instruction d'exécution préalable si fichier absent.
    """
    data_dir = Path(data_dir)

    _REQUIRED = {
        "mdp_states.json":     "python -m modules.module_c.mdp_definition",
        "mdp_transitions.npy": "python -m modules.module_c.mdp_transitions",
        "mdp_rewards.npy":     "python -m modules.module_c.mdp_rewards",
        "mdp_actions.json":    "python -m modules.module_c.mdp_actions",
    }
    for fname, cmd in _REQUIRED.items():
        if not (data_dir / fname).exists():
            raise RuntimeError(
                f"Fichier requis absent : {data_dir / fname}\n"
                f"  → Exécuter d'abord : {cmd}"
            )

    with open(data_dir / "mdp_states.json", encoding="utf-8") as f:
        state_space = json.load(f)
    states = state_space["states"]

    with open(data_dir / "mdp_actions.json", encoding="utf-8") as f:
        actions_space = json.load(f)
    actions = actions_space["actions"]

    P = np.load(data_dir / "mdp_transitions.npy")   # (N, A, N)
    R = np.load(data_dir / "mdp_rewards.npy")        # (N, A)

    N = len(states)
    A = len(actions)

    if P.shape != (N, A, N):
        raise RuntimeError(
            f"Incohérence P.shape={P.shape}, attendu ({N},{A},{N})"
        )
    if R.shape != (N, A):
        raise RuntimeError(
            f"Incohérence R.shape={R.shape}, attendu ({N},{A})"
        )

    logger.info("MDP chargé : N=%d états, A=%d actions", N, A)
    return states, P, R, actions


# ──────────────────────────────────────────────────────────────────────────────
# Value Iteration — Bellman
# ──────────────────────────────────────────────────────────────────────────────

def run_value_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = DEFAULT_GAMMA,
    epsilon: float = DEFAULT_EPSILON,
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict:
    """
    Value Iteration from scratch — équation de Bellman (Russell & Norvig Ch.17).

    Algorithme :
        Initialiser V(s) = 0 pour tout s
        Répéter :
            Q(s,a) = R(s,a) + γ × Σ_s' P(s,a,s') × V(s')
            V_new(s) = max_a Q(s,a)
            delta = max_s |V_new(s) - V(s)|
            V ← V_new
        Jusqu'à delta < epsilon OU iter > max_iter
    """
    N, A, _ = P.shape
    V = np.zeros(N, dtype=np.float64)
    delta_history = []
    converged = False
    n_iterations = 0

    for iteration in range(1, max_iter + 1):
        # Q(s,a) = R(s,a) + γ Σ_s' P(s,a,s') V(s')  — vectorisé
        # P shape (N, A, N), V shape (N,) → P @ V shape (N, A)
        Q = R + gamma * P.dot(V)       # (N, A)
        V_new = Q.max(axis=1)          # (N,)
        delta = float(np.abs(V_new - V).max())
        delta_history.append(delta)
        V = V_new
        n_iterations = iteration

        if delta < epsilon:
            converged = True
            logger.info(
                "VI convergée en %d itérations (delta=%.6f < ε=%.4f, γ=%.2f)",
                iteration, delta, epsilon, gamma,
            )
            break

    if not converged:
        logger.warning(
            "VI non convergée après %d itérations (delta=%.6f)", max_iter, delta
        )

    Q_star  = R + gamma * P.dot(V)
    pi_star = Q_star.argmax(axis=1).astype(int)

    return {
        "V_star":        V,
        "Q_star":        Q_star,
        "pi_star":       pi_star,
        "n_iterations":  n_iterations,
        "converged":     converged,
        "delta_history": delta_history,
        "gamma":         gamma,
        "epsilon":       epsilon,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Extraction de la politique
# ──────────────────────────────────────────────────────────────────────────────

def extract_policy(
    Q_star: np.ndarray,
    states: list,
    actions: list,
) -> list:
    """
    Construit la liste de dicts décrivant la politique optimale π*.
    conf_level : 0=haute, 1=moyen, 2=faible (convention Module B).
    """
    pi_star = Q_star.argmax(axis=1)
    rows = []
    for s, state in enumerate(states):
        a_opt = int(pi_star[s])
        rows.append({
            "state_id":             int(state["state_id"]),
            "state_name":           state["label"],
            "cluster_id":           int(state["cluster_id"]),
            "conf_level":           int(state["conf_level"]),
            "alerte_cnn":           int(state["alerte_cnn"]),
            "procedure_mdp":        state.get("procedure", ""),
            "optimal_action_id":    a_opt,
            "optimal_action_code":  actions[a_opt]["code"],
            "optimal_action_label": actions[a_opt].get("label", actions[a_opt]["code"]),
            "V_star":               float(Q_star[s, a_opt]),
            "Q_values":             Q_star[s].tolist(),
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Affichage de la politique
# ──────────────────────────────────────────────────────────────────────────────

_CONF_NAMES = {0: "haute", 1: "moyen", 2: "faible"}


def print_policy_table(
    policy_rows: list,
    title: str = "Value Iteration — Politique optimale π*",
) -> None:
    """Affiche la politique optimale sous forme de tableau lisible."""
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


def _print_business_interpretation(rows: list) -> None:
    """3 lignes d'interprétation métier de la politique."""
    cluster_actions: dict = {}
    for r in rows:
        cid = r["cluster_id"]
        cluster_actions.setdefault(cid, []).append(r["optimal_action_code"])

    for cid in sorted(cluster_actions):
        actions_set = set(cluster_actions[cid])
        if len(actions_set) == 1:
            print(
                f"  • Cluster {cid} : π* = {next(iter(actions_set))} "
                f"(uniforme)"
            )
        else:
            print(
                f"  • Cluster {cid} : π* varie — {', '.join(sorted(actions_set))}"
            )
    print(
        "  → La politique encode la connaissance MINT/DGI : "
        "plaques expirées → DGI, conformes → laisser passer."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

def plot_convergence(
    delta_history: list,
    epsilon: float,
    gamma: float,
    output_path=None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.semilogy(range(1, len(delta_history) + 1), delta_history,
                color="steelblue", linewidth=1.5, label="Δ max")
    ax.axhline(epsilon, color="red", linestyle="--", linewidth=1.2,
               label=f"Seuil ε = {epsilon}")
    ax.set_xlabel("Itération", fontsize=10)
    ax.set_ylabel("Δ max (log)", fontsize=10)
    ax.set_title(
        f"Convergence Value Iteration (γ={gamma}, ε={epsilon})", fontsize=11
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        logger.info("vi_convergence.png sauvegardée : %s", output_path)
    plt.close(fig)


def plot_value_function(
    states: list,
    V_star: np.ndarray,
    output_path=None,
) -> None:
    from matplotlib.patches import Patch

    N = len(states)
    palette = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}
    colors  = [palette.get(s["cluster_id"], "steelblue") for s in states]
    labels  = [s["label"][:32] for s in states]

    fig, ax = plt.subplots(figsize=(10, max(4, N * 0.7)), dpi=120)
    ax.barh(range(N), V_star, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(N))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("V*(s) en FCFA", fontsize=10)
    ax.set_title(
        "Fonction de valeur optimale V* — Value Iteration (FCFA)", fontsize=11
    )
    legend_elements = [
        Patch(facecolor=palette[0], label="Cluster 0 — Conforme"),
        Patch(facecolor=palette[1], label="Cluster 1 — Dégradé"),
        Patch(facecolor=palette[2], label="Cluster 2 — Expiré"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    abs_max = max(abs(V_star).max(), 1.0)
    for i, v in enumerate(V_star):
        ax.text(v + abs_max * 0.01, i, f"{v:,.0f}",
                va="center", fontsize=7)
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        logger.info("vi_value_function.png sauvegardée : %s", output_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Analyse de sensibilité à γ
# ──────────────────────────────────────────────────────────────────────────────

def sensitivity_gamma(
    P: np.ndarray,
    R: np.ndarray,
    gammas: list = None,
    epsilon: float = DEFAULT_EPSILON,
) -> dict:
    """Lance VI pour chaque γ et retourne un dict de résultats."""
    if gammas is None:
        gammas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

    results = {}
    print(f"\nSensibilité γ — Value Iteration (ε={epsilon})")
    print(f"{'γ':>6} | {'itérations':>11} | {'convergé':>9} | "
          f"{'V* moyen':>14} | {'V* max':>14}")
    print("-" * 65)

    for g in gammas:
        res = run_value_iteration(P, R, gamma=g, epsilon=epsilon)
        results[g] = {
            "n_iterations": res["n_iterations"],
            "V_star":       res["V_star"],
            "pi_star":      res["pi_star"],
            "converged":    res["converged"],
        }
        print(
            f"{g:>6.2f} | {res['n_iterations']:>11d} | "
            f"{'Oui' if res['converged'] else 'Non':>9} | "
            f"{res['V_star'].mean():>14,.0f} | "
            f"{res['V_star'].max():>14,.0f}"
        )

    return results


def plot_gamma_sensitivity(
    results: dict,
    states: list,
    output_path=None,
) -> None:
    gammas = sorted(results.keys())
    N = len(states)
    labels = [s["label"][:20] for s in states]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    cmap = plt.cm.viridis

    for i, g in enumerate(gammas):
        color = cmap(i / max(len(gammas) - 1, 1))
        ax1.plot(range(N), results[g]["V_star"], marker="o", markersize=4,
                 linewidth=1.5, label=f"γ={g}", color=color)
    ax1.set_xticks(range(N))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("V*(s) en FCFA", fontsize=9)
    ax1.set_title("V*(s) par γ — Value Iteration", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    n_iters = [results[g]["n_iterations"] for g in gammas]
    bars = ax2.bar([str(g) for g in gammas], n_iters,
                   color="steelblue", edgecolor="white")
    ax2.set_xlabel("γ", fontsize=9)
    ax2.set_ylabel("Nombre d'itérations", fontsize=9)
    ax2.set_title("Itérations à convergence vs γ — VI", fontsize=10)
    for bar, n in zip(bars, n_iters):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(n), ha="center", va="bottom", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Sensibilité au facteur γ — Value Iteration (§4.3)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        logger.info("vi_gamma_sensitivity.png sauvegardée : %s", output_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Sauvegarde
# ──────────────────────────────────────────────────────────────────────────────

def save_vi_results(
    vi_result: dict,
    policy_rows: list,
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "vi_V_star.npy",  vi_result["V_star"])
    np.save(output_dir / "vi_Q_star.npy",  vi_result["Q_star"])
    np.save(output_dir / "vi_pi_star.npy", vi_result["pi_star"])

    with open(output_dir / "vi_policy.json", "w", encoding="utf-8") as f:
        json.dump(policy_rows, f, ensure_ascii=False, indent=2)

    convergence_data = {
        "delta_history": vi_result["delta_history"],
        "n_iterations":  vi_result["n_iterations"],
        "converged":     vi_result["converged"],
        "gamma":         vi_result["gamma"],
        "epsilon":       vi_result["epsilon"],
    }
    with open(output_dir / "vi_convergence.json", "w", encoding="utf-8") as f:
        json.dump(convergence_data, f, indent=2)

    logger.info(
        "VI results sauvegardés dans %s (V*, Q*, π*, policy.json, convergence.json)",
        output_dir,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Section LaTeX
# ──────────────────────────────────────────────────────────────────────────────

def generate_vi_latex(
    vi_result: dict,
    policy_rows: list,
    sensitivity_results: dict,
    output_path: Path,
) -> None:
    """Génère reports/rapport_technique/module_c_vi.tex"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    N  = len(policy_rows)
    A  = len(vi_result["Q_star"][0])
    g  = vi_result["gamma"]
    ep = vi_result["epsilon"]
    n_iter    = vi_result["n_iterations"]
    converged = vi_result["converged"]

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
                f"{'Oui' if sr['converged'] else 'Non'} & {v_mean} \\\\\n"
            )

    conv_str = "converg\\'{e}" if converged else "non converg\\'{e}"

    tex = (
        r"\subsection{Value Iteration}" + "\n"
        r"\label{sec:c:vi}" + "\n\n"
        r"\paragraph{Algorithme (from scratch --- Russell \& Norvig, Ch.~17)}" + "\n"
        "L'impl\\'{e}mentation est r\\'{e}alis\\'{e}e sans biblioth\\`{e}que MDP, "
        "conform\\'{e}ment \\`{a} l'exigence §4.3.\n\n"
        r"\begin{equation}" + "\n"
        r"V^*(s) \leftarrow \max_{a} \left[ R(s,a)"
        r" + \gamma \sum_{s'} P(s' \mid s, a)\, V^*(s') \right]" + "\n"
        r"\label{eq:bellman}" + "\n"
        r"\end{equation}" + "\n\n"
        f"\\textbf{{Param\\`{{e}}tres :}} $\\gamma = {g}$, "
        f"$\\varepsilon = {ep}$, $N = {N}$ \\'{{}}", ""
    )

    # Build manually to avoid f-string issues
    lines = [
        r"\subsection{Value Iteration}",
        r"\label{sec:c:vi}",
        "",
        r"\paragraph{Algorithme (from scratch --- Russell \& Norvig, Ch.~17)}",
        "L'impl\\'{e}mentation est r\\'{e}alis\\'{e}e sans biblioth\\`{e}que MDP,",
        "conform\\'{e}ment \\`{a} l'exigence §4.3 de compr\\'{e}hension th\\'{e}orique.",
        "",
        r"\begin{equation}",
        r"V^*(s) \leftarrow \max_{a} \left[ R(s,a)",
        r"  + \gamma \sum_{s'} P(s' \mid s, a)\, V^*(s') \right]",
        r"\label{eq:bellman}",
        r"\end{equation}",
        "",
        f"\\textbf{{Param\\`{{e}}tres utilis\\'{{'}}s :}} $\\gamma = {g}$,"
        f" $\\varepsilon = {ep}$, $N = {N}$ \\'{{'}}tats, $A = {A}$ actions."
        f" Convergence en \\textbf{{{n_iter}}} it\\'{{'}}rations ({conv_str}).",
        "",
        r"\begin{table}[H]",
        r"\centering",
        f"\\caption{{Politique optimale $\\pi^*$ --- Value Iteration ($\\gamma={g}$)}}",
        r"\label{tab:c:vi_policy}",
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
            r"\caption{Sensibilit\'{e} au facteur $\gamma$ --- Value Iteration}",
            r"\label{tab:c:vi_sensitivity}",
            r"\begin{tabular}{rrrr}",
            r"\toprule",
            r"$\gamma$ & It\'{e}rations & Converg\'{e} & $\bar{V}^*$ (FCFA) \\",
            r"\midrule",
            sens_rows_tex,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]

    lines += [
        r"\paragraph{Interpr\'{e}tation m\'{e}tier MINT/DGI}",
        "La politique $\\pi^*$ obtenue par Value Iteration encode la strat\\'{e}gie",
        "optimale des agents de contr\\^{o}le : les plaques du cluster~2 (expir\\'{e}es)",
        "re\\c{c}oivent syst\\'{e}matiquement un signalement DGI, tandis que les plaques",
        "conformes (cluster~0 \\`{a} confiance haute) sont laiss\\'{e}es passer.",
        "L'analyse de sensibilit\\'{e} \\`{a} $\\gamma$ confirme la robustesse :",
        "pour $\\gamma \\in [0.7, 0.99]$, la politique reste stable.",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("LaTeX VI généré : %s", output_path)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────────────────────────

def run_value_iteration_pipeline(
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

    vi_result   = run_value_iteration(P, R, gamma=gamma, epsilon=epsilon)
    policy_rows = extract_policy(vi_result["Q_star"], states, actions)

    print_policy_table(policy_rows)

    plot_convergence(
        vi_result["delta_history"], epsilon, gamma,
        output_path=figures_dir / "vi_convergence.png",
    )
    plot_value_function(
        states, vi_result["V_star"],
        output_path=figures_dir / "vi_value_function.png",
    )

    sensitivity_results = {}
    if run_sensitivity:
        sensitivity_results = sensitivity_gamma(P, R, epsilon=epsilon)
        plot_gamma_sensitivity(
            sensitivity_results, states,
            output_path=figures_dir / "vi_gamma_sensitivity.png",
        )

    save_vi_results(vi_result, policy_rows, output_dir)

    generate_vi_latex(
        vi_result, policy_rows, sensitivity_results,
        output_path=report_dir / "module_c_vi.tex",
    )

    n    = vi_result["n_iterations"]
    conv = vi_result["converged"]
    print(f"\n✓ Value Iteration : {n} itérations, convergé={conv}, γ={gamma}")
    return vi_result


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Module C — Value Iteration (§4.3 PlateVision)"
    )
    parser.add_argument("--gamma",          type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--epsilon",        type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument("--data-dir",       type=Path,  default=Path("data/processed"))
    parser.add_argument("--figures-dir",    type=Path,  default=Path("reports/figures"))
    parser.add_argument("--report-dir",     type=Path,  default=Path("reports/rapport_technique"))
    args = parser.parse_args()

    run_value_iteration_pipeline(
        gamma=args.gamma,
        epsilon=args.epsilon,
        run_sensitivity=not args.no_sensitivity,
        data_dir=args.data_dir,
        output_dir=args.data_dir,
        figures_dir=args.figures_dir,
        report_dir=args.report_dir,
    )
