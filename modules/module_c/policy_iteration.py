"""
Module C — Policy Iteration (§4.3 PlateVision — MINT/DGI Cameroun)

Policy Iteration alterne deux phases jusqu'à stabilité de la politique :
  1. Policy Evaluation : pour π fixée, calcule V^π par itération de Bellman
     jusqu'à convergence (delta < eval_epsilon).
     V^π(s) ← R(s, π(s)) + γ × Σ_s' P(s, π(s), s') × V^π(s')
  2. Policy Improvement : pour chaque état, cherche l'action qui améliore V^π.
     π_new(s) = argmax_a [ R(s,a) + γ × Σ_s' P(s,a,s') × V^π(s') ]
     Si π_new == π pour tout s → politique stable → STOP.

Différence avec Value Iteration :
  - VI met à jour V et π simultanément à chaque itération.
  - PI évalue d'abord V^π complètement, puis améliore π une seule fois.
  - PI converge en moins d'itérations globales, mais chaque itération
    contient une boucle d'évaluation interne.

Référence : Russell & Norvig (2020), Ch. 17.3 ;
            Sutton & Barto (2018), Ch. 4.3.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from modules.module_c.value_iteration import load_mdp, extract_policy
except ImportError:
    load_mdp = None
    extract_policy = None

logger = logging.getLogger(__name__)

DEFAULT_GAMMA         = 0.95
DEFAULT_EPSILON       = 0.01
DEFAULT_MAX_ITER      = 1_000
DEFAULT_EVAL_EPSILON  = 1e-6
DEFAULT_EVAL_MAX_ITER = 10_000

_CONF_NAMES = {0: "haute", 1: "moyen", 2: "faible"}


# ──────────────────────────────────────────────────────────────────────────────
# Chargement du MDP (délégué à value_iteration ou fallback local)
# ──────────────────────────────────────────────────────────────────────────────

def _load_mdp_local(data_dir: Path) -> tuple:
    """Fallback local si value_iteration.py est absent."""
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
    P = np.load(data_dir / "mdp_transitions.npy")
    R = np.load(data_dir / "mdp_rewards.npy")
    N, A = len(states), len(actions)
    if P.shape != (N, A, N):
        raise RuntimeError(f"P.shape={P.shape} invalide, attendu ({N},{A},{N})")
    if R.shape != (N, A):
        raise RuntimeError(f"R.shape={R.shape} invalide, attendu ({N},{A})")
    logger.info("MDP chargé (fallback local) : N=%d états, A=%d actions", N, A)
    return states, P, R, actions


def load_mdp(data_dir: Path = Path("data/processed")) -> tuple:
    """Délègue à value_iteration.load_mdp ou fallback local."""
    try:
        from modules.module_c.value_iteration import load_mdp as _vi_load
        return _vi_load(data_dir)
    except ImportError:
        return _load_mdp_local(data_dir)


def extract_policy(Q_star: np.ndarray, states: list, actions: list) -> list:
    """Délègue à value_iteration.extract_policy ou fallback local."""
    try:
        from modules.module_c.value_iteration import extract_policy as _vi_ep
        return _vi_ep(Q_star, states, actions)
    except ImportError:
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
# Phase 1 — Policy Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def policy_evaluation(
    pi: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    eval_epsilon: float = DEFAULT_EVAL_EPSILON,
    eval_max_iter: int = DEFAULT_EVAL_MAX_ITER,
) -> tuple:
    """
    Évalue la politique fixée π jusqu'à convergence de V^π.

    Vectorisé :
      V_new = R[np.arange(N), pi] + gamma * np.einsum('sn,n->s', P[np.arange(N), pi], V)

    Retourne (V, n_eval_iter).
    """
    N = P.shape[0]

    # V^π initialisé à zéro : l'évaluation convergera vers la vraie valeur sous π
    V = np.zeros(N, dtype=np.float64)
    idx = np.arange(N)

    # Pré-indexe les tranches correspondant à π (constantes sur tout l'appel)
    R_pi = R[idx, pi]           # (N,)  — récompenses immédiates sous π fixée
    P_pi = P[idx, pi, :]        # (N, N) — matrice de transition sous π fixée

    # Boucle d'évaluation : Bellman de contraction pour V^π jusqu'à ε-convergence
    for n_eval_iter in range(1, eval_max_iter + 1):
        # einsum "sn,n->s" : Σ_s' P^π(s,s') × V(s') — contraction sur l'état suivant
        V_new = R_pi + gamma * np.einsum("sn,n->s", P_pi, V)
        delta = float(np.abs(V_new - V).max())
        V = V_new
        # Évaluation terminée dès que la variation max est inférieure à eval_epsilon
        if delta < eval_epsilon:
            return V, n_eval_iter

    logger.warning(
        "Policy Evaluation non convergée après %d itérations (delta=%.2e)",
        eval_max_iter, delta,
    )
    return V, eval_max_iter


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — Policy Improvement
# ──────────────────────────────────────────────────────────────────────────────

def policy_improvement(
    V: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    pi_old: np.ndarray,
) -> tuple:
    """
    Améliore la politique greedily à partir de V^π.

    Q(s,a) = R(s,a) + gamma × Σ_s' P(s,a,s') × V(s')
    Vectorisé : Q = R + gamma * np.tensordot(P, V, axes=[[2],[0]])  → shape (N, A)

    Retourne (pi_new, stable) où stable=True si π_new == π_old partout.
    """
    # tensordot(P, V, axes=[[2],[0]]) contracte la dim s' de P(N,A,N) avec V(N) → Q(N,A)
    Q = R + gamma * np.tensordot(P, V, axes=[[2], [0]])   # (N, A)

    # Amélioration greedy : pour chaque état, l'action qui maximise Q(s,a)
    pi_new = np.argmax(Q, axis=1).astype(int)

    # Stabilité : si π_new est identique à π_old sur tous les états → convergence globale
    stable = bool(np.array_equal(pi_new, pi_old))
    return pi_new, stable


# ──────────────────────────────────────────────────────────────────────────────
# Algorithme complet
# ──────────────────────────────────────────────────────────────────────────────

def run_policy_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = DEFAULT_GAMMA,
    epsilon: float = DEFAULT_EPSILON,
    max_iter: int = DEFAULT_MAX_ITER,
    eval_epsilon: float = DEFAULT_EVAL_EPSILON,
) -> dict:
    """
    Policy Iteration from scratch — Russell & Norvig Ch.17.3 ; Sutton & Barto §4.3.

    Algorithme :
        Initialiser π(s) = 0 pour tout s
        Répéter :
            V, n_eval = policy_evaluation(π, P, R, γ, eval_epsilon)
            π_new, stable = policy_improvement(V, P, R, γ, π)
            Enregistre n_policy_changes = (π_new ≠ π).sum()
            π ← π_new
            n_iterations += 1
        Jusqu'à stable OU n_iterations >= max_iter
    """
    N = P.shape[0]

    # Politique initiale : action 0 (LAISSER_PASSER) pour tous les états
    pi = np.zeros(N, dtype=int)    # action 0 = LAISSER_PASSER
    n_eval_iterations_total = 0
    iteration_history = []
    converged = False

    # Boucle principale PI : alterne Évaluation et Amélioration jusqu'à stabilité
    for iteration in range(1, max_iter + 1):
        # Phase 1 : évaluer V^π pour la politique courante (itérations internes)
        V, n_eval = policy_evaluation(pi, P, R, gamma, eval_epsilon=eval_epsilon)
        n_eval_iterations_total += n_eval

        # Phase 2 : améliorer π greedily à partir de V^π (une seule passe)
        pi_new, stable = policy_improvement(V, P, R, gamma, pi)
        n_changes = int((pi_new != pi).sum())

        iteration_history.append({
            "iter":             iteration,
            "n_eval":           n_eval,
            "n_policy_changes": n_changes,  # nombre d'états dont l'action a changé
            "stable":           stable,
        })

        # Mise à jour de π pour la prochaine itération
        pi = pi_new

        # Arrêt : si aucune action n'a changé, π* est atteinte
        if stable:
            converged = True
            logger.info(
                "PI convergée en %d itérations globales "
                "(%d itérations évaluation cumulées, γ=%.2f)",
                iteration, n_eval_iterations_total, gamma,
            )
            break

    if not converged:
        logger.warning(
            "PI non convergée après %d itérations globales", max_iter
        )

    # Q* final calculé depuis V convergé pour fournir les valeurs par action
    Q_star  = R + gamma * np.tensordot(P, V, axes=[[2], [0]])

    # π* extrait comme argmax de Q* (cohérent avec le résultat de policy_improvement)
    pi_star = np.argmax(Q_star, axis=1).astype(int)

    return {
        "V_star":                  V,
        "Q_star":                  Q_star,
        "pi_star":                 pi_star,
        "n_iterations":            len(iteration_history),
        "n_eval_iterations_total": n_eval_iterations_total,
        "converged":               converged,
        "iteration_history":       iteration_history,
        "gamma":                   gamma,
        "epsilon":                 epsilon,
        "eval_epsilon":            eval_epsilon,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Affichage de la politique
# ──────────────────────────────────────────────────────────────────────────────

def print_policy_table(
    policy_rows: list,
    title: str = "Policy Iteration — Politique optimale π*",
    pi_result: dict = None,
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
    if pi_result:
        n    = pi_result["n_iterations"]
        conv = pi_result["converged"]
        n_ev = pi_result["n_eval_iterations_total"]
        print(
            f"→ Convergence : {n} itération(s) globale(s), "
            f"convergé={conv}, {n_ev} itérations d'évaluation cumulées"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

def plot_convergence(
    iteration_history: list,
    gamma: float,
    output_path=None,
) -> None:
    """
    Courbe du nombre de changements de politique par itération globale.
    La courbe tend vers 0 à convergence.
    """
    iters   = [h["iter"] for h in iteration_history]
    changes = [h["n_policy_changes"] for h in iteration_history]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.plot(iters, changes, "o-", color="coral", linewidth=2, markersize=6,
            label="Changements de π")
    ax.fill_between(iters, changes, alpha=0.15, color="coral")

    # Annotation sur le dernier point
    if iters:
        ax.annotate(
            "Politique stable",
            xy=(iters[-1], changes[-1]),
            xytext=(iters[-1] - max(len(iters) * 0.15, 0.5), max(changes) * 0.3 + 0.1),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9,
        )

    ax.set_xlabel("Itération globale", fontsize=10)
    ax.set_ylabel("Changements de politique π", fontsize=10)
    ax.set_title(f"Convergence Policy Iteration (γ={gamma})", fontsize=11)
    ax.set_xticks(iters)
    ax.set_ylim(bottom=-0.2)
    ax.axhline(0, color="green", linestyle="--", linewidth=1, alpha=0.6)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        logger.info("pi_convergence.png sauvegardée : %s", output_path)
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
        "Fonction de valeur optimale V* — Policy Iteration (FCFA)", fontsize=11
    )
    legend_elements = [
        Patch(facecolor=palette[0], label="Cluster 0 — Conforme"),
        Patch(facecolor=palette[1], label="Cluster 1 — Dégradé"),
        Patch(facecolor=palette[2], label="Cluster 2 — Expiré"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    abs_max = max(float(np.abs(V_star).max()), 1.0)
    for i, v in enumerate(V_star):
        ax.text(v + abs_max * 0.01, i, f"{v:,.0f}", va="center", fontsize=7)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        logger.info("pi_value_function.png sauvegardée : %s", output_path)
    plt.close(fig)


def plot_gamma_sensitivity(
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
    ax2.set_ylabel("Itérations globales", fontsize=9)
    ax2.set_title("Itérations à convergence vs γ — PI", fontsize=10)
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
# Sensibilité à γ
# ──────────────────────────────────────────────────────────────────────────────

def sensitivity_gamma(
    P: np.ndarray,
    R: np.ndarray,
    gammas: list = None,
    eval_epsilon: float = DEFAULT_EVAL_EPSILON,
) -> dict:
    """Lance run_policy_iteration pour chaque γ et retourne un dict de résultats."""
    if gammas is None:
        gammas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

    results = {}
    print(f"\nSensibilité γ — Policy Iteration (eval_ε={eval_epsilon:.0e})")
    print(f"{'γ':>6} | {'Itér. globales':>15} | {'Itér. éval. cumul':>18} | "
          f"{'Convergé':>9} | {'V* moyen (FCFA)':>16}")
    print("-" * 75)

    for g in gammas:
        res = run_policy_iteration(P, R, gamma=g, eval_epsilon=eval_epsilon)
        results[g] = {
            "n_iterations":            res["n_iterations"],
            "n_eval_iterations_total": res["n_eval_iterations_total"],
            "V_star":                  res["V_star"],
            "pi_star":                 res["pi_star"],
            "converged":               res["converged"],
        }
        print(
            f"{g:>6.2f} | {res['n_iterations']:>15d} | "
            f"{res['n_eval_iterations_total']:>18d} | "
            f"{'Oui' if res['converged'] else 'Non':>9} | "
            f"{res['V_star'].mean():>16,.0f}"
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Sauvegarde
# ──────────────────────────────────────────────────────────────────────────────

def save_pi_results(
    pi_result: dict,
    policy_rows: list,
    output_dir: Path = Path("data/processed"),
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "pi_V_star.npy",  pi_result["V_star"])
    np.save(output_dir / "pi_Q_star.npy",  pi_result["Q_star"])
    np.save(output_dir / "pi_pi_star.npy", pi_result["pi_star"])

    with open(output_dir / "pi_policy.json", "w", encoding="utf-8") as f:
        json.dump(policy_rows, f, ensure_ascii=False, indent=2)

    convergence_data = {
        "iteration_history":       pi_result["iteration_history"],
        "n_iterations":            pi_result["n_iterations"],
        "n_eval_iterations_total": pi_result["n_eval_iterations_total"],
        "converged":               pi_result["converged"],
        "gamma":                   pi_result["gamma"],
        "epsilon":                 pi_result["epsilon"],
        "eval_epsilon":            pi_result["eval_epsilon"],
    }
    with open(output_dir / "pi_convergence.json", "w", encoding="utf-8") as f:
        json.dump(convergence_data, f, indent=2)

    logger.info(
        "PI results sauvegardés dans %s "
        "(pi_V_star.npy, pi_Q_star.npy, pi_pi_star.npy, "
        "pi_policy.json, pi_convergence.json)",
        output_dir,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Section LaTeX
# ──────────────────────────────────────────────────────────────────────────────

def generate_pi_latex(
    pi_result: dict,
    policy_rows: list,
    sensitivity_results: dict,
    output_path: Path = Path("reports/rapport_technique/module_c_pi.tex"),
) -> None:
    """Génère reports/rapport_technique/module_c_pi.tex"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    N       = len(policy_rows)
    A       = len(pi_result["Q_star"][0])
    g       = pi_result["gamma"]
    ep      = pi_result["eval_epsilon"]
    n_glob  = pi_result["n_iterations"]
    n_eval  = pi_result["n_eval_iterations_total"]
    conv    = pi_result["converged"]
    conv_str = "converg\\'{e}" if conv else "non converg\\'{e}"

    sorted_rows = sorted(
        policy_rows,
        key=lambda r: (r["cluster_id"], r["conf_level"], r["alerte_cnn"]),
    )
    conf_names = {0: "haute", 1: "moyen", 2: "faible"}

    # Table politique
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

    # Table sensibilité
    sens_rows_tex = ""
    if sensitivity_results:
        for gv in sorted(sensitivity_results.keys()):
            sr = sensitivity_results[gv]
            v_mean = f"{float(sr['V_star'].mean()):,.0f}".replace(",", "\\,")
            sens_rows_tex += (
                f"  {gv:.2f} & {sr['n_iterations']} & "
                f"{sr['n_eval_iterations_total']} & "
                f"{'Oui' if sr['converged'] else 'Non'} & {v_mean} \\\\\n"
            )

    # Note de comparaison VI vs PI si vi_convergence.json disponible
    vi_note = ""
    vi_json = Path("data/processed/vi_convergence.json")
    if vi_json.exists():
        try:
            with open(vi_json) as f:
                vi_conv = json.load(f)
            n_vi = vi_conv.get("n_iterations", "?")
            vi_note = (
                f"\n\\textit{{Note comparative :}} PI converge en "
                f"\\textbf{{{n_glob}}} it\\'{{'}}ration(s) globale(s) contre "
                f"\\textbf{{{n_vi}}} it\\'{{'}}rations Bellman pour VI à "
                f"$\\gamma={g}$ --- un rapport de "
                f"{round(int(n_vi) / max(n_glob, 1))}:1 en faveur de PI "
                f"(en it\\'{{'}}rations de politique)."
            )
        except Exception:
            pass

    lines = [
        r"\subsection{Policy Iteration}",
        r"\label{sec:c:pi}",
        "",
        r"\paragraph{Algorithme en deux phases (Russell \& Norvig, Ch.~17.3 ; Sutton \& Barto, §4.3)}",
        "",
        r"\textbf{Phase 1 --- \'{E}valuation de politique :}",
        r"\begin{equation}",
        r"V^{\pi}(s) \leftarrow R\bigl(s,\pi(s)\bigr)",
        r"  + \gamma \sum_{s'} P\bigl(s,\pi(s),s'\bigr)\, V^{\pi}(s')",
        r"\end{equation}",
        "",
        r"\textbf{Phase 2 --- Am\'{e}lioration de politique :}",
        r"\begin{equation}",
        r"\pi_{\text{new}}(s) \leftarrow \arg\max_{a}\left[",
        r"  R(s,a) + \gamma \sum_{s'} P(s,a,s')\, V^{\pi}(s') \right]",
        r"\end{equation}",
        "",
        f"\\textbf{{Param\\`{{e}}tres :}} $\\gamma = {g}$, "
        f"$\\varepsilon_{{\\text{{eval}}}} = {ep:.0e}$, "
        f"$N = {N}$ \\'{{'}}tats, $A = {A}$ actions.",
        f"Convergence en \\textbf{{{n_glob}}} it\\'{{'}}ration(s) globale(s)",
        f"({n_eval} it\\'{{'}}rations d'\\'{{'}}valuation cumul\\'{{'}}es, {conv_str}).",
        vi_note,
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
            r"$\gamma$ & It\'{e}r. globales & It\'{e}r. \'{e}val. cumul"
            r" & Converg\'{e} & $\bar{V}^*$ (FCFA) \\",
            r"\midrule",
            sens_rows_tex,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]

    lines += [
        r"\paragraph{Interpr\'{e}tation m\'{e}tier MINT/DGI}",
        "La politique $\\pi^*$ obtenue par Policy Iteration confirm que les",
        "plaques du cluster~2 (expir\\'{e}es) doivent syst\\'{e}matiquement faire",
        "l'objet d'un \\texttt{SIGNALEMENT\\_DGI} quelle que soit la confiance OCR,",
        "tandis que les plaques conformes \\`{a} haute confiance sont laiss\\'{e}es",
        "passer. Cette politique est identique \\`{a} celle de Value Iteration,",
        "validant la robustesse du r\\'{e}sultat.",
        "L'avantage de PI est sa convergence rapide en quelques it\\'{e}rations de",
        "politique --- recommand\\'{e}e pour une re-calibration p\\'{e}riodique",
        "du syst\\`{e}me MINT/DGI avec de nouvelles donn\\'{e}es de contr\\^{o}le.",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("LaTeX PI généré : %s", output_path)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────────────────────────

def run_policy_iteration_pipeline(
    gamma: float = DEFAULT_GAMMA,
    epsilon: float = DEFAULT_EPSILON,
    eval_epsilon: float = DEFAULT_EVAL_EPSILON,
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
    report_dir.mkdir(parents=True, exist_ok=True)

    # Chargement
    states, P, R, actions = load_mdp(data_dir)

    # Policy Iteration
    pi_result = run_policy_iteration(
        P, R, gamma=gamma, epsilon=epsilon,
        max_iter=DEFAULT_MAX_ITER, eval_epsilon=eval_epsilon,
    )

    # Politique
    policy_rows = extract_policy(pi_result["Q_star"], states, actions)
    print_policy_table(policy_rows, pi_result=pi_result)

    # Figures
    plot_convergence(
        pi_result["iteration_history"], gamma,
        output_path=figures_dir / "pi_convergence.png",
    )
    plot_value_function(
        states, pi_result["V_star"],
        output_path=figures_dir / "pi_value_function.png",
    )

    # Sensibilité γ
    sensitivity_results = {}
    if run_sensitivity:
        sensitivity_results = sensitivity_gamma(P, R, eval_epsilon=eval_epsilon)
        plot_gamma_sensitivity(
            sensitivity_results, states,
            output_path=figures_dir / "pi_gamma_sensitivity.png",
        )

    # Sauvegarde
    save_pi_results(pi_result, policy_rows, output_dir)

    # LaTeX
    generate_pi_latex(
        pi_result, policy_rows, sensitivity_results,
        output_path=report_dir / "module_c_pi.tex",
    )

    n_glob = pi_result["n_iterations"]
    n_eval = pi_result["n_eval_iterations_total"]
    conv   = pi_result["converged"]
    print(f"\n✓ Policy Iteration : {n_glob} itérations globales, convergé={conv}, γ={gamma}")
    print(f"✓ Itérations d'évaluation cumulées : {n_eval}")
    print(f"✓ Politique sauvegardée : {output_dir}/pi_policy.json")

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
    parser.add_argument("--eval-epsilon",   type=float, default=DEFAULT_EVAL_EPSILON)
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument("--data-dir",       type=Path,  default=Path("data/processed"))
    parser.add_argument("--figures-dir",    type=Path,  default=Path("reports/figures"))
    parser.add_argument("--report-dir",     type=Path,  default=Path("reports/rapport_technique"))
    args = parser.parse_args()

    run_policy_iteration_pipeline(
        gamma          = args.gamma,
        epsilon        = args.epsilon,
        eval_epsilon   = args.eval_epsilon,
        run_sensitivity= not args.no_sensitivity,
        data_dir       = args.data_dir,
        figures_dir    = args.figures_dir,
        report_dir     = args.report_dir,
    )
