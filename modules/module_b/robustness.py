"""
Module B — Étape 8 : Robustesse du clustering K-Means (multi-graines + sensibilité n_init)
PlateVision / MINT-DGI Cameroun — §4.2 + §5 cahier des charges

Contexte soutenance §5 : le jury peut modifier random_state et n_init en direct.
Ce module quantifie à l'avance leur impact sur la stabilité du clustering.

Transition B→C §4 : l'instabilité mesurée ici justifie le Module C (MDP)
pour gérer l'incertitude résiduelle des assignations de clusters.
"""

import json
import logging
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT + NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def load_scaled_embeddings(
    data_dir: Path = Path("data/processed"),
    models_dir: Path = Path("models"),
) -> np.ndarray:
    """
    Charge embeddings.npy et applique le scaler kmeans_scaler.pkl.
    Retourne embeddings_scaled (N, 256).
    """
    data_dir   = Path(data_dir)
    models_dir = Path(models_dir)

    emb_path    = data_dir / "embeddings.npy"
    scaler_path = models_dir / "kmeans_scaler.pkl"

    if not emb_path.exists():
        raise RuntimeError(
            "embeddings.npy absent. Exécute d'abord : python main.py --module B1"
        )
    if not scaler_path.exists():
        raise RuntimeError(
            "kmeans_scaler.pkl absent. Exécute d'abord : python main.py --module B4"
        )

    embeddings = np.load(emb_path).astype(np.float32)
    scaler     = joblib.load(scaler_path)
    embeddings_scaled = scaler.transform(embeddings).astype(np.float32)

    logger.info("Embeddings chargés et normalisés : %s", embeddings_scaled.shape)
    return embeddings_scaled


# ══════════════════════════════════════════════════════════════════════════════
# 2. K-MEANS MULTI-GRAINES
# ══════════════════════════════════════════════════════════════════════════════

def run_multiple_seeds(
    embeddings_scaled: np.ndarray,
    k: int,
    seeds: list[int],
    n_init: int = 10,
    max_iter: int = 300,
) -> list[dict]:
    """
    Lance K-Means pour chaque graine dans seeds.
    Retourne une liste de dicts (un par seed) avec inertie, silhouette, labels.
    """
    results = []
    for seed in tqdm(seeds, desc=f"K-Means k={k} — test graines"):
        km = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=n_init,
            max_iter=max_iter,
        )
        km.fit(embeddings_scaled)
        sil = float(silhouette_score(embeddings_scaled, km.labels_,
                                     sample_size=min(2000, len(embeddings_scaled)),
                                     random_state=seed))
        results.append({
            "seed":      int(seed),
            "n_init":    int(n_init),
            "inertia":   float(km.inertia_),
            "n_iter":    int(km.n_iter_),
            "labels":    km.labels_.copy(),
            "silhouette": sil,
        })
        logger.info("Seed %3d | inertia=%.2f | sil=%.4f | n_iter=%d",
                    seed, km.inertia_, sil, km.n_iter_)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. MATRICE ARI PAIRWISE
# ══════════════════════════════════════════════════════════════════════════════

def compute_pairwise_ari(results: list[dict]) -> np.ndarray:
    """
    Calcule la matrice ARI (n_seeds × n_seeds) entre toutes les paires de runs.
    ARI = 1.0 → runs identiques | ARI > 0.8 → stable | ARI < 0.5 → instable.
    """
    n = len(results)
    ari_matrix = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            ari = float(adjusted_rand_score(results[i]["labels"],
                                            results[j]["labels"]))
            ari_matrix[i, j] = ari
            ari_matrix[j, i] = ari
    return ari_matrix


# ══════════════════════════════════════════════════════════════════════════════
# 4. SENSIBILITÉ n_init
# ══════════════════════════════════════════════════════════════════════════════

def run_n_init_sensitivity(
    embeddings_scaled: np.ndarray,
    k: int,
    n_init_values: list[int],
    random_state: int = 42,
) -> list[dict]:
    """
    Teste l'impact de n_init sur l'inertie et le score silhouette.
    Retourne une liste de dicts (un par n_init).
    """
    results = []
    for n_init in tqdm(n_init_values, desc=f"Sensibilité n_init (k={k})"):
        km = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
            max_iter=300,
        )
        km.fit(embeddings_scaled)
        sil = float(silhouette_score(embeddings_scaled, km.labels_,
                                     sample_size=min(2000, len(embeddings_scaled)),
                                     random_state=random_state))
        results.append({
            "n_init":    int(n_init),
            "inertia":   float(km.inertia_),
            "silhouette": sil,
            "n_iter":    int(km.n_iter_),
        })
        logger.info("n_init=%2d | inertia=%.2f | sil=%.4f", n_init, km.inertia_, sil)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. FIGURE ROBUSTESSE
# ══════════════════════════════════════════════════════════════════════════════

def plot_robustness_results(
    seed_results: list[dict],
    ari_matrix: np.ndarray,
    n_init_results: list[dict],
    seeds: list[int],
    k: int,
    out_path: Path,
) -> None:
    """
    Figure 3 sous-graphes :
      Gauche  — Inertie par graine
      Centre  — Heatmap ARI pairwise
      Droite  — Sensibilité n_init (double axe Y)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_seeds   = len(seed_results)
    n_init_ref = seed_results[0]["n_init"] if seed_results else 10

    inerties  = [r["inertia"] for r in seed_results]
    mean_iner = float(np.mean(inerties))
    std_iner  = float(np.std(inerties))

    # Moyenne ARI hors-diagonale
    mask = ~np.eye(n_seeds, dtype=bool)
    mean_ari = float(ari_matrix[mask].mean()) if n_seeds > 1 else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    # ── Gauche : inertie par graine ───────────────────────────────────────────
    ax0 = axes[0]
    x   = np.arange(n_seeds)
    cmap_bar = plt.colormaps["tab10"]
    bars = ax0.bar(x, inerties, color=[cmap_bar(i % 10) for i in range(n_seeds)],
                   edgecolor="black", linewidth=0.5, width=0.7)
    ax0.axhline(mean_iner, color="red", linestyle="--", linewidth=1.2,
                label=f"Moyenne = {mean_iner:.1f}")
    ax0.set_xticks(x)
    ax0.set_xticklabels([str(s) for s in seeds], fontsize=8)
    ax0.set_xlabel("random_state (graine)", fontsize=9)
    ax0.set_ylabel("Inertie", fontsize=9)
    ax0.set_title(
        f"Inertie K-Means par graine aléatoire\n"
        f"(k={k}, n_init={n_init_ref})  σ={std_iner:.1f}",
        fontsize=9,
    )
    ax0.legend(fontsize=8)
    for bar, val in zip(bars, inerties):
        ax0.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(inerties) * 0.005,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=7)

    # ── Centre : heatmap ARI ──────────────────────────────────────────────────
    ax1 = axes[1]

    # Fond de couleur selon stabilité
    if mean_ari > 0.8:
        bg_color = "#e8f5e9"
    elif mean_ari >= 0.5:
        bg_color = "#fff3e0"
    else:
        bg_color = "#ffebee"
    ax1.set_facecolor(bg_color)

    im = ax1.imshow(ari_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax1, label="ARI (0=aléatoire, 1=identique)")

    ax1.set_xticks(range(n_seeds))
    ax1.set_yticks(range(n_seeds))
    ax1.set_xticklabels([str(i) for i in range(n_seeds)], fontsize=8)
    ax1.set_yticklabels([str(i) for i in range(n_seeds)], fontsize=8)
    ax1.set_xlabel("Run (graine)", fontsize=9)
    ax1.set_ylabel("Run (graine)", fontsize=9)
    ax1.set_title(f"Stabilité — ARI entre runs\nARI_moy={mean_ari:.3f}", fontsize=9)

    for i in range(n_seeds):
        for j in range(n_seeds):
            ax1.text(j, i, f"{ari_matrix[i, j]:.2f}",
                     ha="center", va="center", fontsize=7,
                     color="black" if ari_matrix[i, j] < 0.8 else "white")

    # ── Droite : sensibilité n_init ───────────────────────────────────────────
    ax2 = axes[2]
    n_init_vals = [r["n_init"]    for r in n_init_results]
    inert_vals  = [r["inertia"]   for r in n_init_results]
    sil_vals    = [r["silhouette"] for r in n_init_results]

    ax2.plot(n_init_vals, inert_vals, color="steelblue", marker="o",
             linewidth=1.5, markersize=5, label="Inertie")
    ax2.set_xlabel("n_init", fontsize=9)
    ax2.set_ylabel("Inertie", fontsize=9, color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")

    ax2b = ax2.twinx()
    ax2b.plot(n_init_vals, sil_vals, color="green", marker="o",
              linewidth=1.5, markersize=5, label="Silhouette")
    ax2b.set_ylabel("Score silhouette", fontsize=9, color="green")
    ax2b.tick_params(axis="y", labelcolor="green")

    # Légende combinée
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    random_state_ref = 42
    ax2.set_title(
        f"Sensibilité à n_init\n(random_state={random_state_ref})",
        fontsize=9,
    )

    fig.suptitle(
        "Module B — Robustesse du clustering K-Means (§4.2 PlateVision)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure robustesse sauvegardée : %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# 6. INTERPRÉTATION STABILITÉ
# ══════════════════════════════════════════════════════════════════════════════

def interpret_stability(
    ari_matrix: np.ndarray,
    seed_results: list[dict],
    n_init_results: list[dict],
    k: int,
) -> dict:
    """
    Produit l'interprétation automatique de la stabilité du clustering.
    Retourne dict avec verdict, mean_ari, std_inertia, cv_inertia, texte, n_seeds.
    """
    n_seeds = len(seed_results)
    mask    = ~np.eye(n_seeds, dtype=bool)
    mean_ari    = float(ari_matrix[mask].mean()) if n_seeds > 1 else 1.0
    std_inertia = float(np.std([r["inertia"] for r in seed_results]))
    cv_inertia  = std_inertia / float(np.mean([r["inertia"] for r in seed_results]))

    if mean_ari > 0.8 and cv_inertia < 0.01:
        verdict = "STABLE"
        texte = (
            f"Le clustering k={k} est stable : ARI moyen={mean_ari:.3f} "
            f"entre {n_seeds} runs. L'espace d'embeddings CNN "
            f"présente une structure géométrique robuste — les groupes "
            f"de plaques MINT/DGI sont bien séparés."
        )
    elif mean_ari >= 0.5:
        verdict = "MODÉRÉMENT STABLE"
        texte = (
            f"Le clustering k={k} est modérément stable (ARI={mean_ari:.3f}). "
            f"Les groupes sont partiellement reproductibles mais présentent "
            f"une sensibilité à l'initialisation. Recommandation : utiliser "
            f"n_init≥20 en production."
        )
    else:
        verdict = "INSTABLE — LIMITE À DOCUMENTER"
        texte = (
            f"Le clustering k={k} est instable (ARI={mean_ari:.3f}). "
            f"Cette instabilité est une limite importante à documenter "
            f"dans le rapport (§L3) et à évoquer en soutenance. Elle "
            f"renforce la nécessité du Module C (MDP) pour gérer "
            f"l'incertitude décisionnelle malgré l'instabilité du "
            f"clustering sous-jacent."
        )

    logger.info("Verdict stabilité : %s (ARI_moy=%.4f, CV_inertie=%.5f)",
                verdict, mean_ari, cv_inertia)
    return {
        "verdict":     verdict,
        "mean_ari":    mean_ari,
        "std_inertia": std_inertia,
        "cv_inertia":  cv_inertia,
        "texte":       texte,
        "n_seeds":     n_seeds,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. RAPPORT LaTeX ROBUSTESSE
# ══════════════════════════════════════════════════════════════════════════════

def generate_robustness_report_section(
    interpretation: dict,
    seed_results: list[dict],
    n_init_results: list[dict],
    k: int,
    out_dir: Path,
) -> None:
    """
    Génère et sauvegarde la sous-section LaTeX de robustesse.
    Fichier : out_dir/module_b_robustness.tex
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds_list   = [r["seed"]   for r in seed_results]
    n_init_list  = sorted({r["n_init"] for r in n_init_results})
    mean_ari     = interpretation["mean_ari"]
    verdict      = interpretation["verdict"]
    texte        = interpretation["texte"]

    # Optimal n_init : celui qui maximise le score silhouette
    optimal_n_init = max(n_init_results, key=lambda r: r["silhouette"])["n_init"]

    # ── Tableau seeds ─────────────────────────────────────────────────────────
    inerties   = [r["inertia"]    for r in seed_results]
    sils       = [r["silhouette"] for r in seed_results]
    iters_list = [r["n_iter"]     for r in seed_results]
    mean_i = np.mean(inerties); std_i = np.std(inerties)
    mean_s = np.mean(sils);     std_s = np.std(sils)

    seed_rows = "\n".join(
        f"        {r['seed']} & {r['inertia']:.2f} & {r['silhouette']:.4f} & {r['n_iter']} \\\\"
        for r in seed_results
    )
    seed_summary = (
        f"        \\textbf{{Moy. $\\pm$ Écart-type}} & "
        f"${mean_i:.2f} \\pm {std_i:.2f}$ & "
        f"${mean_s:.4f} \\pm {std_s:.4f}$ & — \\\\"
    )

    # ── Tableau n_init ────────────────────────────────────────────────────────
    n_init_rows = "\n".join(
        f"        {r['n_init']} & {r['inertia']:.2f} & {r['silhouette']:.4f} \\\\"
        for r in n_init_results
    )

    latex = (
        r"\subsection{Robustesse et sensibilité du clustering}"
        "\n\n"
        r"\paragraph{Protocole}"
        "\n"
        f"K-Means a été relancé avec {len(seeds_list)} graines aléatoires différentes\n"
        f"({', '.join(str(s) for s in seeds_list)}) et $n_{{\\text{{init}}}} \\in$\n"
        f"\\{{{', '.join(str(n) for n in n_init_list)}\\}}.\n"
        "Pour chaque paire de runs, l'Adjusted Rand Index (ARI) mesure la similarité des\n"
        "assignations.\n\n"
        r"\paragraph{Résultats de stabilité}"
        "\n\n"
        r"\begin{table}[H]"
        "\n"
        r"    \centering"
        "\n"
        r"    \caption{K-Means multi-graines — inertie, silhouette et convergence}"
        "\n"
        r"    \label{tab:robustness_seeds}"
        "\n"
        r"    \small"
        "\n"
        r"    \begin{tabular}{c|r|r|r}"
        "\n"
        r"        \hline"
        "\n"
        r"        \textbf{Graine} & \textbf{Inertie} & \textbf{Score silhouette} & \textbf{Itérations} \\"
        "\n"
        r"        \hline"
        "\n"
        + seed_rows + "\n"
        r"        \hline"
        "\n"
        + seed_summary + "\n"
        r"        \hline"
        "\n"
        r"    \end{tabular}"
        "\n"
        r"\end{table}"
        "\n\n"
        r"\paragraph{Verdict}"
        "\n"
        r"\textit{" + texte.replace("&", r"\&") + "}\n\n"
        r"\paragraph{Recommandation $n_{\text{init}}$}"
        "\n\n"
        r"\begin{table}[H]"
        "\n"
        r"    \centering"
        "\n"
        r"    \caption{Sensibilité à $n_{\text{init}}$ — inertie et silhouette}"
        "\n"
        r"    \label{tab:robustness_ninit}"
        "\n"
        r"    \small"
        "\n"
        r"    \begin{tabular}{c|r|r}"
        "\n"
        r"        \hline"
        "\n"
        r"        \textbf{$n_{\text{init}}$} & \textbf{Inertie} & \textbf{Silhouette} \\"
        "\n"
        r"        \hline"
        "\n"
        + n_init_rows + "\n"
        r"        \hline"
        "\n"
        r"    \end{tabular}"
        "\n"
        r"\end{table}"
        "\n\n"
        f"Valeur recommandée : $n_{{\\text{{init}}}}={optimal_n_init}$ "
        f"(compromis qualité/temps — gain marginal au-delà).\n\n"
        r"\paragraph{Impact sur Module C}"
        "\n"
        f"L'instabilité résiduelle (ARI$={mean_ari:.3f}$) justifie l'approche\n"
        "MDP du Module C : les probabilités de transition $P(s'|s,a)$ absorbent\n"
        "l'incertitude du clustering — un véhicule mal classifié reste gérable\n"
        r"si la politique optimale $\pi^*$ est robuste à cette incertitude (§4.3)."
        "\n\n"
        r"\begin{figure}[H]"
        "\n"
        r"    \centering"
        "\n"
        r"    \includegraphics[width=\textwidth]{figures/robustness_analysis.png}"
        "\n"
        r"    \caption{Robustesse du clustering K-Means — inertie multi-graines, matrice ARI et sensibilité $n_{\text{init}}$}"
        "\n"
        r"    \label{fig:robustness_analysis}"
        "\n"
        r"\end{figure}"
        "\n"
    )

    out_path = out_dir / "module_b_robustness.tex"
    out_path.write_text(latex, encoding="utf-8")
    logger.info("Rapport LaTeX robustesse écrit : %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# 8. PIPELINE COMPLET
# ══════════════════════════════════════════════════════════════════════════════

def run_robustness_pipeline(
    data_dir: Path = Path("data/processed"),
    models_dir: Path = Path("models"),
    figures_dir: Path = Path("reports/rapport_technique/figures"),
    report_dir: Path = Path("reports/rapport_technique"),
    k: "int | None" = None,
    seeds: "list[int] | None" = None,
    n_init_values: "list[int] | None" = None,
    random_state: int = 42,
) -> dict:
    """
    Orchestre l'analyse complète de robustesse du clustering Module B.
    """
    data_dir    = Path(data_dir)
    models_dir  = Path(models_dir)
    figures_dir = Path(figures_dir)
    report_dir  = Path(report_dir)

    if seeds is None:
        seeds = [0, 1, 2, 7, 13, 21, 42]
    if n_init_values is None:
        n_init_values = [1, 5, 10, 20, 50]

    # ── Résolution de k ───────────────────────────────────────────────────────
    if k is None:
        mapping_path = data_dir / "cluster_mapping.json"
        if not mapping_path.exists():
            raise RuntimeError(
                "cluster_mapping.json absent et --k non spécifié. "
                "Exécute d'abord : python -m modules.module_b.interpret_clusters"
            )
        with mapping_path.open(encoding="utf-8") as f:
            k = int(json.load(f)["k"])
        logger.info("k=%d lu depuis cluster_mapping.json", k)

    # ── 1. Chargement ─────────────────────────────────────────────────────────
    embeddings_scaled = load_scaled_embeddings(data_dir, models_dir)

    # ── 2. Multi-graines ──────────────────────────────────────────────────────
    seed_results = run_multiple_seeds(embeddings_scaled, k, seeds, n_init=10)

    # ── 3. Matrice ARI ────────────────────────────────────────────────────────
    ari_matrix = compute_pairwise_ari(seed_results)

    # ── 4. Sensibilité n_init ─────────────────────────────────────────────────
    n_init_results = run_n_init_sensitivity(
        embeddings_scaled, k, n_init_values, random_state=random_state
    )

    # ── 5. Interprétation ─────────────────────────────────────────────────────
    interpretation = interpret_stability(ari_matrix, seed_results, n_init_results, k)

    # ── 6. Figure ─────────────────────────────────────────────────────────────
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "robustness_analysis.png"
    plot_robustness_results(seed_results, ari_matrix, n_init_results, seeds, k, fig_path)

    # ── 7. Rapport LaTeX ──────────────────────────────────────────────────────
    generate_robustness_report_section(
        interpretation, seed_results, n_init_results, k, report_dir
    )

    # ── Optimal n_init ────────────────────────────────────────────────────────
    optimal_n_init = max(n_init_results, key=lambda r: r["silhouette"])["n_init"]

    mean_ari = interpretation["mean_ari"]
    verdict  = interpretation["verdict"]
    n_seeds  = len(seeds)

    print(f"\n=== Module B — Étape 8 terminée ===")
    print(f"K-Means testé sur {n_seeds} graines : {seeds}")
    print(f"ARI moyen       : {mean_ari:.4f}")
    print(f"Verdict         : {verdict}")
    print(f"n_init optimal  : {optimal_n_init}")
    print()
    print("Fichiers générés :")
    print(f"  {fig_path}")
    print(f"  {report_dir}/module_b_robustness.tex")
    print()
    print("Ce résultat sera invoqué en soutenance pour justifier le Module C :")
    print(f"  'Le clustering révèle des groupes mais son instabilité (ARI={mean_ari:.3f})")
    print(f"   justifie un MDP pour décider sous incertitude (§4.3).'")

    return {
        "k":               k,
        "seeds":           seeds,
        "seed_results":    seed_results,
        "ari_matrix":      ari_matrix,
        "n_init_results":  n_init_results,
        "interpretation":  interpretation,
        "fig_path":        fig_path,
        "optimal_n_init":  optimal_n_init,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI — §5 soutenance : jury peut modifier seeds et n_init en direct
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Module B Étape 8 — Robustesse clustering K-Means"
    )
    parser.add_argument("--data-dir",      type=Path, default=Path("data/processed"))
    parser.add_argument("--models-dir",    type=Path, default=Path("models"))
    parser.add_argument("--figures-dir",   type=Path,
                        default=Path("reports/rapport_technique/figures"))
    parser.add_argument("--report-dir",    type=Path,
                        default=Path("reports/rapport_technique"))
    parser.add_argument("--k",             type=int,  default=None)
    parser.add_argument("--seeds",         nargs="+", type=int,
                        default=[0, 1, 2, 7, 13, 21, 42],
                        help="Graines aléatoires à tester")
    parser.add_argument("--n-init-values", nargs="+", type=int,
                        default=[1, 5, 10, 20, 50])
    parser.add_argument("--random-state",  type=int,  default=42)
    args = parser.parse_args()

    run_robustness_pipeline(
        data_dir      = args.data_dir,
        models_dir    = args.models_dir,
        figures_dir   = args.figures_dir,
        report_dir    = args.report_dir,
        k             = args.k,
        seeds         = args.seeds,
        n_init_values = args.n_init_values,
        random_state  = args.random_state,
    )
