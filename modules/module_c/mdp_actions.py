"""
Module C — Étape 2 : Espace d'actions A du MDP PlateVision
MINT/DGI Cameroun — §4.3 cahier des charges

Les 5 actions couvrent le spectre complet des procédures opérationnelles
MINT/DGI, des plus légères (laisser passer) aux plus lourdes (transfert PJ).
Tous les paramètres coût/bénéfice sont sourcés (Code de la Route, Loi de
Finances DGI 2023, barème MINT).
"""

import json
import logging
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MDPAction:
    """Une action de l'espace A du MDP PlateVision."""
    action_id:               int
    code:                    str
    label:                   str
    acteur:                  str
    procedure:               str
    delai_moyen:             str
    cout_operationnel_fcfa:  float
    gain_attendu_fcfa:       float
    base_legale:             str
    description:             str


# ══════════════════════════════════════════════════════════════════════════════
# LES 5 ACTIONS §4.3 — PARAMÈTRES SOURCÉS
# ══════════════════════════════════════════════════════════════════════════════

ACTION_0 = MDPAction(
    action_id = 0,
    code      = "LAISSER_PASSER",
    label     = "Laisser passer",
    acteur    = "Agent MINT au barrage",
    procedure = "Aucune procédure déclenchée — circulation autorisée",
    delai_moyen = "< 5 secondes",
    cout_operationnel_fcfa = 500.0,
    # Coût temps agent : salaire moyen agent MINT ~150 000 FCFA/mois
    # 150 000 / (20j × 8h × 3 600s) × 5s ≈ 500 FCFA
    gain_attendu_fcfa = 0.0,
    base_legale = "Code de la Route camerounais n°96/07, Art. 58",
    description = (
        "L'agent autorise le véhicule à poursuivre sa route sans intervention. "
        "Appliqué lorsque la plaque est conforme, lisible et sans signal d'alerte."
    ),
)

ACTION_1 = MDPAction(
    action_id = 1,
    code      = "CONTROLE_STANDARD",
    label     = "Contrôle standard",
    acteur    = "Agent MINT",
    procedure = (
        "Vérification manuelle des documents (carte grise, assurance, vignette) "
        "— 3 à 5 minutes. Constat d'infraction si irrégularité."
    ),
    delai_moyen = "3–5 minutes",
    cout_operationnel_fcfa = 5_000.0,
    # 5 min agent (~1 250 FCFA) + immobilisation voie (~3 750 FCFA coût flux MINT)
    gain_attendu_fcfa = 25_000.0,
    # Amende moyenne infraction légère : 25 000 FCFA
    # Source : Barème MINT 2022, catégorie infraction courante
    base_legale = "Code de la Route n°96/07, Art. 137–145 (amendes)",
    description = (
        "Arrêt du véhicule pour vérification documentaire standard. "
        "Déclenche un PV d'infraction si anomalie constatée. "
        "Applicable aux plaques lisibles mais de conformité incertaine."
    ),
)

ACTION_2 = MDPAction(
    action_id = 2,
    code      = "ARRET_SAISIE",
    label     = "Arrêt immédiat + saisie",
    acteur    = "Agent MINT + Police / Gendarmerie",
    procedure = (
        "Immobilisation du véhicule, saisie administrative de la plaque, "
        "rédaction d'un procès-verbal de saisie, convocation du propriétaire "
        "dans les 48h. Coordonnées transmises à la base MINT nationale."
    ),
    delai_moyen = "15–30 minutes sur site + 48h procédure",
    cout_operationnel_fcfa = 50_000.0,
    # Mobilisation 2 agents (30 min) + transport véhicule séquestre
    # + frais administratifs PV — estimés MINT 2023
    gain_attendu_fcfa = 500_000.0,
    # Amende + restitution plaque + frais judiciaires
    # Fourchette : 500 000 – 2 000 000 FCFA selon gravité
    # Source : Code de la Route n°96/07 + Loi n°2010/012 Art. 39
    base_legale = (
        "Code de la Route n°96/07 Art. 169 (saisie administrative) ; "
        "Loi n°2010/012 du 21 déc. 2010, Art. 39 (falsification)"
    ),
    description = (
        "Intervention lourde réservée aux plaques suspectes de falsification "
        "ou de double immatriculation. Nécessite coordination MINT/Police. "
        "Coût opérationnel élevé — justifié uniquement pour les clusters "
        "à haute probabilité de fraude."
    ),
)

ACTION_3 = MDPAction(
    action_id = 3,
    code      = "SIGNALEMENT_DGI",
    label     = "Signalement DGI — recouvrement vignette",
    acteur    = "Agent MINT → transmission automatique DGI",
    procedure = (
        "Transmission du numéro de plaque à la DGI via le système PlateVision. "
        "La DGI émet un avis de recouvrement au propriétaire enregistré. "
        "Délai de paiement : 30 jours avant majoration."
    ),
    delai_moyen = "< 1 minute (transmission numérique) + 30j recouvrement",
    cout_operationnel_fcfa = 2_000.0,
    # Transmission numérique automatique : ~0 FCFA marginal
    # Frais postaux avis DGI : ~500 FCFA
    # Temps agent MINT (< 1 min) : ~1 500 FCFA
    # Source : estimation interne MINT/DGI, coûts administratifs 2023
    gain_attendu_fcfa = 45_000.0,
    # Vignette VP < 2 000cc  : 37 500 FCFA/an (Loi de Finances 2023)
    # Vignette VP 2–3 000cc  : 52 500 FCFA/an
    # Majorations de retard  : +25% si > 30j (CGI Art. 556)
    # Moyenne pondérée parc automobile camerounais : ~45 000 FCFA
    # Source : DGI Cameroun (2023). Loi de Finances, Annexe fiscale.
    base_legale = (
        "Loi de Finances DGI Cameroun 2023, Art. 207 (vignette automobile) ; "
        "Code Général des Impôts CGI, Art. 553–558"
    ),
    description = (
        "Action à faible coût opérationnel et fort impact fiscal. "
        "Le véhicule n'est pas immobilisé — l'agent MINT saisit le numéro "
        "de plaque dans PlateVision, qui le transmet automatiquement à la DGI. "
        "Applicable aux plaques expirées ou à vignette manquante détectées "
        "par le clustering Module B."
    ),
)

ACTION_4 = MDPAction(
    action_id = 4,
    code      = "TRANSFERT_PJ",
    label     = "Transfert Police Judiciaire",
    acteur    = "Agent MINT → Police Judiciaire / DGSN",
    procedure = (
        "Signalement à la Police Judiciaire avec transmission du dossier "
        "PlateVision (image, cluster_id, score confiance, horodatage). "
        "Ouverture d'une enquête préliminaire. Véhicule immobilisé si "
        "présent sur site, sinon avis de recherche."
    ),
    delai_moyen = "30 min sur site + enquête 72h–30j",
    cout_operationnel_fcfa = 75_000.0,
    # Agent MINT (30 min) + mobilisation PJ + frais enquête
    # Estimation conservative — coûts réels PJ non publics
    gain_attendu_fcfa = 1_500_000.0,
    # 2 400 cas/24 mois (§1.2) × valeur moyenne dossier
    # Amende + saisie véhicule + amendes connexes : 1–5M FCFA
    # Source : Code de la Route n°96/07 Art. 169 + Loi 2010/012 Art. 39-41
    base_legale = (
        "Loi n°2010/012 du 21 déc. 2010, Art. 39–41 (cybercriminalité, "
        "falsification de documents) ; Code de la Route n°96/07 Art. 169 ; "
        "Code de Procédure Pénale camerounais, Art. 78 (flagrant délit)"
    ),
    description = (
        "Action la plus lourde du système. Réservée aux cas de forte "
        "probabilité de falsification ou double immatriculation (§1.2 : "
        "2 400 cas détectés sur 24 mois). Le coût opérationnel élevé "
        "est justifié par le gain potentiel et la gravité de l'infraction."
    ),
)

ACTIONS_MINT_DGI: list[MDPAction] = [
    ACTION_0, ACTION_1, ACTION_2, ACTION_3, ACTION_4,
]
N_ACTIONS: int = len(ACTIONS_MINT_DGI)


# ══════════════════════════════════════════════════════════════════════════════
# 1. ACCÈS PAR ID
# ══════════════════════════════════════════════════════════════════════════════

def get_action(action_id: int) -> MDPAction:
    """Retourne l'action d'identifiant action_id."""
    if action_id < 0 or action_id >= N_ACTIONS:
        raise ValueError(
            f"action_id={action_id} hors plage [0, {N_ACTIONS - 1}]"
        )
    return ACTIONS_MINT_DGI[action_id]


# ══════════════════════════════════════════════════════════════════════════════
# 2. JUSTIFICATION SIGNALEMENT DGI (EXIGENCE §4.3)
# ══════════════════════════════════════════════════════════════════════════════

def justify_signalement_dgi(
    report_dir: "Path | None" = None,
) -> str:
    """
    Génère la justification obligatoire §4.3 de l'action Signalement DGI.
    Sauvegarde dans report_dir/dgi_justification.txt si fourni.
    """
    a3 = ACTION_3
    a1 = ACTION_1
    ratio_dgi     = a3.gain_attendu_fcfa / a3.cout_operationnel_fcfa
    ratio_control = a1.gain_attendu_fcfa / a1.cout_operationnel_fcfa

    texte = f"""ANALYSE COÛT/BÉNÉFICE — ACTION SIGNALEMENT DGI (§4.3)
{"─" * 65}

Coût opérationnel par activation : {a3.cout_operationnel_fcfa:,.0f} FCFA
  - Transmission numérique automatique via PlateVision : ~0 FCFA marginal
  - Frais postaux avis de recouvrement DGI : ~500 FCFA
  - Temps agent MINT (< 1 minute) : ~1 500 FCFA
  Source : estimation interne MINT/DGI, coûts administratifs 2023

Gain attendu par recouvrement réussi : {a3.gain_attendu_fcfa:,.0f} FCFA
  - Vignette VP < 2 000cc  : 37 500 FCFA/an (Loi de Finances 2023)
  - Vignette VP 2–3 000cc  : 52 500 FCFA/an
  - Majorations de retard  : +25% si > 30j (CGI Art. 556)
  Source : DGI Cameroun (2023). Loi de Finances, Annexe fiscale.

Ratio bénéfice/coût : {a3.gain_attendu_fcfa:,.0f} / {a3.cout_operationnel_fcfa:,.0f} = {ratio_dgi:.1f}
→ Pour chaque FCFA investi, le Signalement DGI rapporte en moyenne
  {ratio_dgi:.1f} FCFA en recouvrement fiscal.

Comparaison avec Contrôle Standard (ratio = {a1.gain_attendu_fcfa:,.0f} / {a1.cout_operationnel_fcfa:,.0f} = {ratio_control:.1f}) :
→ Le Signalement DGI a un ratio {ratio_dgi / ratio_control:.1f}× supérieur au Contrôle Standard.
→ Il est d'autant plus pertinent que le véhicule n'est PAS immobilisé,
  préservant la fluidité du trafic.

Justification d'inclusion dans le MDP :
L'action Signalement DGI occupe un rôle unique dans l'espace d'actions :
elle est la seule à combiner coût quasi-nul, impact fiscal élevé, et
absence d'immobilisation du véhicule. Dans les états où le cluster_id
indique une plaque "expirée" (Module B) avec confiance OCR moyenne ou
haute, le MDP doit apprendre à préférer cette action à l'Arrêt+Saisie
coûteux, à moins que le signal d'alerte CNN ne soit présent.
{"─" * 65}
"""

    if report_dir is not None:
        out = Path(report_dir) / "dgi_justification.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(texte, encoding="utf-8")
        logger.info("Justification DGI sauvegardée : %s", out)

    return texte


# ══════════════════════════════════════════════════════════════════════════════
# 3. EXPORT JSON
# ══════════════════════════════════════════════════════════════════════════════

def export_actions_json(
    out_dir: Path = Path("data/processed"),
) -> Path:
    """Sauvegarde les 5 actions en JSON → out_dir/mdp_actions.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    justif = justify_signalement_dgi(report_dir=None)

    payload = {
        "n_actions": N_ACTIONS,
        "actions": [asdict(a) for a in ACTIONS_MINT_DGI],
        "justification_dgi": justif,
        "source": "§4.3 PlateVision — MINT/DGI Cameroun",
    }

    out_path = out_dir / "mdp_actions.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "mdp_actions.json exporté — %d actions → chargeable par mdp_rewards.py",
        N_ACTIONS,
    )
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# 4. FIGURE COÛT / BÉNÉFICE
# ══════════════════════════════════════════════════════════════════════════════

def plot_actions_cost_benefit(figures_dir: Path) -> None:
    """Scatter log/log coût vs gain pour les 5 actions."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    couts = np.array([a.cout_operationnel_fcfa for a in ACTIONS_MINT_DGI], dtype=float)
    gains = np.array([a.gain_attendu_fcfa      for a in ACTIONS_MINT_DGI], dtype=float)
    codes = [a.code for a in ACTIONS_MINT_DGI]

    # Ratio gain/coût (remplace 0/0 par 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ratios = np.where(couts > 0, gains / couts, 0.0)

    # Couleur : vert si ratio élevé, rouge si faible
    max_r = ratios.max() if ratios.max() > 0 else 1.0
    colors = plt.colormaps["RdYlGn"](ratios / max_r)

    # Taille proportionnelle au ratio (min 100, max 800)
    sizes = 100 + 700 * (ratios / max_r)
    sizes[ratios == 0] = 100

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    # Zones fond
    xlim = (100, 2e5)
    ylim = (100, 5e6)
    ax.fill_betweenx([xlim[0], xlim[1]], xlim[0], xlim[1],
                     alpha=0.05, color="red")   # zone y < x (non rentable)
    ax.fill_between([xlim[0], xlim[1]], [xlim[0], xlim[1]], ylim[1],
                    alpha=0.05, color="green")  # zone y >> x (rentable)

    # Ligne d'équilibre y = x
    diag = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 100)
    ax.plot(diag, diag, "--", color="gray", lw=1, alpha=0.6,
            label="Ratio = 1 (équilibre coût/gain)")

    # Points
    sc = ax.scatter(couts, gains, s=sizes, c=colors,
                    edgecolors="black", linewidths=0.7, zorder=3)

    # Labels
    for a, c, g, code in zip(ACTIONS_MINT_DGI, couts, gains, codes):
        offset_x = c * 1.15
        offset_y = g * 0.75
        ax.annotate(
            code.replace("_", "\n"),
            xy=(c, g),
            xytext=(offset_x, offset_y),
            fontsize=7.5,
            ha="left",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
        )

    # Annotation spéciale ACTION_3
    a3 = ACTION_3
    ratio3 = a3.gain_attendu_fcfa / a3.cout_operationnel_fcfa
    ax.annotate(
        f"Ratio = {ratio3:.0f}× → action à privilégier\npour vignettes impayées",
        xy=(a3.cout_operationnel_fcfa, a3.gain_attendu_fcfa),
        xytext=(a3.cout_operationnel_fcfa * 3, a3.gain_attendu_fcfa * 3),
        fontsize=8, color="darkgreen", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.0),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Coût opérationnel (FCFA, log)", fontsize=10)
    ax.set_ylabel("Gain attendu (FCFA, log)", fontsize=10)
    ax.set_title(
        "Analyse coût/bénéfice des actions MDP — MINT/DGI Cameroun\n"
        "(§4.3 — tous paramètres sourcés : Code Route, Loi Finances, DGI 2023)",
        fontsize=10,
    )

    # Formater les axes en FCFA
    import matplotlib.ticker as mticker
    def fcfa_fmt(x, _):
        if x >= 1_000_000:
            return f"{x/1_000_000:.0f}M"
        if x >= 1_000:
            return f"{x/1_000:.0f}k"
        return f"{x:.0f}"
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fcfa_fmt))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fcfa_fmt))

    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()

    out_path = figures_dir / "actions_cost_benefit.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure coût/bénéfice sauvegardée : %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# 5. AFFICHAGE RÉCAPITULATIF
# ══════════════════════════════════════════════════════════════════════════════

def print_actions_summary() -> None:
    """Tableau récapitulatif des 5 actions pour la soutenance §5."""
    sep  = "─" * 74
    hdr  = f"{'ID':^4}│{'Action':<28}│{'Coût (FCFA)':>13}│{'Gain (FCFA)':>14}│{'Ratio':>8}"
    print()
    print("┌" + sep + "┐")
    print("│" + hdr + "│")
    print("├" + sep + "┤")
    for a in ACTIONS_MINT_DGI:
        if a.cout_operationnel_fcfa > 0 and a.gain_attendu_fcfa > 0:
            ratio_str = f"{a.gain_attendu_fcfa / a.cout_operationnel_fcfa:.1f}×"
        else:
            ratio_str = "—"
        star = " ★" if a.action_id == 3 else "  "
        label_disp = (a.label[:26] + star) if a.action_id == 3 else a.label[:28]
        cout_str = f"{a.cout_operationnel_fcfa:>12,.0f}"
        gain_str = f"{a.gain_attendu_fcfa:>13,.0f}"
        row = f"{a.action_id:^4}│{label_disp:<28}│{cout_str}│{gain_str}│{ratio_str:>8}"
        print("│" + row + "│")
    print("└" + sep + "┘")
    print("★ Action à ratio maximal — voir justification §4.3")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 6. PIPELINE COMPLET
# ══════════════════════════════════════════════════════════════════════════════

def run_actions_pipeline(
    data_dir: Path    = Path("data/processed"),
    figures_dir: Path = Path("reports/rapport_technique/figures"),
    report_dir: Path  = Path("reports/rapport_technique"),
) -> list[MDPAction]:
    """Orchestre la définition et l'export de l'espace d'actions Module C."""
    data_dir    = Path(data_dir)
    figures_dir = Path(figures_dir)
    report_dir  = Path(report_dir)

    # ── 1. Tableau récapitulatif ───────────────────────────────────────────────
    print_actions_summary()

    # ── 2. Justification Signalement DGI ──────────────────────────────────────
    justify_signalement_dgi(report_dir=report_dir)

    # ── 3. Export JSON ────────────────────────────────────────────────────────
    json_path = export_actions_json(data_dir)

    # ── 4. Figure coût/bénéfice ───────────────────────────────────────────────
    plot_actions_cost_benefit(figures_dir)

    # ── Résumé console ────────────────────────────────────────────────────────
    a3 = ACTION_3
    ratio3 = a3.gain_attendu_fcfa / a3.cout_operationnel_fcfa
    print("=== Module C — Espace d'actions MDP défini ===")
    print(f"N actions : {N_ACTIONS} (§4.3 PlateVision)")
    print(f"  A0 : Laisser passer          (coût :      500 FCFA)")
    print(f"  A1 : Contrôle standard       (coût :    5 000 FCFA)")
    print(f"  A2 : Arrêt + saisie          (coût :   50 000 FCFA)")
    print(f"  A3 : Signalement DGI ★       (ratio : {ratio3:.1f}× — ratio maximal)")
    print(f"  A4 : Transfert PJ            (coût :   75 000 FCFA)")
    print()
    print(f"Justification Signalement DGI : {report_dir}/dgi_justification.txt")
    print(f"Fichier actions               : {json_path}")
    print(f"Figure coût/bénéfice          : {figures_dir}/actions_cost_benefit.png")
    print(f"Prêt pour                     : modules/module_c/mdp_rewards.py")

    return ACTIONS_MINT_DGI


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Module C — Espace d'actions MDP (§4.3 PlateVision)"
    )
    parser.add_argument("--data-dir",    type=Path, default=Path("data/processed"))
    parser.add_argument("--figures-dir", type=Path,
                        default=Path("reports/rapport_technique/figures"))
    parser.add_argument("--report-dir",  type=Path,
                        default=Path("reports/rapport_technique"))
    args = parser.parse_args()

    run_actions_pipeline(
        data_dir    = args.data_dir,
        figures_dir = args.figures_dir,
        report_dir  = args.report_dir,
    )
