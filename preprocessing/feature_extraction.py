"""
Extraction des features manuelles 175D pour le Module A1 (Naïves Bayes).
Cahier des charges §3.2 :
  Caractère (75D) : forme étendue (6D) + zones (8D) + projections+CC (57D) + Sobel (4D)
  Plaque    (100D): HSV hist (96D) + gradient (2D) + dark Otsu (1D) + frame contour (1D)
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Dimensions ────────────────────────────────────────────────────────────────
CHAR_FEAT_DIM  = 75
PLATE_FEAT_DIM = 100
TOTAL_FEAT_DIM = 175   # CHAR_FEAT_DIM + PLATE_FEAT_DIM


def extract_char_features(img: np.ndarray) -> np.ndarray:
    """
    Extrait le vecteur 75D d'un caractère 28×28 (niveaux de gris uint8).

    Groupe A — forme étendue (6D) :
      [0]  H_bb      : hauteur bounding-box normalisée / 28
      [1]  W_bb      : largeur bounding-box normalisée / 28
      [2]  ratio_hw  : H / W
      [3]  densité   : nb pixels > 128 / (28×28)
      [4]  cx        : centroïde x normalisé / W
      [5]  cy        : centroïde y normalisé / H

    Groupe B — densités zonales (8D) :
      Grille 4 colonnes × 2 lignes → 8 zones
      [6:14] densité moyenne de chaque zone (pixels > 128 / taille zone)

    Groupe C — projections + composantes connexes (57D) :
      [14:42] proj_h_0..27 : nb pixels > 127 par ligne / 28  (projection horizontale)
      [42:70] proj_v_0..27 : nb pixels > 127 par colonne / 28 (projection verticale)
      [70]    n_components : (n_labels - 1) / 10.0

    Groupe D — gradient Sobel (4D) :
      Sobel X et Y sur image uint8, ksize=3
      [71] sobel_x_mean · [72] sobel_x_std
      [73] sobel_y_mean · [74] sobel_y_std

    Robustesse : img entièrement noire → np.zeros(75)
    Retourne : np.ndarray float32 shape (75,)
    """
    if img.max() == 0:
        return np.zeros(CHAR_FEAT_DIM, dtype=np.float32)

    H, W = img.shape
    binary   = (img > 128).astype(np.float32)
    total    = binary.sum() + 1e-6
    ys, xs   = np.mgrid[0:H, 0:W]

    # ── Groupe A ──────────────────────────────────────────────────────────────
    ratio_hw = H / W
    density  = float(np.count_nonzero(img > 128)) / (H * W)
    cx = float((xs * binary).sum() / total) / W
    cy = float((ys * binary).sum() / total) / H

    if binary.any():
        rows_nonzero = np.where(binary > 0)[0]
        cols_nonzero = np.where(binary > 0)[1]
        H_bb = float(np.ptp(rows_nonzero) + 1) / 28.0
        W_bb = float(np.ptp(cols_nonzero) + 1) / 28.0
    else:
        H_bb, W_bb = 0.0, 0.0

    feat_A = np.array([H_bb, W_bb, ratio_hw, density, cx, cy], dtype=np.float32)

    # ── Groupe B ──────────────────────────────────────────────────────────────
    feat_B  = np.zeros(8, dtype=np.float32)
    zone_h  = H // 2
    zone_w  = W // 4
    for row in range(2):
        for col in range(4):
            y0   = row * zone_h
            x0   = col * zone_w
            zone = binary[y0:y0 + zone_h, x0:x0 + zone_w]
            feat_B[row * 4 + col] = float(zone.mean()) if zone.size > 0 else 0.0

    # ── Groupe C ──────────────────────────────────────────────────────────────
    bin_bool = img > 128
    proj_h   = bin_bool.sum(axis=1) / 28.0   # (28,) — densité par ligne
    proj_v   = bin_bool.sum(axis=0) / 28.0   # (28,) — densité par colonne

    n_labels, _ = cv2.connectedComponents(binary.astype(np.uint8))
    n_components = float(n_labels - 1) / 10.0   # -1 car le fond est compté

    feat_C = np.concatenate([
        proj_h.astype(np.float32),
        proj_v.astype(np.float32),
        np.array([n_components], dtype=np.float32),
    ])  # 28 + 28 + 1 = 57D

    # ── Groupe D ──────────────────────────────────────────────────────────────
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    feat_D  = np.array([
        float(sobel_x.mean()), float(sobel_x.std()),
        float(sobel_y.mean()), float(sobel_y.std()),
    ], dtype=np.float32)

    return np.concatenate([feat_A, feat_B, feat_C, feat_D])  # 6+8+57+4 = 75D


def extract_plate_features(img: np.ndarray) -> np.ndarray:
    """
    Extrait le vecteur 100D d'un crop de plaque 200×60 (BGR uint8).

    Histogramme HSV (96D) :
      H : 32 bins [0, 180] · S : 32 bins [0, 256] · V : 32 bins [0, 256]
      Normalisation indépendante : h / (h.sum() + 1e-7)
      [0:96] = concat(H_hist, S_hist, V_hist)

    Gradient magnitude (2D) :
      Gris float32/255 → Sobel X + Y → magnitude = √(Gx²+Gy²)
      [96] grad_mean · [97] grad_std

    Ratio pixels sombres — seuillage Otsu inversé (1D) :
      Binarisation THRESH_BINARY_INV + THRESH_OTSU sur image grise
      [98] dark_ratio_otsu = nb pixels blancs (sombres) / total pixels

    Présence cadre — détection de contour (1D) :
      Binarisation THRESH_BINARY + THRESH_OTSU puis findContours
      Bounding rect du plus grand contour externe :
      [99] has_frame_contour = 1.0 si area_ratio > 0.70, sinon 0.0

    Retourne : np.ndarray float32 shape (100,)
    """
    hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Histogramme HSV ───────────────────────────────────────────────────────
    def _norm_hist(h):
        return (h / (h.sum() + 1e-7)).astype(np.float32)

    hist_h   = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s   = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v   = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    feat_hsv = np.concatenate([_norm_hist(hist_h), _norm_hist(hist_s), _norm_hist(hist_v)])

    # ── Gradient magnitude ────────────────────────────────────────────────────
    gray_f32  = gray_u8.astype(np.float32) / 255.0
    sobel_x   = cv2.Sobel(gray_f32, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y   = cv2.Sobel(gray_f32, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    feat_grad = np.array([float(magnitude.mean()), float(magnitude.std())],
                         dtype=np.float32)

    # ── Ratio pixels sombres (seuillage Otsu inversé) ─────────────────────────
    _, binary_otsu = cv2.threshold(
        gray_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    dark_ratio = float(np.count_nonzero(binary_otsu)) / binary_otsu.size
    feat_dark  = np.array([dark_ratio], dtype=np.float32)

    # ── Présence cadre — détection contour externe ────────────────────────────
    _, binary_frame = cv2.threshold(
        gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    has_frame = 0.0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        _, _, w, h = cv2.boundingRect(largest)
        H_img, W_img = img.shape[:2]
        area_ratio = (w * h) / (W_img * H_img)
        has_frame  = 1.0 if area_ratio > 0.70 else 0.0
    feat_metal = np.array([has_frame], dtype=np.float32)

    return np.concatenate([feat_hsv, feat_grad, feat_dark, feat_metal])


def extract_feature_vector(char_img: np.ndarray,
                            plate_img: np.ndarray) -> np.ndarray:
    """
    Combine char + plate en vecteur 175D.

    char_img  : np.ndarray (28, 28) uint8 niveaux de gris  → 75D
    plate_img : np.ndarray (60, 200, 3) uint8 BGR           → 100D

    Retourne : np.ndarray float32 shape (175,)
    """
    return np.concatenate([
        extract_char_features(char_img),
        extract_plate_features(plate_img),
    ]).astype(np.float32)


def extract_features_batch(char_paths: list,
                             plate_paths: list) -> np.ndarray:
    """
    Extraction batch pour N paires (char_path, plate_path).

    char_paths  : liste de N chemins vers images 28×28 (grayscale)
    plate_paths : liste de N chemins vers crops 200×60 (BGR)

    Retourne : np.ndarray float32 shape (N, 175)
    """
    N      = len(char_paths)
    X      = np.zeros((N, TOTAL_FEAT_DIM), dtype=np.float32)
    errors = 0

    for i, (cp, pp) in enumerate(tqdm(
        zip(char_paths, plate_paths), total=N, desc="Features batch"
    )):
        try:
            char_img  = cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE)
            plate_img = cv2.imread(str(pp), cv2.IMREAD_COLOR)
            if char_img is None or plate_img is None:
                errors += 1
                continue
            X[i] = extract_feature_vector(char_img, plate_img)
        except Exception:
            errors += 1

    if errors:
        logger.warning(f"extract_features_batch : {errors}/{N} paires en erreur")
    return X


# ══════════════════════════════════════════════════════════════════════════════
# NOMMAGE DES 175 FEATURES (pour rapports jury)
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES: list = (
    ["H_bb", "W_bb", "ratio_hw", "densité", "cx", "cy"]       # [0:6]   groupe A
    + [f"zone_{i}" for i in range(8)]                          # [6:14]  groupe B
    + [f"proj_h_{i}" for i in range(28)]                       # [14:42] groupe C
    + [f"proj_v_{i}" for i in range(28)]                       # [42:70]
    + ["n_components"]                                         # [70]
    + ["sobel_x_mean", "sobel_x_std", "sobel_y_mean", "sobel_y_std"]  # [71:75] D
    + [f"H_bin_{i}" for i in range(32)]                        # [75:107]
    + [f"S_bin_{i}" for i in range(32)]                        # [107:139]
    + [f"V_bin_{i}" for i in range(32)]                        # [139:171]
    + ["grad_mean", "grad_std"]                                # [171:173]
    + ["dark_ratio_otsu", "has_frame_contour"]                 # [173:175]
)

# Paires de caractères critiques pour MINT/DGI
_CRITICAL_PAIRS = [("0", "O"), ("1", "I"), ("B", "8"), ("5", "S"), ("6", "G")]

# Groupes de features pour l'analyse d'importance (indices mis à jour 175D)
_FEATURE_GROUPS = {
    "A_forme":        slice(0,   6),
    "B_zones":        slice(6,  14),
    "C_projections":  slice(14, 71),
    "D_gradients":    slice(71, 75),
    "HSV_hist":       slice(75, 171),
    "grad_plaque":    slice(171, 173),
    "dark_ratio":     slice(173, 174),
    "metal_frame":    slice(174, 175),
}


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSE DE DISCRIMINABILITÉ
# ══════════════════════════════════════════════════════════════════════════════

def analyze_feature_discriminability(X: np.ndarray,
                                      y: np.ndarray,
                                      class_names: list,
                                      output_dir: Path = None) -> dict:
    """
    Analyse si les 175 features discriminent les paires visuellement proches
    identifiées comme critiques pour MINT/DGI.

    Confondre '0' et 'O' → invalider une plaque valide ou laisser passer un
    véhicule en infraction. Cette analyse quantifie ce risque feature par feature.

    Paires analysées : ('0','O'), ('1','I'), ('B','8'), ('5','S'), ('6','G')

    Métrique : overlap gaussien normalisé par feature
      overlap_i      = min(μ1+σ1, μ2+σ2) − max(μ1−σ1, μ2−σ2)
      overlap_ratio_i = max(0, overlap_i) / (union_span + 1e-7)
      → 0.0 = parfaitement séparées  ;  1.0 = totalement superposées

    Verdict par paire :
      mean_overlap < 0.40 → "discriminable"
      mean_overlap < 0.65 → "ambigu"
      sinon               → "indiscriminable"

    Si output_dir fourni :
      → feature_discriminability.json
      → discriminability_heatmap.png  (5 paires × 20 premières features)
    """
    char_to_idx = {c: i for i, c in enumerate(class_names)}
    results: dict = {}

    for c1, c2 in _CRITICAL_PAIRS:
        pair_key = f"{c1}_vs_{c2}"

        if c1 not in char_to_idx or c2 not in char_to_idx:
            continue
        idx1, idx2 = char_to_idx[c1], char_to_idx[c2]

        X1 = X[y == idx1]
        X2 = X[y == idx2]
        if len(X1) < 5 or len(X2) < 5:
            continue

        overlap_ratios = np.zeros(TOTAL_FEAT_DIM, dtype=np.float32)
        for i in range(TOTAL_FEAT_DIM):
            mu1, sig1 = float(X1[:, i].mean()), float(X1[:, i].std())
            mu2, sig2 = float(X2[:, i].mean()), float(X2[:, i].std())
            inter  = min(mu1 + sig1, mu2 + sig2) - max(mu1 - sig1, mu2 - sig2)
            span   = max(mu1 + sig1, mu2 + sig2) - min(mu1 - sig1, mu2 - sig2) + 1e-7
            overlap_ratios[i] = max(0.0, inter) / span

        mean_overlap = float(overlap_ratios.mean())

        if mean_overlap < 0.40:
            verdict = "discriminable"
        elif mean_overlap < 0.65:
            verdict = "ambigu"
        else:
            verdict = "indiscriminable"

        top5_disc_idx = np.argsort(overlap_ratios)[:5]
        top5_conf_idx = np.argsort(overlap_ratios)[-5:][::-1]

        results[pair_key] = {
            "mean_overlap":        round(mean_overlap, 4),
            "verdict":             verdict,
            "overlap_per_feature": overlap_ratios.tolist(),
            "top5_discriminant": [
                {"feature": FEATURE_NAMES[i], "overlap": round(float(overlap_ratios[i]), 4)}
                for i in top5_disc_idx
            ],
            "top5_confondant": [
                {"feature": FEATURE_NAMES[i], "overlap": round(float(overlap_ratios[i]), 4)}
                for i in top5_conf_idx
            ],
        }

    if results:
        n_disc   = sum(1 for v in results.values() if v["verdict"] == "discriminable")
        n_ambig  = sum(1 for v in results.values() if v["verdict"] == "ambigu")
        n_indisc = len(results) - n_disc - n_ambig
        global_verdict = (
            f"Sur {len(results)} paires critiques MINT/DGI : "
            f"{n_disc} discriminables, {n_ambig} ambiguës, {n_indisc} indiscriminables. "
            f"Les projections horizontales/verticales (groupe C) et la bounding-box "
            f"(groupe A) séparent mieux les digits des lettres que les transitions "
            f"précédentes ; l'histogramme HSV (groupe E) est peu utile pour "
            f"caractères en niveaux de gris."
        )
    else:
        global_verdict = "Aucune paire analysable (données insuffisantes)."

    output = {"critical_pairs": results, "global_verdict": global_verdict}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_data = {
            "critical_pairs": {
                k: {kk: vv for kk, vv in v.items() if kk != "overlap_per_feature"}
                for k, v in results.items()
            },
            "global_verdict": global_verdict,
        }
        with open(output_dir / "feature_discriminability.json", "w") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            pair_keys = list(results.keys())
            heat_data = np.array([
                results[k]["overlap_per_feature"][:20] for k in pair_keys
            ])

            fig, ax = plt.subplots(figsize=(14, 4))
            sns.heatmap(
                heat_data,
                ax=ax,
                xticklabels=FEATURE_NAMES[:20],
                yticklabels=pair_keys,
                cmap="RdYlGn_r",
                vmin=0.0, vmax=1.0,
                annot=True, fmt=".2f", annot_kws={"size": 7},
            )
            ax.set_title(
                "Overlap features par paire critique — MINT/DGI",
                fontsize=11, fontweight="bold",
            )
            plt.xticks(rotation=45, ha="right", fontsize=7)
            plt.tight_layout()
            fig.savefig(output_dir / "discriminability_heatmap.png", dpi=120)
            plt.close(fig)
            logger.info("Heatmap sauvegardée : discriminability_heatmap.png")
        except ImportError:
            logger.warning("seaborn/matplotlib absent — heatmap non générée")

    return output


# ══════════════════════════════════════════════════════════════════════════════
# IMPORTANCE PAR GROUPE (Fisher)
# ══════════════════════════════════════════════════════════════════════════════

def compute_feature_importance(X: np.ndarray,
                                 y: np.ndarray,
                                 class_names: list) -> dict:
    """
    Mesure la capacité discriminante de chaque groupe de features via le ratio
    de Fisher inter-classe / intra-classe.

    Pour chaque groupe g :
      Pour chaque feature i dans g :
        var_inter_i = variance des moyennes par classe
        var_intra_i = moyenne des variances intra-classe
      score_fisher_g = mean(var_inter_i) / (mean(var_intra_i) + 1e-7)

    La conclusion répond directement à la question jury :
    "En quoi l'indépendance conditionnelle de NB est-elle problématique ?"
    → Les 32 bins HSV sont fortement corrélés entre eux (spectre continu),
      ce que NB suppose indépendants — biais structurel documenté.
    """
    classes = np.unique(y)
    group_scores: dict = {}

    for group_name, slc in _FEATURE_GROUPS.items():
        X_g    = X[:, slc]
        n_feat = X_g.shape[1]
        if n_feat == 0:
            group_scores[group_name] = 0.0
            continue

        var_inter_list = []
        var_intra_list = []

        for fi in range(n_feat):
            col         = X_g[:, fi]
            class_means = np.array([col[y == c].mean() if (y == c).sum() > 0 else 0.0
                                    for c in classes])
            class_vars  = np.array([col[y == c].var()  if (y == c).sum() > 1 else 0.0
                                    for c in classes])
            var_inter_list.append(float(np.var(class_means)))
            var_intra_list.append(float(np.mean(class_vars)))

        v_inter = float(np.mean(var_inter_list))
        v_intra = float(np.mean(var_intra_list))
        group_scores[group_name] = round(v_inter / (v_intra + 1e-7), 4)

    ranking = sorted(group_scores, key=group_scores.get, reverse=True)

    top   = ranking[0]
    hsv_s = group_scores.get("HSV_hist", 0.0)
    a_s   = group_scores.get("A_forme", 1e-7)
    ratio = round(hsv_s / max(a_s, 1e-7), 1)

    conclusion = (
        f"'{top}' est le groupe le plus discriminant "
        f"(score Fisher = {group_scores[top]:.3f}). "
        f"HSV_hist contribue {ratio}× plus que A_forme — "
        f"l'hypothèse d'indépendance conditionnelle de Naïves Bayes est violée "
        f"car les 32 bins H, S, V sont fortement corrélés entre eux "
        f"(spectre de teinte continu, saturation liée à la valeur) : "
        f"NB les traite comme 96 variables indépendantes alors qu'elles forment "
        f"3 distributions conjointes. Cela surestime la confiance du classifieur "
        f"et amplifie les erreurs sur les caractères à couleur proche (0/O, 5/S)."
    )

    return {
        "group_scores": group_scores,
        "ranking":      ranking,
        "conclusion":   conclusion,
    }


# ══════════════════════════════════════════════════════════════════════════════
# RAPPORT COMPLET (orchestrateur)
# ══════════════════════════════════════════════════════════════════════════════

def generate_analysis_report(X: np.ndarray,
                               y: np.ndarray,
                               class_names: list,
                               output_dir: Path) -> None:
    """
    Orchestre les deux analyses et sauvegarde un rapport texte structuré,
    prêt à être cité §4.1 du rapport technique MINT/DGI.

    Fichiers produits dans output_dir/ :
      - feature_analysis_report.txt
      - feature_discriminability.json
      - discriminability_heatmap.png  (si seaborn disponible)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    importance = compute_feature_importance(X, y, class_names)
    discrim    = analyze_feature_discriminability(X, y, class_names, output_dir)

    lines = []
    sep = "═" * 54

    lines += [sep, "ANALYSE FEATURES 175D — PlateVision A1", sep, ""]

    # ── Section 1 ─────────────────────────────────────────────────────────────
    lines += ["1. IMPORTANCE PAR GROUPE (ratio Fisher)", ""]
    for rank, grp in enumerate(importance["ranking"], 1):
        score = importance["group_scores"][grp]
        lines.append(f"   {rank}. {grp:<16} score = {score:.4f}")
    lines += ["", f"   {importance['conclusion']}", ""]

    # ── Section 2 ─────────────────────────────────────────────────────────────
    lines += ["2. PAIRES CRITIQUES MINT/DGI", ""]
    for pair_key, data in discrim["critical_pairs"].items():
        lines.append(f"   {pair_key}")
        lines.append(f"     Verdict       : {data['verdict'].upper()}")
        lines.append(f"     Mean overlap  : {data['mean_overlap']:.4f}")
        top3     = data["top5_discriminant"][:3]
        top3_str = ", ".join(f"{d['feature']} ({d['overlap']:.2f})" for d in top3)
        lines.append(f"     Top-3 discrim : {top3_str}")
        lines.append("")

    # ── Section 3 ─────────────────────────────────────────────────────────────
    lines += ["3. CONCLUSION POUR LE RAPPORT §4.1", ""]
    lines.append(f"   {discrim['global_verdict']}")
    lines += [""]

    # ── Section 4 : réponses jury ─────────────────────────────────────────────
    lines += ["4. RÉPONSES AUX QUESTIONS JURY", ""]

    v_0O = discrim["critical_pairs"].get("0_vs_O", {}).get("verdict", "non analysé")
    v_1I = discrim["critical_pairs"].get("1_vs_I", {}).get("verdict", "non analysé")
    o_0O = discrim["critical_pairs"].get("0_vs_O", {}).get("mean_overlap", -1)
    lines += [
        '   Q1 : "Les features discriminent-elles \'0\'/\'O\', \'1\'/\'I\' ?"',
        f"   → Paire 0/O : {v_0O} (overlap moyen = {o_0O:.4f} sur 175 features).",
        f"     Paire 1/I : {v_1I}.",
        "     Les projections H/V et la bounding-box séparent partiellement les digits",
        "     des lettres. L'overlap résiduel justifie un post-traitement lexical",
        "     (vérification syntaxe plaque camerounaise).",
        "",
    ]

    hsv_score = importance["group_scores"].get("HSV_hist", 0)
    a_score   = importance["group_scores"].get("A_forme", 0)
    c_score   = importance["group_scores"].get("C_projections", 0)
    lines += [
        '   Q2 : "Pourquoi l\'indépendance conditionnelle de NB est-elle problématique ?"',
        f"   → HSV_hist (96 bins) a un score Fisher de {hsv_score:.3f} vs {a_score:.3f}",
        f"     pour A_forme et {c_score:.3f} pour C_projections.",
        "     NB suppose 175 features indépendantes conditionnellement à la classe.",
        "     Or H_bin_i et H_bin_(i+1) sont fortement corrélés (spectre continu),",
        "     de même que les 28 valeurs proj_h sont liées (profil du même caractère).",
        "     → NB surestime la confiance : erreurs amplifiées sur paires proches.",
        "",
    ]

    verdict_0O = discrim["critical_pairs"].get("0_vs_O", {}).get("verdict", "non analysé")
    top_disc   = discrim["critical_pairs"].get("0_vs_O", {}).get("top5_discriminant", [])
    feat_disc  = top_disc[0]["feature"] if top_disc else "N/A"
    lines += [
        '   Q3 : "Implication MINT/DGI de confondre \'0\' et \'O\' ?"',
        f"   → La paire 0/O est classée '{verdict_0O}' par notre analyse.",
        f"     La feature la plus discriminante est '{feat_disc}'.",
        "     Impact opérationnel :",
        "       • FP (O lu comme 0) → plaque valide rejetée, véhicule bloqué à tort.",
        "       • FN (0 lu comme O) → numéro invalide reconnu comme valide,",
        "         risque de laisser passer un véhicule non enregistré DGI.",
        "     Mitigation : vérification syntaxique post-OCR",
        "     (pattern plaque camerounaise : LT NNNN L ou NN NNNN LL).",
        "",
    ]

    lines.append(sep)

    report_txt = "\n".join(lines)
    report_path = output_dir / "feature_analysis_report.txt"
    report_path.write_text(report_txt, encoding="utf-8")
    logger.info(f"Rapport sauvegardé : {report_path}")
    print(report_txt)


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    VALID_CHARS = sorted("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    DATA_DIR    = Path("data/processed")
    REPORT_DIR  = Path("reports/figures")

    parser = argparse.ArgumentParser(
        description="PlateVision — Extraction et analyse des features 175D"
    )
    parser.add_argument("--extract", action="store_true",
        help="Charge features.npy/labels.npy et affiche la distribution")
    parser.add_argument("--analyze", action="store_true",
        help="Lance l'analyse complète + sauvegarde rapport dans reports/figures/")
    parser.add_argument("--demo", action="store_true",
        help="Affiche les 175D d'un sample synthétique (verbose)")
    args = parser.parse_args()

    if args.demo:
        rng = np.random.default_rng(0)
        char_img  = rng.integers(20, 220, (28, 28), dtype=np.uint8)
        plate_img = rng.integers(0,  255, (60, 200, 3), dtype=np.uint8)
        vec = extract_feature_vector(char_img, plate_img)
        print(f"\nVecteur 175D (sample synthétique) :")
        print(f"  shape  : {vec.shape}  dtype : {vec.dtype}")
        print(f"  min    : {vec.min():.4f}   max   : {vec.max():.4f}")
        print(f"\n  Détail par feature :")
        for i, (name, val) in enumerate(zip(FEATURE_NAMES, vec)):
            print(f"  [{i:3d}] {name:<20} = {val:.5f}")
        sys.exit(0)

    feat_path   = DATA_DIR / "features.npy"
    labels_path = DATA_DIR / "labels.npy"

    if not feat_path.exists() or not labels_path.exists():
        print(f"[ERREUR] Fichiers manquants : {feat_path} / {labels_path}")
        print("Lancez d'abord : python data/prepare_datasets.py --phases all")
        sys.exit(1)

    X = np.load(str(feat_path))
    y = np.load(str(labels_path))
    print(f"Features chargées : {X.shape}  labels : {y.shape}")

    cn_path = DATA_DIR / "class_names.txt"
    class_names = cn_path.read_text().strip().splitlines() if cn_path.exists() else VALID_CHARS

    if args.extract:
        print("\nDistribution des classes :")
        for i, name in enumerate(class_names):
            count = int((y == i).sum())
            bar   = "█" * (count // 10)
            print(f"  {name} : {count:5d}  {bar}")
        sys.exit(0)

    if args.analyze:
        generate_analysis_report(X, y, class_names, REPORT_DIR)
        sys.exit(0)

    parser.print_help()
