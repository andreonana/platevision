"""
PlateVision — Module A — OCR
Extraction et distinction des caractères des plaques
MINT/DGI Cameroun — UCAC-ICAM
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import re
from tqdm import tqdm
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ── Chemins ──
TRAIN_CSV  = r"C:\Users\ibnal\Desktop\PROJET IA X3\archive\train\_annotations.csv"
TEST_CSV   = r"C:\Users\ibnal\Desktop\PROJET IA X3\archive\test\_annotations.csv"
VALID_CSV  = r"C:\Users\ibnal\Desktop\PROJET IA X3\archive\valid\_annotations.csv"
TRAIN_PATH = r"C:\Users\ibnal\Desktop\PROJET IA X3\archive\train"
TEST_PATH  = r"C:\Users\ibnal\Desktop\PROJET IA X3\archive\test"
VALID_PATH = r"C:\Users\ibnal\Desktop\PROJET IA X3\archive\valid"

PATH_PLATES_CROP = "data/processed/plates_cropped"
PATH_OCR_OUT     = "data/processed/ocr_results"
PATH_CONFORMES   = "data/processed/sorted_plates/conformes"
PATH_DOUTEUSES   = "data/processed/sorted_plates/douteuses"
PATH_ILLISIBLES  = "data/processed/sorted_plates/illisibles"
PATH_FIGURES     = "outputs/figures"

for p in [PATH_PLATES_CROP, PATH_OCR_OUT,
          PATH_CONFORMES, PATH_DOUTEUSES,
          PATH_ILLISIBLES, PATH_FIGURES]:
    os.makedirs(p, exist_ok=True)


# ════════════════════════════════════════════════
# CHARGEMENT DES ANNOTATIONS
# ════════════════════════════════════════════════

def charger_annotations(csv_path, images_folder):
    """
    Charge le CSV et vérifie que les images existent.
    """
    if not os.path.exists(csv_path):
        print(f"  ⚠ CSV introuvable : {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df['image_full_path'] = df['filename'].apply(
        lambda f: os.path.join(images_folder, f)
    )
    df['image_exists'] = df['image_full_path'].apply(os.path.exists)
    df_ok = df[df['image_exists']].copy()

    print(f"  [{os.path.basename(images_folder):5s}] "
          f"{len(df_ok):4d} images valides / "
          f"{len(df)} annotations")
    return df_ok


def charger_tous_splits():
    """
    Charge les 3 splits train / test / valid.
    Retourne trois DataFrames.
    """
    print("\n" + "─" * 55)
    print("  CHARGEMENT DES ANNOTATIONS")
    print("─" * 55)
    df_train = charger_annotations(TRAIN_CSV, TRAIN_PATH)
    df_test  = charger_annotations(TEST_CSV,  TEST_PATH)
    df_valid = charger_annotations(VALID_CSV, VALID_PATH)
    total = len(df_train) + len(df_test) + len(df_valid)
    print(f"  Total : {total:,} annotations chargées")
    return df_train, df_test, df_valid


# ════════════════════════════════════════════════
# PRÉTRAITEMENT
# ════════════════════════════════════════════════

def recadrer_plaque(image, xmin, ymin, xmax, ymax, marge=5):
    """Recadre la région plaque avec une petite marge."""
    h, w = image.shape[:2]
    x1 = max(0, int(xmin) - marge)
    y1 = max(0, int(ymin) - marge)
    x2 = min(w, int(xmax) + marge)
    y2 = min(h, int(ymax) + marge)
    return image[y1:y2, x1:x2]


def pretraiter_pour_ocr(plaque_img):
    """
    5 versions prétraitées pour maximiser la lisibilité OCR.
    """
    h, w = plaque_img.shape[:2]
    if w < 150:
        scale      = 150 / w
        plaque_img = cv2.resize(
            plaque_img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC
        )

    gray  = cv2.cvtColor(plaque_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

    v_clahe    = clahe.apply(gray)
    _, v_otsu  = cv2.threshold(
        v_clahe, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, v_otsu_inv = cv2.threshold(
        v_clahe, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    v_denoise  = clahe.apply(
        cv2.fastNlMeansDenoising(gray, h=7)
    )
    big        = cv2.resize(
        plaque_img, (w * 2, h * 2),
        interpolation=cv2.INTER_CUBIC
    )
    v_big      = clahe.apply(
        cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    )

    return {
        'clahe'    : v_clahe,
        'otsu'     : v_otsu,
        'otsu_inv' : v_otsu_inv,
        'denoised' : v_denoise,
        'big_clahe': v_big,
    }


# ════════════════════════════════════════════════
# ANALYSE DES CARACTÈRES — CŒUR DU MODULE OCR
# ════════════════════════════════════════════════

def analyser_caracteres(texte):
    """
    Distingue et sépare PRÉCISÉMENT :
      • LETTRES  (A-Z)
      • CHIFFRES (0-9)
      • SYMBOLES (-, /, ., espace)
    C'est la distinction demandée par le chef.
    """
    if not texte:
        return {
            'texte_clean': '', 'lettres': '',
            'chiffres': '',    'symboles': '',
            'n_lettres': 0,    'n_chiffres': 0,
            'n_symboles': 0,   'longueur': 0,
            'sequence': [],
        }

    texte_upper = texte.upper().strip()
    lettres, chiffres, symboles, sequence = [], [], [], []

    for char in texte_upper:
        if char.isalpha():
            lettres.append(char)
            sequence.append({'char': char, 'type': 'LETTRE'})
        elif char.isdigit():
            chiffres.append(char)
            sequence.append({'char': char, 'type': 'CHIFFRE'})
        elif char in ['-', '/', '.', ' ', '_']:
            symboles.append(char)
            sequence.append({'char': char, 'type': 'SYMBOLE'})

    texte_clean = ''.join(c['char'] for c in sequence)

    return {
        'texte_clean': texte_clean,
        'lettres'    : ''.join(lettres),
        'chiffres'   : ''.join(chiffres),
        'symboles'   : ''.join(symboles),
        'n_lettres'  : len(lettres),
        'n_chiffres' : len(chiffres),
        'n_symboles' : len(symboles),
        'longueur'   : len(texte_clean),
        'sequence'   : sequence,
    }


def ocr_lire_plaque(reader, versions):
    """
    Lance OCR sur toutes les versions prétraitées.
    Retourne le meilleur texte + analyse lettres/chiffres/symboles.
    """
    meilleur_texte = ""
    meilleure_conf = 0.0

    for nom, version in versions.items():
        try:
            resultats = reader.readtext(
                version,
                detail=1,
                paragraph=False,
                min_size=10,
                contrast_ths=0.1,
                adjust_contrast=0.5,
                text_threshold=0.6,
                low_text=0.3,
                link_threshold=0.4,
            )
            for (bbox, texte, conf) in resultats:
                if not texte or conf < 0.1:
                    continue
                analyse_temp = analyser_caracteres(texte)
                t_clean      = analyse_temp['texte_clean']
                score        = conf * min(len(t_clean), 8) / 8
                if score > meilleure_conf and len(t_clean) >= 2:
                    meilleure_conf = score
                    meilleur_texte = t_clean
        except Exception:
            continue

    analyse_finale = analyser_caracteres(meilleur_texte)
    return meilleur_texte, round(meilleure_conf, 4), analyse_finale


def classifier_plaque(texte, confiance):
    """
    CONFORME  : texte lisible, lettres ET chiffres, conf >= 0.5
    DOUTEUSE  : partiellement lisible
    ILLISIBLE : OCR en échec
    """
    if not texte or len(texte) < 2:
        return 'ILLISIBLE', 'texte_vide'
    if confiance < 0.25:
        return 'ILLISIBLE', 'confiance_trop_faible'

    a = analyser_caracteres(texte)
    n_tot = a['n_lettres'] + a['n_chiffres']

    if n_tot < 3:
        return 'ILLISIBLE', 'moins_3_chars_alphanumeriques'
    if a['n_lettres'] == 0:
        return 'DOUTEUSE', 'aucune_lettre'
    if a['n_chiffres'] == 0:
        return 'DOUTEUSE', 'aucun_chiffre'
    if confiance < 0.5:
        return 'DOUTEUSE', 'confiance_moyenne'
    if n_tot < 4:
        return 'DOUTEUSE', 'texte_court'

    return 'CONFORME', 'format_valide'


# ════════════════════════════════════════════════
# PIPELINE OCR PRINCIPAL
# ════════════════════════════════════════════════

def run_ocr(df_train, df_test, df_valid,
            reader, max_images=None):
    """
    Pipeline OCR complet sur les 3 splits.
    Pour chaque plaque annotée :
      recadrer → prétraiter → OCR → analyser → classifier → trier
    """
    print("\n" + "─" * 55)
    print("  OCR — PIPELINE PRINCIPAL")
    print("─" * 55)

    resultats = []
    splits = {
        'train': (df_train, TRAIN_PATH),
        'test' : (df_test,  TEST_PATH),
        'valid': (df_valid, VALID_PATH),
    }

    dossiers_tri = {
        'CONFORME' : PATH_CONFORMES,
        'DOUTEUSE' : PATH_DOUTEUSES,
        'ILLISIBLE': PATH_ILLISIBLES,
    }

    for split_nom, (df_split, img_folder) in splits.items():
        if df_split.empty:
            continue

        df_work = df_split.head(max_images) \
            if max_images else df_split
        print(f"\n  Split [{split_nom}] — "
              f"{len(df_work)} annotations")

        for idx, row in tqdm(
            df_work.iterrows(),
            total=len(df_work),
            desc=f"  OCR {split_nom}"
        ):
            img = cv2.imread(row['image_full_path'])
            if img is None:
                continue

            # Recadrer la plaque
            plaque = recadrer_plaque(
                img,
                row['xmin'], row['ymin'],
                row['xmax'], row['ymax']
            )
            if plaque.size == 0:
                continue

            # Sauvegarder la plaque recadrée
            nom = f"{split_nom}_{idx}_{os.path.basename(row['image_full_path'])}"
            plaque_path = os.path.join(PATH_PLATES_CROP, nom)
            cv2.imwrite(plaque_path, plaque)

            # Prétraitement
            versions = pretraiter_pour_ocr(plaque)

            # OCR + analyse
            texte, conf, analyse = ocr_lire_plaque(reader, versions)

            # Classification
            categorie, raison = classifier_plaque(texte, conf)

            # Tri
            shutil.copy2(
                plaque_path,
                os.path.join(dossiers_tri[categorie], nom)
            )

            resultats.append({
                'split'        : split_nom,
                'filename'     : row['filename'],
                'image_path'   : row['image_full_path'],
                'plaque_path'  : plaque_path,
                'xmin'         : row['xmin'],
                'ymin'         : row['ymin'],
                'xmax'         : row['xmax'],
                'ymax'         : row['ymax'],
                'texte_ocr'    : texte,
                'confiance_ocr': conf,
                'lettres'      : analyse['lettres'],
                'chiffres'     : analyse['chiffres'],
                'symboles'     : analyse['symboles'],
                'n_lettres'    : analyse['n_lettres'],
                'n_chiffres'   : analyse['n_chiffres'],
                'n_symboles'   : analyse['n_symboles'],
                'longueur'     : analyse['longueur'],
                'categorie'    : categorie,
                'raison'       : raison,
            })

    # Sauvegarder
    df_res = pd.DataFrame(resultats)
    csv_out = os.path.join(PATH_OCR_OUT,
                           'ocr_resultats_complets.csv')
    df_res.to_csv(csv_out, index=False, encoding='utf-8')

    # Bilan
    total = len(df_res)
    if total > 0:
        print(f"\n  {'BILAN OCR':^50}")
        print(f"  {'─'*50}")
        print(f"  Total traité : {total:,}")
        for cat in ['CONFORME', 'DOUTEUSE', 'ILLISIBLE']:
            n   = (df_res['categorie'] == cat).sum()
            pct = n / total * 100
            bar = '█' * int(pct / 3)
            print(f"  {cat:10s}: {n:5,} ({pct:5.1f}%) {bar}")
        print(f"\n  Lettres  moy : {df_res['n_lettres'].mean():.1f}")
        print(f"  Chiffres moy : {df_res['n_chiffres'].mean():.1f}")
        print(f"  Symboles moy : {df_res['n_symboles'].mean():.1f}")
        print(f"\n  ✓ CSV : {csv_out}")

    return df_res


# ════════════════════════════════════════════════
# VISUALISATIONS OCR
# ════════════════════════════════════════════════

def visualiser_ocr(df_res):
    """4 figures d'analyse OCR."""
    if df_res.empty:
        print("  ⚠ Pas de données à visualiser")
        return

    print("\n  Génération des figures OCR...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        'OCR — Lettres / Chiffres / Symboles\n'
        'PlateVision MINT/DGI Cameroun',
        fontsize=14, fontweight='bold'
    )

    # 1. Camembert catégories
    cats    = ['CONFORME', 'DOUTEUSE', 'ILLISIBLE']
    counts  = [(df_res['categorie'] == c).sum() for c in cats]
    colors  = ['#2ecc71', '#f39c12', '#e74c3c']
    axes[0, 0].pie(
        counts,
        labels=[f"{c}\n{n:,}" for c, n in zip(cats, counts)],
        colors=colors, autopct='%1.1f%%',
        startangle=90, explode=(0.05, 0.05, 0.05)
    )
    axes[0, 0].set_title('Répartition des catégories',
                          fontweight='bold')

    # 2. Distribution confidences
    axes[0, 1].hist(df_res['confiance_ocr'], bins=30,
                    color='steelblue', edgecolor='white',
                    alpha=0.85)
    axes[0, 1].axvline(x=0.25, color='red',
                        linestyle='--', label='Seuil illisible')
    axes[0, 1].axvline(x=0.5, color='orange',
                        linestyle='--', label='Seuil conforme')
    axes[0, 1].set_title('Distribution des confidences OCR',
                          fontweight='bold')
    axes[0, 1].set_xlabel('Score de confiance')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Lettres vs Chiffres vs Symboles
    labels_bar = ['Lettres', 'Chiffres', 'Symboles']
    means      = [
        df_res['n_lettres'].mean(),
        df_res['n_chiffres'].mean(),
        df_res['n_symboles'].mean(),
    ]
    bar_colors = ['#3498db', '#e74c3c', '#9b59b6']
    bars = axes[1, 0].bar(
        range(3), means, color=bar_colors,
        alpha=0.85, edgecolor='white', width=0.5
    )
    for bar, val in zip(bars, means):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f'{val:.2f}', ha='center',
            fontweight='bold', fontsize=11
        )
    axes[1, 0].set_xticks(range(3))
    axes[1, 0].set_xticklabels(labels_bar, fontsize=12)
    axes[1, 0].set_ylabel('Nombre moyen par plaque')
    axes[1, 0].set_title(
        'Composition moyenne — Lettres / Chiffres / Symboles',
        fontweight='bold'
    )
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 4. Scatter confiance vs longueur
    coul_scatter = df_res['categorie'].map({
        'CONFORME' : '#2ecc71',
        'DOUTEUSE' : '#f39c12',
        'ILLISIBLE': '#e74c3c'
    })
    axes[1, 1].scatter(
        df_res['longueur'], df_res['confiance_ocr'],
        c=coul_scatter, alpha=0.5, s=20
    )
    legend_elems = [
        Patch(facecolor='#2ecc71', label='CONFORME'),
        Patch(facecolor='#f39c12', label='DOUTEUSE'),
        Patch(facecolor='#e74c3c', label='ILLISIBLE'),
    ]
    axes[1, 1].legend(handles=legend_elems, fontsize=9)
    axes[1, 1].set_xlabel('Longueur texte OCR')
    axes[1, 1].set_ylabel('Confiance OCR')
    axes[1, 1].set_title('Confiance vs Longueur',
                          fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    p1 = os.path.join(PATH_FIGURES, 'ocr_analyse.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Figure : {p1}")

    # Exemples par catégorie
    fig2, axes2 = plt.subplots(3, 5, figsize=(18, 10))
    fig2.suptitle(
        'Exemples par Catégorie — Lettres / Chiffres / Symboles\n'
        'PlateVision MINT/DGI Cameroun',
        fontsize=13, fontweight='bold'
    )
    cat_colors = {
        'CONFORME' : '#2ecc71',
        'DOUTEUSE' : '#f39c12',
        'ILLISIBLE': '#e74c3c'
    }
    for row_idx, cat in enumerate(
            ['CONFORME', 'DOUTEUSE', 'ILLISIBLE']
    ):
        df_cat = df_res[df_res['categorie'] == cat].head(5)
        for col_idx in range(5):
            ax = axes2[row_idx, col_idx]
            if col_idx < len(df_cat):
                r   = df_cat.iloc[col_idx]
                img = cv2.imread(r['plaque_path'])
                if img is not None:
                    ax.imshow(
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    )
                    ax.set_title(
                        f"'{r['texte_ocr']}'\n"
                        f"L:{r['lettres']} "
                        f"C:{r['chiffres']} "
                        f"S:{r['symboles']}\n"
                        f"conf:{r['confiance_ocr']:.2f}",
                        fontsize=7,
                        color=cat_colors[cat]
                    )
            if col_idx == 0:
                ax.set_ylabel(
                    cat, fontsize=10, fontweight='bold',
                    color=cat_colors[cat]
                )
            ax.axis('off')

    plt.tight_layout()
    p2 = os.path.join(PATH_FIGURES, 'ocr_exemples.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Figure : {p2}")