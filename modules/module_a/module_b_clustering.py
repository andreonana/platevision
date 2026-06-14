"""
PlateVision — Module B — Clustering Complet
K-Means sur embeddings visuels + OCR
Tout le dataset — MINT/DGI Cameroun — UCAC-ICAM
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    adjusted_rand_score,
)
from sklearn.preprocessing import normalize, LabelEncoder
import json
from tqdm import tqdm
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
PATH_EMBEDDINGS  = "data/processed/embeddings"
PATH_FIGURES     = "outputs/figures"

for p in [PATH_PLATES_CROP, PATH_OCR_OUT,
          PATH_EMBEDDINGS, PATH_FIGURES]:
    os.makedirs(p, exist_ok=True)


# ════════════════════════════════════════════════════════
# PARTIE 1 — OCR COMPLET SUR TOUT LE DATASET
# ════════════════════════════════════════════════════════

def charger_annotations(csv_path, images_folder):
    """Charge le CSV et vérifie que les images existent."""
    if not os.path.exists(csv_path):
        print(f"  ⚠ Introuvable : {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df['image_full_path'] = df['filename'].apply(
        lambda f: os.path.join(images_folder, f)
    )
    df['image_exists'] = df['image_full_path'].apply(
        os.path.exists
    )
    df_ok = df[df['image_exists']].copy()
    print(f"  {os.path.basename(images_folder):6s} : "
          f"{len(df_ok):5d} images valides")
    return df_ok


def recadrer_plaque(image, xmin, ymin, xmax, ymax,
                    marge=5):
    """Recadre la région plaque avec marge."""
    h, w = image.shape[:2]
    x1 = max(0, int(xmin) - marge)
    y1 = max(0, int(ymin) - marge)
    x2 = min(w, int(xmax) + marge)
    y2 = min(h, int(ymax) + marge)
    return image[y1:y2, x1:x2]


def pretraiter(plaque_img):
    """5 versions prétraitées pour l'OCR."""
    h, w = plaque_img.shape[:2]
    if w < 150:
        scale      = 150 / w
        plaque_img = cv2.resize(
            plaque_img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC
        )
    gray  = cv2.cvtColor(plaque_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=3.0, tileGridSize=(4, 4)
    )
    v1 = clahe.apply(gray)
    _, v2 = cv2.threshold(
        v1, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, v3 = cv2.threshold(
        v1, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    v4 = clahe.apply(
        cv2.fastNlMeansDenoising(gray, h=7)
    )
    big = cv2.resize(
        plaque_img, (w * 2, h * 2),
        interpolation=cv2.INTER_CUBIC
    )
    v5 = clahe.apply(
        cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    )
    return {'clahe': v1, 'otsu': v2,
            'otsu_inv': v3, 'denoised': v4, 'big': v5}


def analyser_caracteres(texte):
    """
    Distingue LETTRES / CHIFFRES / SYMBOLES.
    Retourne un dict avec chaque catégorie séparée.
    """
    if not texte or str(texte) == 'nan':
        return {
            'texte_clean': '', 'lettres': '',
            'chiffres': '', 'symboles': '',
            'n_lettres': 0, 'n_chiffres': 0,
            'n_symboles': 0, 'longueur': 0,
        }
    texte_u = str(texte).upper().strip()
    lettres, chiffres, symboles, seq = [], [], [], []
    for c in texte_u:
        if c.isalpha():
            lettres.append(c)
            seq.append(c)
        elif c.isdigit():
            chiffres.append(c)
            seq.append(c)
        elif c in ['-', '/', '.', ' ', '_']:
            symboles.append(c)
            seq.append(c)
    return {
        'texte_clean': ''.join(seq),
        'lettres'    : ''.join(lettres),
        'chiffres'   : ''.join(chiffres),
        'symboles'   : ''.join(symboles),
        'n_lettres'  : len(lettres),
        'n_chiffres' : len(chiffres),
        'n_symboles' : len(symboles),
        'longueur'   : len(seq),
    }


def ocr_lire(reader, versions):
    """OCR sur toutes les versions, retourne le meilleur."""
    meilleur = ''
    conf_max = 0.0
    for nom, version in versions.items():
        try:
            res = reader.readtext(
                version, detail=1, paragraph=False,
                min_size=10, contrast_ths=0.1,
                adjust_contrast=0.5,
                text_threshold=0.6,
                low_text=0.3, link_threshold=0.4,
            )
            for (_, texte, conf) in res:
                if not texte or conf < 0.1:
                    continue
                a = analyser_caracteres(texte)
                sc = conf * min(len(a['texte_clean']), 8) / 8
                if sc > conf_max and len(
                    a['texte_clean']
                ) >= 2:
                    conf_max = sc
                    meilleur = a['texte_clean']
        except Exception:
            continue
    return meilleur, round(conf_max, 4)


def classifier(texte, conf):
    """CONFORME / DOUTEUSE / ILLISIBLE."""
    if not texte or len(str(texte)) < 2:
        return 'ILLISIBLE', 'texte_vide'
    if conf < 0.25:
        return 'ILLISIBLE', 'conf_trop_faible'
    a   = analyser_caracteres(texte)
    tot = a['n_lettres'] + a['n_chiffres']
    if tot < 3:
        return 'ILLISIBLE', 'moins_3_chars'
    if a['n_lettres'] == 0:
        return 'DOUTEUSE', 'aucune_lettre'
    if a['n_chiffres'] == 0:
        return 'DOUTEUSE', 'aucun_chiffre'
    if conf < 0.5:
        return 'DOUTEUSE', 'conf_moyenne'
    return 'CONFORME', 'format_valide'


def run_ocr_complet(df_train, df_test, df_valid, reader):
    """
    OCR sur TOUT le dataset.
    Sauvegarde par batch de 500 pour ne pas tout perdre
    en cas de coupure.
    """
    print("\n" + "═" * 55)
    print("  OCR COMPLET — TOUT LE DATASET")
    print("═" * 55)

    # Vérifier si un fichier de reprise existe
    csv_reprise = os.path.join(
        PATH_OCR_OUT, 'ocr_complet_all.csv'
    )
    if os.path.exists(csv_reprise):
        print(f"\n  ✓ Fichier de reprise trouvé !")
        df_exist = pd.read_csv(csv_reprise)
        print(f"  {len(df_exist)} plaques déjà traitées")
        fichiers_faits = set(df_exist['filename'].tolist())
    else:
        df_exist      = pd.DataFrame()
        fichiers_faits = set()

    splits = {
        'train': (df_train, TRAIN_PATH),
        'test' : (df_test,  TEST_PATH),
        'valid': (df_valid, VALID_PATH),
    }

    tous = []
    compteur = 0

    for split_nom, (df_split, img_folder) in splits.items():
        if df_split.empty:
            continue

        # Filtrer ce qui n'est pas encore fait
        df_todo = df_split[
            ~df_split['filename'].isin(fichiers_faits)
        ]
        print(f"\n  [{split_nom}] {len(df_todo)} restantes "
              f"/ {len(df_split)} total")

        for idx, row in tqdm(
            df_todo.iterrows(),
            total=len(df_todo),
            desc=f"  OCR {split_nom}"
        ):
            img = cv2.imread(row['image_full_path'])
            if img is None:
                continue

            plaque = recadrer_plaque(
                img,
                row['xmin'], row['ymin'],
                row['xmax'], row['ymax']
            )
            if plaque.size == 0:
                continue

            # Sauvegarder la plaque recadrée
            nom = (f"{split_nom}_{idx}_"
                   f"{os.path.basename(row['image_full_path'])}")
            plaque_path = os.path.join(PATH_PLATES_CROP, nom)
            cv2.imwrite(plaque_path, plaque)

            # OCR
            versions    = pretraiter(plaque)
            texte, conf = ocr_lire(reader, versions)
            analyse     = analyser_caracteres(texte)
            cat, raison = classifier(texte, conf)

            tous.append({
                'split'        : split_nom,
                'filename'     : row['filename'],
                'image_path'   : row['image_full_path'],
                'plaque_path'  : plaque_path,
                'xmin'         : row['xmin'],
                'ymin'         : row['ymin'],
                'xmax'         : row['xmax'],
                'ymax'         : row['ymax'],
                'texte_ocr'    : texte if texte else '',
                'confiance_ocr': conf,
                'lettres'      : analyse['lettres'],
                'chiffres'     : analyse['chiffres'],
                'symboles'     : analyse['symboles'],
                'n_lettres'    : analyse['n_lettres'],
                'n_chiffres'   : analyse['n_chiffres'],
                'n_symboles'   : analyse['n_symboles'],
                'longueur'     : analyse['longueur'],
                'categorie'    : cat,
                'raison'       : raison,
            })
            compteur += 1

            # Sauvegarde intermédiaire tous les 500
            if compteur % 500 == 0:
                df_batch = pd.DataFrame(tous)
                if not df_exist.empty:
                    df_save = pd.concat(
                        [df_exist, df_batch],
                        ignore_index=True
                    )
                else:
                    df_save = df_batch
                df_save.to_csv(csv_reprise,
                               index=False,
                               encoding='utf-8')
                print(f"\n  💾 Sauvegarde intermédiaire : "
                      f"{len(df_save)} plaques")

    # Sauvegarde finale
    df_new = pd.DataFrame(tous)
    if not df_exist.empty and len(df_new) > 0:
        df_final = pd.concat(
            [df_exist, df_new], ignore_index=True
        )
    elif not df_exist.empty:
        df_final = df_exist
    else:
        df_final = df_new

    # Nettoyer les NaN dans les colonnes texte
    for col in ['lettres', 'chiffres', 'symboles',
                'texte_ocr']:
        df_final[col] = df_final[col].fillna('')

    df_final.to_csv(csv_reprise, index=False,
                    encoding='utf-8')

    total = len(df_final)
    print(f"\n  {'BILAN OCR COMPLET':^50}")
    print(f"  {'─'*50}")
    print(f"  Total plaques : {total:,}")
    for cat in ['CONFORME', 'DOUTEUSE', 'ILLISIBLE']:
        n   = (df_final['categorie'] == cat).sum()
        pct = n / total * 100 if total > 0 else 0
        bar = '█' * int(pct / 3)
        print(f"  {cat:10s}: {n:5,} ({pct:5.1f}%) {bar}")

    print(f"\n  ✓ CSV complet : {csv_reprise}")
    return df_final


# ════════════════════════════════════════════════════════
# PARTIE 2 — EXTRACTION DES FEATURES POUR CLUSTERING
# ════════════════════════════════════════════════════════

def extraire_features_clustering(df_ocr):
    """
    Extrait 22 features par plaque :
    - 9 features visuelles (luminosité, contraste,
      netteté, ratio, histogramme 8 bins, densité,
      gradients x/y)
    - 6 features OCR (confiance, longueur, ratio
      lettres, ratio chiffres, symboles, catégorie)
    - 7 features de texture (moments de Hu)
    Total : 22 features
    """
    print("\n" + "═" * 55)
    print("  EXTRACTION DES FEATURES — CLUSTERING")
    print("═" * 55)

    features_list = []
    meta_list     = []
    erreurs       = 0

    print(f"\n  {len(df_ocr)} plaques à traiter...")

    for _, row in tqdm(
        df_ocr.iterrows(),
        total=len(df_ocr),
        desc="  Features"
    ):
        plaque_path = str(row.get('plaque_path', ''))
        if not plaque_path or not os.path.exists(plaque_path):
            erreurs += 1
            continue

        img = cv2.imread(plaque_path)
        if img is None:
            erreurs += 1
            continue

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = img.shape[:2]
            feats = []

            # ── 9 features visuelles ──

            # 1. Luminosité moyenne normalisée
            feats.append(float(np.mean(gray)) / 255.0)

            # 2. Contraste (écart-type) normalisé
            feats.append(float(np.std(gray)) / 255.0)

            # 3. Netteté — variance du Laplacien
            lap = float(
                cv2.Laplacian(gray, cv2.CV_64F).var()
            )
            feats.append(float(np.clip(lap / 500.0, 0, 1)))

            # 4. Ratio largeur / hauteur
            feats.append(
                float(w / h) if h > 0 else 1.0
            )

            # 5-12. Histogramme 8 bins normalisé
            hist = cv2.calcHist(
                [gray], [0], None, [8], [0, 256]
            ).flatten()
            s = hist.sum()
            hist_n = hist / s if s > 0 else hist
            feats.extend(hist_n.tolist())

            # 13. Densité pixels sombres (= texte)
            _, binary = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            feats.append(
                float(np.sum(binary == 255)) / (h * w)
            )

            # 14-15. Gradients Sobel X et Y
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            feats.append(
                float(np.mean(np.abs(sx))) / 255.0
            )
            feats.append(
                float(np.mean(np.abs(sy))) / 255.0
            )

            # ── 6 features OCR ──

            # 16. Confiance OCR
            feats.append(
                float(row.get('confiance_ocr', 0))
            )

            # 17. Longueur texte normalisée sur 12
            lon = float(row.get('longueur', 0))
            feats.append(min(lon / 12.0, 1.0))

            # 18. Ratio lettres
            n_l = float(row.get('n_lettres', 0))
            n_c = float(row.get('n_chiffres', 0))
            tot = n_l + n_c
            feats.append(n_l / tot if tot > 0 else 0.0)

            # 19. Ratio chiffres
            feats.append(n_c / tot if tot > 0 else 0.0)

            # 20. Présence symboles (0 ou 1)
            feats.append(
                1.0 if float(
                    row.get('n_symboles', 0)
                ) > 0 else 0.0
            )

            # 21. Catégorie OCR encodée
            feats.append({
                'CONFORME' : 1.0,
                'DOUTEUSE' : 0.5,
                'ILLISIBLE': 0.0,
            }.get(
                str(row.get('categorie', 'ILLISIBLE')),
                0.0
            ))

            # ── 7 features de texture (Moments de Hu) ──
            moments   = cv2.moments(gray)
            hu_moments = cv2.HuMoments(moments).flatten()
            for hu in hu_moments:
                if hu != 0:
                    feats.append(
                        float(
                            -np.sign(hu) * np.log10(abs(hu))
                        )
                    )
                else:
                    feats.append(0.0)

            features_list.append(feats)
            meta_list.append(row.to_dict())

        except Exception:
            erreurs += 1
            continue

    X = np.array(features_list, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

    print(f"\n  Shape features    : {X.shape}")
    print(f"  Features/plaque   : {X.shape[1]}")
    print(f"  Erreurs ignorées  : {erreurs}")
    print(f"  NaN présents      : {np.isnan(X).any()}")

    np.save(
        os.path.join(PATH_EMBEDDINGS, 'features_b.npy'), X
    )
    print(f"  ✓ Features sauvegardées")
    return X, meta_list


# ════════════════════════════════════════════════════════
# PARTIE 3 — DÉTERMINATION DU K OPTIMAL
# ════════════════════════════════════════════════════════

def choisir_k_optimal(X_norm, k_max=8):
    """
    Méthode du coude + score silhouette.
    Retourne k optimal.
    """
    print("\n" + "═" * 55)
    print("  SÉLECTION DU K OPTIMAL")
    print("═" * 55)

    inertias      = []
    sil_scores    = []
    k_range       = range(2, k_max + 1)

    print(f"\n  Test de k = 2 à {k_max}...")

    for k in tqdm(k_range, desc="  Test k"):
        km = KMeans(
            n_clusters=k, init='k-means++',
            n_init=10, random_state=42
        )
        labels = km.fit_predict(X_norm)
        inertias.append(km.inertia_)
        sil_scores.append(
            silhouette_score(X_norm, labels)
        )
        print(f"  k={k} → inertie={km.inertia_:.1f}, "
              f"silhouette={sil_scores[-1]:.4f}")

    k_opt = list(k_range)[np.argmax(sil_scores)]
    print(f"\n  → k optimal retenu : {k_opt}")
    print(f"  → silhouette max   : {max(sil_scores):.4f}")

    # Figure coude + silhouette
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'Module B — Sélection k Optimal\n'
        'PlateVision MINT/DGI Cameroun',
        fontsize=13, fontweight='bold'
    )

    # Coude
    axes[0].plot(
        list(k_range), inertias, 'bo-',
        linewidth=2, markersize=9
    )
    axes[0].axvline(
        x=k_opt, color='red', linestyle='--',
        linewidth=2, label=f'k optimal = {k_opt}'
    )
    axes[0].set_xlabel('Nombre de clusters k', fontsize=12)
    axes[0].set_ylabel('Inertie', fontsize=12)
    axes[0].set_title('Méthode du Coude', fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Silhouette
    axes[1].plot(
        list(k_range), sil_scores, 'gs-',
        linewidth=2, markersize=9
    )
    axes[1].axvline(
        x=k_opt, color='red', linestyle='--',
        linewidth=2, label=f'k optimal = {k_opt}'
    )
    axes[1].set_xlabel('Nombre de clusters k', fontsize=12)
    axes[1].set_ylabel('Score Silhouette', fontsize=12)
    axes[1].set_title('Score Silhouette', fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(PATH_FIGURES, 'B_k_optimal.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Figure : {p}")

    return k_opt, inertias, sil_scores


# ════════════════════════════════════════════════════════
# PARTIE 4 — ENTRAÎNEMENT K-MEANS FINAL
# ════════════════════════════════════════════════════════

def entrainer_kmeans_final(X_norm, k):
    """
    Entraîne K-Means avec k-means++ et 20 runs.
    Analyse la stabilité sur 5 seeds différentes.
    """
    print("\n" + "═" * 55)
    print(f"  ENTRAÎNEMENT K-MEANS FINAL (k={k})")
    print("═" * 55)

    # Stabilité sur 5 seeds
    print("\n  Analyse de stabilité (5 runs)...")
    stab_scores = []
    for seed in range(5):
        km_tmp = KMeans(
            n_clusters=k, init='k-means++',
            n_init=10, random_state=seed
        )
        lbl_tmp = km_tmp.fit_predict(X_norm)
        sc      = silhouette_score(X_norm, lbl_tmp)
        stab_scores.append(sc)
        print(f"    Run {seed} → silhouette={sc:.4f}")

    print(f"  Silhouette moy : {np.mean(stab_scores):.4f} "
          f"± {np.std(stab_scores):.4f}")

    # Entraînement final
    print(f"\n  Entraînement final (n_init=20)...")
    km_final = KMeans(
        n_clusters=k, init='k-means++',
        n_init=20, max_iter=500,
        tol=1e-6, random_state=42
    )
    labels = km_final.fit_predict(X_norm)
    score  = silhouette_score(X_norm, labels)

    print(f"  Silhouette final : {score:.4f}")
    print(f"  Inertie finale   : {km_final.inertia_:.2f}")
    print(f"  Itérations       : {km_final.n_iter_}")

    print(f"\n  Distribution des clusters :")
    unique, counts = np.unique(labels, return_counts=True)
    for cid, cnt in zip(unique, counts):
        pct = cnt / len(labels) * 100
        bar = '█' * int(pct / 3)
        print(f"    Cluster {cid} : {cnt:5d} "
              f"({pct:5.1f}%) {bar}")

    # Sauvegarde
    np.save(
        os.path.join(PATH_EMBEDDINGS, 'labels_b.npy'),
        labels
    )
    return km_final, labels, score


# ════════════════════════════════════════════════════════
# PARTIE 5 — INTERPRÉTATION MÉTIER MINT/DGI
# ════════════════════════════════════════════════════════

def interpreter_clusters(labels, meta_list, k):
    """
    Analyse chaque cluster et l'interprète en termes
    de procédure MINT/DGI.
    """
    print("\n" + "═" * 55)
    print("  INTERPRÉTATION MÉTIER MINT/DGI")
    print("═" * 55)

    interp = {}

    for cid in range(k):
        idx_cluster = [
            i for i in range(len(labels))
            if i < len(meta_list) and labels[i] == cid
        ]
        if not idx_cluster:
            continue

        metas = [meta_list[i] for i in idx_cluster]
        cats  = [
            str(m.get('categorie', 'ILLISIBLE'))
            for m in metas
        ]
        confs = [
            float(m.get('confiance_ocr', 0))
            for m in metas
        ]
        lons  = [
            float(m.get('longueur', 0))
            for m in metas
        ]

        n_conf = cats.count('CONFORME')
        n_dout = cats.count('DOUTEUSE')
        n_ill  = cats.count('ILLISIBLE')
        total  = len(cats)

        conf_moy = np.mean(confs)
        lon_moy  = np.mean(lons)

        # Nommer le cluster selon sa composition
        if n_conf / total > 0.5:
            nom      = f"Cluster {cid} — Plaques CONFORMES"
            proc     = "Laisser passer"
            action   = "LAISSER_PASSER"
            risque   = "Faible"
            couleur  = "#2ecc71"
        elif n_ill / total > 0.5:
            nom      = f"Cluster {cid} — Plaques ILLISIBLES"
            proc     = "Mise en demeure MINT"
            action   = "CONTROLE_STANDARD"
            risque   = "Moyen"
            couleur  = "#e74c3c"
        elif n_dout / total > 0.4:
            if conf_moy > 0.4:
                nom    = f"Cluster {cid} — Plaques DOUTEUSES"
                proc   = "Signalement DGI — vignette"
                action = "SIGNALEMENT_DGI"
                risque = "Moyen-élevé"
                couleur = "#f39c12"
            else:
                nom    = f"Cluster {cid} — Plaques SUSPECTES"
                proc   = "Arrêt immédiat + contrôle PJ"
                action = "ARRET_SAISIE"
                risque = "Élevé"
                couleur = "#e67e22"
        else:
            nom      = f"Cluster {cid} — Plaques MIXTES"
            proc     = "Contrôle standard"
            action   = "CONTROLE_STANDARD"
            risque   = "Moyen"
            couleur  = "#3498db"

        interp[str(cid)] = {
            'cluster_id'  : cid,
            'nom'         : nom,
            'procedure'   : proc,
            'action_mdp'  : action,
            'risque'      : risque,
            'couleur'     : couleur,
            'n_total'     : total,
            'n_conformes' : n_conf,
            'n_douteuses' : n_dout,
            'n_illisibles': n_ill,
            'conf_moy'    : round(conf_moy, 3),
            'lon_moy'     : round(lon_moy, 1),
            'pct_conf'    : round(n_conf/total*100, 1),
            'pct_dout'    : round(n_dout/total*100, 1),
            'pct_ill'     : round(n_ill/total*100, 1),
        }

    # Affichage
    print("\n  ┌─────────────────────────────────────────┐")
    for cid, info in interp.items():
        print(f"  │ {info['nom']}")
        print(f"  │   Procédure MINT/DGI : {info['procedure']}")
        print(f"  │   Action MDP         : {info['action_mdp']}")
        print(f"  │   Risque             : {info['risque']}")
        print(f"  │   Taille             : {info['n_total']} plaques")
        print(f"  │   Conf OCR moy.      : {info['conf_moy']:.3f}")
        print(f"  │   Composition        : "
              f"CONF={info['pct_conf']}% | "
              f"DOUT={info['pct_dout']}% | "
              f"ILL={info['pct_ill']}%")
        print(f"  │")
    print("  └─────────────────────────────────────────┘")

    # Sauvegarder
    with open(
        os.path.join(PATH_EMBEDDINGS, 'B_interpretation.json'),
        'w', encoding='utf-8'
    ) as f:
        json.dump(interp, f, indent=2, ensure_ascii=False)

    return interp


# ════════════════════════════════════════════════════════
# PARTIE 6 — VISUALISATIONS PCA + t-SNE
# ════════════════════════════════════════════════════════

def visualiser_clusters(X_norm, labels, interp, k):
    """
    PCA 2D + t-SNE 2D avec légendes métier MINT/DGI.
    """
    print("\n" + "═" * 55)
    print("  VISUALISATION PCA + t-SNE")
    print("═" * 55)

    couleurs = [
        interp[str(cid)]['couleur']
        for cid in range(k)
        if str(cid) in interp
    ]
    noms = [
        interp[str(cid)]['nom']
        for cid in range(k)
        if str(cid) in interp
    ]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        'Module B — Clustering K-Means\n'
        'PlateVision MINT/DGI Cameroun',
        fontsize=14, fontweight='bold'
    )

    # ── PCA 2D ──
    print("\n  Calcul PCA 2D...")
    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_norm)
    var   = pca.explained_variance_ratio_

    for cid in range(k):
        if str(cid) not in interp:
            continue
        mask = labels == cid
        axes[0].scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=interp[str(cid)]['couleur'],
            label=interp[str(cid)]['nom'],
            alpha=0.6, s=20, edgecolors='none'
        )

    # Centroïdes projetés
    centroides_pca = pca.transform(
        np.array([
            X_norm[labels == cid].mean(axis=0)
            for cid in range(k)
        ])
    )
    axes[0].scatter(
        centroides_pca[:, 0], centroides_pca[:, 1],
        c='black', marker='X', s=200,
        zorder=5, label='Centroïdes'
    )

    axes[0].set_xlabel(
        f'PC1 ({var[0]*100:.1f}% variance)', fontsize=11
    )
    axes[0].set_ylabel(
        f'PC2 ({var[1]*100:.1f}% variance)', fontsize=11
    )
    axes[0].set_title('Projection PCA 2D',
                       fontweight='bold', fontsize=12)
    axes[0].legend(fontsize=7, loc='upper right')
    axes[0].grid(True, alpha=0.2)

    # ── t-SNE 2D ──
    print("  Calcul t-SNE 2D (quelques minutes)...")
    n_pca = min(20, X_norm.shape[1], X_norm.shape[0] - 1)
    X_20  = PCA(
        n_components=n_pca, random_state=42
    ).fit_transform(X_norm)
    perp  = min(30, len(X_norm) - 1)
    tsne  = TSNE(
        n_components=2, perplexity=perp,
        learning_rate=200, max_iter=1000,
        random_state=42, verbose=1
    )
    X_tsne = tsne.fit_transform(X_20)

    for cid in range(k):
        if str(cid) not in interp:
            continue
        mask = labels == cid
        axes[1].scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            c=interp[str(cid)]['couleur'],
            label=interp[str(cid)]['nom'],
            alpha=0.6, s=20, edgecolors='none'
        )

    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=11)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=11)
    axes[1].set_title('Projection t-SNE 2D',
                       fontweight='bold', fontsize=12)
    axes[1].legend(fontsize=7, loc='upper right')
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    p1 = os.path.join(PATH_FIGURES, 'B_pca_tsne.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Figure PCA+t-SNE : {p1}")

    # ── Diagramme silhouette ──
    print("  Génération diagramme silhouette...")
    fig2, ax = plt.subplots(figsize=(10, 7))
    sil_vals = silhouette_samples(X_norm, labels)
    sil_moy  = silhouette_score(X_norm, labels)
    y_lower  = 10

    for cid in range(k):
        vals = np.sort(sil_vals[labels == cid])
        y_upper = y_lower + len(vals)
        couleur = interp.get(
            str(cid), {}
        ).get('couleur', '#3498db')
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, vals,
            alpha=0.7, color=couleur,
            label=interp.get(str(cid), {}).get(
                'nom', f'C{cid}'
            )
        )
        ax.text(-0.05, y_lower + 0.5 * len(vals),
                str(cid), fontsize=10)
        y_lower = y_upper + 10

    ax.axvline(x=sil_moy, color='red', linestyle='--',
               label=f'Moy = {sil_moy:.3f}')
    ax.set_xlabel('Score Silhouette', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    ax.set_title(
        f'Diagramme Silhouette — k={k}\n'
        f'PlateVision MINT/DGI Cameroun',
        fontweight='bold', fontsize=12
    )
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    p2 = os.path.join(PATH_FIGURES, 'B_silhouette.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Figure silhouette : {p2}")

    # ── Heatmap composition des clusters ──
    comp_data = {
        'Cluster': [],
        'CONFORMES (%)': [],
        'DOUTEUSES (%)': [],
        'ILLISIBLES (%)': [],
    }
    for cid in range(k):
        if str(cid) not in interp:
            continue
        info = interp[str(cid)]
        comp_data['Cluster'].append(
            f"C{cid}\n{info['n_total']} plaques"
        )
        comp_data['CONFORMES (%)'].append(info['pct_conf'])
        comp_data['DOUTEUSES (%)'].append(info['pct_dout'])
        comp_data['ILLISIBLES (%)'].append(info['pct_ill'])

    df_comp = pd.DataFrame(comp_data).set_index('Cluster')

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        df_comp.T, annot=True, fmt='.1f',
        cmap='RdYlGn', ax=ax3,
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Pourcentage (%)'}
    )
    ax3.set_title(
        'Composition des Clusters — Module B\n'
        'PlateVision MINT/DGI Cameroun',
        fontweight='bold', fontsize=12
    )
    plt.tight_layout()
    p3 = os.path.join(PATH_FIGURES, 'B_composition.png')
    plt.savefig(p3, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Figure composition : {p3}")


# ════════════════════════════════════════════════════════
# PARTIE 7 — PRÉPARATION DES ÉTATS POUR MODULE C (MDP)
# ════════════════════════════════════════════════════════

def preparer_etats_mdp(labels, meta_list, interp, k):
    """
    Construit les états du MDP à partir des résultats
    du Module B.

    État = (cluster_id, ocr_level, cnn_alert)
    ocr_level : 0=faible(<0.4), 1=moyen(0.4-0.7), 2=élevé(>0.7)
    cnn_alert : 0=pas d'alerte, 1=alerte (ILLISIBLE ou suspect)
    """
    print("\n" + "═" * 55)
    print("  PRÉPARATION DES ÉTATS POUR MODULE C")
    print("═" * 55)

    etats_liste = []

    for i, (lbl, meta) in enumerate(
        zip(labels, meta_list)
    ):
        conf    = float(meta.get('confiance_ocr', 0))
        cat     = str(meta.get('categorie', 'ILLISIBLE'))

        # Discrétiser la confiance OCR en 3 niveaux
        if conf < 0.4:
            ocr_level = 0    # faible
        elif conf < 0.7:
            ocr_level = 1    # moyen
        else:
            ocr_level = 2    # élevé

        # Alerte CNN : 1 si la plaque est suspecte
        cnn_alert = 1 if cat == 'ILLISIBLE' else 0

        # Identifiant unique de l'état
        state_id = int(lbl) * 6 + ocr_level * 2 + cnn_alert

        etats_liste.append({
            'sample_index'    : i,
            'cluster_id'      : int(lbl),
            'ocr_level'       : ocr_level,
            'cnn_alert'       : cnn_alert,
            'state_id'        : state_id,
            'confiance_ocr'   : round(conf, 4),
            'categorie'       : cat,
            'texte_ocr'       : str(
                meta.get('texte_ocr', '')
            ),
            'cluster_nom'     : interp.get(
                str(int(lbl)), {}
            ).get('nom', f'C{lbl}'),
            'action_recommandee': interp.get(
                str(int(lbl)), {}
            ).get('action_mdp', 'CONTROLE_STANDARD'),
        })

    df_etats = pd.DataFrame(etats_liste)

    # Statistiques
    print(f"\n  États générés : {len(df_etats):,}")
    print(f"  États uniques : {df_etats['state_id'].nunique()}")
    print(f"\n  Distribution des états :")
    for sid in sorted(df_etats['state_id'].unique()):
        n   = (df_etats['state_id'] == sid).sum()
        pct = n / len(df_etats) * 100
        ex  = df_etats[
            df_etats['state_id'] == sid
        ].iloc[0]
        print(f"    État {sid:2d} "
              f"(C{ex['cluster_id']}, "
              f"OCR={ex['ocr_level']}, "
              f"A={ex['cnn_alert']}) : "
              f"{n:5d} ({pct:5.1f}%)")

    # Sauvegarder
    path_etats = os.path.join(
        PATH_EMBEDDINGS, 'B_etats_mdp.csv'
    )
    df_etats.to_csv(path_etats, index=False,
                    encoding='utf-8')
    print(f"\n  ✓ États MDP sauvegardés : {path_etats}")
    print(f"  → Prêt pour le Module C (MDP)")

    # Figure distribution des états
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'États MDP issus du Module B\n'
        'PlateVision MINT/DGI Cameroun',
        fontsize=13, fontweight='bold'
    )

    # Distribution par état
    state_counts = df_etats['state_id'].value_counts(
    ).sort_index()
    axes[0].bar(
        state_counts.index.astype(str),
        state_counts.values,
        color='steelblue', alpha=0.8,
        edgecolor='white'
    )
    axes[0].set_xlabel('État MDP (state_id)')
    axes[0].set_ylabel('Nombre de plaques')
    axes[0].set_title('Distribution des états MDP')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Distribution par action recommandée
    action_counts = df_etats[
        'action_recommandee'
    ].value_counts()
    colors_action = ['#2ecc71', '#f39c12',
                     '#e74c3c', '#3498db', '#9b59b6']
    axes[1].pie(
        action_counts.values,
        labels=action_counts.index,
        colors=colors_action[:len(action_counts)],
        autopct='%1.1f%%',
        startangle=90
    )
    axes[1].set_title('Actions recommandées')

    plt.tight_layout()
    p = os.path.join(PATH_FIGURES, 'B_etats_mdp.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Figure états MDP : {p}")

    return df_etats


# ════════════════════════════════════════════════════════
# MAIN — EXÉCUTION MODULE B COMPLET
# ════════════════════════════════════════════════════════

if __name__ == "__main__":

    import easyocr
    import torch

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "█" * 55)
    print("  MODULE B — CLUSTERING COMPLET")
    print(f"  Device : {DEVICE.upper()}")
    print("█" * 55)

    # ── Chargement des annotations ──
    print("\n  Chargement des annotations...")
    df_train = charger_annotations(TRAIN_CSV, TRAIN_PATH)
    df_test  = charger_annotations(TEST_CSV,  TEST_PATH)
    df_valid = charger_annotations(VALID_CSV, VALID_PATH)

    # ── OCR complet ──
    # Vérifier si l'OCR complet existe déjà
    csv_complet = os.path.join(
        PATH_OCR_OUT, 'ocr_complet_all.csv'
    )
    if os.path.exists(csv_complet):
        print(f"\n  OCR complet déjà existant — chargement...")
        df_ocr = pd.read_csv(csv_complet)
        for col in ['lettres', 'chiffres',
                    'symboles', 'texte_ocr']:
            df_ocr[col] = df_ocr[col].fillna('')
        print(f"  ✓ {len(df_ocr):,} plaques chargées")

        reponse = input(
            "\n  Relancer l'OCR complet ? (o/n) : "
        ).strip().lower()
        if reponse == 'o':
            print("\n  Init EasyOCR...")
            reader = easyocr.Reader(
                ['en'],
                gpu=(DEVICE == 'cuda'),
                verbose=False
            )
            df_ocr = run_ocr_complet(
                df_train, df_test, df_valid, reader
            )
    else:
        print("\n  Lancement OCR complet...")
        print("  Init EasyOCR (peut prendre 1-2 min)...")
        reader = easyocr.Reader(
            ['en'],
            gpu=(DEVICE == 'cuda'),
            verbose=False
        )
        df_ocr = run_ocr_complet(
            df_train, df_test, df_valid, reader
        )

    # ── Extraction des features ──
    X, meta = extraire_features_clustering(df_ocr)

    # ── Normalisation L2 ──
    print("\n  Normalisation L2 des features...")
    X_norm = normalize(X, norm='l2')
    print(f"  ✓ Shape normalisé : {X_norm.shape}")

    # ── K optimal ──
    k_opt, inertias, sil_scores = choisir_k_optimal(
        X_norm, k_max=8
    )

    # ── Entraînement K-Means ──
    km, labels, score = entrainer_kmeans_final(
        X_norm, k_opt
    )

    # ── Interprétation métier ──
    interp = interpreter_clusters(labels, meta, k_opt)

    # ── Visualisations ──
    visualiser_clusters(X_norm, labels, interp, k_opt)

    # ── Préparation états pour Module C ──
    df_etats = preparer_etats_mdp(
        labels, meta, interp, k_opt
    )

    print("\n\n" + "✅ " * 18)
    print("  MODULE B TERMINÉ !")
    print(f"  Features    : {PATH_EMBEDDINGS}/features_b.npy")
    print(f"  Labels      : {PATH_EMBEDDINGS}/labels_b.npy")
    print(f"  Interprét.  : {PATH_EMBEDDINGS}/B_interpretation.json")
    print(f"  États MDP   : {PATH_EMBEDDINGS}/B_etats_mdp.csv")
    print(f"  Figures     : {PATH_FIGURES}/B_*.png")
    print("  ")
    print("✅ " * 18)