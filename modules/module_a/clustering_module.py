"""
PlateVision — Module B — Clustering
Regroupement non supervisé des plaques
MINT/DGI Cameroun — UCAC-ICAM
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

PATH_EMBEDDINGS = "data/processed/embeddings"
PATH_FIGURES    = "outputs/figures"

os.makedirs(PATH_EMBEDDINGS, exist_ok=True)
os.makedirs(PATH_FIGURES, exist_ok=True)


# ════════════════════════════════════════════════
# EXTRACTION DES FEATURES
# ════════════════════════════════════════════════

def extraire_features(df_ocr):
    """
    Extrait les features visuelles + OCR de chaque plaque.
    Ces features servent d'entrée au K-Means.
    """
    print("\n" + "─" * 55)
    print("  CLUSTERING — EXTRACTION DES FEATURES")
    print("─" * 55)

    features_list = []
    meta_list     = []

    print(f"\n  {len(df_ocr)} plaques à traiter...")

    for _, row in tqdm(
        df_ocr.iterrows(),
        total=len(df_ocr),
        desc="  Features"
    ):
        plaque_path = row.get('plaque_path', '')
        if not plaque_path or not os.path.exists(plaque_path):
            continue

        img = cv2.imread(plaque_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        feats = []

        # ── Features visuelles ──
        # 1. Luminosité normalisée
        feats.append(float(np.mean(gray)) / 255.0)
        # 2. Contraste normalisé
        feats.append(float(np.std(gray)) / 255.0)
        # 3. Netteté (variance Laplacien)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        feats.append(float(np.clip(lap / 1000.0, 0, 1)))
        # 4. Ratio largeur/hauteur
        feats.append(float(w / h) if h > 0 else 1.0)
        # 5. Histogramme 8 bins normalisé
        hist = cv2.calcHist(
            [gray], [0], None, [8], [0, 256]
        ).flatten()
        hist_n = hist / hist.sum() if hist.sum() > 0 else hist
        feats.extend(hist_n.tolist())
        # 6. Densité pixels sombres (texte)
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        feats.append(
            float(np.sum(binary == 255)) / (h * w)
        )
        # 7+8. Gradients Sobel
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        feats.append(float(np.mean(np.abs(sx))) / 255.0)
        feats.append(float(np.mean(np.abs(sy))) / 255.0)

        # ── Features OCR ──
        # 9. Confiance OCR
        feats.append(float(row.get('confiance_ocr', 0)))
        # 10. Longueur texte normalisée
        feats.append(
            min(float(row.get('longueur', 0)) / 12.0, 1.0)
        )
        # 11+12. Ratio lettres / chiffres
        n_l = float(row.get('n_lettres', 0))
        n_c = float(row.get('n_chiffres', 0))
        tot = n_l + n_c
        feats.append(n_l / tot if tot > 0 else 0.5)
        feats.append(n_c / tot if tot > 0 else 0.5)
        # 13. Présence symboles
        feats.append(
            1.0 if float(row.get('n_symboles', 0)) > 0
            else 0.0
        )
        # 14. Catégorie OCR encodée
        feats.append({
            'CONFORME' : 1.0,
            'DOUTEUSE' : 0.5,
            'ILLISIBLE': 0.0
        }.get(row.get('categorie', 'ILLISIBLE'), 0.0))

        features_list.append(feats)
        meta_list.append(row.to_dict())

    X = np.array(features_list, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

    np.save(
        os.path.join(PATH_EMBEDDINGS, 'features.npy'), X
    )
    print(f"\n  ✓ Shape features : {X.shape}")
    print(f"  ✓ Sauvegardé : {PATH_EMBEDDINGS}/features.npy")
    return X, meta_list


# ════════════════════════════════════════════════
# SÉLECTION DU K OPTIMAL
# ════════════════════════════════════════════════

def choisir_k(X_norm, k_max=8):
    """
    Méthode du coude + score silhouette pour choisir k.
    """
    print("\n  Recherche du k optimal (coude + silhouette)...")

    inertias, sil_scores = [], []
    k_range = range(2, k_max + 1)

    for k in tqdm(k_range, desc="  Test k"):
        km     = KMeans(n_clusters=k, init='k-means++',
                        n_init=10, random_state=42)
        labels = km.fit_predict(X_norm)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_norm, labels))

    k_opt = list(k_range)[np.argmax(sil_scores)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'Sélection k Optimal — K-Means\nPlateVision MINT/DGI',
        fontsize=13, fontweight='bold'
    )
    axes[0].plot(list(k_range), inertias, 'bo-',
                 linewidth=2, markersize=8)
    axes[0].axvline(x=k_opt, color='red', linestyle='--',
                    label=f'k={k_opt}')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Inertie')
    axes[0].set_title('Méthode du Coude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(k_range), sil_scores, 'gs-',
                 linewidth=2, markersize=8)
    axes[1].axvline(x=k_opt, color='red', linestyle='--',
                    label=f'k={k_opt}')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Score Silhouette')
    axes[1].set_title('Score Silhouette')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(PATH_FIGURES, 'clustering_k_optimal.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ k optimal : {k_opt} "
          f"(silhouette={max(sil_scores):.4f})")
    print(f"  ✓ Figure : {p}")
    return k_opt


# ════════════════════════════════════════════════
# ENTRAÎNEMENT K-MEANS
# ════════════════════════════════════════════════

def entrainer_kmeans(X_norm, k):
    """Entraîne K-Means avec k-means++ et 20 runs."""
    print(f"\n  Entraînement K-Means (k={k})...")

    km = KMeans(
        n_clusters=k, init='k-means++',
        n_init=20, max_iter=500,
        random_state=42
    )
    labels = km.fit_predict(X_norm)
    score  = silhouette_score(X_norm, labels)

    print(f"  Silhouette : {score:.4f}")
    print(f"  Inertie    : {km.inertia_:.2f}")
    print(f"  Itérations : {km.n_iter_}")
    print(f"\n  Distribution :")

    unique, counts = np.unique(labels, return_counts=True)
    for cid, cnt in zip(unique, counts):
        pct = cnt / len(labels) * 100
        bar = '█' * int(pct / 3)
        print(f"    Cluster {cid} : {cnt:4d} "
              f"({pct:5.1f}%) {bar}")

    np.save(
        os.path.join(PATH_EMBEDDINGS, 'cluster_labels.npy'),
        labels
    )
    return km, labels


# ════════════════════════════════════════════════
# INTERPRÉTATION MÉTIER + VISUALISATION
# ════════════════════════════════════════════════

def interpreter_et_visualiser(X_norm, labels,
                               meta_list, k):
    """
    Nomme chaque cluster selon sa composition OCR.
    Génère PCA 2D et t-SNE 2D.
    """
    print("\n  Interprétation métier des clusters...")

    interp = {}
    for cid in range(k):
        idx_cluster = [
            i for i in range(len(labels))
            if i < len(meta_list) and labels[i] == cid
        ]
        if not idx_cluster:
            continue

        cats = [
            meta_list[i].get('categorie', 'ILLISIBLE')
            for i in idx_cluster
        ]
        n_conf = cats.count('CONFORME')
        n_dout = cats.count('DOUTEUSE')
        n_ill  = cats.count('ILLISIBLE')
        total  = len(cats)

        if n_conf / total > 0.6:
            nom  = f"C{cid} — Plaques Conformes"
            proc = "Laisser passer"
        elif n_ill / total > 0.6:
            nom  = f"C{cid} — Plaques Illisibles"
            proc = "Mise en demeure MINT"
        elif n_dout / total > 0.4:
            nom  = f"C{cid} — Plaques Douteuses"
            proc = "Contrôle standard DGI"
        else:
            nom  = f"C{cid} — Plaques Mixtes"
            proc = "Contrôle standard"

        interp[cid] = {
            'nom'      : nom,
            'procedure': proc,
            'n_total'  : total,
            'n_conf'   : n_conf,
            'n_dout'   : n_dout,
            'n_ill'    : n_ill,
        }

    print("\n  ┌─ Interprétation MINT/DGI ─────────────┐")
    for cid, info in interp.items():
        print(f"  │ {info['nom']}")
        print(f"  │   Procédure : {info['procedure']}")
        print(f"  │   Conf:{info['n_conf']} | "
              f"Dout:{info['n_dout']} | "
              f"Ill:{info['n_ill']}")
        print(f"  │")
    print("  └───────────────────────────────────────┘")

    colors_map = plt.cm.Set1(np.linspace(0, 1, k))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        'Clustering K-Means — PCA & t-SNE\n'
        'PlateVision MINT/DGI Cameroun',
        fontsize=14, fontweight='bold'
    )

    # PCA 2D
    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_norm)
    var   = pca.explained_variance_ratio_

    for cid in range(k):
        mask = labels == cid
        nom  = interp.get(cid, {}).get('nom', f'C{cid}')
        axes[0].scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=[colors_map[cid]], label=nom,
            alpha=0.6, s=25
        )
    axes[0].set_xlabel(f'PC1 ({var[0]*100:.1f}% var)')
    axes[0].set_ylabel(f'PC2 ({var[1]*100:.1f}% var)')
    axes[0].set_title('PCA 2D', fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2)

    # t-SNE 2D
    print("\n  Calcul t-SNE (1-2 min)...")

    n_pca = min(20, X_norm.shape[1], X_norm.shape[0] - 1)

    X_20 = PCA(
        n_components=n_pca,
        random_state=42
    ).fit_transform(X_norm)

    perp = min(30, len(X_norm) - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate=200,
        max_iter=1000,
        random_state=42,
        verbose=0
    )

    X_tsne = tsne.fit_transform(X_20)

    for cid in range(k):
        mask = labels == cid
        nom  = interp.get(cid, {}).get('nom', f'C{cid}')
        axes[1].scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            c=[colors_map[cid]], label=nom,
            alpha=0.6, s=25
        )
    axes[1].set_xlabel('t-SNE Dim 1')
    axes[1].set_ylabel('t-SNE Dim 2')
    axes[1].set_title('t-SNE 2D', fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    p = os.path.join(PATH_FIGURES, 'clustering_pca_tsne.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Figure : {p}")

    with open(os.path.join(PATH_EMBEDDINGS,
                            'interpretation.json'), 'w') as f:
        json.dump(interp, f, indent=2, ensure_ascii=False)

    return interp