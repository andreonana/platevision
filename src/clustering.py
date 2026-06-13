# glob sert à lister tous les fichiers d'un dossier selon un motif (*.jpg)
import glob
# numpy : calcul sur des tableaux de nombres (les features, les matrices)
import numpy as np
# cv2 (OpenCV) : lire et transformer les images
import cv2
# matplotlib : dessiner et sauvegarder le graphique des clusters
import matplotlib.pyplot as plt
# StandardScaler : met toutes les features à la même échelle (normalisation)
from sklearn.preprocessing import StandardScaler
# KMeans : l'algorithme de clustering qui regroupe les plaques
from sklearn.cluster import KMeans
# PCA : réduit les features à 2 dimensions pour pouvoir les dessiner
from sklearn.decomposition import PCA
# silhouette_score : mesure la qualité du regroupement (entre -1 et 1)
from sklearn.metrics import silhouette_score

# Dossier où l'usine a rangé les plaques découpées
CROPS_DIR = "data/processed/plate_crops"
# Taille commune imposée à toutes les plaques (largeur, hauteur)
TAILLE = (200, 60)

# ── 1. Fonction qui transforme UNE image de plaque en une liste de nombres ──
def extraire_features(chemin):
    # Lit l'image depuis le disque
    img = cv2.imread(chemin)
    # Redimensionne à la taille commune (pour comparer des plaques comparables)
    img = cv2.resize(img, TAILLE)
    # Convertit en niveaux de gris (on ignore la couleur ici)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Feature A : histogramme des niveaux de gris en 32 cases
    # = répartition des zones claires et foncées de la plaque
    hist = cv2.calcHist([gris], [0], None, [32], [0, 256]).flatten()
    # On normalise l'histogramme (somme = 1) pour ne pas dépendre de la taille
    hist = hist / (hist.sum() + 1e-6)

    # Feature B : netteté = variance du Laplacien
    # Un chiffre élevé = plaque nette ; bas = plaque floue/dégradée
    nettete = cv2.Laplacian(gris, cv2.CV_64F).var()

    # Feature C : luminosité moyenne de la plaque (claire ou sombre)
    luminosite = gris.mean()

    # Feature D : proportion de pixels foncés (densité des caractères/encre)
    ratio_fonce = (gris < 100).sum() / gris.size

    # On colle toutes les features bout à bout en un seul vecteur
    return np.concatenate([hist, [nettete, luminosite, ratio_fonce]])

# ── 2. Construire la matrice de features de TOUTES les plaques ──────────────
# Liste tous les fichiers .jpg du dossier des crops
fichiers = glob.glob(f"{CROPS_DIR}/*.jpg")
# Affiche combien de plaques on a trouvées
print(f"{len(fichiers)} plaques trouvees")

# Applique extraire_features() à chaque fichier → tableau (1 ligne par plaque)
X = np.array([extraire_features(f) for f in fichiers])
# Normalise toutes les colonnes pour qu'aucune feature n'écrase les autres
X = StandardScaler().fit_transform(X)

# ── 3. Trouver le meilleur nombre de clusters k via le score silhouette ─────
print("\nRecherche du meilleur k :")
# On teste k de 2 à 6 et on regarde lequel sépare le mieux les groupes
for k in range(2, 7):
    # Entraîne un K-Means avec k clusters (random_state=42 = résultat reproductible)
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    # Calcule la qualité du regroupement
    score = silhouette_score(X, km.labels_)
    # Affiche le score pour ce k (plus c'est haut, mieux c'est)
    print(f"  k={k} : silhouette = {score:.3f}")

# ── 4. Clustering final ─────────────────────────────────────────────────────
# Choisis ici le k qui avait le meilleur score ci-dessus (à ajuster)
K = 3
# Entraîne le K-Means final avec ce nombre de clusters
km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X)
# labels[i] = numéro du cluster attribué à la plaque i
labels = km.labels_

# Affiche combien de plaques sont tombées dans chaque cluster
for c in range(K):
    print(f"Cluster {c} : {(labels == c).sum()} plaques")

# ── 5. Visualiser les clusters en 2D ────────────────────────────────────────
# PCA réduit nos nombreuses features à seulement 2 dimensions (pour le dessin)
X_2d = PCA(n_components=2).fit_transform(X)
# Prépare une figure de 8x6 pouces
plt.figure(figsize=(8, 6))
# Dessine chaque plaque comme un point, coloré selon son cluster
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=10)
# Titre et légendes des axes
plt.title(f"Clustering des plaques (K-Means, k={K})")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
# Sauvegarde l'image dans le dossier outputs
plt.savefig("outputs/clusters.png", dpi=120)
# Confirme que c'est enregistré
print("\nFigure sauvegardee : outputs/clusters.png")