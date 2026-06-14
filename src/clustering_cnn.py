# ── src/clustering_cnn.py ──────────────────────────────────────────────
import glob                                          # lister les fichiers
import numpy as np                                   # calcul sur tableaux
import torch                                          # PyTorch (charger le CNN)
import torch.nn as nn                                 # briques du réseau
import cv2                                            # lire les images
import matplotlib.pyplot as plt                       # dessiner les clusters
from sklearn.preprocessing import StandardScaler      # normaliser les embeddings
from sklearn.cluster import KMeans                    # clustering
from sklearn.decomposition import PCA                 # réduire à 2D pour visualiser
from sklearn.metrics import silhouette_score          # mesurer la qualité des clusters

CROPS_DIR = "data/processed/plate_crops"              # plaques découpées
MODEL_PATH = "models/cnn_chars.pt"                    # le CNN qu'on vient d'entraîner
DEVICE = "cpu"                                         # pas de GPU
MAX_PLATES = 1500                                     # nb de plaques à regrouper

# ── 1. Re-définir EXACTEMENT la même architecture CNN qu'à l'entraînement ──
class CharCNN(nn.Module):                             # identique à train_cnn.py
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)         # la couche EMBEDDING (128 nombres)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, return_embedding=False):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        emb = self.relu(self.fc1(x))                  # l'embedding
        if return_embedding:                          # mode "donne-moi l'embedding"
            return emb
        return self.fc2(emb)

# ── 2. Charger le CNN sauvegardé ──
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)  # charger le fichier .pt
classes = checkpoint["classes"]                       # récupérer la liste des classes
model = CharCNN(len(classes)).to(DEVICE)              # recréer le modèle
model.load_state_dict(checkpoint["model"])            # y charger les poids appris
model.eval()                                          # mode évaluation (pas d'apprentissage)
print(f"CNN charge ({len(classes)} classes).")

# ── 3. Extraire l'embedding de chaque plaque ──
def embedding_de(chemin):
    img = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)    # lire en niveaux de gris
    if img is None:
        return None
    img = cv2.resize(img, (28, 28))                   # même taille que l'entraînement
    t = torch.tensor(img, dtype=torch.float32) / 255  # convertir en tenseur, normaliser 0-1
    t = t.unsqueeze(0).unsqueeze(0)                   # ajouter dimensions (batch, canal)
    with torch.no_grad():                             # pas de calcul de gradient
        emb = model(t, return_embedding=True)         # demander l'embedding
    return emb.squeeze().numpy()                      # convertir en tableau numpy

fichiers = glob.glob(f"{CROPS_DIR}/*.jpg") + glob.glob(f"{CROPS_DIR}/*.png")
fichiers = fichiers[:MAX_PLATES]                      # limiter
print(f"Extraction des embeddings de {len(fichiers)} plaques...")

embeddings = []                                       # liste des embeddings
for f in fichiers:
    e = embedding_de(f)                               # embedding de cette plaque
    if e is not None:
        embeddings.append(e)
X = np.array(embeddings)                              # matrice (1 ligne par plaque, 128 colonnes)
X = StandardScaler().fit_transform(X)                 # normaliser
print(f"{X.shape[0]} embeddings extraits, dimension {X.shape[1]}")

# ── 4. Chercher le meilleur nombre de clusters k ──
print("\nRecherche du meilleur k :")
for k in range(2, 7):                                 # tester k de 2 à 6
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    score = silhouette_score(X, km.labels_)           # qualité des clusters
    print(f"  k={k} : silhouette = {score:.3f}")

# ── 5. Clustering final (ajuste K selon le meilleur score ci-dessus) ──
K = 4                                                # à ajuster après avoir vu les scores
km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X)
labels = km.labels_                                   # cluster de chaque plaque
for c in range(K):
    print(f"Cluster {c} : {(labels == c).sum()} plaques")

# ── 6. Visualiser en 2D avec PCA ──
X_2d = PCA(n_components=2).fit_transform(X)           # réduire à 2 dimensions
plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=10)  # un point par plaque
plt.title(f"Clustering des plaques sur embeddings CNN (k={K})")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.savefig("outputs/clusters_cnn.png", dpi=120)      # sauvegarder l'image
print("\nFigure sauvegardee : outputs/clusters_cnn.png")