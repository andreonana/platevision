# ── src/inspect_clusters.py ────────────────────────────────────────────
import glob                                          # lister les fichiers
import numpy as np                                   # calcul sur tableaux
import torch                                          # charger le CNN
import torch.nn as nn                                 # briques du réseau
import cv2                                            # lire les images
import matplotlib.pyplot as plt                       # afficher la planche
from sklearn.preprocessing import StandardScaler      # normaliser les embeddings
from sklearn.cluster import KMeans                    # clustering

CROPS_DIR = "data/processed/plate_crops"              # plaques découpées
MODEL_PATH = "models/cnn_chars.pt"                    # le CNN entraîné
DEVICE = "cpu"
MAX_PLATES = 1500                                     # même nombre que le clustering
K = 4                                                 # même k que le clustering
N_EXEMPLES = 6                                        # nb de plaques montrées par cluster

# ── Architecture identique à l'entraînement ──
class CharCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x, return_embedding=False):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        emb = self.relu(self.fc1(x))
        if return_embedding:
            return emb
        return self.fc2(emb)

# ── Charger le CNN ──
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)  # charger
classes = ckpt["classes"]
model = CharCNN(len(classes)).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

# ── Extraire les embeddings (en gardant le chemin de chaque plaque) ──
def embedding_de(chemin):
    img = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (28, 28))
    t = torch.tensor(img, dtype=torch.float32) / 255
    t = t.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        return model(t, return_embedding=True).squeeze().numpy()

fichiers = (glob.glob(f"{CROPS_DIR}/*.jpg") + glob.glob(f"{CROPS_DIR}/*.png"))[:MAX_PLATES]
embs, chemins = [], []                                # embeddings + chemins gardés ensemble
for f in fichiers:
    e = embedding_de(f)
    if e is not None:
        embs.append(e); chemins.append(f)
X = StandardScaler().fit_transform(np.array(embs))    # normaliser

# ── Refaire le clustering (mêmes réglages) ──
labels = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(X)

# ── Construire la planche : une ligne par cluster, N exemples par ligne ──
fig, axes = plt.subplots(K, N_EXEMPLES, figsize=(N_EXEMPLES * 2, K * 1.6))
for c in range(K):                                    # pour chaque cluster
    indices = np.where(labels == c)[0][:N_EXEMPLES]   # prendre N plaques de ce cluster
    for j in range(N_EXEMPLES):                       # pour chaque case de la ligne
        ax = axes[c, j]
        ax.axis("off")                                # pas d'axes
        if j < len(indices):
            img = cv2.imread(chemins[indices[j]])     # lire la plaque
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # couleurs correctes
            ax.imshow(img)
        if j == 0:                                    # étiqueter la 1re case de la ligne
            ax.set_title(f"Cluster {c}", loc="left", fontsize=11)

plt.tight_layout()                                    # ajuster l'espacement
plt.savefig("outputs/clusters_exemples.png", dpi=120) # sauvegarder la planche
print("Planche sauvegardee : outputs/clusters_exemples.png")