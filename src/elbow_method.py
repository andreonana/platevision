# ── src/elbow_method.py ────────────────────────────────────────────────
import glob                                          # lister les fichiers
import numpy as np                                   # calcul sur tableaux
import torch                                          # charger le CNN
import torch.nn as nn                                 # briques du réseau
import cv2                                            # lire les images
import matplotlib.pyplot as plt                       # tracer la courbe
from sklearn.preprocessing import StandardScaler      # normaliser les embeddings
from sklearn.cluster import KMeans                    # clustering

CROPS_DIR = "data/processed/plate_crops"              # plaques découpées
MODEL_PATH = "models/cnn_chars.pt"                    # le CNN entraîné
DEVICE = "cpu"
MAX_PLATES = 1500                                     # même nombre que le clustering

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
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)  # charger le .pt
model = CharCNN(len(ckpt["classes"])).to(DEVICE)      # recréer le modèle
model.load_state_dict(ckpt["model"])                  # charger les poids
model.eval()                                          # mode évaluation

# ── Extraire les embeddings ──
def embedding_de(chemin):
    img = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)    # lire en gris
    if img is None:
        return None
    img = cv2.resize(img, (28, 28))                   # taille d'entraînement
    t = torch.tensor(img, dtype=torch.float32) / 255  # tenseur normalisé
    t = t.unsqueeze(0).unsqueeze(0)                   # ajouter dimensions
    with torch.no_grad():
        return model(t, return_embedding=True).squeeze().numpy()  # l'embedding

fichiers = (glob.glob(f"{CROPS_DIR}/*.jpg") + glob.glob(f"{CROPS_DIR}/*.png"))[:MAX_PLATES]
embs = [e for f in fichiers if (e := embedding_de(f)) is not None]  # tous les embeddings
X = StandardScaler().fit_transform(np.array(embs))    # normaliser
print(f"{X.shape[0]} embeddings extraits.")

# ── Méthode du coude : inertie pour chaque k ──
ks = range(1, 9)                                      # on teste k de 1 à 8
inerties = []                                         # liste des inerties
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)  # clustering
    inerties.append(km.inertia_)                      # inertie = somme des distances au centre
    print(f"  k={k} : inertie = {km.inertia_:.0f}")   # afficher

# ── Tracer la courbe du coude ──
plt.figure(figsize=(8, 5))
plt.plot(list(ks), inerties, "o-")                    # courbe avec points
plt.xlabel("Nombre de clusters k")                    # axe X
plt.ylabel("Inertie (compacite des clusters)")        # axe Y
plt.title("Methode du coude")                         # titre
plt.grid(True, alpha=0.3)                              # grille légère
plt.savefig("outputs/elbow.png", dpi=120)             # sauvegarder
print("\nCourbe sauvegardee : outputs/elbow.png")