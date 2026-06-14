# ── src/train_cnn.py ───────────────────────────────────────────────────
import os                                          # gérer les chemins/dossiers
import torch                                        # PyTorch : le moteur du réseau
import torch.nn as nn                               # briques pour construire le réseau
from torch.utils.data import DataLoader, random_split   # charger et découper les données
from torchvision import datasets, transforms        # lire les images rangées par dossier

DATA_DIR = "data/processed/characters_labeled"       # caractères étiquetés (un dossier par lettre)
MODEL_OUT = "models/cnn_chars.pt"                    # où sauvegarder le CNN entraîné
EPOCHS = 15                                          # nb de passages sur les données
BATCH = 64                                           # nb d'images traitées en même temps
DEVICE = "cpu"                                        # on est sur CPU (pas de GPU)

# ── 1. Transformation appliquée à chaque image ──
transform = transforms.Compose([                     # enchaîne plusieurs transformations
    transforms.Grayscale(num_output_channels=1),     # forcer en niveaux de gris (1 canal)
    transforms.Resize((28, 28)),                     # taille fixe 28x28
    transforms.ToTensor(),                           # convertir l'image en tenseur (nombres)
])

# ── 2. Charger les données (ImageFolder lit un dossier par classe) ──
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)  # charge tout le dossier
classes = dataset.classes                            # liste des lettres (0,1,...,A,B,...)
num_classes = len(classes)                           # nombre de classes
print(f"{len(dataset)} caracteres, {num_classes} classes : {classes}")

# Garde-fou : besoin d'au moins 2 classes et d'assez de données
if num_classes < 2 or len(dataset) < 100:
    print("Pas assez de donnees pour entrainer le CNN.")
    raise SystemExit                                 # arrête proprement

# ── 3. Découper en entraînement (80%) et validation (20%) ──
n_val = int(0.2 * len(dataset))                      # taille du lot de validation
n_train = len(dataset) - n_val                       # le reste pour l'entraînement
train_set, val_set = random_split(dataset, [n_train, n_val])  # découpage aléatoire

train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)  # paquets mélangés
val_loader = DataLoader(val_set, batch_size=BATCH)                    # paquets validation

# ── 4. Architecture du CNN ──
class CharCNN(nn.Module):                            # notre réseau hérite de nn.Module
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # 1 canal -> 16 filtres, noyau 3x3
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # 16 -> 32 filtres
        self.pool = nn.MaxPool2d(2, 2)               # réduit la taille de moitié
        self.relu = nn.ReLU()                        # activation (garde l'utile)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)        # couche dense -> 128 valeurs = EMBEDDING
        self.fc2 = nn.Linear(128, num_classes)       # couche finale -> une sortie par lettre

    def forward(self, x, return_embedding=False):    # trajet des données dans le réseau
        x = self.pool(self.relu(self.conv1(x)))      # conv1 -> relu -> pool (28->14)
        x = self.pool(self.relu(self.conv2(x)))      # conv2 -> relu -> pool (14->7)
        x = x.view(x.size(0), -1)                    # aplatir en un vecteur
        emb = self.relu(self.fc1(x))                 # EMBEDDING : résumé de 128 nombres
        if return_embedding:                         # si on veut juste l'embedding (Module B)
            return emb
        return self.fc2(emb)                         # sinon, prédiction finale

model = CharCNN(num_classes).to(DEVICE)              # créer le modèle sur CPU

# ── 5. Outils d'entraînement ──
criterion = nn.CrossEntropyLoss()                    # mesure l'erreur de classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # ajuste les poids

# ── 6. Boucle d'entraînement ──
for epoch in range(EPOCHS):                          # pour chaque passage complet
    model.train()                                    # mode entraînement
    for images, labels in train_loader:              # par paquets
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()                        # remettre les gradients à zéro
        sorties = model(images)                      # prédictions
        perte = criterion(sorties, labels)           # erreur
        perte.backward()                             # calculer les corrections
        optimizer.step()                             # appliquer les corrections

    model.eval()                                     # mode évaluation
    correct = total = 0
    with torch.no_grad():                            # pas d'apprentissage, juste tester
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(1)          # classe prédite = score le plus haut
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total                      # précision en %
    print(f"Epoch {epoch+1}/{EPOCHS} - precision validation : {acc:.1f}%")

# ── 7. Sauvegarder le modèle ET la liste des classes ──
torch.save({"model": model.state_dict(), "classes": classes}, MODEL_OUT)  # enregistrer
print(f"\nCNN sauvegarde : {MODEL_OUT}")