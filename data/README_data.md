# PlateVision — Guide des données et du pipeline de préparation

**Projet** : Reconnaissance automatique de plaques d'immatriculation africaines
**Institution** : UCAC-ICAM / ULC-ICAM — MINT/DGI Cameroun
**Auteur** : [@andreonana](https://github.com/andreonana)

---

## Sommaire

1. [Vue d'ensemble](#1-vue-densemble)
2. [Datasets sources](#2-datasets-sources)
3. [Structure des dossiers](#3-structure-des-dossiers)
4. [Installation des dépendances](#4-installation-des-dépendances)
5. [Exécution du pipeline](#5-exécution-du-pipeline)
6. [Les 7 phases expliquées](#6-les-7-phases-expliquées)
7. [Fichiers produits](#7-fichiers-produits)
8. [Ce que consomme chaque module IA](#8-ce-que-consomme-chaque-module-ia)
9. [Biais connus et limitations](#9-biais-connus-et-limitations)

---

## 1. Vue d'ensemble

Le script `data/prepare_datasets.py` est le **point d'entrée unique** pour transformer les images brutes de plaques d'immatriculation en données prêtes à l'emploi pour les quatre modules IA du projet PlateVision.

```
Données brutes (Roboflow + Mendeley)
          │
          ▼
  prepare_datasets.py
  ┌───────────────────────────────────────────────┐
  │  Phase 1 → Acquisition et unification         │
  │  Phase 2 → EDA (exploration rapide)           │
  │  Phase 3 → Extraction des régions de plaques  │
  │  Phase 4 → Segmentation des caractères        │
  │  Phase 5 → Nettoyage et validation            │
  │  Phase 6 → Extraction des features manuelles  │
  │  Phase 7 → Sauvegarde et split 70/15/15       │
  └───────────────────────────────────────────────┘
          │                        │
          ▼                        ▼
  features.npy (120D)      images/ + labels/ (YOLO)
  labels.npy               yolov8_platevision.yaml
          │                        │
          ▼                        ▼
   Module A1                 Module A2
 (Naïves Bayes)          (YOLOv8 + EasyOCR)
          │                        │
          └──────────┬─────────────┘
                     ▼
              Module B (K-Means)
                     │
                     ▼
               Module C (MDP)
```

---

## 2. Datasets sources

### Roboflow — Nigerian License Plate Dataset

| Propriété | Valeur |
|---|---|
| Source | [universe.roboflow.com/nigerianlpd/nigerian-license-plate](https://universe.roboflow.com/nigerianlpd/nigerian-license-plate) |
| Format | YOLOv5 PyTorch natif |
| Volume | **4 791 images** (train: 4491 / val: 200 / test: 100) |
| Annotations | 1 fichier `.txt` par image — format `class cx cy w h` (normalisé) |
| Classe | `0` = plaque d'immatriculation (une seule classe) |
| Transcriptions | Non disponibles |
| Licence | CC BY 4.0 |

Chaque ligne d'annotation `.txt` :
```
0 0.5015625 0.57109375 0.20703125 0.16953125
│ │         │          │          └─ hauteur bbox (normalisée 0-1)
│ │         │          └──────────── largeur bbox (normalisée 0-1)
│ │         └─────────────────────── centre Y (normalisé 0-1)
│ └───────────────────────────────── centre X (normalisé 0-1)
└─────────────────────────────────── classe (0 = plaque)
```

### Mendeley — Automatic Number Plate Recognition Dataset v2

| Propriété | Valeur |
|---|---|
| Source | [data.mendeley.com/datasets/74zz6dj9vn/2](https://data.mendeley.com/datasets/74zz6dj9vn/2) |
| Format | **Images uniquement** — aucun fichier d'annotation |
| Volume | **14 094 images** `.JPG` |
| Organisation | Sous-dossiers par région du Ghana |
| Type | Gros plans directs de plaques → l'image entière = la plaque |

Répartition par région :

| Région | Images |
|---|---|
| Ashanti Region | 6 202 |
| Greater Accra Region | 5 994 |
| DV (Diplomatic Vehicles) | 665 |
| Motorcycle | 559 |
| Brong Ahafo Region | 440 |
| Central Region | 103 |
| Easter Region | 70 |
| Northern Region | 24 |
| UpperWestRegion | 19 |
| Upper EastRegion | 10 |
| Volta Region | 7 |

> **Note** : le Mendeley étant des gros plans de plaques sans bounding box, le pipeline traite chaque image directement comme un crop de plaque (pas de détection nécessaire).

---

## 3. Structure des dossiers

```
platevision/
├── data/
│   ├── prepare_datasets.py        ← LE SCRIPT PRINCIPAL
│   ├── README_data.md             ← ce fichier
│   │
│   ├── raw/                       ← données brutes (NE PAS MODIFIER)
│   │   ├── roboflow/
│   │   │   └── Nigerian License Plate.v1i.yolov5pytorch/
│   │   │       ├── data.yaml
│   │   │       ├── train/
│   │   │       │   ├── images/    ← 4491 images .jpg
│   │   │       │   └── labels/    ← 4491 annotations .txt
│   │   │       ├── valid/
│   │   │       │   ├── images/    ← 200 images
│   │   │       │   └── labels/    ← 200 annotations
│   │   │       └── test/
│   │   │           ├── images/    ← 100 images
│   │   │           └── labels/    ← 100 annotations
│   │   │
│   │   └── mendeley/
│   │       └── Automatic Number Plate Recognition(2)/
│   │           └── Automatic Number Plate Recognition/
│   │               └── Number Plates/
│   │                   ├── Ashanti Region/    ← 6202 images .JPG
│   │                   ├── Greater Accra Region/
│   │                   ├── DV/
│   │                   └── ...                ← autres régions
│   │
│   └── processed/                 ← généré par le pipeline
│       ├── .pipeline_state.json   ← état de reprise du pipeline
│       ├── eda_report.json        ← rapport EDA (phase 2)
│       ├── validation_report.json ← rapport nettoyage (phase 5)
│       ├── dataset_metadata.json  ← métadonnées complètes (phase 7)
│       ├── pipeline.log           ← logs d'exécution
│       │
│       ├── plate_crops/           ← crops de plaques 200×60 (phase 3)
│       │   ├── roboflow_00000.jpg
│       │   ├── roboflow_00001.jpg
│       │   └── mendeley_04792.jpg
│       │
│       ├── characters/            ← caractères segmentés 28×28 (phase 4)
│       │   ├── A/
│       │   │   └── img_000000.png
│       │   ├── B/
│       │   ├── ...
│       │   ├── 0/
│       │   └── unknown/           ← caractères sans transcription
│       │
│       ├── features.npy           ← X_train (N, 120) float32
│       ├── labels.npy             ← y_train (N,) int32
│       ├── features_val.npy       ← X_val
│       ├── labels_val.npy         ← y_val
│       ├── features_test.npy      ← X_test
│       ├── labels_test.npy        ← y_test
│       ├── class_names.txt        ← mapping index → caractère
│       │
│       ├── images/                ← images YOLO letterbox 640×640 (phase 7)
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       │
│       └── labels/                ← annotations YOLO .txt (phase 7)
│           ├── train/
│           ├── val/
│           └── test/
│
├── models/
│   └── configs/
│       └── yolov8_platevision.yaml  ← config d'entraînement YOLOv8
│
└── reports/
    └── figures/
        └── eda_sample_grid.png    ← grille 4×3 d'images EDA
```

---

## 4. Installation des dépendances

```bash
pip install opencv-python numpy pandas scikit-learn Pillow tqdm
```

Toutes les autres bibliothèques utilisées (`pathlib`, `hashlib`, `json`, `re`, `xml.etree.ElementTree`, `argparse`) sont incluses dans la bibliothèque standard Python.

> **Version Python requise** : 3.10 ou supérieure (utilise la syntaxe `X | Y` pour les types).

---

## 5. Exécution du pipeline

### Exécution complète (phases 1 → 7)

```bash
python data/prepare_datasets.py
```

Durée estimée : **30 à 90 minutes** selon le matériel (le dataset Mendeley contient 14 094 images).

---

### Reprendre à partir d'une phase

Si le pipeline est interrompu, il sauvegarde son état après chaque phase dans `data/processed/.pipeline_state.json`. Pour reprendre sans tout recalculer :

```bash
# Reprendre à partir de la phase 4 (segmentation des caractères)
python data/prepare_datasets.py --from-phase 4

# Reprendre à partir de la phase 6 (extraction des features)
python data/prepare_datasets.py --from-phase 6
```

---

### Exécuter une seule phase

```bash
# Exécuter uniquement la phase 2 (EDA)
python data/prepare_datasets.py --phase-only 2

# Exécuter uniquement la phase 6 (features manuelles)
python data/prepare_datasets.py --phase-only 6
```

---

### Vérifier les résultats après exécution

```bash
# Vérifier les features générées
python3 -c "
import numpy as np
X = np.load('data/processed/features.npy')
y = np.load('data/processed/labels.npy')
print(f'X.shape = {X.shape}')   # attendu : (N, 120)
print(f'y.shape = {y.shape}')   # attendu : (N,)
print(f'Classes : {len(set(y.tolist()))} sur 36')
"

# Vérifier le rapport EDA
cat data/processed/eda_report.json

# Vérifier les métadonnées du dataset
cat data/processed/dataset_metadata.json
```

---

## 6. Les 7 phases expliquées

### Phase 1 — Acquisition (`load_raw_datasets`)

**Rôle** : Charger et unifier les deux sources de données dans une structure commune.

**Ce qu'elle fait :**
- **Roboflow** : parcourt les trois splits (`train/`, `valid/`, `test/`), lit le `data.yaml`, associe chaque image `.jpg` à son fichier d'annotation `.txt`. Signale les images sans annotation.
- **Mendeley** : détecte automatiquement le format d'annotation (CSV / XML / JSON COCO / YOLO txt / aucun). Dans ce dataset, format = `none` → chaque image est traitée comme un gros plan direct de plaque.

**Sortie :**
```python
{
  "roboflow": {
    "images": [Path, ...],        # 4791 chemins
    "labels": [Path | None, ...], # None si annotation absente
    "n_images": 4791,
    "has_text": False,
    "class_names": ["plaque_immatriculation"]
  },
  "mendeley": {
    "dataframe": DataFrame,       # colonnes: image_path, plate_text, x1..y2, source, region
    "n_images": 14094,
    "annotation_format": "none",
    "images": [Path, ...]
  }
}
```

---

### Phase 2 — EDA (`quick_eda`)

**Rôle** : Identifier les problèmes de qualité qui affecteront les phases suivantes.

**Ce qu'elle analyse :**
- **Flou** : variance du Laplacien sur image en niveaux de gris → score < 100 = image floue
- **Luminosité** : canal V (HSV) → moyenne < 50 = image nocturne/sombre
- **Résolutions** : images < 100px ou > 4000px signalées
- **Bounding boxes** : ratio largeur/hauteur des plaques (attendu entre 2.5 et 6.0)
- **Fréquence des caractères** : si transcriptions disponibles

**Sorties** :
- `data/processed/eda_report.json` — rapport complet JSON
- `reports/figures/eda_sample_grid.png` — grille visuelle 4×3 avec bboxes vertes

---

### Phase 3 — Extraction des régions de plaques (`extract_plate_regions`)

**Rôle** : Produire des images de plaques normalisées à **200×60 pixels**.

**Deux méthodes selon la source :**

| Source | Méthode | Description |
|---|---|---|
| Roboflow | `annotation` | Conversion YOLO → pixels + padding 5% + redimensionnement |
| Mendeley | `full_image_crop` | Image entière = plaque (déjà recadrée) |

**Conversion coordonnées YOLO → pixels absolus :**
```
x1 = (cx - w/2) × img_largeur
y1 = (cy - h/2) × img_hauteur
x2 = (cx + w/2) × img_largeur
y2 = (cy + h/2) × img_hauteur
```

**Validation** : crops rejetés si largeur < 20px ou hauteur < 10px.

**Redimensionnement** :
- Upscaling (image trop petite) → `cv2.INTER_CUBIC`
- Downscaling (image trop grande) → `cv2.INTER_AREA`

**Sortie** : fichiers `roboflow_XXXXX.jpg` et `mendeley_XXXXX.jpg` dans `data/processed/plate_crops/`

---

### Phase 4 — Segmentation des caractères (`segment_characters`)

**Rôle** : Extraire chaque caractère individuel depuis les crops 200×60 → images **28×28 pixels**.

**Pipeline de traitement :**
```
Crop 200×60 (BGR)
    ↓ Conversion gris
    ↓ Seuillage adaptatif gaussien (fenêtre 11, constante 2) + THRESH_BINARY_INV
    ↓ Fermeture morphologique kernel 2×2 (supprime le bruit)
    ↓ findContours (contours externes)
    ↓ Filtres : 30% < h < 90% de crop_H
                5px < w < 15% de crop_W
                1.0 < h/w < 4.0
                surface > 100 px²
    ↓ Tri gauche → droite (coordonnée x)
    ↓ Redimensionnement 28×28
```

**Labellisation automatique** : si la transcription est connue et le nombre de contours détectés = longueur du texte, chaque contour reçoit le caractère correspondant (position gauche→droite).

**Sortie** :
```
data/processed/characters/
├── A/img_000000.png    ← caractères labellisés certains
├── B/img_000001.png
├── ...
└── unknown/            ← caractères sans label (Roboflow + Mendeley sans transcription)
```

---

### Phase 5 — Nettoyage et validation (`clean_and_validate`)

**Rôle** : Supprimer les données inutilisables et équilibrer les classes pour le Naïves Bayes.

**Filtres appliqués :**

| Filtre | Condition | Action |
|---|---|---|
| Image noire | `sum(pixels) < 10` | Rejeté |
| Image blanche | `sum(pixels) > 28×28×250` | Rejeté |
| Doublon | Hash MD5 identique | Rejeté |
| Image floue | Variance Laplacien < 5.0 | Rejeté |
| Image corrompue | `shape != (28, 28)` | Rejeté |
| Label invalide | Hors `A-Z0-9` | Rejeté |

**Équilibrage des classes :**
- **> 500 échantillons** → sous-échantillonnage aléatoire à 500 (`seed=42`)
- **< 50 échantillons** → augmentation légère jusqu'à 50 :
  - Rotation aléatoire ±5°
  - Bruit gaussien σ=2
  - Distorsion affine légère

**Sortie** : `data/processed/validation_report.json`

---

### Phase 6 — Extraction des features manuelles (`extract_all_features`)

**Rôle** : Construire le vecteur de features 120D par caractère pour le Naïves Bayes (Module A1).

**Composition du vecteur (120 dimensions) :**

```
┌─────────────────────────────────────────────────────────────┐
│  FEATURES CARACTÈRE 28×28                        (20 dim)   │
│                                                              │
│  Groupe A — Dimensions & densité              (4 dim)        │
│    • Ratio h/w                                               │
│    • Densité (% pixels > 128)                                │
│    • Centroïde cx/28, cy/28                                  │
│                                                              │
│  Groupe B — Distribution spatiale             (8 dim)        │
│    • Densité par zone dans grille 4 col × 2 lignes           │
│                                                              │
│  Groupe C — Transitions 0→1                   (4 dim)        │
│    • Transitions par ligne : moyenne + écart-type            │
│    • Transitions par colonne : moyenne + écart-type          │
│                                                              │
│  Groupe D — Gradients Sobel                   (4 dim)        │
│    • Sobel X : moyenne + écart-type                          │
│    • Sobel Y : moyenne + écart-type                          │
├─────────────────────────────────────────────────────────────┤
│  FEATURES PLAQUE PARENTE 200×60                (100 dim)    │
│                                                              │
│  a) Histogramme HSV                           (96 dim)       │
│     H(32 bins) + S(32 bins) + V(32 bins), normalisés à 1.0  │
│                                                              │
│  b) Gradient moyen                             (2 dim)       │
│     Magnitude Sobel : moyenne + écart-type                   │
│                                                              │
│  c) Ratio pixels foncés                        (1 dim)       │
│     count(pixel < 128) / total_pixels                        │
│                                                              │
│  d) Présence cadre métallique                  (1 dim)       │
│     Binaire : bordure 5px avec S<30 et V>150 (>60%)          │
└─────────────────────────────────────────────────────────────┘
                            TOTAL : 120 dimensions
```

**Encodage des labels :**
```python
VALID_CHARS = sorted('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
# A=0, B=1, ..., Z=25, 0=26, 1=27, ..., 9=35
```

**Sorties** : `X` (N, 120) float32 et `y` (N,) int32

---

### Phase 7 — Sauvegarde et split (`save_and_split`)

**Rôle** : Persister tous les artefacts finaux dans le format attendu par chaque module.

**Split des caractères (stratifié)** :
```
X, y  →  train_test_split(test_size=0.30, stratify=y, seed=42)
              ↓                    ↓
          X_train (70%)       X_temp (30%)
          y_train                  ↓
                        train_test_split(test_size=0.50, stratify, seed=42)
                                ↓           ↓
                            X_val (15%)   X_test (15%)
```

**Letterbox 640×640 pour YOLO** :
```
Image originale (W×H)
    ↓ scale = min(640/W, 640/H)   # conserve le ratio
    ↓ Redimensionnement → (new_W, new_H)
    ↓ Canvas 640×640 gris (114, 114, 114)
    ↓ Centrage + collage
```

**Fichiers produits** :

| Fichier | Usage | Module |
|---|---|---|
| `features.npy` | X_train (N_train, 120) | A1 |
| `labels.npy` | y_train (N_train,) | A1 |
| `features_val.npy` | X_val | A1 |
| `labels_val.npy` | y_val | A1 |
| `features_test.npy` | X_test | A1 |
| `labels_test.npy` | y_test | A1 |
| `class_names.txt` | mapping idx→char | A1 |
| `images/train/*.jpg` | Images 640×640 | A2 (YOLO) |
| `labels/train/*.txt` | Annotations YOLO | A2 (YOLO) |
| `yolov8_platevision.yaml` | Config entraînement | A2 (YOLO) |
| `dataset_metadata.json` | Métadonnées complètes | Tous |

---

## 7. Fichiers produits

### `data/processed/class_names.txt`

Mapping index → caractère pour les 36 classes :
```
0,A
1,B
2,C
...
25,Z
26,0
27,1
...
35,9
```

### `data/processed/dataset_metadata.json`

```json
{
  "sources": ["roboflow", "mendeley"],
  "urls": { "roboflow": "...", "mendeley": "..." },
  "total_images_plaques": 18885,
  "total_chars_labellises": 0,
  "split": { "train": 0, "val": 0, "test": 0 },
  "feature_dim": 120,
  "n_classes_chars": 36,
  "label_to_idx": { "A": 0, "B": 1, ... },
  "biais_identifies": ["..."],
  "date_creation": "2026-06-12"
}
```

### `data/processed/eda_report.json`

```json
{
  "total_images": 18885,
  "blurry_images": ["chemin/image_floue.jpg", ...],
  "dark_images": ["chemin/image_sombre.jpg", ...],
  "corrupted_images": [],
  "resolution_stats": { "min_width": 120, "max_width": 3024, "median_width": 1280, ... },
  "plate_ratio_stats": { "min": 1.8, "max": 7.2, "median": 3.9 },
  "character_frequency": {},
  "recommended_target_size": [640, 640]
}
```

---

## 8. Ce que consomme chaque module IA

### Module A1 — Naïves Bayes Gaussien

```python
import numpy as np

X_train = np.load("data/processed/features.npy")   # (N, 120) float32
y_train = np.load("data/processed/labels.npy")     # (N,) int32
X_val   = np.load("data/processed/features_val.npy")
y_val   = np.load("data/processed/labels_val.npy")

# Lecture des noms de classes
with open("data/processed/class_names.txt") as f:
    class_names = {int(line.split(",")[0]): line.split(",")[1].strip()
                   for line in f}

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
score = clf.score(X_val, y_val)
```

### Module A2 — YOLOv8 + EasyOCR

```bash
# Entraînement YOLO à partir des images préparées
yolo detect train \
  data=models/configs/yolov8_platevision.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640
```

### Module B — K-Means sur embeddings CNN

Consomme les images de `data/processed/images/train/` (640×640, normalisées, letterbox).

### Module C — MDP

Les états `{cluster_id, niveau_OCR, alerte}` sont construits à partir des sorties de A1, A2 et B — pas de consommation directe des fichiers de données brutes.

---

## 9. Biais connus et limitations

| Biais | Description | Impact |
|---|---|---|
| **Géographique Roboflow** | Dataset principalement nigérian | Mauvaise généralisation aux plaques camerounaises |
| **Géographique Mendeley** | Dataset ghanéen exclusivement | Format CEMAC probablement absent |
| **Absence de transcriptions** | Ni Roboflow ni Mendeley ne fournissent le texte des plaques | Module A1 entraîné sur `unknown` uniquement — performances réduites |
| **Conditions nocturnes** | Très peu d'images sombres dans les deux datasets | Faux négatifs en conditions de nuit |
| **Diversité des angles** | Images majoritairement frontales | Difficultés sur plaques de côté/angle |

> **Recommandation** : compléter avec des images de plaques camerounaises annotées (format CEMAC : `XX-XXX-XX`) pour corriger le biais géographique avant déploiement MINT/DGI.

---

## Référence rapide des commandes

```bash
# Exécution complète
python data/prepare_datasets.py

# Reprendre après interruption
python data/prepare_datasets.py --from-phase 3

# Phase EDA uniquement (pour inspecter la qualité)
python data/prepare_datasets.py --phase-only 2

# Voir les logs en direct
tail -f data/processed/pipeline.log

# Vérifier l'état sauvegardé
cat data/processed/.pipeline_state.json
```
