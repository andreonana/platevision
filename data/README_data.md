# Instructions — Téléchargement du Dataset PlateVision

> **Important :** Les images brutes ne sont PAS versionnées dans ce dépôt (voir `.gitignore`).
> Suivre les instructions ci-dessous pour reconstituer le dataset localement.

---

## Datasets recommandés

### Option 1 — OpenALPR Benchmark Dataset (recommandé)
Dataset public de plaques d'immatriculation multi-pays.

```bash
# Téléchargement via Kaggle
pip install kaggle
kaggle datasets download -d andrewmvd/car-plate-detection
unzip car-plate-detection.zip -d data/raw/
```

### Option 2 — CCPD (Chinese City Parking Dataset)
Large dataset avec annotations YOLO compatibles (adapté pour la structure du pipeline).

```bash
# Via gdown (Google Drive)
pip install gdown
gdown --id <ID_FICHIER_GDRIVE> -O data/raw/ccpd.zip
unzip data/raw/ccpd.zip -d data/raw/
```

### Option 3 — Dataset personnalisé MINT/DGI Cameroun
Si vous disposez d'images collectées sur le terrain :

1. Placer les images dans `data/raw/`
2. Annoter avec [Label Studio](https://labelstud.io) ou [Roboflow](https://roboflow.com)
3. Exporter en format YOLO v8 dans `data/annotations/`

---

## Structure attendue après téléchargement

```
data/
├── raw/
│   ├── train/
│   │   ├── images/          # Fichiers .jpg ou .png
│   │   └── labels/          # Annotations YOLO .txt correspondantes
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
├── processed/               # Généré par preprocessing/normalization.py
└── annotations/             # Labels finaux utilisés pour l'entraînement
```

---

## Format des annotations YOLO

Chaque fichier `.txt` correspond à une image et contient une ligne par plaque :

```
<class_id> <x_center> <y_center> <width> <height>
```

Exemple pour une seule plaque dans l'image :
```
0 0.512 0.384 0.234 0.089
```

- `class_id` = 0 (une seule classe : `license_plate`)
- Valeurs normalisées entre 0.0 et 1.0 relativement aux dimensions de l'image

---

## Split recommandé

| Partition | Proportion | Usage |
|-----------|-----------|-------|
| Train     | 70%       | Entraînement YOLOv8 |
| Val       | 15%       | Validation pendant l'entraînement |
| Test      | 15%       | Évaluation finale (mAP, CER, WER) |

---

## Prétraitement après téléchargement

```bash
# Normalisation et redimensionnement
python -c "
from preprocessing.normalization import normalize_images
from pathlib import Path
normalize_images(Path('data/raw/train/images'))
"

# Extraction de features pour Naïves Bayes (Module A1)
python -c "
from preprocessing.feature_extraction import extract_features_from_dataset
from pathlib import Path
extract_features_from_dataset(Path('data/processed'))
"
```

---

## Statistiques cibles du dataset

| Métrique | Valeur cible |
|----------|-------------|
| Nombre total d'images | ≥ 1 000 |
| Images d'entraînement | ≥ 700 |
| Plaques par image (moyenne) | 1–3 |
| Résolution minimale | 640×480 |
| Diversité régionale | ≥ 3 régions camerounaises |

---

*Pour toute question sur le dataset, ouvrir une issue GitHub : `.github/ISSUE_TEMPLATE/dataset_problem.md`*
