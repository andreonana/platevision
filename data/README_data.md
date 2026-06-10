# Instructions — Téléchargement du Dataset PlateVision

Le dossier `data/raw/` n'est **pas versionné** sur Git (trop volumineux).
Suivez ces instructions pour récupérer les données nécessaires.

---

## Dataset Recommandé

### Option 1 : Dataset ALPR Cameroun (interne)
Si vous avez accès au dataset interne fourni par l'encadrant :
```bash
# Télécharger le fichier archive depuis le lien partagé
wget -O data/raw/platevision_dataset.zip [URL_FOURNIE_PAR_ENCADRANT]
cd data/raw && unzip platevision_dataset.zip
```

### Option 2 : Roboflow Public Dataset (Licence publique)
```bash
pip install roboflow
python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='VOTRE_CLE_API')
project = rf.workspace().project('license-plate-recognition-rxg4e')
dataset = project.version(4).download('yolov8')
"
```

### Option 3 : Dataset Kaggle
```bash
pip install kaggle
kaggle datasets download -d andrewmvd/car-plate-detection
unzip car-plate-detection.zip -d data/raw/
```

---

## Structure Attendue

Après téléchargement, votre dossier `data/` doit ressembler à :
```
data/
├── raw/
│   ├── images/
│   │   ├── train/     (images d'entraînement)
│   │   ├── val/       (images de validation)
│   │   └── test/      (images de test)
│   └── labels/
│       ├── train/     (annotations YOLO .txt)
│       ├── val/
│       └── test/
├── processed/         (généré par preprocessing/normalization.py)
└── annotations/       (annotations COCO JSON si besoin)
```

---

## Statistiques du Dataset

À compléter dans le Document de Sélection du Dataset (L1) :

| Statistique | Valeur |
|-------------|--------|
| Total images | |
| Images train | |
| Images validation | |
| Images test | |
| Résolution moyenne | |
| Conditions de prise de vue | |
| Distribution classes | |

---

## Vérification de l'Intégrité

```bash
# Vérifier le nombre d'images
find data/raw -name "*.jpg" -o -name "*.png" | wc -l

# Vérifier les annotations (format YOLO)
python -c "
from pathlib import Path
labels = list(Path('data/raw/labels/train').glob('*.txt'))
print(f'{len(labels)} fichiers annotations trouvés')
"
```
