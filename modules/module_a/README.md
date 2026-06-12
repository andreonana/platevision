# Module A — Détection & Reconnaissance de Plaques

> **PlateVision / MINT-DGI Cameroun — Cahier des charges §4.1**
> Chaîne complète de détection, lecture et évaluation des plaques d'immatriculation.

---

## Vue d'ensemble

Le Module A couvre trois composants :

| Fichier | Composant | Rôle |
|---------|-----------|------|
| `naive_bayes.py` | **A1 — Naïves Bayes** | Baseline statistique sur caractères pré-segmentés |
| `yolo_ocr_pipeline.py` | **A2 — YOLO + OCR** | Détection bout-en-bout + lecture EasyOCR |
| `evaluate.py` | **Comparaison A1 vs A2** | Tableau jury obligatoire §4.1 |

Le classifieur Naïves Bayes sert de **référence de performance** : son incapacité à localiser la plaque dans l'image brute justifie formellement le passage au pipeline YOLO+OCR.

---

## A1 — Naïves Bayes Gaussien (`naive_bayes.py`)

### Démarche algorithmique

```
features.npy / labels.npy
        │
        ▼
 load_features()          ← train / val / test + class_names (36 classes)
        │
        ▼
 preprocess_features()    ← StandardScaler fitté sur train uniquement
        │
        ▼
 train_naive_bayes()      ← GaussianNB(var_smoothing=1e-9)
        │
        ▼
 evaluate_model()         ← accuracy, f1_macro, top3 + matrice confusion 36×36
        │
        ▼
 analyze_errors()         ← top 10 paires confondues (O↔0, I↔1…)
        │
        ▼
 generate_pedagogical_report() ← rapport jury §4.1 (corrélations, impact MINT/DGI)
        │
        ▼
 analyze_ocr_alignment()  ← accord NB vs EasyOCR (optionnel)
        │
        ▼
 save_model()             ← .pkl + métriques .json
```

### Vecteur de features 175D

```
[0:4]      Forme       — ratio h/w, densité pixels, centroïde cx/28, cy/28
[4:12]     Grille      — densités 4×2 zones
[12:16]    Transitions — horizontales/verticales (mean + std)
[16:20]    Gradient    — Sobel X/Y (mean + std)
[20:116]   Couleur     — histogramme HSV normalisé (H×32 + S×32 + V×32)
[116:118]  Gradient    — magnitude mean + std
[118]      Luminance   — ratio pixels sombres
[119]      Cadre       — indicateur binaire cadre métallique
[120:175]  HOG (55D)   — gradient orienté (complément cahier des charges)
```

### Commandes

```bash
# Pipeline complet (entraînement + évaluation + rapport)
python modules/module_a/naive_bayes.py --train

# Prédire le sample d'index 42 de X_test (affiche top-3)
python modules/module_a/naive_bayes.py --predict 42

# Réévaluer le modèle sauvegardé sur le jeu de test
python modules/module_a/naive_bayes.py --evaluate

# Analyser l'alignement NB vs EasyOCR
python modules/module_a/naive_bayes.py --ocr-align
```

### Artefacts produits

```
models/weights/
├── naive_bayes_model.pkl
├── naive_bayes_scaler.pkl
└── naive_bayes_metrics.json

reports/rapport_technique/figures/
├── confusion_matrix_nb.png
├── classification_report_nb.txt
└── naive_bayes_pedagogical_report.txt
```

### Métriques attendues

| Métrique | Valeur typique |
|----------|---------------|
| Accuracy (test) | 0.55 – 0.70 |
| F1 macro (test) | 0.50 – 0.65 |
| Top-3 accuracy | 0.80 – 0.90 |
| Inférence | < 2 ms / vecteur |

Confusions les plus fréquentes : `O ↔ 0`, `I ↔ 1`, `S ↔ 5`, `B ↔ 8`.

---

## A2 — Pipeline YOLO + OCR (`yolo_ocr_pipeline.py`)

### Justification architecture (§4.1 A2)

- **YOLOv8n (nano)** : transfer learning depuis COCO, 3.2M paramètres, < 20 ms/image sur GPU entrée de gamme → compatible contrainte temps réel postes de contrôle MINT
- **EasyOCR** : gère mieux les polices alphanumériques des plaques camerounaises que Tesseract, supporte le fine-tuning, alphabet latin sans configuration complexe (§3.2.2)
- **Alternative Detectron2 écartée** : trop lourd pour déploiement embarqué

### Pipeline complet

```
Image brute (640×640 letterbox)
        │
        ▼
 prepare_yaml()            ← valide / crée yolov8_platevision.yaml (chemin absolu)
        │
        ▼
 train_yolo()              ← YOLO("yolov8n.pt") + transfer learning COCO
        │                     epochs=50, imgsz=640, batch=16
        ▼
 evaluate_yolo()           ← mAP@0.5, mAP@0.5:0.95, précision, rappel
        │                     + temps d'inférence moyen (ms)
        ▼
 plot_training_curves()    ← figure 2×2 : box_loss / cls_loss / mAP50 / mAP50-95
        │
        ▼
 detect_plate()            ← crop plaque + 5% padding, image annotée
        │
        ▼
 deskew_plate_crop()       ← redressement angle oblique (minAreaRect + warpAffine)
        │
        ▼
 read_plate_ocr()          ← EasyOCR allowlist=[A-Z0-9], tri gauche→droite
        │
        ▼
 postprocess_ocr_text()    ← majuscules, suppression espaces/tirets, filtrage
        │
        ▼
 evaluate_full_pipeline()  ← CER, WER, detection_rate, perfect_read_rate
                              + comparaison NB vs YOLO+OCR
```

### Commandes

```bash
# Entraînement YOLOv8n (nécessite données Phase 8)
python modules/module_a/yolo_ocr_pipeline.py --train --epochs 50

# Évaluer le modèle sauvegardé (mAP + temps inférence)
python modules/module_a/yolo_ocr_pipeline.py --evaluate

# Régénérer les courbes d'entraînement depuis results.csv
python modules/module_a/yolo_ocr_pipeline.py --curves

# Inférence YOLO seule sur une image
python modules/module_a/yolo_ocr_pipeline.py --detect data/raw/images/test/img.jpg

# Pipeline complet YOLO+OCR — démo jury (§5)
python modules/module_a/yolo_ocr_pipeline.py --ocr data/raw/images/test/img.jpg

# Évaluation CER/WER sur le jeu de test
python modules/module_a/yolo_ocr_pipeline.py --eval-ocr
```

### Artefacts produits

```
models/weights/
└── yolov8_platevision.pt          ← poids best.pt copié depuis runs/

reports/rapport_technique/figures/
├── yolo_metrics.json              ← mAP@0.5, mAP@0.5:0.95, inférence ms
├── yolo_training_curves.png       ← courbes box_loss / cls_loss / mAP
├── yolo_ocr_evaluation.json       ← CER, WER, detection_rate
└── detection_demo.png             ← image annotée (démo jury)
```

### Métriques cibles (§4.1)

| Métrique | Cible |
|----------|-------|
| mAP@0.5 | ≥ 0.80 |
| mAP@0.5:0.95 | ≥ 0.60 |
| CER moyen | ≤ 0.15 |
| WER moyen | ≤ 0.30 |
| Inférence totale | < 100 ms (seuil temps réel MINT) |

---

## Comparaison obligatoire A1 vs A2 (`evaluate.py`)

### Exigence cahier des charges §4.1

> "La comparaison rigoureuse Pipeline YOLO+OCR vs. Naïves Bayes est obligatoire et doit couvrir : mAP@0.5, mAP@0.5:0.95, CER, WER, temps d'inférence moyen par image (ms)."

Ce module charge les métriques pré-calculées et produit le tableau jury (livrable L3) et la figure de soutenance (livrable L5).

### Commandes

```bash
# Évaluation comparative complète (tableau + figure + JSON)
python modules/module_a/evaluate.py --compare

# Régénérer uniquement le tableau texte
python modules/module_a/evaluate.py --table-only

# Régénérer uniquement la figure comparative
python modules/module_a/evaluate.py --figure-only
```

### Artefacts produits

```
reports/rapport_technique/figures/
├── comparative_figure.png       ← figure 3 panneaux : qualité / inférence / gains
├── comparative_table.txt        ← tableau ASCII copier-coller rapport
└── comparative_evaluation.json  ← résultat complet
```

### Structure du tableau jury

```
┌──────────────────────────┬────────────────┬────────────────┬────────┐
│ Métrique                 │ Naïves Bayes   │ YOLO+OCR       │ Gain   │
│                          │ (A1)           │ (A2)           │        │
├──────────────────────────┼────────────────┼────────────────┼────────┤
│ Accuracy / Perfect Read  │ ...            │ ...            │ +XX%   │
│ F1-macro / (1-CER)       │ ...            │ ...            │ +XX%   │
│ CER moyen                │ N/A            │ ...            │  —     │
│ WER moyen                │ N/A            │ ...            │  —     │
│ mAP@0.5                  │ N/A            │ ...            │  —     │
│ mAP@0.5:0.95             │ N/A            │ ...            │  —     │
│ Temps inférence (ms)     │ < 2 ms         │ < 100 ms       │ ×N     │
│ Temps réel MINT (<100ms) │ OUI            │ OUI            │  —     │
└──────────────────────────┴────────────────┴────────────────┴────────┘
```

---

## Prérequis — données

Le pipeline de préparation doit avoir été exécuté au préalable (`data/README_data.md`).

```
data/processed/
├── features.npy          # (N, 175) float32 — features train
├── labels.npy            # (N,)     int     — labels [0–35]
├── features_val.npy
├── labels_val.npy
├── features_test.npy
├── labels_test.npy
├── class_names.txt       # 36 lignes : A B … Z 0 1 … 9
├── ocr_results.json      # sorties EasyOCR Phase 4
└── yolo/
    └── images/
        ├── train/        # images 640×640 letterbox
        ├── val/
        └── test/
```

---

## Tests

```bash
# A1 — Naïves Bayes
pytest tests/test_naive_bayes.py -v

# A2 — YOLO + OCR (21 tests, aucun GPU requis)
pytest tests/test_yolo_pipeline.py -v

# Comparaison A1 vs A2
pytest tests/test_evaluate.py -v

# Suite complète Module A
pytest tests/test_naive_bayes.py tests/test_yolo_pipeline.py tests/test_evaluate.py -v
```

| Fichier de test | Couverture | Tests |
|-----------------|-----------|-------|
| `test_naive_bayes.py` | load, preprocess, train, predict, save/load, OCR align | 7 |
| `test_yolo_pipeline.py` | prepare_yaml, evaluate, detect, curves, postprocess, CER/WER, deskew | 21 |
| `test_evaluate.py` | load_metrics, compute_gains, figure, tableau, FileNotFoundError | 14 |

---

## Installation des dépendances

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Dépendances clés de ce module :

```
numpy, scikit-learn, joblib       # A1 — Naïves Bayes
ultralytics                       # A2 — YOLOv8
easyocr, torch                    # A2 — OCR
opencv-python, matplotlib, seaborn, pandas  # commun
pyyaml                            # configuration YAML
```
