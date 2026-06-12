# Module A — Détection & Reconnaissance de Plaques

> **Sous-module A1 : Classificateur Naïves Bayes Gaussien**
> Baseline de reconnaissance de caractères alphanumériques (36 classes) sur vecteurs de features 120D.

---

## Contexte et démarche

Le Module A couvre la chaîne complète de reconnaissance de plaques d'immatriculation camerounaises.
Il comprend deux composants complémentaires :

| Composant | Fichier | Rôle |
|-----------|---------|------|
| **A1 — Naïves Bayes** | `naive_bayes.py` | Baseline statistique, interprétable, rapide |
| **A2 — YOLO + OCR** | `yolo_ocr_pipeline.py` | Pipeline de production (deep learning) |

Le classifieur Naïves Bayes sert de **référence de performance** : toute approche plus complexe doit le surpasser pour être justifiée.

---

## Démarche algorithmique

### Pourquoi Naïves Bayes Gaussien ?

L'hypothèse d'indépendance conditionnelle entre les 120 features (bien que non strictement vraie) produit un modèle très rapide à entraîner et dont les probabilités postérieures sont directement interprétables. C'est le choix canonique comme baseline pour la classification multi-classes sur features continues.

### Structure du vecteur 120D

Chaque caractère segmenté est représenté par un vecteur de 120 dimensions :

```
[0:4]     Forme      — ratio h/w, densité pixels, centroïde cx/28, cy/28
[4:12]    Grille     — densités 4×2 zones
[12:16]   Transitions— horizontales/verticales (mean + std)
[16:20]   Gradient   — Sobel X/Y (mean + std)
[20:116]  Couleur    — histogramme HSV normalisé (H×32 + S×32 + V×32)
[116:118] Gradient   — magnitude mean + std
[118]     Luminance  — ratio pixels sombres
[119]     Cadre      — indicateur binaire cadre métallique
```

### Pipeline complet

```
features.npy / labels.npy
        │
        ▼
 load_features()          ← charge train/val/test + class_names
        │
        ▼
 preprocess_features()    ← StandardScaler fitté sur train uniquement
        │
        ▼
 train_naive_bayes()      ← GaussianNB(var_smoothing=1e-9)
        │
        ▼
 evaluate_model()         ← accuracy, f1_macro, top3 + confusion matrix
        │
        ▼
 analyze_errors()         ← top 10 paires confondues (O↔0, I↔1…)
        │
        ▼
 analyze_ocr_alignment()  ← accord NB vs EasyOCR (optionnel)
        │
        ▼
 save_model()             ← .pkl + métriques .json
```

---

## Installation des dépendances

```bash
# Depuis la racine du projet
source .venv/bin/activate          # ou : python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Dépendances utilisées par ce module :

```
numpy
scikit-learn
matplotlib
seaborn
joblib
```

---

## Prérequis — données

Le pipeline de préparation doit avoir été exécuté au préalable (voir `data/README_data.md`).
Les fichiers attendus dans `data/processed/` :

```
data/processed/
├── features.npy          # (N, 120) float32 — features train
├── labels.npy            # (N,)     int     — labels train [0–35]
├── features_val.npy
├── labels_val.npy
├── features_test.npy
├── labels_test.npy
├── class_names.txt       # 36 lignes : A B C … Z 0 1 … 9
└── ocr_results.json      # optionnel — sorties EasyOCR
```

---

## Exécution

Toutes les commandes s'exécutent depuis **la racine du projet**.

### 1. Entraînement complet (pipeline end-to-end)

```bash
python modules/module_a/naive_bayes.py --train
```

Produit :
- `models/weights/naive_bayes_model.pkl`
- `models/weights/naive_bayes_scaler.pkl`
- `models/weights/naive_bayes_metrics.json`
- `reports/figures/confusion_matrix_nb.png`
- `reports/figures/classification_report_nb.txt`

### 2. Prédire un échantillon de test

```bash
# Prédit le sample d'index 42 dans X_test et affiche le top 3
python modules/module_a/naive_bayes.py --predict 42
```

Sortie attendue :
```
Vrai label      : O
Prédit          : O
Confiance       : 0.8731
Top 3 :
  O — 0.8731
  0 — 0.0924
  Q — 0.0215
```

### 3. Réévaluer le modèle sauvegardé

```bash
python modules/module_a/naive_bayes.py --evaluate
```

Recharge `naive_bayes_model.pkl` et recalcule toutes les métriques sur le jeu de test.

### 4. Analyser l'alignement NB–OCR

```bash
python modules/module_a/naive_bayes.py --ocr-align
```

Compare les prédictions du modèle avec les textes lus par EasyOCR (`ocr_results.json`).
Affiche le taux d'accord caractère par caractère.

---

## Tests

```bash
# Depuis la racine du projet
pytest tests/test_naive_bayes.py -v
```

| Test | Ce qu'il vérifie |
|------|-----------------|
| `test_load_features_correct_shapes` | `X_train.shape[1] == 120`, `len(class_names) == 36` |
| `test_load_features_missing_file` | `FileNotFoundError` si `features.npy` absent |
| `test_preprocess_no_data_leakage` | Scaler fitté sur train uniquement (pas de fuite val) |
| `test_train_returns_gnb` | Retourne `GaussianNB` avec `var_smoothing=1e-9` |
| `test_predict_output_structure` | Clés, `confidence ∈ [0,1]`, `len(top3) == 3` |
| `test_save_load_roundtrip` | Sauvegarde → rechargement → prédictions identiques |
| `test_ocr_alignment_missing_file` | Retourne `{}` sans exception si JSON absent |

---

## Sorties et artefacts

```
models/weights/
├── naive_bayes_model.pkl       ← modèle GaussianNB sérialisé
├── naive_bayes_scaler.pkl      ← StandardScaler sérialisé
└── naive_bayes_metrics.json    ← métriques val + test + confusions

reports/figures/
├── confusion_matrix_nb.png     ← heatmap 36×36 (14"×12")
└── classification_report_nb.txt← precision/recall/f1 par classe
```

---

## Métriques attendues (ordre de grandeur)

Sur un dataset équilibré de 36 classes :

| Métrique | Valeur typique |
|----------|---------------|
| Accuracy (test) | 0.55 – 0.70 |
| F1 macro (test) | 0.50 – 0.65 |
| Top-3 accuracy | 0.80 – 0.90 |

Les confusions les plus fréquentes sont structurellement prévisibles : `O ↔ 0`, `I ↔ 1`, `S ↔ 5`, `B ↔ 8`.
Le Module A2 (YOLO + OCR) est conçu pour surpasser ces performances.
