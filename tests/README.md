# Tests — PlateVision

Suite de tests unitaires pour tous les modules du projet.
Tous les tests utilisent **pytest** avec des données synthétiques (`tmp_path`, `np.random`) et n'ont aucune dépendance sur les datasets réels.

---

## Structure

```
tests/
├── test_naive_bayes.py    ← Module A1 : Naïves Bayes Gaussien
├── test_yolo_pipeline.py  ← Module A2 : YOLOv8 + EasyOCR
├── test_clustering.py     ← Module B  : K-Means / clustering
├── test_mdp.py            ← Module C  : MDP Value/Policy Iteration
└── __init__.py
```

---

## Prérequis

```bash
# Depuis la racine du projet
source .venv/bin/activate
pip install -r requirements.txt   # inclut pytest
```

---

## Lancer les tests

### Tous les tests

```bash
pytest -v
```

### Un fichier précis

```bash
pytest tests/test_naive_bayes.py -v
```

### Un test précis

```bash
pytest tests/test_naive_bayes.py::test_save_load_roundtrip -v
```

### Avec rapport de couverture

```bash
pytest --cov=modules --cov-report=term-missing -v
```

---

## Module A1 — `test_naive_bayes.py`

Tests du classificateur Naïves Bayes Gaussien (36 classes, features 120D).
Toutes les données sont générées synthétiquement via `numpy.random` ; aucun fichier réel n'est lu.

| Test | Garantie |
|------|---------|
| `test_load_features_correct_shapes` | `X_train.shape[1] == 120`, `len(class_names) == 36` |
| `test_load_features_missing_file` | `FileNotFoundError` levé si `features.npy` absent |
| `test_preprocess_no_data_leakage` | Scaler fitté sur train → `X_train_scaled.mean() ≈ 0` ; val non centré |
| `test_train_returns_gnb` | Instance `GaussianNB` avec `var_smoothing == 1e-9` |
| `test_predict_output_structure` | Clés `predicted_class / confidence / top3`, `confidence ∈ [0,1]`, `len(top3) == 3` |
| `test_save_load_roundtrip` | Sérialisation `.pkl` + JSON → prédictions identiques après rechargement |
| `test_ocr_alignment_missing_file` | Retourne `{}` sans exception si `ocr_results.json` absent |

### Exécution rapide

```bash
pytest tests/test_naive_bayes.py -v
# attendu : 7 passed
```

---

## Conventions

- **Données** : uniquement `numpy.random` + `tmp_path` pytest — jamais de lecture dans `data/`.
- **Isolation** : chaque test reçoit son propre `tmp_path` (répertoire temporaire unique).
- **Pas de mocks externes** : on instancie directement les vrais objets sklearn.
- **Nommage** : `test_<fonction>_<ce_qui_est_vérifié>()`.

---

## Ajouter un nouveau test

1. Créer (ou compléter) le fichier `tests/test_<module>.py`.
2. Utiliser `tmp_path` pour tout fichier temporaire.
3. Générer les données avec `numpy.random.default_rng(seed)` pour la reproductibilité.
4. Vérifier que `pytest -v` passe avant de commiter.
