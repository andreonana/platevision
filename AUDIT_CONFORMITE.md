# Audit de conformité — PlateVision vs. Cahier des charges

**Date** : 2026-06-15
**Branche** : dataset_nkegoueth
**Auditeur** : andreonana

---

## Résumé exécutif

| Module | Statut | Points manquants |
|--------|--------|-----------------|
| L1 — DSD | ✅ CONFORME | — |
| L2 — Gestion projet | ✅ CONFORME | — |
| L3 — Rapport technique | ✅ CONFORME | — |
| L4 — Code source | ⚠️ PARTIEL | `--module C`, `--gamma`, `--algorithm` absents du CLI |
| Module A (A1 NB) | ✅ CONFORME | — |
| Module A (A2 YOLO+OCR) | ⚠️ PARTIEL | YOLO non entraîné → métriques mAP/CER/WER sont `\todo{}` dans le rapport |
| Module B | ✅ CONFORME | — |
| Module C | ⚠️ PARTIEL | n_states=7 sous le minimum §4.3 (9–16) ; alerte_cnn non exploité |
| Module D | ✅ CONFORME | — |
| CLI §3.2 | ❌ ABSENT (partiel) | `python main.py --module C --algorithm VI --gamma 0.95` → erreur fatale |

**Score estimé** : 48 / 57 critères conformes (**84%**)

---

## Détail par section

---

### L1 — Document de Sélection du Dataset (DSD)

| Critère | Statut | Détail |
|---------|--------|--------|
| `reports/DSD/dsd_platevision.tex` existe et non vide | ✅ | 5 pages PDF généré |
| Tableau comparatif ≥ 4 datasets | ✅ | CCPD, OpenALPR Bench., Roboflow LP, UFPR-ALPR — noté sur 7 critères |
| Justification du choix retenu (≥ 1 page) | ✅ | §3 « Dataset retenu » — proximit CEMAC, double annotation, licence |
| Limites et biais identifiés | ✅ | §4 Tableau 4 types de biais avec impact et mitigation |
| Plan de prétraitement | ✅ | §5 — redimensionnement, normalisation pixel [0,1], 6 augmentations |

**→ L1 : ✅ 5/5 CONFORME**

---

### L2 — Rapport de gestion de projet

| Critère | Statut | Détail |
|---------|--------|--------|
| `reports/gestion_projet/gestion_projet.tex` existe et non vide | ✅ | 7 pages PDF généré |
| Gantt sur 4 jours (minimum) | ✅ | Gantt 5 jours (J1→J5) avec toutes les tâches |
| PERT / dépendances entre tâches | ✅ | §3 Tableau PERT + chemin critique identifié |
| Rôles des 5 membres | ✅ | Tableau §1 : NKE GOUETH, ELIE NJOCK, ALI YOUSSOUF, FOUDA BASILE, NAPANI MAËL |
| Matrice des risques | ✅ | §4 — 7 risques (R1→R7) avec probabilité, impact et mitigation |

**→ L2 : ✅ 5/5 CONFORME**

---

### L3 — Rapport technique intégré

| Critère | Statut | Détail |
|---------|--------|--------|
| `main.tex` compile et inclut A/B/C/D | ✅ | 42 pages, 0 erreur pdflatex. `\input{module_a/b/c/d}` actifs |
| `module_a.tex` non vide | ✅ | Complet — NB + YOLO, 3 questions §4.1 documentées |
| `module_b.tex` non vide | ✅ | Complet — clustering, interprétation métier, robustesse |
| `module_c.tex` non vide | ✅ | Complet — MDP formel, VI/PI, interprétation, sensibilité γ |
| `module_d.tex` non vide | ✅ | Complet — 3 parties + carnet de bord J1–J4 |
| Transitions A→B→C→D documentées | ✅ | `module_a.tex` §« Transition vers A2 » ; `module_b.tex` §14/§B.1/§B.9 ; `module_c.tex` §1 et §6 |

**→ L3 : ✅ 6/6 CONFORME**

---

### L4 — Code source

| Critère | Statut | Détail |
|---------|--------|--------|
| `README.md` avec instructions installation/utilisation | ✅ | 184 lignes — installation venv, exemples CLI tous modules |
| `requirements.txt` versionné | ✅ | 61 lignes, 18 paquets avec versions exactes (ultralytics==8.3.40, etc.) |
| `main.py --help` fonctionne | ✅ | Affiche usage complet A1/A2/A/B/B0–B8 + exemples |
| Historique Git propre | ✅ | Convention Conventional Commits, branche dataset_nkegoueth |
| `--module C` opérationnel | ❌ | `python main.py --module C` → `invalid choice: 'C'` — Module C non câblé dans le dispatcher CLI |
| `--gamma` et `--algorithm` flags | ❌ | Absents de `main.py` — impossible à modifier en direct pour jury §5 |

**→ L4 : ⚠️ 4/6 — 2 critères CLI manquants**

---

### Module A — Naïves Bayes (A1)

| Critère | Statut | Détail |
|---------|--------|--------|
| `modules/module_a/naive_bayes.py` (607 lignes) | ✅ | Non vide |
| Features manuelles (HSV, gradient, ratio, dimensions) | ✅ | 4 familles de features, 96 bins HSV + 8 grad + 3 géom |
| `GaussianNB` de scikit-learn | ✅ | `from sklearn.naive_bayes import GaussianNB`, var_smoothing=1e-9 |
| 3 questions §4.1 dans le rapport | ✅ | (1) indépendance conditionnelle §A1.3 ; (2) confusion '0'/'O' + impact MINT §A1.4 ; (3) limite déploiement temps réel |
| Métriques accuracy, F1-macro, matrice de confusion | ✅ | accuracy, f1_macro, top3_accuracy, confusion_matrix_nb.png |

**→ A1 : ✅ 5/5 CONFORME**

---

### Module A — Pipeline YOLO+OCR (A2)

| Critère | Statut | Détail |
|---------|--------|--------|
| `modules/module_a/yolo_ocr_pipeline.py` (783 lignes) | ✅ | Non vide |
| YOLOv8n pour détection | ✅ | `ultralytics.YOLO("yolov8n.pt")`, transfer learning COCO |
| EasyOCR pour reconnaissance | ✅ | `easyocr==1.7.2`, gestion polices alphanumériques |
| Métriques mAP@0.5, mAP@0.5:0.95, CER, WER, temps inférence | ⚠️ | **Code prêt** mais YOLO non entraîné → toutes les valeurs dans `module_a.tex` sont `\todo{valeur}`. `yolo_metrics.json` et `yolo_ocr_evaluation.json` présents mais potentiellement vides |

**→ A2 : ⚠️ 3.5/4 — métriques à remplir après entraînement**

---

### Module A — Comparaison A1 vs A2

| Critère | Statut | Détail |
|---------|--------|--------|
| `modules/module_a/evaluate.py` (507 lignes) | ✅ | Charge nb_metrics.json + yolo_metrics.json |
| Tableau comparatif NB vs YOLO+OCR, 3 métriques | ⚠️ | Structure complète dans `module_a.tex` (Tableau §A1.6) mais colonne YOLO = `\todo{}` |
| Justification pourquoi YOLO dépasse NB | ✅ | §A1.5 « Analyse critique des limites » — 4 arguments documentés |

**→ Compare A1/A2 : ⚠️ 2.5/3 — tableau incomplet côté A2**

---

### Module B — Clustering

| Critère | Statut | Détail |
|---------|--------|--------|
| `cnn_embeddings.py` (539 lignes) — CharEmbeddingCNN | ✅ | |
| `extract_embeddings.py` (365 lignes) | ✅ | |
| `clustering.py` — elbow + silhouette | ✅ | `compute_elbow()` + `compute_silhouette()` + `find_optimal_k()` |
| Choix de k justifié (conforme/dégradée/falsifiée) | ✅ | k=3 : cluster 0=conforme, 1=illisible/dégradée, 2=expirée |
| `kmeans_fit.py` — enrichit metadata.csv | ✅ | `cluster_id`, `dist_centroid`, `confidence_level`, `mint_dgi_procedure` |
| Produit `cluster_mapping.json` | ✅ | k=3, procédures : Laisser passer / Mise en demeure MINT / Signalement DGI |
| `visualization.py` — PCA + t-SNE | ✅ | Axes labellisés, titres explicatifs |
| `interpret_clusters.py` — procédures MINT/DGI | ✅ | Nommage + base légale par cluster |
| Compare clusters non supervisés vs annotations | ✅ | `cluster_vs_annotations.png` produit |
| `robustness.py` — ARI + verdict STABLE/MODÉRÉ/INSTABLE | ✅ | 3 seuils : STABLE (ARI≥0.8) / MODÉRÉMENT STABLE / INSTABLE |

**→ Module B : ✅ 10/10 CONFORME**

---

### Module C — MDP

| Critère | Statut | Détail |
|---------|--------|--------|
| `mdp_definition.py` (673 lignes) | ✅ | |
| Produit `data/processed/mdp_states.json` | ✅ | Fichier présent |
| **Nombre d'états entre 9 et 16 (§4.3)** | ⚠️ | **n_states = 7** (sous le minimum de 9). Pruning trop agressif : alerte_cnn=1 entièrement éliminé → dimension alerte_cnn inutilisée |
| S = cluster_id × conf_ocr × alerte_cnn | ⚠️ | Structure correcte mais alerte_cnn toujours = 0 dans les 7 états retenus |
| `mdp_actions.py` — 5 actions FCFA sourcées | ✅ | LAISSER_PASSER / CONTROLE_STANDARD / ARRET_SAISIE / SIGNALEMENT_DGI / TRANSFERT_PJ, sources Code Route 96/07 + DGI 2023 + Loi 2010/012 |
| Justification SIGNALEMENT_DGI | ✅ | Ratio 22.5× documenté dans actions.py et module_c.tex |
| `mdp_transitions.py` — shape (N,A,N) | ✅ | shape (7,5,7) vérifié |
| Chaque ligne P somme à 1 | ✅ | `np.allclose(T.sum(axis=2), 1.0)` → True |
| `mdp_rewards.py` — shape (N,A) | ✅ | shape (7,5), gains + coûts + risque contestation |
| Tous paramètres sourcés | ✅ | Code Route 96/07, Loi Finances DGI 2023, CGI Art.556 |
| `value_iteration.py` — from scratch, Bellman | ✅ | Équation de Bellman V*(s)←max_a[R+γΣP·V'] ; n_iterations stocké |
| Déduit π*(s) = argmax_a Q*(s,a) | ✅ | `pi_star = Q_star.argmax(axis=1)` |
| `policy_iteration.py` — 2 phases from scratch | ✅ | Phase 1 Policy Eval + Phase 2 Policy Improvement ; n_iterations globales stockées |
| `compare_vi_pi.py` — vitesse + accord + sensibilité γ | ✅ | Compare n_iter VI vs PI, taux d'accord π*, analyse sur plage γ |

**→ Module C : ⚠️ 12/14 — n_states=7 hors plage §4.3 ; alerte_cnn inactif**

---

### Module D — Analyse Éthique et Sociale

| Critère | Statut | Détail |
|---------|--------|--------|
| `module_d.tex` existe et non vide | ✅ | |
| Partie 1 : Surveillance + libertés + loi camerounaise | ✅ | Loi 2010/012 + Code des Libertés Cameroun + CEMAC + analyse comparative RGPD |
| Partie 2 : Biais algorithmes + équité | ✅ | 5 types de biais (tableau), 3 catégories de discrimination, 4 mitigations |
| Partie 3 : Gouvernance + responsabilité + traçabilité | ✅ | 3 scénarios de responsabilité, tableau 6 acteurs, rétention logs 6 mois |
| Carnet de bord — 1 entrée/jour (J1 à J4) | ✅ | J1 dataset biais géo ; J2 confusion 0/O MINT ; J3 cluster illisible 34% ; J4 garde-fou MDP |

**→ Module D : ✅ 5/5 CONFORME**

---

### CLI §3.2 — Contraintes techniques

| Critère | Statut | Détail |
|---------|--------|--------|
| `requirements.txt` versionné | ✅ | 18 paquets versionnés |
| `python main.py --help` | ✅ | Fonctionne, liste tous les modules A/B |
| `--module A1` | ✅ | `python main.py --module A1 --evaluate` ✅ |
| `--module A2` | ✅ | `python main.py --module A2 --detect image.jpg` ✅ |
| `--module A --compare` | ✅ | Déclenche evaluate.py ✅ |
| `--module B --k 3` | ✅ | Déclenche pipeline B complet ✅ |
| `--module C --algorithm VI --gamma 0.95` | ❌ | **Erreur fatale** : `invalid choice: 'C'`. Le dispatcher CLI ne reconnaît pas `'C'` comme module valide |
| `--k` modifiable | ✅ | Câblé et transmis à K-Means |
| `--gamma` modifiable | ❌ | Flag **absent** de `argparse` dans `main.py` |
| `--conf` modifiable | ✅ | `parser.add_argument("--conf", type=float, default=0.45)` ✅ |
| Démo live pipeline A→B→C | ✅ | `python main.py --demo` → `modules/interface/camera_demo.py` |

**→ CLI §3.2 : ⚠️ 8/11 — 3 critères bloquants pour soutenance**

---

### Grille d'évaluation §7.1

| Critère §7.1 | Fichier(s) | Statut |
|---|---|---|
| Démarche sélection dataset (DSD) | `reports/DSD/dsd_platevision.tex` | ✅ |
| NB : hypothèses, métriques, limites | `naive_bayes.py` + `module_a.tex` | ✅ |
| YOLO+OCR : architecture, mAP, CER | `yolo_ocr_pipeline.py` + `module_a.tex` | ⚠️ métriques \todo{} |
| K-Means : justification k, procédures MINT/DGI | `clustering.py` + `module_b.tex` | ✅ |
| MDP : états issus A+B, transitions, récompenses sourcées | `mdp_*.py` + `module_c.tex` | ⚠️ n_states=7 |
| Résolution MDP : VI vs PI, sensibilité γ | `compare_vi_pi.py` + `module_c.tex` | ✅ |
| Analyse éthique : surveillance, biais, gouvernance | `module_d.tex` | ✅ |
| Qualité code : modulaire, Git, documentation | `main.py` + modules | ⚠️ Module C absent CLI |
| Rapport technique : cohérence A→B→C→D, figures | `main.tex` (42 pages) | ✅ |

---

## Points critiques pour la soutenance (§5)

> Ces éléments **bloquent la démo live ou la grille §7.1** si non corrigés.

### ❌ BLOQUANT 1 — `python main.py --module C` → erreur fatale

```
python main.py --module C --algorithm VI --gamma 0.95
# → error: argument --module: invalid choice: 'C'
```

Le dispatcher de `main.py` (ligne ~458) liste `choices=['A1','A2','A','B','B0',…,'B8']` mais pas `'C'`.
Les flags `--gamma` et `--algorithm` n'existent pas dans `argparse`.
**Impact soutenance** : le jury ne peut pas modifier γ ou choisir VI/PI en direct.

### ⚠️ BLOQUANT 2 — Métriques YOLO+OCR sont des `\todo{}` dans le rapport

Lignes 349–356 de `module_a.tex` : mAP@0.5, mAP@0.5:0.95, CER, WER, temps inférence → toutes `\todo{}`.
Le tableau comparatif A1 vs A2 (§A1.6, lignes 396–404) est incomplet côté A2.
**Impact §7.1** : critère « YOLO+OCR : mAP, CER » non rempli.

### ⚠️ IMPORTANT — n_states = 7 (sous minimum §4.3)

§4.3 exige 9–16 états. Le système en produit 7 car `alerte_cnn=1` est entièrement pruné.
Sur 7 états, 0 ont `alerte_cnn=1` → la dimension alerte_cnn est formellement présente mais inerte.
**Impact §7.1** : la non-conformité est documentable mais la justification du pruning existe dans `module_c.tex`.

---

## Actions restantes avant soutenance

| Priorité | Action | Fichier concerné |
|----------|--------|-----------------|
| 🔴 CRITIQUE | Câbler `--module C` dans le dispatcher `main.py` (choices + handler) | `main.py` ~ligne 458 |
| 🔴 CRITIQUE | Ajouter `--gamma` (float, default=0.95) à argparse et le passer à VI/PI | `main.py` argparse block |
| 🔴 CRITIQUE | Ajouter `--algorithm` (choix `VI`/`PI`, default=`VI`) à argparse | `main.py` argparse block |
| 🔴 CRITIQUE | Entraîner YOLOv8n (ou charger poids pré-entraînés) et lancer `--module A2 --evaluate` pour peupler `yolo_metrics.json` | `main.py --module A2 --train` |
| 🔴 CRITIQUE | Remplacer les `\todo{}` dans `module_a.tex` par les valeurs JSON réelles une fois A2 évalué | `reports/rapport_technique/module_a.tex` lignes 349–404 |
| 🟡 IMPORTANT | Ajuster le seuil de pruning pour obtenir n_states ∈ [9,16] (par ex. freq_min=0.001 au lieu de 0.01) afin d'inclure certains états alerte_cnn=1 | `modules/module_c/mdp_definition.py` |
| 🟡 IMPORTANT | Mettre à jour `module_c.tex` avec le nouveau n_states si recalibration | `reports/rapport_technique/module_c.tex` §2.1 |
| 🟢 MINEUR | Ajouter une section §3.2.2 dans le rapport justifiant chaque dépendance `requirements.txt` | `reports/rapport_technique/module_a.tex` ou annexe |
| 🟢 MINEUR | Compiler une dernière fois `main.tex` après remplissage des \todo{} pour un PDF livrable final propre | `reports/rapport_technique/` |
