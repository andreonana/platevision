# Guide de Contribution — PlateVision

Bienvenue dans le projet PlateVision ! Ce guide définit les conventions
de travail collectif pour l'équipe de 5 étudiants. **Lisez-le intégralement
avant votre premier commit.**

---

## 1. Convention de Nommage des Commits

### Format Obligatoire

```
[MODULE] type(scope): description courte en français

Corps optionnel (explication détaillée si nécessaire)

Références : #numéro-issue
```

### Modules disponibles
| Balise | Description |
|--------|-------------|
| `[A1]` | Classifieur Naïves Bayes |
| `[A2]` | Pipeline YOLO + OCR |
| `[B]`  | Clustering K-Means |
| `[C]`  | MDP (Value/Policy Iteration) |
| `[D]`  | Éthique, carnet de bord, coordination |
| `[PREP]` | Prétraitement (preprocessing/) |
| `[TEST]` | Tests unitaires |
| `[ALL]`  | Modifications transversales |

### Types de commits
| Type | Utilisation |
|------|-------------|
| `feat` | Nouvelle fonctionnalité |
| `fix` | Correction d'un bug |
| `docs` | Documentation uniquement |
| `refactor` | Restructuration sans ajout de fonctionnalité |
| `test` | Ajout ou modification de tests |
| `data` | Ajout/modification de données ou annotations |
| `config` | Fichiers de configuration |
| `chore` | Tâche de maintenance (requirements, .gitignore...) |

### Exemples Valides

```bash
[A1] feat(naive_bayes): ajout extraction features HSV 32 bins
[A2] fix(yolo): correction seuil confiance détection plaque (0.3 → 0.5)
[A2] feat(ocr): intégration EasyOCR multilingue (fr+en)
[B]  feat(clustering): implémentation K-Means sur embeddings ResNet50
[B]  docs(visualization): ajout courbe du coude et interprétation
[C]  feat(mdp): définition états et matrice de transitions MINT/DGI
[C]  fix(vi): correction boucle convergence Value Iteration
[D]  docs(ethics): analyse biais géographiques section 3.1
[D]  docs(logbook): entrée carnet de bord jour 2
[PREP] feat(augmentation): ajout transformation RandomRain albumentations
[TEST] test(mdp): ajout tests unitaires Value Iteration vs Policy Iteration
[ALL] refactor(pipeline): modularisation main.py avec argparse
[ALL] chore(deps): mise à jour ultralytics 8.3.40
```

### Exemples à NE PAS faire

```bash
# ❌ Trop vague
update code
fix bug
work in progress

# ❌ En anglais (les descriptions doivent être en français)
[A2] feat(yolo): add confidence threshold

# ❌ Sans balise de module
feat: ajout clustering

# ❌ Commit fourre-tout (trop de changements à la fois)
[ALL] feat: ajout naïves bayes + yolo + clustering + mdp + éthique
```

---

## 2. Workflow Git de l'Équipe

### Structure des Branches

```
main ──────────────────────────────────────────── (production, protégée)
  └── develop ───────────────────────────────────  (intégration)
        ├── feature/module-a  (Étudiants 1 + 2)
        ├── feature/module-b  (Étudiant 3)
        ├── feature/module-c  (Étudiant 4)
        └── feature/module-d  (Étudiant 5)
```

### Cycle de Travail Quotidien

```bash
# 1. Récupérer les dernières modifications de develop
git fetch origin
git checkout feature/module-x
git merge origin/develop   # ou git rebase origin/develop

# 2. Travailler sur votre fonctionnalité
# ... éditer les fichiers ...

# 3. Commits fréquents et atomiques (au moins 2 par session)
git add modules/module_x/fichier_modifié.py
git commit -m "[X] feat(scope): description courte"

# 4. Pousser régulièrement
git push origin feature/module-x

# 5. Créer une Pull Request vers develop (via GitHub)
```

### Règles Fondamentales

1. **Jamais de push direct sur `main` ou `develop`** — uniquement par PR
2. **Commits atomiques** : un commit = une modification logique cohérente
3. **Au moins 2 commits par session de travail** (minimum 4h)
4. **Toujours partir d'un develop à jour** avant de commencer
5. **Résoudre les conflits sur votre branche**, pas sur develop
6. **Tester avant de créer une PR** : `pytest tests/`

---

## 3. Procédure de Pull Request

### Créer une PR

1. Pousser votre branche : `git push origin feature/module-x`
2. Sur GitHub → "New Pull Request"
3. Base : `develop`, Compare : `feature/module-x`
4. Remplir le template de PR (titre, description, tests effectués)
5. Assigner **1 reviewer** (membre d'un autre module)
6. Attendre la review avant de merger

### Revue de Code (Reviewer)

- Vérifier que le code est commenté en français
- Vérifier qu'un docstring de module est présent
- Lancer les tests : `pytest tests/test_module_x.py`
- Vérifier la convention de nommage des commits
- Approuver ou demander des modifications

### Critères de Merge

- [ ] Au moins 1 approbation d'un reviewer
- [ ] Aucun conflit de merge
- [ ] Tests passants (`pytest` sans erreur)
- [ ] Convention de commits respectée

---

## 4. Merge de Develop vers Main

**Une fois par jour (fin de journée), le chef de projet (Étudiant 5)** :

```bash
# Vérifier que develop est à jour et que les tests passent
git checkout develop
git pull origin develop
pytest tests/

# Créer la PR develop → main sur GitHub
# Titre : "[DAILY] Merge J{X} — {date} — {résumé des modules}"
# Reviewer : n'importe quel autre membre
```

Critères supplémentaires pour merge vers main :
- [ ] Tous les modules du jour fonctionnels
- [ ] Pipeline `python main.py` exécutable sans erreur
- [ ] Rapport et notebooks mis à jour

---

## 5. Gestion des Conflits

```bash
# Si un conflit survient lors du merge de develop dans votre branche :

git checkout feature/module-x
git fetch origin
git merge origin/develop

# Git indique les fichiers en conflit
# Ouvrir le fichier et résoudre :
# <<<<<<< HEAD
# votre code
# =======
# code de develop
# >>>>>>> origin/develop

# Après résolution :
git add fichier_en_conflit.py
git commit -m "[MODULE] fix(merge): résolution conflit avec develop"
git push origin feature/module-x
```

---

## 6. Règles de Protection des Branches (GitHub Settings)

À configurer par le propriétaire du dépôt dans **Settings → Branches** :

### Branche `main`
- ✅ Require a pull request before merging
- ✅ Require 1 approval
- ✅ Dismiss stale pull request approvals when new commits are pushed
- ✅ Do not allow bypassing the above settings
- ❌ Allow force pushes

### Branche `develop`
- ✅ Require a pull request before merging
- ✅ Require 1 approval
- ❌ Allow force pushes

---

## 7. Organisation des Fichiers

Chaque étudiant travaille principalement dans son dossier de module :

| Étudiant | Fichiers principaux |
|----------|---------------------|
| Étudiant 1 | `modules/module_a/naive_bayes.py`, `tests/test_naive_bayes.py` |
| Étudiant 2 | `modules/module_a/yolo_ocr_pipeline.py`, `tests/test_yolo_pipeline.py` |
| Étudiant 3 | `modules/module_b/*.py`, `tests/test_clustering.py` |
| Étudiant 4 | `modules/module_c/*.py`, `tests/test_mdp.py` |
| Étudiant 5 | `modules/module_d/`, `reports/`, coordination `main.py` |

Les fichiers partagés (`main.py`, `preprocessing/`, `requirements.txt`) :
→ Modifications discutées collectivement avant commit.
→ Nommer clairement le commit `[ALL]`.

---

## 8. Checklist Quotidienne

### Début de session
- [ ] `git fetch origin && git merge origin/develop` sur votre branche
- [ ] Vérifier les issues assignées

### Fin de session
- [ ] Tous les changements committés (pas de fichiers modifiés non commités)
- [ ] `git push origin feature/module-x`
- [ ] PR créée si fonctionnalité terminée
- [ ] Mettre à jour le carnet de bord (`modules/module_d/logbook.md`)

---

*Document rédigé par : Étudiant 5 (Chef de projet)*
*Projet PlateVision — UCAC-ICAM / ULC-ICAM*
