# Guide de Contribution — PlateVision

> Ce document définit les conventions de travail collaboratif pour l'équipe PlateVision.
> **Chaque membre doit le lire avant de faire son premier commit.**

---

## 1. Convention de nommage des commits

### Format obligatoire

```
[MODULE] type(scope): description courte en impératif présent
```

**Composants :**

| Composant | Description | Exemples |
|-----------|-------------|---------|
| `[MODULE]` | Module concerné | `[A1]` `[A2]` `[B]` `[C]` `[D]` `[ALL]` |
| `type` | Nature du changement | `feat` `fix` `docs` `test` `refactor` `style` |
| `scope` | Fichier ou fonction concernée | `naive_bayes` `yolo` `clustering` `mdp` |
| `description` | Résumé en 50 caractères max | Action + objet, pas de point final |

### Types de commits autorisés

| Type | Usage |
|------|-------|
| `feat` | Nouvelle fonctionnalité |
| `fix` | Correction de bug |
| `docs` | Documentation uniquement |
| `test` | Ajout ou modification de tests |
| `refactor` | Restructuration sans changement de comportement |
| `style` | Formatage, indentation (pas de logique) |
| `chore` | Maintenance (dépendances, config) |

### Exemples de commits valides

```bash
# Module A1 — Naïves Bayes
[A1] feat(naive_bayes): ajout extraction features HSV et Sobel
[A1] fix(naive_bayes): correction division par zéro dans normalisation
[A1] test(naive_bayes): ajout tests unitaires métriques CER/WER

# Module A2 — YOLO + OCR
[A2] feat(yolo): configuration YOLOv8 pour plaques camerounaises
[A2] fix(yolo): correction seuil de confiance OCR trop permissif
[A2] feat(ocr): intégration EasyOCR avec filtrage par confiance

# Module B — Clustering
[B]  feat(clustering): implémentation K-Means sur embeddings CNN
[B]  feat(visualization): génération graphiques PCA et t-SNE
[B]  fix(clustering): correction normalisation StandardScaler

# Module C — MDP
[C]  feat(mdp): définition états et matrice de transitions MINT/DGI
[C]  feat(value_iteration): algorithme VI avec seuil de convergence
[C]  feat(compare_vi_pi): comparaison convergence VI vs PI

# Module D — Éthique et coordination
[D]  docs(ethics): analyse biais algorithmiques dans le dataset
[D]  docs(logbook): entrée carnet de bord jour 2
[D]  docs(ethics): ajout recommandations gouvernance IA

# Tous modules
[ALL] refactor(pipeline): modularisation main.py avec argparse
[ALL] chore(deps): mise à jour requirements.txt
[ALL] docs(readme): ajout instructions d'installation WSL
```

### Commits invalides (à éviter)

```bash
# ❌ Pas de contexte
git commit -m "fix bug"
git commit -m "update code"
git commit -m "wip"

# ❌ Trop vague
git commit -m "[A] travail sur le module"

# ❌ Trop long dans le titre
git commit -m "[B] feat(clustering): implémentation K-Means avec normalisation StandardScaler et visualisation PCA t-SNE"
```

---

## 2. Workflow Git de l'équipe

### Structure des branches

```
main ──────────────────────────────────── (production, protégée)
  │
  └── develop ───────────────────────────── (intégration quotidienne)
          │
          ├── feature/module-a ─────────── (Étudiants 1 & 2)
          ├── feature/module-b ─────────── (Étudiant 3)
          ├── feature/module-c ─────────── (Étudiant 4)
          └── feature/module-d ─────────── (Étudiant 5)
```

### Flux de travail quotidien

**Étape 1 — Se mettre à jour en début de session**
```bash
git checkout feature/module-x
git fetch origin
git rebase origin/develop
```

**Étape 2 — Travailler par petits commits atomiques**
```bash
# Modifier vos fichiers...
git add modules/module_x/mon_fichier.py
git commit -m "[X] feat(scope): description"
# Répéter pour chaque changement logique distinct
```

**Étape 3 — Pousser votre branche**
```bash
git push origin feature/module-x
```

**Étape 4 — Créer une Pull Request vers `develop`**
1. Aller sur GitHub → votre branche → "Compare & pull request"
2. Titre : `[MODULE] Brève description des changements`
3. Description : lister les fonctionnalités ajoutées, tests effectués
4. Assigner un reviewer (un autre membre de l'équipe)
5. Attendre la review et les retours

**Étape 5 — Review et merge vers `develop`**
- Le reviewer vérifie le code, laisse des commentaires
- L'auteur corrige si nécessaire (nouveaux commits)
- Quand tout est OK → merge via GitHub (pas en ligne de commande)

**Étape 6 — Merge `develop` → `main` (fin de journée uniquement)**
- Effectué par l'Étudiant 5 (chef de projet) après validation de toute l'équipe
- Créer une PR `develop` → `main` avec tag de version `vJ1`, `vJ2`, etc.

---

## 3. Règles absolues

### Ne JAMAIS faire
```bash
# ❌ Push direct sur main
git push origin main

# ❌ Force push sur une branche partagée
git push --force origin feature/module-a

# ❌ Commit de fichiers volumineux ou secrets
git add models/weights/*.pt        # ❌ fichiers modèles
git add data/raw/                  # ❌ données brutes
git add .env                       # ❌ variables d'environnement

# ❌ Committer des notebooks avec outputs lourds
# Nettoyer avant : jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

### Toujours faire
```bash
# ✅ Vérifier avant de committer
git status
git diff --staged

# ✅ Commits atomiques (un seul changement logique à la fois)
git add -p   # Pour sélectionner par hunks

# ✅ Nettoyer les notebooks avant commit
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb

# ✅ S'assurer que les tests passent avant PR
pytest tests/ -v
```

---

## 4. Configuration des protections de branches (GitHub)

L'Étudiant 5 (chef de projet) doit configurer sur GitHub :

### Branche `main`
1. Settings → Branches → Add branch ruleset
2. Nom : `Protect main`
3. Règles :
   - ☑ Require a pull request before merging
   - ☑ Require 1 approving review
   - ☑ Dismiss stale pull request approvals when new commits are pushed
   - ☑ Block force pushes
   - ☑ Restrict deletions

### Branche `develop`
1. Settings → Branches → Add branch ruleset
2. Nom : `Protect develop`
3. Règles :
   - ☑ Require a pull request before merging
   - ☑ Block force pushes

---

## 5. Résolution des conflits

En cas de conflit lors du rebase :

```bash
# 1. Voir les fichiers en conflit
git status

# 2. Ouvrir le fichier et résoudre manuellement les marqueurs <<<, ===, >>>

# 3. Marquer comme résolu
git add fichier_resolu.py

# 4. Continuer le rebase
git rebase --continue

# En cas d'urgence : abandonner le rebase
git rebase --abort
```

---

## 6. Calendrier Git recommandé

| Moment | Action |
|--------|--------|
| Début de session | `git fetch` + `git rebase origin/develop` |
| Toutes les 30-45 min | Commit atomique sur votre branche |
| Fin de session | Push de votre branche |
| Fin de demi-journée | PR vers `develop` + review |
| Fin de journée (J1→J4) | Merge `develop` → `main` par l'Étudiant 5 |

---

*Pour toute question, ouvrir une issue avec le template `question_jury.md` ou discuter en équipe.*
