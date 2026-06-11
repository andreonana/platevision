---
name: "✨ Nouvelle fonctionnalité module"
about: Proposer ou planifier l'ajout d'une fonctionnalité sur un module IA
title: "[FEAT] "
labels: enhancement
assignees: ""
---

## Fonctionnalité proposée

> Décrivez clairement la fonctionnalité à implémenter. Quel est le besoin ?

## Module cible

- [ ] Module A1 — Naïves Bayes (`modules/module_a/naive_bayes.py`)
- [ ] Module A2 — YOLO + OCR (`modules/module_a/yolo_ocr_pipeline.py`)
- [ ] Module A — Évaluation (`modules/module_a/evaluate.py`)
- [ ] Module B — Clustering (`modules/module_b/clustering.py`)
- [ ] Module B — Visualisation (`modules/module_b/visualization.py`)
- [ ] Module B — Interprétation (`modules/module_b/interpret_clusters.py`)
- [ ] Module C — Définition MDP (`modules/module_c/mdp_definition.py`)
- [ ] Module C — Value Iteration (`modules/module_c/value_iteration.py`)
- [ ] Module C — Policy Iteration (`modules/module_c/policy_iteration.py`)
- [ ] Module C — Comparaison VI/PI (`modules/module_c/compare_vi_pi.py`)
- [ ] Module D — Éthique / Carnet de bord
- [ ] Preprocessing
- [ ] Pipeline principal (`main.py`)

## Motivation et contexte

> Pourquoi cette fonctionnalité est-elle nécessaire ?
> - Quel problème résout-elle ?
> - Quel livrable (L1-L5) impacte-t-elle ?

## Description technique

### Comportement attendu
```
python main.py --module X --nouvelle-option valeur
```

### Signature de la fonction envisagée (si applicable)
```python
def nouvelle_fonction(param1: type, param2: type) -> type:
    """Description en français."""
    ...
```

### Algorithme ou approche
> Décrire l'approche technique : quel algorithme, quelle bibliothèque, quel format de données ?

## Critères d'acceptation (Definition of Done)

- [ ] Code implémenté et fonctionnel
- [ ] Tests unitaires ajoutés dans `tests/`
- [ ] Tests pytest passent (`pytest tests/ -v`)
- [ ] Notebook Jupyter mis à jour si applicable
- [ ] Métriques documentées (si applicable)
- [ ] PR mergée vers `develop` après review

## Métriques cibles (si applicable)

| Métrique | Valeur cible |
|----------|-------------|
| (ex: mAP) | > 0.7 |
| (ex: Silhouette) | > 0.4 |

## Dépendances

> Cette fonctionnalité dépend-elle d'une autre issue ou d'un autre module ?
> - Bloqué par : #___
> - Bloque : #___

## Étudiant responsable

> Qui implémente cette fonctionnalité ?
- **Assigné à :** Étudiant ___
- **Branche :** `feature/module-x`
- **Échéance :** Jour ___

---
*Estimation de complexité :* [ ] Simple (< 2h)  [ ] Moyenne (2-4h)  [ ] Complexe (> 4h)
