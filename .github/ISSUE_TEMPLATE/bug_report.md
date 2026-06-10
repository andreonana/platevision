---
name: "🐛 Rapport de Bug"
about: Signaler un bug dans le pipeline PlateVision
title: "[BUG][MODULE_X] Description courte du problème"
labels: ["bug", "needs-triage"]
assignees: ""
---

## Description du Bug

<!-- Décrivez clairement le problème observé. -->

## Module Concerné

- [ ] Module A1 — Naïves Bayes
- [ ] Module A2 — YOLO + OCR
- [ ] Module B  — Clustering
- [ ] Module C  — MDP
- [ ] Preprocessing
- [ ] Pipeline principal (main.py)
- [ ] Autre : _______________

## Commande ou Code pour Reproduire

```bash
# Commande exacte qui produit l'erreur
python main.py --module A --input data/processed/
```

```python
# Ou extrait de code Python
```

## Message d'Erreur Complet

```
# Coller ici le traceback complet
Traceback (most recent call last):
  ...
ErrorType: message d'erreur
```

## Comportement Attendu

<!-- Que devrait-il se passer normalement ? -->

## Comportement Observé

<!-- Que se passe-t-il réellement ? -->

## Environnement

- **OS** : (ex. Ubuntu 22.04, Windows 11, macOS 14)
- **Python version** : `python --version`
- **GPU** : (ex. NVIDIA RTX 3060, CPU uniquement)
- **Branch Git** : (ex. `feature/module-a`)
- **Commit** : `git rev-parse --short HEAD`

## Dataset / Données

- Taille du dataset utilisé :
- Format des images :
- Toutes les images prétraitées ? (oui/non)

## Tentatives de Résolution

<!-- Qu'avez-vous déjà essayé pour corriger le problème ? -->

## Priorité

- [ ] 🔴 Bloquant (empêche de continuer)
- [ ] 🟡 Important (dégrade les résultats)
- [ ] 🟢 Mineur (amélioration souhaitable)
