---
name: "📦 Problème Dataset / Annotations"
about: Signaler un problème lié aux données, annotations ou prétraitement
title: "[DATA] Description du problème dataset"
labels: ["data", "needs-discussion"]
assignees: ""
---

## Type de Problème

- [ ] 🔗 Lien de téléchargement cassé
- [ ] 📁 Structure de dossiers incorrecte
- [ ] 🏷️ Erreur dans les annotations (labels YOLO ou COCO)
- [ ] ⚖️ Dataset déséquilibré (distribution des classes)
- [ ] 🌙 Qualité insuffisante (nuit, flou, résolution)
- [ ] 📏 Format d'image incompatible
- [ ] 🗺️ Biais géographique identifié
- [ ] Autre : _______________

## Description du Problème

<!-- Décrivez précisément le problème observé avec le dataset. -->

## Fichiers ou Images Concernés

```
# Chemins relatifs des fichiers problématiques
data/raw/images/train/image_xxx.jpg
data/raw/labels/train/image_xxx.txt
```

## Illustration du Problème

<!-- Si possible, joindre une capture d'écran ou un exemple.
     Pour les erreurs d'annotation, coller le contenu du fichier .txt -->

```
# Contenu du fichier d'annotation problématique (format YOLO)
# class_id x_center y_center width height
0 0.532 0.721 0.243 0.142
```

## Impact sur les Performances

<!-- Quel est l'impact de ce problème sur les métriques ?
     (mAP, CER, WER, accuracy Naïves Bayes) -->

## Dataset Actuel

| Information | Valeur |
|-------------|--------|
| Source du dataset | |
| Version / date | |
| Taille totale | |
| Nombre d'images concernées | |

## Solution Proposée

<!-- Quelle solution envisagez-vous ?
     - Correction manuelle des annotations ?
     - Filtrage des images défectueuses ?
     - Augmentation de données ciblée ?
     - Téléchargement d'un dataset complémentaire ? -->

## Commande de Vérification

```bash
# Commande pour reproduire la vérification du problème
python -c "
import cv2
from pathlib import Path

# Vérifier les annotations
..."
```

## Urgence

- [ ] 🔴 Bloquant (empêche l'entraînement)
- [ ] 🟡 Important (dégrade significativement les résultats)
- [ ] 🟢 Mineur (à corriger si le temps le permet)

## Discussion

<!-- Discutez ici avec l'équipe des options de résolution. -->
