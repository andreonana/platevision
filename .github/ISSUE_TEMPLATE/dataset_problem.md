---
name: "📁 Problème dataset / annotations"
about: Signaler un problème lié au dataset, aux annotations ou aux données
title: "[DATA] "
labels: data
assignees: ""
---

## Type de problème

- [ ] Données manquantes ou incomplètes
- [ ] Annotations incorrectes ou mal formatées
- [ ] Déséquilibre des classes dans le split train/val/test
- [ ] Images corrompues ou illisibles
- [ ] Biais détecté dans le dataset (géographique, luminosité, etc.)
- [ ] Problème de téléchargement (instructions `data/README_data.md`)
- [ ] Format incompatible (YOLO vs COCO vs Pascal VOC)
- [ ] Autre : ___________

## Description du problème

> Décrivez précisément le problème avec le dataset.

## Localisation du problème

- **Dossier concerné :** `data/raw/` / `data/processed/` / `data/annotations/`
- **Fichiers spécifiques :**
  ```
  data/raw/train/images/nom_image_problematique.jpg
  data/annotations/nom_annotation.txt
  ```
- **Nombre de fichiers affectés :** ___

## Impact sur le pipeline

> Quel module ou quelle étape est impacté ?
- [ ] Preprocessing (`preprocessing/augmentation.py`)
- [ ] Entraînement Module A2 (YOLO)
- [ ] Extraction features Module A1 (Naïves Bayes)
- [ ] Embeddings Module B
- [ ] Évaluation métriques (mAP, CER, WER)

## Exemple concret

### Fichier d'annotation problématique (exemple)
```
# Contenu actuel du fichier .txt :
0 1.23 0.45 0.30 0.15    # ❌ x_center > 1.0 (invalide)

# Devrait être :
0 0.52 0.45 0.30 0.15    # ✅
```

### Message d'erreur lié (si applicable)
```
Coller l'erreur Python ou YOLO ici
```

## Statistiques du dataset actuel

> À remplir pour mieux comprendre l'ampleur du problème :

| Partition | Nb images total | Nb images correctes | Nb images problématiques |
|-----------|----------------|---------------------|--------------------------|
| Train     | ___ | ___ | ___ |
| Val       | ___ | ___ | ___ |
| Test      | ___ | ___ | ___ |

## Distribution des classes (si déséquilibre)

```
Classe 0 (license_plate) : ___ instances
# Autres classes si applicable...
```

## Solution proposée

> Comment corriger ce problème ?
- [ ] Re-annoter les images concernées
- [ ] Supprimer les images corrompues
- [ ] Appliquer une augmentation de données ciblée
- [ ] Télécharger un dataset complémentaire
- [ ] Modifier le script de chargement

## Lien vers la documentation du dataset

> Référencer la source originale du dataset si applicable.
- Source : 
- Licence :
- DOI / URL :

---
*Assigné à :* Étudiant 3 (Module B — Data)  
*Urgence :* [ ] Bloquant  [ ] Important  [ ] Peut attendre
