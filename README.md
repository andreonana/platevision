# PlateVision 🚗

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Licence](https://img.shields.io/badge/Licence-Académique%20UCAC--ICAM-green)
![Statut](https://img.shields.io/badge/Statut-En%20développement-orange)

> **Système de Reconnaissance Automatique de Plaques d'Immatriculation**
> Projet IA — UCAC-ICAM / ULC-ICAM — MINT & DGI Cameroun

---

## Contexte institutionnel

PlateVision s'inscrit dans le cadre de la **Politique Sectorielle des Technologies Numériques et de l'Administration en Ligne du Cameroun (PSTNAC 2023–2028)**, portée par le **Ministère des Postes et Télécommunications (MINT)** et la **Direction Générale des Impôts (DGI)**.

L'objectif est de doter les services publics camerounais d'un système intelligent capable de :
- **Détecter** automatiquement les plaques d'immatriculation sur des images ou flux vidéo
- **Reconnaître** les caractères alphanumériques par OCR
- **Classifier** les véhicules selon les procédures MINT/DGI
- **Optimiser** les flux de contrôle grâce à un modèle de décision séquentielle (MDP)
- **Analyser** les implications éthiques du déploiement en contexte africain

---

## Architecture du projet — Progression des modules

```
Données brutes (images/vidéos)
        │
        ▼
┌───────────────────┐
│  PREPROCESSING    │  Augmentation, normalisation, extraction de features
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────────────────────────────┐
│  MODULE A — Détection & Reconnaissance                    │
│  A1 : Naïves Bayes (classification baseline)              │
│  A2 : YOLOv8 + EasyOCR (pipeline de production)          │
│  Métriques : mAP, CER, WER                                │
└────────┬──────────────────────────────────────────────────┘
         │  embeddings CNN
         ▼
┌───────────────────────────────────────────────────────────┐
│  MODULE B — Analyse non supervisée                        │
│  K-Means sur embeddings CNN                               │
│  Visualisation PCA / t-SNE                                │
│  Correspondance clusters → procédures MINT/DGI            │
└────────┬──────────────────────────────────────────────────┘
         │  politique de décision
         ▼
┌───────────────────────────────────────────────────────────┐
│  MODULE C — Prise de décision (MDP)                       │
│  Définition états, actions, récompenses                   │
│  Value Iteration vs Policy Iteration                      │
│  Analyse de sensibilité au facteur γ                      │
└────────┬──────────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────────────┐
│  MODULE D — Éthique & Gouvernance IA                      │
│  Analyse biais, surveillance, conformité                  │
│  Carnet de bord quotidien                                 │
└───────────────────────────────────────────────────────────┘
```

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/andreonana/platevision.git
cd platevision
```

### 2. Créer un environnement virtuel

```bash
# Linux / macOS / WSL
python3.10 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

```bash
cp .env.example .env
# Éditer .env avec vos chemins locaux
```

### 5. Télécharger le dataset

Consulter `data/README_data.md` pour les instructions de téléchargement.

---

## Utilisation — Interface ligne de commande

```bash
# Module A — Détection et reconnaissance (Naïves Bayes baseline)
python main.py --module A --input data/processed/ --output reports/

# Module A — Pipeline YOLO + OCR complet
python main.py --module A2 --input data/processed/ --output reports/

# Module B — Clustering sur embeddings
python main.py --module B --embeddings models/embeddings.npy

# Module C — MDP avec Value Iteration, γ=0.9
python main.py --module C --gamma 0.9 --algorithm VI

# Module C — MDP avec Policy Iteration
python main.py --module C --gamma 0.95 --algorithm PI

# Pipeline complet sur une vidéo
python main.py --pipeline full --input data/raw/video.mp4
```

---

## Livrables

| Code | Livrable | Description | Échéance |
|------|----------|-------------|----------|
| **L1** | Document de Sélection du Dataset (DSD) | Justification du choix du dataset, statistiques, split | Jour 1 |
| **L2** | Plan de Gestion de Projet | Gantt, PERT, rôles, matrice des risques | Jour 1 |
| **L3** | Rapport Technique Intégré | Modules A → D, résultats, analyses | Jour 4 |
| **L4** | Code Source & Dépôt Git | Historique propre, branches, tests | Jour 4 |
| **L5** | Présentation Soutenance | Slides, démo live, réponses jury | Jour 5 |

---

## Équipe

| Étudiant | Rôle | Module(s) | Branche Git |
|----------|------|-----------|-------------|
| Étudiant 1 | Ingénieur ML — Détection | Module A1 (Naïves Bayes) | `feature/module-a` |
| Étudiant 2 | Ingénieur ML — Vision | Module A2 (YOLO + OCR) | `feature/module-a` |
| Étudiant 3 | Ingénieur Data | Module B (Clustering) | `feature/module-b` |
| Étudiant 4 | Ingénieur IA | Module C (MDP) | `feature/module-c` |
| Étudiant 5 | Chef de projet & Éthique | Module D + coordination | `feature/module-d` |

---

## Structure du dépôt

```
platevision/
├── data/               # Données (raw non versionnées)
├── modules/            # Code source des 4 modules IA
├── preprocessing/      # Pipeline de prétraitement
├── models/             # Poids et configurations des modèles
├── notebooks/          # Notebooks Jupyter d'expérimentation
├── reports/            # Livrables documents (DSD, Gantt, rapport)
├── tests/              # Tests unitaires pytest
├── main.py             # Point d'entrée CLI
└── requirements.txt    # Dépendances versionnées
```

---

## Références

- YOLOv8 : [Ultralytics Documentation](https://docs.ultralytics.com)
- EasyOCR : [GitHub JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)
- PSTNAC 2023-2028 : Ministère des Postes et Télécommunications du Cameroun
- MINT Cameroun : [www.minpostel.gov.cm](http://www.minpostel.gov.cm)

---

*Projet académique — UCAC-ICAM / ULC-ICAM — 2026*
