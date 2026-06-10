# 🚗 PlateVision — Reconnaissance Automatique de Plaques d'Immatriculation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Licence](https://img.shields.io/badge/Licence-Académique-green)
![Institution](https://img.shields.io/badge/Institution-UCAC--ICAM%20%2F%20ULC--ICAM-orange)

---

## Description du Projet

**PlateVision** est un système d'intelligence artificielle pour la **reconnaissance automatique de plaques d'immatriculation** de véhicules, développé dans le cadre du projet académique PSTNAC 2023-2028.

Ce système vise à appuyer les missions du **MINT** (Ministère des Infrastructures et du Transport) et de la **DGI** (Direction Générale des Impôts) du Cameroun pour :
- L'identification automatique des véhicules aux postes de contrôle
- L'automatisation de la vérification de conformité fiscale (vignette, assurance)
- La réduction des fraudes liées aux plaques falsifiées
- La collecte de données statistiques de mobilité urbaine

---

## Contexte Institutionnel

Ce projet s'inscrit dans la **Politique Sectorielle des Transports et des Nouvelles Activités Connexes (PSTNAC 2023-2028)** du Cameroun. Il est développé par des étudiants de **UCAC-ICAM / ULC-ICAM** dans le cadre d'un projet intensif de 4 jours.

Les partenaires institutionnels cibles sont :
| Institution | Utilisation attendue |
|---|---|
| MINT | Contrôle routier, identification véhicules |
| DGI | Vérification fiscale (vignette, patente) |

---

## Architecture des Modules

```
Données Brutes (Images/Vidéos)
         │
         ▼
[Prétraitement] ── augmentation, normalisation, extraction features
         │
         ▼
┌────────────────────────────────────────────┐
│  MODULE A — Détection & Reconnaissance     │
│  A1 : Classifieur Naïves Bayes            │
│  A2 : Pipeline YOLO + OCR                 │
│  Métriques : mAP, CER, WER               │
└────────────────────┬───────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│  MODULE B — Analyse Non Supervisée        │
│  K-Means sur embeddings CNN               │
│  Visualisation PCA / t-SNE               │
│  Interprétation → procédures MINT/DGI    │
└────────────────────┬───────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│  MODULE C — Décision par MDP              │
│  Value Iteration / Policy Iteration       │
│  Comparaison VI vs PI, sensibilité γ      │
└────────────────────┬───────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│  MODULE D — Éthique & Gouvernance         │
│  Analyse biais, surveillance, RGPD        │
│  Carnet de bord J1 → J4                  │
└────────────────────────────────────────────┘
```

---

## Installation

### Prérequis

- Python 3.10 ou supérieur
- Git
- (Recommandé) GPU NVIDIA avec CUDA 11.8+ pour l'entraînement YOLO

### Étapes d'installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/andreonana/platevision.git
cd platevision

# 2. Créer un environnement virtuel
python -m venv venv

# Activer l'environnement (Linux/macOS)
source venv/bin/activate

# Activer l'environnement (Windows CMD)
venv\Scripts\activate.bat

# Activer l'environnement (Windows PowerShell)
venv\Scripts\Activate.ps1

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos chemins locaux
```

---

## Utilisation — Interface en Ligne de Commande

```bash
# Module A — Détection et reconnaissance
python main.py --module A --input data/processed/ --output reports/

# Module B — Clustering et visualisation
python main.py --module B --embeddings models/embeddings.npy

# Module C — MDP (Value Iteration ou Policy Iteration)
python main.py --module C --gamma 0.9 --algorithm VI
python main.py --module C --gamma 0.95 --algorithm PI

# Pipeline complet sur une vidéo
python main.py --pipeline full --input data/raw/video.mp4
```

---

## Structure du Dépôt

```
platevision/
├── data/               # Données (raw non versionnées, processed versionnées)
├── modules/            # Code source par module IA
│   ├── module_a/       # Détection + OCR
│   ├── module_b/       # Clustering
│   ├── module_c/       # MDP
│   └── module_d/       # Éthique + logbook
├── preprocessing/      # Scripts de prétraitement
├── models/             # Poids entraînés et configs YOLO
├── notebooks/          # Jupyter pour expérimentations
├── reports/            # Livrables L1, L2, L3
├── tests/              # Tests unitaires
└── main.py             # Point d'entrée CLI
```

---

## Livrables

| Code | Livrable | Description | Échéance |
|------|----------|-------------|----------|
| L1 | Document de Sélection du Dataset (DSD) | Justification du dataset choisi, statistiques, exemples | Jour 1 |
| L2 | Document de Gestion de Projet | Gantt, PERT, répartition rôles, analyse risques | Jour 1 |
| L3 | Rapport Technique Intégré | Documentation complète modules A→D, résultats, analyse | Jour 4 |
| L4 | Code Source Documenté | Dépôt GitHub complet avec historique de commits | Jour 4 |
| L5 | Présentation Soutenance | Slides + démonstration live du pipeline | Jour 5 |

---

## Équipe

| Étudiant | Rôle | Module(s) responsable(s) | Branche Git |
|----------|------|--------------------------|-------------|
| Étudiant 1 | Développeur IA | Module A1 — Naïves Bayes | `feature/module-a` |
| Étudiant 2 | Développeur IA | Module A2 — YOLO + OCR | `feature/module-a` |
| Étudiant 3 | Développeur ML | Module B — Clustering | `feature/module-b` |
| Étudiant 4 | Développeur IA | Module C — MDP | `feature/module-c` |
| Étudiant 5 | Chef de projet | Module D — Éthique + coordination | `feature/module-d` |

---

## Téléchargement du Dataset

Voir [`data/README_data.md`](data/README_data.md) pour les instructions détaillées.

---

## Licence

Projet académique — Usage éducatif uniquement.
© 2024 UCAC-ICAM / ULC-ICAM — Cameroun
