# Carnet de Bord — Projet PlateVision

**Équipe :** UCAC-ICAM / ULC-ICAM
**Durée :** 4 jours intensifs (J1 → J4) + Soutenance (J5)
**Module D responsable :** Étudiant 5 (Chef de projet / Éthique)

---

## Jour 1 — Initialisation & Exploration

**Date :** ___________

### Réalisations
- [ ] Réunion de lancement : distribution des rôles et modules
- [ ] Création du dépôt GitHub et configuration des branches
- [ ] Installation des environnements Python (venv, requirements.txt)
- [ ] Sélection et téléchargement du dataset (→ Document L1)
- [ ] Rédaction du Document de Gestion de Projet (Gantt, PERT) (→ L2)
- [ ] Exploration initiale du dataset (notebook 01_eda_dataset.ipynb)

### Difficultés rencontrées
<!-- À compléter -->

### Décisions techniques
<!-- À compléter -->

### Métriques du jour
| Métrique | Valeur |
|---------|--------|
| Commits poussés (total équipe) | |
| Issues ouvertes | |
| Issues fermées | |

### Avancement par module
| Module | Avancement | Responsable |
|--------|-----------|-------------|
| A1 (Naïves Bayes) | ▓░░░░ 20% | Étudiant 1 |
| A2 (YOLO+OCR) | ▓░░░░ 20% | Étudiant 2 |
| B (Clustering) | ▓░░░░ 20% | Étudiant 3 |
| C (MDP) | ▓░░░░ 20% | Étudiant 4 |
| D (Éthique) | ▓░░░░ 20% | Étudiant 5 |

---

## Jour 2 — Développement & Premières Expérimentations

**Date :** ___________

### Réalisations
- [ ] Module A1 : extraction features HSV + entraînement Naïves Bayes baseline
- [ ] Module A2 : configuration YOLOv8, tests sur images de validation
- [ ] Module B : extraction embeddings CNN, premier K-Means
- [ ] Module C : définition formelle du MDP (états, actions, transitions)
- [ ] Module D : analyse éthique section 1-3 rédigée
- [ ] Première PR mergée sur `develop`

### Difficultés rencontrées
<!-- À compléter -->

### Décisions techniques
<!-- À compléter -->

### Métriques du jour
| Métrique | Valeur |
|---------|--------|
| Accuracy Naïves Bayes (baseline) | |
| mAP YOLO@0.5 (baseline) | |
| Score Silhouette K-Means | |
| Commits poussés | |

### Avancement par module
| Module | Avancement | Responsable |
|--------|-----------|-------------|
| A1 (Naïves Bayes) | ▓▓▓░░ 60% | Étudiant 1 |
| A2 (YOLO+OCR) | ▓▓░░░ 40% | Étudiant 2 |
| B (Clustering) | ▓▓░░░ 40% | Étudiant 3 |
| C (MDP) | ▓▓░░░ 40% | Étudiant 4 |
| D (Éthique) | ▓▓░░░ 40% | Étudiant 5 |

---

## Jour 3 — Optimisation & Intégration

**Date :** ___________

### Réalisations
- [ ] Module A : optimisation hyperparamètres, calcul CER/WER
- [ ] Module B : visualisations PCA/t-SNE, interprétation clusters
- [ ] Module C : Value Iteration + Policy Iteration convergés
- [ ] Module C : comparaison VI vs PI, analyse sensibilité γ
- [ ] Intégration partielle dans main.py
- [ ] Tests unitaires (tests/)
- [ ] Merge develop → main (fin J3)

### Difficultés rencontrées
<!-- À compléter -->

### Décisions techniques
<!-- À compléter -->

### Métriques du jour
| Métrique | Valeur |
|---------|--------|
| Accuracy Naïves Bayes (optimisé) | |
| mAP YOLO@0.5 (fine-tuning) | |
| CER moyen OCR | |
| WER moyen OCR | |
| K optimal (silhouette) | |
| Itérations VI convergence | |
| Itérations PI convergence | |

### Avancement par module
| Module | Avancement | Responsable |
|--------|-----------|-------------|
| A1 (Naïves Bayes) | ▓▓▓▓░ 80% | Étudiant 1 |
| A2 (YOLO+OCR) | ▓▓▓▓░ 80% | Étudiant 2 |
| B (Clustering) | ▓▓▓▓░ 80% | Étudiant 3 |
| C (MDP) | ▓▓▓▓░ 80% | Étudiant 4 |
| D (Éthique) | ▓▓▓▓░ 80% | Étudiant 5 |

---

## Jour 4 — Finalisation & Rédaction

**Date :** ___________

### Réalisations
- [ ] Finalisation de tous les modules
- [ ] Pipeline complet `python main.py --pipeline full` fonctionnel
- [ ] Rapport technique L3 rédigé (tous les modules)
- [ ] Analyse éthique complète (Module D)
- [ ] Nettoyage du code et documentation
- [ ] Tests finaux sur dataset de test
- [ ] Merge final develop → main (version v1.0)
- [ ] Préparation slides de soutenance

### Rétrospective de l'équipe
<!-- Ce qui a bien fonctionné -->

<!-- Ce qui aurait pu être amélioré -->

<!-- Leçons apprises -->

### Métriques finales
| Métrique | Valeur |
|---------|--------|
| Total commits (tous membres) | |
| Total PRs mergées | |
| Lignes de code Python | |
| Accuracy Naïves Bayes (final) | |
| mAP YOLO@0.5 (final) | |
| CER moyen final | |
| WER moyen final | |
| K final retenu | |

### Avancement par module
| Module | Avancement | Responsable |
|--------|-----------|-------------|
| A1 (Naïves Bayes) | ▓▓▓▓▓ 100% | Étudiant 1 |
| A2 (YOLO+OCR) | ▓▓▓▓▓ 100% | Étudiant 2 |
| B (Clustering) | ▓▓▓▓▓ 100% | Étudiant 3 |
| C (MDP) | ▓▓▓▓▓ 100% | Étudiant 4 |
| D (Éthique) | ▓▓▓▓▓ 100% | Étudiant 5 |

---

## Notes Transversales

### Problèmes Techniques Récurrents
<!-- Liste des problèmes rencontrés et solutions -->

### Décisions Architecturales
<!-- Décisions importantes prises collectivement -->

### Ressources Utiles
<!-- Liens, papiers, tutoriels utilisés -->
