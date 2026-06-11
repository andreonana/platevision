# Module D — Analyse Éthique de PlateVision

**Projet :** PlateVision — Système de reconnaissance de plaques d'immatriculation  
**Commanditaires :** MINT / DGI Cameroun  
**Responsable Module D :** Étudiant 5  
**Date :** À compléter

---

## 1. Contexte et enjeux éthiques

PlateVision est un système de surveillance automatisée déployé dans l'espace public
camerounais. Il collecte, analyse et archive des données sur les déplacements des véhicules
et, par extension, des personnes. Ce type de système soulève des questions fondamentales
relatives aux droits individuels, à la gouvernance algorithmique et à l'équité sociale.

---

## 2. Dimension Surveillance et Vie Privée

### 2.1 Collecte de données personnelles
- Les plaques d'immatriculation sont des **données à caractère personnel** (identifiables)
- Chaque passage capturé constitue un **enregistrement de déplacement**
- Risque de constitution de **profils de mobilité** sans consentement explicite

### 2.2 Cadre légal applicable au Cameroun
- Loi n°2010/012 du 21 décembre 2010 relative à la cybersécurité et à la cybercriminalité
- Absence d'une loi spécifique sur la protection des données personnelles au Cameroun (à 2024)
- Référence aux principes du **RGPD** (Europe) comme cadre éthique de référence
- Recommandations de la **Commission de l'Union Africaine** sur la protection des données

### 2.3 Recommandations
- [ ] Définir une durée maximale de conservation des données (exemple : 30 jours)
- [ ] Chiffrement des données en transit et au repos
- [ ] Accès aux données restreint aux agents habilités de la DGI/MINT
- [ ] Mise en place d'un registre de traitement des données

---

## 3. Biais Algorithmiques

### 3.1 Biais dans les données d'entraînement
Le dataset utilisé pour entraîner YOLOv8 peut introduire des biais systématiques :

| Source de biais | Impact potentiel | Mesure corrective |
|-----------------|-----------------|-------------------|
| Surreprésentation Douala/Yaoundé | Sous-performance dans les régions | Collecter images des 10 régions |
| Conditions d'éclairage urbain | Erreurs la nuit ou en zone rurale | Augmentation données nocturnes |
| Qualité des caméras haut de gamme | Biais vers véhicules récents/aisés | Tester sur caméras basse résolution |
| Plaques abîmées/illisibles | Faux positifs sur véhicules vulnérables | Annoter plaques dégradées |

### 3.2 Biais dans la prise de décision (Module C — MDP)
- La matrice de récompenses encode des **jugements de valeur** (ex : CD → laisser passer)
- Risque de discrimination systémique si les récompenses favorisent certaines catégories
- Recommandation : soumettre la matrice R à une **revue éthique indépendante**

### 3.3 Impact disproportionné
- Les erreurs du système (faux positifs) peuvent pénaliser injustement des citoyens
- Les conducteurs de motos-taxis ou véhicules anciens sont plus vulnérables aux erreurs OCR
- Nécessité d'un **mécanisme de recours** humain accessible

---

## 4. Gouvernance de l'IA

### 4.1 Traçabilité et explicabilité
- YOLOv8 est un modèle de type **boîte noire** — ses décisions ne sont pas directement explicables
- Recommandation : implémenter **Grad-CAM** pour visualiser les zones d'attention du modèle
- Le MDP (Module C) est par nature plus interprétable et auditables

### 4.2 Supervision humaine
- Principe d'**Human-in-the-loop** : toute verbalisation doit être validée par un agent humain
- Le système doit rester un outil d'**aide à la décision**, non un décideur autonome
- Seuils de confiance configurables (`.env`) pour moduler le degré d'automatisation

### 4.3 Responsabilité
- **Qui est responsable** d'une verbalisation injustifiée générée par le système ?
- Recommandation : établir une **charte de responsabilité** signée entre MINT, DGI et développeurs
- Documentation de toutes les décisions du modèle (audit trail)

---

## 5. Inclusivité et équité sociale

### 5.1 Accès numérique
- Les agents de terrain doivent être formés à l'utilisation du système
- Prévoir une interface dégradée fonctionnant **hors ligne** (zones sans connexion)
- Documentation en **français et anglais** (Cameroun bilingue)

### 5.2 Impact sur l'emploi
- PlateVision automatise des tâches de contrôle actuellement manuelles
- Risque de pression sur l'emploi des agents de contrôle routier
- Recommandation : positionner le système comme **amplificateur** et non **remplaçant**

---

## 6. Risques spécifiques au contexte africain

| Risque | Probabilité | Impact | Mitigation |
|--------|------------|--------|------------|
| Détournement à des fins de surveillance politique | Moyenne | Très élevé | Séparation claire des usages DGI vs sécurité |
| Corruption dans l'accès aux données | Haute | Élevé | Logs d'accès, authentification forte |
| Déploiement sans phase de test terrain | Haute | Élevé | Pilote de 3 mois obligatoire |
| Biais géographique (régions rurales) | Haute | Moyen | Dataset multirégional |
| Panne système bloquant la circulation | Faible | Élevé | Mode dégradé manuel prévu |

---

## 7. Recommandations finales

1. **Audit éthique indépendant** avant déploiement production
2. **Consentement implicite** clairement affiché aux points de contrôle (signalétique)
3. **Comité de gouvernance mixte** : MINT, DGI, société civile, experts IA
4. **Tests de performance par région** et par catégorie socioprofessionnelle
5. **Plan de réponse aux incidents** (faux positifs, pannes, violations de données)
6. **Révision annuelle** du système et de ses impacts documentés

---

## 8. Références

- Union Africaine — Convention de Malabo sur la Cybersécurité (2014)
- NIST AI Risk Management Framework (2023)
- UNESCO — Recommandation sur l'éthique de l'IA (2021)
- Amnesty International — Surveillance et reconnaissance faciale en Afrique (2022)
- Article 12 de la Déclaration Universelle des Droits de l'Homme (vie privée)
