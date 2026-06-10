# Analyse Éthique — PlateVision

**Module D — Document d'analyse éthique**
Projet PlateVision — UCAC-ICAM / ULC-ICAM — MINT/DGI Cameroun

---

## 1. Contexte et Enjeux Éthiques

PlateVision est un système de surveillance automatique des véhicules. Son déploiement
soulève des questions fondamentales d'éthique, de vie privée et de gouvernance des données
dans un État de droit.

---

## 2. Risques Liés à la Surveillance

### 2.1 Surveillance de Masse
- Le système permet de tracer les déplacements des véhicules à large échelle
- Risque de profilage comportemental sans consentement explicite
- Absence de cadre légal spécifique aux systèmes ALPR au Cameroun

**Recommandation :** Limiter la collecte aux données strictement nécessaires aux missions MINT/DGI.
Définir des durées de conservation maximales (ex : 30 jours pour les données non litigieuses).

### 2.2 Accès aux Données
- Qui peut interroger la base de données de passages ?
- Risque d'utilisation détournée à des fins politiques ou policières hors mandat
- Nécessité d'un journal d'audit (qui a accédé à quoi, quand)

---

## 3. Biais Algorithmiques

### 3.1 Biais dans le Dataset
| Source de biais | Description | Impact |
|----------------|-------------|--------|
| Biais géographique | Dataset majoritairement constitué de plaques de Yaoundé/Douala | Mauvaise reconnaissance des plaques de l'Adamaoua, du Nord, de l'Est |
| Biais temporel | Images captées en journée uniquement | Faux négatifs élevés la nuit et par temps de pluie |
| Biais de format | Sur-représentation des véhicules particuliers | Mauvaise performance sur motos et véhicules lourds |
| Biais linguistique | EasyOCR entraîné sur texte latin standard | Confusion sur les polices camerounaises non standard |

### 3.2 Impact Différentiel
- Un système moins performant sur certaines catégories (motos, régions) peut conduire à :
  - Un ciblage disproportionné de certaines populations
  - Des injustices dans l'application des contrôles fiscaux

**Recommandation :** Auditer les métriques de performance (CER, WER, mAP) par sous-groupe
(région, type de véhicule, heure de la journée) avant tout déploiement.

---

## 4. Gouvernance des Données

### 4.1 Cadre Légal Applicable
- **Loi n° 2010/012 du 21 décembre 2010** relative à la cybersécurité et la cybercriminalité au Cameroun
- **Décision DGI 2021** sur la dématérialisation des services fiscaux
- Référence internationale : **RGPD (UE)** comme standard de bonnes pratiques

### 4.2 Principes de Protection des Données
| Principe | Application PlateVision |
|---------|------------------------|
| Minimisation | Collecter uniquement numéro de plaque, horodatage, localisation du poste |
| Finalité | Usage limité aux contrôles MINT et DGI, pas de croisement avec d'autres bases |
| Sécurité | Chiffrement des données en transit et au repos |
| Transparence | Information des usagers aux postes de contrôle équipés |
| Droit d'accès | Procédure permettant à un propriétaire de contester une erreur |

### 4.3 Responsabilités
- **MINT** : responsable du traitement des données de contrôle routier
- **DGI** : responsable du traitement des données fiscales
- **Opérateur technique** (intégrateur) : sous-traitant avec obligations contractuelles

---

## 5. Risques de Détournement

### 5.1 Scénarios de Détournement
1. Utilisation du système pour surveiller des journalistes ou opposants politiques
2. Revente des données de traçage à des tiers commerciaux
3. Couplage non autorisé avec des systèmes de reconnaissance faciale

### 5.2 Mesures de Mitigation
- Séparation physique des bases MINT et DGI (pas de base centrale unifiée)
- Habilitation à niveaux pour l'accès aux données (principe du moindre privilège)
- Audit indépendant annuel par une instance nationale (ANTIC ou future APDP camerounaise)
- Interdiction contractuelle de revente ou cession des données

---

## 6. Recommandations Finales

1. **Court terme (déploiement pilote)** : Limiter à 3 postes de contrôle, avec évaluation
   d'impact pendant 6 mois avant extension.

2. **Moyen terme** : Faire adopter un décret spécifique aux systèmes ALPR dans le cadre
   de la révision de la loi cybersécurité 2010.

3. **Long terme** : Créer une Autorité de Protection des Données Personnelles (APDP)
   indépendante au Cameroun, sur le modèle de la CNIL française.

4. **Technique** : Implémenter la **confidentialité différentielle** pour les statistiques
   agrégées publiées (données de mobilité urbaine).

---

## 7. Conclusion

PlateVision offre des bénéfices réels pour la mobilisation fiscale et la sécurité routière
au Cameroun. Cependant, son déploiement sans cadre éthique et légal solide risque d'éroder
la confiance citoyenne et de créer des injustices systémiques. L'équipe recommande une
approche progressive, transparente et supervisée par des instances indépendantes.

---

*Rédigé par : Étudiant 5 (Module D) — UCAC-ICAM / ULC-ICAM*
*Dernière mise à jour : Jour 4*
