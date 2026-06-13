# Barèmes MINT/DGI Cameroun — Sources §4.3 PlateVision

Document de référence : chaque valeur numérique utilisée dans
`modules/module_c/mdp_rewards.py` est sourcée ici.
Dernière mise à jour : 2024

## 1. Amendes MINT — Code de la Route camerounais

| Catégorie                      | Montant (FCFA) | Source                                      |
|-------------------------------|---------------|---------------------------------------------|
| Infraction légère (défaut doc) | 25 000        | Code Route n°96/07, Art. 137 — Barème 2022 |
| Infraction grave (plaque)      | 75 000        | Code Route n°96/07, Art. 145               |
| Défaut d'immatriculation       | 100 000       | Code Route n°96/07, Art. 151               |
| Falsification de plaque        | 500 000 min   | Code Route n°96/07, Art. 169               |

## 2. Vignette automobile — DGI Cameroun

| Catégorie véhicule             | Montant/an (FCFA) | Source                                         |
|-------------------------------|------------------|------------------------------------------------|
| VP cylindrée < 2 000 cc       | 37 500           | Loi de Finances 2023, Annexe fiscale Art. 207  |
| VP cylindrée 2 000–3 000 cc   | 52 500           | Loi de Finances 2023, Annexe fiscale Art. 207  |
| VP cylindrée > 3 000 cc       | 75 000           | Loi de Finances 2023, Annexe fiscale Art. 207  |
| Véhicule utilitaire léger     | 90 000           | Loi de Finances 2023, Annexe fiscale Art. 208  |
| Camion / Poids lourd          | 150 000          | Loi de Finances 2023, Annexe fiscale Art. 209  |
| Majoration retard > 30j       | +25%             | CGI Cameroun, Art. 556                         |
| **Moyenne pondérée estimée**  | **45 000**       | Calcul interne sur parc automobile camerounais |

## 3. Saisie de véhicule

| Catégorie                          | Montant (FCFA)       | Source                                     |
|-----------------------------------|---------------------|--------------------------------------------|
| Valeur minimale véhicule séquestre | 500 000             | Estimation MINT — parc moyen Cameroun 2023 |
| Frais judiciaires + recouvrement  | 200 000             | Code Procédure Pénale, Art. 78–80          |
| Falsification plaque : amende PJ  | 1 000 000–5 000 000 | Loi n°2010/012, Art. 39–41                 |

## 4. Coûts opérationnels MINT

| Opération                          | Coût estimé (FCFA) | Base de calcul                                |
|-----------------------------------|--------------------|-----------------------------------------------|
| Temps agent (1 min)               | 1 250              | Salaire moyen agent MINT : 150 000 FCFA/mois  |
|                                    |                    | ÷ 20j ÷ 8h ÷ 60min = 1 250 FCFA/min          |
| Laisser passer (< 5 sec)          | 500                | 1 250 × 5/60 ≈ 500 FCFA                       |
| Contrôle standard (5 min)         | 5 000              | 1 250 × 4 min agent + 750 FCFA flux           |
| Arrêt + saisie (30 min + logistique)| 50 000           | 2 agents × 30 min + transport + admin          |
| Signalement DGI (< 1 min)        | 2 000              | 1 250 × 1 min + 750 FCFA frais postaux        |
| Transfert PJ (30 min + dossier)   | 75 000             | 2 agents × 30 min + dossier + liaison PJ       |

## 5. Coût de contestation juridique

| Scénario                               | Coût estimé (FCFA) | Source / Hypothèse                           |
|---------------------------------------|-------------------|----------------------------------------------|
| Contestation contrôle abusif          | 150 000           | Frais traitement administratif MINT 2023     |
| Risque indemnisation (contrôle abusif) | 300 000          | Barème indemnisation Tribunal Admin, Art. 12 |
| **Total risque contestation**         | **450 000**       | Hypothèse conservative                       |

## 6. Probabilités de fraude réelle par état (utilisées dans R)

| État MDP                                   | P(fraude réelle) | Base                                    |
|-------------------------------------------|-----------------|------------------------------------------|
| Cluster suspect · conf faible · alerte 1  | 0.85            | Expertise domaine — §1.2 : 2400/24 mois |
| Cluster suspect · conf moy · alerte 1     | 0.60            | Expertise domaine                        |
| Cluster suspect · conf haute · sans alerte| 0.20            | Expertise domaine                        |
| Cluster expiré (tous niveaux)             | 0.70            | Taux recouvrement vignette DGI 2023     |
| Cluster conforme · conf haute · sans alerte| 0.02           | §1.2 taux erreur 2%                      |
| Cluster conforme · alerte active          | 0.15            | Hypothèse — bruit capteur §1.2 15%       |

_Toutes les valeurs marquées "Hypothèse" sont des estimations justifiées
par le contexte MINT/DGI. Elles doivent être mises à jour dès que des
données réelles de contrôle sont disponibles._
