# Analyse des Correspondances Multiples à la Bourdieu et Autoencodeur

Ce projet réalise une Analyse des Correspondances Multiples (ACM) selon l'approche de Pierre Bourdieu et compare cette méthode classique avec un autoencodeur neuronal, qui peut être considéré comme une généralisation non-linéaire de l'ACM.

## Données

L'analyse utilise les données de la vague 11 de l'European Social Survey (ESS11), un sondage international couvrant différents pays européens. Ces données contiennent des informations sur les caractéristiques sociodémographiques, les pratiques culturelles, les opinions politiques et les valeurs des répondants.

## Structure du projet

- `data/` : Contient les données brutes (ESS11.csv) et traitées (ess11_acm.csv)
- `scripts/` : Scripts Python pour l'analyse
  - `prepare_data.py` : Prépare et filtre les données pour l'ACM
  - `acm_analysis.py` : Réalise l'ACM classique avec la bibliothèque prince
  - `autoencoder_acm.py` : Implémente un autoencodeur à deux dimensions pour comparer avec l'ACM
- `run_analysis.py` : Script principal pour exécuter l'analyse complète

## Variables sélectionnées

Les variables ont été sélectionnées pour capturer différentes dimensions de l'espace social selon Bourdieu :

1. **Capital économique** : statut d'emploi (work_status), profession (occupation)
2. **Capital culturel** : niveau d'éducation (eduyrs_cat), utilisation d'Internet (internet_use)
3. **Variables dispositionnelles** : orientation politique (polscale), confiance dans les institutions (trust_cat)
4. **Variables sociodémographiques** : genre (gndr), âge (age_cat), pays (cntry)

## Prérequis

Pour exécuter ce projet, vous aurez besoin des packages Python suivants :
```
pandas
numpy
matplotlib
scikit-learn
prince (pour l'ACM)
torch (pour l'autoencodeur)
seaborn
```

Vous pouvez les installer avec pip :
```bash
pip install pandas numpy matplotlib scikit-learn prince torch seaborn
```

## Exécution de l'analyse

Pour exécuter l'analyse complète en une seule fois :

```bash
python run_analysis.py
```

⚠️ **Note importante** : L'exécution complète peut prendre 15-20 minutes, principalement en raison de l'entraînement de l'autoencodeur.

Cette commande exécutera séquentiellement :
1. La préparation des données
2. L'ACM classique
3. L'analyse par autoencodeur

Si vous préférez exécuter les scripts individuellement :

```bash
# Préparation des données
python scripts/prepare_data.py

# ACM classique
python scripts/acm_analysis.py

# Autoencodeur
python scripts/autoencoder_acm.py
```

## Résultats produits

L'analyse génère plusieurs graphiques :

### ACM classique
- `acm_plan_factoriel.png` : Projection des modalités des variables sur les deux premiers axes
- `acm_individus_par_pays.png` : Projection des individus colorés par pays

### Autoencodeur
- `autoencoder_loss.png` : Courbe de perte pendant l'entraînement
- `autoencoder_latent_space.png` : Espace latent de l'autoencodeur par pays
- `autoencoder_plan_factoriel.png` : Projection des modalités dans l'espace latent (comparable à acm_plan_factoriel.png)
- `autoencoder_by_age.png` : Espace latent par groupe d'âge
- `autoencoder_by_education.png` : Espace latent par niveau d'éducation
- `autoencoder_by_polscale.png` : Espace latent par orientation politique

## Interprétation des résultats

### Plan factoriel de l'ACM
L'ACM produit un plan factoriel où les axes peuvent être interprétés en termes d'opposition entre modalités :
- **Premier axe** : généralement associé au volume global de capital (opposition entre classes sociales supérieures et populaires)
- **Deuxième axe** : généralement associé à la structure du capital (opposition entre capital culturel et économique)

### Plan factoriel de l'autoencodeur
L'autoencodeur produit un espace latent à deux dimensions qui, bien que moins directement interprétable, peut révéler des structures similaires à celles de l'ACM mais en capturant potentiellement des relations non-linéaires.

## Comparaison entre ACM et Autoencodeur

### Différences principales 
1. **Linéarité vs. Non-linéarité** : L'ACM est une méthode linéaire, tandis que l'autoencodeur peut modéliser des relations non-linéaires.
2. **Profondeur** : L'autoencodeur utilise plusieurs couches cachées, ce qui permet d'apprendre des représentations hiérarchiques.
3. **Interprétabilité** : Les axes de l'ACM ont une interprétation directe en termes d'opposition entre modalités, tandis que les dimensions latentes de l'autoencodeur sont plus difficiles à interpréter.
4. **Capacité de reconstruction** : L'autoencodeur est conçu pour reconstruire les données d'origine, ce qui peut être utilisé comme mesure de qualité.

### Perspectives bourdiesiennes
Dans un contexte bourdieusien, les deux méthodes permettent de visualiser l'espace social selon deux dimensions principales :
1. **Volume global de capital** (axe vertical) : opposition entre les classes supérieures et populaires
2. **Structure du capital** (axe horizontal) : opposition entre capital économique et capital culturel

L'autoencodeur pourrait révéler des structures plus subtiles ou non-linéaires dans l'espace social que l'ACM traditionnelle ne capture pas.

## Comment comparer les résultats
Pour comparer les résultats de l'ACM et de l'autoencodeur, examinez les graphiques `acm_plan_factoriel.png` et `autoencoder_plan_factoriel.png` côte à côte. Observez :
- La position relative des modalités dans l'espace
- Les oppositions structurantes (ex: éducation élevée vs faible, âge jeune vs âgé)
- Les proximités entre modalités qui définissent des "styles de vie"

Ces comparaisons permettent d'évaluer si l'autoencodeur capture les mêmes structures sociales que l'ACM ou s'il révèle des relations différentes.

## Modifications possibles

Pour adapter cette analyse à vos besoins, vous pouvez :

1. **Modifier les variables** : Éditez `scripts/prepare_data.py` pour sélectionner différentes variables
2. **Ajuster l'autoencodeur** : Dans `scripts/autoencoder_acm.py`, modifiez l'architecture du réseau (couches, taille, etc.)
3. **Changer les paramètres de l'ACM** : Dans `scripts/acm_analysis.py`, ajustez les paramètres de l'analyse

## Extensions possibles

- Comparer avec d'autres méthodes de réduction de dimensionnalité (t-SNE, UMAP)
- Ajouter des variables supplémentaires comme capital social ou pratiques culturelles
- Développer un autoencodeur variationnel pour une représentation probabiliste de l'espace social