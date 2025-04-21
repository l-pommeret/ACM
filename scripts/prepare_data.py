#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# Charger les données
print("Chargement des données ESS11.csv...")
df = pd.read_csv("data/ESS11.csv")
print(f"Dimensions initiales: {df.shape}")

# Sélectionner les variables pertinentes pour une ACM à la Bourdieu
# Variables sociodémographiques
socio_vars = ['gndr', 'agea', 'edulvlb', 'eisced', 'eduyrs']

# Statut professionnel et classe sociale
work_vars = ['pdwrk', 'uempla', 'isco08', 'emplrel', 'emplno', 'jbspv']

# Capital économique
eco_vars = ['hincsrca', 'hinctnta']

# Capital culturel et social
cultural_vars = ['netusoft', 'netustm', 'sclmeet', 'sclact']

# Attitudes et dispositions
attitude_vars = ['lrscale', 'polintr', 'trstprl', 'trstlgl', 'trstplc', 'happy', 'health']

# Regrouper toutes les variables
selected_vars = socio_vars + work_vars + eco_vars + cultural_vars + attitude_vars + ['cntry']

# Sélectionner les colonnes pertinentes et supprimer les lignes avec trop de valeurs manquantes
print("Filtrage des données...")
df_filtered = df[selected_vars].copy()

# Convertir les codes spéciaux en NaN (6666, 7777, 8888, 9999, 77, 88, 99 sont souvent des codes pour les valeurs manquantes)
for col in df_filtered.columns:
    if df_filtered[col].dtype != 'object':  # Seulement pour les colonnes numériques
        # Remplacer les valeurs spéciales par NaN
        df_filtered[col] = df_filtered[col].replace([6666, 7777, 8888, 9999, 66, 77, 88, 99], np.nan)

# Supprimer les lignes avec plus de 30% de valeurs manquantes
threshold = 0.3 * len(df_filtered.columns)
df_filtered = df_filtered.dropna(thresh=threshold)
print(f"Dimensions après filtrage: {df_filtered.shape}")

# Discrétiser les variables numériques en classes pour l'ACM
# Âge
df_filtered['age_cat'] = pd.cut(df_filtered['agea'], 
                               bins=[0, 25, 40, 55, 70, 100], 
                               labels=['18-25', '26-40', '41-55', '56-70', '71+'])

# Niveau d'éducation (en années)
df_filtered['eduyrs_cat'] = pd.cut(df_filtered['eduyrs'], 
                                  bins=[0, 9, 12, 15, 20, 30], 
                                  labels=['<9', '9-12', '13-15', '16-20', '20+'])

# Orientation politique (gauche-droite)
df_filtered['polscale'] = pd.cut(df_filtered['lrscale'], 
                                bins=[0, 3, 5, 7, 11], 
                                labels=['Gauche', 'Centre-gauche', 'Centre-droite', 'Droite'])

# Confiance dans les institutions (moyenne de trstprl, trstlgl, trstplc)
trust_cols = ['trstprl', 'trstlgl', 'trstplc']
df_filtered['trust_inst'] = df_filtered[trust_cols].mean(axis=1)
df_filtered['trust_cat'] = pd.cut(df_filtered['trust_inst'], 
                                 bins=[0, 3, 5, 7, 11], 
                                 labels=['Très faible', 'Faible', 'Élevée', 'Très élevée'])

# Variables indicatrices pour le statut professionnel
work_status_cols = ['pdwrk', 'uempla']
df_filtered['work_status'] = 'Autre'
df_filtered.loc[df_filtered['pdwrk'] == 1, 'work_status'] = 'Employé'
df_filtered.loc[df_filtered['uempla'] == 1, 'work_status'] = 'Chômeur'

# Encodage des variables catégorielles
cat_vars = ['gndr', 'cntry', 'edulvlb', 'eisced', 'work_status', 'age_cat', 
            'eduyrs_cat', 'polscale', 'trust_cat']

# Nettoyer la variable de fréquence d'utilisation d'Internet
df_filtered['internet_use'] = pd.cut(df_filtered['netusoft'], 
                                   bins=[-1, 0, 2, 4, 6], 
                                   labels=['Jamais', 'Rarement', 'Régulièrement', 'Quotidiennement'])

# Nettoyer les catégories de professions ISCO-08
# Convertir les 4 chiffres de l'ISCO en groupes professionnels majeurs (1er chiffre)
def isco_to_major_group(isco):
    if pd.isna(isco):
        return np.nan
    isco_str = str(int(isco))
    if len(isco_str) >= 1:
        major_group = int(isco_str[0])
        if 1 <= major_group <= 9:
            return major_group
    return np.nan

df_filtered['isco_major'] = df_filtered['isco08'].apply(isco_to_major_group)
df_filtered['occupation'] = df_filtered['isco_major'].map({
    1: 'Cadres/Dirigeants',
    2: 'Professions intellectuelles',
    3: 'Professions intermédiaires',
    4: 'Employés administratifs',
    5: 'Services/Vente',
    6: 'Agriculture/Pêche',
    7: 'Artisans/Métiers',
    8: 'Opérateurs/Assembleurs',
    9: 'Professions élémentaires'
})

# Sélectionner les variables finales pour l'ACM
acm_vars = ['gndr', 'age_cat', 'eduyrs_cat', 'cntry', 'work_status', 
            'occupation', 'polscale', 'trust_cat', 'internet_use']

# Créer un DataFrame final avec seulement les variables pour l'ACM
df_acm = df_filtered[acm_vars].copy()

# Gérer les valeurs manquantes restantes
df_acm = df_acm.dropna()
print(f"Dimensions finales pour l'ACM: {df_acm.shape}")

# Enregistrer le DataFrame préparé
df_acm.to_csv("data/ess11_acm.csv", index=False)
print("Données préparées enregistrées dans data/ess11_acm.csv")

# Afficher un aperçu des données
print("\nAperçu des données:")
print(df_acm.head())

# Afficher les effectifs par variable
print("\nEffectifs par variable:")
for col in df_acm.columns:
    print(f"\n{col}:")
    print(df_acm[col].value_counts())