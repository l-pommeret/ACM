#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import prince
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns

# Charger les données préparées
print("Chargement des données préparées...")
df = pd.read_csv("data/ess11_acm.csv")
print(f"Dimensions: {df.shape}")

# Réaliser l'ACM
print("\nRéalisation de l'ACM...")
mca = prince.MCA(
    n_components=5,        # Nombre de composantes à calculer
    n_iter=3,             # Nombre d'itérations
    copy=True,            # Créer une copie des données
    check_input=True,     # Vérifier les données d'entrée
    engine='sklearn',     # Algorithme de SVD (doit être l'un de 'fbpca', 'scipy', 'sklearn')
    random_state=42       # Pour reproductibilité
)

# Ajuster l'ACM et transformer les données
mca = mca.fit(df)

# Afficher les valeurs propres et la variance expliquée
print("\nValeurs propres:")
print(mca.eigenvalues_)
print("\nInertie expliquée par composante:")
# Calculer manuellement l'inertie expliquée
total_inertia = sum(mca.eigenvalues_)
explained_inertia = [val/total_inertia for val in mca.eigenvalues_]
print(explained_inertia)

# Coordonnées des variables
coords_vars = mca.column_coordinates(df)
print("\nCoordonnées des variables (premiers éléments):")
print(coords_vars.head())

# Coordonnées des individus
coords_ind = mca.row_coordinates(df)
print("\nCoordonnées des individus (premiers éléments):")
print(coords_ind.head())

# Tracer les graphiques
plt.figure(figsize=(12, 10))

# Graphique des valeurs propres (scree plot)
plt.subplot(2, 2, 1)
plt.bar(range(1, len(mca.eigenvalues_) + 1), mca.eigenvalues_)
plt.xlabel('Composante')
plt.ylabel('Valeur propre')
plt.title('Scree plot')
plt.grid(True, linestyle='--', alpha=0.7)

# Projection des variables sur les deux premières composantes
plt.subplot(2, 2, 2)
plt.figure(figsize=(12, 10))
ax = plt.gca()

# Tracer les points des variables
for i, (_, row) in enumerate(coords_vars.iterrows()):
    label = row.name
    x, y = row[0], row[1]
    
    # Diviser le nom de la variable pour extraire le nom de base et la modalité
    parts = label.split('_')
    if len(parts) > 1:
        var_name = parts[0]
        modality = '_'.join(parts[1:])
    else:
        var_name = label
        modality = ""
    
    # Colorer les points par variable
    plt.scatter(x, y, alpha=0.8)
    plt.annotate(modality, (x, y), fontsize=8, alpha=0.7)

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, linestyle='--', alpha=0.4)
plt.xlabel(f'Composante 1 ({explained_inertia[0]:.2%})')
plt.ylabel(f'Composante 2 ({explained_inertia[1]:.2%})')
plt.title('Projection des variables sur les deux premières composantes')
plt.tight_layout()

# Créer une visualisation du plan factoriel complet
plt.figure(figsize=(15, 10))

# Tracer les points des variables avec une couleur par variable de base
var_names = [col.split('_')[0] for col in coords_vars.index]
unique_vars = list(set(var_names))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_vars)))
color_map = dict(zip(unique_vars, colors))

# Tracer les points des modalités
for i, (name, row) in enumerate(coords_vars.iterrows()):
    parts = name.split('_')
    if len(parts) > 1:
        var_name = parts[0]
        modality = '_'.join(parts[1:])
    else:
        var_name = name
        modality = ""
    
    x, y = row[0], row[1]
    color = color_map[var_name]
    
    plt.scatter(x, y, color=color, alpha=0.8)
    plt.annotate(f"{var_name}_{modality}", (x, y), fontsize=8, alpha=0.8)

# Ajouter des légendes pour les axes et les couleurs
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, linestyle='--', alpha=0.4)
plt.xlabel(f'Axe 1 ({explained_inertia[0]:.2%})')
plt.ylabel(f'Axe 2 ({explained_inertia[1]:.2%})')
plt.title('Espace social à la Bourdieu - ACM des variables socio-démographiques et culturelles')

# Créer une légende pour les variables
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[var], 
                     label=var, markersize=8) for var in unique_vars]
plt.legend(handles=handles, title='Variables', loc='best')

plt.tight_layout()
plt.savefig('acm_plan_factoriel.png', dpi=300, bbox_inches='tight')
print("Graphique enregistré: acm_plan_factoriel.png")

# Créer un nuage de points des individus coloré par pays
plt.figure(figsize=(12, 10))
countries = df['cntry'].unique()
colors_countries = plt.cm.tab20(np.linspace(0, 1, len(countries)))
color_map_countries = dict(zip(countries, colors_countries))

# Créer un échantillon pour éviter un graphique trop chargé
sample_size = min(5000, len(coords_ind))
sample_indices = np.random.choice(range(len(coords_ind)), size=sample_size, replace=False)
coords_ind_sample = coords_ind.iloc[sample_indices]
df_sample = df.iloc[sample_indices]

for country in countries:
    mask = df_sample['cntry'] == country
    plt.scatter(
        coords_ind_sample.loc[mask, 0], 
        coords_ind_sample.loc[mask, 1],
        color=color_map_countries[country],
        alpha=0.6,
        label=country,
        s=20
    )

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, linestyle='--', alpha=0.4)
plt.xlabel(f'Axe 1 ({explained_inertia[0]:.2%})')
plt.ylabel(f'Axe 2 ({explained_inertia[1]:.2%})')
plt.title('Projection des individus par pays')
plt.legend(title='Pays', loc='best')
plt.tight_layout()
plt.savefig('acm_individus_par_pays.png', dpi=300, bbox_inches='tight')
print("Graphique enregistré: acm_individus_par_pays.png")

# Interprétation des axes
print("\nInterprétation des axes:")
# Pour chaque axe, afficher les modalités qui contribuent le plus (en valeur absolue)
n_components = min(5, len(mca.eigenvalues_))
for axis in range(n_components):
    print(f"\nAxe {axis+1}:")
    # Trier les coordonnées par valeur absolue décroissante
    sorted_coords = coords_vars[axis].abs().sort_values(ascending=False)
    # Afficher les 10 modalités les plus contributives
    print(f"Top 10 modalités contributives:")
    for name in sorted_coords.index[:10]:
        value = coords_vars.loc[name, axis]
        print(f"  {name}: {value:.4f}")

print("\nAnalyse ACM terminée.")