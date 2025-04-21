#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns

# Charger les données préparées
print("Chargement des données préparées...")
df = pd.read_csv("data/ess11_acm.csv")
print(f"Dimensions: {df.shape}")

# One-hot encoding des variables catégorielles
print("\nPréparation des données pour l'autoencodeur...")
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(df)
feature_names = encoder.get_feature_names_out(df.columns)
print(f"Dimensions après one-hot encoding: {X_encoded.shape}")

# Définir l'architecture de l'autoencodeur
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=2):
        super(Autoencoder, self).__init__()
        
        # Encodeur
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, encoding_dim)
        )
        
        # Décodeur
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        # Decoder
        decoded = self.decoder(encoded)
        return encoded, decoded

# Convertir les données en tenseurs PyTorch
X_tensor = torch.FloatTensor(X_encoded)
dataset = TensorDataset(X_tensor, X_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Définir l'autoencodeur
input_dim = X_encoded.shape[1]
encoding_dim = 2  # Dimension de l'espace latent pour la visualisation
model = Autoencoder(input_dim, encoding_dim)

# Définir la fonction de perte et l'optimiseur
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrainer l'autoencodeur
print("\nEntrainement de l'autoencodeur...")
epochs = 1000
loss_history = []

for epoch in range(epochs):
    epoch_loss = 0
    for data in dataloader:
        # Récupérer les données
        inputs, targets = data
        
        # Réinitialiser les gradients
        optimizer.zero_grad()
        
        # Forward pass
        _, outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Calculer la perte moyenne pour cette époque
    avg_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_loss)
    
    # Afficher la progression
    if (epoch + 1) % 5 == 0:
        print(f"Époque {epoch+1}/{epochs}, Perte: {avg_loss:.6f}")

# Tracer la courbe de perte
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.title('Courbe de perte de l\'autoencodeur')
plt.xlabel('Époque')
plt.ylabel('Perte (BCE)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('autoencoder_loss.png', dpi=300, bbox_inches='tight')
print("Graphique enregistré: autoencoder_loss.png")

# Extraire l'espace latent (représentations apprises)
model.eval()
with torch.no_grad():
    encoded_data, _ = model(X_tensor)
    
latent_space = encoded_data.numpy()
print(f"\nDimensions de l'espace latent: {latent_space.shape}")

# Créer un DataFrame pour l'espace latent
latent_df = pd.DataFrame(
    latent_space,
    columns=[f'Dimension_{i+1}' for i in range(encoding_dim)]
)

# Ajouter le pays pour la coloration
latent_df['cntry'] = df['cntry'].values

# Visualiser l'espace latent
plt.figure(figsize=(12, 10))
countries = df['cntry'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(countries)))
color_map = dict(zip(countries, colors))

# Créer un échantillon pour éviter un graphique trop chargé
sample_size = min(5000, len(latent_df))
sample_indices = np.random.choice(range(len(latent_df)), size=sample_size, replace=False)
latent_sample = latent_df.iloc[sample_indices]

for country in countries:
    mask = latent_sample['cntry'] == country
    plt.scatter(
        latent_sample.loc[mask, 'Dimension_1'],
        latent_sample.loc[mask, 'Dimension_2'],
        color=color_map[country],
        alpha=0.6,
        label=country,
        s=20
    )

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, linestyle='--', alpha=0.4)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Espace latent de l\'autoencodeur par pays')
plt.legend(title='Pays', loc='best')
plt.tight_layout()
plt.savefig('autoencoder_latent_space.png', dpi=300, bbox_inches='tight')
print("Graphique enregistré: autoencoder_latent_space.png")

# PLAN FACTORIEL DES MODALITÉS (similaire à acm_plan_factoriel.png)
print("\nCréation du plan factoriel des modalités (similaire à ACM)...")

# Créer un DataFrame pour stocker les coordonnées moyennes de chaque modalité dans l'espace latent
modality_coords = {}

# Pour chaque variable catégorielle
for col in df.columns:
    unique_vals = df[col].unique()
    
    # Pour chaque modalité de la variable
    for val in unique_vals:
        # Filtrer le DataFrame pour cette modalité
        mask = df[col] == val
        
        if mask.sum() > 0:  # S'assurer qu'il y a des données pour cette modalité
            # Calculer les coordonnées moyennes dans l'espace latent
            mean_coords = latent_space[mask].mean(axis=0)
            
            # Stocker dans le dictionnaire avec une clé formatée "colonne_modalité"
            modality_name = f"{col}__{val}"
            modality_coords[modality_name] = mean_coords

# Convertir en DataFrame pour faciliter la visualisation
modality_df = pd.DataFrame.from_dict(modality_coords, orient='index', 
                                     columns=['Dimension_1', 'Dimension_2'])

# Créer le graphique du plan factoriel
plt.figure(figsize=(15, 12))

# Extraire les noms de variables de base
var_names = [idx.split('__')[0] for idx in modality_df.index]
unique_vars = list(set(var_names))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_vars)))
color_map = dict(zip(unique_vars, colors))

# Tracer les points des modalités
for name, row in modality_df.iterrows():
    parts = name.split('__')
    var_name = parts[0]
    modality = '__'.join(parts[1:])
    
    x, y = row['Dimension_1'], row['Dimension_2']
    color = color_map[var_name]
    
    plt.scatter(x, y, color=color, alpha=0.8, s=100)
    plt.annotate(f"{var_name}_{modality}", (x, y), fontsize=9, alpha=0.8)

# Configurer les limites des axes pour concentrer la vue sur les données
x_min, x_max = modality_df['Dimension_1'].min(), modality_df['Dimension_1'].max()
y_min, y_max = modality_df['Dimension_2'].min(), modality_df['Dimension_2'].max()
# Ajouter 10% de marge autour des données
x_margin = (x_max - x_min) * 0.1
y_margin = (y_max - y_min) * 0.1
plt.xlim(x_min - x_margin, x_max + x_margin)
plt.ylim(y_min - y_margin, y_max + y_margin)

plt.grid(True, linestyle='--', alpha=0.4)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Espace social à la Bourdieu - Autoencodeur: Position des modalités dans l\'espace latent')

# Créer une légende pour les variables
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[var], 
                     label=var, markersize=8) for var in unique_vars]
plt.legend(handles=handles, title='Variables', loc='best')

plt.tight_layout()
plt.savefig('autoencoder_plan_factoriel.png', dpi=300, bbox_inches='tight')
print("Graphique enregistré: autoencoder_plan_factoriel.png")

# Visualiser l'espace latent selon l'âge
plt.figure(figsize=(12, 10))
age_categories = df['age_cat'].unique()
colors_age = plt.cm.viridis(np.linspace(0, 1, len(age_categories)))
color_map_age = dict(zip(age_categories, colors_age))

# Ajouter l'âge au DataFrame latent
latent_sample = latent_df.iloc[sample_indices].copy()
latent_sample['age_cat'] = df.iloc[sample_indices]['age_cat'].values

for age in age_categories:
    mask = latent_sample['age_cat'] == age
    plt.scatter(
        latent_sample.loc[mask, 'Dimension_1'],
        latent_sample.loc[mask, 'Dimension_2'],
        color=color_map_age[age],
        alpha=0.6,
        label=age,
        s=20
    )

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, linestyle='--', alpha=0.4)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Espace latent de l\'autoencodeur par groupe d\'âge')
plt.legend(title='Âge', loc='best')
plt.tight_layout()
plt.savefig('autoencoder_by_age.png', dpi=300, bbox_inches='tight')
print("Graphique enregistré: autoencoder_by_age.png")

# Visualiser l'espace latent selon l'éducation
plt.figure(figsize=(12, 10))
edu_categories = df['eduyrs_cat'].unique()
colors_edu = plt.cm.plasma(np.linspace(0, 1, len(edu_categories)))
color_map_edu = dict(zip(edu_categories, colors_edu))

# Ajouter l'éducation au DataFrame latent
latent_sample['eduyrs_cat'] = df.iloc[sample_indices]['eduyrs_cat'].values

for edu in edu_categories:
    mask = latent_sample['eduyrs_cat'] == edu
    plt.scatter(
        latent_sample.loc[mask, 'Dimension_1'],
        latent_sample.loc[mask, 'Dimension_2'],
        color=color_map_edu[edu],
        alpha=0.6,
        label=edu,
        s=20
    )

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, linestyle='--', alpha=0.4)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Espace latent de l\'autoencodeur par niveau d\'éducation')
plt.legend(title='Niveau d\'éducation', loc='best')
plt.tight_layout()
plt.savefig('autoencoder_by_education.png', dpi=300, bbox_inches='tight')
print("Graphique enregistré: autoencoder_by_education.png")

# Visualiser l'espace latent selon l'orientation politique
plt.figure(figsize=(12, 10))
polscale_categories = df['polscale'].unique()
colors_polscale = plt.cm.coolwarm(np.linspace(0, 1, len(polscale_categories)))
color_map_polscale = dict(zip(polscale_categories, colors_polscale))

# Ajouter l'orientation politique au DataFrame latent
latent_sample['polscale'] = df.iloc[sample_indices]['polscale'].values

for polscale in polscale_categories:
    mask = latent_sample['polscale'] == polscale
    plt.scatter(
        latent_sample.loc[mask, 'Dimension_1'],
        latent_sample.loc[mask, 'Dimension_2'],
        color=color_map_polscale[polscale],
        alpha=0.6,
        label=polscale,
        s=20
    )

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, linestyle='--', alpha=0.4)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Espace latent de l\'autoencodeur par orientation politique')
plt.legend(title='Orientation politique', loc='best')
plt.tight_layout()
plt.savefig('autoencoder_by_polscale.png', dpi=300, bbox_inches='tight')
print("Graphique enregistré: autoencoder_by_polscale.png")

# Comparer ACM classique et autoencodeur
print("\nComparaison entre ACM classique et autoencodeur terminée.")
print("Les graphiques générés permettent de comparer les deux approches.")
print("""
Différences principales entre ACM classique et autoencodeur :
1. L'ACM est linéaire tandis que l'autoencodeur peut capturer des relations non-linéaires.
2. L'autoencodeur a plus de couches et peut potentiellement apprendre des représentations plus complexes.
3. L'interprétabilité: l'ACM offre une interprétation plus directe des axes, tandis que l'autoencodeur est plus "boîte noire".
4. L'autoencodeur peut potentiellement mieux reconstruire les données d'origine, grâce à sa capacité à modéliser des relations non-linéaires.

Dans un contexte bourdieusien:
- L'ACM est traditionnellement utilisée car elle préserve mieux les distances chi-carrées entre modalités.
- L'autoencodeur pourrait révéler des structures plus subtiles ou non-linéaires dans l'espace social.
- L'espace à deux dimensions de l'autoencodeur est comparable à l'espace social à deux dimensions de Bourdieu (volume de capital / structure du capital).
""")