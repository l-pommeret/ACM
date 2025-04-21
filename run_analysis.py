#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script principal pour exécuter l'analyse ACM et autoencodeur
sur les données de l'European Social Survey (ESS11)
"""

import os
import sys
import subprocess
import time

def run_script(script_path, description):
    """Exécute un script Python et affiche sa sortie en temps réel"""
    print(f"\n{'='*80}")
    print(f"Exécution de: {description}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Exécuter le script et capturer sa sortie
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Afficher la sortie en temps réel
    for line in process.stdout:
        print(line, end='')
    
    # Attendre que le processus se termine
    process.wait()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nTerminé en {elapsed_time:.2f} secondes")
    
    # Vérifier si le script s'est terminé avec succès
    if process.returncode != 0:
        print(f"ERREUR: Le script s'est terminé avec le code {process.returncode}")
        return False
    
    return True

def main():
    """Fonction principale qui exécute tous les scripts d'analyse"""
    print("Début de l'analyse ACM et autoencodeur pour les données ESS11")
    
    # Vérifier si le dossier de sortie existe, sinon le créer
    os.makedirs("outputs", exist_ok=True)
    
    # Liste des scripts à exécuter dans l'ordre
    scripts = [
        ("scripts/prepare_data.py", "Préparation des données"),
        ("scripts/acm_analysis.py", "Analyse ACM classique"),
        ("scripts/autoencoder_acm.py", "Analyse avec autoencodeur à deux dimensions")
    ]
    
    # Exécuter chaque script
    for script_path, description in scripts:
        success = run_script(script_path, description)
        
        if not success:
            print(f"\nERREUR lors de l'exécution de {script_path}. Abandon.")
            return
    
    print("\nAnalyse complète terminée avec succès!")
    print("Les résultats graphiques ont été enregistrés dans le répertoire courant.")

if __name__ == "__main__":
    main()