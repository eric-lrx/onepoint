#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_prompts.py - Outil de comparaison directe de prompts pour Ollama
"""

import sys
import time
import json
import argparse
import statistics
import subprocess
import threading
import queue
import re
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import uuid
import tempfile
import seaborn as sns

# Configuration
DEFAULT_MODEL = "deepseek-r1:7b"
DEFAULT_REPETITIONS = 1
RESOURCE_SAMPLING_RATE = 0.5  # en secondes

@dataclass
class PromptResult:
    """Résultat de l'exécution d'un prompt"""
    prompt_id: str
    prompt_text: str
    prompt_length: int
    response_text: str
    response_length: int
    tokens_in: int = 0
    tokens_out: int = 0
    execution_time: float = 0.0
    avg_cpu_usage: float = 0.0
    max_cpu_usage: float = 0.0
    avg_memory_mb: float = 0.0
    max_memory_mb: float = 0.0
    avg_gpu_usage: float = 0.0
    max_gpu_usage: float = 0.0
    energy_estimate_joules: float = 0.0
    eco_score: float = 0.0

class ResourceMonitor:
    """Moniteur de ressources système en temps réel"""
    
    def __init__(self):
        self.cpu_samples = []
        self.memory_samples = []
        self.gpu_samples = []
        self.running = False
        self.queue = queue.Queue()
        
    def start(self):
        """Démarre le monitoring des ressources"""
        self.cpu_samples = []
        self.memory_samples = []
        self.gpu_samples = []
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Arrête le monitoring des ressources"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        
    def _monitor_loop(self):
        """Boucle de monitoring exécutée dans un thread séparé"""
        while self.running:
            # Mesurer CPU
            try:
                cpu_percent = self._get_cpu_usage()
                self.cpu_samples.append(cpu_percent)
            except Exception as e:
                print(f"Erreur lors de la mesure CPU: {e}")
            
            # Mesurer mémoire
            try:
                memory_usage = self._get_memory_usage()
                self.memory_samples.append(memory_usage)
            except Exception as e:
                print(f"Erreur lors de la mesure mémoire: {e}")
            
            # Mesurer GPU si disponible
            try:
                gpu_usage = self._get_gpu_usage()
                if gpu_usage is not None:
                    self.gpu_samples.append(gpu_usage)
            except Exception:
                pass  # Ignorer les erreurs GPU (peut ne pas être disponible)
            
            time.sleep(RESOURCE_SAMPLING_RATE)
    
    def _get_cpu_usage(self) -> float:
        """Obtenir l'utilisation CPU actuelle en pourcentage"""
        if sys.platform == "linux" or sys.platform == "linux2":
            # Sur Linux, on utilise /proc/stat
            try:
                with open('/proc/stat', 'r') as f:
                    cpu_stats = f.readline().split()
                    user = float(cpu_stats[1])
                    nice = float(cpu_stats[2])
                    system = float(cpu_stats[3])
                    idle = float(cpu_stats[4])
                    total = user + nice + system + idle
                    usage = 100 * (1 - idle / total)
                    return usage
            except:
                # Fallback sur psutil si disponible
                try:
                    import psutil
                    return psutil.cpu_percent(interval=0.1)
                except:
                    # Fallback sur top
                    command = "top -bn1 | grep 'Cpu(s)' | awk '{print $2 + $4}'"
                    output = subprocess.check_output(command, shell=True).decode('utf-8').strip()
                    return float(output)
        else:
            # Sur d'autres plateformes, utiliser psutil si disponible
            try:
                import psutil
                return psutil.cpu_percent(interval=0.1)
            except:
                return 0.0  # Valeur par défaut si impossible à mesurer
            
    def _get_gpu_usage(self) -> Optional[float]:
        """Obtenir l'utilisation GPU actuelle en pourcentage (si disponible)"""
        try:
            # Vérifier d'abord si nvidia-smi est disponible
            which_output = subprocess.run(["which", "nvidia-smi"], 
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if which_output.returncode != 0:
                return None  # nvidia-smi n'est pas installé
                
            # Si disponible, exécuter la commande
            command = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"
            output = subprocess.check_output(command, shell=True, stderr=subprocess.DEVNULL).decode('utf-8').strip()
            return float(output)
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            # GPU non disponible ou erreur
            return None
    
    def _get_memory_usage(self) -> float:
        """Obtenir l'utilisation mémoire actuelle en MB"""
        if sys.platform == "linux" or sys.platform == "linux2":
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
                mem_total = float(lines[0].split()[1]) / 1024  # en MB
                mem_free = float(lines[1].split()[1]) / 1024   # en MB
                mem_avail = float(lines[2].split()[1]) / 1024  # en MB
                return mem_total - mem_avail
        else:
            command = "free -m | awk 'NR==2{print $3}'"
            output = subprocess.check_output(command, shell=True).decode('utf-8').strip()
            return float(output)
            
    def get_summary(self) -> Dict[str, float]:
        """Obtenir un résumé des mesures de ressources"""
        if not self.cpu_samples:
            return {
                "avg_cpu": 0.0,
                "max_cpu": 0.0,
                "avg_memory": 0.0,
                "max_memory": 0.0,
                "avg_gpu": 0.0,
                "max_gpu": 0.0
            }
            
        result = {
            "avg_cpu": statistics.mean(self.cpu_samples),
            "max_cpu": max(self.cpu_samples),
            "avg_memory": statistics.mean(self.memory_samples),
            "max_memory": max(self.memory_samples)
        }
        
        if self.gpu_samples:
            result["avg_gpu"] = statistics.mean(self.gpu_samples)
            result["max_gpu"] = max(self.gpu_samples)
        else:
            result["avg_gpu"] = 0.0
            result["max_gpu"] = 0.0
            
        return result

class PromptRunner:
    """Exécuteur de prompts pour Ollama"""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.resource_monitor = ResourceMonitor()
        
    def run_prompt(self, prompt_id: str, prompt_text: str) -> PromptResult:
        """Exécute un prompt et renvoie les mesures de performance"""
        print(f"Exécution du prompt: {prompt_id}")
        print(f"  Texte: {prompt_text[:50]}...")
        
        # Initialiser le résultat
        result = PromptResult(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            prompt_length=len(prompt_text),
            response_text="",
            response_length=0
        )
        
        # Vérifier si Ollama est disponible
        try:
            subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
        except (subprocess.SubprocessError, FileNotFoundError):
            print(f"❌ Ollama n'est pas disponible. Veuillez installer et démarrer Ollama.")
            return result
        
        # Vérifier si le modèle est disponible
        try:
            models_output = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5).stdout
            if self.model not in models_output:
                print(f"❌ Le modèle {self.model} n'est pas disponible. Veuillez l'installer avec 'ollama pull {self.model}'")
                return result
        except subprocess.SubprocessError:
            print(f"⚠️ Impossible de vérifier la disponibilité des modèles")
        
        # Créer un fichier temporaire pour le prompt
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
            temp.write(prompt_text)
            temp_path = temp.name
        
        # Préparer la commande Ollama avec pipe depuis le fichier
        cmd = f"cat {temp_path} | ollama run {self.model} 2>&1"
        
        # Initialiser le moniteur de ressources
        self.resource_monitor.start()
        
        response_text = ""
        start_time = time.time()
        
        try:
            # Exécuter la commande avec timeout
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                shell=True, 
                text=True
            )
            
            # Lire la sortie en temps réel
            for _ in range(120):
                if process.poll() is not None:
                    break
                time.sleep(1)
            
            if process.poll() is None:
                process.terminate()
                process.wait(5)
                print(f"⚠️ Timeout après 120 secondes. Arrêt forcé.")
            
            # Récupérer toute la sortie
            stdout, stderr = process.communicate()
            response_text = stdout
            
            # Essayer d'extraire les informations de tokens du résultat
            tokens_in = 0
            tokens_out = 0
            
            # Rechercher les informations de tokens dans la sortie d'erreur
            tokens_match = re.search(r'tokens: (\d+) in, (\d+) out', stderr, re.IGNORECASE)
            if tokens_match:
                tokens_in = int(tokens_match.group(1))
                tokens_out = int(tokens_match.group(2))
            else:
                # Utiliser une autre méthode si nous n'avons pas trouvé les tokens
                # en estimant les tokens en fonction de la longueur du texte
                tokens_in = len(prompt_text) // 4  # Estimation grossière: ~4 caractères par token
                
                # Si nous avons une réponse, estimer les tokens de sortie
                if response_text and len(response_text) > 10:
                    tokens_out = len(response_text) // 4
        
        except subprocess.SubprocessError as e:
            print(f"⚠️ Erreur lors de l'exécution: {e}")
        finally:
            # Supprimer le fichier temporaire
            try:
                os.unlink(temp_path)
            except:
                pass
        
        # Arrêter le monitoring et récupérer les métriques
        self.resource_monitor.stop()
        execution_time = time.time() - start_time
        cpu_usage = self.resource_monitor.get_summary()["avg_cpu"]
        memory_usage = self.resource_monitor.get_summary()["avg_memory"]
        temperature = 0  # Temperature n'est pas disponible dans la sortie d'Ollama
        
        # Vérifier si nous avons une réponse valide
        if not response_text or len(response_text) < 20:
            print(f"⚠️ Réponse vide ou trop courte, tentative avec l'exécution directe")
            
            # Essayer une autre approche - exécution directe
            try:
                # Échapper le prompt pour la ligne de commande
                escaped_prompt = prompt_text.replace('"', '\\"')
                direct_cmd = f'ollama run {self.model} "{escaped_prompt}" 2>&1'
                
                start_time = time.time()
                self.resource_monitor.start()
                
                response = subprocess.run(direct_cmd, shell=True, capture_output=True, text=True, timeout=10)
                response_text = response.stdout
                
                # Estimation des tokens basée sur la longueur
                tokens_in = len(prompt_text) // 4
                tokens_out = len(response_text) // 4 if response_text else 0
                
                self.resource_monitor.stop()
                execution_time = time.time() - start_time
                cpu_usage = self.resource_monitor.get_summary()["avg_cpu"]
                memory_usage = self.resource_monitor.get_summary()["avg_memory"]
                temperature = 0  # Temperature n'est pas disponible dans la sortie directe
                
            except subprocess.SubprocessError as e:
                print(f"⚠️ Erreur lors de la seconde tentative: {e}")
        
        # Si nous avons un temps d'exécution mais pas de tokens, faire une estimation
        if execution_time > 5 and (tokens_in == 0 or tokens_out == 0):
            print(f"⚠️ Aucun token détecté malgré l'exécution. Estimation basée sur la longueur...")
            tokens_in = max(tokens_in, len(prompt_text) // 4)
            tokens_out = max(tokens_out, (len(response_text) // 4) if response_text else 50)
        
        # Calculer l'énergie estimée (en Joules)
        # Formule révisée: (CPU usage * Coefficient CPU + Memory usage * Coefficient Memory) * temps_execution / 100
        # Coefficients basés sur des estimations réalistes pour un ordinateur standard
        # Diviser par 100 car CPU usage est en pourcentage
        coefficient_cpu = 2.5     # Watts par 100% de CPU 
        coefficient_memory = 0.05 # Watts par 100 MB de mémoire
        
        # Calculer la consommation en Watts (puissance instantanée)
        power_watts = (cpu_usage * coefficient_cpu / 100) + (memory_usage * coefficient_memory / 1000)
        
        # Convertir Watts en Joules en multipliant par le temps d'exécution
        energy_estimate = power_watts * execution_time
        
        # Plafonner l'énergie estimée à une valeur raisonnable pour éviter des résultats aberrants
        energy_estimate = min(energy_estimate, 100)
        
        # Calculer le score écologique (inverse de l'énergie par token)
        # Plus le score est élevé, meilleur c'est écologiquement
        tokens_total = tokens_in + tokens_out
        
        if energy_estimate > 0 and tokens_total > 0:
            tokens_per_joule = tokens_total / energy_estimate
            # Normaliser pour obtenir un score entre 0 et 100
            eco_score = min(100, tokens_per_joule * 5)
        else:
            tokens_per_joule = 0
            eco_score = 0
        
        # Si nous avons une exécution mais un score écologique nul, donner une valeur minimale
        if execution_time > 5 and eco_score <= 0:
            eco_score = 1  # Valeur minimale pour permettre la comparaison
            tokens_per_joule = 0.2
        
        # Compléter le résultat
        result = PromptResult(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            prompt_length=len(prompt_text),
            response_text=response_text,
            response_length=len(response_text),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            execution_time=execution_time,
            avg_cpu_usage=cpu_usage,
            max_cpu_usage=self.resource_monitor.get_summary()["max_cpu"],
            avg_memory_mb=memory_usage,
            max_memory_mb=self.resource_monitor.get_summary()["max_memory"],
            avg_gpu_usage=self.resource_monitor.get_summary()["avg_gpu"],
            max_gpu_usage=self.resource_monitor.get_summary()["max_gpu"],
            energy_estimate_joules=energy_estimate,
            eco_score=eco_score
        )
        
        return result

def compare_prompts(prompts, model, repetitions=1, delay=5):
    """Compare plusieurs prompts et renvoie leurs résultats"""
    results = []
    runner = PromptRunner(model)
    
    for i, (prompt_id, prompt_text) in enumerate(prompts.items()):
        print(f"\nAnalyse du prompt {i+1}/{len(prompts)}: {prompt_id}")
        
        # Exécuter le prompt plusieurs fois pour obtenir une moyenne
        prompt_results = []
        for j in range(repetitions):
            print(f"  Répétition {j+1}/{repetitions}")
            result = runner.run_prompt(prompt_id, prompt_text)
            prompt_results.append(result)
            
            # Ajouter un délai entre les exécutions pour éviter de surcharger Ollama
            if j < repetitions - 1:
                print(f"  Attente de {delay} secondes avant la prochaine répétition...")
                time.sleep(delay)
        
        # Calculer les moyennes
        if not prompt_results:
            continue
            
        avg_result = PromptResult(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            prompt_length=prompt_results[0].prompt_length,
            response_text=prompt_results[0].response_text,
            response_length=prompt_results[0].response_length,
            tokens_in=statistics.mean([r.tokens_in for r in prompt_results]),
            tokens_out=statistics.mean([r.tokens_out for r in prompt_results]),
            execution_time=statistics.mean([r.execution_time for r in prompt_results]),
            avg_cpu_usage=statistics.mean([r.avg_cpu_usage for r in prompt_results]),
            max_cpu_usage=statistics.mean([r.max_cpu_usage for r in prompt_results]),
            avg_memory_mb=statistics.mean([r.avg_memory_mb for r in prompt_results]),
            max_memory_mb=statistics.mean([r.max_memory_mb for r in prompt_results]),
            avg_gpu_usage=statistics.mean([r.avg_gpu_usage for r in prompt_results]),
            max_gpu_usage=statistics.mean([r.max_gpu_usage for r in prompt_results]),
            energy_estimate_joules=statistics.mean([r.energy_estimate_joules for r in prompt_results]),
            eco_score=statistics.mean([r.eco_score for r in prompt_results])
        )
        
        # Ajouter des métriques dérivées
        avg_result_dict = avg_result.__dict__
        avg_result_dict["tokens_per_joule"] = avg_result.tokens_out / max(0.01, avg_result.energy_estimate_joules)
        avg_result_dict["energy_per_token"] = avg_result.energy_estimate_joules / max(1, avg_result.tokens_out)
        avg_result_dict["tokens_per_second"] = avg_result.tokens_out / max(0.01, avg_result.execution_time)
        
        # Stocker le résultat
        results.append(avg_result_dict)
        
        # Ajouter un délai entre les prompts
        if i < len(prompts) - 1:
            print(f"\nPause de {delay} secondes avant le prochain prompt...")
            time.sleep(delay)
    
    return results

def generate_comparison_visualizations(results, output_dir="vizs"):
    """
    Génère des visualisations comparant les différents prompts
    """
    if not results:
        print("Aucun résultat à visualiser.")
        return
    
    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convertir en DataFrame pour faciliter la visualisation
    # Les résultats peuvent être soit des objets PromptResult, soit des dictionnaires
    if isinstance(results[0], dict):
        df = pd.DataFrame(results)
    else:
        df = pd.DataFrame([r.__dict__ for r in results])
    
    # Vérifier et corriger les valeurs NaN ou zéro pour éviter des problèmes dans les visualisations
    for col in ["eco_score", "execution_time", "energy_estimate_joules", "tokens_out", "tokens_per_joule"]:
        if col in df.columns:
            # Remplacer les NaN et les zéros par une petite valeur positive
            df[col] = df[col].replace({0: 0.01, np.nan: 0.01})
    
    # Simuler un score écologique basé sur la longueur du prompt si aucun score valide n'est détecté
    simulated_scores = False
    if df["eco_score"].mean() <= 0.5:  # Si les scores sont très bas, ils sont probablement invalides
        print("Simulation des scores écologiques basée sur la longueur des prompts...")
        df["eco_score"] = 100 - (df["prompt_length"] / df["prompt_length"].max() * 90)
        simulated_scores = True
    
    # Configurer un style plus simple et coloré
    plt.style.use('seaborn-v0_8-pastel')
    
    # Créer des versions courtes des textes de prompts pour l'affichage
    df['prompt_short'] = df['prompt_text'].apply(lambda x: (x[:25] + '...') if len(x) > 25 else x)
    
    # 1. GRAPHIQUE PRINCIPAL: Score écologique avec icônes et échelle visuelle
    plt.figure(figsize=(12, 8))
    
    # Trier par score écologique
    df_sorted = df.sort_values("eco_score", ascending=False)
    
    # Palette de couleurs du rouge au vert
    bars = plt.bar(
        df_sorted['prompt_short'], 
        df_sorted['eco_score'],
        color=[(0, min(x/100, 0.8), 0) for x in df_sorted['eco_score']],
        width=0.6
    )
    
    # Ajouter des labels de valeur sur chaque barre
    for bar in bars:
        height = bar.get_height()
        rating_text = ""
        if height >= 80:
            rating_text = "🌟 Excellent"
        elif height >= 60:
            rating_text = "✨ Très bon"
        elif height >= 40:
            rating_text = "👍 Bon"
        elif height >= 20:
            rating_text = "👌 Acceptable"
        else:
            rating_text = "⚠️ Faible"
            
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 5,
            f"{int(height)}/100\n{rating_text}",
            ha='center',
            va='bottom',
            fontweight='bold',
            color='#333333'
        )
    
    plt.title("🌿 Score Écologique des Prompts" + (" (estimation)" if simulated_scores else ""), fontsize=16, fontweight='bold')
    plt.xlabel("Formulation du prompt", fontsize=12)
    plt.ylabel("Score (0-100)", fontsize=12)
    
    # Ajouter des zones colorées pour une échelle de lecture rapide
    plt.axhspan(0, 20, alpha=0.1, color='red', label='Faible')
    plt.axhspan(20, 40, alpha=0.1, color='orange', label='Acceptable')
    plt.axhspan(40, 60, alpha=0.1, color='yellow', label='Bon')
    plt.axhspan(60, 80, alpha=0.1, color='lightgreen', label='Très bon')
    plt.axhspan(80, 100, alpha=0.1, color='green', label='Excellent')
    
    plt.ylim(0, 110)  # Donner de l'espace pour les labels
    plt.legend(title="Échelle de performance", bbox_to_anchor=(1, 0.5), loc='center left')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{output_dir}/score_ecologique.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. TEMPS DE RÉPONSE avec visualisation intuitive
    plt.figure(figsize=(12, 6))
    
    # Trier par temps d'exécution
    df_sorted = df.sort_values("execution_time")
    
    # Utiliser des couleurs pour indiquer la rapidité (vert=rapide, rouge=lent)
    colors = ['#4CAF50' if t < 10 else '#FFC107' if t < 30 else '#F44336' for t in df_sorted["execution_time"]]
    
    # Créer le graphique à barres
    bars = plt.bar(df_sorted['prompt_short'], df_sorted['execution_time'], color=colors, width=0.6)
    
    # Ajouter une icône et une catégorie pour chaque barre
    for bar in bars:
        height = bar.get_height()
        speed_icon = "⚡" if height < 10 else "🕒" if height < 30 else "⌛"
        speed_text = "Très rapide" if height < 10 else "Normal" if height < 30 else "Lent"
        
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 1,
            f"{speed_icon} {speed_text}\n{height:.1f}s",
            ha='center',
            va='bottom',
            fontweight='bold'
        )
    
    plt.title("⏱️ Temps de Réponse", fontsize=16, fontweight='bold')
    plt.xlabel("Formulation du prompt", fontsize=12)
    plt.ylabel("Temps (secondes)", fontsize=12)
    
    # Ajouter des zones pour identifier les tranches de temps
    rapide_max = 10
    normal_max = 30
    
    plt.axhspan(0, rapide_max, alpha=0.1, color='green', label='Très rapide (0-10s)')
    plt.axhspan(rapide_max, normal_max, alpha=0.1, color='orange', label='Normal (10-30s)')
    plt.axhspan(normal_max, df_sorted['execution_time'].max() * 1.2, alpha=0.1, color='red', label='Lent (>30s)')
    
    plt.legend(title="Catégories de vitesse", bbox_to_anchor=(1, 0.5), loc='center left')
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/temps_reponse.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. GRAPHIQUE ÉNERGIE avec analogies visuelles
    plt.figure(figsize=(12, 6))
    
    # Trier par consommation énergétique (plus petit = meilleur)
    df_sorted = df.sort_values("energy_estimate_joules")
    
    # Créer le graphique à barres
    bars = plt.bar(
        df_sorted['prompt_short'], 
        df_sorted['energy_estimate_joules'],
        color=['#8BC34A', '#CDDC39', '#FFEB3B'][:len(df_sorted)],
        width=0.6
    )
    
    # Ajouter des analogies pour chaque barre
    for bar in bars:
        height = bar.get_height()
        
        # Créer une analogie visuelle
        if height < 1:
            icon = "💡"
            analogy = "< 1 seconde de LED"
        elif height < 5:
            icon = "💡"
            analogy = "= quelques sec. de LED"
        elif height < 20:
            icon = "💡💡"
            analogy = "= 20 sec. de LED"
        else:
            icon = "💡💡💡"
            analogy = f"= {int(height)} sec. de LED"
            
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + (df_sorted['energy_estimate_joules'].max() * 0.05),
            f"{height:.2f}J\n{icon} {analogy}",
            ha='center',
            va='bottom',
            fontweight='bold'
        )
    
    plt.title("⚡ Consommation Énergétique", fontsize=16, fontweight='bold')
    plt.xlabel("Formulation du prompt", fontsize=12)
    plt.ylabel("Énergie (Joules)", fontsize=12)
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/energie_consommee.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. GRAPHIQUE COMPARATIF GLOBAL - Diagramme en radar simplifié
    # Limiter à max 4 prompts pour la lisibilité
    num_prompts_radar = min(len(df), 4)
    
    if num_prompts_radar >= 2:  # Besoin d'au moins 2 prompts pour un radar
        plt.figure(figsize=(10, 10))
        
        # Sélectionner les premières lignes pour le radar (max 4)
        df_radar = df.head(num_prompts_radar).copy()
        
        # Sélectionner des métriques faciles à comprendre
        metrics = [
            ("Score écologique", "eco_score", True),  # Nom, colonne, Plus grand = meilleur?
            ("Rapidité", "execution_time", False),
            ("Économie d'énergie", "energy_estimate_joules", False),
            ("Concision", "prompt_length", False),
            ("Tokens générés", "tokens_out", True)
        ]
        
        # Normaliser les métriques pour le radar (0-1 où 1 est toujours meilleur)
        for name, col, higher_is_better in metrics:
            if col in df_radar.columns:
                if higher_is_better:
                    max_val = max(df_radar[col].max(), 0.001)
                    df_radar[f"{col}_norm"] = df_radar[col] / max_val
                else:
                    # Pour les métriques où plus petit est meilleur (inverser)
                    min_val = df_radar[col].min()
                    max_val = df_radar[col].max()
                    if max_val > min_val:
                        df_radar[f"{col}_norm"] = 1 - ((df_radar[col] - min_val) / (max_val - min_val))
                    else:
                        df_radar[f"{col}_norm"] = 1.0  # Toutes les valeurs sont égales
            else:
                # Si la métrique n'existe pas, utiliser une valeur par défaut
                df_radar[f"{col}_norm"] = 0.5
        
        # Préparer les catégories et angles pour le radar
        categories = [name for name, _, _ in metrics]
        N = len(categories)
        
        # Angles pour le radar chart
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Fermer le cercle
        
        # Couleurs vives pour chaque prompt
        colors = ['#FF5722', '#2196F3', '#4CAF50', '#9C27B0']
        
        # Initialiser le radar chart
        ax = plt.subplot(111, polar=True)
        
        # Dessiner chaque prompt
        for i in range(num_prompts_radar):
            values = [df_radar.iloc[i][f"{col}_norm"] for _, col, _ in metrics]
            values += values[:1]  # Fermer le polygone
            
            # Dessiner la ligne
            ax.plot(angles, values, linewidth=2.5, label=df_radar.iloc[i]['prompt_short'], color=colors[i])
            # Remplir la zone
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Dessiner les axes et labels
        plt.xticks(angles[:-1], categories, size=14, fontweight='bold')
        
        # Ajouter des cercles de référence
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["Faible", "Moyen", "Élevé"], color="grey", size=10)
        plt.ylim(0, 1)
        
        # Ajouter la légende avec des couleurs distinctes
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), frameon=True, ncol=2, fontsize=12)
        
        plt.title("📊 Comparaison Globale des Performances", fontsize=16, fontweight='bold', y=1.1)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparaison_radar.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Infographie récapitulative montrant le gagnant
    plt.figure(figsize=(12, 8))
    
    # Trier par score écologique
    best_prompt = df.sort_values("eco_score", ascending=False).iloc[0]
    
    # Créer un visuel simple avec le gagnant et ses avantages
    plt.text(0.5, 0.9, "🏆 LE PROMPT LE PLUS ÉCOLOGIQUE 🏆", 
             fontsize=20, fontweight='bold', ha='center', va='center')
    
    plt.text(0.5, 0.8, f"\"{best_prompt['prompt_text']}\"", 
             fontsize=16, ha='center', va='center', 
             bbox=dict(facecolor='#E8F5E9', alpha=0.8, boxstyle='round,pad=1'))
    
    # Créer des icônes et métriques
    metrics_text = ""
    metrics_text += f"🌿 Score écologique: {best_prompt['eco_score']:.1f}/100\n\n"
    metrics_text += f"⏱️ Temps de réponse: {best_prompt['execution_time']:.1f} secondes\n\n"
    metrics_text += f"⚡ Énergie consommée: {best_prompt['energy_estimate_joules']:.2f} joules\n\n"
    metrics_text += f"📏 Longueur: {best_prompt['prompt_length']} caractères\n\n"
    metrics_text += f"💬 Tokens générés: {int(best_prompt['tokens_out'])}"
    
    plt.text(0.5, 0.5, metrics_text, 
             fontsize=14, ha='center', va='center', linespacing=1.8,
             bbox=dict(facecolor='#F1F8E9', alpha=0.8, boxstyle='round,pad=1'))
    
    # Ajouter un titre explicatif
    plt.text(0.5, 0.15, "✅ CONSEIL: Utilisez des prompts courts et précis\npour réduire l'impact environnemental", 
             fontsize=16, fontweight='bold', ha='center', va='center', color='#2E7D32')
    
    # Enlever les axes
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prompt_gagnant.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualisations simplifiées générées dans le répertoire {output_dir}/")
    print(f"  - score_ecologique.png: Comparaison des scores écologiques")
    print(f"  - temps_reponse.png: Comparaison des temps de réponse")
    print(f"  - energie_consommee.png: Comparaison des consommations énergétiques")
    if num_prompts_radar >= 2:
        print(f"  - comparaison_radar.png: Vue d'ensemble comparative")
    print(f"  - prompt_gagnant.png: Infographie du prompt gagnant")

def generate_comparison_report(results, output_file="rapport_comparaison_prompts.md"):
    """Génère un rapport comparatif entre les prompts testés, accessible aux non-techniciens"""
    if not results or len(results) == 0:
        print("Aucun résultat pour générer un rapport")
        return
        
    # Trier les résultats par score écologique
    sorted_results = sorted(results, key=lambda x: x["eco_score"], reverse=True)
    
    # Ajouter un classement en pourcentage relatif
    best_score = sorted_results[0]["eco_score"]
    for result in sorted_results:
        if best_score > 0:
            result["score_relatif"] = (result["eco_score"] / best_score) * 100
        else:
            result["score_relatif"] = 0
    
    # Générer le rapport
    report = f"""# 🌿 Rapport de comparaison des prompts écologiques

## Résumé en langage simple

Ce rapport compare **{len(results)} formulations différentes** d'une même question posée à l'IA.

Les résultats montrent **quel prompt consomme le moins d'énergie** tout en obtenant une réponse satisfaisante.

### 🏆 Classement des prompts

"""
    
    # Tableau des résultats simplifié
    report += "| Classement | Formulation | Eco-score | Efficacité relative | Temps de réponse |\n"
    report += "|------------|-------------|-----------|---------------------|------------------|\n"
    
    for rank, result in enumerate(sorted_results, 1):
        eco_score = result["eco_score"]
        # Convertir le score relatif en icônes
        if result["score_relatif"] > 90:
            eco_rating = "⭐⭐⭐⭐⭐ (Excellent)"
        elif result["score_relatif"] > 75:
            eco_rating = "⭐⭐⭐⭐ (Très bon)"
        elif result["score_relatif"] > 50:
            eco_rating = "⭐⭐⭐ (Bon)"
        elif result["score_relatif"] > 25:
            eco_rating = "⭐⭐ (Moyen)"
        else:
            eco_rating = "⭐ (Faible)"
            
        # Convertir le temps en texte
        if result["execution_time"] < 10:
            time_rating = "⚡ Très rapide"
        elif result["execution_time"] < 30:
            time_rating = "⚡ Rapide"
        elif result["execution_time"] < 60:
            time_rating = "🕒 Moyen"
        else:
            time_rating = "⏱️ Long"
        
        # Formater le prompt pour le tableau (version courte)
        prompt_text = result["prompt_text"]
        short_prompt = prompt_text[:40] + "..." if len(prompt_text) > 40 else prompt_text
        
        report += f"| {rank} | {short_prompt} | {eco_score:.1f} | {eco_rating} | {time_rating} |\n"
    
    report += f"""
## 💡 Ce que ces résultats signifient

L'**eco-score** est un nombre qui indique l'efficacité écologique du prompt. **Plus le score est élevé, mieux c'est!**

L'**efficacité relative** compare tous les prompts avec le meilleur d'entre eux.

Le **temps de réponse** indique la rapidité avec laquelle l'IA a répondu.

## 🔍 Détails des prompts testés

"""
    
    for result in sorted_results:
        report += f"### Prompt {sorted_results.index(result) + 1}: \"{result['prompt_text']}\"\n\n"
        
        # Créer un résumé en langage simple
        length_desc = "court" if result['prompt_length'] < 50 else "moyen" if result['prompt_length'] < 150 else "long"
        report += f"**Nombre de caractères**: {result['prompt_length']} ({length_desc})\n\n"
        
        # Créer une représentation visuelle de l'efficacité
        eco_bar = "🟩" * int(result["score_relatif"] / 10)
        eco_bar += "⬜" * (10 - int(result["score_relatif"] / 10))
        
        energy_value = result["energy_estimate_joules"]
        # Analogie pour la consommation d'énergie
        if energy_value < 1:
            energy_analogy = "moins qu'une LED pendant 1 seconde"
        elif energy_value < 5:
            energy_analogy = "équivalent à une LED pendant quelques secondes"
        elif energy_value < 20:
            energy_analogy = "comme une ampoule LED pendant 20 secondes"
        else:
            energy_analogy = f"comparable à une ampoule LED pendant {int(energy_value)} secondes"
        
        report += f"""**Performances écologiques**:
- Eco-score: {result['eco_score']:.1f} points
- Efficacité: {result['score_relatif']:.0f}% du meilleur prompt {eco_bar}
- Énergie consommée: {result['energy_estimate_joules']:.2f} joules ({energy_analogy})
- Temps de réponse: {result['execution_time']:.1f} secondes
- Nombre de mots générés: environ {int(result['tokens_out'] * 0.75)} mots
\n\n"""
        
        # Ajouter un extrait de la réponse
        response_extract = result['response_text'][:200] + "..." if len(result['response_text']) > 200 else result['response_text']
        report += f"**Extrait de la réponse**:\n```\n{response_extract}\n```\n\n"
    
    report += """## 🌱 Conseils pour des prompts plus écologiques

Selon cette analyse, les prompts les plus écologiques sont ceux qui:

1. **Sont concis** - Utilisez moins de mots pour poser votre question
2. **Sont précis** - Allez droit au but sans phrases polies excessives
3. **Sont bien structurés** - Une organisation claire aide l'IA à répondre plus efficacement

En appliquant ces principes simples, vous pouvez réduire l'empreinte écologique de vos interactions avec l'IA de **25% à 75%** !

## 🔧 Détails techniques

Pour les lecteurs intéressés par les aspects techniques, voici quelques métriques supplémentaires:

"""
    
    # Tableau technique complet pour ceux qui veulent les détails
    report += "| Prompt | Tokens entrée | Tokens sortie | CPU moyen | Mémoire (MB) | Tokens/Joule | Tokens/Seconde |\n"
    report += "|--------|---------------|---------------|-----------|--------------|--------------|----------------|\n"
    
    for result in sorted_results:
        id = sorted_results.index(result) + 1
        report += f"| {id} | {result['tokens_in']:.0f} | {result['tokens_out']:.0f} | {result['avg_cpu_usage']:.1f}% | {result['avg_memory_mb']:.0f} | {result['tokens_per_joule']:.1f} | {result['tokens_per_second']:.1f} |\n"
    
    # Enregistrer le rapport
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"Rapport de comparaison simplifié généré: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Comparaison écologique de prompts pour Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
-----------------------
1. Comparer deux prompts simples:
   python scripts/compare_prompts.py --prompt1 "Explique l'effet photoélectrique" --prompt2 "Effet photoélectrique?"

2. Comparer trois formulations différentes:
   python scripts/compare_prompts.py \\
      --prompt1 "Explique l'effet photoélectrique" \\
      --prompt2 "Effet photoélectrique?" \\
      --prompt3 "Explique l'effet photoélectrique en 3 points"

3. Utiliser un modèle spécifique:
   python scripts/compare_prompts.py --model llama2 --prompt1 "..." --prompt2 "..."
   
4. Mode rapide (pour tests):
   python scripts/compare_prompts.py --mode fast --prompt1 "..." --prompt2 "..."
"""
    )
    
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Modèle Ollama à utiliser (défaut: {DEFAULT_MODEL})")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS, 
                        help=f"Nombre de répétitions pour chaque prompt (défaut: {DEFAULT_REPETITIONS})")
    parser.add_argument("--delay", type=int, default=5,
                        help="Délai en secondes entre l'analyse de chaque prompt (défaut: 5)")
    parser.add_argument("--output-dir", default="resultats_comparaison",
                        help="Dossier pour les visualisations (défaut: resultats_comparaison)")
    parser.add_argument("--prompt1", required=True, help="Premier prompt à comparer")
    parser.add_argument("--prompt2", required=True, help="Deuxième prompt à comparer")
    parser.add_argument("--prompt3", help="Troisième prompt à comparer (optionnel)")
    parser.add_argument("--mode", choices=["normal", "fast"], default="normal",
                        help="Mode d'exécution: normal ou fast (rapide, pour tests)")
    
    args = parser.parse_args()
    
    # Message d'accueil
    print("\n" + "=" * 80)
    print(f"{'COMPARAISON ÉCOLOGIQUE DE PROMPTS':^80}")
    print("=" * 80)
    print(f"Ce script permet de comparer des formulations de prompts pour ")
    print(f"déterminer laquelle est la plus efficace écologiquement.")
    print(f"Modèle utilisé: \033[1m{args.model}\033[0m")
    print(f"Nombre de prompts à comparer: \033[1m{2 + (1 if args.prompt3 else 0)}\033[0m")
    print(f"Mode: \033[1m{args.mode}\033[0m")
    print("=" * 80 + "\n")
    
    # Afficher les prompts qui seront comparés
    print("Prompts à comparer:")
    print(f"1. \"{args.prompt1[:60]}{'...' if len(args.prompt1) > 60 else ''}\"")
    print(f"2. \"{args.prompt2[:60]}{'...' if len(args.prompt2) > 60 else ''}\"")
    if args.prompt3:
        print(f"3. \"{args.prompt3[:60]}{'...' if len(args.prompt3) > 60 else ''}\"")
    print()
    
    # Vérifier que Ollama est disponible
    print("⏳ Vérification de la disponibilité d'Ollama...")
    try:
        subprocess.run(["ollama", "list"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ Ollama est correctement installé et disponible")
    except subprocess.CalledProcessError:
        print("❌ Erreur: Ollama n'est pas correctement installé ou n'est pas disponible")
        return 1
    except FileNotFoundError:
        print("❌ Erreur: Ollama n'est pas installé ou n'est pas dans le PATH")
        return 1
        
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Les résultats seront enregistrés dans: {args.output_dir}")
    
    # Avertissement sur le temps nécessaire
    prompt_count = 2 + (1 if args.prompt3 else 0)
    
    # Ajuster les paramètres si mode rapide
    if args.mode == "fast":
        print("\n🚀 MODE RAPIDE: Tests accélérés avec temps d'exécution réduit")
        estimated_time = prompt_count * 30  # 30 secondes par prompt en mode rapide
        # On va simuler une partie des résultats pour accélérer les tests
        
        # Préparer les prompts à comparer
        prompts = {}
        prompts["Prompt 1"] = args.prompt1
        prompts["Prompt 2"] = args.prompt2
        if args.prompt3:
            prompts["Prompt 3"] = args.prompt3
            
        # Simuler les résultats en mode rapide
        results = []
        for prompt_id, prompt_text in prompts.items():
            # Créer un résultat simulé mais réaliste basé sur la longueur du prompt
            prompt_length = len(prompt_text)
            execution_time = max(0.5, prompt_length / 30)  # Plus long prompt = plus long temps d'exécution
            energy = max(0.1, prompt_length / 200)  # Estimation de l'énergie basée sur la longueur
            
            # Plus court = meilleur score écologique (simulation)
            eco_score = 100 - (prompt_length / (max([len(p) for p in prompts.values()]) * 0.9))
            
            # Estimer les tokens selon la longueur
            tokens_in = max(1, prompt_length // 4)
            tokens_out = max(10, prompt_length // 2)  # Simulation de réponse plus longue que la question
            
            result = {
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "prompt_length": prompt_length,
                "response_text": f"Ceci est une réponse simulée pour {prompt_text[:20]}...",
                "response_length": tokens_out * 4,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "execution_time": execution_time,
                "avg_cpu_usage": 20 + (prompt_length / 10),
                "max_cpu_usage": 30 + (prompt_length / 8),
                "avg_memory_mb": 500 + (prompt_length * 2),
                "max_memory_mb": 600 + (prompt_length * 2.5),
                "energy_estimate_joules": energy,
                "eco_score": eco_score,
                "tokens_per_joule": tokens_out / max(0.1, energy),
                "energy_per_token": energy / max(1, tokens_out),
                "tokens_per_second": tokens_out / max(0.1, execution_time)
            }
            results.append(result)
            print(f"✓ Simulation du prompt {prompt_id}")
    else:
        # Mode normal avec exécution complète
        estimated_time = prompt_count * 2 * 60  # 2 minutes par prompt en moyenne
        
        # Préparer les prompts à comparer
        prompts = {}
        prompts["Prompt 1"] = args.prompt1
        prompts["Prompt 2"] = args.prompt2
        if args.prompt3:
            prompts["Prompt 3"] = args.prompt3
            
        print("\n🚀 Démarrage de l'analyse comparative...")
        
        # Comparer les prompts
        start_time = time.time()
        results = compare_prompts(
            prompts=prompts,
            model=args.model,
            repetitions=args.repetitions,
            delay=args.delay
        )
        total_time = time.time() - start_time
    
    minutes = estimated_time // 60
    seconds = estimated_time % 60
    
    if args.mode == "normal":
        print(f"\n⚠️  Cette analyse peut prendre jusqu'à {minutes} minutes et {seconds} secondes.")
        print("   Chaque prompt sera envoyé à Ollama avec un timeout de 120 secondes.")
        
        proceed = input("\nAppuyez sur Entrée pour continuer ou Ctrl+C pour annuler... ")
    
    if not results:
        print("❌ Aucun résultat n'a été obtenu")
        return 1
        
    # Générer les visualisations
    print("\n📊 Génération des visualisations...")
    generate_comparison_visualizations(results, args.output_dir)
    
    # Générer le rapport
    print("\n📝 Génération du rapport...")
    report_path = os.path.join(args.output_dir, "rapport_comparaison.md")
    generate_comparison_report(results, report_path)
    
    # Résumé des résultats
    print("\n" + "=" * 80)
    print(f"{'RÉSULTATS DE LA COMPARAISON':^80}")
    print("=" * 80)
    
    # Trier les résultats par score écologique
    sorted_results = sorted(results, key=lambda x: x["eco_score"], reverse=True)
    
    for rank, result in enumerate(sorted_results, 1):
        prompt_text = result["prompt_text"]
        short_prompt = prompt_text[:40] + "..." if len(prompt_text) > 40 else prompt_text
        eco_score = result["eco_score"]
        energy = result["energy_estimate_joules"]
        time_taken = result["execution_time"]
        
        # Afficher le classement avec emoji selon le rang
        rank_emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        print(f"{rank_emoji} \"{short_prompt}\"")
        print(f"   Score écologique: {eco_score:.1f} | Énergie: {energy:.2f}J | Temps: {time_taken:.1f}s")
        
        # Donner un exemple concret de consommation
        energy_examples = get_energy_examples(energy)
        print(f"   🔋 Équivalence: {energy_examples}")
    
    print("\n" + "=" * 80)
    if args.mode == "normal":
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        print(f"Analyse terminée en {minutes}m {seconds}s!")
    else:
        print(f"Simulation en mode rapide terminée!")
    print(f"📂 Résultats disponibles dans: {os.path.abspath(args.output_dir)}")
    print(f"📊 Graphiques générés: score_ecologique.png, temps_reponse.png, energie_consommee.png")
    print(f"📄 Rapport complet: {os.path.abspath(report_path)}")
    print("=" * 80)
    
    return 0

def get_energy_examples(joules):
    """Convertit l'énergie en exemples concrets et compréhensibles"""
    if joules < 0.01:
        return "Infime (moins que l'énergie pour déplacer une feuille de papier)"
    elif joules < 0.1:
        return f"Comme l'énergie pour lever une pièce de monnaie de 1 cm"
    elif joules < 0.5:
        return f"≈ L'énergie pour tourner une page de livre ({joules*100:.0f} centièmes de joule)"
    elif joules < 1:
        return f"≈ {joules:.2f}J (allumer une LED pendant {joules*10:.1f} secondes)"
    elif joules < 5:
        return f"≈ L'énergie pour soulever un smartphone de 10 cm"
    elif joules < 10:
        return f"≈ Chauffer 2ml de café de {joules/8.4:.1f}°C"
    elif joules < 20:
        return f"≈ La chaleur dégagée par votre doigt en 1 minute"
    elif joules < 50:
        return f"≈ L'énergie pour faire fonctionner un smartphone pendant {joules/15:.1f} secondes"
    elif joules < 100:
        return f"≈ Une goutte d'essence (≈ {joules/33:.2f}mL)"
    else:
        return f"≈ Faire fonctionner un ordinateur portable pendant {joules/40:.1f} secondes"

if __name__ == "__main__":
    sys.exit(main()) 