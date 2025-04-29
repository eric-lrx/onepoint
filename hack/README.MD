# Étude sur les prompts écologiques pour Ollama

Ce projet vise à analyser et optimiser l'efficacité écologique des prompts utilisés avec les modèles de langage Ollama, en mesurant la consommation de ressources et en identifiant les bonnes pratiques pour réduire l'empreinte environnementale.

## Structure du projet

- `scripts/` : Scripts d'analyse et de mesure
- `data/` : Prompts à tester et résultats
- `visualisations/` : Graphiques générés
- `docs/` : Documentation méthodologique

## Prérequis

- Python 3.6 ou supérieur
- Ollama installé et accessible dans le PATH
- Bibliothèques Python : pandas, numpy, matplotlib, pyRAPL

## Installation

```bash
pip install pandas numpy matplotlib pyRAPL



commande : python scripts/compare_prompts.py --mode fast --prompt1 "Explique l'effet photoélectrique" --prompt2 "Effet photoélectrique?"