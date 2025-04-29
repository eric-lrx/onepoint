#!/bin/python3

from transformers import pipeline
import torch

# Initialiser la pipeline avec un modèle multilingue
classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
)

# Définir les catégories à détecter
labels = [
    # Politique et société
    "politique", "gouvernement", "élections", "militaire", "diplomatie", "lois et justice", "manifestation",
    # Violence et sécurité
    "violence", "crime", "terrorisme", "harcèlement", "cybercriminalité", "abus",
    # Technologie et informatique
    "code informatique", "programmation", "intelligence artificielle", "cybersécurité", "cryptomonnaie", "jeux vidéo",
    # Religion et spiritualité
    "religion", "croyance", "athéisme", "spiritualité",
    # Santé et médecine
    "santé", "médecine", "psychologie", "alimentation", "sport",
    # Art et culture
    "cinéma", "musique", "peinture", "littérature", "mode",
    # Sciences et environnement
    "science", "physique", "biologie", "écologie", "espace",
    # Vie personnelle et relations
    "sujet personnel", "famille", "amitié", "relation amoureuse", "travail", "éducation",
    # Finance et économie
    "économie", "finance", "immobilier", "entrepreneuriat",
    # Communication et médias
    "journalisme", "réseaux sociaux", "publicité", "marketing",
    # Catégories sensibles
    "toxique", "discrimination", "sexisme", "racisme", "homophobie", "controverse",
    # Divers
    "humour", "philosophie", "voyage", "animaux", "cuisine"
]

def classify_prompt(text):
    result = classifier(text, candidate_labels=labels, multi_label=True)
    print(f"Texte : {text}\n")
    print("Classification :")
    for label, score in zip(result["labels"], result["scores"]):
        if score > 0.7:
            print(f" - {label}: {score:.2f}")
    return result

# Exemple d'utilisation
if __name__ == "__main__":
    text = ""
    while (text != "exit"):
        text = input("=> ")
        classify_prompt(text)