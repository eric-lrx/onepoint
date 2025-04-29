#!/bin/python3

from transformers import pipeline
from ollama import chat
import re

# Appel à Ollama
def Call_Ollama(text):
    response = chat(
        model='deepseek-r1:7b',
        messages=[{'role': 'user', 'content': text}]
    )
    raw_output = response["message"]["content"]
    clean_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)
    print(clean_output.strip())
    return clean_output.strip()

# Initialiser la pipeline avec un modèle multilingue
classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
)

# Définir les catégories à détecter
labels = [
    "remerciement"
]


def classify_prompt(text):
    result = classifier(text, candidate_labels=labels, multi_label=True)
    # print(f"Texte : {text}\n")
    # print("Classification :")
    for label, score in zip(result["labels"], result["scores"]):
        if score > 0.7:
            print(f" - {label}: {score:.2f}")
            return False 
    return True

# Exemple d'utilisation
if __name__ == "__main__":
    text = ""
    call_api = False
    while (text != "exit"):
        text = input("=> ")
        call_api = classify_prompt(text)
        if call_api == True:
            Call_Ollama(text)
        else:
            print("Message pas conforme")
