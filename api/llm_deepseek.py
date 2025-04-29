from transformers import pipeline
from ollama import chat
import re

def Call_Ollama(text):
    response = chat(
        model='deepseek-r1:7b',
        messages=[{'role': 'user', 'content': text}]
    )
    raw_output = response["message"]["content"]
    clean_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)
    print(clean_output.strip())
    return clean_output.strip()

classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
)

labels = [
    "code informatique",
    "chercharble sur google facilement",
    "bibliothèque standard C",
]


def classify_prompt(text):
    result = classifier(text, candidate_labels=labels, multi_label=True)
    scores = dict(zip(result["labels"], result["scores"]))

    code_score = scores.get("code informatique", 0)
    google_score = scores.get("cherchable sur google facilement", 0)
    lib_score = scores.get("bibliothèque standard C", 0)

    if code_score > 0.7:
        print("code informatique")
        if lib_score > 0.3:
            print("bibliothèque standard C")
            return True, "renvoyer vers man"
        elif google_score > 0.8:
            print("cherchable sur google facilement")
            return True, f"https://letmegooglethat.com/?q={text.replace(" ", "+")}"
        else:
            print("code informatique")
            return True, "réponse correcte (code informatique)"
    else:
        print("fakse")
        return False, "Question hors sujet utilise une autre IA"

def ask_ai(question: str):
    call_api = classify_prompt(question)
    if call_api[0] == True:
        return Call_Ollama(question)
    else:
        return call_api[1]
