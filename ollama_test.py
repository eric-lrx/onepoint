#!/bin/python3

from ollama import chat
import re

# Appel à Ollama
response = chat(
    model='deepseek-r1:7b',
    messages=[{'role': 'user', 'content': "bonjour comment tu vas"}]
)

raw_output = response["message"]["content"]

clean_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)

print(clean_output.strip())
