import nltk
import json
from nltk.tokenize import word_tokenize

with open("./data/intents.json") as file:
    data = json.load(file)

patterns = []
labels = []

# tokenize the patterns and store them along with their corresponding labels
for intent in data["intents"]:
    print(f"Intent: {intent['label']}")
    for pattern in intent["patterns"]:
        tokens = word_tokenize(pattern)
        print(f"Pattern: {pattern} -> Tokens: {tokens}")
        clean_pattern = " ".join(tokens)
        patterns.append(clean_pattern)
        labels.append(intent["label"])
        



