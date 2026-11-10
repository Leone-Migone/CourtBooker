import nltk
import json
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
import re
from datetime import datetime, timedelta

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

# mention not removing stop words as they might be important for the context of the patterns in report

X_train, X_test, y_train, y_test = train_test_split(patterns, labels, test_size=0.24, random_state=42, stratify = labels)

# Create a CountVectorizer and TfidfTransformer to convert the text data into numerical features
count_vectorizer = CountVectorizer()
X_train_counts = count_vectorizer.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts) 

# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB().fit(X_train_tfidf, y_train)

# Transform the test data using the same vectorizer and transformer
X_new_counts = count_vectorizer.transform(X_test)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# Make predictions on the test data
predicted = classifier.predict(X_new_tfidf)

print("Classification Report:")
print(classification_report(y_test, predicted))
print("Accuracy Score:", accuracy_score(y_test, predicted))

# try some more trial patterns to see how the model performs
trial_patterns = [ "I'd like to reserve a court for tomorrow evening.","Do you have anything free after 6", "I can't make my game anymore", "When's my next session", "What am I able to do here?" ]
trial_vectors = count_vectorizer.transform(trial_patterns)
trial_tfidf = tfidf_transformer.transform(trial_vectors)
trial_predictions = classifier.predict(trial_tfidf)

print("Trial Predictions:")
for pattern, prediction in zip(trial_patterns, trial_predictions):
    print(f"Pattern: {pattern} -> Predicted Intent: {prediction}")  


SPORTS = ["tennis", "badminton", "football"]

# map number words to their corresponding integer
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5
}



def extract_slots(text):
    slots = {
        "sport": None,
        "date": None,
        "time": None,
        "duration": None
    }

    text = text.lower().strip()

    #extract sport
    for sport in SPORTS:
        if re.search(rf"\b{sport}\b", text):
            slots["sport"] = sport
            break

    #Extract date
    today = datetime.now().date()

    if re.search(r"\btomorrow\b", text):
        slots["date"] = (today + timedelta(days=1)).isoformat()

    elif re.search(r"\btoday\b", text):
        slots["date"] = today.isoformat()

    else:
        weekdays = [
            "monday", "tuesday", "wednesday",
            "thursday", "friday", "saturday", "sunday"
        ]

        for i, day in enumerate(weekdays):
            if re.search(rf"\b{day}\b", text):
                days_ahead = (i - today.weekday()) % 7
                slots["date"] = (
                    today + timedelta(days=days_ahead)
                ).isoformat()
                break

    #Extract time
    time_match = re.search(
        r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
        text
    )

    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2) or 0)
        period = time_match.group(3)

        if 1 <= hour <= 12 and 0 <= minute <= 59:
            hour = hour % 12
            if period == "pm":
                hour += 12

            slots["time"] = f"{hour:02d}:{minute:02d}"

    else:
        time_match = re.search(
            r"\b(\d{1,2}):(\d{2})\b",
            text
        )

        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))

            if 0 <= hour <= 23 and 0 <= minute <= 59:
                slots["time"] = f"{hour:02d}:{minute:02d}"

    # extract duration
    duration_match = re.search(
        r"\b(\d+|one|two|three|four|five)\s*(hours?|hrs?|minutes?|mins?)\b",
        text
    )

    # if duration found convert it to minutes if neededand store in slots
    if duration_match:
        amount_text = duration_match.group(1)
        unit = duration_match.group(2)

        amount = NUMBER_WORDS.get(amount_text)
        if amount is None:
            amount = int(amount_text)

        if unit.startswith(("hour", "hr")):
            slots["duration"] = amount * 60
        else:
            slots["duration"] = amount

    return slots



examples = [
    "I'd like to book a tennis court tomorrow at 5pm for two hours",
    "Can I reserve badminton on Friday at 14:30?",
    "Book football today at 7 pm for 90 minutes",
    "I want a squash court on Monday",
]

for sentence in examples:
    print(sentence)
    print(extract_slots(sentence))


