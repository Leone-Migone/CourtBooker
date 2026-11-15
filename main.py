import nltk
import json
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
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



def predict_intent(text):
    count_vector = count_vectorizer.transform([text])
    tfidf = tfidf_transformer.transform(count_vector)

    probabilities = classifier.predict_proba(tfidf)[0]
    # get the confidence of the prediction
    confidence = max(probabilities)
    # get intent with highest probability 
    predicted_intent = classifier.classes_[probabilities.argmax()]

    # if the confidence is below a certain threshold, return None and ask for more information
    if confidence < 0.35:
        return None, confidence
    
    return predicted_intent, confidence

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
    intent, confidence = predict_intent(sentence)
    print(confidence,intent)
    if intent is None:
        print("I'm not sure what you mean. Could you please provide more details?")
    elif intent == "BOOK_COURT":
        print(extract_slots(sentence))
    

# conversation state to keep track of name
conversation = {
    "name": None,
    "awaiting_name": False
}



def chatbot_response(user_input):
    text = user_input.strip()
    lower_text = text.lower()

    # Handle greetings
    if lower_text in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]:
        if conversation["name"] is None:
            conversation["awaiting_name"] = True
            return "Hello! Welcome to the Sports Centre Booking System. What's your name?"

        return f"Hello {conversation['name']}! How can I help you?"

    #Handle name 
    name_match = re.search(
        r"\b(?:my name is|call me|i am|i'm|im)\s+([a-zA-Z]+)\b",
        text,
        re.IGNORECASE
    )
    
    if name_match:
        conversation["name"] = name_match.group(1).capitalize()
        conversation["awaiting_name"] = False

        return f"Nice to meet you, {conversation['name']}!"

    #Handle answers to "What's your name?"
    if conversation["awaiting_name"]:
        if re.fullmatch(r"[a-zA-Z]+", text):
            conversation["name"] = text.capitalize()
            conversation["awaiting_name"] = False

            return f"Nice to meet you, {conversation['name']}!"

        return "Sorry, could you tell me just your first name?"

    #Recall the user's name
    if lower_text in ["what is my name?", "what is my name",
                      "what's my name?", "what's my name", "whats my name?"]:
        if conversation["name"]:
            return f"Your name is {conversation['name']}."

        return "I don't know your name yet. What should I call you?"

    #classify the intent of the user input
    intent, confidence = predict_intent(text)

    if intent == "HELP":
        return (
            "I can help you book sports courts, check availability, "
            "cancel reservations, and view your bookings. "
            "What would you like to do?"
        )

    # temps for other intents
    if intent == "BOOK_COURT":
        return "Sure! I can help you book a court."

    if intent == "CHECK_AVAILABILITY":
        return "I can help you check which courts are available."

    if intent == "CANCEL_BOOKING":
        return "I can help you cancel a reservation."

    if intent == "VIEW_BOOKINGS":
        return "I can help you view your existing bookings."

    if intent == "INFORMATIONS":
        return "What would you like to know about our sports centre?"

    return "Sorry, I didn't understand. Could you rephrase that?"

def main():
    print("Welcome to the Sports Centre Chatbot!")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye!")
            break

        if not user_input:
            continue

        response = chatbot_response(user_input)
        print(f"Bot: {response}")


if __name__ == "__main__":
    main()