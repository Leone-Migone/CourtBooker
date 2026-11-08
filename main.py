import nltk
import json
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score

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

