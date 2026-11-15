import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#load qa dataset
qa_data = {}

with open("./data/COMP3074-CW1-Dataset.csv", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        question = row["Question"].strip().lower()
        answer = row["Answer"].strip()
        #some Q have multiple answers so store in list
        if answer and answer not in qa_data.setdefault(question, []):
            qa_data[question].append(answer)

questions = list(qa_data.keys())

# Create a vectorizer to convert the questions into numerical features
qa_vectorizer = TfidfVectorizer(stop_words="english")
question_vectors = qa_vectorizer.fit_transform(questions)





def answer_question(user_input):
    #transform using the vectorizer
    user_vector = qa_vectorizer.transform([user_input])

    #comparison with dataset questions
    similarities = cosine_similarity(user_vector, question_vectors)[0]

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    if best_score < 0.65:
        return None

    matched_question = questions[best_index]
    answers = qa_data[matched_question]

    return " ".join(answers)
