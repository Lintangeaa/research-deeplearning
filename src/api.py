from fastapi import FastAPI
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

with open("models/rf/intent_model.pkl", "rb") as model_file:
    intent_model = pickle.load(model_file)

with open("models/rf/tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

MODEL_PATH = "models/faiss/"
FAISS_INDEX_FILE = MODEL_PATH + "faq_index.faiss"
FAQ_QUESTIONS_FILE = MODEL_PATH + "faq_questions.pkl"
FAQ_ANSWERS_FILE = MODEL_PATH + "faq_answers.pkl"
FAQ_EMBEDDINGS_FILE = MODEL_PATH + "faq_embeddings.pkl"
FAQ_QUESTION_CLASSIFER = MODEL_PATH + "faq_classifier.pkl"

sbert_model = SentenceTransformer("models/search_sbert")

faiss_index = faiss.read_index(FAISS_INDEX_FILE)

with open(FAQ_QUESTIONS_FILE, "rb") as f:
    faq_questions = pickle.load(f)

with open(FAQ_ANSWERS_FILE, "rb") as f:
    faq_answers = pickle.load(f)

with open(FAQ_EMBEDDINGS_FILE, "rb") as f:
    faq_embeddings = pickle.load(f)

with open(FAQ_QUESTION_CLASSIFER, "rb") as f:
    faq_classifier = pickle.load(f)

faq_embeddings = faq_embeddings / np.linalg.norm(faq_embeddings, axis=1, keepdims=True)

class TextInput(BaseModel):
    text: str

@app.post("/predict/")
def predict_intent(input_data: TextInput):
    new_text_tfidf = vectorizer.transform([input_data.text])
    prediction = intent_model.predict(new_text_tfidf)[0]
    return {"text": input_data.text, "intent": prediction}


@app.post("/search/")
def search_faq(input_data: TextInput):
    query = input_data.text

    try:
        predicted_question = faq_classifier.predict([query])[0]
        predicted_proba = faq_classifier.predict_proba([query])

        print(np.max(predicted_proba))
        print(predicted_question)
    except Exception:
        predicted_question = None

    print('PREDICTED: ', predicted_question)

    if predicted_question:
        query_embedding = sbert_model.encode([predicted_question], normalize_embeddings=True)
    else:
        query_embedding = sbert_model.encode([query], normalize_embeddings=True)

    query_embedding = np.array(query_embedding, dtype=np.float32)

    distances, indices = faiss_index.search(query_embedding, k=5)

    best_match = None
    best_score = -1

    for i in range(5):
        idx = indices[0][i]
        score = cosine_similarity([query_embedding[0]], [faq_embeddings[idx]])[0][0]

        if score > best_score:
            best_score = score
            best_match = idx

    if best_score < 0.7:
        return {
            "query": query,
            "matched_question": None,
            "answer": "Maaf, saya tidak mengerti pertanyaan Anda.",
            "cosine_similarity": float(best_score)
        }

    matched_question = faq_questions[best_match]
    answer = faq_answers[best_match]

    return {
        "query": query,
        "matched_question": matched_question,
        "answer": answer,
        "cosine_similarity": float(best_score)
    }

