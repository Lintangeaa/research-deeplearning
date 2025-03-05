from fastapi import FastAPI
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
import joblib

app = FastAPI()

with open("models/rf/intent_model.pkl", "rb") as model_file:
    intent_model = pickle.load(model_file)

with open("models/rf/tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load model Logistic Regression FAQ
faq_model = joblib.load("models/faq/logistic_faq_model.pkl")

# Load TF-IDF Vectorizer
vectorizer = joblib.load("models/faq/vectorizer.pkl")

class TextInput(BaseModel):
    text: str

@app.post("/predict/")
def predict_intent(input_data: TextInput):
    new_text_tfidf = vectorizer.transform([input_data.text])
    prediction = intent_model.predict(new_text_tfidf)[0]
    return {"text": input_data.text, "intent": prediction}


@app.post("/faq/")
def get_faq_answer(input_data: TextInput):
    # Konversi teks input ke fitur TF-IDF
    input_tfidf = vectorizer.transform([input_data.text])

    # Prediksi intent/jawaban langsung dari model Logistic Regression
    predicted_answer = faq_model.predict(input_tfidf)[0]

    return {
        "question": input_data.text,
        "answer": predicted_answer
    }