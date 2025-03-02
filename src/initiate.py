import pandas as pd
import os
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Fungsi untuk memuat model dan vectorizer
def load_model_and_vectorizer():
    try:
        with open('../models/intent_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('../models/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except FileNotFoundError:
        print("Model atau vectorizer tidak ditemukan. Membuat model baru.")
        return MultinomialNB(), TfidfVectorizer()

# Fungsi untuk menyimpan model dan vectorizer
def save_model_and_vectorizer(model, vectorizer):
    if not os.path.exists('../models'):
        os.makedirs('../models')
    with open('../models/intent_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('../models/tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

# Fungsi untuk membaca data dari file JSON
def load_data_from_json():
    try:
        with open('../data/intent_json.json', 'r') as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        print("File JSON tidak ditemukan. Menggunakan data kosong.")
        return {'text': [], 'intent': []}
    except json.JSONDecodeError:
        print("Error dalam membaca file JSON. Menggunakan data kosong.")
        return {'text': [], 'intent': []}

# Fungsi untuk melatih model dengan data yang ada dalam JSON
def train_model_from_json():
    # Memuat data dari file JSON
    data = load_data_from_json()

    if not data['text'] or not data['intent']:
        print("Data kosong. Tidak dapat melatih model.")
        return

    # Memuat model dan vectorizer yang ada
    model, vectorizer = load_model_and_vectorizer()

    # Membuat dataframe dari data yang ada
    df = pd.DataFrame(data)

    # Memisahkan fitur dan target
    X = df['text']
    y = df['intent']

    # Transformasi data menggunakan vectorizer
    X_tfidf = vectorizer.fit_transform(X)

    # Melatih model dengan data
    model.fit(X_tfidf, y)

    # Menyimpan model dan vectorizer yang sudah terlatih
    save_model_and_vectorizer(model, vectorizer)

    print("Model berhasil dilatih dan disimpan.")

# Proses utama untuk pelatihan model
if __name__ == "__main__":
    train_model_from_json()
