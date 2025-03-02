import os
import json
import re
import string
import pickle
import pandas as pd
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

stop_words = set(stopwords.words('indonesian'))

# Inisialisasi stemmer untuk Bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Daftar kata-kata informal yang sering muncul
slang_dict = {
    "gmn": "bagaimana",
    "knp": "kenapa",
    "tdk": "tidak",
    "bgt": "banget",
    "sdh": "sudah",
    "sy": "saya",
    "lg": "lagi"
}


def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'\d+', '', text)  # Hapus angka
    text = text.translate(str.maketrans('', '', string.punctuation))  # Hapus tanda baca
    text = text.strip()  # Hapus spasi berlebih
    text = re.sub(r'[^\w\s]', '', text)  # Hapus karakter non-alfabet

    # Ganti kata slang dengan kata baku
    words = text.split()
    words = [slang_dict[word] if word in slang_dict else word for word in words]

    # Hapus stopwords
    words = [word for word in words if word not in stop_words]

    # Stemming
    words = [stemmer.stem(word) for word in words]

    return ' '.join(words)


def load_model_and_vectorizer():
    try:
        with open('../models/intent_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('../models/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except FileNotFoundError:
        print("Model atau vectorizer tidak ditemukan. Membuat yang baru.")
        return MultinomialNB(), TfidfVectorizer()


def save_model_and_vectorizer(model, vectorizer):
    if not os.path.exists('../models'):
        os.makedirs('../models')
    with open('../models/intent_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('../models/tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)


def load_data_from_json():
    try:
        with open('../data/intent_json.json', 'r') as json_file:
            data = json.load(json_file)
            return data if data else {'text': [], 'intent': []}
    except (FileNotFoundError, json.JSONDecodeError):
        print("File JSON tidak ditemukan atau rusak. Menggunakan data kosong.")
        return {'text': [], 'intent': []}


def save_data_to_json(new_data):
    if not os.path.exists('../data'):
        os.makedirs('../data')
    existing_data = load_data_from_json()
    new_data['text'] = [preprocess_text(text) for text in new_data['text']]
    existing_data['text'].extend(new_data['text'])
    existing_data['intent'].extend(new_data['intent'])
    with open('../data/intent_json.json', 'w') as json_file:
        json.dump(existing_data, json_file)


def train_model_with_new_data():
    data = load_data_from_json()
    df = pd.DataFrame(data)
    df['text'] = df['text'].apply(preprocess_text)
    X, y = df['text'], df['intent']

    model, vectorizer = load_model_and_vectorizer()

    if not X.empty:
        X_tfidf = vectorizer.fit_transform(X)
        model.fit(X_tfidf, y)
        save_model_and_vectorizer(model, vectorizer)
    else:
        print("Data tidak cukup untuk melatih model.")

    return model, vectorizer


def get_user_input():
    intent_options = {
        1: 'stok',
        2: 'status_order',
        3: 'faq',
        4: 'greeting',
        5: 'harga'
    }
    new_data = {'text': [], 'intent': []}

    print("Masukkan kalimat dan pilih intentnya (ketik 'selesai' untuk berhenti):")
    while True:
        user_text = input("Masukkan kalimat: ")
        if user_text.lower() == 'selesai':
            break

        print("\nPilih intent untuk kalimat ini:")
        for key, value in intent_options.items():
            print(f"{key}. {value}")

        while True:
            try:
                user_intent_number = int(input("\nMasukkan nomor intent (1-5): "))
                if user_intent_number in intent_options:
                    user_intent = intent_options[user_intent_number]
                    break
                else:
                    print("Nomor intent tidak valid. Pilih antara 1 dan 5.")
            except ValueError:
                print("Input tidak valid. Masukkan angka antara 1 dan 5.")

        new_data['text'].append(user_text)
        new_data['intent'].append(user_intent)

    save_data_to_json(new_data)
    return new_data


if __name__ == "__main__":
    new_data = get_user_input()
    train_model_with_new_data()