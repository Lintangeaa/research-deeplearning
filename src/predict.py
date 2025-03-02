import pickle

# Memuat model dan vectorizer dari folder 'models'
with open('../models/rf/intent_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('../models/rf/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Input dari pengguna
new_text = []
print("Masukkan kalimat yang ingin diprediksi (ketik 'selesai' untuk berhenti):")

while True:
    # Menerima input dari pengguna
    user_input = input("Masukkan kalimat: ")

    # Jika input 'selesai' maka berhentiap
    if user_input.lower() == 'selesai':
        break

    new_text.append(user_input)

# Melakukan transformasi pada teks baru
new_text_tfidf = vectorizer.transform(new_text)

# Memprediksi intent dari teks baru
predictions = model.predict(new_text_tfidf)

# Menampilkan hasil prediksi
for i, text in enumerate(new_text):
    print(f"Teks: '{text}' => Prediksi Intent: {predictions[i]}")
