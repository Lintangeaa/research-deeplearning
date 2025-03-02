import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# 🔹 Path penyimpanan dataset & model
dataset_path = "../../models/faiss/faq_dataset.pkl"
model_path = "../../models/faiss/faq_classifier.pkl"

# 🔹 Load dataset lama jika ada
try:
    with open(dataset_path, "rb") as f:
        df = pickle.load(f)
        print("✅ Dataset lama ditemukan! Total pertanyaan:", len(df))
except FileNotFoundError:
    df = pd.DataFrame(columns=["user_question", "faq_question"])
    print("🚀 Tidak ada dataset lama, memulai dataset baru...")

while True:
    print("\n📌 Tambahkan FAQ baru atau ketik 'exit' untuk keluar.")
    faq = input("Masukkan pertanyaan utama (FAQ): ").strip()
    if faq.lower() == "exit":
        break

    while True:
        user_question = input(f"Masukkan variasi pertanyaan untuk '{faq}' (atau ketik 'done'): ").strip()
        if user_question.lower() == "done":
            break
        df = pd.concat([df, pd.DataFrame([{"user_question": user_question, "faq_question": faq}])], ignore_index=True)

with open(dataset_path, "wb") as f:
    pickle.dump(df, f)
print(f"✅ Dataset diperbarui! Total pertanyaan: {len(df)}")

X_train, X_test, y_train, y_test = train_test_split(df["user_question"], df["faq_question"], test_size=0.2, random_state=42)

# 🔹 Buat pipeline TF-IDF + RandomForest
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# 🔹 Latih model ulang dengan dataset baru
model.fit(X_train, y_train)

# 🔹 Simpan model yang sudah diperbarui
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("✅ Model Random Forest berhasil diperbarui dan disimpan!")
