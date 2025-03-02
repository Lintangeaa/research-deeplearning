import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pickle
from meilisearch import Client

# 🔹 Path untuk penyimpanan dataset & model
dataset_path = "../../models/faiss/faq_dataset.pkl"
model_path = "../../models/faiss/faq_classifier.pkl"

# 🔹 Inisialisasi Meilisearch
client = Client(api_key="ms29sk298u23n928o12ij918j21jnmk2192", url="https://mlsrch.knitto.org")
index = client.index('faq')

# 🔹 Ambil data FAQ dari Meilisearch
faq_data_meili = index.search('', {"limit": 500})  # Ambil lebih banyak data
faq_questions = [item['pertanyaan'] for item in faq_data_meili['hits']]

# 🔹 Buat dataset awal (tanpa variasi pertanyaan pengguna)
data = {
    "user_question": faq_questions,   # Input = pertanyaan asli
    "faq_question": faq_questions     # Output = pertanyaan asli
}

df = pd.DataFrame(data)

# 🔹 Simpan dataset ke file
with open(dataset_path, "wb") as f:
    pickle.dump(df, f)

print(f"✅ Dataset berhasil disimpan di {dataset_path}")

# 🔹 Split data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(df["user_question"], df["faq_question"], test_size=0.2, random_state=42)

# 🔹 Buat pipeline TF-IDF + RandomForest
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# 🔹 Latih model
model.fit(X_train, y_train)

# 🔹 Simpan model ke file
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model Random Forest berhasil dilatih dan disimpan di {model_path}!")
