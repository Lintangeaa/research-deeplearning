import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from meilisearch import Client
import numpy as np

# 🔹 Path file model
model_path = "../../models/faiss/faq_classifier.pkl"

# 🔹 Inisialisasi Meilisearch
client = Client(api_key="ms29sk298u23n928o12ij918j21jnmk2192", url="https://mlsrch.knitto.org")
index = client.index('faq')

# 🔹 Ambil data FAQ dari Meilisearch untuk evaluasi
faq_data_meili = index.search('', {"limit": 200})  # Ambil sebagian data sebagai test set
faq_questions = [item['pertanyaan'] for item in faq_data_meili['hits']]
faq_ids = [item['id'] for item in faq_data_meili['hits']]

# 🔹 Buat dataset uji (question, question)
X_test = faq_questions
y_test = faq_questions  # Ground truth sama dengan pertanyaan asli

# 🔹 Load model Random Forest
with open(model_path, "rb") as f:
    model = pickle.load(f)

# 🔹 Prediksi dengan model
proba = model.predict_proba(X_test)  # Ambil probabilitas prediksi
y_pred = model.predict(X_test)

# 🔹 Evaluasi model
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, average='macro', zero_division=1),
    "Recall": recall_score(y_test, y_pred, average='macro', zero_division=1),
    "F1-Score": f1_score(y_test, y_pred, average='macro', zero_division=1)
}

# 🔹 Buat DataFrame untuk analisis
confidence_scores = [max(proba[i]) for i in range(len(y_pred))]  # Ambil nilai confidence tertinggi
results = pd.DataFrame({
    "ID": faq_ids,
    "Pertanyaan Asli": y_test,
    "Prediksi Model": y_pred,
    "Confidence": confidence_scores,
    "Benar/Salah": [1 if y_test[i] == y_pred[i] else 0 for i in range(len(y_test))]
})

# 🔹 Filter hanya prediksi yang salah
df_salah = results[results["Benar/Salah"] == 0]
print("Data yang salah diprediksi oleh model:")
print(df_salah[["ID", "Pertanyaan Asli", "Prediksi Model", "Confidence"]])

# 🔹 Simpan hasil salah prediksi ke file CSV
df_salah.to_csv("prediksi_salah.csv", index=False, encoding="utf-8-sig")

# 🔹 Tampilkan hasil evaluasi
print("Evaluasi Model Random Forest")
for metric, value in metrics.items():
    print(f"\u2705 {metric}: {value:.4f}")
