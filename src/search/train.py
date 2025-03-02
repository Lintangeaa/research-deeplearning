from sentence_transformers import SentenceTransformer
import faiss
from meilisearch import Client
import numpy as np
import pickle

# Load SBERT model (pastikan path benar)
fine_tuned_model = SentenceTransformer("../../models/search_sbert")

# Inisialisasi Meilisearch
client = Client(api_key="ms29sk298u23n928o12ij918j21jnmk2192", url="https://mlsrch.knitto.org")
index = client.index('faq')

# 🔹 Ambil data FAQ dari Meilisearch
faq_data_meili = index.search('', {"limit": 200})
faq_questions = [item['pertanyaan'] for item in faq_data_meili['hits']]
faq_answers = [item['jawaban'] for item in faq_data_meili['hits']]

# 🔹 Generate embedding pertanyaan FAQ
faq_embeddings = fine_tuned_model.encode(faq_questions, normalize_embeddings=True)  # Normalisasi agar cocok dengan IndexFlatIP
faq_embeddings = np.array(faq_embeddings, dtype=np.float32)  # Pastikan float32

# 🔹 Buat FAISS Index dengan Inner Product (IP) untuk Cosine Similarity
dimension = faq_embeddings.shape[1]
index_faiss = faiss.IndexFlatIP(dimension)  # Menggunakan IP agar mendukung Cosine Similarity
index_faiss.add(faq_embeddings)

# 🔹 Simpan FAISS ke file
faiss.write_index(index_faiss, "../../models/faiss/faq_index.faiss")

# 🔹 Simpan FAQ Questions & Answers
with open("../../models/faiss/faq_questions.pkl", "wb") as f:
    pickle.dump(faq_questions, f)

with open("../../models/faiss/faq_answers.pkl", "wb") as f:
    pickle.dump(faq_answers, f)

# 🔹 Simpan Embeddings ke File
with open("../../models/faiss/faq_embeddings.pkl", "wb") as f:
    pickle.dump(faq_embeddings, f)

print("✅ FAISS training selesai! Model 'search_sbert' telah digunakan untuk pencarian.")
