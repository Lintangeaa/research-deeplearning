import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 📌 Path Model & Data
MODEL_PATH = "../../models/faiss/"
FAISS_INDEX_FILE = MODEL_PATH + "faq_index.faiss"
FAQ_QUESTIONS_FILE = MODEL_PATH + "faq_questions.pkl"
FAQ_ANSWERS_FILE = MODEL_PATH + "faq_answers.pkl"
FAQ_EMBEDDINGS_FILE = MODEL_PATH + "faq_embeddings.pkl"

# Load Model SBERT
sbert_model = SentenceTransformer('../../models/search_sbert')

# Load FAISS Index
faiss_index = faiss.read_index(FAISS_INDEX_FILE)

# Load Data FAQ (Pertanyaan & Jawaban)
with open(FAQ_QUESTIONS_FILE, "rb") as f:
    faq_questions = pickle.load(f)

with open(FAQ_ANSWERS_FILE, "rb") as f:
    faq_answers = pickle.load(f)

# Load FAQ Embeddings
with open(FAQ_EMBEDDINGS_FILE, "rb") as f:
    faq_embeddings = pickle.load(f)

# ✅ Fungsi Pencarian FAQ
def search_faq(query):
    query_embedding = sbert_model.encode([query], normalize_embeddings=True)  # Pastikan normalisasi
    query_embedding = np.array(query_embedding, dtype=np.float32)

    # Cari di FAISS
    distances, indices = faiss_index.search(query_embedding, 1)

    best_index = indices[0][0]
    matched_question = faq_questions[best_index]
    answer = faq_answers[best_index]

    # Hitung cosine similarity dengan embedding FAQ
    cos_sim = cosine_similarity([query_embedding[0]], [faq_embeddings[best_index]])[0][0]

    return {
        "query": query,
        "matched_question": matched_question,
        "answer": answer,
        "cosine_similarity": cos_sim
    }

# 🔍 Evaluasi dengan Sample Query
test_queries = [
    ("Apakah sudah ringspun?", "Apakah kain knitto sudah ringspun ?"),
    ("Benangnya sudah import?", "Apakah benang kita import?"),
    ("Apa sih enzyme?", "Apa itu enzyme"),
    ("kalo beli banyak bisa dapat diskon?", "Ada diskon tidak?")
]

correct_count = 0
total_count = len(test_queries)

print("\n📊 Evaluasi Pencocokan FAQ\n" + "=" * 50)
for query, expected_question in test_queries:
    result = search_faq(query)
    print(f"Query: {query}")
    print(f"Predicted Question: {result['matched_question']}")
    print(f"Expected Question: {expected_question}")
    print(f"Cosine Similarity: {result['cosine_similarity']:.4f}")
    print("=" * 50)

    # Evaluasi: Jika pertanyaan yang diprediksi sama dengan ekspektasi
    if result['matched_question'] == expected_question:
        correct_count += 1

# 🔥 Hitung Akurasi
accuracy = correct_count / total_count * 100
print(f"\n✅ Akurasi Pencocokan: {accuracy:.2f}%")