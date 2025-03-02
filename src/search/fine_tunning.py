from sentence_transformers import SentenceTransformer, InputExample, SentencesDataset, losses
from torch.utils.data import DataLoader
from meilisearch import Client
import  os

model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-dot-v1')

# 2. Inisialisasi Meilisearch client
client = Client(api_key="ms29sk298u23n928o12ij918j21jnmk2192", url="https://mlsrch.knitto.org")  # Pastikan URL sesuai dengan instance Meilisearch Anda
index = client.index('faq')  #

# 2. Ambil data FAQ dari Meilisearch
faq_data_meili = index.search('', {"limit": 200})

# 3. Ekstrak pertanyaan dan jawaban dari data FAQ
faq_questions = [item['pertanyaan'] for item in faq_data_meili['hits']]
faq_answers = [item['jawaban'] for item in faq_data_meili['hits']]

# 4. Membuat pasangan pertanyaan dan jawaban untuk fine-tuning
train_samples = []
for question, answer in zip(faq_questions, faq_answers):
    train_samples.append(InputExample(texts=[question, answer], label=1))  # Label 1 berarti relevan

# 5. Membuat dataset dan dataloader
train_dataset = SentencesDataset(train_samples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)

# 6. Definisikan loss function yang akan digunakan untuk fine-tuning
train_loss = losses.CosineSimilarityLoss(model)

# 7. Fine-tuning model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

# 8. Simpan model yang sudah di-fine-tune dengan nama 'search_sbert'
model_save_path = "../../models/search_sbert"

# Cek dan buat folder jika belum ada
os.makedirs(model_save_path, exist_ok=True)

model.save(model_save_path)
print("✅ Fine-tuning selesai! Model 'search_sbert' telah disimpan.")
