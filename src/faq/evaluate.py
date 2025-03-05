import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Muat model dan vectorizer
model = joblib.load("../../models/faq/logistic_faq_model.pkl")
vectorizer = joblib.load("../../models/faq/vectorizer.pkl")

# Data evaluasi
test_questions = [
    "Bagaimana cara mengganti password?",
    "Dimana saya bisa mendapatkan bantuan pelanggan?",
    "Apakah ada aplikasi untuk iPhone?",
    "Saya ingin mendaftar akun baru, bagaimana caranya?",
    "Bagaimana cara menutup akun saya?",
    "Dimana letak menu reset password?",
    "Bisakah saya menghubungi customer service lewat email?",
    "Saya ingin tahu apakah ada aplikasi di Play Store?",
    "Langkah-langkah membuat akun pengguna baru?",
    "Apa prosedur untuk menghapus akun saya?",
    "Kalo mau hapus akun dimana?",
    "Registrasinya dimana si?"
]

test_labels = [
    "Untuk mereset password, klik tombol 'Lupa Password' pada halaman login, masukkan email Anda, lalu ikuti instruksi yang dikirimkan ke email Anda.",
    "Anda dapat menghubungi customer support melalui email support@example.com atau menggunakan fitur live chat di dalam aplikasi.",
    "Ya, aplikasi ini tersedia untuk Android dan iOS. Anda dapat mengunduhnya di Google Play Store dan Apple App Store dengan mencari nama aplikasi kami.",
    "Untuk mendaftar akun baru, buka aplikasi, pilih 'Daftar', masukkan informasi yang diminta, lalu verifikasi melalui email atau nomor telepon.",
    "Untuk menghapus akun Anda, masuk ke pengaturan akun, pilih opsi 'Hapus Akun', dan ikuti instruksinya. Jika butuh bantuan lebih lanjut, hubungi customer support.",
    "Untuk mereset password, klik tombol 'Lupa Password' pada halaman login, masukkan email Anda, lalu ikuti instruksi yang dikirimkan ke email Anda.",
    "Anda dapat menghubungi customer support melalui email support@example.com atau menggunakan fitur live chat di dalam aplikasi.",
    "Ya, aplikasi ini tersedia untuk Android dan iOS. Anda dapat mengunduhnya di Google Play Store dan Apple App Store dengan mencari nama aplikasi kami.",
    "Untuk mendaftar akun baru, buka aplikasi, pilih 'Daftar', masukkan informasi yang diminta, lalu verifikasi melalui email atau nomor telepon.",
    "Untuk menghapus akun Anda, masuk ke pengaturan akun, pilih opsi 'Hapus Akun', dan ikuti instruksinya. Jika butuh bantuan lebih lanjut, hubungi customer support.",
    "Untuk menghapus akun Anda, masuk ke pengaturan akun, pilih opsi 'Hapus Akun', dan ikuti instruksinya. Jika butuh bantuan lebih lanjut, hubungi customer support.",
    "Untuk mendaftar akun baru, buka aplikasi, pilih 'Daftar', masukkan informasi yang diminta, lalu verifikasi melalui email atau nomor telepon.",
]

# Transformasi data uji
test_X = vectorizer.transform(test_questions)

# Prediksi hasil
predictions = model.predict(test_X)

# Evaluasi model
accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions, average='weighted')
recall = recall_score(test_labels, predictions, average='weighted')
f1 = f1_score(test_labels, predictions, average='weighted')
conf_matrix = confusion_matrix(test_labels, predictions)

# Cetak hasil evaluasi
print("Akurasi:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)

print("\nPertanyaan yang terjawab salah:")
for i in range(len(test_questions)):
    if test_labels[i] != predictions[i]:  # Jika prediksi salah
        print(f"- Pertanyaan: {test_questions[i]}")
        print(f"  Jawaban yang benar: {test_labels[i]}")
        print(f"  Jawaban yang diprediksi: {predictions[i]}\n")