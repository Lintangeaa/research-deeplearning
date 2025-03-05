import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Data FAQ dari JSON
faq_list = [
    {
        "pertanyaan": "Bagaimana cara reset password?",
        "keyword": [
            "Bagaimana cara reset password?",
            "Gimana ganti password akun saya?",
            "Saya lupa password, bagaimana menggantinya?",
            "Cara ubah password akun?",
            "Dimana menu reset password?",
            "Bagaimana jika saya lupa password akun saya?",
            "Reset password akun gimana ya?",
            "Bisakah saya mengatur ulang password saya?"
        ],
        "jawaban": "Untuk mereset password, klik tombol 'Lupa Password' pada halaman login, masukkan email Anda, lalu ikuti instruksi yang dikirimkan ke email Anda."
    },
    {
        "pertanyaan": "Bagaimana cara menghubungi customer support?",
        "keyword": [
            "Bagaimana cara menghubungi customer support?",
            "Gimana kontak admin?",
            "Dimana saya bisa meminta bantuan?",
            "Cara menghubungi tim support?",
            "Ada nomor telepon customer service?",
            "Saya butuh bantuan, hubungi siapa?",
            "Dimana bisa mengajukan keluhan?",
            "Siapa yang bisa saya hubungi untuk masalah akun?"
        ],
        "jawaban": "Anda dapat menghubungi customer support melalui email support@example.com atau menggunakan fitur live chat di dalam aplikasi."
    },
    {
        "pertanyaan": "Apakah ada versi mobile untuk aplikasi ini?",
        "keyword": [
            "Apakah ada versi mobile untuk aplikasi ini?",
            "Bisa dipakai di Android atau iOS?",
            "Apakah ada aplikasi untuk HP?",
            "Apakah aplikasi ini tersedia di Play Store?",
            "Dimana saya bisa download aplikasi mobile ini?",
            "Aplikasi ini bisa diinstal di iPhone?",
            "Ada aplikasi khusus untuk ponsel?",
            "Apakah ada aplikasi untuk tablet?"
        ],
        "jawaban": "Ya, aplikasi ini tersedia untuk Android dan iOS. Anda dapat mengunduhnya di Google Play Store dan Apple App Store dengan mencari nama aplikasi kami."
    },
    {
        "pertanyaan": "Bagaimana cara mendaftar akun baru?",
        "keyword": [
            "Bagaimana cara mendaftar akun baru?",
            "Cara buat akun di aplikasi ini?",
            "Gimana cara daftar?",
            "Dimana bisa registrasi akun?",
            "Saya ingin buat akun, bagaimana caranya?",
            "Langkah-langkah membuat akun baru?",
            "Cara sign up di aplikasi ini?",
            "Bagaimana cara membuat akun pengguna?",
            "Kalo untuk registrasinya itu dimana?"
        ],
        "jawaban": "Untuk mendaftar akun baru, buka aplikasi, pilih 'Daftar', masukkan informasi yang diminta, lalu verifikasi melalui email atau nomor telepon."
    },
    {
        "pertanyaan": "Bagaimana cara menghapus akun saya?",
        "keyword": [
            "Bagaimana cara menghapus akun saya?",
            "Cara delete akun?",
            "Dimana opsi hapus akun?",
            "Bagaimana jika saya ingin menonaktifkan akun?",
            "Bisa gak akun saya dihapus?",
            "Saya mau menghapus akun ini, gimana caranya?",
            "Apakah ada cara untuk menutup akun?",
            "Bisakah saya membatalkan akun saya?"
        ],
        "jawaban": "Untuk menghapus akun Anda, masuk ke pengaturan akun, pilih opsi 'Hapus Akun', dan ikuti instruksinya. Jika butuh bantuan lebih lanjut, hubungi customer support."
    }
]


# Buat dataset
texts = []
labels = []

for faq in faq_list:
    for keyword in faq["keyword"]:
        texts.append(keyword)
        labels.append(faq["jawaban"])

# Ubah teks menjadi fitur numerik dengan TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

# Latih model Logistic Regression
model = LogisticRegression()
model.fit(X, y)

# Simpan model dan vectorizer ke file
joblib.dump(model, "../../models/faq/logistic_faq_model.pkl")
joblib.dump(vectorizer, "../../models/faq/vectorizer.pkl")

print("✅ Model telah dilatih dan disimpan!")
