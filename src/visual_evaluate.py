import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Muat data dari file JSON
with open('../data/intent_json.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Hitung jumlah masing-masing intent
intent_counts = df['intent'].value_counts()

# Pisahkan data menjadi fitur dan label
X = df['text']
y = df['intent']

# Muat model dan vectorizer dari file
with open('../models/rf/intent_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('../models/rf/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Transformasi data ke TF-IDF
X_tfidf = vectorizer.transform(X)

# Prediksi
y_pred = model.predict(X_tfidf)

# Evaluasi Akurasi dan Classification Report
eval_report = classification_report(y, y_pred, zero_division=1, output_dict=True)
accuracy = accuracy_score(y, y_pred)

# Confusion Matrix
cm = confusion_matrix(y, y_pred)

# Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_tfidf, y, cv=cv, scoring='f1_weighted')

# ==========================
# 1. Figure 1 - Plot Distribusi Data Intent
# ==========================
fig1 = plt.figure(figsize=(10, 5))
sns.barplot(x=intent_counts.index, y=intent_counts.values, hue=intent_counts.index, palette='viridis', legend=False)
plt.title('Distribusi Data per Intent')
plt.xlabel('Intent')
plt.ylabel('Jumlah Data')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ==========================
# 2. Figure 2 - Plot Confusion Matrix
# ==========================
fig2 = plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ==========================
# 3. Figure 3 - Classification Report Heatmap
# ==========================
df_report = pd.DataFrame(eval_report).transpose().iloc[:-3, :]
fig3 = plt.figure(figsize=(10, 5))
sns.heatmap(df_report.iloc[:, :-1], annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Classification Report Heatmap')
plt.show()

# ==========================
# 4. Menampilkan Cross Validation Scores dalam bentuk tabel
# ==========================
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

cv_table = pd.DataFrame({
    "Metric": ["F1 Weighted Mean", "F1 Weighted Std"],
    "Value": [cv_mean, cv_std]
})

print("\nCross Validation Scores:")
print(cv_table.to_string(index=False))  # Menampilkan dalam format tabel rapi di terminal

# ==========================
# 5. Plot Hasil Prediksi per Kalimat
# ==========================
result_df = pd.DataFrame({'Text': X, 'True Intent': y, 'Predicted Intent': y_pred})
result_df['Correct'] = result_df['True Intent'] == result_df['Predicted Intent']

fig5, ax = plt.subplots(figsize=(12, 6))
colors = result_df['Correct'].map({True: 'black', False: 'red'})
for i, row in result_df.iterrows():
    ax.text(0, i, f"{row['Text']} => {row['Predicted Intent']}", fontsize=10, color=colors[i])

ax.set_xlim([-1, 1])
ax.set_ylim([-1, len(result_df)])
ax.axis('off')
plt.title('Hasil Prediksi Intent (Merah = Salah)')
plt.show()
