# -*- coding: utf-8 -*-
"""
Skrip Analisis Data Siswa

Deskripsi:
Skrip ini melakukan analisis lengkap pada dataset siswa, mencakup:
1.  Pemuatan dan penggabungan data dari dua mata pelajaran (Matematika dan B. Portugis).
2.  Exploratory Data Analysis (EDA) dengan visualisasi.
3.  Analisis Regresi Linear untuk memprediksi nilai akhir.
4.  Clustering K-Means untuk segmentasi siswa.
5.  Klasifikasi Decision Tree untuk memprediksi kelulusan.

Dependensi: pandas, numpy, matplotlib, seaborn, scikit-learn
"""

# 1. SETUP DAN PEMUATAN DATA
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import warnings

# Mengabaikan peringatan untuk kejelasan output
warnings.filterwarnings('ignore')

print("Memulai proses analisis data siswa...")

# Memuat dataset dari file CSV
try:
    d1 = pd.read_csv('student-mat.csv', sep=';')
    d2 = pd.read_csv('student-por.csv', sep=';')
    print("Dataset berhasil dimuat.")
except FileNotFoundError:
    print("Error: Pastikan file 'student-mat.csv' dan 'student-por.csv' berada di direktori yang sama.")
    exit()

# Menggabungkan dataset berdasarkan atribut umum
merge_keys = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"
]
d3 = pd.merge(d1, d2, on=merge_keys, suffixes=('_math', '_por'))
print(f"Data berhasil digabungkan. Jumlah total siswa: {len(d3)}")


# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
print("\n--- Memulai Exploratory Data Analysis (EDA) ---")

# Statistik deskriptif untuk kolom numerik
print("Statistik Deskriptif untuk Data Gabungan:")
print(d3.describe())

# Mengatur style plot
sns.set_style("whitegrid")

# Visualisasi 1: Distribusi Usia dan Jenis Kelamin
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(d3['age'], bins=8, kde=True, color="skyblue")
plt.title('Distribusi Usia Siswa')
plt.xlabel('Usia')
plt.ylabel('Jumlah')

plt.subplot(1, 2, 2)
sns.countplot(x='sex', data=d3, palette='pastel')
plt.title('Distribusi Jenis Kelamin')
plt.xlabel('Jenis Kelamin')
plt.ylabel('Jumlah')
plt.tight_layout()
plt.savefig('distribusi_usia_gender.png')
plt.close()

# Visualisasi 2: Distribusi Waktu Belajar dan Absensi (B. Portugis)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='studytime_por', data=d3, palette='viridis')
plt.title('Distribusi Waktu Belajar Mingguan (B. Portugis)')
plt.xlabel('Waktu Belajar (Kategori)')
plt.ylabel('Jumlah')

plt.subplot(1, 2, 2)
sns.histplot(d3['absences_por'], bins=20, kde=True, color='red')
plt.title('Distribusi Jumlah Absensi (B. Portugis)')
plt.xlabel('Jumlah Absensi')
plt.ylabel('Jumlah')
plt.xlim(0, 40) # Batasi untuk visualisasi yang lebih baik
plt.tight_layout()
plt.savefig('distribusi_belajar_absen.png')
plt.close()

# Visualisasi 3: Heatmap Korelasi
numeric_cols_por = [
    'age', 'Medu', 'Fedu', 'traveltime_por', 'studytime_por', 'failures_por',
    'famrel_por', 'freetime_por', 'goout_por', 'Dalc_por', 'Walc_por',
    'health_por', 'absences_por', 'G1_por', 'G2_por', 'G3_por'
]
plt.figure(figsize=(14, 10))
correlation_matrix = d3[numeric_cols_por].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 8})
plt.title('Heatmap Korelasi Variabel Numerik (B. Portugis)', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('heatmap_korelasi.png')
plt.close()

print("Visualisasi EDA telah disimpan: 'distribusi_usia_gender.png', 'distribusi_belajar_absen.png', 'heatmap_korelasi.png'")


# 3. ANALISIS REGRESI LINEAR
# ==============================================================================
print("\n--- Memulai Analisis Regresi Linear ---")

# Variabel: studytime_por, failures_por -> G3_por
X_reg = d3[['studytime_por', 'failures_por']]
y_reg = d3['G3_por']

# Membagi data latih dan uji
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Melatih model
model_reg = LinearRegression()
model_reg.fit(X_train_reg, y_train_reg)

# Evaluasi
y_pred_reg = model_reg.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"Koefisien Regresi: {model_reg.coef_}")
print(f"Intercept: {model_reg.intercept_}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")

# Visualisasi hasil regresi
plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.7, edgecolors='k')
plt.plot([y_reg.min(), y_reg.max()], [y_reg.min(), y_reg.max()], '--', lw=2, color='red')
plt.xlabel("Nilai Aktual (G3 por)")
plt.ylabel("Nilai Prediksi (G3 por)")
plt.title("Regresi Linear: Nilai Aktual vs. Prediksi")
plt.savefig('regresi_aktual_vs_prediksi.png')
plt.close()
print("Visualisasi Regresi telah disimpan: 'regresi_aktual_vs_prediksi.png'")


# 4. CLUSTERING (SEGMENTASI SISWA)
# ==============================================================================
print("\n--- Memulai Clustering (Segmentasi Siswa) ---")

# Fitur: studytime_por, absences_por
features_cluster = d3[['studytime_por', 'absences_por']]

# Standarisasi fitur
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_cluster)

# Menentukan jumlah cluster optimal dengan Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method untuk Menentukan Jumlah Cluster Optimal')
plt.xlabel('Jumlah Clusters')
plt.ylabel('WCSS')
plt.savefig('elbow_method.png')
plt.close()

# Melakukan clustering dengan k=4
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
d3['cluster'] = kmeans.fit_predict(features_scaled)

# Visualisasi hasil clustering
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=d3['studytime_por'],
    y=d3['absences_por'],
    hue=d3['cluster'],
    palette=sns.color_palette("hsv", n_colors=optimal_clusters),
    s=100,
    alpha=0.8,
    edgecolor='k'
)
plt.title('Segmentasi Siswa berdasarkan Waktu Belajar dan Absensi')
plt.xlabel('Waktu Belajar Mingguan (Kategori)')
plt.ylabel('Jumlah Absensi')
plt.legend(title='Cluster')
plt.savefig('segmentasi_siswa.png')
plt.close()
print("Visualisasi Clustering telah disimpan: 'elbow_method.png', 'segmentasi_siswa.png'")


# 5. KLASIFIKASI
# ==============================================================================
print("\n--- Memulai Klasifikasi (Prediksi Kelulusan) ---")

# Membuat variabel target biner: lulus jika G3_por >= 10
d3_class = d3.copy()
d3_class['passed'] = np.where(d3_class['G3_por'] >= 10, 1, 0)

# Mengubah variabel kategorikal 'yes'/'no' menjadi numerik
d3_class['higher_por'] = d3_class['higher_por'].apply(lambda x: 1 if x == 'yes' else 0)
d3_class['schoolsup_por'] = d3_class['schoolsup_por'].apply(lambda x: 1 if x == 'yes' else 0)

# Fitur: failures_por, higher_por, schoolsup_por
X_class = d3_class[['failures_por', 'higher_por', 'schoolsup_por']]
y_class = d3_class['passed']

# Membagi data latih dan uji
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.25, random_state=42, stratify=y_class)

# Melatih model Decision Tree
model_class = DecisionTreeClassifier(max_depth=3, random_state=42)
model_class.fit(X_train_class, y_train_class)

# Evaluasi
y_pred_class = model_class.predict(X_test_class)
print("Laporan Klasifikasi:")
print(classification_report(y_test_class, y_pred_class))

# Visualisasi Confusion Matrix
cm = confusion_matrix(y_test_class, y_pred_class)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Tidak Lulus', 'Lulus'], yticklabels=['Tidak Lulus', 'Lulus'])
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Visualisasi Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(
    model_class,
    filled=True,
    feature_names=X_class.columns,
    class_names=['Tidak Lulus', 'Lulus'],
    rounded=True,
    fontsize=10
)
plt.title("Pohon Keputusan untuk Prediksi Kelulusan", fontsize=16)
plt.savefig('decision_tree.png')
plt.close()
print("Visualisasi Klasifikasi telah disimpan: 'confusion_matrix.png', 'decision_tree.png'")

print("\nAnalisis selesai. Semua output dan visualisasi telah berhasil dibuat.")