
# Heart Disease EDA
# ==================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Membaca Dataset
df = pd.read_csv("heart_disease_uci.csv")

# 5 Data Teratas
print("🧠 5 Data Teratas:")
print(df.head())

# Info Dataset
print("\n📊 Info Dataset:")
print(df.info())

# Statistik Deskriptif
print("\n📈 Statistik Deskriptif:")
print(df.describe())

# Cek Missing Values
print("\n🔍 Cek Missing Values:")
print(df.isnull().sum())

# 2. Menangani Missing Values

# Mengisi missing values numerik dengan median
num_cols_miss = ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
for col in num_cols_miss:
    df[col] = df[col].fillna(df[col].median())

# Mengisi missing values kategorikal dengan modus
cat_cols_miss = ['fbs', 'restecg', 'exang', 'slope', 'thal']
for col in cat_cols_miss:
    df[col] = df[col].fillna(df[col].mode()[0])

# 3. Ubah Target Jadi Biner
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# 4. Visualisasi Data
# Distribusi kolom 'num'
sns.countplot(data=df, x='num', palette='pastel', hue='num', legend=False)
plt.title("Distribusi Penyakit Jantung (0 = Tidak, 1 = Ya)")
plt.xlabel("Penyakit Jantung")
plt.ylabel("Jumlah")
plt.show()

# Distribusi usia pasien
plt.figure(figsize=(8,6))
sns.histplot(df['age'], kde=True, color='blue')
plt.title("Distribusi Usia Pasien")
plt.xlabel("Usia")
plt.ylabel("Frekuensi")
plt.show()

# Heatmap korelasi antar fitur numerik
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Heatmap Korelasi Antar Fitur")
plt.show()

# 5. Visualisasi per Kategori
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=col, hue='num', palette='Set2')
    plt.title(f'Distribusi {col} berdasarkan Penyakit Jantung')
    plt.xlabel(col)
    plt.ylabel('Jumlah')
    plt.legend(title='Penyakit Jantung')
    plt.show()

# 6. Analisis Lanjutan EDA

# Boxplot: Usia berdasarkan penyakit jantung
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x='num', y='age', palette='Set2')
plt.title("Perbandingan Usia Pasien Berdasarkan Penyakit Jantung")
plt.xlabel("Penyakit Jantung (0 = Tidak, 1 = Ya)")
plt.ylabel("Usia")
plt.show()

# Scatterplot: Usia vs Tekanan Darah
sns.lmplot(data=df, x='age', y='trestbps', hue='num', palette='Set1')
plt.title("Hubungan Usia & Tekanan Darah berdasarkan Penyakit Jantung")
plt.show()

# Scatterplot: Usia vs Kolesterol
sns.lmplot(data=df, x='age', y='chol', hue='num', palette='Set2')
plt.title("Hubungan Usia & Kolesterol berdasarkan Penyakit Jantung")
plt.show()

# Jenis kelamin vs penyakit jantung
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='sex', hue='num', palette='cool')
plt.title("Distribusi Jenis Kelamin & Penyakit Jantung")
plt.xlabel("Jenis Kelamin")
plt.ylabel("Jumlah")
plt.legend(title="Penyakit Jantung")
plt.show()

# Crosstab: tipe nyeri dada vs penyakit jantung
print("\n🧾 Tabel Crosstab - Tipe Nyeri Dada (cp) vs Penyakit Jantung:")
print(pd.crosstab(df['cp'], df['num'], normalize='index') * 100)

# Crosstab: thalassemia vs penyakit jantung
print("\n🧾 Tabel Crosstab - Thalassemia (thal) vs Penyakit Jantung:")
print(pd.crosstab(df['thal'], df['num'], normalize='index') * 100)
