# Capstone Project — Prediksi Status Stunting Anak (Data Mining)

Proyek ini memprediksi **status stunting** anak di Kabupaten Temanggung menggunakan
model machine learning (XGBoost & Random Forest), berdasarkan fitur usia, jenis
kelamin, tinggi badan, dan tahun lahir.

## 📁 Struktur Repository

```
capstone-project-data-mining/
│
├── data/
│   ├── raw/               # Data mentah (data_anak_stunting_temanggung.csv)
│   ├── processed/         # Data hasil preprocessing (train/test split)
│   └── external/          # Data referensi eksternal (jika ada)
├── notebooks/
│   ├── 01_eda.ipynb              # EDA dan preprocessing
│   ├── 02_modeling.ipynb         # Pemodelan dan evaluasi
│   └── 03_interpretation.ipynb   # Interpretasi model
├── src/
│   ├── data_preprocessing.py   # Script preprocessing
│   ├── train_model.py          # Script training
│   ├── evaluate_model.py       # Script evaluasi
│   └── utils.py                # Fungsi utilitas
├── models/
│   ├── best_model.pkl          # Model terbaik (dibuat setelah training)
│   └── preprocessing.pkl       # Pipeline preprocessing
├── app/
│   ├── app.py                  # Aplikasi Streamlit utama
│   ├── pages/                  # Halaman tambahan Streamlit
│   └── assets/                 # Gambar, CSS, dll.
├── reports/
│   ├── final_report.pdf        # Laporan akhir (opsional)
│   └── presentation.pptx       # Slide presentasi (opsional)
├── requirements.txt
├── README.md
└── .gitignore
```

## 🔄 Alur Kerja (Pipeline)

1. **EDA & Preprocessing** (`notebooks/01_eda.ipynb` / `src/data_preprocessing.py`)
   - Load data mentah dari `data/raw/`
   - Cek info, korelasi, deskripsi statistik, duplikat, dan missing value
   - Feature engineering: `tanggal_lahir` → `tahun_lahir`, encoding `jenis_kelamin`
   - Penanganan outlier dengan metode IQR (clipping)
   - Binning fitur numerik untuk keperluan visualisasi
   - Normalisasi (`MinMaxScaler`) dan split data train/test

2. **Pemodelan** (`notebooks/02_modeling.ipynb` / `src/train_model.py`)
   - **XGBoost** (`n_estimators=200`, `max_depth=4`, dll.)
   - **Random Forest** (parameter yang sudah dioptimalkan manual)
   - **Random Forest + GridSearchCV** (tuning hyperparameter otomatis)
   - Evaluasi: Accuracy, F1 Score, Precision, Recall, Confusion Matrix

3. **Interpretasi** (`notebooks/03_interpretation.ipynb` / `src/evaluate_model.py`)
   - Feature importance (fitur paling berpengaruh terhadap status stunting)
   - Analisis overfitting (perbandingan akurasi training vs testing)

4. **Deployment** (`app/app.py`)
   - Aplikasi Streamlit sederhana untuk memprediksi status stunting dari input manual

## 🚀 Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Siapkan data mentah
Letakkan file `data_anak_stunting_temanggung.csv` di dalam folder `data/raw/`.

### 3. Jalankan pipeline dari script (`src/`)
```bash
cd src
python data_preprocessing.py   # cleaning, feature engineering, split data
python train_model.py          # training + perbandingan model + simpan model terbaik
python evaluate_model.py       # evaluasi lengkap + simpan grafik ke reports/
```

Atau jalankan langkah yang sama secara interaktif melalui notebook di folder `notebooks/`.

### 4. Jalankan aplikasi Streamlit
```bash
streamlit run app/app.py
```

## 📊 Ringkasan Model

Empat model dilatih dan dibandingkan pada dataset asli (2200 baris, 3 kelas `status_stunting`:
Normal, Stunting Ringan, Stunting Berat):

| Model                          | Accuracy | F1 (weighted) | Precision | Recall |
|---------------------------------|----------|---------------|-----------|--------|
| XGBoost                         | 77.95%   | 68.78%        | 61.54%    | 77.95% |
| Random Forest (Optimized)       | 69.32%   | 68.10%        | 67.14%    | 69.32% |
| Random Forest (Grid Search)     | 75.91%   | 71.50%        | 69.23%    | 75.91% |
| Random Forest + SMOTEENN        | 38.41%   | 45.80%        | 66.75%    | 38.41% |

Model dengan F1 Score (weighted) tertinggi otomatis dipilih & disimpan sebagai
`models/best_model.pkl` oleh `src/train_model.py` -- bukan Accuracy tertinggi. Saat ini
model terbaik adalah Random Forest (Grid Search, tuned dengan class_weight='balanced').
Alasan memilih F1 dijelaskan di catatan di bawah.

### Catatan penting: ketidakseimbangan kelas (class imbalance)

Dataset ini sangat imbalanced: Normal (1727 baris) jauh lebih banyak dari Stunting
Ringan (359) dan Stunting Berat (114). Konsekuensinya:
- Model yang dinilai hanya dari Accuracy bisa "curang" dengan selalu menebak Normal
  dan tetap terlihat baik, padahal gagal mendeteksi kasus stunting yang sebenarnya justru
  paling penting untuk ditangani.
- Karena itu src/train_model.py:
  - Memakai class_weight='balanced' pada Random Forest,
  - Meng-tuning GridSearchCV dengan scoring='f1_weighted' (bukan accuracy),
  - Menambahkan kandidat model dengan resampling SMOTEENN (library ini sudah diimpor di
    notebook awal tapi belum pernah dipakai -- sekarang benar-benar diterapkan).
- Model terbaik saat ini masih belum sempurna dalam mendeteksi kelas minoritas -- recall
  untuk Stunting Ringan/Berat di bawah 15% (lihat reports/confusion_matrix_Best_Model.png).
  Ini indikasi kuat kalau performa lebih baik butuh salah satu (atau kombinasi) dari:
  - Lebih banyak data anak dengan status stunting (khususnya Stunting Berat, cuma 114 baris),
  - Fitur tambahan (misal riwayat gizi, berat lahir, ASI eksklusif, dsb.),
  - Threshold/tuning tambahan atau algoritma lain (misal cost-sensitive learning).
- Ada juga indikasi overfitting ringan: akurasi training 96.4% vs testing 75.9%
  (selisih sekitar 20%) -- lihat output src/evaluate_model.py.

Untuk laporan/skripsi, poin-poin ini sebaiknya dituliskan sebagai keterbatasan (limitation)
dan saran penelitian lanjutan, bukan disembunyikan.

## 🧠 Catatan

- Kolom `skor_z_haz` dan `tanggal_lahir` dikeluarkan dari fitur (X) karena `skor_z_haz`
  berkorelasi langsung dengan target dan `tanggal_lahir` sudah direpresentasikan oleh `tahun_lahir`.
- Kolom `id`, `id_anak`, `jenis_data`, dan `tanggal_pengukuran` (ada di dataset asli tapi tidak
  dibahas di notebook awal) juga dikeluarkan karena merupakan identifier unik/metadata dengan
  kardinalitas sangat tinggi, bukan fitur klinis yang relevan untuk prediksi.
- Fitur numerik dinormalisasi dengan `MinMaxScaler`, fitur kategorikal (`kecamatan`, `desa`,
  dll.) di-encode dengan `OneHotEncoder`, keduanya digabung dalam satu `ColumnTransformer`
  agar konsisten antara data training dan testing (mencegah data leakage).
