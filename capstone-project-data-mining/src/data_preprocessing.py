"""
data_preprocessing.py
Script untuk memuat data mentah, melakukan cleaning, feature engineering,
normalisasi, dan menyimpan data yang sudah diproses + pipeline preprocessing.

Menjalankan file ini secara langsung akan:
1. Membaca data mentah dari data/raw/
2. Membersihkan duplikat & missing value
3. Menangani outlier (IQR clipping)
4. Feature engineering (tahun_lahir, encoding jenis_kelamin)
5. Split train/test
6. Membuat & menyimpan ColumnTransformer (preprocessing.pkl)
7. Menyimpan data hasil proses ke data/processed/
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from utils import handle_outliers_iqr

RAW_DATA_PATH = os.path.join("data", "raw", "data_anak_stunting_temanggung.csv")
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"

TARGET_COL = "status_stunting"
# Kolom yang dibuang dari fitur (X):
# - status_stunting  -> target
# - skor_z_haz       -> berkorelasi langsung/dipakai membentuk target, dibuang agar tidak leakage
# - tanggal_lahir     -> sudah direpresentasikan oleh tahun_lahir
# - id, id_anak       -> identifier unik, bukan fitur prediktif
# - jenis_data        -> flag sumber data ("Anak"/"data_tambahan"), bukan fitur klinis
# - tanggal_pengukuran-> tanggal pengukuran, nyaris seluruhnya unik (high-cardinality, tidak informatif)
DROP_COLS = [
    "status_stunting",
    "skor_z_haz",
    "tanggal_lahir",
    "id",
    "id_anak",
    "jenis_data",
    "tanggal_pengukuran",
]
OUTLIER_COLS = ["usia_bulan", "tinggi_badan_cm", "skor_z_haz"]


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Memuat data mentah dari CSV."""
    df = pd.read_csv(path)
    print(f"Data dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus duplikat & missing value, lalu menangani outlier (clipping IQR)."""
    df = df.copy()

    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    print(f"Menghapus {before - df.shape[0]} baris duplikat")

    before = df.shape[0]
    df.dropna(inplace=True)
    print(f"Menghapus {before - df.shape[0]} baris dengan missing value")

    df_clean, outlier_summary = handle_outliers_iqr(df, OUTLIER_COLS)
    print("Ringkasan outlier (IQR):")
    print(outlier_summary)

    return df_clean


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Mengubah tanggal_lahir menjadi tahun_lahir
    - Meng-encode jenis_kelamin menjadi biner (1 = Laki-laki, 0 = Perempuan)
    """
    df = df.copy()
    df["tanggal_lahir"] = pd.to_datetime(df["tanggal_lahir"], errors="coerce")
    df["tahun_lahir"] = df["tanggal_lahir"].dt.year
    df["jenis_kelamin"] = df["jenis_kelamin"].map({"Laki-laki": 1, "Perempuan": 0})
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Memisahkan fitur (X) & target (y), lalu membagi menjadi train/test."""
    x = df.drop(columns=DROP_COLS)
    y = df[TARGET_COL]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Data training: {x_train.shape[0]} baris | Data testing: {x_test.shape[0]} baris")
    return x_train, x_test, y_train, y_test


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    """Membangun ColumnTransformer: MinMaxScaler untuk numerik, OneHotEncoder untuk kategorikal."""
    numeric_features = x_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = x_train.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough",
    )
    print(f"Preprocessor dibuat -> numerik: {numeric_features}, kategorikal: {categorical_features}")
    return preprocessor


def run_pipeline():
    """Menjalankan seluruh alur preprocessing dari data mentah sampai siap dilatih."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_data()
    df = feature_engineering(df)
    df = clean_data(df)

    x_train, x_test, y_train, y_test = split_data(df)
    preprocessor = build_preprocessor(x_train)

    # simpan data hasil proses
    x_train.to_csv(os.path.join(PROCESSED_DIR, "x_train.csv"), index=False)
    x_test.to_csv(os.path.join(PROCESSED_DIR, "x_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

    # simpan preprocessing pipeline (belum di-fit, akan di-fit ulang di dalam model pipeline saat training)
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessing.pkl"))

    print("\nPreprocessing selesai. Data & pipeline tersimpan di data/processed/ dan models/.")
    return x_train, x_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    run_pipeline()
