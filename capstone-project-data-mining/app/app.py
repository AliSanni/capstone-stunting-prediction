"""
app.py
Aplikasi Streamlit untuk memprediksi status stunting anak
berdasarkan model machine learning terbaik (models/best_model.pkl).

Jalankan dengan:
    streamlit run app/app.py

PENTING - rentang input dibatasi sesuai data training:
Model ini dilatih HANYA pada data anak usia 0-60 bulan (balita) dari 16
kecamatan tertentu di Kabupaten Temanggung. Memasukkan usia di luar rentang
itu (mis. usia orang dewasa) atau kecamatan/desa yang tidak ada di data
training akan membuat prediksi tidak bisa dipercaya (model mengekstrapolasi
di luar apa yang pernah ia pelajari, atau kehilangan info lokasi karena
OneHotEncoder mengabaikan kategori yang tidak dikenal).
"""

import os
import json
import datetime
import joblib
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(APP_DIR, "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
KECAMATAN_DESA_PATH = os.path.join(APP_DIR, "assets", "kecamatan_desa.json")

# Rentang data training (lihat data/raw/data_anak_stunting_temanggung.csv):
# usia_bulan: 0-60, tinggi_badan_cm: 45.3-155.9, tahun lahir: 2021-2025
USIA_BULAN_MIN, USIA_BULAN_MAX = 0, 60
TINGGI_MIN, TINGGI_MAX = 45.0, 130.0
TAHUN_LAHIR_MIN = datetime.date.today().year - 6  # anak balita, beri sedikit buffer
TAHUN_LAHIR_MAX = datetime.date.today().year


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    label_encoder = None
    if os.path.exists(LABEL_ENCODER_PATH):
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, label_encoder


@st.cache_data
def load_kecamatan_desa():
    with open(KECAMATAN_DESA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    st.set_page_config(page_title="Prediksi Status Stunting Anak", page_icon="👶", layout="centered")

    st.title("👶 Prediksi Status Stunting Anak")
    st.write(
        "Aplikasi ini memprediksi status stunting anak **balita (0-60 bulan)** di "
        "Kabupaten Temanggung berdasarkan usia, jenis kelamin, tinggi badan, tahun "
        "lahir, kecamatan, dan desa."
    )
    st.info(
        "Model hanya dilatih pada data anak usia 0-60 bulan dari kecamatan/desa "
        "tertentu. Semua pilihan di bawah sudah dibatasi agar sesuai data training, "
        "supaya hasil prediksi tidak menyesatkan.",
        icon="ℹ️",
    )

    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model belum tersedia. Jalankan `python src/train_model.py` terlebih dahulu "
            "untuk melatih dan menyimpan model ke `models/best_model.pkl`."
        )
        return

    model, label_encoder = load_model()
    kecamatan_desa = load_kecamatan_desa()

    st.subheader("Masukkan Data Anak")
    col1, col2 = st.columns(2)

    with col1:
        usia_bulan = st.number_input(
            "Usia (bulan)",
            min_value=USIA_BULAN_MIN,
            max_value=USIA_BULAN_MAX,
            value=24,
            help=f"Model hanya valid untuk anak usia {USIA_BULAN_MIN}-{USIA_BULAN_MAX} bulan (balita).",
        )
        jenis_kelamin_label = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        tahun_lahir = st.number_input(
            "Tahun Lahir",
            min_value=TAHUN_LAHIR_MIN,
            max_value=TAHUN_LAHIR_MAX,
            value=TAHUN_LAHIR_MAX - 2,
        )

    with col2:
        tinggi_badan_cm = st.number_input(
            "Tinggi Badan (cm)",
            min_value=TINGGI_MIN,
            max_value=TINGGI_MAX,
            value=80.0,
            help=f"Rentang data training: {TINGGI_MIN}-{TINGGI_MAX} cm.",
        )
        kecamatan = st.selectbox("Kecamatan", sorted(kecamatan_desa.keys()))
        desa = st.selectbox("Desa", sorted(kecamatan_desa[kecamatan]))

    jenis_kelamin = 1 if jenis_kelamin_label == "Laki-laki" else 0

    if st.button("Prediksi", type="primary"):
        input_df = pd.DataFrame([{
            "usia_bulan": usia_bulan,
            "jenis_kelamin": jenis_kelamin,
            "tinggi_badan_cm": tinggi_badan_cm,
            "tahun_lahir": tahun_lahir,
            "kecamatan": kecamatan,
            "desa": desa,
        }])

        try:
            prediction = model.predict(input_df)
            if label_encoder is not None:
                prediction = label_encoder.inverse_transform(prediction)
            st.success(f"Hasil Prediksi: **{prediction[0]}**")

            # Tampilkan probabilitas per kelas jika tersedia, agar pengguna tahu
            # seberapa yakin model (bukan cuma label tunggal).
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0]
                classes = (
                    label_encoder.inverse_transform(model.classes_)
                    if label_encoder is not None
                    else model.classes_
                )
                proba_df = pd.DataFrame({"Status": classes, "Probabilitas": proba})
                proba_df = proba_df.sort_values("Probabilitas", ascending=False)
                st.write("Detail probabilitas per kelas:")
                st.dataframe(proba_df.set_index("Status"), use_container_width=True)
        except Exception as e:
            st.error(
                "Terjadi kesalahan saat prediksi. Pastikan kolom input sesuai dengan fitur "
                f"yang dipakai model saat training.\n\nDetail error: {e}"
            )

    st.markdown("---")
    st.caption(
        "Model: Random Forest (tuned via GridSearchCV, class_weight='balanced'). "
        "Perhatikan bahwa recall untuk kelas 'Stunting Ringan'/'Stunting Berat' masih "
        "rendah pada data uji (lihat reports/confusion_matrix_Best_Model.png) karena "
        "data training didominasi kelas 'Normal'. Lihat notebooks/02_modeling.ipynb "
        "dan README.md untuk detail keterbatasan model."
    )


if __name__ == "__main__":
    main()
