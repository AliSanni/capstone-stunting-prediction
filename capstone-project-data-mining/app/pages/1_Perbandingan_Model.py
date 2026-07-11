"""
1_Perbandingan_Model.py
Halaman tambahan Streamlit untuk menampilkan gambar hasil evaluasi model
(confusion matrix & feature importance) yang dihasilkan oleh src/evaluate_model.py.
"""

import os
import streamlit as st

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports")

st.set_page_config(page_title="Perbandingan Model", page_icon="📊")
st.title("📊 Perbandingan & Evaluasi Model")

st.write(
    "Halaman ini menampilkan visualisasi hasil evaluasi model yang disimpan di folder "
    "`reports/` setelah menjalankan `python src/evaluate_model.py`."
)

if not os.path.isdir(REPORTS_DIR):
    st.info("Belum ada laporan. Jalankan `python src/evaluate_model.py` terlebih dahulu.")
else:
    image_files = [f for f in os.listdir(REPORTS_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not image_files:
        st.info("Belum ada gambar laporan di folder reports/.")
    for img in sorted(image_files):
        st.image(os.path.join(REPORTS_DIR, img), caption=img)
