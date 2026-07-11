"""
utils.py
Fungsi-fungsi utilitas yang dipakai bersama oleh script preprocessing,
training, dan evaluasi model prediksi stunting.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def handle_outliers_iqr(df: pd.DataFrame, columns: list):
    """
    Mendeteksi dan menangani outlier pada kolom numerik menggunakan metode IQR.
    Outlier ditangani dengan clipping ke batas bawah/atas (bukan dihapus).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe input.
    columns : list
        Daftar nama kolom numerik yang akan dicek outlier-nya.

    Returns
    -------
    df_clean : pd.DataFrame
        Dataframe setelah outlier di-clip.
    summary : pd.DataFrame
        Ringkasan Q1, Q3, IQR, batas bawah/atas, dan jumlah outlier per kolom.
    """
    df_clean = df.copy()
    summary = {}

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        summary[col] = {
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outliers_count": len(outliers),
        }

        df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

    return df_clean, pd.DataFrame(summary).T


def bin_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat kolom kategori (binned) dari beberapa fitur numerik:
    usia_bulan, tinggi_badan_cm, dan skor_z_haz.
    Berguna untuk keperluan eksplorasi/visualisasi, bukan untuk fitur model.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe yang minimal punya kolom usia_bulan, tinggi_badan_cm, skor_z_haz.

    Returns
    -------
    pd.DataFrame
        Dataframe dengan tambahan kolom *_bin.
    """
    df_binned = df.copy()

    # 1. Binning usia_bulan -> kategori umur
    age_bins = [0, 12, 24, 36, 48, 60, 72]
    age_labels = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
    df_binned["usia_bulan_bin"] = pd.cut(
        df["usia_bulan"], bins=age_bins, labels=age_labels, right=False
    )

    # 2. Binning tinggi_badan_cm -> rentang tinggi badan
    tb_bins = [0, 70, 85, 100, df["tinggi_badan_cm"].max() + 1]
    tb_labels = ["Pendek", "Normal", "Tinggi", "Sangat Tinggi"]
    df_binned["tinggi_badan_cm_bin"] = pd.cut(
        df["tinggi_badan_cm"], bins=tb_bins, labels=tb_labels, right=False
    )

    # 3. Binning skor_z_haz -> kategori Z-score
    zscore_bins = [-np.inf, -3, -2, 0, np.inf]
    zscore_labels = ["Severely Stunted", "Stunted", "Normal", "Tall"]
    df_binned["skor_z_haz_bin"] = pd.cut(
        df["skor_z_haz"], bins=zscore_bins, labels=zscore_labels, right=False
    )

    return df_binned


def metrics_dict(y_true, y_pred) -> dict:
    """
    Menghitung ringkasan metrik evaluasi klasifikasi (accuracy, f1, precision, recall)
    dalam bentuk persen, dibulatkan 2 angka desimal.
    """
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "F1 Score": round(f1_score(y_true, y_pred, average="weighted") * 100, 2),
        "Precision": round(
            precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2
        ),
        "Recall": round(recall_score(y_true, y_pred, average="weighted") * 100, 2),
    }


def analyze_overfitting(model_name, model, X_train, y_train, X_test, y_test, threshold=5.0):
    """
    Membandingkan akurasi training vs testing untuk mendeteksi indikasi overfitting.

    Parameters
    ----------
    threshold : float
        Selisih akurasi (train - test) dalam persen yang dianggap indikasi overfitting.

    Returns
    -------
    dict berisi train_accuracy, test_accuracy, difference, dan indikasi overfitting.
    """
    train_accuracy = model.score(X_train, y_train) * 100
    test_accuracy = model.score(X_test, y_test) * 100
    difference = train_accuracy - test_accuracy
    is_overfit = difference > threshold

    print(f"\n=== Overfitting Analysis for {model_name} ===")
    print(f"  Training Accuracy: {train_accuracy:.2f}%")
    print(f"  Testing Accuracy: {test_accuracy:.2f}%")
    print(f"  Difference (Train - Test): {difference:.2f}%")
    print(f"  Indication: {'Potential Overfitting' if is_overfit else 'No significant overfitting'}")

    return {
        "model_name": model_name,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "difference": difference,
        "is_overfit": is_overfit,
    }
