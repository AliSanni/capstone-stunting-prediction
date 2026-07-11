"""
train_model.py
Melatih 3 kandidat model klasifikasi status stunting:
  1. XGBoost
  2. Random Forest (parameter manual/optimized)
  3. Random Forest + GridSearchCV (tuning)

Model dengan akurasi testing terbaik disimpan sebagai models/best_model.pkl.
"""

import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

from data_preprocessing import run_pipeline, build_preprocessor
from utils import metrics_dict

MODELS_DIR = "models"


def train_xgboost(x_train, y_train, x_test, y_test, preprocessor):
    """Melatih pipeline XGBoost dan mengembalikan pipeline + label encoder + metrik."""
    xgb_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )),
    ])

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    xgb_pipeline.fit(x_train, y_train_encoded)
    y_pred_encoded = xgb_pipeline.predict(x_test)

    metrics = metrics_dict(y_test_encoded, y_pred_encoded)
    print("=== XGBoost ===")
    print(metrics)

    return xgb_pipeline, label_encoder, metrics


def train_random_forest(x_train, y_train, x_test, y_test, preprocessor):
    """Melatih Random Forest dengan parameter yang sudah dioptimalkan manual."""
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    rf_pipeline.fit(x_train, y_train)
    y_pred = rf_pipeline.predict(x_test)

    metrics = metrics_dict(y_test, y_pred)
    print("=== Random Forest (Optimized) ===")
    print(metrics)

    return rf_pipeline, metrics


def train_random_forest_gridsearch(x_train, y_train, x_test, y_test, preprocessor):
    """Melatih Random Forest dengan tuning hyperparameter via GridSearchCV."""
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt"],
    }
    # class_weight='balanced' + scoring='f1_weighted' agar tuning tidak terjebak
    # memilih model yang hanya menebak kelas mayoritas ('Normal') pada data yang imbalanced.
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
    pipe_gs = Pipeline([("pre", preprocessor), ("model", rf_base)])

    gs = GridSearchCV(pipe_gs, param_grid=param_grid, cv=3, scoring="f1_weighted", n_jobs=-1)
    gs.fit(x_train, y_train)

    best_pipeline = gs.best_estimator_
    y_pred = best_pipeline.predict(x_test)

    metrics = metrics_dict(y_test, y_pred)
    print("=== Random Forest (GridSearch) ===")
    print("Best params:", gs.best_params_)
    print(metrics)

    return best_pipeline, metrics


def train_random_forest_smoteenn(x_train, y_train, x_test, y_test, preprocessor):
    """
    Melatih Random Forest dengan resampling SMOTEENN pada data training untuk
    mengatasi ketidakseimbangan kelas pada status_stunting (mayoritas 'Normal').
    SMOTEENN hanya diterapkan ke data training (bukan data testing) agar evaluasi tetap adil.
    """
    smoteenn_pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("resampler", SMOTEENN(random_state=42)),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    smoteenn_pipeline.fit(x_train, y_train)
    y_pred = smoteenn_pipeline.predict(x_test)

    metrics = metrics_dict(y_test, y_pred)
    print("=== Random Forest + SMOTEENN ===")
    print(metrics)

    return smoteenn_pipeline, metrics


def run_training():

    """Menjalankan preprocessing lalu melatih & membandingkan 3 model, menyimpan model terbaik."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    x_train, x_test, y_train, y_test, _ = run_pipeline()
    preprocessor = build_preprocessor(x_train)

    xgb_pipeline, label_encoder, xgb_metrics = train_xgboost(
        x_train, y_train, x_test, y_test, preprocessor
    )
    rf_pipeline, rf_metrics = train_random_forest(
        x_train, y_train, x_test, y_test, preprocessor
    )
    rf_gs_pipeline, rf_gs_metrics = train_random_forest_gridsearch(
        x_train, y_train, x_test, y_test, preprocessor
    )
    rf_smoteenn_pipeline, rf_smoteenn_metrics = train_random_forest_smoteenn(
        x_train, y_train, x_test, y_test, preprocessor
    )

    comparison_df = pd.DataFrame({
        "XGBoost": xgb_metrics,
        "Random Forest (Optimized)": rf_metrics,
        "Random Forest (Grid Search)": rf_gs_metrics,
        "Random Forest (SMOTEENN)": rf_smoteenn_metrics,
    })
    print("\n=== Perbandingan Metrik Semua Model ===")
    print(comparison_df)

    # Catatan: status_stunting sangat imbalanced (mayoritas 'Normal'), sehingga Accuracy
    # saja bisa menyesatkan (model bisa 'curang' dengan selalu menebak kelas mayoritas).
    # Model terbaik dipilih berdasarkan F1 Score (weighted), bukan Accuracy mentah.
    candidates = {
        "xgboost": (xgb_pipeline, xgb_metrics["F1 Score"]),
        "random_forest_optimized": (rf_pipeline, rf_metrics["F1 Score"]),
        "random_forest_gridsearch": (rf_gs_pipeline, rf_gs_metrics["F1 Score"]),
        "random_forest_smoteenn": (rf_smoteenn_pipeline, rf_smoteenn_metrics["F1 Score"]),
    }
    best_name, (best_pipeline, best_f1) = max(candidates.items(), key=lambda kv: kv[1][1])
    print(f"\nModel terbaik: {best_name} (F1 Score weighted: {best_f1}%)")

    joblib.dump(best_pipeline, os.path.join(MODELS_DIR, "best_model.pkl"))
    if best_name == "xgboost":
        joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))

    print(f"Model tersimpan di {os.path.join(MODELS_DIR, 'best_model.pkl')}")
    return best_name, best_pipeline, comparison_df


if __name__ == "__main__":
    run_training()
