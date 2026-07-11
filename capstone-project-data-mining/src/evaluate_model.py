"""
evaluate_model.py
Memuat model terbaik (models/best_model.pkl) dan data testing (data/processed/)
untuk menampilkan evaluasi lengkap: accuracy, F1, confusion matrix,
classification report, feature importance, dan analisis overfitting.
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from utils import metrics_dict, analyze_overfitting

MODELS_DIR = "models"
PROCESSED_DIR = os.path.join("data", "processed")
REPORTS_DIR = "reports"


def load_test_data():
    x_train = pd.read_csv(os.path.join(PROCESSED_DIR, "x_train.csv"))
    x_test = pd.read_csv(os.path.join(PROCESSED_DIR, "x_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).iloc[:, 0]
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).iloc[:, 0]
    return x_train, x_test, y_train, y_test


def load_model(model_name: str = "best_model.pkl"):
    model_path = os.path.join(MODELS_DIR, model_name)
    return joblib.load(model_path)


def plot_confusion_matrix(cm, title: str, save_path: str = None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "Count"})
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix disimpan di {save_path}")
    plt.show()


def evaluate(model, x_train, y_train, x_test, y_test, model_name: str = "Best Model"):
    """Menjalankan evaluasi lengkap sebuah pipeline model yang sudah dilatih."""
    y_pred = model.predict(x_test)

    metrics = metrics_dict(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== Evaluasi {model_name} ===")
    print(metrics)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(REPORTS_DIR, exist_ok=True)
    plot_confusion_matrix(
        cm,
        f"Confusion Matrix - {model_name}",
        save_path=os.path.join(REPORTS_DIR, f"confusion_matrix_{model_name.replace(' ', '_')}.png"),
    )

    overfit_result = analyze_overfitting(model_name, model, x_train, y_train, x_test, y_test)

    return {"metrics": metrics, "confusion_matrix": cm, "overfitting": overfit_result}


def plot_feature_importance(model, top_n: int = 10, save_path: str = None):
    """
    Menampilkan feature importance untuk model berbasis tree (RandomForest/XGBoost).
    Mendukung beberapa variasi nama step pipeline: 'classifier'/'model' untuk
    estimator, dan 'preprocessor'/'pre' untuk ColumnTransformer.
    """
    classifier_keys = ["classifier", "model"]
    preprocessor_keys = ["preprocessor", "pre"]

    classifier = next((model.named_steps[k] for k in classifier_keys if k in model.named_steps), None)
    preprocessor_step = next((model.named_steps[k] for k in preprocessor_keys if k in model.named_steps), None)

    if classifier is None or preprocessor_step is None or not hasattr(classifier, "feature_importances_"):
        print("Model tidak memiliki feature_importances_ atau nama step pipeline tidak dikenali.")
        return None

    importances = classifier.feature_importances_
    feature_names = preprocessor_step.get_feature_names_out()

    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(top_n), x="Importance", y="Feature")
    plt.title(f"Top {top_n} Fitur Terpenting")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Feature importance disimpan di {save_path}")
    plt.show()

    return importance_df


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_test_data()
    model = load_model()
    evaluate(model, x_train, y_train, x_test, y_test, model_name="Best Model")
    plot_feature_importance(
        model,
        save_path=os.path.join(REPORTS_DIR, "feature_importance.png"),
    )
