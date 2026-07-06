"""
Credit Scoring Model using Machine Learning
CodeAlpha Machine Learning Internship - Task 1

This script trains classification models to predict creditworthiness
using financial history data.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


DATASET_PATH = "dataset.csv"
TARGET_COLUMN = "Credit_Score"


def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Keep dataset.csv in the same folder.")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and handle missing values."""
    df = df.drop_duplicates().copy()

    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].median())

    return df


def encode_data(df: pd.DataFrame):
    """Encode categorical columns using LabelEncoder."""
    label_encoders = {}
    data = df.copy()

    for column in data.columns:
        if data[column].dtype == "object":
            encoder = LabelEncoder()
            data[column] = encoder.fit_transform(data[column])
            label_encoders[column] = encoder

    return data, label_encoders


def plot_accuracy(results_df: pd.DataFrame):
    """Save model accuracy comparison graph."""
    os.makedirs("images", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Model"], results_df["Accuracy"])
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("images/accuracy.png", dpi=200)
    plt.close()


def plot_confusion_matrix(y_test, y_pred, classes):
    """Save confusion matrix graph."""
    os.makedirs("images", exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("images/confusion_matrix.png", dpi=200)
    plt.close()


def plot_feature_importance(model, feature_names):
    """Save feature importance graph."""
    os.makedirs("images", exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        return

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("images/feature_importance.png", dpi=200)
    plt.close()


def plot_roc_curve(model, model_name, X_test, X_test_scaled, y_test, classes):
    """Save multi-class ROC curve graph."""
    os.makedirs("images", exist_ok=True)

    try:
        if model_name == "Logistic Regression":
            y_score = model.predict_proba(X_test_scaled)
        else:
            y_score = model.predict_proba(X_test)

        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        auc_score = roc_auc_score(y_test_bin, y_score, multi_class="ovr")

        plt.figure(figsize=(8, 6))
        for i, class_name in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            plt.plot(fpr, tpr, label=f"{class_name}")

        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"ROC Curve - {model_name} | AUC: {auc_score:.4f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig("images/roc_curve.png", dpi=200)
        plt.close()

        return auc_score
    except Exception as error:
        print("ROC curve could not be generated:", error)
        return None


def main():
    print("Loading dataset...")
    df = load_dataset(DATASET_PATH)

    print("Dataset shape:", df.shape)
    print("First five rows:")
    print(df.head())

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    print("\nCleaning data...")
    df = clean_data(df)

    print("\nEncoding categorical values...")
    data, label_encoders = encode_data(df)

    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = []

    print("\nTraining models...")
    for model_name, model in models.items():
        if model_name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        results.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1 Score": f1_score(y_test, y_pred, average="weighted"),
        })

    results_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(results_df)

    best_model_name = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]["Model"]
    best_model = models[best_model_name]

    if best_model_name == "Logistic Regression":
        y_pred_best = best_model.predict(X_test_scaled)
    else:
        y_pred_best = best_model.predict(X_test)

    print("\nBest Model:", best_model_name)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best))

    classes = label_encoders[TARGET_COLUMN].classes_ if TARGET_COLUMN in label_encoders else np.unique(y)

    plot_accuracy(results_df)
    plot_confusion_matrix(y_test, y_pred_best, classes)
    plot_feature_importance(best_model, X.columns)
    auc_score = plot_roc_curve(best_model, best_model_name, X_test, X_test_scaled, y_test, classes)

    if auc_score is not None:
        print("ROC-AUC Score:", auc_score)

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/credit_scoring_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(label_encoders, "models/label_encoders.pkl")

    print("\nFiles saved successfully.")
    print("Model: models/credit_scoring_model.pkl")
    print("Scaler: models/scaler.pkl")
    print("Encoders: models/label_encoders.pkl")
    print("Images saved in images/ folder.")


if __name__ == "__main__":
    main()
