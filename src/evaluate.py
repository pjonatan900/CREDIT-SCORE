from __future__ import annotations

"""evaluate.py

Evalúa el modelo entrenado (`models/best_model.pkl`) usando el conjunto de
validación `data/processed/credit_val.csv`.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

TARGET_COL = "Credit_Score"


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_validation_data() -> tuple[pd.DataFrame, pd.Series]:
    root = get_project_root()
    val_path = root / "data" / "processed" / "credit_val.csv"
    if not val_path.exists():
        raise FileNotFoundError(
            f"No se encontró {val_path}. Ejecuta primero: python train.py"
        )
    df = pd.read_csv(val_path)
    if TARGET_COL not in df.columns:
        raise ValueError("El archivo de validación no tiene la columna objetivo.")

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y


def load_model():
    root = get_project_root()
    model_path = root / "models" / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {model_path}. Entrena primero el modelo."
        )
    return joblib.load(model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluar modelo Credit Score (ANN)")
    _ = parser.parse_args()

    print("📥 Cargando datos de validación...")
    X_val, y_val = load_validation_data()

    print("📦 Cargando modelo...")
    model = load_model()

    print("📊 Calculando métricas...")
    preds = model.predict(X_val)

    acc = accuracy_score(y_val, preds)
    cm = confusion_matrix(y_val, preds)
    report = classification_report(y_val, preds, digits=4)

    print(f"Accuracy: {acc:.4f}")
    print("Matriz de confusión:")
    print(cm)
    print("Classification report:")
    print(report)


if __name__ == "__main__":
    main()
