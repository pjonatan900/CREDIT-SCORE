from __future__ import annotations

"""train.py

Entrena un modelo de crédito usando el dataset tabular `Score.csv`.

Flujo:
  1. Leer `data/raw/Score.csv`.
  2. Separar `Credit_Score` como variable objetivo (3 clases).
  3. Dividir en train/validación.
  4. Crear un pequeño set de `score` (sin target) para pruebas de `predict.py`.
  5. Entrenar un pipeline con preprocesamiento y un `MLPClassifier` (ANN).
  6. Guardar:
     - `data/processed/credit_train.csv`
     - `data/processed/credit_val.csv`
     - `data/processed/credit_score.csv`
     - `models/best_model.pkl`
"""

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


TARGET_COL = "Credit_Score"


def get_project_root() -> Path:
    """Devuelve la raíz del proyecto asumiendo que este archivo está en src/."""
    return Path(__file__).resolve().parents[1]


def load_raw_data() -> pd.DataFrame:
    """Carga `Score.csv` desde `data/raw`.")

    root = get_project_root()
    raw_path = root / "data" / "raw" / "Score.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo esperado: {raw_path}")
    df = pd.read_csv(raw_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{TARGET_COL}' en Score.csv")
    return df


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, score_size: float = 0.1, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide el dataset completo en train, val y score (este último sin target)."""
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + score_size, random_state=random_state, stratify=y
    )

    # De X_temp sacamos validación y score
    relative_score = score_size / (test_size + score_size)
    X_val, X_score, y_val, y_score = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_score,
        random_state=random_state,
        stratify=y_temp,
    )

    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train
    val_df = X_val.copy()
    val_df[TARGET_COL] = y_val

    score_df = X_score.copy()  # sin columna objetivo

    return train_df, val_df, score_df


def build_pipeline(df: pd.DataFrame) -> Pipeline:
    """Construye un pipeline de preprocesamiento + MLPClassifier."""
    X = df.drop(columns=[TARGET_COL])
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=100,
        random_state=42,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )
    return pipe


def save_processed(train_df: pd.DataFrame, val_df: pd.DataFrame, score_df: pd.DataFrame) -> None:
    root = get_project_root()
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "credit_train.csv"
    val_path = processed_dir / "credit_val.csv"
    score_path = processed_dir / "credit_score.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    score_df.to_csv(score_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenar modelo Credit Score (ANN)")
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Proporción para validación (default: 0.2)",
    )
    parser.add_argument(
        "--score_size",
        type=float,
        default=0.1,
        help="Proporción para score (default: 0.1 del total)",
    )
    args = parser.parse_args()

    print("📥 Cargando datos crudos...")
    df = load_raw_data()

    print("✂️ Generando splits train / val / score...")
    train_df, val_df, score_df = split_data(
        df,
        test_size=args.val_size,
        score_size=args.score_size,
        random_state=42,
    )
    save_processed(train_df, val_df, score_df)

    y_train = train_df[TARGET_COL]
    X_train = train_df.drop(columns=[TARGET_COL])

    y_val = val_df[TARGET_COL]
    X_val = val_df.drop(columns=[TARGET_COL])

    print("⚙️ Construyendo pipeline (preprocesamiento + MLPClassifier)...")
    pipe = build_pipeline(train_df)

    print("🚀 Entrenando modelo...")
    pipe.fit(X_train, y_train)

    print("📊 Evaluando accuracy en validación...")
    val_pred = pipe.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    print(f"Accuracy en validación: {acc:.4f}")

    print("💾 Guardando modelo en models/best_model.pkl...")
    root = get_project_root()
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "best_model.pkl"
    joblib.dump(pipe, model_path)

    print(f"✅ Modelo guardado en: {model_path}")


if __name__ == "__main__":
    main()
