from __future__ import annotations

"""predict.py

Genera probabilidades de cada clase de `Credit_Score` usando el modelo
entrenado sobre el archivo `data/processed/credit_score.csv` y guarda
el resultado en `data/scores/final_score.csv`.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_score_data(path: Path | None = None) -> pd.DataFrame:
    """Carga el dataset de scoring.

    Si `path` es None, se usa `data/processed/credit_score.csv`.
    """
    root = get_project_root()
    if path is None:
        path = root / "data" / "processed" / "credit_score.csv"

    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de scoring: {path}")

    df = pd.read_csv(path)
    return df


def load_model():
    root = get_project_root()
    model_path = root / "models" / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {model_path}. Entrena primero el modelo."
        )
    return joblib.load(model_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generar probabilidades de Credit_Score (scoring)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help=(
            "Ruta opcional al CSV de scoring. "
            "Si se omite, usa data/processed/credit_score.csv."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help=(
            "Ruta del CSV de salida. "
            "Si se omite, usa data/scores/final_score.csv."
        ),
    )
    args = parser.parse_args()

    custom_input = Path(args.input) if args.input else None

    print("📥 Cargando datos de scoring...")
    X_score = load_score_data(custom_input)

    print("📦 Cargando modelo...")
    model = load_model()

    print("🔮 Generando probabilidades...")
    proba = model.predict_proba(X_score)
    class_labels = model.classes_

    root = get_project_root()
    scores_dir = root / "data" / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = scores_dir / "final_score.csv"

    out_df = pd.DataFrame(proba, columns=[f"proba_{c}" for c in class_labels])
    out_df.to_csv(out_path, index=False)
    print(f"✅ Archivo de scoring generado en: {out_path}")


if __name__ == "__main__":
    main()
