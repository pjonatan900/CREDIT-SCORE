"""make_dataset.py

En esta adaptación el preprocesamiento principal se realiza dentro de `train.py`.
Este script se deja como stub para mantener compatibilidad con la estructura del
proyecto guía `model-credit`.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def get_project_root() -> Path:
    """Devuelve la raíz del proyecto asumiendo que este archivo está en src/."""
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stub de make_dataset. Actualmente el flujo principal está en train.py."
        )
    )
    _ = parser.parse_args()
    print(
        "make_dataset.py (stub): no se ejecutó ninguna transformación.
"
        "El flujo estándar es usar directamente src/train.py."
    )


if __name__ == "__main__":
    main()
