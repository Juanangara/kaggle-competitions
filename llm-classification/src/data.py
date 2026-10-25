"""Carga de datos de LLM Classification Finetuning.

Las columnas `prompt`, `response_a` y `response_b` vienen como listas JSON
serializadas (una entrada por turno de la conversacion). Aqui se parsean y se
agrega la etiqueta de 3 clases.
"""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEXT_COLS = ["prompt", "response_a", "response_b"]
TARGET_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]
CLASS_NAMES = ["a", "b", "tie"]


def parse_turns(raw):
    """Convierte una lista JSON serializada en una lista de strings.

    Un turno puede venir como null; se reemplaza por "" para no romper
    las operaciones de texto posteriores.
    """
    return ["" if turn is None else turn for turn in json.loads(raw)]


def join_turns(turns):
    """Une los turnos en un solo texto, separados por una linea en blanco."""
    return "\n\n".join(turns)


def _prepare(df):
    for col in TEXT_COLS:
        df[col + "_turns"] = df[col].map(parse_turns)
        df[col] = df[col + "_turns"].map(join_turns)
    return df


def load_train(data_dir=DATA_DIR):
    """Train con textos parseados y la columna `label` (0=a, 1=b, 2=tie)."""
    df = _prepare(pd.read_csv(Path(data_dir) / "train.csv"))
    df["label"] = df[TARGET_COLS].to_numpy().argmax(axis=1)
    return df


def load_test(data_dir=DATA_DIR):
    return _prepare(pd.read_csv(Path(data_dir) / "test.csv"))


if __name__ == "__main__":
    train = load_train()
    print(train.shape)
    print(train["label"].value_counts(normalize=True).sort_index().round(3))
    print("turnos por fila:", train["prompt_turns"].map(len).describe().round(2).to_dict())
