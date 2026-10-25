"""Loading for LLM Classification Finetuning.

`prompt`, `response_a` and `response_b` are serialized JSON lists (one entry per
conversation turn). They are parsed here and a 3-class `label` is added.
"""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEXT_COLS = ["prompt", "response_a", "response_b"]
TARGET_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]


def parse_turns(raw):
    # A turn can be null; use "" so later text operations do not break.
    return ["" if turn is None else turn for turn in json.loads(raw)]


def _prepare(df):
    for col in TEXT_COLS:
        df[col + "_turns"] = df[col].map(parse_turns)
        df[col] = df[col + "_turns"].map("\n\n".join)
    return df


def load_train(data_dir=DATA_DIR):
    """Train with parsed texts and `label` (0 = a, 1 = b, 2 = tie)."""
    df = _prepare(pd.read_csv(Path(data_dir) / "train.csv"))
    df["label"] = df[TARGET_COLS].to_numpy().argmax(axis=1)
    return df


def load_test(data_dir=DATA_DIR):
    return _prepare(pd.read_csv(Path(data_dir) / "test.csv"))


if __name__ == "__main__":
    train = load_train()
    print(train.shape)
    print(train["label"].value_counts(normalize=True).sort_index().round(3))
    print("turns per row:", train["prompt_turns"].map(len).describe().round(2).to_dict())
