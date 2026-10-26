"""Check whether the out-of-fold LightGBM probabilities need calibration, and test the LoRA blend.

Needs data/oof_embeddings.npy (`python src/train.py --embeddings`). The blend part also needs
data/lora_val_pred.npz, the validation predictions written by notebooks/lora_colab.ipynb.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from data import TARGET_COLS

DATA = Path(__file__).resolve().parent.parent / "data"


def normalize(p):
    return p / p.sum(axis=1, keepdims=True)


def calibration():
    train = pd.read_csv(DATA / "train.csv", usecols=["prompt", *TARGET_COLS])
    y = train[TARGET_COLS].to_numpy().argmax(axis=1)
    groups = pd.factorize(train["prompt"])[0]
    p = normalize(np.load(DATA / "oof_embeddings.npy"))
    logits = np.log(np.clip(p, 1e-7, 1))

    # Each correction is fitted on 4 folds and scored on the 5th, grouped by prompt.
    temperature, logreg = np.zeros_like(p), np.zeros_like(p)
    for fit, held in GroupKFold(n_splits=5).split(p, y, groups):
        best = min(np.arange(0.8, 1.3, 0.02), key=lambda t: log_loss(y[fit], normalize(p[fit] ** (1 / t))))
        temperature[held] = normalize(p[held] ** (1 / best))
        logreg[held] = LogisticRegression(C=10, max_iter=1000).fit(logits[fit], y[fit]).predict_proba(logits[held])

    print(f"raw                {log_loss(y, p):.5f}")
    print(f"temperature        {log_loss(y, temperature):.5f}")
    print(f"logreg on logits   {log_loss(y, logreg):.5f}")

    confidence = p.max(axis=1)
    edges = np.quantile(confidence, [0, 0.25, 0.5, 0.75, 0.95, 1])
    print("\nconfidence bin     mean conf   accuracy")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidence >= lo) & (confidence <= hi)
        print(f"{lo:.2f}-{hi:.2f}          {confidence[m].mean():.3f}       {(p[m].argmax(axis=1) == y[m]).mean():.3f}")


def blend():
    path = DATA / "lora_val_pred.npz"
    if not path.exists():
        print("\nlora_val_pred.npz not found, skipping the blend")
        return
    ids = pd.read_csv(DATA / "train.csv", usecols=["id"])["id"].to_numpy()
    z = np.load(path)
    rows = pd.Series(np.arange(len(ids)), index=ids)[z["id"]].to_numpy()
    y, llm, gbm = z["label"], normalize(z["pred"]), normalize(np.load(DATA / "oof_embeddings.npy")[rows])

    print(f"\nValidation rows: {len(y):,}")
    print(f"LLM (LoRA)         {log_loss(y, llm):.4f}")
    print(f"LightGBM (OOF)     {log_loss(y, gbm):.4f}")
    for w in (0.2, 0.3, 0.4, 0.5, 0.6):
        print(f"blend, {w:.0%} LLM     {log_loss(y, w * llm + (1 - w) * gbm):.4f}")


if __name__ == "__main__":
    calibration()
    blend()
