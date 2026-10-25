"""Inference from saved artifacts: used by the Kaggle notebook and for local checks."""
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from features import SWAP, build_features, embedding_features, load_projections, make_X


def predict(df, emb, artifacts):
    """Probabilities (a, b, tie) for `df`, averaging each row with its A/B mirror."""
    artifacts = Path(artifacts)
    booster = lgb.Booster(model_file=str(artifacts / "lgbm.txt"))
    projections = load_projections(artifacts / "projections.npz")

    fa, fb, shared = build_features(df)
    ea, eb, es, _ = embedding_features(emb, projections=projections)
    fa, fb, shared = (pd.concat([x, y], axis=1) for x, y in ((fa, ea), (fb, eb), (shared, es)))

    cols = booster.feature_name()
    p = booster.predict(make_X(fa, fb, shared)[cols])
    p_swapped = booster.predict(make_X(fb, fa, shared)[cols])[:, SWAP]
    return (p + p_swapped) / 2
