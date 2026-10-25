import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import load_test, load_train
from features import build_features, embedding_features, make_X

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
SWAP = [1, 0, 2]  # class order after swapping A and B

LGB_PARAMS = dict(
    objective="multiclass", learning_rate=0.05, n_estimators=2000, num_leaves=31,
    min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
    reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1,
)


def augment(fa, fb, shared, y):
    """Original rows plus the mirrored rows (A and B swapped, labels swapped)."""
    X = pd.concat([make_X(fa, fb, shared), make_X(fb, fa, shared)], ignore_index=True)
    return X, np.concatenate([y, np.array(SWAP)[y]])


def predict_tta(model, fa, fb, shared):
    """Average the prediction for (A, B) with the mirrored one for (B, A)."""
    p = model.predict_proba(make_X(fa, fb, shared))
    p_swapped = model.predict_proba(make_X(fb, fa, shared))[:, SWAP]
    return p, (p + p_swapped) / 2


def main(use_embeddings=False):
    train, test = load_train(), load_test()
    y = train["label"].to_numpy()
    groups = pd.factorize(train["prompt"])[0]

    print("Building features...")
    fa, fb, shared = build_features(train)
    if use_embeddings:
        emb = np.load(ROOT / "data" / "emb_train.npz")
        assert (emb["id"] == train["id"].to_numpy()).all(), "embeddings are not aligned with train"
        ea, eb, es, _ = embedding_features(emb)
        fa, fb, shared = (pd.concat([x, y_], axis=1) for x, y_ in ((fa, ea), (fb, eb), (shared, es)))
    tag = "embeddings" if use_embeddings else "baseline"

    cv = GroupKFold(n_splits=5)
    oof = {name: np.zeros((len(train), 3)) for name in ("prior", "logreg", "lgbm", "lgbm_tta")}
    best_iters = []

    for fold, (tr, va) in enumerate(cv.split(train, y, groups)):
        part = lambda idx: (fa.iloc[idx], fb.iloc[idx], shared.iloc[idx])

        oof["prior"][va] = np.bincount(y[tr], minlength=3) / len(tr)

        X_tr, y_tr = augment(*part(tr), y[tr])
        X_va = make_X(*part(va))

        diff_cols = [c for c in X_tr.columns if c.startswith("d_") or c in shared.columns]
        logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        logreg.fit(X_tr[diff_cols], y_tr)
        oof["logreg"][va] = logreg.predict_proba(X_va[diff_cols])

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(X_tr, y_tr, eval_X=X_va, eval_y=y[va], eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        best_iters.append(model.best_iteration_)
        oof["lgbm"][va], oof["lgbm_tta"][va] = predict_tta(model, *part(va))
        print(f"fold {fold}: lgbm_tta log loss = {log_loss(y[va], oof['lgbm_tta'][va]):.4f}"
              f" (best iteration {model.best_iteration_})")

    print()
    labels = {"prior": "Class proportions", "logreg": "Logistic regression",
              "lgbm": "LightGBM", "lgbm_tta": "LightGBM + A/B averaging"}
    for name, p in oof.items():
        print(f"{labels[name]:28s} log loss = {log_loss(y, p):.4f}")
    np.save(ROOT / "data" / f"oof_{tag}.npy", oof["lgbm_tta"])

    # Final model on all of train, with the number of trees found in CV.
    X_all, y_all = augment(fa, fb, shared, y)
    params = {**LGB_PARAMS, "n_estimators": int(np.mean(best_iters) * 1.1)}
    final = lgb.LGBMClassifier(**params).fit(X_all, y_all)

    importance = pd.Series(final.booster_.feature_importance("gain"), index=X_all.columns)
    print("\nTop features by gain:")
    print((importance / importance.sum()).nlargest(10).round(3).to_string())

    if use_embeddings:
        return  # test embeddings are computed in the Kaggle notebook, not locally

    ta, tb, ts = build_features(test)
    _, p_test = predict_tta(final, ta, tb, ts)
    submission = pd.DataFrame(p_test, columns=["winner_model_a", "winner_model_b", "winner_tie"])
    submission.insert(0, "id", test["id"])
    out = ROOT / "submission.csv"
    submission.to_csv(out, index=False)
    print(f"\nWrote {out} ({len(submission)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", action="store_true", help="add sentence-embedding features")
    main(parser.parse_args().embeddings)
