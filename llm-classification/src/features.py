"""Hand-made features for the pairwise preference problem.

Features are computed once per response (A and B) plus a few that do not depend
on the order of the pair. `make_X` builds the model matrix; calling it with the
two responses swapped gives the mirrored row, which is used for augmentation and
test-time averaging.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REFUSAL = r"(?i)as an ai|i cannot|i can't|i'm sorry|i am sorry|i apologize"
SWAP = [1, 0, 2]  # class order after swapping A and B


def _jaccard(a, b):
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def response_features(responses, prompt_words):
    s = responses
    f = pd.DataFrame(index=s.index)
    f["chars"] = s.str.len()
    f["lines"] = s.str.count("\n")
    f["bullets"] = s.str.count(r"(?m)^\s*(?:[-*•]|\d+[.)])\s")
    f["bold"] = s.str.count(r"\*\*") / 2
    f["code_blocks"] = s.str.count("```") / 2
    f["headers"] = s.str.count(r"(?m)^#{1,6}\s")
    f["urls"] = s.str.count(r"https?://")
    f["questions"] = s.str.count(r"\?")
    f["exclaims"] = s.str.count("!")
    f["sentences"] = s.str.count(r"[.!?](?:\s|$)")
    f["digit_ratio"] = s.str.count(r"\d") / f["chars"].clip(lower=1)
    f["upper_ratio"] = s.str.count(r"[A-Z]") / f["chars"].clip(lower=1)
    f["refusal"] = s.str.contains(REFUSAL).astype(int)
    f["empty"] = (f["chars"] == 0).astype(int)

    words = s.str.lower().str.findall(r"\w+")
    f["words"] = words.str.len()
    f["unique_ratio"] = words.map(lambda w: len(set(w)) / len(w) if w else 0.0)
    f["avg_word_len"] = words.map(lambda w: sum(map(len, w)) / len(w) if w else 0.0)
    f["prompt_overlap"] = [_jaccard(set(w), p) for w, p in zip(words, prompt_words)]
    f["log_chars"] = np.log1p(f["chars"])
    return f, words


def build_features(df):
    """Per-response frames for A and B, plus order-independent features."""
    prompt_words = df["prompt"].str.lower().str.findall(r"\w+").map(set)
    fa, words_a = response_features(df["response_a"], prompt_words)
    fb, words_b = response_features(df["response_b"], prompt_words)

    shared = pd.DataFrame(index=df.index)
    shared["same_response"] = (df["response_a"] == df["response_b"]).astype(int)
    shared["ab_overlap"] = [_jaccard(set(a), set(b)) for a, b in zip(words_a, words_b)]
    shared["prompt_chars"] = df["prompt"].str.len()
    shared["prompt_turns"] = df["prompt_turns"].map(len)
    shared["prompt_code"] = df["prompt"].str.contains("```|def |class |import ").astype(int)
    return fa, fb, shared


def make_X(fa, fb, shared):
    """Model matrix: A features, B features, A - B differences and shared features."""
    return pd.concat([
        fa.add_prefix("a_"),
        fb.add_prefix("b_"),
        (fa - fb).add_prefix("d_"),
        shared,
    ], axis=1)


class Projection:
    """Linear projection (a fitted PCA) kept as plain arrays, so it can be saved
    and applied without depending on the scikit-learn version."""

    def __init__(self, mean, components):
        self.mean, self.components = mean, components

    def transform(self, x):
        return (x - self.mean) @ self.components.T


def _fit_projection(x, n_components):
    pca = PCA(n_components, random_state=0).fit(x)
    return Projection(pca.mean_.astype(np.float32), pca.components_.astype(np.float32))


def save_projections(projections, path):
    (rm, rc), (pm, pc) = ((q.mean, q.components) for q in projections)
    np.savez(path, response_mean=rm, response_components=rc, prompt_mean=pm, prompt_components=pc)


def load_projections(path):
    z = np.load(path)
    return (Projection(z["response_mean"], z["response_components"]),
            Projection(z["prompt_mean"], z["prompt_components"]))


def embedding_features(emb, n_response=32, n_prompt=16, projections=None):
    """Features from unit-norm sentence embeddings (dot product = cosine).

    Returns frames shaped like `build_features` (per-response A and B, shared) so
    they can be concatenated, plus the projections to reuse on another split.
    """
    p, a, b = (emb[k].astype(np.float32) for k in ("prompt", "response_a", "response_b"))
    if projections is None:
        projections = (_fit_projection(np.vstack([a, b]), n_response), _fit_projection(p, n_prompt))
    response_proj, prompt_proj = projections

    def per_response(r):
        f = pd.DataFrame(response_proj.transform(r), columns=[f"emb{i}" for i in range(n_response)])
        f["prompt_cos"] = (r * p).sum(axis=1)
        return f

    shared = pd.DataFrame(prompt_proj.transform(p), columns=[f"pemb{i}" for i in range(n_prompt)])
    shared["ab_cos"] = (a * b).sum(axis=1)
    return per_response(a), per_response(b), shared, projections
