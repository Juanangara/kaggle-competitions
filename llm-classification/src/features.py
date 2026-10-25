"""Hand-made features for the pairwise preference problem.

Features are computed once per response (A and B) plus a few that do not depend
on the order of the pair. `make_X` builds the model matrix; calling it with the
two responses swapped gives the mirrored row, which is used for augmentation and
test-time averaging.
"""
import numpy as np
import pandas as pd

REFUSAL = r"(?i)as an ai|i cannot|i can't|i'm sorry|i am sorry|i apologize"


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
