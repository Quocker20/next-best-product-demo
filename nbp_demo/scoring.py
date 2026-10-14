"""sameDest, score fusion, probabilities and the two-step user list (numpy only).

Same maths as the research repo (`nbp.hybrid.*`, `nbp.eval.two_step`). Cluster ids here are the raw
hotel clusters 0..99; the model works on ids c + 1 (0 = padding).
"""

from __future__ import annotations

import numpy as np

K = 100
TOP = 20


def same_dest_share(hist_clusters: list[int], hist_dest: list[int], q_dest: int, base: float) -> np.ndarray:
    """(100,) recency-weighted share of the history booked at the searched destination.

    weight of a booking = base ** lag (lag 0 = most recent); divided by the weight of the whole history.
    """
    out = np.zeros(K)
    n = len(hist_clusters)
    total = sum(base ** (n - 1 - j) for j in range(n)) or 1e-12
    for j, (c, d) in enumerate(zip(hist_clusters, hist_dest)):
        if d == q_dest:
            out[c] += base ** (n - 1 - j)
    return out / total


def same_dest_distribution(s: np.ndarray) -> np.ndarray:
    """Normalised sameDest scores; uniform when the user never booked at that destination."""
    tot = s.sum()
    return s / tot if tot > 0 else np.full(len(s), 1.0 / len(s))


def smoothed_log(p: np.ndarray, alpha: float) -> np.ndarray:
    return np.log((1.0 - alpha) * p + alpha / len(p))


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def rank_order(score: np.ndarray) -> np.ndarray:
    """Clusters sorted best first, ties to the lower cluster id."""
    return np.argsort(-score, kind="stable")


def merge_top(ids1, p1, ids2, p2, k: int = TOP) -> list[int]:
    """Reranker m2_merge: union of two scored lists, max probability per cluster, top k."""
    full = np.full(K, -1.0)
    full[ids1] = p1
    full[ids2] = np.maximum(full[ids2], p2)
    return [int(c) for c in np.argsort(-full, kind="stable")[:k]]
