"""Query fields of a search and their encoding (same definitions as the research repo)."""

from __future__ import annotations

import numpy as np

LEAD_BINS = [-1, 0, 2, 6, 13, 29, 59, 89, 179, 364, 10_000]  # days between search and check-in

QUERY_FIELDS = ["dest", "dest_type", "ci_month", "lead", "stay", "adults", "children", "rooms",
                "package", "mobile", "channel", "site", "posa", "country"]


def encode(codes: np.ndarray, vocab: list[np.ndarray]) -> np.ndarray:
    """(n, F) raw field codes -> (n, F) embedding indices; a value outside the vocabulary -> 0."""
    out = np.zeros(codes.shape, dtype=np.int64)
    for j, ids in enumerate(vocab):
        if len(ids) == 0:
            continue
        pos = np.searchsorted(ids, codes[:, j])
        hit = (pos < len(ids)) & (ids[np.minimum(pos, len(ids) - 1)] == codes[:, j])
        out[:, j] = np.where(hit, pos + 1, 0)
    return out
