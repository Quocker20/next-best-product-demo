"""SMLP4Rec with a query embedding, inference only, in numpy (no torch at run time).

History = the user's last S bookings (item ids, 0 = padding, right-padded). The current search is one
extra position S: a learned mask vector plus one embedding per query field. Two passes through the
same mixer block, the hidden state of the query position is multiplied with the item embedding table.
Weights come from `checkpoints/weights.npz`, exported from the research checkpoint by
`tools/export_weights.py` (parameter names as in the research repo's state dict).
"""

from __future__ import annotations

from math import erf

import numpy as np

_erf = np.frompyfunc(erf, 1, 1)


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)).astype(np.float64))


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


class QueryModel:
    def __init__(self, weights: dict[str, np.ndarray], n_layers: int = 2) -> None:
        self.w = {k: np.asarray(v, np.float64) for k, v in weights.items()}
        self.n_layers = n_layers
        self.n_fields = sum(k.startswith("query_emb.") for k in self.w)

    def query_vector(self, q_idx: np.ndarray) -> np.ndarray:
        """(F,) embedding indices -> (H,) query vector."""
        q = self.w["query_mask"].copy()
        for j in range(self.n_fields):
            q += self.w[f"query_emb.{j}.weight"][q_idx[j]]
        return q

    def _mixer(self, x: np.ndarray):
        """x (S+1, H) -> (xn, channel branch, sequence branch, hidden branch)."""
        w = self.w
        xn = layer_norm(x, w["layers.0.norm2.weight"], w["layers.0.norm2.bias"], 1e-5)
        # channel mixing acts on a dimension of size 1: elementwise Linear(1,3) - GELU - Linear(3,1)
        c1, c2 = w["layers.0.c.0.weight"][:, 0], w["layers.0.c.3.weight"][0]
        x0 = (gelu(xn[..., None] * c1) * c2).sum(-1)
        # sequence mixing: MLP over the S+1 positions, for every hidden unit
        x1 = w["layers.0.h.3.weight"] @ gelu(w["layers.0.h.0.weight"] @ xn)
        # hidden-unit mixing: MLP over the H features, for every position
        x2 = gelu(xn @ w["layers.0.w.0.weight"].T) @ w["layers.0.w.3.weight"].T
        return xn, x0, x1, x2

    def _forward(self, seq: np.ndarray, q_idx: np.ndarray) -> dict:
        w = self.w
        h = w["item_embedding.weight"][seq]  # (S, H)
        qv = self.query_vector(q_idx)
        x_cat = np.concatenate([h, qv[None]], 0)  # (S+1, H)
        xn, x0, x1, x2 = self._mixer(x_cat)
        y1 = x0 + x1 + x2
        y2 = sum(self._mixer(y1)[1:])
        for _ in range(self.n_layers - 2):
            y2 = sum(self._mixer(y2)[1:])
        out = layer_norm(y2[-1], w["LayerNorm.weight"], w["LayerNorm.bias"], 1e-12)
        return {"history_emb": h, "query_vec": qv, "combined": x_cat, "layernorm": xn, "seq_mix": x1,
                "feat_mix": x0, "chan_mix": x2, "layer1": y1, "layer2": y2, "query_hidden": out,
                "logits": out @ w["item_embedding.weight"].T}

    def scores(self, seq: np.ndarray, q_idx: np.ndarray) -> np.ndarray:
        """(n_items,) logits of one row; entry 0 is the padding id (callers drop it)."""
        return self._forward(seq, q_idx)["logits"]

    def trace(self, seq: np.ndarray, q_idx: np.ndarray) -> dict:
        """Intermediate tensors of one row, for the architecture view of the page."""
        t = self._forward(seq, q_idx)
        t["tensor"] = t["combined"][None, None]
        return t
