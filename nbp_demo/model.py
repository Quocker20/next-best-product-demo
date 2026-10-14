"""SMLP4Rec with a query embedding, in plain PyTorch (no RecBole), loadable from the research checkpoints.

History = the user's last S bookings (item ids, 0 = padding, right-padded). The current search is one
extra position S: a learned mask vector plus one embedding per query field. Two passes through the
same mixer block, the hidden state of the query position is multiplied with the item embedding table.
Parameter names match the research repo (`nbp.models.smlprec_query.SMLPRECQuery`), so its `best.pth`
state dict loads with `strict=True`.
"""

from __future__ import annotations

import torch
from torch import nn

EXPANSION = 3


def _mlp(dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, EXPANSION * dim, bias=False), nn.GELU(), nn.Dropout(0.0),
                         nn.Linear(EXPANSION * dim, dim, bias=False), nn.Dropout(0.0))


class Mixer(nn.Module):
    """Channel (c), sequence (h) and hidden (w) MLPs on a LayerNorm-ed (B, 1, S+1, H) tensor, summed."""

    def __init__(self, shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.c, self.h, self.w = _mlp(shape[0]), _mlp(shape[1]), _mlp(shape[2])
        self.norm2 = nn.LayerNorm(shape[2])

    def branches(self, x: torch.Tensor):
        xn = self.norm2(x)
        x0 = self.c(xn.transpose(1, 3).contiguous()).transpose(3, 1).contiguous()
        x1 = self.h(xn.transpose(2, 3).contiguous()).transpose(3, 2).contiguous()
        x2 = self.w(xn)
        return xn, x0, x1, x2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, x0, x1, x2 = self.branches(x)
        return x0 + x1 + x2


class QueryModel(nn.Module):
    def __init__(self, n_items: int, field_sizes: list[int], seq_len: int = 20, hidden: int = 64,
                 n_layers: int = 2, eps: float = 1e-12) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.item_embedding = nn.Embedding(n_items, hidden, padding_idx=0)
        self.layers = nn.ModuleList([Mixer((1, seq_len + 1, hidden))])  # one block, reused n_layers times
        self.LayerNorm = nn.LayerNorm(hidden, eps=eps)
        self.query_emb = nn.ModuleList([nn.Embedding(n, hidden) for n in field_sizes])
        self.query_mask = nn.Parameter(torch.zeros(hidden))

    def query_vector(self, q_idx: torch.Tensor) -> torch.Tensor:
        q = self.query_mask.expand(q_idx.shape[0], -1)
        for j, emb in enumerate(self.query_emb):
            q = q + emb(q_idx[:, j].long())
        return q

    @torch.no_grad()
    def scores(self, seq: torch.Tensor, q_idx: torch.Tensor) -> torch.Tensor:
        """(B, n_items) logits; column 0 is the padding id (callers mask it)."""
        h = self.item_embedding(seq)
        x = torch.cat([h, self.query_vector(q_idx).unsqueeze(1)], dim=1).unsqueeze(1)  # (B, 1, S+1, H)
        for _ in range(self.n_layers):
            x = self.layers[0](x)
        return self.LayerNorm(x[:, 0, -1]) @ self.item_embedding.weight.T

    @torch.no_grad()
    def trace(self, seq: torch.Tensor, q_idx: torch.Tensor) -> dict:
        """Intermediate tensors of one row, for the architecture view of the page."""
        h = self.item_embedding(seq)
        qv = self.query_vector(q_idx)
        x_cat = torch.cat([h, qv.unsqueeze(1)], dim=1)
        x = x_cat.unsqueeze(1)
        blk = self.layers[0]
        xn, x0, x1, x2 = blk.branches(x)
        y1 = x0 + x1 + x2
        y2 = blk(y1)
        out = self.LayerNorm(y2[:, 0, -1])
        logits = out @ self.item_embedding.weight.T
        return {"history_emb": h, "query_vec": qv, "combined": x_cat, "tensor": x, "layernorm": xn,
                "seq_mix": x1, "feat_mix": x0, "chan_mix": x2, "layer1": y1, "layer2": y2,
                "query_hidden": out, "logits": logits}
