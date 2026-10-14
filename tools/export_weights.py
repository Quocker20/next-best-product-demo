"""Convert the research checkpoint (torch state dict) to numpy weights, so the server needs no torch.

    python tools/export_weights.py        # needs torch; run once after build_assets.py
"""

from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sd = torch.load(HERE / "checkpoints" / "query_d0.2_s2024.pth", map_location="cpu", weights_only=True)
np.savez_compressed(HERE / "checkpoints" / "weights.npz", **{k: v.numpy() for k, v in sd.items()})
print(len(sd), "tensors written")
