"""Export everything the demo needs from the research repo (run once; the output is committed).

    <research repo>/.venv/Scripts/python.exe tools/build_assets.py --research <research repo>

Writes:
- checkpoints/query_d0.2_s2024.pth + meta.json: the reported checkpoint (SMLP4Rec + query embedding,
  seed 2024) and the sameDest weights (alpha, w, recency) of `results/week6_retrain/main_s2024.json`;
- checkpoints/vocab.npz: per-field vocabulary of the query embedding (train window only);
- checkpoints/baselines.npz: popularity, first-booking popularity and the ItemKNN similarity (K = 50);
- data/sample_users.json: a curated sample of test users (see `pick_users`).

The sample is NOT representative: it is biased toward users the main model ranks well, and keeps cold
users and users the model gets wrong so that the demo shows failures too.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--research", type=Path, required=True)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()
R = args.research.resolve()
HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / "src"), str(R)]

from nbp.baselines.itemknn import fit_itemknn, user_item_matrix  # noqa: E402
from nbp.data.query_features import QUERY_FIELDS, build_vocab  # noqa: E402

SEED = 2024
RUN = "2026-10-13_expedia_query-embedding_d0.2-s2024"
N_PICK = {  # tag -> number of users
    "warm_good": 22,     # user NDCG@10 >= 0.5
    "warm_partial": 6,   # 0 < NDCG@10 < 0.5
    "warm_wrong": 8,     # none of the user's clusters in the top-20
    "cold_good": 8,      # first test booking has no history, NDCG@10 >= 0.5
    "cold_wrong": 4,     # cold, none in the top-20
}
MAX_TEST = 6

rows = np.load(R / "data/processed/expedia_date_split/rows.npz")
W, USER, Y, L = rows["window"], rows["user"], rows["y"].astype(np.int64), rows["L"]
assert (rows["event"] == np.arange(len(W))).all()
rk = np.load(R / "data/interim/week6/ranks_main_s2024.npz")
tf_rank = dict(zip(rk["event_test"].tolist(), rk["test_tf"].tolist()))  # teacher-forced rank of the true cluster
top = np.load(R / "data/interim/week6/user_top20.npz")
main = top[f"main_{SEED}_m2_merge"].astype(np.int64)  # (u, 20) clusters 0..99
users = top["user"]

# ---- per-user outcome ---------------------------------------------------------------------------
te = np.flatnonzero(W == 2)
order = te[np.lexsort((te, USER[te]))]
us = USER[order]
start = np.flatnonzero(np.r_[True, us[1:] != us[:-1]])
end = np.r_[start[1:], len(order)]
assert np.array_equal(us[start], users)
n_test = end - start
cold = L[order[start]] == 0
disc = [1.0 / math.log2(r + 1) for r in range(1, 21)]
ndcg10 = np.zeros(len(users))
rec20 = np.zeros(len(users))
for i in range(len(users)):
    truth = set((Y[order[start[i]:end[i]]] - 1).tolist())
    hits = [c in truth for c in main[i]]
    ndcg10[i] = sum(d for d, h in zip(disc[:10], hits[:10]) if h) / sum(disc[:min(10, len(truth))])
    rec20[i] = sum(hits) / len(truth)

ok = n_test <= MAX_TEST
tags = {
    "warm_good": ~cold & ok & (ndcg10 >= 0.5),
    "warm_partial": ~cold & ok & (ndcg10 > 0) & (ndcg10 < 0.5),
    "warm_wrong": ~cold & ok & (rec20 == 0),
    "cold_good": cold & ok & (ndcg10 >= 0.5),
    "cold_wrong": cold & ok & (rec20 == 0),
}
rng = np.random.default_rng(args.seed)
picked: list[tuple[int, str]] = []
for tag, n in N_PICK.items():
    pool = np.flatnonzero(tags[tag])
    for i in rng.choice(pool, size=n, replace=False):
        picked.append((int(i), tag))
rng.shuffle(picked)

# ---- bookings of the picked users -----------------------------------------------------------------
sample = []
for i, tag in picked:
    uid = int(users[i])
    idx = np.flatnonzero(USER == uid)  # event order
    t_idx = idx[W[idx] == 2]
    prev = idx[W[idx] != 2][-20:]
    bk = []
    for e in list(prev) + list(t_idx):
        b = {"event": int(e), "cluster": int(Y[e] - 1), "dest": int(rows["dest"][e]), "test": bool(W[e] == 2)}
        if b["test"]:
            b["fields"] = {f: int(c) for f, c in zip(QUERY_FIELDS, rows["codes"][e])}
            b["official_rank"] = int(tf_rank[int(e)])
        bk.append(b)
    first = t_idx[0]
    sl = int(rows["seq_len"][first])
    assert [b["cluster"] for b in bk[:len(prev)]][-sl:] == (rows["seq"][first][:sl] - 1).tolist() or sl == 0, uid
    sample.append({"user_id": uid, "tag": tag, "cold": bool(cold[i]), "ndcg10": round(float(ndcg10[i]), 4),
                   "bookings": bk})

cfg = yaml.safe_load((R / "configs/split.yaml").read_text(encoding="utf-8"))
res = json.loads((R / "results/week6_retrain/user_top20.json").read_text(encoding="utf-8"))["results"]["main"]
info = {"split": f"train < {cfg['valid_start']} <= valid < {cfg['test_start']} <= test < {cfg['test_end']}",
        "seed": SEED, "n_sample": len(sample),
        "composition": {t: sum(s["tag"] == t for s in sample) for t in N_PICK},
        "population": {"test_users": int(len(users)), "cold_share": round(float(cold.mean()), 4),
                       "ndcg@10_all": round(res["all"]["ndcg@10"], 4), "ndcg@10_cold": round(res["cold"]["ndcg@10"], 4),
                       "recall@20_all": round(res["all"]["recall@20"], 4)},
        "sample_ndcg@10": round(float(np.mean([s["ndcg10"] for s in sample])), 4)}
(HERE / "data").mkdir(exist_ok=True)
(HERE / "data/sample_users.json").write_text(json.dumps({"info": info, "users": sample}, separators=(",", ":")), encoding="utf-8")

# ---- checkpoint, vocabulary, baselines --------------------------------------------------------------
(HERE / "checkpoints").mkdir(exist_ok=True)
shutil.copy(R / "experiments" / RUN / "best.pth", HERE / "checkpoints/query_d0.2_s2024.pth")
tr = np.flatnonzero(W == 0)
vocab = build_vocab(rows["codes"], rows["event"][tr])
np.savez(HERE / "checkpoints/vocab.npz", **{f"vocab_{f}": v for f, v in zip(QUERY_FIELDS, vocab)})
n_items = int(Y.max()) + 1
sd = json.loads((R / "results/week6_retrain/main_s2024.json").read_text(encoding="utf-8"))["samedest"]
(HERE / "checkpoints/meta.json").write_text(json.dumps({
    "label": "SMLP4Rec + query embedding + sameDest (seed 2024)", "file": "query_d0.2_s2024.pth", "n_items": n_items,
    "samedest": {"alpha": sd["alpha"], "w": sd["w"], "recency": sd["recency"]}, "run": RUN}, indent=1), encoding="utf-8")

y_tr = Y[tr]
pop = np.bincount(y_tr, minlength=n_items) / len(y_tr)
first = tr[L[tr] == 0]
pop_first = np.bincount(Y[first], minlength=n_items) / len(first)
k = json.loads((R / "results/week6_retrain/itemknn.json").read_text(encoding="utf-8"))["k_chosen"]
sim = fit_itemknn(user_item_matrix(USER[tr], y_tr, n_items), k).toarray()
tie = yaml.safe_load((R / "configs/train.yaml").read_text(encoding="utf-8"))["itemknn_tie_eps"]
np.savez(HERE / "checkpoints/baselines.npz", pop=pop[1:], pop_first=pop_first[1:], sim=sim[1:, 1:].astype(np.float32),
         tie_eps=np.float64(tie))
print(json.dumps(info, indent=1))
