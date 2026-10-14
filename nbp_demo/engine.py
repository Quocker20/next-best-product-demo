"""Loads the checkpoint, the vocabulary, the baselines and the sample users; answers the page's requests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from nbp_demo.features import LEAD_BINS, QUERY_FIELDS, encode
from nbp_demo.model import QueryModel
from nbp_demo.scoring import (K, rank_order, same_dest_distribution, same_dest_share,
                              smoothed_log, softmax)

ROOT = Path(__file__).resolve().parents[1]
S = 20  # history length
TOP_K = 5
SMALL = {"dest_type", "ci_month", "lead", "stay", "adults", "children", "rooms", "package",
         "mobile", "channel", "site", "posa"}  # dropdowns; dest and country are free numbers


def tensor_summary(t, max_samples: int = 16) -> dict:
    a = np.asarray(t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t, dtype=float)
    f = a.flatten()
    return {"shape": list(a.shape), "min": round(float(f.min()), 4), "max": round(float(f.max()), 4),
            "mean": round(float(f.mean()), 4), "std": round(float(f.std()), 4),
            "l2_norm": round(float(np.linalg.norm(f)), 4), "sample": [round(float(x), 4) for x in f[:max_samples]]}


class Engine:
    def __init__(self, root: Path = ROOT) -> None:
        meta = json.loads((root / "checkpoints" / "meta.json").read_text(encoding="utf-8"))
        self.meta = meta
        vz = np.load(root / "checkpoints" / "vocab.npz")
        self.vocab = [vz[f"vocab_{f}"] for f in QUERY_FIELDS]
        self.model = QueryModel(meta["n_items"], [len(v) + 1 for v in self.vocab], seq_len=S)
        self.model.load_state_dict(torch.load(root / "checkpoints" / meta["file"], map_location="cpu", weights_only=True))
        self.model.eval()
        sd = meta["samedest"]
        self.alpha, self.w, self.recency = sd["alpha"], sd["w"], sd["recency"]
        b = np.load(root / "checkpoints" / "baselines.npz")
        self.pop, self.pop_first, self.sim = b["pop"], b["pop_first"], b["sim"]  # (100,), (100,), (100, 100)
        self.tie_eps = float(b["tie_eps"])
        users = json.loads((root / "data" / "sample_users.json").read_text(encoding="utf-8"))
        self.info = users["info"]
        self.users = {u["user_id"]: u for u in users["users"]}

    # ---- inputs -------------------------------------------------------------------------------
    def history(self, user: dict, upto: int) -> list[dict]:
        """The user's bookings before booking index `upto`, last S, oldest first."""
        return user["bookings"][:upto][-S:]

    def _tensors(self, hist_clusters: list[int], fields: dict[str, int]):
        seq = np.zeros((1, S), np.int64)
        seq[0, :len(hist_clusters)] = [c + 1 for c in hist_clusters]
        codes = np.array([[fields[f] for f in QUERY_FIELDS]], np.int64)
        q_idx = torch.from_numpy(encode(codes, self.vocab))
        return torch.from_numpy(seq), q_idx

    # ---- scoring ------------------------------------------------------------------------------
    def main_score(self, hist_clusters: list[int], hist_dest: list[int], fields: dict[str, int]):
        """(fused score (100,), model probability (100,), sameDest distribution (100,), known dest)."""
        seq, q_idx = self._tensors(hist_clusters, fields)
        logits = self.model.scores(seq, q_idx).numpy()[0, 1:].astype(np.float64)
        p = softmax(logits)
        share = same_dest_share(hist_clusters, hist_dest, fields["dest"], self.recency)
        sd_p = same_dest_distribution(share)
        score = smoothed_log(p, self.alpha) + self.w * smoothed_log(sd_p, self.alpha)
        return score, p, sd_p, bool(share.sum() > 0)

    def knn_prob(self, hist_clusters: list[int]) -> np.ndarray:
        if not hist_clusters:
            s = self.pop_first.astype(np.float64)
        else:
            h = np.zeros(K)
            for c in hist_clusters:
                h[c] += 1
            s = h @ self.sim + self.tie_eps * self.pop
        return s / s.sum()

    # ---- API ----------------------------------------------------------------------------------
    def meta_json(self) -> dict:
        fields = []
        for f, v in zip(QUERY_FIELDS, self.vocab):
            e = {"name": f}
            if f in SMALL:
                opts = []
                for c in [0] + [int(x) for x in v if int(x) != 0]:
                    if f == "lead":
                        lab = ("không rõ" if c == 0 else f"{LEAD_BINS[c - 1] + 1}-{LEAD_BINS[c]} ngày"
                               if LEAD_BINS[c] < 10_000 else f">{LEAD_BINS[c - 1]} ngày")
                    elif f == "stay":
                        lab = "không rõ" if c == 0 else f"{c - 1} đêm"
                    elif f == "ci_month":
                        lab = "không rõ" if c == 0 else f"tháng {c}"
                    else:
                        lab = "không rõ / 0" if c == 0 else str(c)
                    opts.append([c, f"{c}: {lab}"])
                e["options"] = opts
            else:
                e["hint"] = f"{len(v)} id trong từ vựng của mô hình; id khác = 'không rõ'"
                e["datalist"] = [int(x) for x in v[:400]]
            fields.append(e)
        return {"fields": fields, "info": self.info, "model": self.meta["label"],
                "users": [{"user_id": u["user_id"], "tag": u["tag"], "cold": u["cold"],
                           "n_test": sum(b["test"] for b in u["bookings"]),
                           "n_history": sum(not b["test"] for b in u["bookings"]),
                           "ndcg10": u["ndcg10"]} for u in self.users.values()]}

    def api_user(self, req: dict) -> dict:
        uid = int(req["user_id"])
        if uid not in self.users:
            raise ValueError(f"user {uid} không nằm trong tập mẫu của demo")
        u = self.users[uid]
        tests = [{"index": i, "event": b["event"], "truth": b["cluster"], "n_prior": i,
                  "fields": b["fields"]} for i, b in enumerate(u["bookings"]) if b["test"]]
        return {"user_id": uid, "tag": u["tag"], "cold": u["cold"], "n_total": len(u["bookings"]),
                "n_test": len(tests), "tests": tests}

    def api_recommend(self, req: dict) -> dict:
        uid = req.get("user_id")
        uid = int(uid) if uid not in (None, "") else None
        fields = {f: int(req["fields"][f]) for f in QUERY_FIELDS}
        truth = req.get("truth")
        truth = int(truth) if truth not in (None, "") else None
        if uid is not None and uid not in self.users:
            raise ValueError(f"user {uid} không nằm trong tập mẫu của demo")
        hist: list[dict] = []
        if uid is not None:
            idx = req.get("index")
            n = len(self.users[uid]["bookings"])
            hist = self.history(self.users[uid], int(idx) if idx not in (None, "") else n)
        hc, hd = [b["cluster"] for b in hist], [b["dest"] for b in hist]
        score, p, sd_p, known = self.main_score(hc, hd, fields)
        final = softmax(score)
        order = rank_order(score)
        rows = [{"rank": i + 1, "cluster": int(c), "score": round(float(final[c]), 4),
                 "model_p": round(float(p[c]), 4), "samedest_share": round(float(sd_p[c]), 4)}
                for i, c in enumerate(order)]
        out = {"top": rows[:TOP_K], "all_ranking": rows, "known_destination": known, "w_samedest": self.w,
               "alpha": self.alpha, "history_len": len(hist),
               "dest_in_vocab": bool(encode(np.array([[fields[f] for f in QUERY_FIELDS]]), self.vocab)[0, 0] > 0),
               "activations": self.activations(hc, fields, score, p, sd_p, known)}
        if truth is not None:
            r = int(np.flatnonzero(order == truth)[0]) + 1
            out["truth"] = {"cluster": truth, "rank": r, "in_top": r <= TOP_K}
        out["history"] = [{"seq": i + 1, "event": b["event"], "cluster": b["cluster"], "dest": b["dest"]}
                          for i, b in enumerate(hist)]
        out["total_history_count"] = len(hist)
        out["baselines"] = self.baselines(hc, truth)
        return out

    def baselines(self, hc: list[int], truth: int | None) -> dict:
        res = {}
        note_knn = "chỉ dùng lịch sử" + ("" if hc else " (rỗng: popularity của lượt đặt đầu)")
        for name, prob, note in (
            ("ItemKNN", self.knn_prob(hc), note_knn),
            ("Popularity", self.pop / self.pop.sum(), "cụm phổ biến toàn cục, không dùng lịch sử hay ngữ cảnh"),
        ):
            order = rank_order(prob)
            e = {"note": note, "top": [int(c) for c in order[:TOP_K]],
                 "all_ranking": [{"rank": i + 1, "cluster": int(c)} for i, c in enumerate(order)]}
            if truth is not None:
                e["truth_rank"] = int(np.flatnonzero(order == truth)[0]) + 1
            res[name] = e
        return res

    def activations(self, hc, fields, score, p, sd_p, known) -> dict:
        seq, q_idx = self._tensors(hc, fields)
        t = self.model.trace(seq, q_idx)
        ts = tensor_summary
        return {
            "embedding": {"name": "Lớp Embedding",
                          "role": "Ánh xạ chuỗi lịch sử (item_id) và 14 trường tìm kiếm thành các vector không gian ẩn H=64.",
                          "formula": "h = ItemEmb(seq) || QueryVector(q_idx)",
                          "history_emb": ts(t["history_emb"]), "query_vec": ts(t["query_vec"]), "combined": ts(t["combined"])},
            "tensor3d": {"name": "Tensor 3D/4D Đầu vào",
                         "role": "Cấu trúc tensor đa chiều [Batch=1, Channel=1, Sequence=21, Hidden=64] đưa vào Mixer block.",
                         "formula": "X \\in \\mathbb{R}^{B \\times 1 \\times (S+1) \\times H}", "tensor": ts(t["tensor"])},
            "layernorm": {"name": "Chuẩn hóa Layer Norm",
                          "role": "Chuẩn hóa phân phối kích hoạt trên chiều đặc trưng ẩn trước khi trộn (Mean ≈ 0, Std ≈ 1).",
                          "formula": "X_{norm} = \\mathrm{LayerNorm}(X)", "tensor": ts(t["layernorm"])},
            "seq_mix": {"name": "Sequence Mixing (Trộn chuỗi)",
                        "role": "MLP biến đổi dọc theo chiều chuỗi thời gian (21 vị trí) để học mối tương quan giữa các booking và query.",
                        "formula": "X_{seq} = \\mathrm{MLP}_{seq}(X_{norm})", "tensor": ts(t["seq_mix"])},
            "feat_mix": {"name": "Feature Mixing (Trộn đặc trưng)",
                         "role": "MLP biến đổi dọc theo chiều kênh/đặc trưng (channel dimension).",
                         "formula": "X_{feat} = \\mathrm{MLP}_{feat}(X_{norm})", "tensor": ts(t["feat_mix"])},
            "chan_mix": {"name": "Channel Mixing (Trộn chiều ẩn)",
                         "role": "MLP biến đổi giữa các đơn vị biểu diễn ẩn (hidden units H=64) để học đặc trưng trừu tượng cao cấp.",
                         "formula": "X_{chan} = \\mathrm{MLP}_{chan}(X_{norm})", "tensor": ts(t["chan_mix"])},
            "merge": {"name": "Hợp nhất & Lặp 2 tầng",
                      "role": "Cộng tổng kết quả từ 3 nhánh trộn (x0 + x1 + x2), chuyển tiếp qua khối Mixer thứ 2.",
                      "formula": "Y = X_{seq} + X_{feat} + X_{chan} \\quad (\\times 2 \\text{ layers})",
                      "layer1": ts(t["layer1"]), "layer2": ts(t["layer2"])},
            "output": {"name": "Lớp đầu ra (Query Position)",
                       "role": "Trích xuất biểu diễn ẩn tại vị trí cuối cùng (Query Embedding), chuẩn hóa LayerNorm và nhân ma trận với Embedding Table để ra Logits 100 cụm.",
                       "formula": "\\mathrm{logits} = \\mathrm{LayerNorm}(Y_{query}) \\cdot W_{emb}^T",
                       "query_hidden": ts(t["query_hidden"]), "logits": ts(t["logits"][0, 1:]), "model_softmax_p": ts(p)},
            "samedest": {"name": "Thành phần sameDest",
                         "role": "Phân phối xác suất tiền nghiệm dựa trên các cụm đã từng đặt trước đó tại cùng điểm đến đang tìm kiếm.",
                         "formula": "p_{\\mathrm{sameDest}} \\propto \\sum_{t: \\mathrm{dest}_t = \\mathrm{dest}_q} \\gamma^{L-t}",
                         "active": known, "weight_w": self.w, "distribution": ts(sd_p)},
            "fusion": {"name": "Khối Dung hợp (Score Fusion)",
                       "role": "Kết hợp log-score làm mịn giữa SMLP4Rec và sameDest theo trọng số w, sinh ra phân phối xác suất cuối cùng.",
                       "formula": "\\mathrm{Score} = \\log(p_{\\mathrm{model}} + \\alpha) + w \\cdot \\log(p_{\\mathrm{sameDest}} + \\alpha)",
                       "alpha": self.alpha, "weight_w": self.w, "final_scores": ts(score)},
        }
