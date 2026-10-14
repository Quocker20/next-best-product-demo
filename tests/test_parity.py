"""The demo engine must reproduce the research repo's teacher-forced rank of the true cluster
(main model, seed 2024) for every test booking of the sample users."""

import json
from pathlib import Path

from nbp_demo.engine import Engine

ROOT = Path(__file__).resolve().parents[1]


def test_booking_ranks_match_research_run():
    eng = Engine(ROOT)
    raw = json.loads((ROOT / "data" / "sample_users.json").read_text(encoding="utf-8"))["users"]
    bad, n = [], 0
    for u in raw:
        for i, b in enumerate(u["bookings"]):
            if not b["test"]:
                continue
            n += 1
            out = eng.api_recommend({"user_id": u["user_id"], "index": i, "fields": b["fields"], "truth": b["cluster"]})
            if out["truth"]["rank"] != b["official_rank"]:
                bad.append((u["user_id"], i, out["truth"]["rank"], b["official_rank"]))
    assert n > 0 and not bad, f"{len(bad)} / {n} bookings differ: {bad[:5]}"
