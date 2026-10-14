"""The demo engine must reproduce the research repo's fixed top-20 per user (reranker m2_merge, seed 2024)."""

import json
from pathlib import Path

from nbp_demo.engine import Engine

ROOT = Path(__file__).resolve().parents[1]


def test_user_lists_match_research_run():
    eng = Engine(ROOT)
    raw = json.loads((ROOT / "data" / "sample_users.json").read_text(encoding="utf-8"))["users"]
    bad = [u["user_id"] for u in raw if eng.user_list(eng.users[u["user_id"]])["top"] != u["official_top20"]]
    assert not bad, f"{len(bad)} users differ from the research run: {bad}"
