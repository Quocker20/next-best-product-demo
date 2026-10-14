# Next-Best Product demo

Offline demo of the next-best hotel-cluster recommender (SMLP4Rec + query embedding + sameDest) on the
public Expedia data. Only the code and assets needed to run the demo live here; training, evaluation and
the report are in the research repo (`next-best-product-recommendation`).

- Model: checkpoint `checkpoints/query_d0.2_s2024.pth` (reported seed 2024) + sameDest fusion (alpha 0.1, w 0.5, recency 0.7).
- Baselines: ItemKNN (K = 50, history only) and global popularity.
- Data: 48 test users (`data/sample_users.json`, window 2014-11-01 to 2015-01-01), each with the last 20 earlier bookings. The demo replays one booking at a time, like a customer searching: the history is every booking before it (teacher-forced), the search fields are the 14 fields of that booking.
- `cluster` is an anonymous proxy of room category / package. Offline evaluation only.

## The sample is biased on purpose

Groups come from the user-level evaluation (one fixed top-20 per user), used only to pick the sample; the demo itself replays single bookings.

| Group | Users | Rule |
|---|---|---|
| warm, good | 22 | has history, user NDCG@10 >= 0.5 |
| warm, partial | 6 | has history, 0 < NDCG@10 < 0.5 |
| warm, wrong | 8 | has history, no booked cluster in the top-20 |
| cold, good | 8 | first test booking without history, NDCG@10 >= 0.5 |
| cold, wrong | 4 | cold, no booked cluster in the top-20 |

Mean NDCG@10 of the sample is 0.504 against 0.447 on all 222,428 test users: the demo leans toward users
the model ranks well, but keeps cold users and misses. Do not quote sample numbers as results.

## Run

```bash
pip install -r requirements.txt
python server.py            # http://127.0.0.1:8765, PORT env var is honoured
python -m pytest tests      # engine reproduces the research rank of the true cluster for every test booking of the 48 users
```

`index.html` is the UI (served at `/`); `/healthz` is the liveness probe.

## Deploy

The page calls the Python API, so it needs a server, not static hosting. Any container or Python host works:

```bash
docker build -t nbp-demo . && docker run -p 8080:8080 nbp-demo
```

Render / Railway / Fly: build with `pip install -r requirements.txt`, start with `python -u server.py`
(CPU torch, about 300 MB RAM).

## Rebuild the assets

`tools/build_assets.py` regenerates `checkpoints/` and `data/` from the research repo (needs its data and venv):

```bash
<research repo>/.venv/Scripts/python.exe tools/build_assets.py --research <research repo>
```
