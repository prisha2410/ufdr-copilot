# Data Setup Guide

> The raw data and all generated indexes are **not committed to git** (too large).  
> Every team member must download and process the data locally by following these steps exactly.

---

## 1. Download CERT r4.2

Go to: https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247

Download **`r4.2.tar.bz2`** and **`answers.tar.bz2`**.

Extract both:
```bash
# Linux / Mac
tar -xvjf r4.2.tar.bz2
tar -xvjf answers.tar.bz2

# Windows — use 7-Zip or WinRAR, right-click → Extract Here
```

After extraction you should have:

```
r4.2/
├── email.csv
├── logon.csv
├── device.csv
├── file.csv
├── http.csv
├── psychometric.csv
└── LDAP/
    ├── 2009-12.csv
    ├── 2010-01.csv
    └── ...

answers/
├── insiders.csv
└── scenarios.txt
```

---

## 2. Configure Paths

Edit **`configs/paths.py`** to match your local folder layout:

```python
DATA_PATH     = "D:/dl_proj/r4.2"          # ← point to extracted r4.2 folder
PROCESSED_DIR = "D:/dl_proj/processed_data"
PAGEINDEX_DIR = "D:/dl_proj/pageindex_data"
PAGEINDEX_STORE = "D:/dl_proj/pageindex_store"
CHROMA_DIR    = "D:/dl_proj/chroma_store"
FAISS_DIR     = "D:/dl_proj/faiss_index"
INSIDERS_CSV  = "D:/dl_proj/answers/insiders.csv"
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support (if you have CUDA):
```bash
pip install faiss-gpu
# and the correct torch version from https://pytorch.org
```

---

## 4. Run the Pipeline (in order)

```bash
# Step 1 — Raw CSVs → JSONL  (~20–40 min on SSD)
python data_pipeline/data_pipeline.py

# Step 2 — Add PageIndex fields  (~10–20 min)
python data_pipeline/enhance_pageindex.py

# Step 3 — Build structured index maps  (~5–15 min)
python indexing/build_pageindex.py

# Step 4 — Embed + persist to ChromaDB  (~30–90 min, GPU speeds this up)
python indexing/build_chroma.py

# Step 5 — Build FAISS index from ChromaDB embeddings  (~5–10 min)
python indexing/build_faiss.py
```

After Step 5 your folder layout should be:

```
processed_data/    ← email.jsonl, logon.jsonl, ...
pageindex_data/    ← same files with extra fields
pageindex_store/   ← user_index.pkl, action_index.pkl, date_index.pkl,
                      hour_index.pkl, record_store.pkl
chroma_store/      ← ChromaDB files
faiss_index/       ← events.index, page_id_map.pkl
```

---

## 5. Start the API Server (Member 1 only)

```bash
python api/server.py
```

Server starts at `http://0.0.0.0:8000`.  
Share your **local IP** with the team so they can set `RETRIEVER_HOST`.

Other members — set this environment variable (or add to a `.env` file):

```bash
# .env  (never commit this file)
RETRIEVER_HOST=http://192.168.x.x:8000
```

Test it:
```bash
curl http://192.168.x.x:8000/health
# → {"status":"ok","service":"UFDR Retriever"}
```

---

## Dataset Overview

| File | Contents | Size (approx) |
|------|----------|---------------|
| email.csv | Communication logs | ~2 GB |
| logon.csv | Login/logout events | ~1 GB |
| device.csv | USB connect/disconnect | ~500 MB |
| file.csv | File access records | ~1.5 GB |
| http.csv | Web browsing logs | ~3 GB |
| LDAP/*.csv | Employee directory | ~10 MB |
| psychometric.csv | Personality scores | ~1 MB |

**Temporal filter applied:** Jan 1, 2010 → Jun 30, 2010  
**Reason:** Preserves all insider threat scenarios while reducing size by ~50%.
