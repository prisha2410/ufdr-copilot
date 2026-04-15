# Data & Indexing

**Module 1 (Data Layer)** + **Module 2 (Indexing)**

---

## Pipeline Overview

```
Raw CSVs (DATA_PATH)
    │
    ▼  Step 1: data_pipeline/data_pipeline.py
JSONL (PROCESSED_DIR)  — one file per event type
    │
    ▼  Step 2: data_pipeline/enhance_pageindex.py
Enriched JSONL (PAGEINDEX_DIR)  — adds page_id, date, action, keywords, normalized_text
    │
    ├──▶  Step 3: indexing/build_index.py
    │    PageIndex (EMBEDDINGS_DIR/pageindex/)
    │    user_index / action_index / date_index / hour_index / page_store
    │
    └──▶  Step 4: indexing/build_faiss.py
         FAISS (EMBEDDINGS_DIR/faiss/)
         index.faiss + id_map.json
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure paths
cp .env.example .env
# Edit .env — set DATA_PATH to wherever the CERT CSVs live
```

---

## Run

```bash
# All 4 steps
python run_pipeline.py

# Specific steps only (e.g. re-run indexing after a change)
python run_pipeline.py --steps 3 4
```

---

## Output consumed by other members

| Member | Consumes |
|--------|----------|
| Member 2 (Retrieval) | `PAGEINDEX_DIR/*.jsonl`, `EMBEDDINGS_DIR/pageindex/`, `EMBEDDINGS_DIR/faiss/` |
| Member 3 (Agent)     | `EMBEDDINGS_DIR/pageindex/page_store.json` |
| Member 4 (Report)    | `EMBEDDINGS_DIR/pageindex/page_store.json` |

---

## Data & Embeddings (gitignored)

Raw data, processed JSONL, and embeddings are all gitignored.
Add `data/` to your local clone manually, or via DVC / shared storage.

```
data/
├── raw/          ← CERT r4.2 CSVs  (gitignored)
├── processed/    ← Step 1 output    (gitignored)
├── pageindex/    ← Step 2 output    (gitignored)
└── embeddings/   ← Step 3+4 output  (gitignored)
```