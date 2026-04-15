# UFDR Copilot — Grounded Forensic Investigation System

MSc Data Science Group Project | CERT Insider Threat Dataset r4.2 | Due May 5, 2026

---

## Setup (Each Member — One Time)

### Step 1 — Clone repo
```bash
git clone https://github.com/prisha2410/ufdr-copilot.git
cd ufdr-copilot
pip install -r requirements.txt
```

### Step 2 — Get indexes from Member 1
Member 1 will give you a hard disk with:
```
pageindex_store/   → pkl index maps
pageindex_data/    → JSONL event files
faiss_index/       → FAISS vectors
chroma_store/      → ChromaDB (optional)
answers/           → ground truth files
```
Copy these folders anywhere on your machine.

### Step 3 — Create .env file
Create a file called `.env` in the repo root:
```
DATA_PATH=C:/your_path/r4.2
PROCESSED_DIR=C:/your_path/processed_data
PAGEINDEX_DIR=C:/your_path/pageindex_data
PAGEINDEX_STORE=C:/your_path/pageindex_store
FAISS_DIR=C:/your_path/faiss_index
CHROMA_DIR=C:/your_path/chroma_store
INSIDERS_CSV=C:/your_path/answers/insiders.csv
SCENARIOS_FILE=C:/your_path/answers/scenarios.txt
```
Replace `C:/your_path/` with where you saved the data.

### Step 4 — Test setup
```python
from indexing.retriever_api import get_by_filter
records = get_by_filter(user="LAP0338", top_k=5)
print(f"Found {len(records)} records")
```

---

## How to Use the Retriever

```python
from indexing.retriever_api import get_by_filter, get_by_vector, get_by_id

# Structured filter — fast, uses index maps
records = get_by_filter(user="LAP0338", action="connect_device", hour_min=18)
records = get_by_filter(date="2010-02-01", event_type="email")
records = get_by_filter(action="file_copy", hour_min=18, hour_max=23)

# Semantic search — natural language
records = get_by_vector("employee stealing data before leaving company")
records = get_by_vector("suspicious USB activity after hours")

# HTTP browsing search (needs http FAISS index)
records = get_by_vector("wikileaks upload", search_http=True)

# Single record lookup
record = get_by_id("email_000123")

# Full user timeline
from indexing.retriever_api import get_user_timeline
timeline = get_user_timeline("LAP0338")
```

---

## Team Roles & Branches

| Member | Role | Branch | Uses |
|--------|------|--------|------|
| Member 1 | Data & Indexing | indexing | builds everything |
| Member 2 | Retrieval & Reranker | retrieval | get_by_filter, get_by_vector |
| Member 3 | Agent & Tool Policy | agent | get_by_filter, get_by_vector |
| Member 4 | Case, Graph & Report | case-engine | get_by_id, get_user_timeline |

---

## Project Structure

```
ufdr-copilot/
├── configs/
│   └── paths.py              ← paths read from .env
├── data_pipeline/
│   ├── data_pipeline.py      ← Step 1: CSV to JSONL
│   └── enhance_pageindex.py  ← Step 2: add PageIndex fields
├── indexing/
│   ├── build_pageindex.py    ← Step 3: build index maps
│   ├── build_chroma.py       ← Step 4: build ChromaDB
│   ├── build_faiss.py        ← Step 5: build FAISS (all events)
│   ├── build_faiss_http.py   ← Step 6: build FAISS for http
│   └── retriever_api.py      ← what teammates import
├── retrieval/                ← Member 2 work here
├── agent/                    ← Member 3 work here
├── case_engine/              ← Member 4 work here
├── evaluation/               ← evaluation pipeline
└── answers/
    ├── insiders.csv          ← ground truth
    └── scenarios.txt         ← scenario descriptions
```

---

## Dataset

CERT Insider Threat Dataset r4.2

| File | Records |
|------|---------|
| email.jsonl | 1,000,037 |
| logon.jsonl | 329,511 |
| device.jsonl | 156,911 |
| file.jsonl | 171,112 |
| http.jsonl | ~3 million |
| ldap.jsonl | 16,743 |
| psychometric.jsonl | 1,000 |

Temporal filter: Jan 1 – Jun 30, 2010

---

## Insider Threat Scenarios

| # | Type | Detection Query |
|---|------|----------------|
| 1 | USB + Wikileaks | get_by_filter(action="connect_device", hour_min=18) + get_by_vector("wikileaks", search_http=True) |
| 2 | Job switching | get_by_vector("job search linkedin", search_http=True) |
| 3 | Malicious admin | get_by_vector("suspicious file access mass email") |
| 4 | Insider snooping | get_by_filter(user="X", event_type="logon") |
| 5 | Dropbox revenge | get_by_vector("dropbox upload", search_http=True) |

Ground truth: `answers/insiders.csv`