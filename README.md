# UFDR Copilot — Grounded Forensic Investigation System

MSc Data Science Group Project | CERT Insider Threat Dataset r4.2 | Due May 5, 2026

---

## Live API

```
https://ufdr-copilot-api.onrender.com
https://ufdr-copilot-api.onrender.com/docs
```

> Note: Free tier may take 50 seconds to wake up after inactivity. First request after wake-up will be slow (~30 sec) as indexes load from disk.

---

## Team Roles

| Member | Role | Branch |
|--------|------|--------|
| Member 1 (Prisha) | Data & Indexing | indexing |
| Member 2 | Retrieval & Reranker | retrieval |
| Member 3 | Agent & Tool Policy | agent |
| Member 4 | Case, Graph & Report | case-engine |

---

## Quick Start (All Members)

```bash
# 1. Clone
git clone https://github.com/prisha2410/ufdr-copilot.git
cd ufdr-copilot

# 2. Install
pip install -r requirements.txt

# 3. Create .env
echo "RETRIEVER_HOST=https://ufdr-copilot-api.onrender.com" > .env

# 4. Use the client
python -c "
from client.retriever_client import RetrieverClient
client = RetrieverClient()
records = client.filter(user='LAP0338', action='connect_device')
print(records[:2])
"
```

---

## API Endpoints

| Endpoint | Description | Example |
|----------|-------------|---------|
| GET /health | Liveness check | /health |
| GET /filter | Structured PageIndex query | /filter?user=LAP0338&action=connect_device |
| GET /vector | Semantic FAISS search | /vector?query=data+theft+after+hours |
| GET /record/{page_id} | Single record lookup | /record/email_000123 |
| GET /stats | Index statistics | /stats |
| GET /http_search | Browsing log search | /http_search?keyword=wikileaks |

---

## Client Usage

```python
from client.retriever_client import RetrieverClient
client = RetrieverClient()

# Structured filter
records = client.filter(user="LAP0338", action="connect_device", hour_min=18)
records = client.filter(date="2010-02-01", event_type="email")
records = client.filter(action="file_copy", hour_min=18, hour_max=23)

# Semantic search
records = client.vector("employee stealing data before leaving company")
records = client.vector("suspicious USB activity after hours")

# HTTP browsing search
records = client.http_search(keyword="wikileaks")
records = client.http_search(keyword="dropbox", user="LAP0338")

# Single record
record = client.get_record("email_000123")
```

---

## Architecture

```
CERT r4.2 CSVs
      ↓
data_pipeline.py        → JSONL files (1.67M records)
enhance_pageindex.py    → adds page_id, date, hour, action fields
      ↓
build_pageindex.py      → user/action/date/hour index maps (pkl)
build_chroma.py         → ChromaDB vector store
build_faiss.py          → FAISS index (384-dim, 1.67M vectors)
      ↓
retriever_api.py        → core retrieval logic (disk-based, no OOM)
server.py               → FastAPI server (6 endpoints)
      ↓
https://ufdr-copilot-api.onrender.com
```

### Infrastructure

```
Indexes:     HuggingFace — pisha2410/ufdr-indexes (public dataset)
JSONL files: HuggingFace — pageindex_data/ folder
Deployment:  Render free tier (512MB RAM)
RAM usage:   ~110MB at startup (4 index maps only)
Records:     Read from JSONL files on demand (disk-based)
```

---

## Dataset

CERT Insider Threat Dataset r4.2 — Carnegie Mellon University SEI

Download: https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247

| File | Records | Notes |
|------|---------|-------|
| email.jsonl | 1,000,037 | Full |
| logon.jsonl | 329,511 | Full |
| device.jsonl | 156,911 | Full |
| file.jsonl | 171,112 | Full |
| http_sample.jsonl | 1,082,684 | 10% unbiased sample |
| ldap.jsonl | 16,743 | Full |
| psychometric.jsonl | 1,000 | Full |

Temporal filter: Jan 1 - Jun 30, 2010

---

## Insider Threat Scenarios

| # | Type | Key Signals | Best Endpoints |
|---|------|-------------|----------------|
| 1 | USB + Wikileaks | after-hours login, USB, wikileaks URL | /filter + /http_search |
| 2 | Job switching + data theft | job site URLs, file downloads | /http_search + /filter |
| 3 | Malicious admin | unusual file access, mass email | /filter + /vector |
| 4 | Insider snooping | login to other machines | /filter |
| 5 | Layoff revenge via Dropbox | dropbox URLs, bulk uploads | /http_search + /filter |

Ground truth: answers/insiders.csv + answers/scenarios.txt

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Recall@K | % of relevant evidence in top-K results |
| Precision@K | % of retrieved results that are relevant |
| Grounding Score | % of claims backed by cited page_id |
| Tool Accuracy | % correct tool decisions by agent |

---

## Local Development

```bash
# Run server locally
python api/server.py

# Server runs at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Local paths (edit configs/paths.py)

```python
DATA_PATH       = "D:/dl_proj/r4.2"
PROCESSED_DIR   = "D:/dl_proj/processed data"
PAGEINDEX_DIR   = "D:/dl_proj/pageindex_data"
PAGEINDEX_STORE = "D:/dl_proj/pageindex_store"
FAISS_DIR       = "D:/dl_proj/faiss_index"
```

---

## Notes

- /http_search uses a 10% unbiased random sample of http.jsonl (1.08M records)
- FAISS and embedding model load lazily on first /vector request (~30 sec delay)
- Free Render tier sleeps after 15 min inactivity - first request takes ~50 sec to wake
- For demo day: hit /health every 10 min to keep server awake