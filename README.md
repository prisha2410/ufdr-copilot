# UFDR Copilot — Grounded Forensic Investigation System

MSc Data Science Group Project | CERT Insider Threat Dataset r4.2 | Due May 5, 2026

---

## How to Use the API (All Members Read This)

There are two ways to access the API. Use **Option A** for all real work.

### Option A — Google Colab (RECOMMENDED — Full API, Free)

**This is the main way to run the API.**

1. Open `colab_server.ipynb` from this repo in Google Colab
2. Go to **Runtime → Run all**
3. Wait 5-10 minutes for indexes to download
4. Copy the ngrok URL from Cell 6 output
5. Share URL in group chat
6. Keep the Colab tab open while team works

**Anyone on the team can do this.** Whoever starts work first runs the notebook.

### Option B — Render (Health Check Only)

```
https://ufdr-copilot-api.onrender.com/health
```

This always works but only for `/health`. Real queries need Colab (Option A).

---

## Quick Start for Members

```bash
# 1. Clone repo
git clone https://github.com/prisha2410/ufdr-copilot.git
cd ufdr-copilot

# 2. Install
pip install -r requirements.txt

# 3. Create .env with the Colab URL shared in group chat
echo "RETRIEVER_HOST=https://YOUR-NGROK-URL.ngrok-free.app" > .env

# 4. Use the client
python -c "
from client.retriever_client import RetrieverClient
client = RetrieverClient()
print(client.filter(user='LAP0338', top_k=3))
"
```

> Note: Update RETRIEVER_HOST in .env each time a new Colab session starts (URL changes per session)

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
| GET /docs | Full API documentation | /docs |

---

## Client Usage

```python
from client.retriever_client import RetrieverClient
client = RetrieverClient()  # reads RETRIEVER_HOST from .env

# Structured filter — fast, uses index maps
records = client.filter(user="LAP0338", action="connect_device", hour_min=18)
records = client.filter(date="2010-02-01", event_type="email")
records = client.filter(action="file_copy", hour_min=18, hour_max=23)

# Semantic search — natural language
records = client.vector("employee stealing data before leaving company")
records = client.vector("suspicious USB activity after hours")

# HTTP browsing search
records = client.http_search(keyword="wikileaks")
records = client.http_search(keyword="dropbox", user="LAP0338")

# Single record lookup
record = client.get_record("email_000123")
```

---

## Team Roles

| Member | Role | Branch | Status |
|--------|------|--------|--------|
| Member 1 (Prisha) | Data & Indexing | indexing | DONE ✅ |
| Member 2 | Retrieval & Reranker | retrieval | In progress |
| Member 3 | Agent & Tool Policy | agent | In progress |
| Member 4 | Case, Graph & Report | case-engine | In progress |

---

## Architecture

```
CERT r4.2 CSVs
      ↓
data_pipeline.py        → JSONL files (1.67M records)
enhance_pageindex.py    → adds page_id, date, hour, action
      ↓
build_pageindex.py      → user/action/date/hour index maps
build_chroma.py         → ChromaDB vector store
build_faiss.py          → FAISS index (384-dim, 1.67M vectors)
      ↓
retriever_api.py        → core retrieval logic
server.py               → FastAPI server (6 endpoints)
      ↓
Google Colab            → full API (12GB RAM, free)
Render                  → health check only (512MB RAM limit)
```

### Where data lives

```
HuggingFace pisha2410/ufdr-indexes:
  pageindex_store/     → pkl index maps
  pageindex_data/      → JSONL event files
  faiss_index/         → FAISS vectors
  user_index_split/    → per-user JSON files
```

---

## Dataset

CERT Insider Threat Dataset r4.2 (Carnegie Mellon SEI)

| File | Records | Notes |
|------|---------|-------|
| email.jsonl | 1,000,037 | Full |
| logon.jsonl | 329,511 | Full |
| device.jsonl | 156,911 | Full |
| file.jsonl | 171,112 | Full |
| http_sample.jsonl | 1,082,684 | 10% unbiased sample |
| ldap.jsonl | 16,743 | Full |
| psychometric.jsonl | 1,000 | Full |

Temporal filter: Jan 1 – Jun 30, 2010

---

## Insider Threat Scenarios

| # | Type | Key Signals | Best Endpoints |
|---|------|-------------|----------------|
| 1 | USB + Wikileaks exfiltration | after-hours login, USB, wikileaks URL | /filter + /http_search |
| 2 | Job switching + data theft | job site URLs, file downloads | /http_search + /filter |
| 3 | Malicious admin (keylogger) | unusual file access, mass email | /filter + /vector |
| 4 | Insider snooping | login to other machines | /filter |
| 5 | Layoff revenge via Dropbox | dropbox URLs, bulk uploads | /http_search + /filter |

Ground truth: `answers/insiders.csv` + `answers/scenarios.txt`

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Recall@K | % of relevant evidence in top-K |
| Precision@K | % of retrieved results that are relevant |
| Grounding Score | % of claims backed by cited page_id |
| Tool Accuracy | % correct tool decisions by agent |

---

## Local Development

```bash
# Run server locally (needs local indexes)
python api/server.py

# Server at http://localhost:8000
# Docs at http://localhost:8000/docs

# Edit configs/paths.py for your local paths
```

---

## Notes

- `/vector` requires FAISS + sentence-transformers (works on Colab, not Render free tier)
- `/http_search` uses 10% unbiased random sample of http.jsonl
- Colab URL changes each session — update .env and share with team
- For demo day: run Colab 10 min early, share URL before demo starts