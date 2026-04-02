# UFDR Copilot — Forensic Investigation Support System

A hybrid Agentic RAG system that retrieves forensic evidence, constructs cases,
builds knowledge graphs, and generates explainable UFDR-style reports.

**Dataset:** CERT Insider Threat Dataset r4.2 (CMU SEI)

---

## Team & Responsibilities

| Member | Role | Modules |
|--------|------|---------|
| Member 1 | Data & Indexing | `data_pipeline/`, `indexing/`, `api/` |
| Member 2 | Retrieval & Reranker | `retrieval/` |
| Member 3 | Agent & Tool Policy | `agent/` |
| Member 4 | Case, Graph & Report | `case_engine/`, `evaluation/` |

---

## Architecture

```
User Query
    ↓
[Agent]  agent/agent.py
    ↓
[Tool Policy]  agent/tool_policy.py
    ↓
[Hybrid Retriever]
 ├── PageIndex (structured filter)   → GET /filter
 ├── FAISS     (semantic search)     → GET /vector
 └── HTTP logs (browsing search)     → GET /http_search
    ↓
[Reranker]          retrieval/reranker.py
    ↓
[Case Builder]      case_engine/case_builder.py
    ↓
[Graph Builder]     case_engine/graph_builder.py
    ↓
[Report Generator]  case_engine/report_generator.py
    ↓
[Evaluation]        evaluation/evaluate.py
```

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/prisha2410/ufdr-copilot.git
cd ufdr-copilot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up data
See **[data/README.md](data/README.md)** for full instructions.

### 4. Configure paths
Edit `configs/paths.py` — this is the **only file you need to change** for your machine.

### 5. Connect to the API
```bash
# Add to .env file (never commit this)
RETRIEVER_HOST=https://ufdr-api.onrender.com
```

### 6. Use the client
```python
from client.retriever_client import RetrieverClient

client = RetrieverClient()

# Structured search (PageIndex)
records = client.filter(user="LAP0338", action="connect_device", hour_min=18)

# Semantic search (FAISS)
records = client.vector("employee copying files after hours", top_k=20)

# Browsing logs search (http.jsonl)
records = client.http_search(keyword="wikileaks")
records = client.http_search(user="LAP0338", keyword="dropbox")

# Single record lookup
rec = client.get_record("email_000123")

# Server stats
info = client.stats()
```

---

## Pipeline Execution Order (Member 1 only)

```bash
python data_pipeline/data_pipeline.py      # Step 1 — CSV → JSONL
python data_pipeline/enhance_pageindex.py  # Step 2 — add PageIndex fields
python indexing/build_pageindex.py         # Step 3 — build dict maps (http skipped)
python indexing/build_chroma.py            # Step 4 — embed + persist ChromaDB (http skipped)
python indexing/build_faiss.py             # Step 5 — build FAISS index
python api/server.py                       # Step 6 — start API server
```

> **Note:** http.jsonl (7.76 GB) is excluded from ChromaDB/FAISS to save time.
> HTTP browsing logs are served via the `/http_search` keyword endpoint instead.
> For full http semantic search, run `python indexing/build_chroma_http.py` (6–8 hrs).

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server liveness check |
| GET | `/filter` | Structured PageIndex query |
| GET | `/vector` | Semantic FAISS query |
| GET | `/record/{page_id}` | Single record lookup by page_id |
| GET | `/stats` | Index statistics |
| GET | `/http_search` | Search browsing logs by user/keyword/date |

Full interactive docs: `https://ufdr-api.onrender.com/docs`

---

## Branching Strategy

```
main                ← stable, merged code only
member1/indexing    ← Member 1 work
member2/retrieval   ← Member 2 work
member3/agent       ← Member 3 work
member4/case-engine ← Member 4 work
```

Each member works on their branch and opens a Pull Request to `main`.  
Never push directly to `main`.

---

## Insider Threat Scenarios (from CERT r4.2)

The dataset contains 5 insider threat scenarios your system must detect:

| # | Type | Key Signals |
|---|------|-------------|
| 1 | Data exfiltration via USB + Wikileaks | after-hours login, USB spike, wikileaks in http |
| 2 | Job switching + data theft | job sites in http, increased downloads, USB usage |
| 3 | Malicious admin (keylogger) | unusual file access, impersonation, mass email |
| 4 | Insider snooping | login to other machines, external email |
| 5 | Layoff revenge via Dropbox | dropbox in http, bulk file uploads |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Recall@K | % of relevant evidence retrieved |
| Precision@K | % of retrieved evidence that is relevant |
| Grounding Score | % of answers backed by cited evidence (doc_id + span) |
| Tool Accuracy | % of correct tool call decisions by agent |

**Baseline:** plain FAISS → LLM answer (no tools, no preference tuning)  
**Improved:** PageIndex + FAISS + Reranker + Tool Policy Agent + Preference Alignment

---

## Ground Truth

```
D:/dl_proj/answers/insiders.csv   ← malicious users + time ranges
D:/dl_proj/answers/scenarios.txt  ← scenario descriptions
```

Use `insiders.csv` to evaluate whether your system correctly identifies
the malicious users defined in each scenario.

