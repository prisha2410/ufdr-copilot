# UFDR Copilot — Forensic Investigation Support System

A hybrid Agentic RAG system that retrieves forensic evidence, constructs cases, builds knowledge graphs, and generates explainable UFDR-style reports.

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
[Agent] api/server.py  ←  member 1 hosts this
    ↓
[Hybrid Retriever]
 ├── PageIndex (structured filter)   → indexing/retriever_api.py
 └── FAISS     (semantic search)     → indexing/retriever_api.py
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

### 5. Connect to Member 1's API
```bash
# Add to .env (never commit this)
RETRIEVER_HOST=http://MEMBER1_IP:8000
```

### 6. Use the client
```python
from client.retriever_client import RetrieverClient

client = RetrieverClient()

# Structured search
records = client.filter(user="LAP0338", action="connect_device", hour_min=18)

# Semantic search
records = client.vector("employee copying files after hours", top_k=20)

# Single record
rec = client.get_record("email_000123")
```

---

## Pipeline Execution Order (Member 1)

```bash
python data_pipeline/data_pipeline.py      # Step 1 — CSV → JSONL
python data_pipeline/enhance_pageindex.py  # Step 2 — add PageIndex fields
python indexing/build_pageindex.py         # Step 3 — build dict maps
python indexing/build_chroma.py            # Step 4 — embed + persist ChromaDB
python indexing/build_faiss.py             # Step 5 — build FAISS index
python api/server.py                       # Step 6 — start API server
```

---

## Branching Strategy

```
main          ← stable, merged code
member1/indexing
member2/retrieval
member3/agent
member4/case-engine
```

Each member works on their branch and opens a Pull Request to `main`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server liveness |
| GET | `/filter` | Structured PageIndex query |
| GET | `/vector` | Semantic FAISS query |
| GET | `/record/{page_id}` | Single record lookup |
| GET | `/stats` | Index statistics |

Full API docs at: `http://MEMBER1_IP:8000/docs` (FastAPI auto-generated)

---

## Evaluation Metrics

- Recall@K
- Precision@K  
- Grounding Score (% answers backed by cited evidence)
- Tool Accuracy (correct tool call rate)

Baseline: plain FAISS → LLM answer  
Improved: PageIndex + FAISS + Reranker + Tool Policy Agent
