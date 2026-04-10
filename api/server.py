"""
api/server.py
--------------
FastAPI server — works both locally and on Render.

Local machine  → full API including /vector (16GB RAM available)
Render free    → /filter, /record, /stats, /http_search only
               → /vector returns 503 (needs 2GB RAM)

Render strategy:
- Only date/hour/action indexes loaded at startup (~70MB)
- JSONL files downloaded from HuggingFace on first request
- Per-user JSON downloaded from HuggingFace on demand
- Total startup RAM: ~100MB ✅ never OOMs on Render
"""

import os
import sys
import json
import pickle
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
from contextlib import asynccontextmanager

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
PAGEINDEX_STORE = os.getenv("PAGEINDEX_STORE", "D:/dl_proj/pageindex_store")
PAGEINDEX_DIR   = os.getenv("PAGEINDEX_DIR",   "D:/dl_proj/pageindex_data")
FAISS_DIR       = os.getenv("FAISS_DIR",       "D:/dl_proj/faiss_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
API_PORT        = int(os.getenv("API_PORT", "8000"))
HF_REPO         = "pisha2410/ufdr-indexes"
BASE_PATH       = os.path.join(os.path.dirname(__file__), "../data")

# Detect if running locally or on Render
IS_LOCAL = os.path.exists(PAGEINDEX_STORE) and os.path.exists(
    os.path.join(PAGEINDEX_STORE, "user_index.pkl")
)

_PREFIX_TO_FILE = {
    "email":        "email.jsonl",
    "logon":        "logon.jsonl",
    "device":       "device.jsonl",
    "file":         "file.jsonl",
    "ldap":         "ldap.jsonl",
    "psychometric": "psychometric.jsonl",
    "http":         "http_sample.jsonl",
}

# ─────────────────────────────────────────────
# INDEX LOADING
# ─────────────────────────────────────────────
_hour_index   = {}
_date_index   = {}
_action_index = {}
_user_index   = None  # only loaded locally

def _load_indexes():
    global _hour_index, _date_index, _action_index, _user_index

    if IS_LOCAL:
        # Local — load all 4 indexes from disk
        print("  Running locally — loading all indexes...")
        for name, target in [
            ("hour_index.pkl",   "_hour_index"),
            ("date_index.pkl",   "_date_index"),
            ("action_index.pkl", "_action_index"),
            ("user_index.pkl",   "_user_index"),
        ]:
            try:
                with open(os.path.join(PAGEINDEX_STORE, name), "rb") as f:
                    globals()[target] = pickle.load(f)
                print(f"  {name} loaded ✅")
            except Exception as e:
                print(f"  {name} failed: {e}")
    else:
        # Render — only load 3 small indexes (~70MB)
        print("  Running on Render — loading small indexes only...")
        for name, target in [
            ("hour_index.pkl",   "_hour_index"),
            ("date_index.pkl",   "_date_index"),
            ("action_index.pkl", "_action_index"),
        ]:
            try:
                with open(os.path.join(PAGEINDEX_STORE, name), "rb") as f:
                    globals()[target] = pickle.load(f)
                print(f"  {name} loaded ✅")
            except Exception as e:
                print(f"  {name} failed: {e}")

# ─────────────────────────────────────────────
# HuggingFace lazy download (Render only)
# ─────────────────────────────────────────────
def _ensure_jsonl(fname: str) -> str:
    fpath = os.path.join(PAGEINDEX_DIR, fname)
    if not os.path.exists(fpath):
        from huggingface_hub import hf_hub_download
        print(f"  Downloading {fname} from HuggingFace...")
        os.makedirs(PAGEINDEX_DIR, exist_ok=True)
        hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename=f"pageindex_data/{fname}",
            local_dir=BASE_PATH,
        )
        print(f"  {fname} ready!")
    return fpath

# ─────────────────────────────────────────────
# USER INDEX
# ─────────────────────────────────────────────
_user_cache = {}
_USER_CACHE_MAX = 5

def _get_user_page_ids(user_id: str):
    if IS_LOCAL and _user_index is not None:
        # Local — instant lookup from RAM
        return _user_index.get(user_id, [])
    else:
        # Render — download per-user JSON from HuggingFace
        if user_id in _user_cache:
            return _user_cache[user_id]
        from huggingface_hub import hf_hub_download
        page_ids = None
        for batch_num in range(11):
            batch = f"batch_{batch_num:02d}"
            try:
                local = hf_hub_download(
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    filename=f"user_index_split/{batch}/{user_id}.json",
                    local_dir=BASE_PATH,
                )
                with open(local, "r") as f:
                    page_ids = json.load(f)
                break
            except Exception:
                continue
        if page_ids is None:
            return []
        if len(_user_cache) >= _USER_CACHE_MAX:
            _user_cache.pop(next(iter(_user_cache)))
        _user_cache[user_id] = page_ids
        return page_ids

# ─────────────────────────────────────────────
# RECORD FETCHING
# ─────────────────────────────────────────────
def _get_prefix(page_id):
    return page_id.split("_")[0] if "_" in page_id else ""

def _fetch_by_page_ids(page_ids, top_k):
    by_file = {}
    for pid in list(page_ids)[:top_k * 3]:
        fname = _PREFIX_TO_FILE.get(_get_prefix(pid))
        if fname:
            by_file.setdefault(fname, set()).add(pid)

    results = {}
    for fname, pids in by_file.items():
        fpath = _ensure_jsonl(fname)
        remaining = set(pids)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not remaining:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    pid = rec.get("page_id")
                    if pid in remaining:
                        results[pid] = rec
                        remaining.discard(pid)
                except Exception:
                    continue
    return list(results.values())

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    mode = "LOCAL (full)" if IS_LOCAL else "RENDER (minimal)"
    print(f"Starting UFDR Retriever Server [{mode}]...")
    print(f"  PAGEINDEX_STORE : {PAGEINDEX_STORE}")
    print(f"  PAGEINDEX_DIR   : {PAGEINDEX_DIR}")
    print(f"  FAISS_DIR       : {FAISS_DIR}")
    _load_indexes()
    print("Server ready!")
    yield

app = FastAPI(
    title="UFDR Forensic Retriever API",
    description="Hybrid PageIndex + FAISS retrieval for forensic evidence.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "UFDR Retriever",
        "mode": "local" if IS_LOCAL else "render",
        "vector_search": "available" if IS_LOCAL else "unavailable (needs 2GB RAM)"
    }


@app.get("/filter")
def filter_endpoint(
    user:       Optional[str] = Query(None),
    action:     Optional[str] = Query(None),
    date:       Optional[str] = Query(None),
    hour_min:   Optional[int] = Query(None, ge=0, le=23),
    hour_max:   Optional[int] = Query(None, ge=0, le=23),
    event_type: Optional[str] = Query(None),
    top_k:      int           = Query(50, ge=1, le=200),
):
    candidate_sets = []

    if user:
        page_ids = _get_user_page_ids(user)
        if not page_ids:
            return {"count": 0, "results": [], "message": f"User '{user}' not found"}
        candidate_sets.append(set(page_ids))

    if date and _date_index:
        candidate_sets.append(set(_date_index.get(date, [])))
    if action and _action_index:
        candidate_sets.append(set(_action_index.get(action, [])))
    if (hour_min is not None or hour_max is not None) and _hour_index:
        hour_pages = set()
        h_min = hour_min if hour_min is not None else 0
        h_max = hour_max if hour_max is not None else 23
        for h in range(h_min, h_max + 1):
            hour_pages.update(_hour_index.get(h, []))
        candidate_sets.append(hour_pages)

    if not candidate_sets:
        return {"count": 0, "results": [], "message": "Provide at least one filter"}

    result_ids = candidate_sets[0]
    for s in candidate_sets[1:]:
        result_ids = result_ids & s

    if event_type:
        result_ids = {pid for pid in result_ids if _get_prefix(pid) == event_type}

    records = _fetch_by_page_ids(result_ids, top_k)
    records.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return {"count": len(records[:top_k]), "results": records[:top_k]}


@app.get("/vector")
def vector_endpoint(
    query:      str           = Query(..., description="Natural language query"),
    top_k:      int           = Query(20, ge=1, le=100),
    user:       Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
):
    if not IS_LOCAL:
        return JSONResponse(status_code=503, content={
            "error": "Vector search needs 2GB RAM — unavailable on Render free tier.",
            "suggestion": "Use /filter for structured queries.",
            "local": "Run server locally: python run_server.py"
        })

    from indexing.retriever_api import get_by_vector
    results = get_by_vector(
        query_text=query, top_k=top_k,
        user=user, event_type=event_type
    )
    return {"count": len(results), "results": results}


@app.get("/record/{page_id}")
def record_endpoint(page_id: str):
    fname = _PREFIX_TO_FILE.get(_get_prefix(page_id))
    if not fname:
        raise HTTPException(status_code=404, detail="Not found")
    fpath = _ensure_jsonl(fname)
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("page_id") == page_id:
                    return rec
            except Exception:
                continue
    raise HTTPException(status_code=404, detail=f"'{page_id}' not found")


@app.get("/stats")
def stats_endpoint():
    return {
        "mode":           "local" if IS_LOCAL else "render",
        "unique_dates":   len(_date_index),
        "unique_actions": list(_action_index.keys()),
        "unique_hours":   len(_hour_index),
        "user_index":     "loaded in RAM" if IS_LOCAL else "per-user JSON on HuggingFace",
        "vector_search":  "available" if IS_LOCAL else "unavailable on free tier",
    }


@app.get("/http_search")
def http_search_endpoint(
    user:    Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
    date:    Optional[str] = Query(None),
    top_k:   int           = Query(50, ge=1, le=200),
):
    fpath = _ensure_jsonl("http_sample.jsonl")
    results = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if user and rec.get("user") != user:
                    continue
                if date and rec.get("date") != date:
                    continue
                if keyword and keyword.lower() not in rec.get("text", "").lower():
                    continue
                results.append(rec)
                if len(results) >= top_k:
                    break
            except Exception:
                continue
    return {"count": len(results), "results": results}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", API_PORT))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)