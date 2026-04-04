"""
api/server.py
--------------
FastAPI server for Render free tier (512MB RAM).

RAM strategy:
- date_index, hour_index, action_index loaded at startup (~70MB)
- user_index split into per-user JSON files on HuggingFace
- Per-user file downloaded on demand (~100KB per request)
- Records read from JSONL files on disk
- Total startup RAM: ~150MB ✅
"""

import os
import json
import pickle
import tempfile
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
PAGEINDEX_STORE = os.getenv("PAGEINDEX_STORE", "D:/dl_proj/pageindex_store")
PAGEINDEX_DIR   = os.getenv("PAGEINDEX_DIR",   "D:/dl_proj/pageindex_data")
API_PORT        = int(os.getenv("API_PORT", "8000"))
HF_REPO         = "pisha2410/ufdr-indexes"

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
# SMALL INDEXES (loaded at startup ~70MB)
# ─────────────────────────────────────────────
_hour_index   = {}
_date_index   = {}
_action_index = {}

def _load_small_indexes():
    global _hour_index, _date_index, _action_index
    for name, target in [
        ("hour_index.pkl",   "_hour_index"),
        ("date_index.pkl",   "_date_index"),
        ("action_index.pkl", "_action_index"),
    ]:
        try:
            path = os.path.join(PAGEINDEX_STORE, name)
            with open(path, "rb") as f:
                globals()[target] = pickle.load(f)
            print(f"  {name} loaded")
        except Exception as e:
            print(f"  {name} failed: {e}")

# ─────────────────────────────────────────────
# PER-USER INDEX (download on demand from HF)
# ─────────────────────────────────────────────

# Simple in-memory cache for recently used users
_user_cache = {}
_USER_CACHE_MAX = 5

def _get_user_page_ids(user_id: str):
    """Download per-user JSON from HuggingFace and return page_ids."""
    if user_id in _user_cache:
        return _user_cache[user_id]

    # Find which batch this user is in (alphabetical, 1000 per batch)
    # Try all batches until found
    page_ids = None
    for batch_num in range(11):
        batch = f"batch_{batch_num:02d}"
        try:
            local = hf_hub_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                filename=f"user_index_split/{batch}/{user_id}.json",
                cache_dir=tempfile.gettempdir(),
            )
            with open(local, "r") as f:
                page_ids = json.load(f)
            break
        except Exception:
            continue

    if page_ids is None:
        return []

    # Cache with size limit
    if len(_user_cache) >= _USER_CACHE_MAX:
        _user_cache.pop(next(iter(_user_cache)))
    _user_cache[user_id] = page_ids
    return page_ids

# ─────────────────────────────────────────────
# RECORD FETCHING (from JSONL files on disk)
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
        fpath = os.path.join(PAGEINDEX_DIR, fname)
        if not os.path.exists(fpath):
            continue
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
    print("Starting UFDR Retriever Server ...")
    print(f"  PAGEINDEX_STORE : {PAGEINDEX_STORE}")
    print(f"  PAGEINDEX_DIR   : {PAGEINDEX_DIR}")
    _load_small_indexes()
    print("Server ready — startup RAM ~150MB")
    yield

app = FastAPI(
    title="UFDR Forensic Retriever API",
    description="Hybrid PageIndex + FAISS retrieval for forensic evidence.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "UFDR Retriever"}


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

    # User index — download from HuggingFace on demand
    if user:
        page_ids = _get_user_page_ids(user)
        if not page_ids:
            return {"count": 0, "results": [], "message": f"User '{user}' not found"}
        candidate_sets.append(set(page_ids))

    # Small indexes — already in RAM
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

    # Intersect all filter sets
    result_ids = candidate_sets[0]
    for s in candidate_sets[1:]:
        result_ids = result_ids & s

    if event_type:
        result_ids = {pid for pid in result_ids if _get_prefix(pid) == event_type}

    records = _fetch_by_page_ids(result_ids, top_k)
    records.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return {"count": len(records[:top_k]), "results": records[:top_k]}


@app.get("/vector")
def vector_endpoint(query: str = Query(...)):
    return JSONResponse(status_code=503, content={
        "error": "Vector search needs 2GB RAM — unavailable on free tier.",
        "suggestion": "Use /filter for structured queries.",
        "local": "Run server locally for /vector: python api/server.py"
    })


@app.get("/record/{page_id}")
def record_endpoint(page_id: str):
    fname = _PREFIX_TO_FILE.get(_get_prefix(page_id))
    if not fname:
        raise HTTPException(status_code=404, detail="Not found")
    fpath = os.path.join(PAGEINDEX_DIR, fname)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="File not found on server")
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
        "unique_dates":   len(_date_index),
        "unique_actions": list(_action_index.keys()),
        "unique_hours":   len(_hour_index),
        "user_index":     "per-user JSON files on HuggingFace (downloaded on demand)",
        "note":           "startup RAM ~150MB, user queries add ~3-5s for HF download"
    }


@app.get("/http_search")
def http_search_endpoint(
    user:    Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
    date:    Optional[str] = Query(None),
    top_k:   int           = Query(50, ge=1, le=200),
):
    path = os.path.join(PAGEINDEX_DIR, "http_sample.jsonl")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="http_sample.jsonl not found")

    results = []
    with open(path, "r", encoding="utf-8") as f:
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