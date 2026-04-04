"""
api/server.py
--------------
Minimal FastAPI server for Render free tier (512MB RAM).

Available endpoints:
    GET /health       → liveness check
    GET /filter       → structured PageIndex query
    GET /record/{id}  → single record lookup
    GET /stats        → index statistics
    GET /http_search  → browsing log search
    GET /vector       → returns 503 (needs 2GB RAM, use locally)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import pickle
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
from contextlib import asynccontextmanager

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
PAGEINDEX_STORE = os.getenv("PAGEINDEX_STORE", "D:/dl_proj/pageindex_store")
PAGEINDEX_DIR   = os.getenv("PAGEINDEX_DIR",   "D:/dl_proj/pageindex_data")
API_HOST        = os.getenv("API_HOST",        "0.0.0.0")
API_PORT        = int(os.getenv("API_PORT",    "8000"))

# ─────────────────────────────────────────────
# INDEX MAPS (loaded once at startup ~110MB)
# ─────────────────────────────────────────────
_user_index   = None
_action_index = None
_date_index   = None
_hour_index   = None

def _load_indexes():
    global _user_index, _action_index, _date_index, _hour_index
    print(f"  Loading user_index...")
    with open(os.path.join(PAGEINDEX_STORE, "user_index.pkl"), "rb") as f:
        _user_index = pickle.load(f)
    print(f"  Loading action_index...")
    with open(os.path.join(PAGEINDEX_STORE, "action_index.pkl"), "rb") as f:
        _action_index = pickle.load(f)
    print(f"  Loading date_index...")
    with open(os.path.join(PAGEINDEX_STORE, "date_index.pkl"), "rb") as f:
        _date_index = pickle.load(f)
    print(f"  Loading hour_index...")
    with open(os.path.join(PAGEINDEX_STORE, "hour_index.pkl"), "rb") as f:
        _hour_index = pickle.load(f)
    print("  All indexes loaded!")

# ─────────────────────────────────────────────
# DISK-BASED RECORD LOOKUP
# ─────────────────────────────────────────────
_PREFIX_TO_FILE = {
    "email":        "email.jsonl",
    "logon":        "logon.jsonl",
    "device":       "device.jsonl",
    "file":         "file.jsonl",
    "ldap":         "ldap.jsonl",
    "psychometric": "psychometric.jsonl",
    "http":         "http_sample.jsonl",
}

def _get_prefix(page_id):
    return page_id.split("_")[0] if "_" in page_id else ""

def _fetch_records(page_ids):
    by_file = {}
    for pid in page_ids:
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
    return results

def _fetch_one(page_id):
    fname = _PREFIX_TO_FILE.get(_get_prefix(page_id))
    if not fname:
        return None
    fpath = os.path.join(PAGEINDEX_DIR, fname)
    if not os.path.exists(fpath):
        return None
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
    return None

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting UFDR Retriever Server ...")
    print(f"  PAGEINDEX_STORE : {PAGEINDEX_STORE}")
    print(f"  PAGEINDEX_DIR   : {PAGEINDEX_DIR}")
    _load_indexes()
    print("Server ready")
    yield

app = FastAPI(
    title="UFDR Forensic Retriever API",
    description="Hybrid PageIndex retrieval for forensic evidence.",
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
    return {"status": "ok", "service": "UFDR Retriever"}


@app.get("/filter")
def filter_endpoint(
    user:       Optional[str] = Query(None),
    action:     Optional[str] = Query(None),
    date:       Optional[str] = Query(None),
    hour_min:   Optional[int] = Query(None, ge=0, le=23),
    hour_max:   Optional[int] = Query(None, ge=0, le=23),
    event_type: Optional[str] = Query(None),
    top_k:      int           = Query(50, ge=1, le=500),
):
    candidate_sets = []

    if user:
        candidate_sets.append(set(_user_index.get(user, [])))
    if action:
        candidate_sets.append(set(_action_index.get(action, [])))
    if date:
        candidate_sets.append(set(_date_index.get(date, [])))
    if hour_min is not None or hour_max is not None:
        hour_pages = set()
        h_min = hour_min if hour_min is not None else 0
        h_max = hour_max if hour_max is not None else 23
        for h in range(h_min, h_max + 1):
            hour_pages.update(_hour_index.get(h, []))
        candidate_sets.append(hour_pages)

    if not candidate_sets:
        return {"count": 0, "results": [], "error": "Provide at least one filter"}

    result_ids = candidate_sets[0]
    for s in candidate_sets[1:]:
        result_ids = result_ids & s

    if event_type:
        result_ids = {pid for pid in result_ids if _get_prefix(pid) == event_type}

    result_ids = set(list(result_ids)[:top_k * 3])
    record_map = _fetch_records(result_ids)
    records = list(record_map.values())
    records.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return {"count": len(records[:top_k]), "results": records[:top_k]}


@app.get("/vector")
def vector_endpoint(query: str = Query(...)):
    return JSONResponse(
        status_code=503,
        content={
            "error": "Vector search unavailable on free tier (requires 2GB RAM).",
            "suggestion": "Run server locally: python api/server.py",
            "local_docs": "http://localhost:8000/docs"
        }
    )


@app.get("/record/{page_id}")
def record_endpoint(page_id: str):
    rec = _fetch_one(page_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"'{page_id}' not found")
    return rec


@app.get("/stats")
def stats_endpoint():
    return {
        "unique_users":   len(_user_index),
        "unique_actions": list(_action_index.keys()),
        "unique_dates":   len(_date_index),
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


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", API_PORT))
    uvicorn.run("api.server:app", host="0.0.0.0", port=port, reload=False)