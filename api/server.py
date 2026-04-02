"""
api/server.py
--------------
FastAPI server that exposes the retrieval layer to the entire team.

Member 1 runs this on their machine.
All other members call it via http://MEMBER1_IP:8000

Start the server:
    python api/server.py

Or with uvicorn for auto-reload during development:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET /health                    → liveness check
    GET /filter                    → structured PageIndex query
    GET /vector                    → semantic FAISS query
    GET /record/{page_id}          → single record by page_id
    GET /stats                     → index statistics
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from contextlib import asynccontextmanager

from indexing.retriever_api import get_by_filter, get_by_vector, get_by_id, warmup
from indexing.retriever_api import (
    _load_user_index, _load_action_index, _load_record_store
)
from configs.paths import API_HOST, API_PORT


# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting UFDR Retriever Server ...")
    warmup()
    print("✅ Server ready")
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
    """Liveness check — Member 2/3/4 call this first."""
    return {"status": "ok", "service": "UFDR Retriever"}


@app.get("/filter")
def filter_endpoint(
    user:       Optional[str] = Query(None, description="e.g. LAP0338"),
    action:     Optional[str] = Query(None, description="e.g. connect_device"),
    date:       Optional[str] = Query(None, description="e.g. 2010-02-01"),
    hour_min:   Optional[int] = Query(None, ge=0, le=23),
    hour_max:   Optional[int] = Query(None, ge=0, le=23),
    event_type: Optional[str] = Query(None, description="email|logon|device|file|http"),
    top_k:      int           = Query(50, ge=1, le=500),
):
    """
    Structured retrieval — uses PageIndex maps, no ML.
    Fast and exact. All parameters are ANDed together.

    Example:
        GET /filter?user=LAP0338&action=connect_device&hour_min=18
    """
    results = get_by_filter(
        user=user, action=action, date=date,
        hour_min=hour_min, hour_max=hour_max,
        event_type=event_type, top_k=top_k,
    )
    return {"count": len(results), "results": results}


@app.get("/vector")
def vector_endpoint(
    query:      str           = Query(..., description="Natural language query"),
    top_k:      int           = Query(20, ge=1, le=100),
    user:       Optional[str] = Query(None, description="Post-filter by user"),
    event_type: Optional[str] = Query(None, description="Post-filter by event type"),
):
    """
    Semantic retrieval — FAISS cosine similarity search.
    Each result includes a 'score' field (higher = more similar).

    Example:
        GET /vector?query=employee+copying+files+after+hours&top_k=20
    """
    results = get_by_vector(
        query_text=query, top_k=top_k, user=user, event_type=event_type
    )
    return {"count": len(results), "results": results}


@app.get("/record/{page_id}")
def record_endpoint(page_id: str):
    """
    Fetch a single record by its page_id.

    Example:
        GET /record/email_000123
    """
    rec = get_by_id(page_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"page_id '{page_id}' not found")
    return rec


@app.get("/stats")
def stats_endpoint():
    """Index statistics."""
    user_idx   = _load_user_index()
    action_idx = _load_action_index()
    rec_store  = _load_record_store()
    return {
        "total_records": len(rec_store),
        "unique_users":  len(user_idx),
        "unique_actions": list(action_idx.keys()),
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host=API_HOST, port=API_PORT, reload=False)
