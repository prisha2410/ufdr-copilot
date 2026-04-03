"""
indexing/retriever_api.py
--------------------------
Core retrieval logic — used by api/server.py.

Member 2 does NOT import this directly; they call the FastAPI endpoints.
But this module is the heart of what every endpoint does.

Two primary functions:
    get_by_filter(user, action, date, hour_range, event_type, top_k)
        → structured PageIndex lookup (zero ML, very fast)

    get_by_vector(query_text, top_k, filters)
        → semantic FAISS search (with optional post-filter)

    get_by_id(page_id)
        → single record lookup from record_store
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pickle
import numpy as np
from functools import lru_cache
from typing import Optional

# ─────────────────────────────────────────────
# PATHS — read directly from env vars
# (bypasses configs/paths.py to ensure Render
#  env vars are always picked up correctly)
# ─────────────────────────────────────────────
PAGEINDEX_STORE = os.getenv("PAGEINDEX_STORE", "D:/dl_proj/pageindex_store")
FAISS_DIR       = os.getenv("FAISS_DIR",       "D:/dl_proj/faiss_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


# ─────────────────────────────────────────────
# LAZY LOADERS  (load once, cache in RAM)
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_user_index():
    with open(os.path.join(PAGEINDEX_STORE, "user_index.pkl"), "rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=1)
def _load_action_index():
    with open(os.path.join(PAGEINDEX_STORE, "action_index.pkl"), "rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=1)
def _load_date_index():
    with open(os.path.join(PAGEINDEX_STORE, "date_index.pkl"), "rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=1)
def _load_hour_index():
    with open(os.path.join(PAGEINDEX_STORE, "hour_index.pkl"), "rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=1)
def _load_record_store():
    with open(os.path.join(PAGEINDEX_STORE, "record_store.pkl"), "rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=1)
def _load_faiss():
    import faiss
    index = faiss.read_index(os.path.join(FAISS_DIR, "events.index"))
    with open(os.path.join(FAISS_DIR, "page_id_map.pkl"), "rb") as f:
        page_id_map = pickle.load(f)
    return index, page_id_map

@lru_cache(maxsize=1)
def _load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def get_by_filter(
    user:        Optional[str] = None,
    action:      Optional[str] = None,
    date:        Optional[str] = None,
    hour_min:    Optional[int] = None,
    hour_max:    Optional[int] = None,
    event_type:  Optional[str] = None,
    top_k:       int           = 50,
) -> list[dict]:
    """
    Structured retrieval using PageIndex maps.
    All filters are AND-ed together.
    Returns a list of full record dicts.
    """
    record_store = _load_record_store()
    candidate_sets = []

    if user:
        idx = _load_user_index()
        candidate_sets.append(set(idx.get(user, [])))

    if action:
        idx = _load_action_index()
        candidate_sets.append(set(idx.get(action, [])))

    if date:
        idx = _load_date_index()
        candidate_sets.append(set(idx.get(date, [])))

    if hour_min is not None or hour_max is not None:
        idx = _load_hour_index()
        hour_pages = set()
        h_min = hour_min if hour_min is not None else 0
        h_max = hour_max if hour_max is not None else 23
        for h in range(h_min, h_max + 1):
            hour_pages.update(idx.get(h, []))
        candidate_sets.append(hour_pages)

    # Intersect all filter sets
    if candidate_sets:
        result_ids = candidate_sets[0]
        for s in candidate_sets[1:]:
            result_ids = result_ids & s
    else:
        result_ids = set(record_store.keys())

    # Apply event_type post-filter
    records = []
    for pid in result_ids:
        rec = record_store.get(pid)
        if rec is None:
            continue
        if event_type and rec.get("event_type") != event_type:
            continue
        records.append(rec)

    # Sort by timestamp descending, cap at top_k
    records.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return records[:top_k]


def get_by_vector(
    query_text: str,
    top_k:      int = 20,
    user:       Optional[str] = None,
    event_type: Optional[str] = None,
) -> list[dict]:
    """
    Semantic retrieval using FAISS.
    Optional user / event_type post-filters applied after vector search.
    Returns a list of full record dicts with an added 'score' field.
    """
    import faiss  # noqa

    index, page_id_map = _load_faiss()
    record_store       = _load_record_store()
    model              = _load_embedding_model()

    # Embed query
    vec = model.encode([query_text], normalize_embeddings=True).astype(np.float32)

    # Search — fetch extra candidates to absorb post-filter losses
    fetch_k = min(top_k * 5, index.ntotal)
    scores, positions = index.search(vec, fetch_k)

    results = []
    for score, pos in zip(scores[0], positions[0]):
        if pos < 0 or pos >= len(page_id_map):
            continue
        pid = page_id_map[pos]
        rec = record_store.get(pid)
        if rec is None:
            continue
        if user and rec.get("user") != user:
            continue
        if event_type and rec.get("event_type") != event_type:
            continue
        results.append({**rec, "score": float(score)})
        if len(results) >= top_k:
            break

    return results


def get_by_id(page_id: str) -> Optional[dict]:
    """Exact record lookup by page_id."""
    return _load_record_store().get(page_id)


def warmup():
    """Pre-load all indexes into RAM. Call this on server startup."""
    print(f"  PAGEINDEX_STORE: {PAGEINDEX_STORE}")
    print(f"  FAISS_DIR: {FAISS_DIR}")
    print("  Warming up PageIndex maps ...")
    _load_user_index()
    _load_action_index()
    _load_date_index()
    _load_hour_index()
    _load_record_store()
    print("  Warming up FAISS index ...")
    _load_faiss()
    print("  Warming up embedding model ...")
    _load_embedding_model()
    print("  Warmup complete")