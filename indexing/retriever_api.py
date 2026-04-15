"""
indexing/retriever_api.py
--------------------------
Core local retrieval logic for UFDR Copilot.

All team members import from this file directly.
No server, no HTTP — pure local Python.

Usage:
    from indexing.retriever_api import get_by_filter, get_by_vector, get_by_id

Environment variables required (set in .env):
    PAGEINDEX_STORE = path to pageindex_store/ folder
    PAGEINDEX_DIR   = path to pageindex_data/ folder
    FAISS_DIR       = path to faiss_index/ folder
"""

import os
import sys
import json
import pickle
import numpy as np
from functools import lru_cache
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from configs.paths import PAGEINDEX_STORE, PAGEINDEX_DIR, FAISS_DIR, EMBEDDING_MODEL

# ─────────────────────────────────────────────
# FILE MAP
# ─────────────────────────────────────────────
_PREFIX_TO_FILE = {
    "email":        "email.jsonl",
    "logon":        "logon.jsonl",
    "device":       "device.jsonl",
    "file":         "file.jsonl",
    "ldap":         "ldap.jsonl",
    "psychometric": "psychometric.jsonl",
    "http":         "http.jsonl",
}

# ─────────────────────────────────────────────
# LAZY LOADERS
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
def _load_faiss(index_name: str = "events"):
    import faiss
    index = faiss.read_index(os.path.join(FAISS_DIR, f"{index_name}.index"))
    with open(os.path.join(FAISS_DIR, f"{index_name.replace('events', 'page_id_map')}.pkl"), "rb") as f:
        page_id_map = pickle.load(f)
    return index, page_id_map

@lru_cache(maxsize=1)
def _load_http_faiss():
    import faiss
    http_index_path = os.path.join(FAISS_DIR, "http_events.index")
    http_map_path   = os.path.join(FAISS_DIR, "http_page_id_map.pkl")
    if not os.path.exists(http_index_path):
        return None, None
    index = faiss.read_index(http_index_path)
    with open(http_map_path, "rb") as f:
        page_id_map = pickle.load(f)
    return index, page_id_map

@lru_cache(maxsize=1)
def _load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _get_prefix(page_id: str) -> str:
    return page_id.split("_")[0] if "_" in page_id else ""

def _fetch_records_from_disk(page_ids: set) -> dict:
    """Read records for given page_ids from JSONL files."""
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
    use_record_store: bool     = True,
) -> list[dict]:
    """
    Structured retrieval using PageIndex maps.

    Args:
        user:       e.g. "LAP0338"
        action:     e.g. "connect_device", "send_email", "file_copy", "login"
        date:       e.g. "2010-02-01"
        hour_min:   e.g. 18 (after 6 PM)
        hour_max:   e.g. 23
        event_type: e.g. "email", "logon", "device", "file"
        top_k:      max records to return
        use_record_store: True = fast (RAM), False = disk-based (low memory)

    Returns:
        list of record dicts

    Examples:
        get_by_filter(user="LAP0338", action="connect_device", hour_min=18)
        get_by_filter(date="2010-02-01", event_type="email")
    """
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

    if not candidate_sets:
        return []

    result_ids = candidate_sets[0]
    for s in candidate_sets[1:]:
        result_ids = result_ids & s

    if event_type:
        result_ids = {pid for pid in result_ids if _get_prefix(pid) == event_type}

    if use_record_store:
        record_store = _load_record_store()
        records = []
        for pid in result_ids:
            rec = record_store.get(pid)
            if rec:
                records.append(rec)
    else:
        record_map = _fetch_records_from_disk(set(list(result_ids)[:top_k * 3]))
        records = list(record_map.values())

    records.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return records[:top_k]


def get_by_vector(
    query_text:  str,
    top_k:       int           = 20,
    user:        Optional[str] = None,
    event_type:  Optional[str] = None,
    search_http: bool          = False,
) -> list[dict]:
    """
    Semantic FAISS search using natural language.

    Args:
        query_text:  natural language query
        top_k:       max records to return
        user:        optional post-filter by user
        event_type:  optional post-filter by event type
        search_http: True = search http.jsonl FAISS index

    Returns:
        list of record dicts with 'score' field

    Examples:
        get_by_vector("employee copying files before resignation")
        get_by_vector("suspicious USB after hours", user="LAP0338")
        get_by_vector("wikileaks upload", search_http=True)
    """
    import faiss

    if search_http:
        index, page_id_map = _load_http_faiss()
        if index is None:
            print("HTTP FAISS index not found. Run build_faiss_http.py first.")
            return []
    else:
        index, page_id_map = _load_faiss("events")

    record_store = _load_record_store()
    model        = _load_embedding_model()

    vec = model.encode([query_text], normalize_embeddings=True).astype(np.float32)
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
    """Get a single record by page_id."""
    return _load_record_store().get(page_id)


def get_user_timeline(user: str, top_k: int = 200) -> list[dict]:
    """
    Get full activity timeline for a user sorted by timestamp.
    Useful for building case reports.
    """
    records = get_by_filter(user=user, top_k=top_k)
    records.sort(key=lambda r: r.get("timestamp") or "")
    return records


def warmup():
    """Pre-load all indexes into RAM. Call at startup for faster first queries."""
    print("Loading PageIndex maps...")
    _load_user_index()
    _load_action_index()
    _load_date_index()
    _load_hour_index()
    print("Loading record store...")
    _load_record_store()
    print("Loading FAISS index...")
    _load_faiss("events")
    print("Loading embedding model...")
    _load_embedding_model()
    print("Warmup complete!")