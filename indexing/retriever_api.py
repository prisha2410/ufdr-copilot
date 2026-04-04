"""
indexing/retriever_api.py
--------------------------
Core retrieval logic — optimized for Render free tier (512MB RAM).

Strategy:
- 4 index maps load at startup (~110MB total) → stay in RAM
- Records are read from JSONL files on demand → zero extra RAM
- FAISS loads lazily on first /vector request
- No record_store.pkl ever loaded (saves 1.2GB RAM)

Trade-off: slightly slower per request (~1-3 sec) but never OOMs.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import pickle
import numpy as np
from functools import lru_cache
from typing import Optional

# ─────────────────────────────────────────────
# PATHS — read directly from env vars
# ─────────────────────────────────────────────
PAGEINDEX_STORE = os.getenv("PAGEINDEX_STORE", "D:/dl_proj/pageindex_store")
PAGEINDEX_DIR   = os.getenv("PAGEINDEX_DIR",   "D:/dl_proj/pageindex_data")
FAISS_DIR       = os.getenv("FAISS_DIR",       "D:/dl_proj/faiss_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Map page_id prefix → JSONL filename
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
# LAZY LOADERS — index maps only (small)
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
# DISK-BASED RECORD LOOKUP (no RAM overhead)
# ─────────────────────────────────────────────

def _get_prefix(page_id: str) -> str:
    """Extract event type prefix from page_id e.g. 'email' from 'email_000123'."""
    return page_id.split("_")[0] if "_" in page_id else ""


def _fetch_records_from_disk(page_ids: set) -> dict:
    """
    Read records for given page_ids from JSONL files.
    Groups by source file for efficient single-pass scanning.
    Returns dict of {page_id: record}.
    """
    by_file = {}
    for pid in page_ids:
        prefix = _get_prefix(pid)
        fname = _PREFIX_TO_FILE.get(prefix)
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


def _fetch_single_record(page_id: str) -> Optional[dict]:
    """Fetch a single record by page_id from its JSONL file."""
    prefix = _get_prefix(page_id)
    fname  = _PREFIX_TO_FILE.get(prefix)
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
    Records are read from JSONL files — no record_store RAM needed.
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

    # Must have at least one filter
    if not candidate_sets:
        return []

    # Intersect all filter sets
    result_ids = candidate_sets[0]
    for s in candidate_sets[1:]:
        result_ids = result_ids & s

    # Filter by event_type if provided
    if event_type:
        result_ids = {pid for pid in result_ids if _get_prefix(pid) == event_type}

    # Cap candidates before disk read
    result_ids = set(list(result_ids)[:top_k * 3])

    # Fetch records from disk
    record_map = _fetch_records_from_disk(result_ids)
    records = list(record_map.values())

    # Sort by timestamp descending
    records.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return records[:top_k]


def get_by_vector(
    query_text: str,
    top_k:      int = 20,
    user:       Optional[str] = None,
    event_type: Optional[str] = None,
) -> list[dict]:
    """
    Semantic FAISS search. Records fetched from disk after search.
    """
    import faiss  # noqa

    index, page_id_map = _load_faiss()
    model = _load_embedding_model()

    vec = model.encode([query_text], normalize_embeddings=True).astype(np.float32)

    fetch_k = min(top_k * 5, index.ntotal)
    scores, positions = index.search(vec, fetch_k)

    candidates = []
    for score, pos in zip(scores[0], positions[0]):
        if pos < 0 or pos >= len(page_id_map):
            continue
        pid = page_id_map[pos]
        if event_type and _get_prefix(pid) != event_type:
            continue
        candidates.append((pid, float(score)))

    pids = {pid for pid, _ in candidates}
    record_map = _fetch_records_from_disk(pids)

    score_map = {pid: score for pid, score in candidates}
    results = []
    for pid, rec in record_map.items():
        if user and rec.get("user") != user:
            continue
        results.append({**rec, "score": score_map.get(pid, 0.0)})

    results.sort(key=lambda r: r.get("score", 0), reverse=True)
    return results[:top_k]


def get_by_id(page_id: str) -> Optional[dict]:
    """Single record lookup from JSONL file."""
    return _fetch_single_record(page_id)


def _load_record_store():
    """Stub — not used in disk-based mode."""
    return {}


def warmup():
    """
    Load only the 4 small index maps at startup (~110MB).
    Everything else is disk-based — no OOM risk.
    """
    print(f"  PAGEINDEX_STORE : {PAGEINDEX_STORE}")
    print(f"  PAGEINDEX_DIR   : {PAGEINDEX_DIR}")
    print(f"  FAISS_DIR       : {FAISS_DIR}")

    print("  Loading user_index   (~38MB) ...")
    _load_user_index()
    print("  Loading action_index (~24MB) ...")
    _load_action_index()
    print("  Loading date_index   (~24MB) ...")
    _load_date_index()
    print("  Loading hour_index   (~24MB) ...")
    _load_hour_index()

    print("  Warmup complete — total RAM ~110MB")
    print("  Records served from JSONL files on demand")
    print("  FAISS loads on first /vector request")