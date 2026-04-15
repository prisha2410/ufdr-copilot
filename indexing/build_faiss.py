"""
indexing/build_faiss.py
------------------------
STEP 5 of the pipeline.

Builds TWO FAISS indexes in one run:
    1. events.index      - email/logon/device/file/ldap/psychometric (~1.67M records)
    2. http_events.index - http browsing logs (sample or full)

Saves:
    faiss_index/events.index          <- main FAISS index
    faiss_index/page_id_map.pkl       <- position -> page_id for events
    faiss_index/http_events.index     <- http FAISS index
    faiss_index/http_page_id_map.pkl  <- position -> page_id for http

Run after build_pageindex.py:
    python indexing/build_faiss.py

Environment variables (set in .env):
    PAGEINDEX_DIR   = path to pageindex_data/ folder
    FAISS_DIR       = path to faiss_index/ folder
    EMBEDDING_MODEL = sentence transformer model name
    USE_FULL_HTTP   = false (default, uses http_sample.jsonl ~1.08M records)
                      true  (uses full http.jsonl ~10.8M records)

Estimated time on RTX 3050:
    events index (1.67M records)        -> ~1 hour
    http_sample index (1.08M records)   -> ~20-30 minutes
    Total                               -> ~1.5 hours
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import json
import pickle
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from configs.paths import PAGEINDEX_DIR, FAISS_DIR, EMBEDDING_MODEL, check_paths

# ─────────────────────────────────────────────
# VALIDATE
# ─────────────────────────────────────────────
if not check_paths():
    sys.exit(1)

os.makedirs(FAISS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
USE_FULL_HTTP = os.getenv("USE_FULL_HTTP", "false").lower() == "true"
HTTP_FILE     = "http.jsonl" if USE_FULL_HTTP else "http_sample.jsonl"

EVENTS_FILES = [
    "email.jsonl",
    "logon.jsonl",
    "device.jsonl",
    "file.jsonl",
    "ldap.jsonl",
    "psychometric.jsonl",
]

ENCODE_BATCH = 1_000
READ_BATCH   = 1_000
DIM          = 384

# ─────────────────────────────────────────────
# DETECT GPU
# ─────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("STEP 5: Build FAISS Indexes (events + http)")
print("=" * 60)
print(f"  Source       : {PAGEINDEX_DIR}")
print(f"  Output       : {FAISS_DIR}")
print(f"  Model        : {EMBEDDING_MODEL}")
print(f"  Device       : {device.upper()}")
if device == "cuda":
    print(f"  GPU          : {torch.cuda.get_device_name(0)}")
print(f"  HTTP file    : {HTTP_FILE}")
print()

# ─────────────────────────────────────────────
# LOAD MODEL ONCE (reused for both indexes)
# ─────────────────────────────────────────────
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL, device=device)
print(f"  Model loaded on {device.upper()}\n")


# ─────────────────────────────────────────────
# HELPER — build index from list of JSONL files
# ─────────────────────────────────────────────
def build_index(files, index_file, map_file, label):
    print("=" * 60)
    print(f"Building: {label}")
    print("=" * 60)

    # Count
    file_counts = {}
    for fname in files:
        path = os.path.join(PAGEINDEX_DIR, fname)
        if os.path.exists(path):
            count = sum(1 for _ in open(path, "r", encoding="utf-8"))
            file_counts[fname] = count
            print(f"  {fname}: {count:,}")
        else:
            print(f"  {fname}: NOT FOUND - skipping")

    total = sum(file_counts.values())
    secs  = 2 if device == "cuda" else 20
    est   = (total // READ_BATCH * secs) // 60
    print(f"\n  Total   : {total:,} records")
    print(f"  Est time: ~{est} minutes\n")

    index        = faiss.IndexFlatIP(DIM)
    all_page_ids = []
    grand_total  = 0

    for fname in files:
        path = os.path.join(PAGEINDEX_DIR, fname)
        if not os.path.exists(path):
            continue

        total_lines = file_counts.get(fname, 0)
        print(f"  Embedding {fname}...")

        batch_texts = []
        batch_ids   = []
        file_count  = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    text   = record.get("normalized_text") or record.get("text", "")
                    pid    = record.get("page_id")
                    if not pid:
                        continue
                    batch_texts.append(text)
                    batch_ids.append(pid)
                except Exception:
                    continue

                if len(batch_texts) >= READ_BATCH:
                    embs = model.encode(
                        batch_texts,
                        batch_size=ENCODE_BATCH,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    ).astype(np.float32)
                    faiss.normalize_L2(embs)
                    index.add(embs)
                    all_page_ids.extend(batch_ids)
                    file_count += len(batch_ids)
                    pct = file_count / total_lines * 100 if total_lines else 0
                    print(f"    {file_count:,} / {total_lines:,} ({pct:.1f}%)", end="\r")
                    batch_texts = []
                    batch_ids   = []

        if batch_texts:
            embs = model.encode(
                batch_texts,
                batch_size=ENCODE_BATCH,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)
            faiss.normalize_L2(embs)
            index.add(embs)
            all_page_ids.extend(batch_ids)
            file_count += len(batch_ids)

        print(f"\n    {fname} done — {file_count:,} records")
        grand_total += file_count

    # Save
    print(f"\n  Saving {os.path.basename(index_file)}...")
    faiss.write_index(index, index_file)
    with open(map_file, "wb") as f:
        pickle.dump(all_page_ids, f)

    idx_gb = os.path.getsize(index_file) / 1024 / 1024 / 1024
    map_mb = os.path.getsize(map_file) / 1024 / 1024
    print(f"  {os.path.basename(index_file)} saved ({idx_gb:.2f} GB)")
    print(f"  {os.path.basename(map_file)} saved ({map_mb:.1f} MB)")
    print(f"  Total: {index.ntotal:,} vectors\n")

    return index.ntotal


# ─────────────────────────────────────────────
# BUILD EVENTS INDEX
# ─────────────────────────────────────────────
events_total = build_index(
    files      = EVENTS_FILES,
    index_file = os.path.join(FAISS_DIR, "events.index"),
    map_file   = os.path.join(FAISS_DIR, "page_id_map.pkl"),
    label      = "Events Index (email/logon/device/file/ldap/psychometric)"
)

# ─────────────────────────────────────────────
# BUILD HTTP INDEX
# ─────────────────────────────────────────────
http_path = os.path.join(PAGEINDEX_DIR, HTTP_FILE)

if os.path.exists(http_path):
    http_total = build_index(
        files      = [HTTP_FILE],
        index_file = os.path.join(FAISS_DIR, "http_events.index"),
        map_file   = os.path.join(FAISS_DIR, "http_page_id_map.pkl"),
        label      = f"HTTP Index ({HTTP_FILE})"
    )
else:
    print(f"WARNING: {HTTP_FILE} not found in {PAGEINDEX_DIR}")
    print("HTTP index skipped.")
    print("Create http_sample.jsonl or set USE_FULL_HTTP=true in .env")
    http_total = 0

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 5 COMPLETE")
print("=" * 60)
print(f"  events.index      : {events_total:,} vectors")
print(f"  http_events.index : {http_total:,} vectors")
print()
print("Usage:")
print("  from indexing.retriever_api import get_by_vector")
print("  get_by_vector('suspicious file copy')               # events")
print("  get_by_vector('wikileaks upload', search_http=True) # http")