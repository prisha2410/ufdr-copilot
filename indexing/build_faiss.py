"""
indexing/build_faiss.py
------------------------
STEP 5 of the pipeline.

Builds TWO FAISS indexes in one run:
    1. events.index      - email/logon/device/file/ldap/psychometric
                           IndexFlatIP (exact search, 2.4GB)
    2. http_events.index - full http.jsonl (10.8M records)
                           IndexIVFFlat (approximate, ~4GB, fits in RAM)

Saves:
    faiss_index/events.index          <- main FAISS index (exact)
    faiss_index/page_id_map.pkl       <- position -> page_id for events
    faiss_index/http_events.index     <- http FAISS index (approximate IVF)
    faiss_index/http_page_id_map.pkl  <- position -> page_id for http

Run after build_pageindex.py:
    python indexing/build_faiss.py

Environment variables (set in .env):
    PAGEINDEX_DIR   = path to pageindex_data/ folder
    FAISS_DIR       = path to faiss_index/ folder
    EMBEDDING_MODEL = sentence transformer model name

Estimated time on RTX 3050:
    events index (1.67M records)  -> ~1 hour
    http index   (10.8M records)  -> ~3-4 hours
    Total                         -> ~4-5 hours
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

EVENTS_FILES = [
    "email.jsonl",
    "logon.jsonl",
    "device.jsonl",
    "file.jsonl",
    "ldap.jsonl",
    "psychometric.jsonl",
]

HTTP_FILE = "http.jsonl"

ENCODE_BATCH = 1_000
READ_BATCH   = 1_000
DIM          = 384

# IVF config for http
# nlist = number of clusters (more = faster search, less accurate)
# nprobe = clusters to search at query time (more = more accurate, slower)
HTTP_NLIST  = 4096   # good for 10M+ records
HTTP_NPROBE = 64     # search 64 clusters per query

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
print(f"  Events index : IndexFlatIP (exact search)")
print(f"  HTTP index   : IndexIVFFlat (approximate, fits in 16GB RAM)")
print()

# ─────────────────────────────────────────────
# LOAD MODEL ONCE
# ─────────────────────────────────────────────
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL, device=device)
print(f"  Model loaded on {device.upper()}\n")


# ─────────────────────────────────────────────
# HELPER — embed and add to index in batches
# ─────────────────────────────────────────────
def embed_files(files, index, label):
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

    return all_page_ids, grand_total


# ─────────────────────────────────────────────
# BUILD EVENTS INDEX (FlatIP — exact search)
# ─────────────────────────────────────────────
events_index_file = os.path.join(FAISS_DIR, "events.index")
events_map_file   = os.path.join(FAISS_DIR, "page_id_map.pkl")

if os.path.exists(events_index_file):
    print("events.index already exists — skipping!")
    print("Delete events.index to rebuild.\n")
    events_total = faiss.read_index(events_index_file).ntotal
else:
    print("=" * 60)
    print("Building: Events Index (FlatIP exact search)")
    print("=" * 60)

    events_index = faiss.IndexFlatIP(DIM)
    events_pids, events_total = embed_files(EVENTS_FILES, events_index, "Events")

    print(f"\n  Saving events.index...")
    faiss.write_index(events_index, events_index_file)
    with open(events_map_file, "wb") as f:
        pickle.dump(events_pids, f)

    idx_gb = os.path.getsize(events_index_file) / 1024 / 1024 / 1024
    map_mb = os.path.getsize(events_map_file) / 1024 / 1024
    print(f"  events.index    saved ({idx_gb:.2f} GB)")
    print(f"  page_id_map.pkl saved ({map_mb:.1f} MB)")
    print(f"  Total: {events_total:,} vectors\n")
    del events_index  # free RAM before building http


# ─────────────────────────────────────────────
# BUILD HTTP INDEX (IVFFlat — fits in 16GB RAM)
# ─────────────────────────────────────────────
http_index_file = os.path.join(FAISS_DIR, "http_events.index")
http_map_file   = os.path.join(FAISS_DIR, "http_page_id_map.pkl")
http_path       = os.path.join(PAGEINDEX_DIR, HTTP_FILE)

if os.path.exists(http_index_file):
    print("http_events.index already exists — skipping!")
    print("Delete http_events.index to rebuild.\n")
    http_total = faiss.read_index(http_index_file).ntotal
elif not os.path.exists(http_path):
    print(f"WARNING: {HTTP_FILE} not found — http index skipped.")
    http_total = 0
else:
    print("=" * 60)
    print("Building: HTTP Index (IVFFlat approximate search)")
    print(f"  nlist  = {HTTP_NLIST} clusters")
    print(f"  nprobe = {HTTP_NPROBE} at query time")
    print(f"  RAM when loaded: ~4GB (fits in 16GB)")
    print("=" * 60)

    # Count http records for training
    print("\nCounting http records...")
    http_count = sum(1 for _ in open(http_path, "r", encoding="utf-8"))
    print(f"  Total: {http_count:,} records")

    # IVF needs training data first
    # Sample 10x nlist records for training (minimum requirement)
    train_size = min(HTTP_NLIST * 40, http_count)
    print(f"\nCollecting {train_size:,} records for IVF training...")

    train_texts = []
    with open(http_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(train_texts) >= train_size:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec  = json.loads(line)
                text = rec.get("normalized_text") or rec.get("text", "")
                if text:
                    train_texts.append(text)
            except Exception:
                continue

    print(f"  Encoding training vectors...")
    train_vecs = model.encode(
        train_texts,
        batch_size=ENCODE_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    faiss.normalize_L2(train_vecs)

    # Build IVF index
    quantizer  = faiss.IndexFlatIP(DIM)
    http_index = faiss.IndexIVFFlat(quantizer, DIM, HTTP_NLIST, faiss.METRIC_INNER_PRODUCT)

    print(f"\n  Training IVF index on {len(train_vecs):,} vectors...")
    http_index.train(train_vecs)
    http_index.nprobe = HTTP_NPROBE
    print(f"  Training complete!")
    del train_vecs, train_texts

    # Now add all records
    print(f"\n  Adding all {http_count:,} records to index...")
    http_pids, http_total = embed_files([HTTP_FILE], http_index, "HTTP")

    print(f"\n  Saving http_events.index...")
    faiss.write_index(http_index, http_index_file)
    with open(http_map_file, "wb") as f:
        pickle.dump(http_pids, f)

    idx_gb = os.path.getsize(http_index_file) / 1024 / 1024 / 1024
    map_mb = os.path.getsize(http_map_file) / 1024 / 1024
    print(f"  http_events.index    saved ({idx_gb:.2f} GB)")
    print(f"  http_page_id_map.pkl saved ({map_mb:.1f} MB)")
    print(f"  Total: {http_total:,} vectors")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 COMPLETE")
print("=" * 60)
print(f"  events.index      : {events_total:,} vectors (exact search)")
print(f"  http_events.index : {http_total:,} vectors (approximate IVF)")
print()
print("Usage:")
print("  from indexing.retriever_api import get_by_vector")
print("  get_by_vector('suspicious file copy')               # events")
print("  get_by_vector('wikileaks upload', search_http=True) # http")