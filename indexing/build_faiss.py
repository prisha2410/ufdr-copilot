"""
indexing/build_faiss.py
------------------------
STEP 5 of the pipeline.

Builds a single unified FAISS index for ALL event types including http.
Embeds directly from JSONL files using GPU (no ChromaDB dependency).

Saves:
    faiss_index/events.index      <- FAISS binary index (all event types)
    faiss_index/page_id_map.pkl   <- position -> page_id lookup

Run after build_pageindex.py:
    python indexing/build_faiss.py

Environment variables required (set in .env):
    PAGEINDEX_DIR   = path to pageindex_data/ folder
    FAISS_DIR       = path to faiss_index/ folder
    EMBEDDING_MODEL = sentence transformer model name

Estimated time on RTX 3050:
    email/logon/device/file/ldap/psychometric (~1.67M records) -> ~1 hour
    http.jsonl (~10.8M records)                                -> ~3-4 hours
    Total                                                      -> ~4-5 hours
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

INDEX_FILE = os.path.join(FAISS_DIR, "events.index")
MAP_FILE   = os.path.join(FAISS_DIR, "page_id_map.pkl")

# All event types including http
JSONL_FILES = [
    "email.jsonl",
    "logon.jsonl",
    "device.jsonl",
    "file.jsonl",
    "ldap.jsonl",
    "psychometric.jsonl",
    "http.jsonl",
]

ENCODE_BATCH = 1_000   # bigger batch = faster on GPU
READ_BATCH   = 1_000
DIM          = 384

# ─────────────────────────────────────────────
# DETECT GPU
# ─────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("STEP 5: Build Unified FAISS Index (all event types)")
print("=" * 60)
print(f"  Source  : {PAGEINDEX_DIR}")
print(f"  Output  : {FAISS_DIR}")
print(f"  Model   : {EMBEDDING_MODEL}")
print(f"  Device  : {device.upper()}")
if device == "cuda":
    print(f"  GPU     : {torch.cuda.get_device_name(0)}")
print(f"  Files   : {', '.join(JSONL_FILES)}")
print()

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL, device=device)
print(f"  Model loaded on {device.upper()}")

# ─────────────────────────────────────────────
# COUNT TOTAL FOR ESTIMATE
# ─────────────────────────────────────────────
print("\nCounting records...")
file_counts = {}
for fname in JSONL_FILES:
    path = os.path.join(PAGEINDEX_DIR, fname)
    if os.path.exists(path):
        count = sum(1 for _ in open(path, "r", encoding="utf-8"))
        file_counts[fname] = count
        print(f"  {fname}: {count:,}")
    else:
        print(f"  {fname}: NOT FOUND - will skip")

total_records = sum(file_counts.values())
secs_per_batch = 2 if device == "cuda" else 20
est_min = (total_records // READ_BATCH * secs_per_batch) // 60
print(f"\n  Total records : {total_records:,}")
print(f"  Estimated time: ~{est_min} minutes on {device.upper()}")
print()

# ─────────────────────────────────────────────
# BUILD INDEX
# ─────────────────────────────────────────────
all_embeddings = []
all_page_ids   = []
grand_total    = 0

for fname in JSONL_FILES:
    path = os.path.join(PAGEINDEX_DIR, fname)
    if not os.path.exists(path):
        print(f"  SKIP (not found): {fname}")
        continue

    total_lines = file_counts.get(fname, 0)
    print(f"\n  Embedding {fname} ({total_lines:,} records)...")

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
                all_embeddings.append(embs)
                all_page_ids.extend(batch_ids)
                file_count += len(batch_ids)
                pct = file_count / total_lines * 100 if total_lines else 0
                print(f"    {file_count:,} / {total_lines:,} ({pct:.1f}%)", end="\r")
                batch_texts = []
                batch_ids   = []

    # Final batch
    if batch_texts:
        embs = model.encode(
            batch_texts,
            batch_size=ENCODE_BATCH,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        all_embeddings.append(embs)
        all_page_ids.extend(batch_ids)
        file_count += len(batch_ids)

    print(f"\n    {fname} done — {file_count:,} records")
    grand_total += file_count

print(f"\n  Total embedded: {grand_total:,} records")

# ─────────────────────────────────────────────
# BUILD FAISS INDEX IN BATCHES
# ─────────────────────────────────────────────
print("\nBuilding FAISS index in batches...")

dim   = DIM
index = faiss.IndexFlatIP(dim)

all_page_ids  = []
grand_total   = 0

for fname in JSONL_FILES:
    path = os.path.join(PAGEINDEX_DIR, fname)
    if not os.path.exists(path):
        print(f"  SKIP (not found): {fname}")
        continue

    total_lines = file_counts.get(fname, 0)
    print(f"\n  Embedding {fname} ({total_lines:,} records)...")

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

    # Final batch
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

print(f"\n  Total indexed: {grand_total:,} vectors")

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
print("\nSaving...")
faiss.write_index(index, INDEX_FILE)
with open(MAP_FILE, "wb") as f:
    pickle.dump(all_page_ids, f)