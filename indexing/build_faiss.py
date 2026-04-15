"""
indexing/build_faiss.py
------------------------
STEP 5 of the pipeline.

Builds a single unified FAISS index for ALL event types including http.

Strategy:
- email/logon/device/file/ldap/psychometric → load embeddings from ChromaDB (fast, no re-embedding)
- http.jsonl → embed fresh from JSONL using GPU

Saves:
    faiss_index/events.index      <- FAISS binary index (all event types)
    faiss_index/page_id_map.pkl   <- position -> page_id lookup

Run after build_chroma.py:
    python indexing/build_faiss.py

Environment variables required (set in .env):
    PAGEINDEX_DIR   = path to pageindex_data/ folder
    FAISS_DIR       = path to faiss_index/ folder
    CHROMA_DIR      = path to chroma_store/ folder
    EMBEDDING_MODEL = sentence transformer model name
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
from configs.paths import PAGEINDEX_DIR, FAISS_DIR, CHROMA_DIR, EMBEDDING_MODEL, check_paths

# ─────────────────────────────────────────────
# VALIDATE
# ─────────────────────────────────────────────
if not check_paths():
    sys.exit(1)

os.makedirs(FAISS_DIR, exist_ok=True)

INDEX_FILE = os.path.join(FAISS_DIR, "events.index")
MAP_FILE   = os.path.join(FAISS_DIR, "page_id_map.pkl")

# Files to load from ChromaDB (already embedded)
CHROMA_FILES = [
    "email.jsonl",
    "logon.jsonl",
    "device.jsonl",
    "file.jsonl",
    "ldap.jsonl",
    "psychometric.jsonl",
]

# Files to embed fresh from JSONL
FRESH_FILES = [
    "http.jsonl",
]

COLLECTION  = "ufdr_events"
ENCODE_BATCH = 512
READ_BATCH   = 1_000
DIM          = 384

# ─────────────────────────────────────────────
# DETECT GPU
# ─────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("STEP 5: Build Unified FAISS Index")
print("=" * 60)
print(f"  ChromaDB files : {', '.join(CHROMA_FILES)}")
print(f"  Fresh embed    : {', '.join(FRESH_FILES)}")
print(f"  Device         : {device.upper()}")
if device == "cuda":
    print(f"  GPU            : {torch.cuda.get_device_name(0)}")
print()

all_embeddings = []
all_page_ids   = []

# ─────────────────────────────────────────────
# STEP A: Load from ChromaDB (fast)
# ─────────────────────────────────────────────
print("Loading embeddings from ChromaDB...")

import chromadb
client     = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION)
total_chroma = collection.count()
print(f"  ChromaDB has {total_chroma:,} records")

CHROMA_BATCH = 5000
offset = 0

while offset < total_chroma:
    result = collection.get(
        limit=CHROMA_BATCH,
        offset=offset,
        include=["embeddings", "ids"]
    )
    if not result["ids"]:
        break

    embs = np.array(result["embeddings"], dtype=np.float32)
    ids  = result["ids"]

    all_embeddings.append(embs)
    all_page_ids.extend(ids)
    offset += len(ids)
    print(f"  Loaded {offset:,} / {total_chroma:,} from ChromaDB...", end="\r")

print(f"\n  ChromaDB load complete — {len(all_page_ids):,} records")

# ─────────────────────────────────────────────
# STEP B: Embed http.jsonl fresh (GPU)
# ─────────────────────────────────────────────
print(f"\nEmbedding fresh files using {device.upper()}...")
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL, device=device)
print(f"  Model loaded on {device.upper()}")

for fname in FRESH_FILES:
    path = os.path.join(PAGEINDEX_DIR, fname)
    if not os.path.exists(path):
        print(f"  SKIP (not found): {fname}")
        continue

    # Count lines for progress
    print(f"\n  Counting records in {fname}...")
    total_lines = sum(1 for _ in open(path, "r", encoding="utf-8"))
    print(f"  Total records: {total_lines:,}")

    # Estimate time
    batches = total_lines // READ_BATCH
    secs_per_batch = 2 if device == "cuda" else 20
    est_min = (batches * secs_per_batch) // 60
    print(f"  Estimated time: ~{est_min} minutes on {device.upper()}")
    print(f"\n  Embedding {fname}...")

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
                pct = file_count / total_lines * 100
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

    print(f"\n  {fname} done — {file_count:,} records embedded")

# ─────────────────────────────────────────────
# BUILD FAISS INDEX
# ─────────────────────────────────────────────
print(f"\nBuilding FAISS index...")
print(f"  Total vectors: {len(all_page_ids):,}")

matrix = np.vstack(all_embeddings).astype(np.float32)
faiss.normalize_L2(matrix)

dim   = matrix.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(matrix)
print(f"  Index built — {index.ntotal:,} vectors, dim={dim}")

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
print("\nSaving...")
faiss.write_index(index, INDEX_FILE)
with open(MAP_FILE, "wb") as f:
    pickle.dump(all_page_ids, f)

idx_gb = os.path.getsize(INDEX_FILE) / 1024 / 1024 / 1024
map_mb = os.path.getsize(MAP_FILE) / 1024 / 1024
print(f"  events.index    saved ({idx_gb:.2f} GB)")
print(f"  page_id_map.pkl saved ({map_mb:.1f} MB)")

print(f"\nSTEP 5 COMPLETE — {index.ntotal:,} vectors in unified FAISS index")
print("Includes: email, logon, device, file, ldap, psychometric, http")