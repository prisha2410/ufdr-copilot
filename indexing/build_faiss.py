"""
indexing/build_faiss.py
------------------------
STEP 5 of the pipeline.

Builds FAISS index directly from PageIndex JSONL files.
Uses GPU for embedding if available (much faster).

Saves:
    faiss_index/events.index      ← FAISS binary index
    faiss_index/page_id_map.pkl   ← position → page_id lookup

Run after build_chroma.py:
    python indexing/build_faiss.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import pickle
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from configs.paths import PAGEINDEX_DIR, FAISS_DIR, EMBEDDING_MODEL

JSONL_FILES = [
    "email.jsonl",
    "logon.jsonl",
    "device.jsonl",
    "file.jsonl",
    "ldap.jsonl",
    "psychometric.jsonl",
    "http.jsonl",
]

INDEX_FILE = os.path.join(FAISS_DIR, "events.index")
MAP_FILE   = os.path.join(FAISS_DIR, "page_id_map.pkl")


def build_faiss():
    # ── Device detection ──────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Batch size — bigger on GPU ────────────────────────────────────
    ENCODE_BATCH = 512 if device == "cpu" else 1_000
    READ_BATCH   = 1_000

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    all_embeddings = []
    all_page_ids   = []

    for fname in JSONL_FILES:
        path = os.path.join(PAGEINDEX_DIR, fname)
        if not os.path.exists(path):
            print(f"  SKIP (not found): {fname}")
            continue

        print(f"\n  Processing {fname} ...", flush=True)
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
                    )
                    all_embeddings.append(embs)
                    all_page_ids.extend(batch_ids)
                    file_count += len(batch_ids)
                    print(f"    Encoded {file_count:,}", end="\r")
                    batch_texts = []
                    batch_ids   = []

        # Last batch
        if batch_texts:
            embs = model.encode(
                batch_texts,
                batch_size=ENCODE_BATCH,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            all_embeddings.append(embs)
            all_page_ids.extend(batch_ids)
            file_count += len(batch_ids)

        print(f"    Done — {file_count:,} records")

    print(f"\n  Total records embedded: {len(all_page_ids):,}")

    # ── Build FAISS index ─────────────────────────────────────────────
    print("  Building FAISS index ...")
    matrix = np.vstack(all_embeddings).astype(np.float32)
    faiss.normalize_L2(matrix)

    dim   = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    print(f"  Index built — {index.ntotal:,} vectors, dim={dim}")

    # ── Save ──────────────────────────────────────────────────────────
    faiss.write_index(index, INDEX_FILE)
    with open(MAP_FILE, "wb") as f:
        pickle.dump(all_page_ids, f)

    idx_mb = os.path.getsize(INDEX_FILE) / 1_048_576
    map_mb = os.path.getsize(MAP_FILE)   / 1_048_576
    print(f"\n  Saved events.index    ({idx_mb:.1f} MB)")
    print(f"  Saved page_id_map.pkl ({map_mb:.1f} MB)")

    return index.ntotal


if __name__ == "__main__":
    print("=" * 50)
    print("STEP 5: Build FAISS Index (from JSONL files)")
    print(f"  Source : {PAGEINDEX_DIR}")
    print(f"  Output : {FAISS_DIR}")
    print(f"  Model  : {EMBEDDING_MODEL}")
    print("=" * 50)

    total = build_faiss()
    print(f"\n✅  STEP 5 COMPLETE — {total:,} vectors in FAISS index")