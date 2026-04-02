"""
indexing/build_faiss.py
------------------------
STEP 5 of Member 1's pipeline.

Loads all embeddings from ChromaDB (so they are never recomputed),
builds a FAISS flat index, and saves:

    faiss_index/events.index      ← FAISS binary index
    faiss_index/page_id_map.pkl   ← position → page_id lookup

Run after build_chroma.py:
    python indexing/build_faiss.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pickle
import numpy as np
import chromadb
from configs.paths import CHROMA_DIR, FAISS_DIR

COLLECTION  = "ufdr_events"
FETCH_BATCH = 5_000
INDEX_FILE  = os.path.join(FAISS_DIR, "events.index")
MAP_FILE    = os.path.join(FAISS_DIR, "page_id_map.pkl")


def load_all_embeddings_from_chroma(collection) -> tuple:
    """
    Pull every embedding + page_id from ChromaDB in batches.
    Returns (embeddings_matrix, page_id_list).
    """
    total    = collection.count()
    all_embs = []
    all_ids  = []

    print(f"  Fetching {total:,} embeddings from ChromaDB ...")

    offset = 0
    while offset < total:
        result = collection.get(
            limit=FETCH_BATCH,
            offset=offset,
            include=["embeddings"],
        )
        embs     = result["embeddings"]
        page_ids = result["ids"]

        all_embs.extend(embs)
        all_ids.extend(page_ids)
        offset += len(page_ids)
        print(f"    Fetched {offset:,} / {total:,}", end="\r")

    print(f"\n  Done — {len(all_embs):,} embeddings loaded")
    return np.array(all_embs, dtype=np.float32), all_ids


def build_faiss_index(embeddings: np.ndarray):
    import faiss

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)    # Inner-product (cosine after normalisation)

    # L2-normalise for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    print(f"  Index built — {index.ntotal:,} vectors, dim={dim}")
    return index


if __name__ == "__main__":
    import faiss  # fail fast if not installed

    print("=" * 50)
    print("STEP 5: Build FAISS Index (from ChromaDB embeddings)")
    print(f"  ChromaDB : {CHROMA_DIR}")
    print(f"  Output   : {FAISS_DIR}")
    print("=" * 50)

    # ── Connect to ChromaDB ───────────────────────────────────────────
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION)
    print(f"  ChromaDB collection '{COLLECTION}': {collection.count():,} docs")

    # ── Load embeddings ───────────────────────────────────────────────
    embeddings, page_id_map = load_all_embeddings_from_chroma(collection)

    # ── Build FAISS ───────────────────────────────────────────────────
    index = build_faiss_index(embeddings)

    # ── Save ──────────────────────────────────────────────────────────
    faiss.write_index(index, INDEX_FILE)
    with open(MAP_FILE, "wb") as f:
        pickle.dump(page_id_map, f)

    idx_mb = os.path.getsize(INDEX_FILE) / 1_048_576
    map_mb = os.path.getsize(MAP_FILE)   / 1_048_576
    print(f"\n  Saved events.index    ({idx_mb:.1f} MB)")
    print(f"  Saved page_id_map.pkl ({map_mb:.1f} MB)")
    print(f"\n✅  STEP 5 COMPLETE — FAISS index at {FAISS_DIR}")