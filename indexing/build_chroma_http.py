"""
indexing/build_chroma_http.py
------------------------------
Optional script to embed http.jsonl separately.
Run this only if you need semantic search on browsing logs.
Takes 6-8 hours on CPU.

Run:
    python indexing/build_chroma_http.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import chromadb
from sentence_transformers import SentenceTransformer
from configs.paths import PAGEINDEX_DIR, CHROMA_DIR, EMBEDDING_MODEL

BATCH_SIZE = 512
COLLECTION = "ufdr_http"    # separate collection, doesn't touch ufdr_events


def build_chroma_http():
    print("Loading embedding model:", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to ChromaDB at:", CHROMA_DIR)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Collection '{COLLECTION}' ready (existing: {collection.count()})")

    path = os.path.join(PAGEINDEX_DIR, "http.jsonl")
    if not os.path.exists(path):
        print("ERROR: http.jsonl not found at", path)
        return

    # Stream file — don't load all into memory
    batch_records = []
    total = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                batch_records.append(json.loads(line))
            except Exception:
                continue

            if len(batch_records) >= BATCH_SIZE:
                _upsert_batch(batch_records, collection, model)
                total += len(batch_records)
                print(f"  Upserted {total:,} records", end="\r")
                batch_records = []

    # Last batch
    if batch_records:
        _upsert_batch(batch_records, collection, model)
        total += len(batch_records)

    print(f"\n  Total: {total:,} records")
    print(f"✅ DONE — http embeddings in ChromaDB collection '{COLLECTION}'")


def _upsert_batch(records, collection, model):
    texts     = [r.get("normalized_text") or r.get("text", "") for r in records]
    ids       = [r["page_id"] for r in records]
    metadatas = [{
        "page_id":    str(r.get("page_id", "")),
        "user":       str(r.get("user", "")),
        "date":       str(r.get("date") or ""),
        "hour":       int(r.get("hour") or -1),
        "url":        str(r.get("entities", {}).get("urls", [""])[0]),
    } for r in records]

    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=False
    ).tolist()

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )


if __name__ == "__main__":
    print("=" * 50)
    print("OPTIONAL: Build ChromaDB for http.jsonl")
    print(f"  Source  : {PAGEINDEX_DIR}/http.jsonl")
    print(f"  ChromaDB: {CHROMA_DIR}")
    print(f"  WARNING : This takes 6-8 hours on CPU")
    print("=" * 50)
    build_chroma_http()

    