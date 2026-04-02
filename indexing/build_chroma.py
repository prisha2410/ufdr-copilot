"""
indexing/build_chroma.py
-------------------------
STEP 4 of Member 1's pipeline.

Reads all PageIndex-enhanced JSONL files, embeds the normalized_text field
using sentence-transformers, and persists every record into ChromaDB.

ChromaDB is the persistent storage layer — run this ONCE.
build_faiss.py and retriever_api.py load from here so embeddings are
never recomputed.

Run after build_pageindex.py:
    python indexing/build_chroma.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from configs.paths import PAGEINDEX_DIR, CHROMA_DIR, EMBEDDING_MODEL

JSONL_FILES = [
    "email.jsonl",
    "logon.jsonl",
    "device.jsonl",
    "file.jsonl",
    "http.jsonl",
    "ldap.jsonl",
    "psychometric.jsonl",
]

BATCH_SIZE = 512          # records per ChromaDB upsert call
COLLECTION = "ufdr_events"


def load_records_from_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def chroma_metadata(record: dict) -> dict:
    """
    ChromaDB metadata must be flat (str/int/float/bool only).
    We keep the fields Member 2 needs for filtering.
    """
    return {
        "page_id":     str(record.get("page_id", "")),
        "user":        str(record.get("user", "")),
        "event_type":  str(record.get("event_type", "")),
        "action":      str(record.get("action", "")),
        "date":        str(record.get("date") or ""),
        "hour":        int(record.get("hour") or -1),
        "source_file": str(record.get("source_file", "")),
        "line_number": int(record.get("line_number") or 0),
        "pc":          str(record.get("pc") or ""),
    }


def build_chroma():
    print("Loading embedding model:", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to ChromaDB at:", CHROMA_DIR)
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMA_DIR,
        anonymized_telemetry=False,
    ))

    # Get or create collection
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Collection '{COLLECTION}' ready (existing docs: {collection.count()})")

    total_added = 0

    for fname in JSONL_FILES:
        path = os.path.join(PAGEINDEX_DIR, fname)
        if not os.path.exists(path):
            print(f"  SKIP: {fname}")
            continue

        records = load_records_from_jsonl(path)
        print(f"\n  {fname}: {len(records):,} records")

        for start in range(0, len(records), BATCH_SIZE):
            batch = records[start : start + BATCH_SIZE]

            texts     = [r.get("normalized_text") or r.get("text", "") for r in batch]
            ids       = [r["page_id"] for r in batch]
            metadatas = [chroma_metadata(r) for r in batch]

            # Embed
            embeddings = model.encode(texts, batch_size=64, show_progress_bar=False).tolist()

            # Upsert (safe to re-run)
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            total_added += len(batch)
            print(f"    Upserted {start + len(batch):,} / {len(records):,}", end="\r")

        print(f"    {fname} — done ({len(records):,} records)")

    # Persist to disk
    client.persist()
    print(f"\n✅  STEP 4 COMPLETE — {total_added:,} records in ChromaDB at {CHROMA_DIR}")
    return collection.count()


if __name__ == "__main__":
    print("=" * 50)
    print("STEP 4: Build ChromaDB (persistent vector store)")
    print(f"  Source    : {PAGEINDEX_DIR}")
    print(f"  ChromaDB  : {CHROMA_DIR}")
    print(f"  Model     : {EMBEDDING_MODEL}")
    print("=" * 50)

    count = build_chroma()
    print(f"  Total docs in collection: {count:,}")
