"""
indexing/build_chroma.py
-------------------------
STEP 4 of Member 1's pipeline.

Reads all PageIndex-enhanced JSONL files, embeds the normalized_text field
using sentence-transformers, and persists every record into ChromaDB.

ChromaDB is an OPTIONAL persistent vector store backup.
- FAISS is built independently from JSONL files (not from ChromaDB)
- retriever_api.py uses FAISS for search, NOT ChromaDB
- ChromaDB is useful as a backup — if FAISS needs rebuilding,
  embeddings are already stored here (no re-embedding needed)

Safe to re-run — already embedded files are skipped automatically.

Run after build_pageindex.py:
    python indexing/build_chroma.py

Environment variables required (set in .env):
    PAGEINDEX_DIR   = path to pageindex_data/ folder
    CHROMA_DIR      = path to chroma_store/ folder
    EMBEDDING_MODEL = sentence transformer model name
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import json
import chromadb
from sentence_transformers import SentenceTransformer
from configs.paths import PAGEINDEX_DIR, CHROMA_DIR, EMBEDDING_MODEL, check_paths

# ─────────────────────────────────────────────
# VALIDATE
# ─────────────────────────────────────────────
if not check_paths():
    sys.exit(1)

JSONL_FILES = [
    "email.jsonl",
    "logon.jsonl",
    "device.jsonl",
    "file.jsonl",
    "ldap.jsonl",
    "psychometric.jsonl",
]

EXPECTED_COUNTS = {
    "email.jsonl":        1_000_037,
    "logon.jsonl":          329_511,
    "device.jsonl":         156_911,
    "file.jsonl":           171_112,
    "ldap.jsonl":              None,
    "psychometric.jsonl":      None,
}

BATCH_SIZE = 512
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
    """ChromaDB metadata must be flat (str/int/float/bool only)."""
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


def already_embedded(collection, fname: str) -> bool:
    """Check if a file's records are already in ChromaDB."""
    expected = EXPECTED_COUNTS.get(fname)
    if expected is None:
        return False
    try:
        result = collection.get(
            where={"source_file": {"$eq": fname}},
            limit=1,
            include=[],
        )
        if len(result["ids"]) > 0:
            print(f"    Already embedded - skipping {fname}")
            return True
    except Exception:
        pass
    return False


def build_chroma():
    print("Loading embedding model:", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to ChromaDB at:", CHROMA_DIR)
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.count()
    print(f"Collection '{COLLECTION}' ready (existing docs: {existing:,})")

    if existing >= 1_657_571:
        print("\n  All records already in ChromaDB - nothing to do!")
        print(f"\nSTEP 4 COMPLETE - {existing:,} records in ChromaDB")
        return existing

    total_added = 0

    for fname in JSONL_FILES:
        path = os.path.join(PAGEINDEX_DIR, fname)
        if not os.path.exists(path):
            print(f"  SKIP (not found): {fname}")
            continue

        if already_embedded(collection, fname):
            expected = EXPECTED_COUNTS.get(fname) or 0
            total_added += expected
            continue

        records = load_records_from_jsonl(path)
        print(f"\n  {fname}: {len(records):,} records")

        for start in range(0, len(records), BATCH_SIZE):
            batch     = records[start : start + BATCH_SIZE]
            texts     = [r.get("normalized_text") or r.get("text", "") for r in batch]
            ids       = [r["page_id"] for r in batch]
            metadatas = [chroma_metadata(r) for r in batch]

            embeddings = model.encode(
                texts, batch_size=64, show_progress_bar=False
            ).tolist()

            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            total_added += len(batch)
            print(f"    Upserted {start + len(batch):,} / {len(records):,}", end="\r")

        print(f"    {fname} done ({len(records):,} records)")

    final_count = collection.count()
    print(f"\nSTEP 4 COMPLETE - {final_count:,} records in ChromaDB at {CHROMA_DIR}")
    return final_count


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 4: Build ChromaDB (optional persistent vector backup)")
    print(f"  Source   : {PAGEINDEX_DIR}")
    print(f"  ChromaDB : {CHROMA_DIR}")
    print(f"  Model    : {EMBEDDING_MODEL}")
    print()
    print("  NOTE: ChromaDB is a backup store only.")
    print("  FAISS (build_faiss.py) is used for actual search.")
    print("  Already embedded files are skipped automatically.")
    print("=" * 60)

    count = build_chroma()
    print(f"  Total docs in collection: {count:,}")