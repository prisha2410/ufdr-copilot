"""
indexing/build_pageindex.py
----------------------------
STEP 3 of Member 1's pipeline.

Reads all PageIndex-enhanced JSONL files and builds four dictionary maps:

    user_index   : user_id  → [page_id, ...]
    action_index : action   → [page_id, ...]
    date_index   : date     → [page_id, ...]
    hour_index   : hour     → [page_id, ...]

Also builds a full record lookup:
    record_store : page_id → full record dict

All are saved as pickle files in PAGEINDEX_STORE.
Member 2's retriever loads these at query time for O(1) structured filtering.

Run after enhance_pageindex.py:
    python indexing/build_pageindex.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import pickle
from collections import defaultdict
from configs.paths import PAGEINDEX_DIR, PAGEINDEX_STORE

JSONL_FILES = [
    "email.jsonl",
    "logon.jsonl",
    "device.jsonl",
    "file.jsonl",
    "http.jsonl",
    "ldap.jsonl",
    "psychometric.jsonl",
]


def build_indexes():
    user_index   = defaultdict(list)   # user_id   → [page_ids]
    action_index = defaultdict(list)   # action    → [page_ids]
    date_index   = defaultdict(list)   # date str  → [page_ids]
    hour_index   = defaultdict(list)   # hour int  → [page_ids]
    record_store = {}                  # page_id   → full record

    total = 0

    for fname in JSONL_FILES:
        path = os.path.join(PAGEINDEX_DIR, fname)
        if not os.path.exists(path):
            print(f"  SKIP (not found): {path}")
            continue

        print(f"  Indexing {fname} ...", flush=True)
        file_count = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record  = json.loads(line)
                    page_id = record.get("page_id")
                    if not page_id:
                        continue

                    # ── full record store ──────────────────
                    record_store[page_id] = record

                    # ── user index ────────────────────────
                    user = record.get("user")
                    if user:
                        user_index[user].append(page_id)

                    # also index target_users
                    for tu in record.get("target_users", []):
                        user_index[tu].append(page_id)

                    # ── action index ──────────────────────
                    action = record.get("action")
                    if action:
                        action_index[action].append(page_id)

                    # ── date index ────────────────────────
                    date = record.get("date")
                    if date:
                        date_index[date].append(page_id)

                    # ── hour index ────────────────────────
                    hour = record.get("hour")
                    if hour is not None:
                        hour_index[int(hour)].append(page_id)

                    file_count += 1

                except Exception:
                    continue

        print(f"    → {file_count:,} records indexed")
        total += file_count

    return user_index, action_index, date_index, hour_index, record_store, total


def save_pickle(obj, filename):
    path = os.path.join(PAGEINDEX_STORE, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    size_mb = os.path.getsize(path) / 1_048_576
    print(f"  Saved {filename}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    print("=" * 50)
    print("STEP 3: Build PageIndex Maps")
    print(f"  Source : {PAGEINDEX_DIR}")
    print(f"  Output : {PAGEINDEX_STORE}")
    print("=" * 50)

    user_index, action_index, date_index, hour_index, record_store, total = build_indexes()

    print(f"\n  Total records indexed: {total:,}")
    print(f"  Unique users   : {len(user_index):,}")
    print(f"  Unique actions : {len(action_index):,}")
    print(f"  Unique dates   : {len(date_index):,}")
    print(f"  Hour range     : {min(hour_index.keys())} – {max(hour_index.keys())}")

    print("\n  Saving pickle files ...")
    save_pickle(dict(user_index),   "user_index.pkl")
    save_pickle(dict(action_index), "action_index.pkl")
    save_pickle(dict(date_index),   "date_index.pkl")
    save_pickle(dict(hour_index),   "hour_index.pkl")
    save_pickle(record_store,       "record_store.pkl")

    print(f"\n✅  STEP 3 COMPLETE — PageIndex saved to {PAGEINDEX_STORE}")
    print("    Files: user_index.pkl, action_index.pkl, date_index.pkl,")
    print("           hour_index.pkl, record_store.pkl")
