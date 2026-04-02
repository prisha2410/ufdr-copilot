import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import pickle
from collections import defaultdict
from configs.paths import PAGEINDEX_DIR, PAGEINDEX_STORE

# HTTP skipped here - too large for PageIndex
# It will still be included in FAISS (semantic search)
JSONL_FILES = [
    "email.jsonl",
    "logon.jsonl",
    "device.jsonl",
    "file.jsonl",
    "ldap.jsonl",
    "psychometric.jsonl",
]

def build_indexes():
    user_index   = defaultdict(list)
    action_index = defaultdict(list)
    date_index   = defaultdict(list)
    hour_index   = defaultdict(list)
    total = 0

    for fname in JSONL_FILES:
        path = os.path.join(PAGEINDEX_DIR, fname)
        if not os.path.exists(path):
            print(f"  SKIP (not found): {path}")
            continue

        print(f"  Indexing {fname} ...", flush=True)
        file_count = 0
        file_records = {}

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

                    # indexes
                    user = record.get("user")
                    if user:
                        user_index[user].append(page_id)
                    for tu in record.get("target_users", []):
                        user_index[tu].append(page_id)

                    action = record.get("action")
                    if action:
                        action_index[action].append(page_id)

                    date = record.get("date")
                    if date:
                        date_index[date].append(page_id)

                    hour = record.get("hour")
                    if hour is not None:
                        hour_index[int(hour)].append(page_id)

                    file_records[page_id] = record
                    file_count += 1

                except Exception:
                    continue

        # Save this file's records, free RAM
        store_path = os.path.join(
            PAGEINDEX_STORE,
            fname.replace(".jsonl", "_records.pkl")
        )
        with open(store_path, "wb") as f:
            pickle.dump(file_records, f)

        del file_records
        print(f"    → {file_count:,} records indexed + saved")
        total += file_count

    return user_index, action_index, date_index, hour_index, total


def save_pickle(obj, filename):
    path = os.path.join(PAGEINDEX_STORE, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    size_mb = os.path.getsize(path) / 1_048_576
    print(f"  Saved {filename}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    print("=" * 50)
    print("STEP 3: Build PageIndex Maps (http.jsonl skipped)")
    print(f"  Source : {PAGEINDEX_DIR}")
    print(f"  Output : {PAGEINDEX_STORE}")
    print("=" * 50)

    user_index, action_index, date_index, hour_index, total = build_indexes()

    print(f"\n  Total records indexed: {total:,}")
    print(f"  Unique users   : {len(user_index):,}")
    print(f"  Unique actions : {len(action_index):,}")
    print(f"  Unique dates   : {len(date_index):,}")

    print("\n  Saving index maps ...")
    save_pickle(dict(user_index),   "user_index.pkl")
    save_pickle(dict(action_index), "action_index.pkl")
    save_pickle(dict(date_index),   "date_index.pkl")
    save_pickle(dict(hour_index),   "hour_index.pkl")

    print(f"\n✅  STEP 3 COMPLETE")