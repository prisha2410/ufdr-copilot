"""
data_pipeline/data_pipeline.py
--------------------------------
STEP 1 of Member 1's pipeline.

Reads all raw CERT r4.2 CSV files, applies a temporal filter (Jan–Jun 2010),
and writes one JSONL file per event type to PROCESSED_DIR.

Run once:
    python data_pipeline/data_pipeline.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import json
import uuid
from glob import glob
from configs.paths import DATA_PATH, PROCESSED_DIR, START_DATE, END_DATE

CHUNK_SIZE = 100_000
BATCH_SIZE = 5_000


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def generate_id():
    return str(uuid.uuid4())

def safe_split(x):
    if pd.isna(x):
        return []
    return [s.strip() for s in str(x).split(";") if s.strip()]

def write_batch(batch, file_path):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r) for r in batch) + "\n")

def filter_time(df):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]

def process_stream(input_file, output_file, transform_fn):
    print(f"  Processing {input_file} ...", flush=True)
    for chunk in pd.read_csv(
        os.path.join(DATA_PATH, input_file),
        chunksize=CHUNK_SIZE,
        engine="c",
        low_memory=True,
    ):
        chunk = filter_time(chunk)
        batch = []
        for row in chunk.itertuples(index=False):
            try:
                batch.append(transform_fn(row))
                if len(batch) >= BATCH_SIZE:
                    write_batch(batch, output_file)
                    batch = []
            except Exception:
                continue
        if batch:
            write_batch(batch, output_file)
    print(f"  Done → {output_file}")


# ─────────────────────────────────────────────
# TRANSFORM FUNCTIONS
# ─────────────────────────────────────────────

def email_transform(row):
    return {
        "doc_id": generate_id(),
        "timestamp": str(row.date),
        "user": row.user,
        "pc": row.pc,
        "event_type": "email",
        "text": f"{row.user} emailed {row.to} content {str(row.content)[:200]}",
        "entities": {
            "users": safe_split(row.to) + safe_split(row.cc) + safe_split(row.bcc),
            "devices": [row.pc],
        },
        "metadata": {"size": row.size, "attachments": row.attachments},
    }

def logon_transform(row):
    return {
        "doc_id": generate_id(),
        "timestamp": str(row.date),
        "user": row.user,
        "pc": row.pc,
        "event_type": "logon",
        "text": f"{row.user} {row.activity} on {row.pc}",
        "entities": {"users": [row.user], "devices": [row.pc]},
        "metadata": {"activity": row.activity},
    }

def device_transform(row):
    return {
        "doc_id": generate_id(),
        "timestamp": str(row.date),
        "user": row.user,
        "pc": row.pc,
        "event_type": "device",
        "text": f"{row.user} {row.activity} device on {row.pc}",
        "entities": {"users": [row.user], "devices": [row.pc]},
        "metadata": {"activity": row.activity},
    }

def file_transform(row):
    return {
        "doc_id": generate_id(),
        "timestamp": str(row.date),
        "user": row.user,
        "pc": row.pc,
        "event_type": "file",
        "text": f"{row.user} accessed {row.filename} content {str(row.content)[:150]}",
        "entities": {"users": [row.user], "files": [row.filename], "devices": [row.pc]},
        "metadata": {},
    }

def http_transform(row):
    return {
        "doc_id": generate_id(),
        "timestamp": str(row.date),
        "user": row.user,
        "pc": row.pc,
        "event_type": "http",
        "text": f"{row.user} visited {row.url} content {str(row.content)[:100]}",
        "entities": {"users": [row.user], "urls": [row.url], "devices": [row.pc]},
        "metadata": {},
    }

def process_ldap():
    print("  Processing LDAP ...", flush=True)
    output_file = os.path.join(PROCESSED_DIR, "ldap.jsonl")
    for file in glob(os.path.join(DATA_PATH, "LDAP", "*.csv")):
        df = pd.read_csv(file)
        batch = []
        for row in df.itertuples(index=False):
            batch.append({
                "doc_id": generate_id(),
                "timestamp": os.path.basename(file).replace(".csv", ""),
                "user": row.user_id,
                "event_type": "ldap",
                "text": f"{row.employee_name} role {row.role} dept {row.department}",
                "entities": {"users": [row.user_id]},
                "metadata": {
                    "email": row.email,
                    "role": row.role,
                    "department": row.department,
                    "supervisor": row.supervisor,
                },
            })
        write_batch(batch, output_file)
    print(f"  Done → {output_file}")

def process_psychometric():
    print("  Processing Psychometric ...", flush=True)
    df = pd.read_csv(os.path.join(DATA_PATH, "psychometric.csv"))
    output_file = os.path.join(PROCESSED_DIR, "psychometric.jsonl")
    batch = []
    for row in df.itertuples(index=False):
        batch.append({
            "doc_id": generate_id(),
            "user": row.user_id,
            "event_type": "psychometric",
            "text": f"user {row.user_id} personality O:{row.O} C:{row.C} E:{row.E} A:{row.A} N:{row.N}",
            "entities": {"users": [row.user_id]},
            "metadata": {"O": row.O, "C": row.C, "E": row.E, "A": row.A, "N": row.N},
        })
    write_batch(batch, output_file)
    print(f"  Done → {output_file}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("STEP 1: RAW CSV → JSONL")
    print(f"  Source : {DATA_PATH}")
    print(f"  Output : {PROCESSED_DIR}")
    print(f"  Filter : {START_DATE} → {END_DATE}")
    print("=" * 50)

    # Clear old output
    for fname in ["email", "logon", "device", "file", "http", "ldap", "psychometric"]:
        path = os.path.join(PROCESSED_DIR, f"{fname}.jsonl")
        if os.path.exists(path):
            os.remove(path)

    process_stream("email.csv",  os.path.join(PROCESSED_DIR, "email.jsonl"),  email_transform)
    process_stream("logon.csv",  os.path.join(PROCESSED_DIR, "logon.jsonl"),  logon_transform)
    process_stream("device.csv", os.path.join(PROCESSED_DIR, "device.jsonl"), device_transform)
    process_stream("file.csv",   os.path.join(PROCESSED_DIR, "file.jsonl"),   file_transform)
    process_stream("http.csv",   os.path.join(PROCESSED_DIR, "http.jsonl"),   http_transform)
    process_ldap()
    process_psychometric()

    print("\n✅  STEP 1 COMPLETE — JSONL files ready in", PROCESSED_DIR)
