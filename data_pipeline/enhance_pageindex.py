"""
data_pipeline/enhance_pageindex.py
------------------------------------
STEP 2 of Member 1's pipeline.

Reads the JSONL files produced by data_pipeline.py and enriches every
record with PageIndex fields:

    page_id, source_file, line_number     ← traceability
    date, hour                            ← temporal indexing
    action                                ← normalised action label
    target_users, objects, keywords       ← structured entities
    normalized_text                       ← clean embedding text

Output goes to PAGEINDEX_DIR.

Run once (after data_pipeline.py):
    python data_pipeline/enhance_pageindex.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
from datetime import datetime
from configs.paths import PROCESSED_DIR, PAGEINDEX_DIR


# ─────────────────────────────────────────────
# ACTION MAPPING
# ─────────────────────────────────────────────

ACTION_MAP = {
    "email":       "send_email",
    "logon":       "login",
    "file":        "file_copy",
    "http":        "visit_url",
    "ldap":        "user_metadata",
    "psychometric":"personality_record",
}

KEYWORD_MAP = {
    "email":       ["communication"],
    "file":        ["file_transfer"],
    "device":      ["usb_activity"],
    "http":        ["web_activity"],
    "logon":       ["authentication"],
    "ldap":        ["identity"],
    "psychometric":["personality"],
}

def map_action(event_type: str, record: dict) -> str:
    if event_type == "device":
        return "connect_device" if "Connect" in record.get("text", "") else "disconnect_device"
    return ACTION_MAP.get(event_type, "unknown")

def extract_time_features(timestamp: str):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(timestamp[:19], fmt[:len(timestamp[:19])])
            return dt.strftime("%Y-%m-%d"), dt.hour
        except Exception:
            pass
    return None, None

def normalize_text(record: dict) -> str:
    return f"{record.get('user', '')} performed {record.get('action', '')}"


# ─────────────────────────────────────────────
# FILE PROCESSOR
# ─────────────────────────────────────────────

def process_file(input_path: str, output_path: str, prefix: str):
    print(f"  Processing {os.path.basename(input_path)} ...", flush=True)
    count = 0

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for i, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)

                # ── Traceability ──────────────────────────
                record["page_id"]     = f"{prefix}_{i:06d}"
                record["source_file"] = os.path.basename(input_path)
                record["line_number"] = i

                # ── Temporal ──────────────────────────────
                ts = record.get("timestamp") or ""
                record["date"], record["hour"] = extract_time_features(str(ts))

                # ── Action ────────────────────────────────
                event_type        = record.get("event_type", "")
                record["action"]  = map_action(event_type, record)

                # ── Structured entities ───────────────────
                entities = record.get("entities", {})
                record["target_users"] = entities.get("users", [])
                record["objects"]      = [event_type]
                record["keywords"]     = KEYWORD_MAP.get(event_type, [])

                # ── Clean embedding text ──────────────────
                record["normalized_text"] = normalize_text(record)

                f_out.write(json.dumps(record) + "\n")
                count += 1

            except Exception:
                continue

    print(f"  Done → {output_path}  ({count:,} records)")
    return count


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

FILES = {
    "email.jsonl":        "email",
    "logon.jsonl":        "logon",
    "device.jsonl":       "device",
    "file.jsonl":         "file",
    "http.jsonl":         "http",
    "ldap.jsonl":         "ldap",
    "psychometric.jsonl": "psychometric",
}

if __name__ == "__main__":
    print("=" * 50)
    print("STEP 2: JSONL → PageIndex-enhanced JSONL")
    print(f"  Source : {PROCESSED_DIR}")
    print(f"  Output : {PAGEINDEX_DIR}")
    print("=" * 50)

    total = 0
    for file_name, prefix in FILES.items():
        input_path  = os.path.join(PROCESSED_DIR, file_name)
        output_path = os.path.join(PAGEINDEX_DIR, file_name)
        if os.path.exists(input_path):
            total += process_file(input_path, output_path, prefix)
        else:
            print(f"  SKIP (not found): {input_path}")

    print(f"\n✅  STEP 2 COMPLETE — {total:,} records written to {PAGEINDEX_DIR}")
