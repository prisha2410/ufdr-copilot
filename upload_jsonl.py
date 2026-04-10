"""
Run this to upload all JSONL files to HuggingFace.
These are needed for disk-based record lookup on Render.
"""

from huggingface_hub import HfApi
import os

api = HfApi()
REPO_ID = "pisha2410/ufdr-indexes"
BASE    = "D:/dl_proj/pageindex_data"

files = [
    "email.jsonl",
    "logon.jsonl",
    "device.jsonl",
    "file.jsonl",
    "ldap.jsonl",
    "psychometric.jsonl",
    # http_sample.jsonl is already uploading separately
]

for fname in files:
    path = os.path.join(BASE, fname)
    if not os.path.exists(path):
        print(f"SKIP {fname} — not found at {path}")
        continue

    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"Uploading {fname} ({size_mb:.0f} MB)...")

    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=f"pageindex_data/{fname}",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"  {fname} done!")

print("\nAll JSONL files uploaded!")
print("Now push retriever_api.py and deploy on Render.")
