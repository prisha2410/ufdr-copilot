"""
api/startup.py - Downloads essential indexes + JSONL files from HuggingFace
"""
import os
from huggingface_hub import hf_hub_download

HF_REPO   = "pisha2410/ufdr-indexes"
BASE_PATH = os.path.join(os.path.dirname(__file__), "../data")
STORE     = os.path.join(BASE_PATH, "pageindex_store")
JSONL_DIR = os.path.join(BASE_PATH, "pageindex_data")

os.makedirs(STORE,    exist_ok=True)
os.makedirs(JSONL_DIR, exist_ok=True)

print("=" * 40)
print("Downloading indexes...")

files_to_download = [
    ("pageindex_store/hour_index.pkl",   STORE),
    ("pageindex_store/date_index.pkl",   STORE),
    ("pageindex_store/action_index.pkl", STORE),
    ("pageindex_data/email.jsonl",       JSONL_DIR),
    ("pageindex_data/logon.jsonl",       JSONL_DIR),
    ("pageindex_data/device.jsonl",      JSONL_DIR),
    ("pageindex_data/file.jsonl",        JSONL_DIR),
    ("pageindex_data/ldap.jsonl",        JSONL_DIR),
    ("pageindex_data/psychometric.jsonl",JSONL_DIR),
    ("pageindex_data/http_sample.jsonl", JSONL_DIR),
]

for repo_path, local_dir in files_to_download:
    fname = repo_path.split("/")[-1]
    out   = os.path.join(local_dir, fname)
    if not os.path.exists(out):
        print(f"  Downloading {fname}...")
        hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename=repo_path,
            local_dir=BASE_PATH,
        )
    else:
        print(f"  {fname} exists — skip")

print("All files ready!")
print("=" * 40)