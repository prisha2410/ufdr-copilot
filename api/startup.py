"""
api/startup.py
Downloads only 3 small index pkl files at startup (~70MB).
JSONL files download lazily on first request.
"""
import os
from huggingface_hub import hf_hub_download

HF_REPO   = "pisha2410/ufdr-indexes"
BASE_PATH = os.path.join(os.path.dirname(__file__), "../data")
STORE     = os.path.join(BASE_PATH, "pageindex_store")

os.makedirs(STORE, exist_ok=True)

print("=" * 40)
print("Downloading small indexes only...")

for fname in ["hour_index.pkl", "date_index.pkl", "action_index.pkl"]:
    out = os.path.join(STORE, fname)
    if not os.path.exists(out):
        print(f"  Downloading {fname}...")
        hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename=f"pageindex_store/{fname}",
            local_dir=BASE_PATH,
        )
    else:
        print(f"  {fname} exists — skip")

print("Startup complete! (~70MB RAM used)")
print("JSONL files will download on first request")
print("=" * 40)