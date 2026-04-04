"""
api/startup.py
---------------
Downloads only essential indexes from HuggingFace.
JSONL files are downloaded lazily on first request.
"""

import os
from huggingface_hub import hf_hub_download

HF_REPO   = "pisha2410/ufdr-indexes"
BASE_PATH = os.path.join(os.path.dirname(__file__), "../data")
STORE     = os.path.join(BASE_PATH, "pageindex_store")
FAISS     = os.path.join(BASE_PATH, "faiss_index")

os.makedirs(STORE, exist_ok=True)
os.makedirs(FAISS, exist_ok=True)

print("=" * 40)
print("Downloading essential indexes ...")

# Only download the 4 small index maps + FAISS map
essential = [
    ("pageindex_store/user_index.pkl",   STORE),
    ("pageindex_store/action_index.pkl", STORE),
    ("pageindex_store/date_index.pkl",   STORE),
    ("pageindex_store/hour_index.pkl",   STORE),
    ("faiss_index/page_id_map.pkl",      FAISS),
]

for repo_path, local_dir in essential:
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

print("Essential indexes ready!")
print("JSONL files load on first request from HuggingFace")
print("=" * 40)