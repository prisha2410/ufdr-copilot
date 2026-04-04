"""
api/startup.py
---------------
Downloads indexes from HuggingFace on Render startup.
Includes http_sample.jsonl (10% sample, unbiased).
"""

import os
from huggingface_hub import snapshot_download

HF_REPO   = "pisha2410/ufdr-indexes"
BASE_PATH = os.path.join(os.path.dirname(__file__), "../data")

os.makedirs(BASE_PATH, exist_ok=True)

print("=" * 40)
print("Downloading indexes from HuggingFace ...")
print(f"  Repo : {HF_REPO}")
print(f"  Path : {BASE_PATH}")

snapshot_download(
    repo_id=HF_REPO,
    repo_type="dataset",
    local_dir=BASE_PATH,
    ignore_patterns=["*.md", ".gitattributes"],
)

print("All indexes ready!")
print("=" * 40)