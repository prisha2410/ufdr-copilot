"""
api/startup.py
---------------
Downloads indexes from HuggingFace + http.jsonl from Google Drive.
"""

import os
import gdown
from huggingface_hub import snapshot_download

HF_REPO    = "pisha2410/ufdr-indexes"
BASE_PATH  = os.path.join(os.path.dirname(__file__), "../data")
HTTP_FILE_ID = "1uxhWo7UXHpg_smKlxdZZ4ygvyDmhsNPf"

PAGEINDEX_DIR = os.getenv("PAGEINDEX_DIR", os.path.join(BASE_PATH, "pageindex_data"))

os.makedirs(BASE_PATH,     exist_ok=True)
os.makedirs(PAGEINDEX_DIR, exist_ok=True)

print("=" * 40)
print("Downloading indexes from HuggingFace ...")
snapshot_download(
    repo_id=HF_REPO,
    repo_type="dataset",
    local_dir=BASE_PATH,
    ignore_patterns=["*.md", ".gitattributes"],
)
print("All indexes ready!")

# Download http.jsonl from Google Drive
http_path = os.path.join(PAGEINDEX_DIR, "http.jsonl")
if not os.path.exists(http_path):
    print("Downloading http.jsonl from Google Drive ...")
    gdown.download(
        f"https://drive.google.com/uc?id={HTTP_FILE_ID}",
        http_path,
        quiet=False
    )
    print("http.jsonl ready!")
else:
    print("http.jsonl already exists — skipping")

print("=" * 40)