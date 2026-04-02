"""
api/startup.py
---------------
Downloads indexes from Google Drive on Render startup.
Runs automatically before server starts.
"""

import os
import gdown

# ── Google Drive folder IDs ────────────────────────────────
FAISS_FOLDER_ID     = "1V6J87QTAXSRCr0tZEC-Za94nWGzCsmgr"
PAGEINDEX_FOLDER_ID = "1q7AHgFWdVB0GOrCb8hucvl1jkd0iCpPA"

# ── Local paths on Render ──────────────────────────────────
FAISS_DIR       = "/data/faiss_index"
PAGEINDEX_STORE = "/data/pageindex_store"

os.makedirs(FAISS_DIR,       exist_ok=True)
os.makedirs(PAGEINDEX_STORE, exist_ok=True)

def download_if_missing(folder_id, dest_dir, name):
    files = os.listdir(dest_dir)
    if files:
        print(f"  {name} already exists ({len(files)} files) — skipping")
        return
    print(f"  Downloading {name} from Google Drive ...")
    gdown.download_folder(
        id=folder_id,
        output=dest_dir,
        quiet=False,
        use_cookies=False,
    )
    print(f"  {name} downloaded!")

print("=" * 40)
print("Checking indexes ...")
download_if_missing(FAISS_FOLDER_ID,     FAISS_DIR,       "faiss_index")
download_if_missing(PAGEINDEX_FOLDER_ID, PAGEINDEX_STORE, "pageindex_store")
print("All indexes ready!")
print("=" * 40)