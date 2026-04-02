"""
configs/paths.py
----------------
Central path configuration for the UFDR Copilot project.
Each team member edits ONLY this file to match their local setup.
"""

import os

# ─────────────────────────────────────────────
# RAW DATA  (downloaded from CMU KiltHub)
# ─────────────────────────────────────────────
DATA_PATH = "D:/dl_proj/r4.2"          # folder containing email.csv, logon.csv, etc.

# ─────────────────────────────────────────────
# PIPELINE OUTPUT DIRS
# ─────────────────────────────────────────────
PROCESSED_DIR   = "D:/dl_proj/processed_data"    # output of data_pipeline.py
PAGEINDEX_DIR   = "D:/dl_proj/pageindex_data"    # output of enhance_pageindex.py

# ─────────────────────────────────────────────
# INDEX DIRS  (built by Member 1, shared / rebuilt locally)
# ─────────────────────────────────────────────
PAGEINDEX_STORE = "D:/dl_proj/pageindex_store"   # pickle files for structured indexes
CHROMA_DIR      = "D:/dl_proj/chroma_store"      # ChromaDB persistent folder
FAISS_DIR       = "D:/dl_proj/faiss_index"       # FAISS binary index + page_id map

# ─────────────────────────────────────────────
# GROUND TRUTH (from answers.tar.bz2)
# ─────────────────────────────────────────────
INSIDERS_CSV    = "D:/dl_proj/answers/insiders.csv"
SCENARIOS_FILE  = "D:/dl_proj/answers/scenarios.txt"

# ─────────────────────────────────────────────
# API SERVER
# ─────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ─────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast + good quality, ~80MB download

# ─────────────────────────────────────────────
# DATA FILTER  (temporal scope of processed data)
# ─────────────────────────────────────────────
START_DATE = "2010-01-01"
END_DATE   = "2010-06-30"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
for _dir in [PROCESSED_DIR, PAGEINDEX_DIR, PAGEINDEX_STORE, CHROMA_DIR, FAISS_DIR]:
    os.makedirs(_dir, exist_ok=True)
