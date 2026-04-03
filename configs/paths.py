"""
configs/paths.py
----------------
Central path configuration for the UFDR Copilot project.

Local development: edit the default values below to match your machine.
Render deployment: environment variables automatically override defaults.
"""

import os

# ─────────────────────────────────────────────
# RAW DATA  (downloaded from CMU KiltHub)
# ─────────────────────────────────────────────
DATA_PATH = os.getenv("DATA_PATH", "D:/dl_proj/r4.2")

# ─────────────────────────────────────────────
# PIPELINE OUTPUT DIRS
# ─────────────────────────────────────────────
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "D:/dl_proj/processed data")
PAGEINDEX_DIR = os.getenv("PAGEINDEX_DIR", "D:/dl_proj/pageindex_data")

# ─────────────────────────────────────────────
# INDEX DIRS  (built once, shared via Google Drive)
# ─────────────────────────────────────────────
PAGEINDEX_STORE = os.getenv("PAGEINDEX_STORE", "D:/dl_proj/pageindex_store")
CHROMA_DIR      = os.getenv("CHROMA_DIR",      "D:/dl_proj/chroma_store")
FAISS_DIR       = os.getenv("FAISS_DIR",       "D:/dl_proj/faiss_index")

# ─────────────────────────────────────────────
# GROUND TRUTH (from answers.tar.bz2)
# ─────────────────────────────────────────────
INSIDERS_CSV   = os.getenv("INSIDERS_CSV",   "D:/dl_proj/answers/insiders.csv")
SCENARIOS_FILE = os.getenv("SCENARIOS_FILE", "D:/dl_proj/answers/scenarios.txt")

# ─────────────────────────────────────────────
# API SERVER
# ─────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ─────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

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