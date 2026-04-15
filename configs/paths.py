"""
configs/paths.py
----------------
Central path configuration for the UFDR Copilot project.

Each team member sets environment variables for their machine.
Either set them in your OS or create a .env file in the repo root.

Example .env file:
    DATA_PATH=C:/ufdr_data/r4.2
    PROCESSED_DIR=C:/ufdr_data/processed_data
    PAGEINDEX_DIR=C:/ufdr_data/pageindex_data
    PAGEINDEX_STORE=C:/ufdr_data/pageindex_store
    FAISS_DIR=C:/ufdr_data/faiss_index
    CHROMA_DIR=C:/ufdr_data/chroma_store
    INSIDERS_CSV=C:/ufdr_data/answers/insiders.csv
"""

import os
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# ─────────────────────────────────────────────
# RAW DATA
# ─────────────────────────────────────────────
DATA_PATH = os.getenv("DATA_PATH", "")

# ─────────────────────────────────────────────
# PIPELINE OUTPUT
# ─────────────────────────────────────────────
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "")
PAGEINDEX_DIR = os.getenv("PAGEINDEX_DIR", "")

# ─────────────────────────────────────────────
# INDEX DIRS
# ─────────────────────────────────────────────
PAGEINDEX_STORE = os.getenv("PAGEINDEX_STORE", "")
CHROMA_DIR      = os.getenv("CHROMA_DIR",      "")
FAISS_DIR       = os.getenv("FAISS_DIR",       "")

# ─────────────────────────────────────────────
# GROUND TRUTH
# ─────────────────────────────────────────────
INSIDERS_CSV   = os.getenv("INSIDERS_CSV",   "")
SCENARIOS_FILE = os.getenv("SCENARIOS_FILE", "")

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ─────────────────────────────────────────────
# DATA FILTER
# ─────────────────────────────────────────────
START_DATE = "2010-01-01"
END_DATE   = "2010-06-30"

# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
def check_paths():
    """Call this to verify all required paths are set."""
    required = {
        "PAGEINDEX_STORE": PAGEINDEX_STORE,
        "PAGEINDEX_DIR":   PAGEINDEX_DIR,
        "FAISS_DIR":       FAISS_DIR,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        print("ERROR: Missing environment variables:")
        for m in missing:
            print(f"  {m}")
        print("\nCreate a .env file in the repo root with your paths.")
        print("See configs/paths.py for example.")
        return False
    return True