"""
run_server.py
--------------
Run this on your local machine to start the API server.
Your machine has enough RAM (16GB) for the full server.

Usage:
    python run_server.py

Requirements:
    pip install pyngrok
    Get free ngrok token at: https://dashboard.ngrok.com/signup
"""

import os
import sys
import time
import subprocess
import threading

# ─────────────────────────────────────────────
# CONFIGURATION — edit these if needed
# ─────────────────────────────────────────────

NGROK_TOKEN = "3CA2A81bo3T7jEMRLlDh6bDXTHI_3Un7Em6UGUpUP5qMQWFmt"  # ← get free token at dashboard.ngrok.com

# Local paths (should already be set in configs/paths.py)
os.environ.setdefault("PAGEINDEX_STORE", "D:/dl_proj/pageindex_store")
os.environ.setdefault("PAGEINDEX_DIR",   "D:/dl_proj/pageindex_data")
os.environ.setdefault("FAISS_DIR",       "D:/dl_proj/faiss_index")

# ─────────────────────────────────────────────
# START SERVER
# ─────────────────────────────────────────────

def stream_output(proc):
    for line in proc.stdout:
        print(line, end="")

print("=" * 60)
print("UFDR Copilot — Starting API Server")
print("=" * 60)

# Restore original full server.py from git
print("\nUsing original full server...")

# Start the server
server = subprocess.Popen(
    [sys.executable, "api/server.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    cwd=os.path.dirname(os.path.abspath(__file__))
)

# Stream server output in background
t = threading.Thread(target=stream_output, args=(server,), daemon=True)
t.start()

print("Server starting... waiting 15 seconds for indexes to load...")
time.sleep(15)

# ─────────────────────────────────────────────
# START NGROK
# ─────────────────────────────────────────────

try:
    from pyngrok import ngrok

    if NGROK_TOKEN == "PASTE_YOUR_NGROK_TOKEN_HERE":
        print("\n⚠️  No ngrok token set!")
        print("Get free token at: https://dashboard.ngrok.com/signup")
        print("Paste it in run_server.py line 17")
        print("\nServer running locally at: http://localhost:8000")
        print("Teammates cannot access this without ngrok token.")
    else:
        ngrok.set_auth_token(NGROK_TOKEN)
        url = ngrok.connect(8000)

        print("\n" + "=" * 60)
        print(f"✅ API URL  : {url}")
        print(f"✅ Docs     : {url}/docs")
        print(f"✅ Health   : {url}/health")
        print("=" * 60)
        print("\nShare this URL with your team!")
        print("Add to .env: RETRIEVER_HOST=" + str(url))
        print("\nKeep this terminal open while team works.")
        print("Press Ctrl+C to stop the server.")

except ImportError:
    print("\npyngrok not installed. Run: pip install pyngrok")
    print("Server running locally at: http://localhost:8000")

# ─────────────────────────────────────────────
# KEEP RUNNING
# ─────────────────────────────────────────────

try:
    server.wait()
except KeyboardInterrupt:
    print("\nShutting down server...")
    server.terminate()
    print("Server stopped.")