"""
client/retriever_client.py
---------------------------
HTTP client for the UFDR Retriever API.

Members 2, 3, and 4 import this — they never touch the server or indexes.

Usage:
    from client.retriever_client import RetrieverClient

    client = RetrieverClient()   # reads RETRIEVER_HOST from env or uses default

    # Structured filter
    records = client.filter(user="LAP0338", action="connect_device", hour_min=18)

    # Semantic search
    records = client.vector("employee copying files after hours", top_k=20)

    # Browsing logs search
    records = client.http_search(keyword="wikileaks")

    # Single record
    rec = client.get_record("email_000123")

Set RETRIEVER_HOST in your environment or .env file:
    RETRIEVER_HOST=http://192.168.x.x:8000
"""

import os
import requests
from typing import Optional

_DEFAULT_HOST = "http://localhost:8000"


class RetrieverClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or os.getenv("RETRIEVER_HOST", _DEFAULT_HOST)).rstrip("/")
        self._check_health()

    # ─────────────────────────────────────────
    # INTERNALS
    # ─────────────────────────────────────────

    def _get(self, endpoint: str, params: dict) -> dict:
        clean = {k: v for k, v in params.items() if v is not None}
        url   = f"{self.base_url}{endpoint}"
        try:
            resp = requests.get(url, params=clean, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot reach UFDR Retriever at {self.base_url}. "
                "Make sure Member 1 has started api/server.py and set "
                "RETRIEVER_HOST correctly."
            )

    def _check_health(self):
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            resp.raise_for_status()
        except Exception:
            print(
                f"Warning: UFDR Retriever at {self.base_url} is not responding. "
                "Set RETRIEVER_HOST or ask Member 1 to start the server."
            )

    # ─────────────────────────────────────────
    # PUBLIC METHODS
    # ─────────────────────────────────────────

    def filter(
        self,
        user:       Optional[str] = None,
        action:     Optional[str] = None,
        date:       Optional[str] = None,
        hour_min:   Optional[int] = None,
        hour_max:   Optional[int] = None,
        event_type: Optional[str] = None,
        top_k:      int           = 50,
    ) -> list[dict]:
        """
        Structured retrieval using PageIndex maps.

        Parameters
        ----------
        user       : e.g. "LAP0338"
        action     : "connect_device" | "send_email" | "login" | "file_copy" | "visit_url"
        date       : e.g. "2010-02-01"
        hour_min   : 0-23  (e.g. hour_min=18 for after-hours start)
        hour_max   : 0-23  (e.g. hour_max=7  for early morning)
        event_type : "email" | "logon" | "device" | "file" | "http"
        top_k      : max records to return (default 50)

        Returns list of record dicts, sorted newest-first.

        Example:
            # All after-hours device connections
            records = client.filter(action="connect_device", hour_min=18)

            # All emails by a specific user
            records = client.filter(user="LAP0338", event_type="email")
        """
        data = self._get("/filter", {
            "user": user, "action": action, "date": date,
            "hour_min": hour_min, "hour_max": hour_max,
            "event_type": event_type, "top_k": top_k,
        })
        return data.get("results", [])

    def vector(
        self,
        query:      str,
        top_k:      int           = 20,
        user:       Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Semantic search using FAISS.
        Each result has an added 'score' field (0-1, higher = more relevant).

        Parameters
        ----------
        query      : natural language query string
        top_k      : number of results (default 20)
        user       : optional post-filter by user
        event_type : optional post-filter by event type

        Example:
            records = client.vector("employee copying files after hours", top_k=20)
            records = client.vector("data exfiltration", user="LAP0338")
        """
        data = self._get("/vector", {
            "query": query, "top_k": top_k,
            "user": user, "event_type": event_type,
        })
        return data.get("results", [])

    def http_search(
        self,
        user:    Optional[str] = None,
        keyword: Optional[str] = None,
        date:    Optional[str] = None,
        top_k:   int           = 50,
    ) -> list[dict]:
        """
        Search browsing logs (http.jsonl) by user, keyword, or date.
        Streams directly — no index needed.

        Parameters
        ----------
        user    : e.g. "LAP0338"
        keyword : e.g. "wikileaks", "dropbox", "jobsite"
        date    : e.g. "2010-02-01"
        top_k   : max records to return (default 50)

        Example:
            # Find wikileaks uploads
            records = client.http_search(keyword="wikileaks")

            # Find dropbox activity by specific user
            records = client.http_search(user="LAP0338", keyword="dropbox")

            # All browsing on a specific date
            records = client.http_search(date="2010-02-01")
        """
        data = self._get("/http_search", {
            "user": user, "keyword": keyword,
            "date": date, "top_k": top_k,
        })
        return data.get("results", [])

    def get_record(self, page_id: str) -> Optional[dict]:
        """
        Fetch a single evidence record by its page_id.

        Example:
            rec = client.get_record("email_000123")
        """
        try:
            return self._get(f"/record/{page_id}", {})
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    def stats(self) -> dict:
        """Return index statistics from the server."""
        return self._get("/stats", {})

    def health(self) -> bool:
        """Return True if server is up."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False