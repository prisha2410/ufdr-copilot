"""
client/retriever_client.py
---------------------------
HTTP client for the UFDR Retriever API.

Usage:
    from client.retriever_client import RetrieverClient
    client = RetrieverClient()
    records = client.filter(user="LAP0338", action="connect_device")

Setup:
    Create .env file in repo root:
    RETRIEVER_HOST=https://YOUR-NGROK-URL.ngrok-free.app

Note:
    The Colab URL changes every session.
    Update RETRIEVER_HOST in .env each time a new session starts.
"""

import os
import requests
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class RetrieverClient:
    def __init__(self, host: str = None):
        self.host = (host or os.getenv("RETRIEVER_HOST", "http://localhost:8000")).rstrip("/")

    def _get(self, endpoint: str, params: dict = None):
        url = f"{self.host}{endpoint}"
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to API at {self.host}\n"
                f"Make sure the Colab server is running and RETRIEVER_HOST is set in .env"
            )
        except Exception as e:
            raise RuntimeError(f"API error: {e}")

    def health(self) -> dict:
        """Check if server is running."""
        return self._get("/health")

    def filter(
        self,
        user:       Optional[str] = None,
        action:     Optional[str] = None,
        date:       Optional[str] = None,
        hour_min:   Optional[int] = None,
        hour_max:   Optional[int] = None,
        event_type: Optional[str] = None,
        top_k:      int           = 50,
    ) -> list:
        """
        Structured retrieval using PageIndex maps.

        Args:
            user:       e.g. "LAP0338"
            action:     e.g. "connect_device", "send_email", "file_copy", "login"
            date:       e.g. "2010-02-01"
            hour_min:   e.g. 18 (6 PM)
            hour_max:   e.g. 23 (11 PM)
            event_type: e.g. "email", "logon", "device", "file"
            top_k:      max records to return (default 50)

        Returns:
            list of record dicts

        Examples:
            client.filter(user="LAP0338", action="connect_device", hour_min=18)
            client.filter(date="2010-02-01", event_type="email")
            client.filter(action="file_copy", hour_min=18, hour_max=23)
        """
        params = {k: v for k, v in {
            "user": user, "action": action, "date": date,
            "hour_min": hour_min, "hour_max": hour_max,
            "event_type": event_type, "top_k": top_k,
        }.items() if v is not None}

        result = self._get("/filter", params)
        return result.get("results", [])

    def vector(
        self,
        query:      str,
        top_k:      int           = 20,
        user:       Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> list:
        """
        Semantic FAISS search using natural language.

        Args:
            query:      natural language query
            top_k:      max records to return (default 20)
            user:       optional post-filter by user
            event_type: optional post-filter by event type

        Returns:
            list of record dicts with 'score' field

        Examples:
            client.vector("employee stealing data before leaving company")
            client.vector("suspicious USB activity after hours")
            client.vector("insider threat wikileaks upload", user="LAP0338")
        """
        params = {k: v for k, v in {
            "query": query, "top_k": top_k,
            "user": user, "event_type": event_type,
        }.items() if v is not None}

        result = self._get("/vector", params)
        if isinstance(result, dict) and "error" in result:
            print(f"Vector search unavailable: {result['error']}")
            return []
        return result.get("results", [])

    def http_search(
        self,
        keyword:    Optional[str] = None,
        user:       Optional[str] = None,
        date:       Optional[str] = None,
        top_k:      int           = 50,
    ) -> list:
        """
        Search browsing logs (http_sample.jsonl — 10% unbiased sample).

        Args:
            keyword: URL keyword e.g. "wikileaks", "dropbox", "linkedin"
            user:    optional filter by user
            date:    optional filter by date e.g. "2010-02-01"
            top_k:   max records to return

        Returns:
            list of http record dicts

        Examples:
            client.http_search(keyword="wikileaks")
            client.http_search(keyword="dropbox", user="LAP0338")
        """
        params = {k: v for k, v in {
            "keyword": keyword, "user": user,
            "date": date, "top_k": top_k,
        }.items() if v is not None}

        result = self._get("/http_search", params)
        return result.get("results", [])

    def get_record(self, page_id: str) -> Optional[dict]:
        """
        Get a single record by page_id.

        Args:
            page_id: e.g. "email_000123", "logon_000456"

        Returns:
            record dict or None if not found
        """
        try:
            return self._get(f"/record/{page_id}")
        except Exception:
            return None

    def stats(self) -> dict:
        """Get index statistics."""
        return self._get("/stats")