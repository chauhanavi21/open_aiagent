"""Persistent long-term memory with TF-IDF retrieval (JSON file, atomic writes)."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from .nlp import TfidfIndex

MAX_ITEMS = 10_000
MAX_TEXT = 2_000


class Memory:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._items: list[dict] = []
        self._index = TfidfIndex()
        if self._path.exists():
            try:
                loaded = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                loaded = []
            for item in loaded if isinstance(loaded, list) else []:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    self._items.append(item)
                    self._index.add(item["text"], item["text"])

    def __len__(self) -> int:
        return len(self._items)

    def add(self, text: str) -> int:
        text = text.strip()[:MAX_TEXT]
        if not text:
            raise ValueError("nothing to remember")
        with self._lock:
            if len(self._items) >= MAX_ITEMS:
                raise ValueError("memory is full")
            self._items.append({"id": len(self._items) + 1, "text": text, "ts": int(time.time())})
            self._index.add(text, text)
            self._save()
            return len(self._items)

    def search(self, query: str, k: int = 3, min_score: float = 0.15) -> list[str]:
        with self._lock:
            return [text for _, text in self._index.query(query, k=k, min_score=min_score)]

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._items, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)
