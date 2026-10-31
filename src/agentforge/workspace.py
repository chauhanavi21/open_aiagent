"""Sandboxed workspace: every file operation is confined to one root directory."""
from __future__ import annotations

from pathlib import Path

ALLOWED_SUFFIXES = {".md", ".txt", ".json", ".csv"}
MAX_BYTES = 200_000


class WorkspaceError(Exception):
    pass


class Workspace:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, rel: str) -> Path:
        rel = (rel or "").strip()
        if not rel or "\x00" in rel or len(rel) > 200:
            raise WorkspaceError("invalid path")
        candidate = Path(rel)
        if candidate.is_absolute() or candidate.drive:
            raise WorkspaceError("absolute paths are not allowed")
        full = (self.root / candidate).resolve()
        try:
            full.relative_to(self.root)
        except ValueError:
            raise WorkspaceError("path escapes the workspace") from None
        if full.suffix.lower() not in ALLOWED_SUFFIXES:
            raise WorkspaceError("file type not allowed")
        return full

    def read(self, rel: str) -> str:
        path = self.resolve(rel)
        if not path.is_file():
            raise WorkspaceError("file not found")
        if path.stat().st_size > MAX_BYTES:
            raise WorkspaceError("file too large")
        return path.read_text(encoding="utf-8", errors="replace")

    def write(self, rel: str, content: str) -> str:
        data = content.encode("utf-8")
        if len(data) > MAX_BYTES:
            raise WorkspaceError("content too large")
        path = self.resolve(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path.relative_to(self.root)).replace("\\", "/")
