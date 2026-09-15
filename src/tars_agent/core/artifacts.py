from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path


class ArtifactStore:
    """Locate run artifacts and save session notes; SQLite owns history and run state."""

    def __init__(self, root: Path) -> None:
        self._root = root.expanduser()

    def session_dir(self, session_id: str) -> Path:
        return self._root / session_id

    def runs_dir(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "runs"

    def run_dir(self, session_id: str, run_id: str) -> Path:
        return self.runs_dir(session_id) / run_id

    def read_notes(self, session_id: str) -> str:
        path = self.session_dir(session_id) / "notes.md"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def append_note(self, session_id: str, content: str, run_id: str) -> None:
        directory = self.session_dir(session_id)
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).isoformat()
        with (directory / "notes.md").open("a", encoding="utf-8") as handle:
            handle.write(f"## Note ({timestamp}, {run_id})\n{content}\n\n")
