from pathlib import Path

from tars_agent.core.artifacts import ArtifactStore


def test_artifact_store_preserves_session_run_layout(tmp_path: Path):
    store = ArtifactStore(tmp_path / "sessions")
    assert store.run_dir("session-1", "run-1") == tmp_path / "sessions/session-1/runs/run-1"
    assert not (tmp_path / "sessions").exists()


def test_notes_survive_reopening_without_creating_history_or_metadata(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    assert store.read_notes("session-1") == ""
    store.append_note("session-1", "中文事实", "run-1")
    store.append_note("session-1", "第二条", "run-2")
    notes = ArtifactStore(tmp_path).read_notes("session-1")
    assert "中文事实" in notes and "第二条" in notes
    assert "run-1" in notes and "run-2" in notes
    assert [path.name for path in (tmp_path / "session-1").iterdir()] == ["notes.md"]
