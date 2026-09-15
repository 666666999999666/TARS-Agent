from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest


@pytest.mark.parametrize("entry", ["cli", "core", "tui", "web"])
def test_entrypoints_do_not_probe_old_homes(
    entry: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    user = tmp_path / "user"
    legacy_roots = [user / ".kama", user / ".tars"]
    for root in legacy_roots:
        (root / "sessions" / "old").mkdir(parents=True)
        (root / "sessions" / "old" / "meta.json").write_text("private sentinel", encoding="utf-8")
        (root / "policy.toml").write_text('[always]\nbash = "allow"\n', encoding="utf-8")
    home = tmp_path / "new-home"
    home.mkdir()
    (home / "config.toml").write_text(
        '[trace]\nenabled = false\n[logging]\nfile = ""\n[sandbox]\nmode = "preferred"\n',
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("USERPROFILE", str(user))
    monkeypatch.setenv("HOME", str(user))
    monkeypatch.setenv("TARS_HOME", str(home))
    monkeypatch.setenv("TARS_CONFIG", str(home / "config.toml"))
    monkeypatch.chdir(workspace)
    touched: list[str] = []

    def guarded(method: Any) -> Any:
        def wrapped(path: Path, *args: Any, **kwargs: Any) -> Any:
            absolute = path.absolute()
            if any(absolute == old or absolute.is_relative_to(old) for old in legacy_roots):
                touched.append(str(path))
                raise AssertionError("entrypoint accessed an old HOME")
            return method(path, *args, **kwargs)
        return wrapped

    with monkeypatch.context() as guard:
        for name in ("open", "exists", "glob", "iterdir"):
            guard.setattr(Path, name, guarded(getattr(Path, name)))
        if entry == "cli":
            module = importlib.import_module("tars_agent.cli.main")
            guard.setattr(sys, "argv", ["tars", "ping"])
            guard.setattr(module, "cmd_ping", lambda config: None)
            module.main()
        elif entry == "core":
            module = importlib.import_module("tars_agent.core.app")
            async def stop_at_runtime(_config: Any) -> Any:
                raise RuntimeError("test boundary after fresh database bootstrap")
            guard.setattr(module, "initialize_runtime_router", stop_at_runtime)
            with pytest.raises(RuntimeError, match="test boundary"):
                asyncio.run(module.CoreApp().run())
        elif entry == "tui":
            module = importlib.import_module("tars_agent.tui.__main__")
            guard.setattr(sys, "argv", ["tars-tui"])
            guard.setattr(module.TarsTuiApp, "run", lambda self: None)
            module.main()
        else:
            module = importlib.import_module("tars_agent.web.cli")
            guard.setattr(sys, "argv", ["tars-web"])
            guard.setattr(module.uvicorn, "run", lambda *args, **kwargs: None)
            module.main()
        capsys.readouterr()  # Consume the local test-only Web bootstrap URL.
    assert touched == []
    for root in legacy_roots:
        assert (root / "sessions/old/meta.json").read_text(encoding="utf-8") == "private sentinel"
        assert (root / "policy.toml").read_text(encoding="utf-8") == '[always]\nbash = "allow"\n'
    if entry == "core":
        assert (home / "state.db").is_file()
