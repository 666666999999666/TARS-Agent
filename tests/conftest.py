from __future__ import annotations

import asyncio
import atexit
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path

import pytest

_COLLECTION_HOME = tempfile.TemporaryDirectory(prefix="tars-test-collection-")
atexit.register(_COLLECTION_HOME.cleanup)
# Explicit real-model tests keep only the user's requested model credentials. Ordinary
# collection cannot use real configuration or accidentally inherit an enabled endpoint.
_REAL_MODEL_SELECTED = "--real-model" in sys.argv
if not _REAL_MODEL_SELECTED:
    for _key in tuple(os.environ):
        if _key.startswith(("TARS_", "KAMA_", "ANTHROPIC_", "OPENAI_")):
            os.environ.pop(_key, None)
if not _REAL_MODEL_SELECTED:
    os.environ["TARS_HOME"] = _COLLECTION_HOME.name
    os.environ["TARS_CONFIG"] = str(Path(_COLLECTION_HOME.name) / "config.toml")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--real-model", action="store_true", default=False,
                     help="Explicitly allow budgeted real model tests")


@pytest.fixture(autouse=True)
def restore_application_logging() -> Iterator[None]:
    loggers = [logging.getLogger(), logging.getLogger("tars_agent"),
               logging.getLogger("tars_agent.core.llm.provider")]
    states = [(logger, list(logger.handlers), logger.level, logger.propagate, logger.disabled)
              for logger in loggers]
    disabled = logging.root.manager.disable
    yield
    for logger, handlers, level, propagate, logger_disabled in states:
        for handler in list(logger.handlers):
            if handler not in handlers:
                handler.close()
        logger.handlers = handlers
        logger.setLevel(level)
        logger.propagate = propagate
        logger.disabled = logger_disabled
    logging.disable(disabled)



@pytest.fixture(autouse=True)
def isolated_runtime_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if _REAL_MODEL_SELECTED:
        return
    root = tmp_path / "tars-test-home"
    root.mkdir(exist_ok=True)
    monkeypatch.setenv("TARS_HOME", str(root))
    monkeypatch.setenv("TARS_CONFIG", str(root / "config.toml"))
    from tars_agent.core.config import get_config
    clear = getattr(get_config, "cache_clear", None)
    if clear is not None:
        clear()


@pytest.fixture
def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return port  # socket released; daemon can bind to this port


@pytest.fixture
async def running_daemon(
    free_port: int,
    tmp_path: Path,
) -> AsyncGenerator[subprocess.Popen[bytes], None]:
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    env["TARS_PORT"] = str(free_port)
    env["TARS_LOG_FILE"] = ""
    env["TARS_LOG_LEVEL"] = "WARNING"
    # IPC fixture only; this is never counted as real sandbox acceptance.
    env["TARS_SANDBOX_MODE"] = "preferred"
    env["TARS_TRACE_ENABLED"] = "false"
    env["TARS_HOME"] = str(tmp_path / "tars-home")

    proc = subprocess.Popen(
        [sys.executable, "-m", "tars_agent.core"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            pytest.fail(
                "Daemon exited before startup "
                f"(code={proc.returncode})\n"
                f"stdout:\n{stdout.decode(errors='replace')}\n"
                f"stderr:\n{stderr.decode(errors='replace')}"
            )
        try:
            _reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
            writer.close()
            await writer.wait_closed()
            break
        except (ConnectionRefusedError, OSError):
            pass
    else:
        proc.terminate()
        stdout, stderr = proc.communicate(timeout=2)
        pytest.fail(
            "Daemon did not start within 10 seconds\n"
            f"stdout:\n{stdout.decode(errors='replace')}\n"
            f"stderr:\n{stderr.decode(errors='replace')}"
        )

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
