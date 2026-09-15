from pathlib import Path

import pytest
from scripts.check_architecture import check


def test_runtime_and_web_persistence_boundaries() -> None:
    assert check() == []


def _probe_source(tmp_path: Path, relative: str, source: str) -> Path:
    root = tmp_path / "tars_agent"
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    return root


def test_web_core_protocol_allowlist_accepts_only_declared_symbols(tmp_path: Path) -> None:
    root = _probe_source(
        tmp_path,
        "web/core.py",
        "\n".join(
            (
                "from tars_agent.core.transport.socket_client import SocketClient",
                "from tars_agent.core.bus.envelope import EventPushEnvelope",
                "from tars_agent.core.bus.envelope import EventOverflowEnvelope",
            )
        ),
    )

    assert check(root) == []


def test_web_cli_has_only_file_scoped_config_exception(tmp_path: Path) -> None:
    root = _probe_source(
        tmp_path,
        "web/cli.py",
        "from tars_agent.core.config import get_config\n",
    )

    assert check(root) == []


@pytest.mark.parametrize(
    "statement",
    [
        "from tars_agent.core.runtime.service import RuntimeService",
        "from tars_agent.core.persistence import StateRepository",
        "from tars_agent.core.tools.runtime import DockerRuntime",
        "from tars_agent.core.permissions.manager import PermissionManager",
        "from tars_agent.core.transport.socket_client import IpcError",
        "import tars_agent.core.transport.socket_client",
        "from tars_agent.core.config import get_config",
    ],
)
def test_web_rejects_every_unlisted_core_dependency(
    tmp_path: Path,
    statement: str,
) -> None:
    root = _probe_source(tmp_path, "web/adapter.py", statement + "\n")

    errors = check(root)

    assert len(errors) == 1
    assert "Web adapter Core" in errors[0]
    assert "not allowlisted" in errors[0]
