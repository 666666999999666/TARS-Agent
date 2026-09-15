"""Start tars-web from the current isolated environment and probe it once."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def _free_ports() -> tuple[int, int]:
    """Reserve two distinct loopback ports until both numbers are selected."""

    with socket.socket() as web_listener, socket.socket() as core_listener:
        web_listener.bind(("127.0.0.1", 0))
        core_listener.bind(("127.0.0.1", 0))
        return (
            int(web_listener.getsockname()[1]),
            int(core_listener.getsockname()[1]),
        )


def main() -> int:
    port, unavailable_core_port = _free_ports()
    environment = os.environ.copy()
    environment.update(
        {
            "TARS_PORT": str(unavailable_core_port),
            "TARS_LOG_FILE": "",
            "TARS_TRACE_ENABLED": "false",
        }
    )
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from tars_agent.web.cli import main; main()",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise RuntimeError(
                    f"tars-web exited with {process.returncode}: "
                    f"{stdout.decode(errors='replace')} "
                    f"{stderr.decode(errors='replace')}"
                )
            try:
                with opener.open(  # fixed loopback, explicitly bypass system proxies
                    f"http://127.0.0.1:{port}/healthz",
                    timeout=4,
                ) as response:
                    if response.status == 200:
                        return 0
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.05)
        raise TimeoutError("isolated tars-web did not become ready")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
