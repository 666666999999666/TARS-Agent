from __future__ import annotations

import argparse
import ipaddress
import socket
import threading
import time
import webbrowser

import uvicorn

from tars_agent.core.config import get_config
from tars_agent.web.app import create_app
from tars_agent.web.auth import LocalWebAuth
from tars_agent.web.core import SocketCoreReader


def _loopback_host(value: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("host must be a loopback IP address") from exc
    if not address.is_loopback:
        raise argparse.ArgumentTypeError("tars-web refuses non-loopback hosts")
    return value


def _open_when_listening(
    url: str,
    host: str,
    port: int,
    *,
    timeout_s: float = 15.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser(prog="tars-web")
    parser.add_argument("--host", type=_loopback_host, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7438)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    args = parser.parse_args()
    if not 1 <= args.port <= 65_535:
        parser.error("port must be between 1 and 65535")

    config = get_config()
    auth, bootstrap_token = LocalWebAuth.issue()
    url_host = f"[{args.host}]" if ":" in args.host else args.host
    url = f"http://{url_host}:{args.port}/#bootstrap={bootstrap_token}"
    print(f"One-time local URL (expires in 60 seconds):\n{url}", flush=True)
    if args.open_browser:
        threading.Thread(
            target=_open_when_listening,
            args=(url, args.host, args.port),
            name="tars-web-browser-bootstrap",
            daemon=True,
        ).start()
    # The launcher is the only component that ever receives the raw bootstrap
    # credential. Drop its references before entering the long-lived server
    # loop; LocalWebAuth retains only the digest. With --open, the short-lived
    # browser helper owns the URL only until it has opened the page.
    del bootstrap_token
    del url
    app = create_app(
        SocketCoreReader(config.host, config.port),
        auth=auth,
        allowed_host=args.host,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        access_log=False,
        log_config=None,
    )


__all__ = ["main"]
