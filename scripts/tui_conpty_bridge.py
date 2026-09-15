"""Private QA bridge: real Windows ConPTY -> local xterm.js; never a shell server."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
MAX_BODY = 65536
MAX_BUFFER = 8 * 1024 * 1024


class TuiPty:
    def __init__(self, args):
        self.args = args
        self.rows, self.cols = args.rows, args.cols
        self.generation = 0
        self.pid = None
        self.exit_code = None
        self.error = None
        self.pty = None
        self.reader = None
        self.reader_stop = threading.Event()
        self.state_lock = threading.RLock()
        self.lifecycle_lock = threading.Lock()
        self.chunks = deque()
        self.seq = 0
        self.buffer_size = 0
        self.origin = ""
        self.nonce = secrets.token_urlsafe(32)
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.state_file = self.output / "bridge.json"

    def alive(self):
        pty = self.pty
        try:
            return pty is not None and pty.isalive()
        except Exception:
            return False

    def persist(self):
        with self.state_lock:
            record = {"bridge_pid": os.getpid(), "pty_pid": self.pid,
                      "origin": self.origin, "nonce": self.nonce,
                      "url": self.origin + "/#nonce=" + self.nonce,
                      "backend": "Windows ConPTY", "renderer": "xterm.js 6.0.0",
                      "session_id": self.args.session, "core_port": self.args.core_port,
                      "workspace": str(self.args.workspace), "home": str(self.args.home),
                      "generation": self.generation, "running": self.alive(),
                      "exit_code": self.exit_code, "rows": self.rows, "cols": self.cols}
            temporary = self.state_file.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.state_file)

    def start(self):
        from winpty import PTY, Backend
        with self.lifecycle_lock:
            if self.alive():
                raise ValueError("TUI is already running")
            self._release()
            allowed = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "TEMP", "TMP",
                       "USERPROFILE", "APPDATA", "LOCALAPPDATA", "COMSPEC"}
            env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
            # The client neither inherits model keys nor reads the Core's credential config.
            client_config = self.output / "client-config.toml"
            client_config.write_text("[core]\nhost = '127.0.0.1'\nport = " + str(self.args.core_port) + "\n", encoding="utf-8")
            env.update({"TARS_HOME": str(self.args.home), "TARS_CONFIG": str(client_config),
                        "TARS_HOST": "127.0.0.1", "TARS_PORT": str(self.args.core_port),
                        "TARS_TUI_LOG_FILE": str(self.output / "tui.log"),
                        "PYTHONPATH": str(ROOT / "src"), "PYTHONUTF8": "1",
                        "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                        "TERM": "xterm-256color", "COLORTERM": "truecolor"})
            pty = PTY(self.cols, self.rows, backend=Backend.ConPTY)
            command = " " + subprocess.list2cmdline(["-m", "tars_agent.tui", "--resume", self.args.session])
            if not pty.spawn(str(Path(sys.executable).resolve()), cmdline=command,
                             cwd=str(self.args.workspace), env="\0".join(f"{k}={v}" for k, v in env.items()) + "\0"):
                raise RuntimeError("ConPTY could not start the fixed TUI process")
            with self.state_lock:
                self.pty = pty
                self.pid = pty.pid
                self.exit_code = None
                self.error = None
                self.generation += 1
                self.chunks.clear()
                self.seq = 0
                self.buffer_size = 0
                self.reader_stop = threading.Event()
            self.reader = threading.Thread(target=self._read, args=(pty, self.reader_stop, self.generation), daemon=True)
            self.reader.start()
            self.persist()

    def _read(self, pty, stop, generation):
        path = self.output / f"pty-output-{generation}.ansi.log"
        try:
            with path.open("w", encoding="utf-8", newline="") as log:
                while not stop.is_set():
                    data = pty.read(blocking=False)
                    if data:
                        log.write(data)
                        log.flush()
                        with self.state_lock:
                            self.seq += 1
                            self.chunks.append({"seq": self.seq, "text": data})
                            self.buffer_size += len(data.encode("utf-8"))
                            while self.buffer_size > MAX_BUFFER and self.chunks:
                                self.buffer_size -= len(self.chunks.popleft()["text"].encode("utf-8"))
                    if not pty.isalive():
                        self.exit_code = pty.get_exitstatus()
                        break
                    stop.wait(0.01)
        except Exception as exc:
            self.error = type(exc).__name__
        finally:
            self.persist()

    def poll(self, after):
        with self.state_lock:
            first = self.chunks[0]["seq"] if self.chunks else self.seq + 1
            return {"chunks": [chunk for chunk in self.chunks if chunk["seq"] > after],
                    "cursor": self.seq, "truncated": after < first - 1,
                    "generation": self.generation, "running": self.alive(), "pid": self.pid,
                    "exit_code": self.exit_code, "error": self.error,
                    "rows": self.rows, "cols": self.cols}

    def write(self, text):
        if not isinstance(text, str) or len(text.encode("utf-8")) > 16384:
            raise ValueError("input must be at most 16 KiB of text")
        if not self.alive():
            raise ValueError("TUI exited; explicitly restart the same TUI")
        self.pty.write(text)

    def resize(self, rows, cols):
        if isinstance(rows, bool) or isinstance(cols, bool) or not isinstance(rows, int) or not isinstance(cols, int):
            raise ValueError("terminal dimensions must be integers")
        if not 10 <= rows <= 120 or not 40 <= cols <= 300:
            raise ValueError("terminal size must be 40..300 columns and 10..120 rows")
        self.rows, self.cols = rows, cols
        if self.alive():
            self.pty.set_size(cols, rows)
        if self.origin:
            self.persist()

    def _release(self):
        self.reader_stop.set()
        if self.pty is not None:
            try:
                self.pty.cancel_io()
            except Exception:
                pass
        if self.reader is not None:
            self.reader.join(timeout=2)
        self.pty = None
        self.reader = None

    def close(self):
        with self.lifecycle_lock:
            if self.alive():
                self.pty.write("\x11")  # Ctrl+Q exits TUI, preserving the Core session.
                deadline = time.monotonic() + 4
                while self.alive() and time.monotonic() < deadline:
                    time.sleep(0.05)
            if self.alive():
                # PID belongs to the still-live PTY handle created by this harness.
                taskkill = Path(os.environ["SystemRoot"]) / "System32/taskkill.exe"
                subprocess.run([str(taskkill), "/PID", str(self.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=5, check=False)
                deadline = time.monotonic() + 2
                while self.alive() and time.monotonic() < deadline:
                    time.sleep(0.05)
            if self.alive():
                raise RuntimeError("owned TUI process did not exit")
            self._release()
            self.persist()


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(3)

    def log_message(self, *_):
        pass  # Do not log request credentials or terminal input.

    def common(self, status, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; font-src 'self'; frame-ancestors 'none'")

    def respond(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.common(status, "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def valid_host(self):
        return self.headers.get("Host") == self.server.bridge.origin.removeprefix("http://")

    def do_GET(self):
        if not self.valid_host():
            self.respond(403, {"error": "invalid host"})
            return
        assets = {
            "/": ROOT / "scripts/tui_bridge_assets/index.html",
            "/bridge.js": ROOT / "scripts/tui_bridge_assets/bridge.js",
            "/vendor/xterm.mjs": ROOT / "web/node_modules/@xterm/xterm/lib/xterm.mjs",
            "/vendor/addon-fit.mjs": ROOT / "web/node_modules/@xterm/addon-fit/lib/addon-fit.mjs",
            "/vendor/xterm.css": ROOT / "web/node_modules/@xterm/xterm/css/xterm.css",
        }
        path = assets.get(urlsplit(self.path).path)
        if path is None or not path.is_file():
            self.respond(404, {"error": "not found"})
            return
        data = path.read_bytes()
        kind = "text/javascript" if path.suffix in {".js", ".mjs"} else mimetypes.guess_type(path)[0] or "application/octet-stream"
        self.common(200, kind + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        bridge = self.server.bridge
        token = self.headers.get("X-Tars-Bridge-Token", "")
        if (not self.valid_host() or self.headers.get("Origin") != bridge.origin
                or not secrets.compare_digest(token.encode("utf-8"), bridge.nonce.encode("ascii"))):
            self.respond(403, {"error": "bridge authentication or Origin failed"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                raise ValueError("invalid body size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            route = urlsplit(self.path).path
            if route == "/api/poll":
                after = payload.get("after", 0)
                if not isinstance(after, int) or isinstance(after, bool) or after < 0:
                    raise ValueError("invalid cursor")
                result = bridge.poll(after)
            elif route == "/api/input":
                bridge.write(payload.get("text"))
                result = {"ok": True}
            elif route == "/api/resize":
                bridge.resize(payload.get("rows"), payload.get("cols"))
                result = {"ok": True}
            elif route == "/api/restart" and not payload:
                bridge.start()
                result = {"ok": True}
            elif route == "/api/shutdown" and not payload:
                result = {"ok": True}
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self.respond(404, {"error": "unknown fixed bridge operation"})
                return
            self.respond(200, result)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.respond(409, {"error": str(exc)})
        except Exception as exc:
            self.respond(500, {"error": type(exc).__name__})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-port", required=True, type=int)
    parser.add_argument("--session", required=True)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--home", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "build/qa/tui-conpty")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--rows", type=int, default=42)
    parser.add_argument("--cols", type=int, default=140)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("this QA bridge requires Windows ConPTY")
    if not re.fullmatch(r"sess-[a-f0-9]{12}", args.session):
        parser.error("invalid durable session identifier")
    if not 1 <= args.core_port <= 65535 or not 0 <= args.port <= 65535:
        parser.error("invalid port")
    args.workspace = args.workspace.resolve(strict=True)
    args.home = args.home.resolve(strict=True)
    if not args.workspace.is_dir() or not args.home.is_dir():
        parser.error("workspace and home must be existing directories")
    bridge = TuiPty(args)
    bridge.resize(args.rows, args.cols)
    server = BridgeServer(("127.0.0.1", args.port), Handler)
    server.bridge = bridge
    bridge.origin = f"http://127.0.0.1:{server.server_port}"
    try:
        bridge.start()
        print(json.dumps({"bridge_pid": os.getpid(), "origin": bridge.origin,
                          "state_file": str(bridge.state_file)}), flush=True)
        server.serve_forever(poll_interval=0.1)
    finally:
        bridge.close()
        server.server_close()


if __name__ == "__main__":
    main()
