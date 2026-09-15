from __future__ import annotations

import asyncio
import codecs
import os
import sys
import threading


class TerminalInput:
    """Read console lines without making asyncio wait for a blocked input thread on exit."""

    def __init__(self) -> None:
        self._lines: asyncio.Queue[str | None] = asyncio.Queue()
        self._stopped = threading.Event()
        self._started = False
        self._console_line = ""
        self._extended_key = False

    async def read(self) -> str | None:
        if os.name == "nt" and sys.stdin.isatty():
            return await self._read_console()
        if not self._started:
            self._started = True
            loop = asyncio.get_running_loop()
            threading.Thread(target=self._read_lines, args=(loop,), daemon=True,
                             name="tars-cli-input").start()
        return await self._lines.get()

    async def _read_console(self) -> str | None:
        import msvcrt

        # Windows ReadConsole reports both Ctrl+C and Ctrl+Z as empty reads.
        # Read available keys instead, so only Ctrl+Z is treated as EOF. Ctrl+C
        # continues through the client's normal SIGINT cancellation handler.
        while not self._stopped.is_set():
            if not msvcrt.kbhit():
                await asyncio.sleep(0.02)
                continue
            character = msvcrt.getwch()
            if self._extended_key:
                self._extended_key = False
            elif character in {"\x00", "\xe0"}:
                self._extended_key = True
            elif character == "\x1a":
                print(flush=True)
                return None
            elif character in {"\r", "\n"}:
                line, self._console_line = self._console_line, ""
                print(flush=True)
                return line
            elif character in {"\b", "\x7f"}:
                if self._console_line:
                    self._console_line = self._console_line[:-1]
                    print("\b \b", end="", flush=True)
            elif character >= " ":
                self._console_line += character
                print(character, end="", flush=True)
        return None

    def close(self) -> None:
        self._stopped.set()

    def _read_lines(self, loop: asyncio.AbstractEventLoop) -> None:
        def deliver(line: str | None) -> None:
            if not self._stopped.is_set():
                try:
                    loop.call_soon_threadsafe(self._lines.put_nowait, line)
                except RuntimeError:
                    pass  # The client has already closed its event loop.

        # Read the raw console/file stream, not BufferedReader: a blocked buffered
        # stdin reader can otherwise hold a Python IO lock during interpreter exit.
        raw = getattr(getattr(sys.stdin, "buffer", None), "raw", None)
        try:
            if raw is None:
                while not self._stopped.is_set():
                    line = sys.stdin.readline()
                    if not line:
                        break
                    deliver(line.rstrip("\r\n"))
            else:
                decoder = codecs.getincrementaldecoder(sys.stdin.encoding or "utf-8")("replace")
                pending = ""
                while not self._stopped.is_set():
                    try:
                        chunk = raw.read(4096)
                    except InterruptedError:
                        continue
                    if not chunk:
                        pending += decoder.decode(b"", final=True)
                        if pending:
                            deliver(pending.rstrip("\r"))
                        break
                    pending += decoder.decode(chunk)
                    while "\n" in pending:
                        line, pending = pending.split("\n", 1)
                        deliver(line.rstrip("\r"))
        except (OSError, ValueError):
            pass
        finally:
            deliver(None)
