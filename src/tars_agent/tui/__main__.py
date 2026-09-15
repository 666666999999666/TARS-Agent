from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
from pathlib import Path

from tars_agent.core.config import get_config
from tars_agent.core.paths import tars_home
from tars_agent.tui.app import TarsTuiApp


# TUI 文件日志初始化：不写 stderr（避免干扰 Textual 渲染），只写滚动文件
def _setup_logging(level: str) -> None:
    log_path = Path(os.environ.get(
        "TARS_TUI_LOG_FILE", str(tars_home() / "logs" / "tui.log"),
    )).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter(
            'level=%(levelname)s ts=%(asctime)s source=%(name)s msg="%(message)s"',
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    root.handlers.clear()
    root.addHandler(handler)


# tars-tui 入口：可恢复既有 Session，并从持久化 cursor 续传事件
def main() -> None:
    parser = argparse.ArgumentParser(prog="tars-tui", description="TARS-Agent TUI")
    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="Resume an existing durable session",
    )
    args = parser.parse_args()

    config = get_config()
    _setup_logging(config.logging.level)
    app = TarsTuiApp(config.host, config.port, resume_session_id=args.resume)
    app.run()


if __name__ == "__main__":
    main()
