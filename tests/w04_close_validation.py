"""Replay the reviewed W04 assertions in a new directory after the close fix."""
from __future__ import annotations

import argparse
import asyncio
import json
import os

from tests.w04_support import EXECUTION, ISOLATED, PREP, container_ids, save

if not os.environ.get("W04_EXECUTION_DIR") or ISOLATED == PREP / "isolated":
    raise RuntimeError("close verification requires explicit new evidence and isolation directories")

from tests.w04_real_validation import (  # noqa: E402
    LOG,
    Core,
    cli_approval,
    main,
    record,
    tool,
    write_new,
)


async def cli_close():
    core = Core("docker-lifecycle", "close-fix-cli")
    prefix = "same-prefix-" * 200
    plan = [tool(f"approval-{i}", "write_file", {"path": "approval.txt", "content": prefix + tail})
            for i, tail in enumerate(("DENY_A", "ALLOW_B", "DENY_C"))]
    write_new(core.home / "w04-plans.json", json.dumps({"W04 cli approval": plan}))
    before = container_ids(LOG)
    try:
        await core.start()
        await cli_approval(core, plan)  # Unchanged complete-parameter, denial and strict exit=1 assertions.
    finally:
        await core.stop()
        after = container_ids(LOG)
        save(EXECUTION / f"cli-close-containers-{core.launch}.json", {"before": before, "after": after})
        assert before == after
    record("cli_close_stage", passed=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["cli", "observe", "reclaim"])
    selected = parser.parse_args().stage
    asyncio.run(cli_close() if selected == "cli" else main(selected))
