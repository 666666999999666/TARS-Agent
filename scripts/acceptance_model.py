"""One explicitly invoked real model probe through the production request budget."""
from __future__ import annotations

import argparse
import asyncio
import json
import secrets
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tars_agent.core.config import get_config
from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm.provider import AnthropicProvider
from tars_agent.core.persistence.request_budget import RequestLedger


async def probe(output: Path) -> int:
    config = get_config()
    ledger = RequestLedger(config.llm.request_budget_path)
    before = ledger.counts()
    nonce = "TARS-CHECK-" + secrets.token_hex(8)
    result: dict[str, Any] = {
        "kind": "real_provider_probe", "started_at": datetime.now(UTC).isoformat(),
        "requested_model": config.llm.default_model, "endpoint": config.llm.base_url,
        "budget_before": before, "status": "failed", "docker_required_task": False,
    }
    provider: AnthropicProvider | None = None
    try:
        provider = AnthropicProvider.from_config(config.llm)
        async with asyncio.timeout(150):
            response = await provider.chat(
                [{"role": "user", "content": "请只原样回复这个标记，不添加其他文字：" + nonce}],
                [], EventBus(), "acceptance-provider-probe", system="Follow the user's exact output request.",
            )
        matched = response.text.strip() == nonce
        result.update(status="passed" if matched else "failed", nonce_matched=matched,
                      stop_reason=response.stop_reason, response=response.text,
                      usage=asdict(response.usage) if response.usage else None)
    except Exception as exc:
        detail = str(exc)
        for value in (config.llm.api_key, config.llm.anthropic_api_key):
            if value:
                detail = detail.replace(value, "[REDACTED]")
        result.update(error_type=type(exc).__name__, error=detail[:2000])
    finally:
        if provider is not None:
            await provider.close()
        result["budget_after"] = ledger.counts()
        result["finished_at"] = datetime.now(UTC).isoformat()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "budget_after": result["budget_after"],
                      "evidence": str(output)}, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return asyncio.run(probe(parser.parse_args().output))


if __name__ == "__main__":
    raise SystemExit(main())
