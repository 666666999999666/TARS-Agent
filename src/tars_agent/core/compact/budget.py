from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tars_agent.core.config import LlmConfig

DEFAULT_CONTEXT_BUDGET = 32_768
DEFAULT_CONTEXT_MARGIN = 1_024
ESTIMATOR = "utf8_json_bytes_plus_message_framing_v1"


class ContextBudgetError(ValueError):
    """A request cannot safely retain its required context under the local policy."""


def message_estimate(message: dict[str, Any]) -> int:
    # Conservative local estimate, not a tokenizer or a verified endpoint window.
    return len(json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 32


def is_user_input(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return message.get("role") == "user" and not (
        isinstance(content, list)
        and any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)
    )


def validate_tool_pairs(messages: list[dict[str, Any]]) -> None:
    pending: set[str] = set()
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            raise ContextBudgetError("context_history_invalid: unsupported message role")
        content = message.get("content")
        blocks = content if isinstance(content, list) else []
        if any(not isinstance(block, dict) for block in blocks):
            raise ContextBudgetError("context_history_invalid: invalid content block")
        calls = [block.get("id") for block in blocks if block.get("type") == "tool_use"]
        results = [block.get("tool_use_id") for block in blocks
                   if block.get("type") == "tool_result"]
        if any(not isinstance(value, str) or not value for value in calls + results):
            raise ContextBudgetError("context_history_invalid: missing tool identifier")
        if pending:
            if role != "user" or len(results) != len(set(results)) or set(results) != pending:
                raise ContextBudgetError(
                    "context_history_invalid: tool result IDs do not match calls"
                )
            pending = set()
        elif results:
            raise ContextBudgetError("context_history_invalid: tool result without a matching call")
        if calls:
            if role != "assistant" or len(calls) != len(set(calls)):
                raise ContextBudgetError(
                    "context_history_invalid: duplicate or misplaced tool calls"
                )
            pending = set(calls)
    if pending:
        raise ContextBudgetError("context_history_invalid: tool calls are missing their results")


def history_groups(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for message in messages:
        if not groups or is_user_input(message):
            groups.append([])
        groups[-1].append(message)
    return groups


@dataclass(frozen=True)
class ContextBudget:
    total: int = DEFAULT_CONTEXT_BUDGET
    output_reserve: int = 8_192
    safety_margin: int = DEFAULT_CONTEXT_MARGIN

    def __post_init__(self) -> None:
        if (type(self.total) is not int or type(self.output_reserve) is not int
                or type(self.safety_margin) is not int or self.output_reserve <= 0
                or self.safety_margin < 0 or self.input_limit <= 0):
            raise ValueError("context budget must exceed output reserve plus nonnegative margin")

    @classmethod
    def from_config(cls, config: LlmConfig) -> ContextBudget:
        return cls(config.context_budget_tokens, config.max_tokens, config.context_safety_margin)

    @property
    def input_limit(self) -> int:
        return self.total - self.output_reserve - self.safety_margin

    def exceeded(self, stage: str, estimate: int) -> ContextBudgetError:
        return ContextBudgetError(
            f"context_budget_exceeded: {stage}; estimated_input={estimate}; "
            f"input_budget={self.input_limit}; output_reserve={self.output_reserve}; "
            f"safety_margin={self.safety_margin}; local_budget={self.total}"
        )

    def estimate(
        self, messages: list[dict[str, Any]], system: str, tools: list[dict[str, object]],
    ) -> int:
        fixed = json.dumps({"system": system, "tools": tools}, ensure_ascii=False,
                           separators=(",", ":")).encode("utf-8")
        return 64 + len(fixed) + sum(message_estimate(message) for message in messages)

    def require_fits(
        self, messages: list[dict[str, Any]], system: str, tools: list[dict[str, object]],
        *, stage: str,
    ) -> int:
        estimate = self.estimate(messages, system, tools)
        if estimate > self.input_limit:
            raise self.exceeded(stage, estimate)
        return estimate

    def select(
        self, messages: list[dict[str, Any]], current_start: int,
        system: str, tools: list[dict[str, object]],
    ) -> list[dict[str, Any]]:
        current = messages[current_start:]
        validate_tool_pairs(current)
        estimate = self.require_fits(current, system, tools, stage="current input and constraints")
        selected: list[list[dict[str, Any]]] = []
        groups = history_groups(messages[:current_start])
        for group in reversed(groups):
            required = estimate + sum(message_estimate(message) for message in group)
            if required > self.input_limit:
                if not selected:
                    raise self.exceeded("latest complete history group", required)
                break
            validate_tool_pairs(group)
            selected.append(group)
            estimate = required
        return [message for group in reversed(selected) for message in group] + current

TOOL_RESULT_LIMIT = 8_000
TOOL_RESULT_KEEP = 4_000


# Compatibility helper only; production context selection does not call this function.
def truncate_tool_results(
    messages: list[dict[str, Any]],
    limit: int = TOOL_RESULT_LIMIT,
    keep: int = TOOL_RESULT_KEEP,
) -> list[dict[str, Any]]:
    result = []
    for msg in messages:
        if msg.get("role") != "user":
            result.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            result.append(msg)
            continue
        new_blocks = []
        for block in content:
            if block.get("type") == "tool_result" and isinstance(block.get("content"), str):
                text = block["content"]
                if len(text) > limit:
                    omitted = len(text) - keep
                    block = dict(block)
                    block["content"] = (
                        text[:keep]
                        + f"\n[... {omitted} chars omitted. Full output in run events.]"
                    )
            new_blocks.append(block)
        result.append({**msg, "content": new_blocks})
    return result
