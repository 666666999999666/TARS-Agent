"""The single, explicit prompt difference in the AppWorld completion experiment."""

from __future__ import annotations

import hashlib
import json
from typing import Any

TOOL_NAME = "appworld__appworld_execute"
COMPARISON_ID = "completion-contract-v1"
_COMPLETION_A = "apis.supervisor.complete_task(answer=...) when appropriate. "
_COMPLETION_B = (
    "apis.supervisor.complete_task(...), following the task's answer requirements. "
    "Pass answer only when the task requests an answer; for action-only tasks omit answer. "
    "Do not put a narrative completion summary in answer. "
)


def prompt_template(variant: str = "A") -> str:
    if variant not in {"A", "B"}:
        raise ValueError("unknown AppWorld completion prompt variant")
    return (
        f"Solve the current AppWorld task using only {TOOL_NAME}. "
        "It runs Python with persistent variables and the public `apis` object. "
        "Discover APIs using apis.api_docs.show_app_descriptions(), "
        "apis.api_docs.show_api_descriptions(app_name=...), and "
        "apis.api_docs.show_api_doc(app_name=..., api_name=...). "
        "Print API results you need. Follow documented APIs and explicitly finish via "
        + (_COMPLETION_A if variant == "A" else _COMPLETION_B)
        + "Do not access filesystem, evaluation data, internal modules or real networks. "
        "Use the task world's time, not the host clock.\nPublic task information:\n"
    )


def template_sha256(variant: str = "A") -> str:
    return hashlib.sha256(prompt_template(variant).encode("utf-8")).hexdigest()


def task_prompt(public_task: dict[str, Any], variant: str = "A") -> str:
    return prompt_template(variant) + json.dumps(public_task, ensure_ascii=False)
