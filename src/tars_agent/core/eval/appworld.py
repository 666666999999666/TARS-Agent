"""Opt-in AppWorld orchestration with owned Docker resources and durable checkpoints."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Mapping
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values

from tars_agent.core.config import TarsConfig
from tars_agent.core.control import CoreHomeLock
from tars_agent.core.eval.appworld_prompts import COMPARISON_ID, template_sha256
from tars_agent.core.eval.models import (
    AppWorldSpec,
    EvalRunResult,
    EvalStatus,
    EvalSuiteManifest,
    RunProvenance,
    TaskAttemptResult,
    TaskSelectionRecord,
    ToolMetrics,
    UsageMetrics,
    summarize_attempts,
)
from tars_agent.core.eval.provenance import safe_config_snapshot
from tars_agent.core.eval.runner import utc_now
from tars_agent.core.persistence.cost_budget import CostLedger
from tars_agent.core.persistence.request_budget import (
    RequestLedger,
    validate_existing_request_ledger,
)

OWNER_LABEL = "com.tars-agent.appworld.owner"
DEEPSEEK_MODEL = "deepseek-flash"
DEEPSEEK_PROFILE = "deepseek-flash-official-anthropic-v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/anthropic"
DEEPSEEK_PRIVATE_ENV = Path(__file__).resolve().parents[4] / "build/internship/deepseek-private.env"
_PAUSING_MODEL_REASONS = frozenset({
    "llm_rate_limited", "llm_model_mismatch", "llm_request_budget_exhausted",
})


def load_deepseek_config(
    *, private_env: Path | None = None, environment: Mapping[str, str] | None = None,
) -> TarsConfig:
    """Load only the official DeepSeek credential and the existing shared request ledger."""
    source = os.environ if environment is None else environment
    path = DEEPSEEK_PRIVATE_ENV if private_env is None else private_env
    private = dotenv_values(path, interpolate=False) if path.is_file() else {}
    key = source.get("DEEPSEEK_API_KEY", private.get("DEEPSEEK_API_KEY"))
    if not key or not key.strip() or any(character.isspace() for character in key):
        raise ValueError("DeepSeek requires its dedicated DEEPSEEK_API_KEY")
    ledger_value = source.get(
        "DEEPSEEK_REQUEST_BUDGET_PATH", private.get("DEEPSEEK_REQUEST_BUDGET_PATH")
    )
    if not ledger_value:
        raise ValueError("DeepSeek requires an existing DEEPSEEK_REQUEST_BUDGET_PATH")
    ledger = validate_existing_request_ledger(Path(ledger_value))
    limit = source.get("DEEPSEEK_REQUEST_LIMIT", private.get("DEEPSEEK_REQUEST_LIMIT"))
    if limit is not None and (not limit.isdecimal() or int(limit) <= 0):
        raise ValueError("DeepSeek request limit must be a positive cumulative request count")
    config = TarsConfig()
    config.llm.default_model = DEEPSEEK_MODEL
    config.llm.expected_model = DEEPSEEK_MODEL
    config.llm.base_url = DEEPSEEK_BASE_URL
    config.llm.api_key = key
    config.llm.anthropic_api_key = ""
    config.llm.attempts = 2
    config.llm.retry_delay_s = 1.0
    config.llm.context_budget_tokens = 131072
    config.llm.request_budget_path = ledger
    config.llm.request_limit = int(limit) if limit is not None else None
    cost_path = source.get("DEEPSEEK_COST_BUDGET_PATH", private.get("DEEPSEEK_COST_BUDGET_PATH"))
    if cost_path:
        candidate = Path(cost_path).expanduser()
        if not candidate.is_absolute() or not candidate.is_file():
            raise ValueError("DeepSeek cost budget must be an existing absolute file")
        config.llm.cost_budget_path = candidate.resolve(strict=True)
        CostLedger(config.llm.cost_budget_path).summary()
    config.compaction.auto_threshold = 0
    config.trace.include_llm_payload = False
    config.mcp.servers = []
    validate_deepseek_config(config)
    return config


def validate_deepseek_config(config: TarsConfig) -> None:
    if (config.llm.default_model != DEEPSEEK_MODEL
            or config.llm.expected_model != DEEPSEEK_MODEL
            or config.llm.base_url != DEEPSEEK_BASE_URL or not config.llm.api_key
            or config.llm.anthropic_api_key or config.llm.attempts != 2
            or config.llm.retry_delay_s != 1.0):
        raise ValueError(
            "DeepSeek model, official endpoint, credential, guard or retry policy changed"
        )


def model_protocol_binding(config: TarsConfig) -> dict[str, Any]:
    binding: dict[str, Any] = {
        "protocol": "anthropic_messages",
        "requested_model": config.llm.default_model,
        "response_model_guard": {
            "expected_model": config.llm.expected_model,
            "enabled": bool(config.llm.expected_model),
            "missing_or_different_model": "reject" if config.llm.expected_model else "not_enforced",
        },
        "thinking": {"request_parameter": "omitted", "behavior": "provider_default"},
    }
    if config.llm.default_model == DEEPSEEK_MODEL and config.llm.base_url == DEEPSEEK_BASE_URL:
        binding.update(provider_profile=DEEPSEEK_PROFILE, base_url=DEEPSEEK_BASE_URL,
                       model_fallback="not_requested")
    return binding


async def complete_owned[T](operation: Awaitable[T]) -> T:
    """Let an owned mutation record its identity before propagating cancellation."""
    task = asyncio.ensure_future(operation)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError()
    return result


def worker_config(config: TarsConfig, job: dict[str, Any]) -> TarsConfig:
    import copy

    selected = copy.deepcopy(config)
    if job.get("profile") == DEEPSEEK_PROFILE:
        validate_deepseek_config(selected)
        if (
            job["model"] != DEEPSEEK_MODEL
            or Path(job["request_budget_path"]).resolve()
            != selected.llm.request_budget_path.resolve()
            or (selected.llm.request_limit is not None
                and int(job["request_limit"]) > selected.llm.request_limit)
        ):
            raise ValueError(
                "DeepSeek worker cannot change its model, existing ledger or trusted request cap"
            )
    elif job.get("profile") is not None:
        raise ValueError("unknown AppWorld model profile")
    selected.llm.default_model = str(job["model"])
    selected.llm.request_limit = int(job["request_limit"])
    selected.llm.request_budget_path = Path(job["request_budget_path"]).resolve()
    if job.get("cost_budget_path") is not None:
        expected_cost_path = getattr(selected.llm, "cost_budget_path", None)
        if (expected_cost_path is None
                or Path(job["cost_budget_path"]).resolve(strict=True)
                != expected_cost_path.resolve(strict=True)):
            raise ValueError("AppWorld worker cannot change its trusted cost ledger")
    elif getattr(selected.llm, "cost_budget_path", None) is not None:
        raise ValueError("AppWorld worker is missing its trusted cost ledger binding")
    selected.agent.max_steps = int(job["max_steps"])
    selected.compaction.auto_threshold = 0
    selected.mcp.servers = []
    return selected


def config_fingerprint(config: TarsConfig) -> str:
    snapshot = safe_config_snapshot(config)
    transport: dict[str, Any] = {
        "model_protocol": model_protocol_binding(config),
        "endpoint_sha256": hashlib.sha256(
            (config.llm.base_url or "https://api.anthropic.com").encode()
        ).hexdigest(),
        "connect_timeout_s": config.llm.connect_timeout_s,
        "read_timeout_s": config.llm.read_timeout_s,
        "write_timeout_s": config.llm.write_timeout_s,
        "pool_timeout_s": config.llm.pool_timeout_s,
        "ledger_path_sha256": hashlib.sha256(
            str(config.llm.request_budget_path.resolve()).encode()
        ).hexdigest(),
    }
    cost_path = getattr(config.llm, "cost_budget_path", None)
    if cost_path is not None:
        transport["cost_ledger_path_sha256"] = hashlib.sha256(
            str(cost_path.resolve(strict=True)).encode()
        ).hexdigest()
        cost_summary = CostLedger(cost_path).summary()
        transport["cost_policy"] = {
            key: cost_summary[key] for key in ("budget_id", "policy")
        }
    snapshot["transport"] = transport
    return digest_json(snapshot)


def canonical_mount(value: str) -> str:
    normalized = value.replace("\\", "/").rstrip("/")
    prefix = "/run/desktop/mnt/host/"
    if normalized.startswith(prefix):
        tail = normalized[len(prefix) :]
        if len(tail) > 1 and tail[1] == "/":
            normalized = tail[0] + ":" + tail[1:]
    return normalized.casefold() if len(normalized) > 1 and normalized[1] == ":" else normalized


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def data_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def command(args: list[str], *, timeout: float = 120) -> str:
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode:
        raise RuntimeError(f"{args[0]} failed ({completed.returncode}): {completed.stderr[-2000:]}")
    return completed.stdout.strip()


class WorldContainer:
    """Durable creation intent and exact identity checks for this world's resources."""

    def __init__(self, image_id: str, data: Path, output: Path, owner: str) -> None:
        self.image_id = image_id
        self.data = data.resolve(strict=True)
        self.output = output.resolve()
        self.owner = owner
        self.container_id: str | None = None
        self.name: str | None = None
        self.url = ""
        self.network: str | None = None
        self.network_id: str | None = None
        self.record = self.output / "container.json"

    def _record(self, phase: str) -> None:
        write_json(
            self.record,
            {
                "owner": self.owner,
                "name": self.name,
                "container_id": self.container_id,
                "image_id": self.image_id,
                "network": self.network,
                "network_id": self.network_id,
                "data": str(self.data),
                "output": str(self.output),
                "url": self.url,
                "phase": phase,
            },
        )

    def _restore(self) -> bool:
        if not self.record.exists():
            return False
        previous = json.loads(self.record.read_text(encoding="utf-8"))
        if previous.get("phase") == "removed":
            return False
        if (
            previous.get("owner") != self.owner
            or previous.get("image_id") != self.image_id
            or previous.get("data") != str(self.data)
            or previous.get("output") != str(self.output)
            or not previous.get("name")
            or not previous.get("network")
        ):
            raise RuntimeError("Uncertain prior AppWorld identity; refusing recovery")
        self.name = previous["name"]
        self.container_id = previous.get("container_id")
        self.network = previous["network"]
        self.network_id = previous.get("network_id")
        return True

    async def _find_container(self) -> bool:
        selector = f"id={self.container_id}" if self.container_id else f"name=^/{self.name}$"
        found = (
            await asyncio.to_thread(
                command,
                [
                    "docker",
                    "ps",
                    "--all",
                    "--quiet",
                    "--no-trunc",
                    "--filter",
                    selector,
                ],
            )
        ).splitlines()
        if not found:
            self.container_id = None
            return False
        if len(found) != 1 or (self.container_id is not None and found[0] != self.container_id):
            raise RuntimeError("ambiguous AppWorld container identity")
        self.container_id = found[0]
        await self.inspect()
        self._record("reconciled")
        return True

    async def start(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        if self._restore():
            await self.close()
        self.name = f"tars-appworld-{uuid.uuid4().hex[:16]}"
        self.network = self.name + "-network"
        # Both names and mount identity survive a crash before Docker returns an ID.
        self._record("network_creating")
        try:

            async def create_network() -> None:
                self.network_id = await asyncio.to_thread(
                    command,
                    [
                        "docker",
                        "network",
                        "create",
                        "--label",
                        f"{OWNER_LABEL}={self.owner}",
                        str(self.network),
                    ],
                )
                self._record("network_ready")

            await complete_owned(create_network())

            async def create_container() -> None:
                self._record("container_creating")
                self.container_id = await asyncio.to_thread(
                    command,
                    [
                        "docker",
                        "create",
                        "--name",
                        str(self.name),
                        "--label",
                        f"{OWNER_LABEL}={self.owner}",
                        "--network",
                        str(self.network),
                        "--memory",
                        "3g",
                        "--cpus",
                        "2",
                        "--publish",
                        "127.0.0.1::8000",
                        "--mount",
                        f"type=bind,source={self.data},target=/run/data,readonly",
                        "--mount",
                        f"type=bind,source={self.output},target=/run/experiments",
                        self.image_id,
                        "environment",
                        "--port",
                        "8000",
                    ],
                )
                self._record("created")

            await complete_owned(create_container())

            async def start_container() -> None:
                await asyncio.to_thread(command, ["docker", "start", str(self.container_id)])
                self._record("started")

            await complete_owned(start_container())
            info = await self.inspect()
            ports = info["NetworkSettings"]["Ports"].get("8000/tcp") or []
            if len(ports) != 1 or ports[0].get("HostIp") != "127.0.0.1":
                raise RuntimeError("AppWorld service must expose exactly one loopback port")
            self.url = f"http://127.0.0.1:{ports[0]['HostPort']}"
            self._record("running")
            async with httpx.AsyncClient(timeout=3, trust_env=False) as client:
                for _ in range(60):
                    try:
                        response = await client.get(self.url + "/")
                        response.raise_for_status()
                        return
                    except (httpx.HTTPError, OSError):
                        await asyncio.sleep(1)
            raise RuntimeError("AppWorld environment did not become ready")
        except BaseException:
            await complete_owned(self.close())
            raise

    async def inspect(self) -> dict[str, Any]:
        if self.container_id is None:
            raise RuntimeError("AppWorld container identity is not confirmed")
        items = json.loads(
            await asyncio.to_thread(
                command,
                [
                    "docker",
                    "inspect",
                    self.container_id,
                ],
            )
        )
        item: dict[str, Any] = items[0]
        if (
            item["Id"] != self.container_id
            or item["Image"] != self.image_id
            or (item["Config"].get("Labels") or {}).get(OWNER_LABEL) != self.owner
            or not self.name
            or item.get("Name") != "/" + self.name
        ):
            raise RuntimeError("AppWorld container ownership mismatch; refusing operation")
        mounts = {mount["Destination"]: mount for mount in item["Mounts"]}
        if set(mounts) != {"/run/data", "/run/experiments"}:
            raise RuntimeError("AppWorld container mount set mismatch")
        for target, expected, writable in (
            ("/run/data", self.data, False),
            ("/run/experiments", self.output, True),
        ):
            actual = mounts[target]
            if (
                actual.get("Type") != "bind"
                or canonical_mount(actual["Source"]) != canonical_mount(str(expected))
                or actual["RW"] is not writable
            ):
                raise RuntimeError("AppWorld container mount identity mismatch")
        networks = item["NetworkSettings"]["Networks"]
        if item["HostConfig"]["NetworkMode"] != self.network or (
            networks and set(networks) != {self.network}
        ):
            raise RuntimeError("AppWorld container network identity mismatch")
        attached_id = networks.get(self.network, {}).get("NetworkID")
        if (
            self.network_id
            and attached_id
            and attached_id != self.network_id
            or item["State"]["Running"]
            and attached_id != self.network_id
        ):
            raise RuntimeError("AppWorld network ID mismatch")
        return item

    async def close(self) -> None:
        async def remove_owned() -> None:
            if self.container_id is None and self.network is None and self.name is None:
                if not self._restore():
                    return
            if self.name or self.container_id:
                if await self._find_container():
                    await asyncio.to_thread(
                        command,
                        [
                            "docker",
                            "rm",
                            "--force",
                            str(self.container_id),
                        ],
                    )
                    if await self._find_container():
                        raise RuntimeError("AppWorld container still exists after removal")
                    self._record("container_removed")
            if self.network:
                names = (
                    await asyncio.to_thread(
                        command,
                        [
                            "docker",
                            "network",
                            "ls",
                            "--format",
                            "{{.Name}}",
                            "--filter",
                            f"name=^{self.network}$",
                        ],
                    )
                ).splitlines()
                if self.network in names:
                    info = json.loads(
                        await asyncio.to_thread(
                            command,
                            [
                                "docker",
                                "network",
                                "inspect",
                                self.network,
                            ],
                        )
                    )[0]
                    if (
                        (info.get("Labels") or {}).get(OWNER_LABEL) != self.owner
                        or info.get("Containers")
                        or (self.network_id and info["Id"] != self.network_id)
                    ):
                        raise RuntimeError("AppWorld network ownership/busy check failed")
                    await asyncio.to_thread(command, ["docker", "network", "rm", self.network])
                self.network = None
                self.network_id = None
            self.container_id = None
            self.name = None
            self._record("removed")

        await complete_owned(remove_owned())


async def image_metadata(spec: AppWorldSpec, data: Path) -> tuple[str, dict[str, Any]]:
    info = json.loads(await asyncio.to_thread(command, ["docker", "image", "inspect", spec.image]))[
        0
    ]
    labels = info.get("Config", {}).get("Labels", {}) or {}
    if labels.get("org.opencontainers.image.revision") != spec.source_ref:
        raise ValueError("AppWorld image does not match the pinned source revision")
    image_id: str = info["Id"]
    # This preparation container executes no model/user code and removes itself on exit.
    code = (
        "import json,appworld; from appworld import load_task_ids; "
        "from appworld.task import task_id_to_generator_id,task_id_to_number; "
        "from appworld.common.constants import DATA_VERSION,DB_VERSION; "
        f"ids=load_task_ids({spec.dataset!r}); "
        "print(json.dumps(dict(version=appworld.__version__,data_version=DATA_VERSION,"
        "db_version=DB_VERSION,task_ids=ids,"
        "scenarios={i:[task_id_to_generator_id(i),task_id_to_number(i)] for i in ids})))"
    )
    result = await asyncio.to_thread(
        command,
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "python",
            "--label",
            f"{OWNER_LABEL}=metadata",
            "--mount",
            f"type=bind,source={data},target=/run/data,readonly",
            image_id,
            "-c",
            code,
        ],
    )
    metadata: dict[str, Any] = json.loads(result)
    ids = metadata["task_ids"]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("invalid/duplicate official task IDs")
    validate_task_set(spec.dataset, metadata)
    return image_id, metadata


def validate_task_set(dataset: str, metadata: dict[str, Any]) -> None:
    ids = metadata["task_ids"]
    groups: dict[str, list[int]] = defaultdict(list)
    for task_id in ids:
        scenario, number = metadata["scenarios"][task_id]
        groups[str(scenario)].append(int(number))
    if dataset == "test_normal" and (
        len(ids) != 168
        or len(groups) != 56
        or any(sorted(numbers) != [1, 2, 3] for numbers in groups.values())
    ):
        raise ValueError(
            "pinned test_normal requires 168 tasks and 56 complete three-task scenarios"
        )
    metadata["task_count"] = len(ids)
    metadata["scenario_count"] = len(groups)
    metadata["scenario_variants_complete"] = all(
        sorted(numbers) == [1, 2, 3] for numbers in groups.values()
    )


def selected_scenarios_complete(selected: list[str], metadata: dict[str, Any]) -> bool:
    """Check only official task-ID groups; never inspect task contents or scores."""
    groups: dict[str, list[int]] = defaultdict(list)
    for task_id in selected:
        scenario, number = metadata["scenarios"][task_id]
        groups[str(scenario)].append(int(number))
    return bool(groups) and all(sorted(numbers) == [1, 2, 3] for numbers in groups.values())


def reported_official_metrics(
    aggregate: dict[str, Any], *, selected_scenario_variants_complete: bool,
) -> dict[str, Any]:
    metrics = dict(aggregate)
    if not selected_scenario_variants_complete:
        metrics["scenario_goal_completion"] = None
    return metrics


def select_task_ids(spec: AppWorldSpec, metadata: dict[str, Any]) -> list[str]:
    official = list(metadata["task_ids"])
    if spec.task_ids is None:
        return official if spec.task_limit is None else official[:spec.task_limit]
    if spec.dataset == "test_normal" or spec.task_limit is not None:
        raise ValueError("explicit task_ids are allowed only for train/dev without task_limit")
    selected = list(spec.task_ids)
    if not selected or len(selected) != len(set(selected)) or not set(selected).issubset(official):
        raise ValueError("explicit AppWorld task_ids must be unique official dataset members")
    return selected


def checkpoint_binding(
    manifest: EvalSuiteManifest, provenance: RunProvenance, metadata: dict[str, Any], image_id: str
) -> dict[str, Any]:
    return {
        "manifest": manifest.model_dump(mode="json"),
        "tree_digest": provenance.tree_digest,
        "config_hash": provenance.config_hash,
        "image_id": image_id,
        "data": metadata,
    }


def dependency_fingerprint(site_packages: Path) -> str:
    installed = []
    for distribution in importlib.metadata.distributions(path=[str(site_packages)]):
        installed.append((distribution.metadata["Name"], distribution.version))
    records: dict[str, str] = {}
    for pattern in ("*.pth", "*.dist-info/METADATA", "*.dist-info/RECORD"):
        for path in sorted(site_packages.glob(pattern)):
            records[path.relative_to(site_packages).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return digest_json({"installed": sorted(installed), "records": records})


def python_environment() -> dict[str, str]:
    # The uv venv python.exe can be a trampoline. Own the base interpreter process itself.
    executable = Path(getattr(sys, "_base_executable", sys.executable)).resolve(strict=True)
    site_packages = Path(str(importlib.metadata.distribution("mcp").locate_file(""))).resolve()
    return {
        "base_executable": str(executable),
        "site_packages": str(site_packages),
        "dependencies_digest": dependency_fingerprint(site_packages),
        "interpreter_digest": hashlib.sha256(executable.read_bytes()).hexdigest(),
    }


def module_argv(module: str, arguments: list[str], environment: dict[str, str]) -> list[str]:
    code = (
        "import runpy,site,sys; site.addsitedir(sys.argv[1]); "
        "name=sys.argv[2]; sys.argv=[name]+sys.argv[3:]; "
        "runpy.run_module(name,run_name='__main__')"
    )
    return [
        environment["base_executable"],
        "-c",
        code,
        environment["site_packages"],
        module,
        *arguments,
    ]


def _worker_argv(job_path: Path, output: Path) -> list[str]:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    return module_argv(
        "tars_agent.core.eval.appworld_worker",
        [str(job_path), str(output)],
        job["python_environment"],
    )


def _powershell(script: str) -> str:
    executable = (
        Path(os.environ.get("SystemRoot", r"C:\Windows"))
        / "System32"
        / "WindowsPowerShell"
        / "v1.0"
        / "powershell.exe"
    )
    return command(
        [
            str(executable),
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); " + script,
        ]
    )


def process_identity(pid: int) -> dict[str, Any] | None:
    if os.name == "nt":
        text = _powershell(
            f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}'; "
            "if($null -ne $p){ @{pid=[int]$p.ProcessId; "
            "started=$p.CreationDate.ToUniversalTime().Ticks; "
            "command_line=$p.CommandLine; executable=$p.ExecutablePath} "
            "| ConvertTo-Json -Compress }"
        )
        return json.loads(text) if text else None
    path = Path("/proc") / str(int(pid))
    try:
        return {
            "pid": pid,
            "started": path.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19],
            "argv": path.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0"),
            "executable": str(path.joinpath("exe").resolve(strict=True)),
        }
    except FileNotFoundError:
        return None


def _matching_workers(argv: list[str]) -> list[int]:
    if os.name == "nt":
        expected = subprocess.list2cmdline(argv).replace("'", "''")
        found = _powershell(
            f"$expected='{expected}'; Get-CimInstance Win32_Process | "
            "Where-Object { $_.CommandLine -eq $expected } | ForEach-Object { $_.ProcessId }"
        )
        return [int(pid) for pid in found.splitlines() if pid.strip()]
    result = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            identity = process_identity(int(path.name))
            if identity and identity.get("argv") == argv:
                result.append(int(path.name))
    return result


async def stop_recorded_worker(directory: Path) -> None:
    record_path = directory / "process.json"
    if not record_path.exists():
        return
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("phase") == "finished":
        return
    expected_argv = _worker_argv(directory / "job.json", directory / "result.json")
    if record.get("argv") != expected_argv:
        raise RuntimeError("recorded worker command does not belong to this attempt")
    identity = record.get("identity")
    if identity is not None:
        current = await asyncio.to_thread(process_identity, int(identity["pid"]))
        if current is None or current != identity:
            return  # A reused PID is not this attempt's worker.
    else:
        matches = await asyncio.to_thread(_matching_workers, record["argv"])
        if len(matches) > 1:
            raise RuntimeError("ambiguous orphan AppWorld worker identity")
        if not matches:
            return
        identity = await asyncio.to_thread(process_identity, matches[0])
        if identity is None:
            return
    # Check the creation identity again immediately before terminating only this worker tree.
    if await asyncio.to_thread(process_identity, int(identity["pid"])) != identity:
        return
    matches_command = (
        identity.get("command_line") == subprocess.list2cmdline(expected_argv)
        if os.name == "nt"
        else identity.get("argv") == expected_argv
    )
    if not matches_command:
        raise RuntimeError("worker process command identity mismatch")
    if os.name == "nt":
        await asyncio.to_thread(command, ["taskkill", "/PID", str(identity["pid"]), "/T", "/F"])
    else:
        import signal

        os.kill(int(identity["pid"]), signal.SIGTERM)
    write_json(record_path, {**record, "phase": "retired", "identity": identity})


def validate_worker_result(raw: dict[str, Any], job: dict[str, Any]) -> None:
    for key in ("task_id", "experiment_name", "job_fingerprint"):
        if raw.get(key) != job.get(key):
            raise ValueError(f"AppWorld worker result binding mismatch: {key}")
    for key in ("prompt_variant", "prompt_template_sha256"):
        if key in job and raw.get(key) != job[key]:
            raise ValueError(f"AppWorld worker result binding mismatch: {key}")


async def run_worker_process(job: dict[str, Any], directory: Path) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    job_path, output = directory / "job.json", directory / "result.json"
    write_json(job_path, job)
    argv = _worker_argv(job_path, output)
    record_path = directory / "process.json"
    write_json(record_path, {"phase": "starting", "argv": argv})
    process: asyncio.subprocess.Process | None = None
    with (directory / "worker.log").open("wb") as log:

        async def launch() -> None:
            nonlocal process
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=log,
                stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            identity = await asyncio.to_thread(process_identity, process.pid)
            write_json(
                record_path,
                {"pid": process.pid, "identity": identity, "argv": argv, "phase": "running"},
            )

        try:
            await complete_owned(launch())
            assert process is not None
            async with asyncio.timeout(float(job["task_timeout_s"]) + 300):
                await process.wait()
        except (TimeoutError, asyncio.CancelledError) as exc:

            async def retire() -> None:
                if process is not None and process.returncode is None:
                    if os.name == "nt":
                        await asyncio.to_thread(
                            command,
                            [
                                "taskkill",
                                "/PID",
                                str(process.pid),
                                "/T",
                                "/F",
                            ],
                        )
                    else:
                        process.kill()
                    await process.wait()
                write_json(record_path, {"phase": "finished", "argv": argv})

            await complete_owned(retire())
            if isinstance(exc, asyncio.CancelledError):
                raise
            return {
                "task_id": job["task_id"],
                "experiment_name": job["experiment_name"],
                "job_fingerprint": job["job_fingerprint"],
                **prompt_binding(job),
                "saved": False,
                "closed": False,
                "error": "worker_timeout",
            }
    write_json(
        record_path,
        {
            "phase": "finished",
            "argv": argv,
            "returncode": process.returncode if process else None,
            "pid": process.pid if process else None,
        },
    )
    if not output.exists():
        return {
            "task_id": job["task_id"],
            "experiment_name": job["experiment_name"],
            "job_fingerprint": job["job_fingerprint"],
            **prompt_binding(job),
            "saved": False,
            "closed": False,
            "error": "worker_result_missing",
        }
    raw: dict[str, Any] = json.loads(output.read_text(encoding="utf-8"))
    validate_worker_result(raw, job)
    if process is None or raw.get("worker_pid") != process.pid:
        raise ValueError("AppWorld worker PID differs from the owned interpreter process")
    return raw


_OUTPUT_MARKER = ".tars-checkpoint.json"


def output_digest(root: Path) -> str:
    if root.is_symlink() or root.is_junction() or root.resolve() != root.absolute():
        raise ValueError("AppWorld output root is not owned")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if (
            path.is_symlink()
            or path.is_junction()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError("AppWorld output contains an unowned link")
        if path.is_file() and path != root / _OUTPUT_MARKER:
            digest.update(path.relative_to(root).as_posix().encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def publish_output(source: Path, canonical: Path, identity: dict[str, Any]) -> None:
    if not (source / "dbs").is_dir():
        raise RuntimeError("saved AppWorld task database is missing")
    expected = {"binding": identity, "output_digest": output_digest(source)}
    if canonical.exists():
        marker = canonical / _OUTPUT_MARKER
        if (
            not marker.is_file()
            or json.loads(marker.read_text(encoding="utf-8")) != expected
            or output_digest(canonical) != expected["output_digest"]
        ):
            raise RuntimeError("existing canonical output does not match this completed attempt")
        return
    canonical.parent.mkdir(parents=True, exist_ok=True)
    staging = canonical.with_name(f".{canonical.name}-{identity['repetition']}.pending")
    if staging.exists():
        intent = staging / "intent.json"
        if (
            staging.is_symlink()
            or staging.resolve() != staging.absolute()
            or not intent.is_file()
            or json.loads(intent.read_text(encoding="utf-8")) != identity
        ):
            raise RuntimeError("unowned AppWorld output staging directory")
        shutil.rmtree(staging)
    staging.mkdir()
    write_json(staging / "intent.json", identity)
    payload = staging / "payload"
    shutil.copytree(source, payload)
    write_json(payload / _OUTPUT_MARKER, expected)
    if output_digest(payload) != expected["output_digest"]:
        raise RuntimeError("AppWorld output changed during canonical copy")
    payload.replace(canonical)
    # The intent is kept until the payload has been atomically published.
    (staging / "intent.json").unlink()
    staging.rmdir()


def scorable_attempt(raw: dict[str, Any]) -> bool:
    return bool(
        raw.get("saved") is True
        and raw.get("reason") not in _PAUSING_MODEL_REASONS
        and raw.get("closed") is True
        and raw.get("agent_started") is True
        and raw.get("run_id")
        and raw.get("infrastructure_error") is False
        and raw.get("run_terminal_status") in {"succeeded", "failed", "cancelled", "interrupted"}
        and (raw.get("model_requests") or {}).get("reserved", 0) > 0
    )


def attempt_result(raw: dict[str, Any], task_id: str, repetition: int) -> TaskAttemptResult:
    complete = scorable_attempt(raw)
    status = (
        EvalStatus.error
        if not complete or raw.get("error") or raw.get("cleanup_errors")
        else EvalStatus.passed
        if raw.get("run_terminal_status") == "succeeded"
        else EvalStatus.failed
    )
    return TaskAttemptResult(
        task_id=task_id,
        source_task_id=task_id,
        repetition=repetition,
        status=status,
        run_terminal_status=raw.get("run_terminal_status"),
        goal_completed=None,
        started_at=raw.get("started_at", utc_now()),
        finished_at=raw.get("finished_at", utc_now()),
        latency_ms=raw.get("latency_ms", 0),
        score=None,
        error_type=(raw.get("reason") if raw.get("reason") in _PAUSING_MODEL_REASONS
                    else "appworld_infrastructure_error" if not complete else raw.get("reason")),
        error=raw.get("error"),
        usage=UsageMetrics.model_validate(raw.get("usage", {})),
        tools=ToolMetrics.model_validate(raw.get("tools", {})),
        evaluation={
            "world_saved": complete,
            "official_goal_score": "pending_aggregate",
            "experiment_name": raw.get("experiment_name"),
            "model_calls_started": raw.get("model_calls_started"),
            "agent_started": raw.get("agent_started", False),
            "infrastructure_error": raw.get("infrastructure_error", True),
            "model_requests": raw.get("model_requests"),
            "usage_complete": raw.get("usage_complete"),
            "confirmed_usage": raw.get("confirmed_usage"),
            "cleanup_errors": raw.get("cleanup_errors", []),
            "model_protocol": raw.get("model_protocol"),
            "prompt_variant": raw.get("prompt_variant", "A"),
            "prompt_template_sha256": raw.get("prompt_template_sha256"),
            "prompt_sha256": raw.get("prompt_sha256"),
        },
    )


def prompt_binding(job: dict[str, Any]) -> dict[str, Any]:
    return {key: job[key] for key in ("prompt_variant", "prompt_template_sha256") if key in job}


def task_schedule(
    selected: list[str], *, paired: bool, variant: str = "A",
) -> list[tuple[str, str, str]]:
    """Return durable key, official task ID and variant, alternating which arm runs first."""
    if not paired:
        return [(task_id, task_id, variant) for task_id in selected]
    return [
        (f"{variant}:{task_id}", task_id, variant)
        for index, task_id in enumerate(sorted(selected))
        for variant in (("A", "B") if index % 2 == 0 else ("B", "A"))
    ]


async def run_appworld_suite(
    manifest: EvalSuiteManifest,
    config: TarsConfig,
    provenance: RunProvenance,
    artifact_dir: Path,
) -> EvalRunResult:
    artifact_dir = artifact_dir.resolve()
    lock = CoreHomeLock(artifact_dir / "controller-state")
    lock.acquire()
    try:
        return await _run_appworld_suite_locked(manifest, config, provenance, artifact_dir)
    finally:
        lock.release()


async def _run_appworld_suite_locked(
    manifest: EvalSuiteManifest,
    config: TarsConfig,
    provenance: RunProvenance,
    artifact_dir: Path,
) -> EvalRunResult:
    spec = manifest.appworld
    if spec is None:
        raise ValueError("AppWorld configuration is required")
    deepseek_profile = manifest.model_config_ref.reference == DEEPSEEK_PROFILE
    if deepseek_profile:
        validate_deepseek_config(config)
        if (spec.dataset not in {"train", "dev"}
                or spec.workers != 1 or spec.max_steps != 60 or spec.task_timeout_s != 900
                or spec.infrastructure_retries != 1 or spec.task_limit is not None
                or config.llm.max_tokens != 8192 or config.llm.context_budget_tokens != 131072):
            raise ValueError("DeepSeek comparison differs from the frozen paired contract")
        if config.llm.cost_budget_path is None:
            raise ValueError("DeepSeek AppWorld requires its initialized cost budget")
        CostLedger(config.llm.cost_budget_path).summary()
    data = (Path(spec.data_root).resolve() / "data").resolve(strict=True)
    image_id, metadata = await complete_owned(image_metadata(spec, data))
    metadata["data_digest"] = await asyncio.to_thread(data_digest, data)
    selected = select_task_ids(spec, metadata)
    if spec.paired_comparison or deepseek_profile:
        selected = sorted(selected)
    if deepseek_profile:
        from tars_agent.core.eval.appworld_comparison import validate_comparison_selection

        validate_comparison_selection(spec, selected, metadata)
    schedule = task_schedule(
        selected, paired=spec.paired_comparison, variant=spec.prompt_variant,
    )
    rows_by_key = {key: (task_id, variant) for key, task_id, variant in schedule}
    selected_variants_complete = selected_scenarios_complete(selected, metadata)
    binding = checkpoint_binding(manifest, provenance, metadata, image_id)
    binding["base_config_fingerprint"] = config_fingerprint(config)
    binding["python_environment"] = python_environment()
    binding["model_protocol"] = model_protocol_binding(config)
    binding["completion_prompts"] = {
        "comparison": COMPARISON_ID if spec.paired_comparison else None,
        "templates": {variant: template_sha256(variant)
                      for variant in (("A", "B") if spec.paired_comparison
                                      else (spec.prompt_variant,))},
        "schedule": [list(row) for row in schedule],
    }
    freeze, checkpoint = artifact_dir / "freeze.json", artifact_dir / "checkpoint.json"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        frozen = json.loads(freeze.read_text(encoding="utf-8")) if freeze.exists() else None
        if state.get("binding", frozen) != binding or (frozen is not None and frozen != binding):
            raise ValueError("frozen benchmark inputs changed; use a new artifact directory")
        if frozen is None:
            write_json(freeze, binding)
    else:
        if freeze.exists() and json.loads(freeze.read_text(encoding="utf-8")) != binding:
            raise ValueError("frozen benchmark inputs changed; use a new artifact directory")
        count = RequestLedger(config.llm.request_budget_path).counts()["real"]
        allowance = (
            len(schedule) * spec.max_steps * config.llm.attempts * (1 + spec.infrastructure_retries)
        )
        limit = count + allowance
        if config.llm.request_limit is not None:
            limit = min(limit, config.llm.request_limit)
        if limit <= count:
            raise ValueError("trusted request limit has no remaining allowance")
        state = {
            "run_id": f"appworld-{uuid.uuid4().hex[:16]}",
            "started_at": utc_now(),
            "request_limit": limit,
            "requests_before": count,
            "results": {},
            "attempts": [],
            "inflight": {},
            "worlds": [],
            "binding": binding,
        }
        # A freeze-only crash is recoverable: the empty checkpoint is durable first.
        write_json(checkpoint, state)
        write_json(freeze, binding)
    state.setdefault("inflight", {})
    state.setdefault("worlds", [])
    # An explicit controller restart may resume, but cannot reset recorded attempt slots.
    state.pop("paused_reason", None)
    run_id = str(state["run_id"])
    def variant_root(variant: str) -> Path:
        return artifact_dir / "variants" / variant if spec.paired_comparison else artifact_dir

    for result_key, completed in state["results"].items():
        if result_key not in rows_by_key:
            raise ValueError("checkpoint contains an unknown task or prompt variant")
        task_id, variant = rows_by_key[result_key]
        canonical = variant_root(variant) / "official" / "outputs" / run_id / "tasks" / task_id
        marker = canonical / _OUTPUT_MARKER
        if (
            str(canonical.relative_to(artifact_dir)) != completed["output"]
            or not (canonical / "dbs").is_dir()
            or not marker.is_file()
        ):
            raise ValueError("completed checkpoint has no bound official output")
        receipt = json.loads(marker.read_text(encoding="utf-8"))
        if (
            receipt["binding"]["task_id"] != task_id
            or receipt["binding"]["run_id"] != run_id
            or receipt["binding"]["repetition"] != completed["attempt"]
            or (spec.paired_comparison and (
                receipt["binding"].get("prompt_variant") != variant
                or receipt["binding"].get("prompt_template_sha256") != template_sha256(variant)))
            or receipt["output_digest"] != completed["output_digest"]
            or output_digest(canonical) != completed["output_digest"]
        ):
            raise ValueError("completed checkpoint output changed")
    state_lock = asyncio.Lock()
    stop = asyncio.Event()
    containers: list[WorldContainer] = []

    def commit(row: dict[str, Any], raw: dict[str, Any]) -> None:
        job = row["job"]
        validate_worker_result(raw, job)
        task_id, repetition = row["task_id"], row["repetition"]
        result_key = row.get("result_key", task_id)
        variant = str(job.get("prompt_variant", "A"))
        if scorable_attempt(raw):
            source = (
                artifact_dir
                / row["world_output"]
                / "outputs"
                / job["experiment_name"]
                / "tasks"
                / task_id
            )
            if not (source / "dbs").is_dir():
                raw = {**raw, "saved": False, "error": "saved AppWorld database is missing"}
            else:
                canonical = (
                    variant_root(variant) / "official" / "outputs" / run_id / "tasks" / task_id
                )
                identity = {
                    "run_id": run_id,
                    "task_id": task_id,
                    "repetition": repetition,
                    "job_fingerprint": job["job_fingerprint"],
                    **prompt_binding(job),
                }
                publish_output(source, canonical, identity)
                state["results"][result_key] = {
                    "attempt": repetition,
                    "output": str(canonical.relative_to(artifact_dir)),
                    "output_digest": output_digest(canonical),
                }
        state["attempts"].append(attempt_result(raw, task_id, repetition).model_dump(mode="json"))
        state["inflight"].pop(result_key, None)
        if raw.get("reason") in _PAUSING_MODEL_REASONS:
            state["paused_reason"] = raw["reason"]
            stop.set()
        write_json(checkpoint, state)

    # Retire any old worker before retiring its world. Do not kill a reused PID.
    for row in state["inflight"].values():
        await complete_owned(stop_recorded_worker(artifact_dir / row["attempt_dir"]))
    for owner in state["worlds"]:
        world = WorldContainer(image_id, data, artifact_dir / owner["output"], owner["owner"])
        await world.close()
    for row in list(state["inflight"].values()):
        directory = artifact_dir / row["attempt_dir"]
        result_path = directory / "result.json"
        if result_path.is_file():
            raw = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            job = row["job"]
            raw = {
                "task_id": job["task_id"],
                "experiment_name": job["experiment_name"],
                "job_fingerprint": job["job_fingerprint"],
                **prompt_binding(job),
                "saved": False,
                "closed": False,
                "error": "controller interrupted before a durable worker result",
            }
        commit(row, raw)

    pending: asyncio.Queue[tuple[str, str, str]] = asyncio.Queue()
    for row in schedule:
        if row[0] not in state["results"]:
            pending.put_nowait(row)

    async def new_world(output: Path, owner: str) -> WorldContainer:
        world = WorldContainer(image_id, data, output, owner)
        containers.append(world)
        async with state_lock:
            record = {"output": str(output.relative_to(artifact_dir)), "owner": owner}
            if record not in state["worlds"]:
                state["worlds"].append(record)
            write_json(checkpoint, state)
        await world.start()
        return world

    async def worker(index: int) -> None:
        world: WorldContainer | None = None
        while not stop.is_set():
            try:
                result_key, task_id, variant = pending.get_nowait()
            except asyncio.QueueEmpty:
                return
            previous = sum(
                item["task_id"] == task_id
                and item["evaluation"].get("prompt_variant", "A") == variant
                for item in state["attempts"]
            )
            if previous >= 1 + spec.infrastructure_retries:
                async with state_lock:
                    state["paused_reason"] = "infrastructure_retries_exhausted"
                    write_json(checkpoint, state)
                raise RuntimeError("an AppWorld task exhausted its infrastructure retries")
            if world is None:
                world = await new_world(artifact_dir / f"world-{index}", f"{run_id}-{index}")
            for repetition in range(previous + 1, 2 + spec.infrastructure_retries):
                attempt_base = artifact_dir / "attempts"
                if spec.paired_comparison:
                    attempt_base /= variant
                attempt_dir = attempt_base / task_id / f"attempt-{repetition}"
                experiment = f"{run_id}-{variant}-{task_id}-r{repetition}"
                job = {
                    "task_id": task_id,
                    "experiment_name": experiment,
                    "environment_url": world.url,
                    "attempt_root": str(attempt_dir / "runtime"),
                    "model": config.llm.default_model,
                    "max_steps": spec.max_steps,
                    "task_timeout_s": spec.task_timeout_s,
                    "request_limit": state["request_limit"],
                    "request_budget_path": str(config.llm.request_budget_path),
                    "python_environment": binding["python_environment"],
                    "prompt_variant": variant,
                    "prompt_template_sha256": template_sha256(variant),
                }
                cost_path = getattr(config.llm, "cost_budget_path", None)
                if cost_path is not None:
                    job["cost_budget_path"] = str(cost_path)
                if deepseek_profile:
                    job["profile"] = DEEPSEEK_PROFILE
                job["config_fingerprint"] = config_fingerprint(worker_config(config, job))
                job["job_fingerprint"] = digest_json(job)
                row = {
                    "task_id": task_id,
                    "repetition": repetition,
                    "job": job,
                    "attempt_dir": str(attempt_dir.relative_to(artifact_dir)),
                    "world_output": str(world.output.relative_to(artifact_dir)),
                    "result_key": result_key,
                }
                async with state_lock:
                    state["inflight"][result_key] = row
                    write_json(checkpoint, state)
                raw = await run_worker_process(job, attempt_dir)
                async with state_lock:
                    commit(row, raw)
                    print(f"AppWorld progress {len(state['results'])}/{len(schedule)}", flush=True)
                if (raw.get("reason") == "llm_request_budget_exhausted"
                        or raw.get("reason") in _PAUSING_MODEL_REASONS):
                    stop.set()
                    break
                if result_key in state["results"]:
                    if raw.get("reason") == "llm_request_budget_exhausted":
                        stop.set()
                    break
                await world.close()
                if repetition > spec.infrastructure_retries:
                    stop.set()
                    break
                world = await new_world(
                    artifact_dir / f"world-{index}-retry-{variant}-{task_id}",
                    f"{run_id}-{index}-{variant}-{task_id}",
                )

    async def guarded_worker(index: int) -> None:
        try:
            await worker(index)
        except BaseException:
            stop.set()
            raise

    errors: list[str] = []
    try:
        outcomes = await asyncio.gather(
            *(guarded_worker(i) for i in range(min(spec.workers, pending.qsize()))),
            return_exceptions=True,
        )
        errors.extend(str(item) for item in outcomes if isinstance(item, BaseException))
    finally:

        async def close_all_worlds() -> None:
            for world in containers:
                try:
                    await world.close()
                except Exception as exc:
                    errors.append(str(exc))

        await complete_owned(close_all_worlds())
    attempts = [TaskAttemptResult.model_validate(item) for item in state["attempts"]]
    benchmark: dict[str, Any] = {
        "dataset": spec.dataset,
        "expected_tasks": len(schedule),
        "unique_tasks": len(selected),
        "completed_tasks": len(state["results"]),
        "official_metrics": None,
        "complete": False,
        "errors": errors,
        "paused_reason": state.get("paused_reason"),
        "model_protocol": binding["model_protocol"],
        "completion_prompts": binding["completion_prompts"],
        "interface": "remote-code via task-bound MCP bridge",
        "source_ref": spec.source_ref,
        "official_task_count": metadata.get("task_count", len(metadata["task_ids"])),
        "official_scenario_count": metadata.get("scenario_count"),
        "official_source_scenarios_complete": metadata.get("scenario_variants_complete"),
        "selected_scenario_variants_complete": selected_variants_complete,
        "scenario_goal_completion_note": (
            None if selected_variants_complete else
            "所选任务未覆盖每个场景的全部三个变体，SGC 不可解释；"
            "官方原始汇总保留在 official/aggregate.json。"
        ),
        "network_boundary": (
            "Dedicated Docker bridge with loopback-only management; container egress is not "
            "blocked by Docker. AppWorld syntax/execution safety guards remain enabled; "
            "no credentials or Docker socket are mounted."
        ),
    }
    if len(state["results"]) == len(schedule) and not errors:
        try:
            if spec.paired_comparison:
                from tars_agent.core.eval.appworld_comparison import comparison_summary

                metrics = {}
                for variant in ("A", "B"):
                    metrics[variant] = await complete_owned(
                        score_outputs(spec, image_id, data, variant_root(variant), run_id, selected)
                    )
                benchmark["comparison"] = comparison_summary(
                    metrics, selected, state["attempts"],
                )
                benchmark["complete"] = True
            else:
                aggregate = await complete_owned(
                    score_outputs(spec, image_id, data, artifact_dir, run_id, selected)
                )
                benchmark.update(
                    official_metrics=reported_official_metrics(
                        aggregate,
                        selected_scenario_variants_complete=selected_variants_complete,
                    ),
                    complete=True,
                )
        except Exception as exc:
            errors.append(f"official scoring failed: {exc}")
    return EvalRunResult(
        run_id=run_id,
        suite_id=manifest.suite_id,
        suite_name=manifest.name,
        suite_manifest="manifest.json",
        adapter="appworld",
        execution_mode="agent_tasks",
        started_at=state["started_at"],
        finished_at=utc_now(),
        provenance=provenance,
        model_config_ref=manifest.model_config_ref,
        selected_task_ids=selected,
        selection=TaskSelectionRecord(
            algorithm="paired_ab_ba" if spec.paired_comparison else "official_split",
            source_task_ids=selected,
            default_repetitions=1,
            selected_ids_digest=digest_json(selected),
        ),
        attempts=attempts,
        summary=summarize_attempts(attempts),
        benchmark=benchmark,
    )


async def score_outputs(
    spec: AppWorldSpec,
    image_id: str,
    data: Path,
    artifact_dir: Path,
    run_id: str,
    selected: list[str],
) -> dict[str, Any]:
    # The comparison uses only train/dev. The historical test adapter exposes aggregate only.
    details = spec.paired_comparison and spec.dataset in {"train", "dev"}
    official = artifact_dir / "official"
    scoring_binding: dict[str, Any] = {}
    metrics_relative = "metrics.json"
    if details:
        scoring_binding = {
            "image_id": image_id, "source_ref": spec.source_ref, "dataset": spec.dataset,
            "run_id": run_id, "task_ids": selected,
            "outputs_sha256": output_digest(official / "outputs" / run_id),
        }
        receipt = official / "scoring-receipt.json"
        if receipt.exists():
            recorded = json.loads(receipt.read_text(encoding="utf-8"))
            cached: dict[str, Any] = json.loads((official / "metrics.json").read_text())
            if (recorded["binding"] != scoring_binding
                    or recorded["metrics_sha256"] != digest_json(cached)
                    or json.loads((official / "aggregate.json").read_text())
                    != cached["aggregate"]):
                raise ValueError("completed official scoring changed; refusing to overwrite it")
            return cached
        attempt = official / "scoring-attempts" / uuid.uuid4().hex
        attempt.mkdir(parents=True)
        metrics_relative = (attempt / "metrics.json").relative_to(official).as_posix()
    code = (
        "import json; from pathlib import Path; from appworld.evaluator import evaluate_tasks; "
        f"r=evaluate_tasks({selected!r}, experiment_name={run_id!r}, include_details=True, "
        f"save_reports={not details!r}); "
        + (f"Path('/run/experiments/{metrics_relative}').write_text(json.dumps(r))" if details
           else "Path('/run/experiments/aggregate.json').write_text(json.dumps(r['aggregate']))")
    )
    await asyncio.to_thread(
        command,
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "python",
            "--label",
            f"{OWNER_LABEL}={run_id}-score",
            "--mount",
            f"type=bind,source={data},target=/run/data,readonly",
            "--mount",
            f"type=bind,source={artifact_dir / 'official'},target=/run/experiments",
            image_id,
            "-c",
            code,
        ],
        timeout=1800,
    )
    result: dict[str, Any] = json.loads(
        (official / (metrics_relative if details else "aggregate.json")).read_text()
    )
    if details:
        for name, value in (("metrics.json", result), ("aggregate.json", result["aggregate"])):
            path = official / name
            if path.exists():
                if json.loads(path.read_text()) != value:
                    raise ValueError("prior official score differs; refusing to overwrite it")
            else:
                write_json(path, value)
        write_json(official / "scoring-receipt.json", {
            "binding": scoring_binding, "metrics_sha256": digest_json(result),
        })
    return result
