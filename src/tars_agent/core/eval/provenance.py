from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue

from tars_agent.core.config import TarsConfig
from tars_agent.core.eval.models import ModelConfigReference, RunProvenance


# 返回文件的 SHA-256；文件不存在时返回 None
def hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 在指定仓库执行只读 Git 命令并返回文本，失败时返回 None
def _run_git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


# 返回 Git 管理及未忽略的未跟踪文件路径，保留删除项供摘要标记
def _git_tree_paths(root: Path) -> list[Path] | None:
    output = _run_git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    if output is None:
        return None
    return [Path(item) for item in output.split("\0") if item]


# 判断路径是否位于排除列表中
def _is_excluded(path: Path, excluded: set[Path]) -> bool:
    return any(path == item or item in path.parents for item in excluded)


# 计算当前工作树内容摘要，覆盖已跟踪、删除和未忽略的未跟踪文件
def compute_tree_digest(root: Path, *, exclude: tuple[Path, ...] = ()) -> str:
    resolved_root = root.resolve()
    excluded: set[Path] = set()
    for path in exclude:
        candidate = path if path.is_absolute() else resolved_root / path
        excluded.add(candidate.resolve(strict=False))
    paths = _git_tree_paths(resolved_root)
    if paths is None:
        paths = [
            path.relative_to(resolved_root)
            for path in resolved_root.rglob("*")
            if path.is_file() and ".git" not in path.parts
        ]
    digest = hashlib.sha256()
    for relative in sorted(paths, key=lambda item: item.as_posix()):
        if any(part in {".git", ".venv", "node_modules", "build", "dist", "artifacts",
                        ".tars-baseline", ".tars", ".kama", "__pycache__"}
               for part in relative.parts):
            continue
        if ((relative.name == ".env" or (relative.name.startswith(".env.")
                                               and relative.name != ".env.example"))
                or relative.suffix in {".db", ".sqlite", ".sqlite3", ".log"}):
            continue
        absolute = (resolved_root / relative).resolve(strict=False)
        if not absolute.is_relative_to(resolved_root):
            continue
        if _is_excluded(absolute, excluded):
            continue
        encoded_path = relative.as_posix().encode("utf-8", errors="surrogatepass")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        if absolute.is_file():
            digest.update(b"\0file\0")
            with absolute.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            digest.update(b"\0missing\0")
    return digest.hexdigest()


# 生成不含凭证、MCP 请求头和环境变量的配置快照
def safe_config_snapshot(config: TarsConfig) -> dict[str, JsonValue]:
    return {
        "agent": {"max_steps": config.agent.max_steps},
        "llm": {
            "default_model": config.llm.default_model,
            "max_tokens": config.llm.max_tokens,
            "total_timeout_s": config.llm.total_timeout_s,
            "attempts": config.llm.attempts,
            "request_limit": config.llm.request_limit,
            "retry_delay_s": config.llm.retry_delay_s,
        },
        "trace": {
            "enabled": config.trace.enabled,
            "include_llm_payload": config.trace.include_llm_payload,
        },
        "permission": {"timeout_s": config.permission.timeout_s},
        "sandbox": {
            "mode": config.sandbox.mode,
            "image": config.sandbox.image,
            "memory": config.sandbox.memory,
            "memory_swap": config.sandbox.memory_swap,
            "cpus": config.sandbox.cpus,
            "pids_limit": config.sandbox.pids_limit,
            "nofile_limit": config.sandbox.nofile_limit,
            "output_limit_bytes": config.sandbox.output_limit_bytes,
            "network_mode": "disabled",
        },
        "compaction": {
            "auto_threshold": config.compaction.auto_threshold,
            "tool_result_limit": config.compaction.tool_result_limit,
            "tool_result_keep": config.compaction.tool_result_keep,
        },
        "mcp": {
            "server_count": len(config.mcp.servers),
            "server_names": [server.name for server in config.mcp.servers],
        },
    }


# 只读查询配置镜像的内容摘要；Docker 不可用、超时或镜像缺失时返回 None
def _docker_image_digest(config: TarsConfig) -> str | None:
    try:
        completed = subprocess.run(
            [
                config.sandbox.docker_binary,
                "image",
                "inspect",
                "--format={{.Id}}",
                config.sandbox.image,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    digest = completed.stdout.strip()
    return digest or None


# 采集一次 Eval 所需的仓库、依赖锁、平台、模型和脱敏配置证据
def collect_provenance(
    root: Path,
    config: TarsConfig,
    *,
    model: str | None,
    model_config_ref: ModelConfigReference,
    eval_config: dict[str, JsonValue] | None = None,
    exclude: tuple[Path, ...] = (),
) -> RunProvenance:
    resolved_root = root.resolve()
    git_sha = _run_git(resolved_root, "rev-parse", "HEAD")
    status = _run_git(
        resolved_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    config_snapshot = safe_config_snapshot(config)
    if eval_config is not None:
        config_snapshot["eval"] = eval_config
    config_hash = hashlib.sha256(
        json.dumps(
            config_snapshot,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return RunProvenance(
        collected_at=datetime.now(UTC).isoformat(),
        git_sha=git_sha,
        git_dirty=(bool(status) if status is not None else None),
        tree_digest=compute_tree_digest(resolved_root, exclude=exclude),
        lock_hash=hash_file(resolved_root / "uv.lock"),
        config_hash=config_hash,
        repository_root=".",
        platform=platform.system(),
        platform_release=platform.release(),
        python_version=platform.python_version(),
        model=model,
        model_config_ref=model_config_ref,
        docker_image=config.sandbox.image,
        docker_image_digest=_docker_image_digest(config),
        config=config_snapshot,
    )


__all__ = [
    "collect_provenance",
    "compute_tree_digest",
    "hash_file",
    "safe_config_snapshot",
]
