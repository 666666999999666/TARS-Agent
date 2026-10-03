from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from tars_agent.core.config import SandboxConfig
from tars_agent.core.control import CoreHomeLock

_LABEL = "com.tars-agent."
_FULL_ID = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_HOME_ID = re.compile(r"[0-9a-f]{64}\Z")
RECOVERY_COMMAND_TIMEOUT_S = 15.0


def _safe_diagnostic(value: str) -> str:
    for key, secret in os.environ.items():
        if secret and any(part in key.upper() for part in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"(?i)(https?://)[^/\s@]+@", r"\1[REDACTED]@", value)
    value = re.sub(r"(?i)(Bearer\s+)[^\s\"'<>]+", r"\1[REDACTED]", value)
    value = re.sub(
        r"(?i)((?:api[_-]?key|token|password|secret|authorization)[\"']?\s*[:=]\s*[\"']?)"
        r"[^\s\"'<>,;]+", r"\1[REDACTED]", value,
    )
    return value.strip()[:2048]


class DockerCommand(Protocol):
    async def __call__(
        self, args: list[str], *, timeout_s: float | None = None,
    ) -> tuple[int, str, str]: ...


class SandboxRecoveryError(RuntimeError):
    """An unresolved identity or operation prevents Core from becoming ready."""

    def __init__(
        self, reason: str, resource_id: str | None = None, *, container_id: str | None = None,
        operation: str | None = None, exit_code: int | None = None, diagnostic: str | None = None,
    ) -> None:
        self.reason = reason
        self.resource_id = resource_id
        self.container_id = container_id
        self.operation = operation
        self.exit_code = exit_code
        self.diagnostic = _safe_diagnostic(diagnostic) if diagnostic else None
        suffix = f" (resource={resource_id})" if resource_id else ""
        if container_id is not None:
            suffix += f" (container={container_id})"
        if operation is not None:
            suffix += f" (operation={operation}, exit_code={exit_code})"
        if self.diagnostic:
            suffix += f": {self.diagnostic}"
        super().__init__(f"sandbox recovery blocked: {reason}{suffix}")


class SandboxResourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    version: Literal[1]
    resource_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    home_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    launch_id: str = Field(min_length=1)
    instance_id: str = Field(pattern=r"^[0-9a-f]{12}$")
    run_id: str = Field(min_length=1)
    engine_id: str = Field(min_length=1)
    name: str
    workspace_root: str
    image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    phase: Literal["intent", "confirmed"] = "intent"
    container_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    created: str | None = None

    @model_validator(mode="after")
    def check_identity(self) -> Self:
        digest = hashlib.sha256(self.run_id.encode()).hexdigest()[:20]
        if self.name != f"tars-{self.instance_id}-{digest}":
            raise ValueError("container name does not match the recorded run")
        if not Path(self.workspace_root).is_absolute():
            raise ValueError("workspace must be absolute")
        if self.phase == "confirmed":
            if self.container_id is None or self.created is None:
                raise ValueError("confirmed resource lacks its complete identity")
            if datetime.fromisoformat(self.created.replace("Z", "+00:00")).tzinfo is None:
                raise ValueError("container creation time must include a timezone")
        elif self.container_id is not None or self.created is not None:
            raise ValueError("intent cannot contain a partial confirmed identity")
        return self

    @property
    def labels(self) -> dict[str, str]:
        return {
            _LABEL + "sandbox": "true",
            _LABEL + "purpose": "core-managed",
            _LABEL + "home": self.home_id,
            _LABEL + "launch": self.launch_id,
            _LABEL + "instance": self.instance_id,
            _LABEL + "run": self.run_id,
            _LABEL + "resource": self.resource_id,
        }


class SandboxResourceStore:
    """Crash-persistent records, usable only while this Core holds its HOME lock."""

    def __init__(self, lock: CoreHomeLock) -> None:
        self.lock = lock
        self.home_id = hashlib.sha256(os.path.normcase(str(lock.home)).encode()).hexdigest()
        self.directory = lock.home / "control" / "sandbox-resources"
        self.require_lock()

    def require_lock(self) -> None:
        if not self.lock.acquired:
            raise SandboxRecoveryError("home_lock_not_held")

    def load(self) -> list[SandboxResourceRecord]:
        self.require_lock()
        try:
            if not self.directory.exists():
                return []
            if self.directory.is_symlink():
                raise SandboxRecoveryError("resource_directory_is_symlink")
            records = [self.read(path.stem) for path in sorted(self.directory.glob("*.json"))]
            if len({record.name for record in records}) != len(records):
                raise SandboxRecoveryError("duplicate_container_name")
            return records
        except OSError as exc:
            raise SandboxRecoveryError("resource_store_unavailable") from exc

    def _path(self, resource_id: str) -> Path:
        if re.fullmatch(r"[0-9a-f]{32}", resource_id) is None:
            raise SandboxRecoveryError("invalid_resource_filename")
        return self.directory / f"{resource_id}.json"

    def read(self, resource_id: str) -> SandboxResourceRecord:
        self.require_lock()
        path = self._path(resource_id)
        try:
            if path.is_symlink():
                raise SandboxRecoveryError("resource_record_is_symlink", resource_id)
            record = SandboxResourceRecord.model_validate_json(path.read_bytes())
            if record.resource_id != resource_id or record.home_id != self.home_id:
                raise SandboxRecoveryError("resource_record_owner_mismatch", resource_id)
            return record
        except (OSError, ValidationError) as exc:
            raise SandboxRecoveryError("invalid_or_missing_resource_record", resource_id) from exc

    def save(self, record: SandboxResourceRecord) -> None:
        self.require_lock()
        if record.home_id != self.home_id:
            raise SandboxRecoveryError("resource_record_owner_mismatch", record.resource_id)
        path = self._path(record.resource_id)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            if self.directory.is_symlink() or path.is_symlink():
                raise SandboxRecoveryError("resource_store_is_symlink", record.resource_id)
            with temporary.open("x", encoding="utf-8") as handle:
                os.chmod(temporary, 0o600)
                handle.write(record.model_dump_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            _sync_directory(self.directory)
        except OSError as exc:
            raise SandboxRecoveryError("resource_record_write_failed", record.resource_id) from exc
        finally:
            temporary.unlink(missing_ok=True)

    def forget(self, record: SandboxResourceRecord) -> None:
        self.require_lock()
        try:
            self._path(record.resource_id).unlink()
            _sync_directory(self.directory)
        except OSError as exc:
            raise SandboxRecoveryError("resource_record_remove_failed", record.resource_id) from exc


def _sync_directory(directory: Path) -> None:
    if os.name == "nt":
        return  # Windows has no portable directory fsync; file replace is atomic.
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def mount_identity(value: str) -> str:
    """Compare Windows bind sources as returned by the Docker Desktop Linux engine."""
    normalized = value.replace("\\", "/").rstrip("/")
    prefix = "/run/desktop/mnt/host/"
    if normalized.startswith(prefix):
        remainder = normalized[len(prefix):]
        if len(remainder) >= 2 and remainder[0].isalpha() and remainder[1] == "/":
            normalized = remainder[0] + ":/" + remainder[2:]
    if re.match(r"^[A-Za-z]:/", normalized):
        return normalized.casefold()
    return normalized or "/"


def validate_container(
    record: SandboxResourceRecord, inspected: dict[str, Any],
) -> SandboxResourceRecord:
    """Only a complete match may be converted to a removable, full Docker ID."""
    resource_id = record.resource_id
    try:
        container_id = inspected["Id"]
        created = inspected["Created"]
        labels = inspected["Config"]["Labels"]
        mounts = inspected["Mounts"]
        matches = (
            isinstance(container_id, str) and _FULL_ID.fullmatch(container_id) is not None
            and isinstance(created, str)
            and inspected["Name"] == "/" + record.name
            and inspected["Image"] == record.image_id
            and isinstance(labels, dict)
            and all(labels.get(key) == value for key, value in record.labels.items())
            and len(mounts) == 1
            and mounts[0]["Type"] == "bind"
            and mounts[0]["Destination"] == "/workspace"
            and mounts[0]["RW"] is True
            and mount_identity(mounts[0]["Source"]) == mount_identity(record.workspace_root)
            and inspected["HostConfig"]["NetworkMode"] == "none"
            and inspected["HostConfig"]["ReadonlyRootfs"] is True
            and (record.container_id is None or record.container_id == container_id)
            and (record.created is None or record.created == created)
        )
        if not matches:
            raise ValueError("container identity mismatch")
        return SandboxResourceRecord.model_validate({
            **record.model_dump(), "phase": "confirmed", "container_id": container_id,
            "created": created,
        })
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise SandboxRecoveryError("container_identity_mismatch", resource_id) from exc


async def _engine_id(config: SandboxConfig, command: DockerCommand) -> str:
    code, stdout, stderr = await command(
        [config.docker_binary, "info", "--format", "{{.ID}}"],
        timeout_s=RECOVERY_COMMAND_TIMEOUT_S,
    )
    engine = stdout.strip()
    if code != 0 or not engine or any(character.isspace() for character in engine):
        raise SandboxRecoveryError(
            "docker_engine_unavailable", operation="info", exit_code=code,
            diagnostic=stderr or "Docker returned an empty or malformed engine ID",
        )
    return engine


async def _inspect(
    config: SandboxConfig, command: DockerCommand, target: str,
) -> dict[str, Any] | None:
    code, stdout, stderr = await command(
        [config.docker_binary, "inspect", target], timeout_s=RECOVERY_COMMAND_TIMEOUT_S,
    )
    if code != 0:
        if "no such container" in stderr.casefold() or "no such object" in stderr.casefold():
            return None
        raise SandboxRecoveryError(
            "docker_inspect_failed", operation="inspect", exit_code=code, diagnostic=stderr,
        )
    try:
        objects = json.loads(stdout)
        if not isinstance(objects, list) or len(objects) != 1 or not isinstance(objects[0], dict):
            raise ValueError("inspect must return exactly one object")
        if _FULL_ID.fullmatch(target) is not None and objects[0].get("Id") != target:
            raise ValueError("inspect returned a different container id")
        return objects[0]
    except (ValueError, TypeError) as exc:
        raise SandboxRecoveryError("invalid_docker_inspect") from exc


class SandboxOwner:
    def __init__(
        self, store: SandboxResourceStore, *, launch_id: str, engine_id: str,
    ) -> None:
        self.store = store
        self.launch_id = launch_id
        self.engine_id = engine_id

    async def check_engine(self, config: SandboxConfig, command: DockerCommand) -> None:
        self.store.require_lock()
        if await _engine_id(config, command) != self.engine_id:
            raise SandboxRecoveryError("docker_engine_changed")

    async def create_intent(
        self, config: SandboxConfig, command: DockerCommand, *, instance_id: str,
        run_id: str, name: str, workspace_root: Path,
    ) -> SandboxResourceRecord:
        root = workspace_root.resolve()
        home = self.store.lock.home
        if root.is_relative_to(home) or home.is_relative_to(root):
            raise SandboxRecoveryError("workspace_overlaps_core_home")
        await self.check_engine(config, command)
        code, stdout, stderr = await command(
            [config.docker_binary, "image", "inspect", config.image],
            timeout_s=RECOVERY_COMMAND_TIMEOUT_S,
        )
        try:
            objects = json.loads(stdout)
            image_id = objects[0]["Id"]
            if code != 0 or len(objects) != 1 or _IMAGE_ID.fullmatch(image_id) is None:
                raise ValueError("image identity unavailable")
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise SandboxRecoveryError(
                "docker_image_identity_unavailable", operation="image inspect", exit_code=code,
                diagnostic=stderr or "Docker returned an invalid image identity",
            ) from exc
        record = SandboxResourceRecord(
            version=1, resource_id=uuid.uuid4().hex, home_id=self.store.home_id,
            launch_id=self.launch_id, instance_id=instance_id, run_id=run_id,
            engine_id=self.engine_id, name=name, workspace_root=str(root),
            image_id=image_id,
        )
        self.store.save(record)
        return record

    async def confirm(
        self, record: SandboxResourceRecord, container_id: str,
        config: SandboxConfig, command: DockerCommand,
    ) -> SandboxResourceRecord:
        await self.check_engine(config, command)
        if _FULL_ID.fullmatch(container_id) is None:
            raise SandboxRecoveryError("invalid_container_id", record.resource_id)
        inspected = await _inspect(config, command, container_id)
        if inspected is None:
            raise SandboxRecoveryError("created_container_missing", record.resource_id)
        confirmed = validate_container(record, inspected)
        self.store.save(confirmed)
        return confirmed

    async def _confirm_absent(
        self, record: SandboxResourceRecord, config: SandboxConfig, command: DockerCommand,
    ) -> None:
        assert record.container_id is not None
        code, stdout, stderr = await command(
            [config.docker_binary, "ps", "--all", "--quiet", "--no-trunc", "--filter",
             f"id={record.container_id}"], timeout_s=RECOVERY_COMMAND_TIMEOUT_S,
        )
        if code != 0 or stdout.strip():
            raise SandboxRecoveryError(
                "container_removal_unconfirmed", record.resource_id, operation="confirm absence",
                exit_code=code, diagnostic=stderr or "Docker still lists the recorded container",
            )
        await self.check_engine(config, command)

    async def remove(
        self, record: SandboxResourceRecord, config: SandboxConfig, command: DockerCommand,
    ) -> None:
        # Reload because a previous attempt may have persisted confirmation before
        # failing to remove it. An absent confirmed ID is conclusive; absent intent is not.
        record = self.store.read(record.resource_id)
        await self.check_engine(config, command)
        if record.engine_id != self.engine_id:
            raise SandboxRecoveryError("record_engine_mismatch", record.resource_id)
        inspected = await _inspect(config, command, record.container_id or record.name)
        if inspected is None:
            if record.phase == "intent":
                raise SandboxRecoveryError("creation_outcome_unknown", record.resource_id)
            await self._confirm_absent(record, config, command)
        else:
            confirmed = validate_container(record, inspected)
            if record.phase == "intent":
                # Commit the exact identity before deletion, so a crash after rm
                # can safely recover by observing this known ID's absence.
                self.store.save(confirmed)
            await self.check_engine(config, command)
            assert confirmed.container_id is not None
            code, stdout, stderr = await command(
                [config.docker_binary, "rm", "--force", confirmed.container_id],
                timeout_s=RECOVERY_COMMAND_TIMEOUT_S,
            )
            if code != 0:
                raise SandboxRecoveryError(
                    "container_removal_failed", record.resource_id, operation="rm", exit_code=code,
                    diagnostic=stderr or stdout,
                )
            await self._confirm_absent(confirmed, config, command)
        self.store.forget(record)

    async def recover(
        self, config: SandboxConfig, command: DockerCommand, *, instance_id: str | None = None,
    ) -> int:
        await self.check_engine(config, command)
        records = self.store.load()
        by_id = {record.resource_id: record for record in records}
        if any(record.engine_id != self.engine_id for record in records):
            raise SandboxRecoveryError("record_engine_mismatch")
        code, stdout, stderr = await command(
            [config.docker_binary, "ps", "--all", "--quiet", "--no-trunc", "--filter",
             f"label={_LABEL}sandbox=true"], timeout_s=RECOVERY_COMMAND_TIMEOUT_S,
        )
        if code != 0:
            raise SandboxRecoveryError(
                "docker_inventory_failed", operation="inventory", exit_code=code,
                diagnostic=stderr or stdout,
            )
        # Validate the complete inventory before deleting anything. Legacy containers
        # cannot be assigned to a HOME safely and require explicit manual diagnosis.
        for container_id in stdout.split():
            if _FULL_ID.fullmatch(container_id) is None:
                raise SandboxRecoveryError("invalid_docker_inventory")
            inspected = await _inspect(config, command, container_id)
            if inspected is None:
                continue  # Another owner may finish an ephemeral container concurrently.
            config_data = inspected.get("Config")
            labels = config_data.get("Labels") if isinstance(config_data, dict) else None
            if not isinstance(labels, dict):
                raise SandboxRecoveryError(
                    "unclassified_legacy_container", container_id=container_id,
                )
            purpose = labels.get(_LABEL + "purpose")
            if purpose == "ephemeral":
                continue
            home_id = labels.get(_LABEL + "home")
            if (purpose != "core-managed" or not isinstance(home_id, str)
                    or _HOME_ID.fullmatch(home_id) is None):
                raise SandboxRecoveryError(
                    "unclassified_legacy_container", container_id=container_id,
                )
            if home_id != self.store.home_id:
                continue
            resource_id = labels.get(_LABEL + "resource")
            if not isinstance(resource_id, str) or resource_id not in by_id:
                raise SandboxRecoveryError(
                    "owned_container_has_no_record", container_id=container_id,
                )
            validate_container(by_id[resource_id], inspected)
        selected = [record for record in records
                    if instance_id is None or record.instance_id == instance_id]
        for record in selected:
            await self.remove(record, config, command)
        await self.check_engine(config, command)
        return len(selected)


async def recover_core_sandboxes(
    config: SandboxConfig, store: SandboxResourceStore, *, launch_id: str,
    command: DockerCommand,
) -> SandboxOwner:
    owner = SandboxOwner(store, launch_id=launch_id, engine_id=await _engine_id(config, command))
    await owner.recover(config, command)
    return owner
