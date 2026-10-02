from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest

from tars_agent.core.eval import appworld


class Docker:
    def __init__(self, world: appworld.WorldContainer) -> None:
        self.world = world
        self.networks: dict[str, dict[str, Any]] = {}
        self.containers: dict[str, dict[str, Any]] = {}
        self.calls: list[list[str]] = []
        self.created = threading.Event()
        self.release = threading.Event()
        self.pause_creation = False

    def command(self, args: list[str], **kwargs: Any) -> str:
        self.calls.append(args)
        if args[1:3] == ["network", "create"]:
            intent = json.loads(self.world.record.read_text())
            assert intent["phase"] == "network_creating" and intent["network"] == args[-1]
            name = args[-1]
            self.networks[name] = {"Id": hashlib.sha256(name.encode()).hexdigest(),
                                   "Labels": {appworld.OWNER_LABEL: self.world.owner}, "Containers": {}}
            return self.networks[name]["Id"]
        if args[1] == "create":
            intent = json.loads(self.world.record.read_text())
            assert intent["phase"] == "container_creating"
            name, network = args[args.index("--name") + 1], args[args.index("--network") + 1]
            identifier = hashlib.sha256(name.encode()).hexdigest()
            self.containers[identifier] = {
                "Id": identifier, "Name": "/" + name, "Image": self.world.image_id,
                "Config": {"Labels": {appworld.OWNER_LABEL: self.world.owner}},
                "State": {"Running": False}, "HostConfig": {"NetworkMode": network},
                "Mounts": [{"Type": "bind", "Destination": target, "Source": str(source), "RW": rw}
                           for target, source, rw in [("/run/data", self.world.data, False),
                                                     ("/run/experiments", self.world.output, True)]],
                "NetworkSettings": {"Networks": {network: {"NetworkID": ""}}, "Ports": {}},
            }
            self.created.set()
            if self.pause_creation:
                assert self.release.wait(5)
            return identifier
        if args[1] == "start":
            item = self.containers[args[2]]
            network = item["HostConfig"]["NetworkMode"]
            item["State"]["Running"] = True
            item["NetworkSettings"] = {"Networks": {network: {"NetworkID": self.networks[network]["Id"]}},
                                       "Ports": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8123"}]}}
            self.networks[network]["Containers"][args[2]] = {}
            return args[2]
        if args[1] == "inspect":
            return json.dumps([self.containers[args[2]]])
        if args[1] == "ps":
            selector = args[-1]
            if selector.startswith("id="):
                return selector[3:] if selector[3:] in self.containers else ""
            name = selector.removeprefix("name=^/").removesuffix("$")
            return "\n".join(key for key, value in self.containers.items() if value["Name"] == "/" + name)
        if args[1] == "rm":
            identifier = args[-1]
            item = self.containers.pop(identifier)
            self.networks[item["HostConfig"]["NetworkMode"]]["Containers"].pop(identifier, None)
            return identifier
        if args[1:3] == ["network", "ls"]:
            name = args[-1].removeprefix("name=^").removesuffix("$")
            return name if name in self.networks else ""
        if args[1:3] == ["network", "inspect"]:
            return json.dumps([self.networks[args[-1]]])
        if args[1:3] == ["network", "rm"]:
            self.networks.pop(args[-1])
            return args[-1]
        raise AssertionError(args)


def world_and_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[appworld.WorldContainer, Docker]:
    data = tmp_path / "data"
    data.mkdir()
    world = appworld.WorldContainer("image", data, tmp_path / "output", "owned")
    docker = Docker(world)
    monkeypatch.setattr(appworld, "command", docker.command)
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(lambda request: httpx.Response(200)), **kwargs,
    ))
    return world, docker


async def test_creation_intents_and_full_identity_support_idempotent_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    world, docker = world_and_docker(tmp_path, monkeypatch)
    await world.start()
    assert world.url == "http://127.0.0.1:8123"
    assert len(docker.containers) == len(docker.networks) == 1
    await world.close()
    await world.close()
    assert not docker.containers and not docker.networks
    assert json.loads(world.record.read_text())["phase"] == "removed"


async def test_cancel_during_docker_creation_waits_for_id_then_reclaims_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    world, docker = world_and_docker(tmp_path, monkeypatch)
    docker.pause_creation = True
    starting = asyncio.create_task(world.start())
    assert await asyncio.to_thread(docker.created.wait, 5)
    starting.cancel()
    await asyncio.sleep(0)
    assert not starting.done()
    docker.release.set()
    with pytest.raises(asyncio.CancelledError):
        await starting
    assert not docker.containers and not docker.networks
    assert json.loads(world.record.read_text())["phase"] == "removed"


async def test_recovery_finds_container_by_recorded_name_when_id_was_not_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    world, docker = world_and_docker(tmp_path, monkeypatch)
    await world.start()
    record = json.loads(world.record.read_text())
    record["container_id"] = None
    record["phase"] = "container_creating"
    appworld.write_json(world.record, record)
    recovered = appworld.WorldContainer("image", world.data, world.output, "owned")
    await recovered.close()
    assert not docker.containers and not docker.networks


@pytest.mark.parametrize("mismatch", ["name", "owner", "image", "mount", "extra_mount", "network"])
async def test_cleanup_refuses_any_mismatched_container_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str,
) -> None:
    world, docker = world_and_docker(tmp_path, monkeypatch)
    await world.start()
    item = docker.containers[world.container_id]
    if mismatch == "name":
        item["Name"] = "/foreign"
    elif mismatch == "owner":
        item["Config"]["Labels"][appworld.OWNER_LABEL] = "foreign"
    elif mismatch == "image":
        item["Image"] = "foreign"
    elif mismatch == "mount":
        item["Mounts"][0]["Source"] = str(tmp_path / "foreign")
    elif mismatch == "extra_mount":
        item["Mounts"].append({"Destination": "/foreign", "Source": "/private", "RW": True})
    else:
        item["HostConfig"]["NetworkMode"] = "foreign"
    before = len(docker.calls)
    with pytest.raises(RuntimeError, match="identity|ownership|mismatch"):
        await world.close()
    assert not any(call[1] == "rm" for call in docker.calls[before:])
    assert len(docker.containers) == 1


async def test_base_interpreter_launcher_owns_actual_pid_and_loads_selected_site_packages(tmp_path: Path) -> None:
    (tmp_path / "dependency_marker.py").write_text("VALUE = 'selected environment'\n")
    (tmp_path / "pid_probe.py").write_text(
        "import json,os,sys\nfrom dependency_marker import VALUE\n"
        "print(json.dumps({'pid':os.getpid(),'value':VALUE}),flush=True)\n"
        "sys.stdin.readline()\n",
    )
    environment = appworld.python_environment()
    environment["site_packages"] = str(tmp_path)
    argv = appworld.module_argv("pid_probe", [], environment)
    process = await asyncio.create_subprocess_exec(
        *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        assert process.stdout is not None and process.stdin is not None
        response = json.loads(await asyncio.wait_for(process.stdout.readline(), 10))
        assert response == {"pid": process.pid, "value": "selected environment"}
        identity = await asyncio.to_thread(appworld.process_identity, process.pid)
        assert identity is not None and identity["pid"] == process.pid
        assert identity.get("command_line", subprocess.list2cmdline(identity.get("argv", []))) == subprocess.list2cmdline(argv)
        process.stdin.write(b"finish\n")
        await process.stdin.drain()
        await asyncio.wait_for(process.wait(), 5)
        assert process.returncode == 0
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
