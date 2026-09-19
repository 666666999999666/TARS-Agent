"""Explicit W04 scenarios only. No automatic full test gate or real model calls."""
from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sqlite3
import subprocess
import sys
import time
import uuid

from scripts.acceptance_cli import ANSI, CliPty

from tars_agent.core.control import read_control_file
from tars_agent.core.transport.socket_client import SocketClient
from tests.w04_preparation import check_demo
from tests.w04_support import (
    EXECUTION,
    ROOT,
    append,
    checked,
    clean_environment,
    container_ids,
    digest,
    docker,
    inspect_container,
    kill_verified_core,
    process_identity,
    remove_exact,
    save,
    scope_path,
    verify_container,
    verify_process,
)

IMAGE = json.loads((EXECUTION / "image-result.json").read_text())["selected"]
LOG = EXECUTION / "validation-commands.jsonl"
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
RESULTS = []


def record(name, **data):
    value = {"case": name, "at": time.time(), **data}
    RESULTS.append(value)
    append(EXECUTION / "results.jsonl", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


async def wait_for(predicate, timeout, description):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.1)
    raise TimeoutError(description)


async def wait_until(wall_time):
    while time.time() < wall_time:
        await asyncio.sleep(min(0.5, wall_time - time.time()))


def rows(home, query, params=()):
    with sqlite3.connect(f"file:{(home / 'state.db').as_posix()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(query, params)]


def write_new(path, content):
    if path.exists():
        assert path.read_text(encoding="utf-8") == content, "refuse to overwrite an existing artifact"
        return
    with path.open("x", encoding="utf-8") as file:
        file.write(content)


def case_script(workspace, name, *, delay=18, heartbeat=False):
    # Bounded synthetic worker: one start marker, at most 100 heartbeats or one delayed write.
    text = (
        "import hashlib, json, os, time\nfrom pathlib import Path\n"
        f"name={name!r}\ndelay={delay!r}\n"
        "start=time.time()\n"
        "identity={'wall':start,'late_due':start+delay,'pid':os.getpid(),'ppid':os.getppid(),"
        "'start_ticks':Path('/proc/self/stat').read_text().split()[21],"
        "'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}\n"
        "Path(name+'.started.json').write_text(json.dumps(identity))\n"
    )
    if heartbeat:
        text += ("for counter in range(100):\n"
                 "    with Path(name+'.heartbeat.jsonl').open('a') as f:\n"
                 "        f.write(json.dumps({'counter':counter,'wall':time.time()})+'\\n')\n"
                 "    if Path(name+'.stop').exists(): break\n"
                 "    time.sleep(1)\n")
    else:
        text += ("time.sleep(max(0,start+delay-time.time()))\n"
                 "with Path(name+'.late.jsonl').open('a') as f:\n"
                 "    f.write(json.dumps({'wall':time.time(),'pid':os.getpid()})+'\\n')\n")
    write_new(workspace / f"{name}.py", text)
    return {"command": f"python {name}.py", "timeout": 110 if heartbeat else max(60, delay + 15)}


def tool(tool_id, name, params):
    return {"id": tool_id, "name": name, "input": params}


class Core:
    def __init__(self, scope, name, plan_name="w04-plans.json"):
        self.scope = scope
        self.root = scope_path(scope)
        self.home = self.root / "home"
        self.workspace = self.root / "workspace"
        self.evidence = self.root / "evidence" / ("real-103341-" + name)
        self.evidence.mkdir(exist_ok=False)
        self.proc = None
        self.client = None
        self.reader = None
        self.identity = None
        self.session = None
        self.plan_name = plan_name

    async def start(self):
        check_demo(self.workspace, self.home)
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            self.port = server.getsockname()[1]
        self.launch = uuid.uuid4().hex
        env = clean_environment(self.home, self.port, IMAGE["tag"])
        env.update(W04_SCOPE=self.scope, W04_EVIDENCE=str(self.evidence), TARS_CORE_LAUNCH_ID=self.launch,
                   W04_PLAN_FILE=self.plan_name)
        output = (self.evidence / "core.log").open("xb")
        args = [sys.executable, "-B", "-m", "tests.w04_real_daemon"]
        self.proc = subprocess.Popen(args, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        output.close()
        self.launcher_identity = await asyncio.to_thread(process_identity, self.proc.pid)
        save(self.evidence / "launcher-process.json", {"identity": self.launcher_identity, "args": args,
             "scope": self.scope, "home": str(self.home), "workspace": str(self.workspace),
             "port": self.port, "launch_id": self.launch})
        self.control_path = self.home / "control" / f"tars-core-{self.port}.json"

        def ready():
            if self.proc.poll() is not None:
                raise RuntimeError(f"Core exited early: {self.evidence / 'core.log'}")
            control = read_control_file(self.control_path)
            source_file = self.evidence / "core-source.json"
            if not source_file.exists():
                return False
            source_pid = json.loads(source_file.read_text())["pid"]
            return control and control.pid == source_pid and control.launch_id == self.launch

        await wait_for(ready, 45, "Core startup deadline")
        control = read_control_file(self.control_path)
        self.identity = await asyncio.to_thread(process_identity, control.pid)
        assert (self.identity["ProcessId"] == self.proc.pid
                or self.identity["ParentProcessId"] == self.proc.pid), "Core is not the launched process or its direct child"
        assert "-m tests.w04_real_daemon" in self.identity["CommandLine"]
        save(self.evidence / "core-process.json", {"identity": self.identity,
             "launcher_identity": self.launcher_identity, "port": self.port, "launch_id": self.launch,
             "home": str(self.home), "workspace": str(self.workspace)})
        self.client = SocketClient("127.0.0.1", self.port)
        await self.client.connect()
        self.reader = asyncio.create_task(self.client.run_event_loop())
        pong = await self.rpc("core.ping", {"client": "W04 controlled validation"})
        source = json.loads((self.evidence / "core-source.json").read_text())
        for name, sha in source["source_files"].items():
            assert digest(ROOT / name) == sha, "Core source changed during startup"
        assert source["app_file"] == str(ROOT / "src/tars_agent/core/app.py")
        record("core_started", scope=self.scope, pid=self.identity["ProcessId"], launcher_pid=self.proc.pid, port=self.port,
               launch_id=self.launch, source_verified=True, pong=pong)

    async def rpc(self, method, params):
        assert self.client is not None
        response = await self.client.send_command(method, params, timeout_s=25)
        logged = {key: value for key, value in params.items() if key != "token"}
        append(self.evidence / "rpc.jsonl", {"method": method, "params": logged, "response": response})
        return response

    async def submit(self, goal):
        # Revalidate before every new task, including a task after recovery.
        check_demo(self.workspace, self.home)
        if self.session is None:
            self.session = (await self.rpc("session.create", {"mode": "chat", "workspace_root": str(self.workspace)}))["session_id"]
        result = await self.rpc("session.send_message", {"session_id": self.session,
            "content": "/demo " + goal, "client_message_id": uuid.uuid4().hex})
        return result["run_id"]

    def events(self, run, kind):
        return [json.loads(row["payload"]) for row in rows(self.home,
            "SELECT payload FROM events WHERE run_id=? AND event_type=? ORDER BY cursor", (run, kind))]

    async def approvals(self, run, expected):
        approved = set()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            for event in self.events(run, "permission.requested"):
                request = event["request_id"]
                if request in approved:
                    continue
                call = next((value for value in expected if value["id"] == event["tool_use_id"]), None)
                assert call is not None and call["name"] == event["tool_name"] and call["input"] == event["params"]
                assert event["request_kind"] == "tool", "host fallback forbidden"
                await self.rpc("permission.respond", {"request_id": request, "session_id": self.session,
                                                     "decision": "allow_once"})
                approved.add(request)
            if len(approved) == sum(call["name"] in {"write_file", "bash"} for call in expected):
                return
            await asyncio.sleep(0.1)
        raise TimeoutError("expected tool approval did not arrive")

    async def terminal(self, run, timeout=35):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = await self.rpc("run.get", {"run_id": run})
            if value["status"] in TERMINAL:
                return value
            await asyncio.sleep(0.2)
        raise TimeoutError("run did not become terminal before cleanup")

    def container(self, run):
        found = [json.loads(path.read_text()) for path in (self.evidence / "containers").glob("*.json")]
        matching = [value for value in found if value["labels"].get("com.tars-agent.run") == run]
        assert len(matching) == 1, "expected exactly one known container for this run"
        return matching[0]

    async def verify_live(self):
        actual = await asyncio.to_thread(verify_process, self.identity)
        control = read_control_file(self.control_path)
        assert control and control.pid == self.identity["ProcessId"] and control.launch_id == self.launch
        return actual

    async def kill_for_fault(self, run):
        await self.verify_live()
        owned = self.container(run)
        await asyncio.to_thread(verify_container, owned, LOG)
        append(self.evidence / "fault.jsonl", {"action": "kill_only_owned_A_core", "identity": self.identity,
            "container": owned, "run_id": run, "launch_id": self.launch})
        await asyncio.to_thread(kill_verified_core, self.identity)
        await asyncio.to_thread(self.proc.wait, 10)
        record("A_core_killed", scope=self.scope, pid=self.identity["ProcessId"], exit_code=self.proc.returncode,
               container_id=owned["id"], run_id=run)
        return owned

    async def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            await self.verify_live()
            control = read_control_file(self.control_path)
            await self.rpc("core.shutdown", {"token": control.token})
            # No implicit process termination. A timeout is a failed shutdown to investigate.
            await asyncio.to_thread(self.proc.wait, 30)
        try:
            if self.client is not None:
                await self.client.close()
        except ConnectionResetError as error:
            record("ipc_close_error", scope=self.scope, error=repr(error),
                   source="src/tars_agent/core/transport/socket_client.py:61",
                   production_fix_applied=False)
            raise  # Keep the whole stage nonzero; physical cleanup is a separate assertion.
        finally:
            if self.reader is not None:
                await asyncio.gather(self.reader, return_exceptions=True)
            record("core_stopped", scope=self.scope,
                   core_pid=self.identity["ProcessId"] if self.identity else None,
                   launcher_pid=self.proc.pid if self.proc else None,
                   exit_code=self.proc.returncode if self.proc else None)


async def marker(core, name, run):
    path = core.workspace / f"{name}.started.json"
    await wait_for(lambda: path.is_file(), 25, "actual worker start marker")
    value = json.loads(path.read_text())
    owned = core.container(run)
    await asyncio.to_thread(verify_container, owned, LOG)
    # Corroborate the in-container PID, start ticks, command line and boot identity.
    code = ("import json,pathlib; p=pathlib.Path('/proc')/" + repr(str(value["pid"])) + "; "
            "print(json.dumps({'stat':(p/'stat').read_text(),'cmdline':(p/'cmdline').read_bytes().decode().replace(chr(0),' '),"
            "'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))")
    observed = json.loads(checked(await asyncio.to_thread(docker, ["exec", owned["id"], "python", "-c", code], LOG)))
    assert observed["stat"].split()[21] == value["start_ticks"] and observed["boot_id"] == value["boot_id"]
    assert f"{name}.py" in observed["cmdline"]
    save(core.evidence / f"{name}-started.json", {"marker": value, "container": owned, "process": observed})
    return value, owned


async def cli_approval(core, plan):
    check_demo(core.workspace, core.home)
    terminal = CliPty([str(ROOT / ".venv/Scripts/tars.exe"), "run", "--goal", "/demo W04 cli approval"],
                      core.workspace, core.home / "cli-home", core.port, core.evidence / "formal-cli")
    identity = await asyncio.to_thread(process_identity, terminal.pty.pid)
    save(core.evidence / "formal-cli/identity.json", identity)
    run = None
    answered = set()
    decisions = ["n", "a", "n"]
    try:
        deadline = time.monotonic() + 65
        while time.monotonic() < deadline:
            terminal.pump()
            if run is None:
                matches = rows(core.home, "SELECT * FROM runs ORDER BY created_at")
                if matches:
                    assert len(matches) == 1
                    run = matches[0]
            if run is not None:
                for event in core.events(run["id"], "permission.requested"):
                    rid = event["request_id"]
                    if rid in answered or not terminal.approval_ready(rid):
                        continue
                    index = len(answered)
                    assert event["params"] == plan[index]["input"]
                    plain = ANSI.sub("", terminal.text)
                    fragment = plain[plain.index(f"[approval: {rid}]"):]
                    # ConPTY inserts wrap line breaks; remove only CR/LF, retaining every parameter character.
                    assert json.dumps(event["params"], ensure_ascii=False, sort_keys=True) in fragment.replace("\r", "").replace("\n", "")
                    if index < 2:
                        assert not (core.workspace / "approval.txt").exists()
                    else:
                        assert (core.workspace / "approval.txt").read_text() == plan[1]["input"]["content"]
                    append(core.evidence / "formal-cli/decisions.jsonl", {"run_id": run["id"], "event": event,
                           "full_parameters_visible_before_answer": True, "key": decisions[index]})
                    terminal.send(decisions[index] + "\r", "response after complete parameter comparison")
                    answered.add(rid)
                if len(answered) == 3 and not terminal.pty.isalive():
                    break
            await asyncio.sleep(0.1)
        assert len(answered) == 3, "all three distinct approvals must be shown"
        await terminal.finish(1)  # Two deliberate tool denials must not be advertised as a clean success.
        assert (core.workspace / "approval.txt").read_text() == plan[1]["input"]["content"]
        invocations = rows(core.home, "SELECT * FROM tool_invocations WHERE run_id=?", (run["id"],))
        assert len(invocations) == 3 and sum(row["status"] == "succeeded" for row in invocations) == 1
        save(core.evidence / "formal-cli/tool-invocations.json", invocations)
        owned = core.container(run["id"])
        assert await asyncio.to_thread(inspect_container, owned["id"], LOG) is None
        record("formal_cli_approval", passed=True, run_id=run["id"], session_id=run["session_id"],
               approvals=3, actual_writes=1, cli_exit_code=1, provider="ScriptedProvider", terminal="Windows ConPTY")
    finally:
        # The existing cleanup helper is used only after natural exit was observed, so its kill fallback is unreachable.
        if terminal.record["exit_code"] is not None:
            await terminal.cleanup()
        elif terminal.pty.isalive():
            await asyncio.to_thread(verify_process, identity)
            terminal.send("\x03", "failed test cleanup: request cancellation of this CLI task")
            try:
                await terminal.finish(130)
            finally:
                if not terminal.pty.isalive():
                    terminal.pty.cancel_io()


async def lifecycle():
    core = Core("docker-lifecycle", "lifecycle-v2")
    prefix = "same-prefix-" * 200
    approval = [tool(f"approval-{i}", "write_file", {"path": "approval.txt", "content": prefix + tail})
                for i, tail in enumerate(("DENY_A", "ALLOW_B", "DENY_C"))]
    normal = [tool("list", "list_dir", {"path": "."}), tool("read", "read_file", {"path": "sample.txt"}),
              tool("write", "write_file", {"path": "normal.txt", "content": "W04 real Docker"}),
              tool("bash", "bash", {"command": "printf W04-normal; test -f normal.txt", "timeout": 15})]
    timeout = case_script(core.workspace, "timeout", delay=22)
    timeout["timeout"] = 12
    cancel = case_script(core.workspace, "cancel", delay=22)
    plans = {"W04 cli approval": approval, "W04 normal": normal,
             "W04 timeout": [tool("timeout", "bash", timeout)], "W04 cancel": [tool("cancel", "bash", cancel)]}
    write_new(core.home / "w04-plans.json", json.dumps(plans))
    try:
        await core.start()
        await cli_approval(core, approval)
        run = await core.submit("W04 normal")
        await core.approvals(run, normal)
        final = await core.terminal(run)
        tools = rows(core.home, "SELECT * FROM tool_invocations WHERE run_id=?", (run,))
        assert final["status"] == "succeeded" and len(tools) == 4 and all(row["status"] == "succeeded" for row in tools)
        assert (core.workspace / "normal.txt").read_text() == "W04 real Docker"
        owned = core.container(run)
        assert await asyncio.to_thread(inspect_container, owned["id"], LOG) is None
        record("normal", passed=True, run=final, container=owned["id"], tools=tools, manual_cleanup=False)
        for name in ("timeout", "cancel"):
            run = await core.submit("W04 " + name)
            await core.approvals(run, plans["W04 " + name])
            started, owned = await marker(core, name, run)
            if name == "cancel":
                await core.verify_live()
                await asyncio.to_thread(verify_container, owned, LOG)
                await core.rpc("run.cancel", {"run_id": run})
            final = await core.terminal(run)
            tools = rows(core.home, "SELECT * FROM tool_invocations WHERE run_id=?", (run,))
            removed = await asyncio.to_thread(inspect_container, owned["id"], LOG) is None
            await wait_until(started["late_due"] + 2)
            late = (core.workspace / f"{name}.late.jsonl").exists()
            passed = removed and not late
            if name == "cancel":
                passed &= final["status"] == "cancelled"
            else:
                passed &= len(tools) == 1 and tools[0]["status"] == "failed" and tools[0]["error_class"] == "timeout"
            record(name, passed=passed, run=final, tools=tools, container=owned["id"],
                   start=started, observed_at=time.time(), removed_by_project=removed, late_exists=late,
                   manual_cleanup=False)
            if not removed:
                await asyncio.to_thread(remove_exact, owned, LOG, reason=f"fallback after FAILED {name} cleanup")
            assert passed, f"{name} failed before fallback cleanup"
    finally:
        await core.stop()


def heartbeat(core, name):
    path = core.workspace / f"{name}.heartbeat.jsonl"
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    return json.loads(lines[-1]) if lines else None


async def pair(kind):
    a = Core(f"f05-{kind}/A", "first-v2", "w04-plans-v2.json")
    b = Core(f"f05-{kind}/B", "first-v2", "w04-plans-v2.json")
    name_a, name_b = f"{kind}-A-v2", f"{kind}-B-v2"
    calls_a = [tool("delayed", "bash", case_script(a.workspace, name_a, delay=35))]
    calls_b = [tool("heartbeat", "bash", case_script(b.workspace, name_b, heartbeat=True))]
    write_new(a.home / a.plan_name, json.dumps({"W04 A": calls_a, "W04 recovered": [tool("recovery-read", "read_file", {"path": "sample.txt"})]}))
    write_new(b.home / b.plan_name, json.dumps({"W04 B": calls_b}))
    recovered = None
    owned_a = None
    try:
        await a.start()
        await b.start()
        run_b = await b.submit("W04 B")
        await b.approvals(run_b, calls_b)
        _, owned_b = await marker(b, name_b, run_b)
        await wait_for(lambda: heartbeat(b, name_b), 10, "B heartbeat starts")
        run_a = await a.submit("W04 A")
        await a.approvals(run_a, calls_a)
        started, owned_a = await marker(a, name_a, run_a)
        before_b = heartbeat(b, name_b)
        await b.verify_live()
        await asyncio.to_thread(verify_container, owned_b, LOG)
        await a.kill_for_fault(run_a)
        residual = await asyncio.to_thread(verify_container, owned_a, LOG)
        assert residual["state"]["Running"], "F05 residual was not running at observation"
        if kind == "reclaim":
            assert time.time() < started["late_due"], "manual recovery missed the planned write window"
            await asyncio.to_thread(remove_exact, owned_a, LOG, reason="authorized precise manual F05 recovery")
        await wait_until(started["late_due"] + 2)
        late_path = a.workspace / f"{name_a}.late.jsonl"
        late = late_path.exists()
        residual_after = await asyncio.to_thread(inspect_container, owned_a["id"], LOG)
        await b.verify_live()
        after_b_container = await asyncio.to_thread(verify_container, owned_b, LOG)
        after_b = heartbeat(b, name_b)
        assert after_b["counter"] > before_b["counter"] and time.time() - after_b["wall"] < 4
        assert after_b_container["started"] == owned_b["started"]
        assert (await b.rpc("run.get", {"run_id": run_b}))["status"] == "running"
        record(f"f05_{kind}", risk_observed=(late and residual_after is not None) if kind == "observe" else None,
               manual_recovery_passed=(not late and residual_after is None) if kind == "reclaim" else None,
               automatic_recovery=False, A_run=run_a, B_run=run_b, A_container=owned_a["id"],
               B_container=owned_b["id"], start=started, late_exists=late, late_contents=late_path.read_text() if late else None,
               B_before=before_b, B_after=after_b, B_identity_unchanged=True, observed_at=time.time())
        if kind == "observe":
            assert late and residual_after is not None, "observed behavior differs; preserve evidence and inspect"
            await asyncio.to_thread(remove_exact, owned_a, LOG, reason="cleanup after separate F05 risk observation")
        else:
            assert not late and residual_after is None
        # B is stopped only after the above identity and continued-heartbeat verification.
        write_new(b.workspace / f"{name_b}.stop", "normal completion after B isolation check\n")
        b_result = await b.terminal(run_b)
        assert b_result["status"] == "succeeded"
        assert await asyncio.to_thread(inspect_container, owned_b["id"], LOG) is None
        await b.stop()
        # Old A workspace is reused only after the exact old container is confirmed gone.
        recovered = Core(a.scope, "recovered-v2", a.plan_name)
        await recovered.start()
        old = await recovered.rpc("run.get", {"run_id": run_a})
        assert old["status"] == "interrupted"
        invocations_before = rows(a.home, "SELECT id FROM tool_invocations ORDER BY id")
        late_before = late_path.read_bytes() if late else None
        await asyncio.sleep(2)
        assert not (recovered.evidence / "provider.jsonl").exists(), "recovery automatically invoked Provider"
        assert rows(a.home, "SELECT id FROM tool_invocations ORDER BY id") == invocations_before
        assert (late_path.read_bytes() if late_path.exists() else None) == late_before
        new_run = await recovered.submit("W04 recovered")
        new_result = await recovered.terminal(new_run)
        assert new_result["status"] == "succeeded"
        assert (late_path.read_bytes() if late_path.exists() else None) == late_before
        record(f"{kind}_recovery", passed=True, old_run=old, new_run=new_result,
               no_auto_provider_call=True, no_auto_tool_replay=True, recovered_preset_rechecked=True)
    finally:
        if recovered is not None:
            await recovered.stop()
        # Normal shutdown still performs only project cleanup guarded by exact identities.
        if b.proc is not None and b.proc.poll() is None:
            current = heartbeat(b, name_b)
            record("B_before_final_cleanup", scope=b.scope, heartbeat=current)
            await b.stop()
        try:
            await a.stop()
        finally:
            if owned_a is not None:
                await asyncio.to_thread(remove_exact, owned_a, LOG, reason="final owned-A cleanup; does not change test verdict")


async def main(stage):
    attempt = f"{stage}-{int(time.time())}"
    before = container_ids(LOG)
    save(EXECUTION / f"{attempt}-containers-before.json", before)
    try:
        await (lifecycle() if stage == "lifecycle" else pair(stage))
    except Exception as error:
        record(stage + "_stage", passed=False, error=repr(error))
        raise
    finally:
        after = container_ids(LOG)
        save(EXECUTION / f"{attempt}-containers-after.json", after)
        record(stage + "_resources", before=before, after=after, same_container_ids=before == after)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["lifecycle", "observe", "reclaim"])
    arguments = parser.parse_args()
    asyncio.run(main(arguments.stage))
