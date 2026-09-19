"""Verify the sandbox worker, building a separate local image only when needed."""
from __future__ import annotations

import json
import shutil

from tars_agent.core.config import SandboxConfig
from tars_agent.core.tools.runtime.docker import DockerRuntime
from tars_agent.core.tools.runtime.models import ToolExecutionRequest
from tests.w04_support import (
    EXECUTION,
    ROOT,
    checked,
    container_ids,
    digest,
    docker,
    register_container,
    remove_exact,
    save,
    scope_path,
)


def probe(image, label):
    path = scope_path("docker-lifecycle")
    log = EXECUTION / "image-commands.jsonl"
    image_id = checked(docker(["image", "inspect", "--format", "{{.Id}}", image], log)).strip()
    runtime = DockerRuntime(SandboxConfig(image=image))
    request = ToolExecutionRequest("image-probe", f"w04-{label}", "image-check", "probe", "read_file", {}, path / "workspace", 20)
    args = runtime.build_container_args(request, f"w04-{label}-{runtime._instance_id}")
    cid = checked(docker(args[1:], log)).strip()
    owned = register_container(cid, workspace=request.workspace_root, instance=runtime._instance_id,
                               run=request.run_id, image=image_id, log=log)
    save(EXECUTION / f"{label}-container.json", owned)
    # The container uses the project's unmodified security arguments. cp only reads its worker.
    checked(docker(["cp", f"{cid}:/opt/tars/worker.py", str(EXECUTION / f"{label}-worker.py")], log))
    worker_hash = digest(EXECUTION / f"{label}-worker.py")
    remove_exact(owned, log, reason="completed owned image code verification")
    return {"tag": image, "id": image_id, "worker_sha256": worker_hash, "container": cid,
            "container_removed": True}


def main():
    log = EXECUTION / "image-commands.jsonl"
    save(EXECUTION / "containers-before.json", container_ids(log))
    source = ROOT / "src/tars_agent/sandbox/worker.py"
    original = probe("tars-agent-sandbox:0.8.0", "original-image")
    selected = original
    if original["worker_sha256"] != digest(source):
        context = EXECUTION / "image-context"
        context.mkdir()
        for name in ("Dockerfile", "worker.py"):
            shutil.copyfile(ROOT / "src/tars_agent/sandbox" / name, context / name)
        assert sorted(file.name for file in context.iterdir()) == ["Dockerfile", "worker.py"]
        # Exact same two-file build as the existing `tars sandbox build`, with a distinct tag.
        tag = "tars-agent-sandbox:w04-local-20260917-103341"
        checked(docker(["build", "--file", str(context / "Dockerfile"), "--tag", tag, str(context)], log, timeout=600))
        selected = probe(tag, "candidate-image")
    assert selected["worker_sha256"] == digest(source), "candidate worker differs from the working copy"
    original_after = checked(docker(["image", "inspect", "--format", "{{.Id}}", original["tag"]], log)).strip()
    assert original_after == original["id"], "original image changed"
    report = {"source_path": str(source), "source_sha256": digest(source), "original": original,
              "selected": selected, "original_image_unchanged": True,
              "container_project_code": "/opt/tars/worker.py only; no project source mount",
              "remaining_container_ids": container_ids(log)}
    save(EXECUTION / "image-result.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
