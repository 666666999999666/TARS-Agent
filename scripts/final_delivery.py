"""Export and restore an exact main snapshot and an optional stable tag."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MAIN_REF = "refs/heads/main"
PRIVATE_PARTS = {".git", ".venv", "node_modules", ".env", ".tars", ".kama", ".tars-baseline", "__pycache__", "artifacts", "build"}


class DeliveryError(RuntimeError):
    pass


def git(repo: Path, *args: str, input_data: bytes | None = None) -> bytes:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update({"GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"})
    result = subprocess.run(["git", "--no-optional-locks", "-c", "core.quotePath=false",
                             "-c", "core.fsmonitor=false", *args], cwd=repo, env=env,
                            input=input_data, capture_output=True, timeout=300, check=False)
    if result.returncode:
        raise DeliveryError(f"git {args[0]} failed with exit {result.returncode}")
    return result.stdout


def resolve(repo: Path, reference: str) -> str:
    return git(repo, "rev-parse", "--verify", "--end-of-options", reference + "^{commit}").decode().strip()


def baseline_commit(repo: Path, reference: str) -> str:
    roots = git(repo, "rev-list", "--max-parents=0", reference).decode().splitlines()
    if len(roots) != 1:
        raise DeliveryError("selected history must have one repository initialization commit")
    return roots[0]


def check_independent(repo: Path, *, require_clean: bool) -> dict:
    alternates = Path(git(repo, "rev-parse", "--git-path", "objects/info/alternates").decode().strip())
    if not alternates.is_absolute():
        alternates = repo / alternates
    if alternates.exists() or os.environ.get("GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        raise DeliveryError("repository depends on alternate object storage")
    if git(repo, "rev-parse", "--is-shallow-repository").strip() != b"false":
        raise DeliveryError("shallow history is not a self-contained handoff")
    if require_clean and git(repo, "status", "--porcelain=v1", "--untracked-files=all").strip():
        raise DeliveryError("main has uncommitted or untracked files; commit the chosen candidate first")
    return {"alternates_absent": True, "shallow": False,
            "working_tree_clean": not git(repo, "status", "--porcelain=v1", "--untracked-files=all").strip()}


def tree_entries(repo: Path, commit: str) -> dict[str, str]:
    entries = {}
    for row in git(repo, "ls-tree", "-r", "-z", commit).split(b"\0"):
        if not row:
            continue
        metadata, raw_path = row.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        path = raw_path.decode("utf-8")
        item = PurePosixPath(path)
        if mode not in {"100644", "100755"} or kind != "blob":
            raise DeliveryError(f"non-portable symlink/submodule entry: {path}")
        if item.is_absolute() or ".." in item.parts or set(item.parts) & PRIVATE_PARTS:
            raise DeliveryError(f"private or unsafe committed path: {path}")
        if item.suffix in {".db", ".sqlite", ".sqlite3", ".log", ".pyc"}:
            raise DeliveryError(f"private runtime file is committed: {path}")
        entries[path] = oid
    if len({path.casefold() for path in entries}) != len(entries):
        raise DeliveryError("tree contains case-colliding paths that cannot restore on Windows")
    return entries


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blob_identifier(data: bytes, algorithm: str) -> str:
    if algorithm not in {"sha1", "sha256"}:
        raise DeliveryError("unsupported Git object format")
    # Git's object identity, not a security/authentication digest. Artifact hashes use SHA-256.
    return hashlib.new(algorithm, b"blob " + str(len(data)).encode() + b"\0" + data,
                       usedforsecurity=False).hexdigest()


def verify_zip(repo: Path, commit: str, archive: Path, prefix: str = "TARS-Agent/") -> dict:
    entries = tree_entries(repo, commit)
    algorithm = git(repo, "rev-parse", "--show-object-format").decode().strip()
    with zipfile.ZipFile(archive) as package:
        members = [item for item in package.infolist() if not item.is_dir()]
        names = [item.filename for item in members]
        if len(names) != len(set(names)) or set(names) != {prefix + path for path in entries}:
            raise DeliveryError("ZIP paths differ from the exact Git tree (check export-ignore attributes)")
        for item in members:
            relative = item.filename.removeprefix(prefix)
            content = package.read(item)
            if content.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
                raise DeliveryError(f"ZIP contains an external Git LFS pointer: {relative}")
            if blob_identifier(content, algorithm) != entries[relative]:
                raise DeliveryError(f"ZIP content differs from committed blob: {relative}")
    return {"exact_tree_paths": True, "exact_blob_bytes": True, "file_count": len(entries)}


def snapshot_refs(source: Path, bare: Path, refs: dict[str, str], primary_ref: str) -> None:
    git(bare.parent, "init", "--bare", str(bare))
    # Fetch object closures into a new object database; never copy .git or use alternates/hardlinks.
    git(bare, "fetch", "--no-tags", "--no-write-fetch-head", str(source), *sorted(set(refs.values())))
    for name, oid in refs.items():
        git(bare, "update-ref", name, oid)
    git(bare, "symbolic-ref", "HEAD", primary_ref)
    actual = set(git(bare, "for-each-ref", "--format=%(refname)").decode().splitlines())
    if actual != set(refs):
        raise DeliveryError("snapshot includes unexpected refs")


def verify_bundle(source: Path, bundle: Path, refs: dict[str, str], stage: Path,
                  primary_ref: str = MAIN_REF) -> dict:
    head_rows = git(source, "bundle", "list-heads", str(bundle)).decode().splitlines()
    actual = {row.split(" ", 1)[1]: row.split(" ", 1)[0] for row in head_rows}
    if actual != refs:
        raise DeliveryError("bundle refs do not exactly match the requested snapshots")
    with bundle.open("rb") as handle:
        first = handle.readline()
        if not first.startswith(b"# v"):
            raise DeliveryError("invalid Git bundle header")
        for row in handle:
            if row == b"\n":
                break
            if row.startswith(b"-"):
                raise DeliveryError("bundle has history prerequisites and is not standalone")
    clone = stage / "restored"
    hooks = stage / "empty-hooks"
    hooks.mkdir()
    branch = primary_ref.removeprefix("refs/heads/")
    git(stage, "-c", "core.autocrlf=false", "-c", f"core.hooksPath={hooks}",
        "clone", "--no-checkout", "--no-local", "--origin", "handoff", "--branch", branch,
        str(bundle), str(clone))
    git(clone, "config", "core.autocrlf", "false")
    git(clone, "config", "core.hooksPath", str(hooks))
    git(clone, "bundle", "verify", str(bundle))
    git(clone, "fsck", "--full", "--no-reflogs")
    restored = []
    for refname, oid in refs.items():
        if not refname.startswith("refs/heads/"):
            if git(clone, "rev-parse", refname).decode().strip() != oid:
                raise DeliveryError("annotated/lightweight tag identity was not restored")
            continue
        name = refname.removeprefix("refs/heads/")
        if refname == primary_ref:
            git(clone, "checkout", name)
        else:
            git(clone, "checkout", "--no-guess", "-b", name, "handoff/" + name)
        if resolve(clone, "HEAD") != oid:
            raise DeliveryError(f"restored branch points to a different commit: {name}")
        independence = check_independent(clone, require_clean=True)
        entries = tree_entries(clone, oid)
        algorithm = git(clone, "rev-parse", "--show-object-format").decode().strip()
        for path, expected in entries.items():
            restored_file = clone.joinpath(*PurePosixPath(path).parts)
            if not restored_file.is_file() or blob_identifier(restored_file.read_bytes(), algorithm) != expected:
                raise DeliveryError(f"restored checkout differs from Git tree: {path}")
        restored.append({"branch": name, "commit": oid, "tree": git(clone, "rev-parse", oid + "^{tree}").decode().strip(),
                         "exact_worktree_blob_bytes": True, "file_count": len(entries), **independence})
    primary_commit = refs[primary_ref]
    baseline = baseline_commit(clone, primary_commit)
    git(clone, "merge-base", "--is-ancestor", baseline, primary_commit)
    git(clone, "diff", "--check", baseline, primary_commit)
    clone_env = {key: value for key, value in os.environ.items()
                 if not key.startswith(("GIT_", "TARS_", "ANTHROPIC_", "OPENAI_"))}
    clone_env.update({"PYTHONPATH": str(clone / "src"), "PYTHONDONTWRITEBYTECODE": "1"})
    branch_names = [ref.removeprefix("refs/heads/") for ref in refs if ref.startswith("refs/heads/")]
    public = subprocess.run([sys.executable, str(clone / "scripts/check_public_content.py"),
                             "--history", "--refs", *branch_names], cwd=clone,
                            env=clone_env, capture_output=True, timeout=300, check=False)
    if public.returncode:
        raise DeliveryError("bundle clone public worktree/history gate failed")
    clone_scan = json.loads((clone / "build/qa/public-content.json").read_text(encoding="utf-8"))
    if set(clone_scan["refs"]) != set(branch_names) or clone_scan["findings"]:
        raise DeliveryError("bundle clone public scan did not cover the selected restored branches")
    return {"exact_refs": actual, "prerequisites": 0, "fsck": "passed",
            "baseline_commit": baseline, "baseline_reachable": True,
            "diff_check": "passed", "public_history_refs": clone_scan["refs"],
            "public_history_blob_count": clone_scan["history_blobs"],
            "public_history_unresolved_findings": 0,
            "qa_interpreter": "authorized formal environment; clone scripts/ROOT and PYTHONPATH",
            "restored_branches": restored}


def preflight(repo: Path, main_ref: str, stable_tag: str | None) -> tuple[str, dict]:
    if git(repo, "branch", "--show-current").decode().strip() != "main":
        raise DeliveryError("leave the source checkout on main before final export")
    check_independent(repo, require_clean=True)
    if git(repo, "stash", "list", "--format=%H").strip():
        raise DeliveryError("resolve pending stash entries before final export")
    main = resolve(repo, main_ref)
    if resolve(repo, MAIN_REF) != main:
        raise DeliveryError("M must be the exact tip of main")
    baseline = baseline_commit(repo, main)
    git(repo, "diff", "--check", baseline, main)
    tree_entries(repo, main)
    refs = {MAIN_REF: main}
    if stable_tag:
        tag_ref = "refs/tags/" + stable_tag
        git(repo, "check-ref-format", tag_ref)
        if resolve(repo, tag_ref) != main:
            raise DeliveryError("requested stable tag does not point to M")
        refs[tag_ref] = git(repo, "rev-parse", "--verify", tag_ref).decode().strip()
    return main, refs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--main-ref", required=True, help="Exact selected main commit or ref")
    parser.add_argument("--stable-tag", help="Omit for a candidate: no stability claim or invented tag")
    parser.add_argument("--output", type=Path, required=True, help="New empty directory outside the repository")
    parser.add_argument("--acceptance-summary", type=Path, help="Sanitized final-M QA JSON; statuses/skips are copied unchanged")
    parser.add_argument("--preflight", action="store_true", help="Read-only checks only; no exports")
    args = parser.parse_args()
    repo, output = args.repo.resolve(strict=True), args.output.resolve()
    m, refs = preflight(repo, args.main_ref, args.stable_tag)
    acceptance = (json.loads(args.acceptance_summary.read_text(encoding="utf-8"))
                  if args.acceptance_summary else {"status": "not_supplied", "pending": ["runtime acceptance summary not supplied"]})
    if not isinstance(acceptance, dict):
        raise DeliveryError("acceptance summary must be a sanitized JSON object")
    if output == repo or output.is_relative_to(repo):
        raise DeliveryError("delivery output must be outside the source repository")
    if args.preflight:
        print(json.dumps({"preflight": "passed", "commit": m, "stable_tag": args.stable_tag}))
        return 0
    if output.exists() and any(output.iterdir()):
        raise DeliveryError("refusing to overwrite a non-empty delivery directory")
    # This checker reports only detector identities/fingerprints, never secret values.
    branch_names = [ref.removeprefix("refs/heads/") for ref in refs if ref.startswith("refs/heads/")]
    scan = subprocess.run([sys.executable, str(repo / "scripts/check_public_content.py"), "--history",
                           "--refs", *branch_names], cwd=repo,
                          capture_output=True, timeout=300, check=False)
    if scan.returncode:
        raise DeliveryError("public worktree/reachable-history scan failed; inspect build/qa/public-content.json")
    public_scan = json.loads((repo / "build/qa/public-content.json").read_text(encoding="utf-8"))
    if set(public_scan["refs"]) != set(branch_names) or public_scan["findings"]:
        raise DeliveryError("public scan did not cover the selected delivery branches cleanly")
    output.mkdir(parents=True, exist_ok=True)
    zip_path, bundle_path = output / f"TARS-Agent-main-{m[:12]}.zip", output / f"TARS-Agent-{m[:12]}.bundle"
    with tempfile.TemporaryDirectory(prefix=".verify-", dir=output) as temporary:
        stage = Path(temporary).resolve()
        if not stage.is_relative_to(output):
            raise DeliveryError("temporary verification directory escaped output")
        bare = stage / "snapshot.git"
        snapshot_refs(repo, bare, refs, MAIN_REF)
        git(bare, "archive", "--format=zip", "--prefix=TARS-Agent/", f"--output={zip_path}", m)
        zip_check = verify_zip(bare, m, zip_path)
        git(bare, "bundle", "create", str(bundle_path), *refs)
        bundle_check = verify_bundle(bare, bundle_path, refs, stage)
    # A concurrent source ref/worktree change must not be silently called the final handoff.
    confirmed_m, confirmed_refs = preflight(repo, m, args.stable_tag)
    if (confirmed_m, confirmed_refs) != (m, refs):
        raise DeliveryError("source refs changed during export")
    verification = {"zip": zip_check, "bundle": bundle_check, "public_scan": public_scan}
    verification_path = output / "verification.json"
    verification_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    artifacts = [{"file": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size}
                 for path in (zip_path, bundle_path, verification_path)]
    manifest = {"schema_version": 2, "created_utc": datetime.now(UTC).isoformat(),
                "qualification": "tagged_snapshot" if args.stable_tag else "candidate_no_stable_tag",
                "baseline_commit": baseline_commit(repo, m), "commit": m,
                "stable_tag": args.stable_tag, "refs": refs,
                "main_tree": git(repo, "rev-parse", m + "^{tree}").decode().strip(),
                "main_lock_sha256": hashlib.sha256(git(repo, "show", m + ":uv.lock")).hexdigest(),
                "artifacts": artifacts,
                "acceptance": acceptance,
                "acceptance_summary_sha256": file_sha256(args.acceptance_summary) if args.acceptance_summary else None,
                "acceptance_boundary": "Snapshot/export verification only. Runtime gates and skips retain their separately recorded status; a Git tag is not proof of passed runtime acceptance.",
                "hash_boundary": "This manifest intentionally excludes its own checksum and the checksum sidecars to avoid a self-reference cycle."}
    manifest_path = output / "handoff-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_sha = file_sha256(manifest_path)
    (output / "handoff-manifest.sha256").write_text(f"{manifest_sha}  handoff-manifest.json\n", encoding="ascii")
    sums = [f"{item['sha256']}  {item['file']}" for item in artifacts]
    sums.append(f"{manifest_sha}  handoff-manifest.json")
    (output / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="ascii")
    print(json.dumps({"status": "exported_and_restored", "commit": m,
                      "qualification": manifest["qualification"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DeliveryError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print("Delivery blocked: " + str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
