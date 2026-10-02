"""Scan public worktree files and all reachable blobs of the named delivery refs."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from detect_secrets.core.scan import scan_file
from detect_secrets.settings import default_settings

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PARTS = {".env", ".venv", "node_modules", ".kama", ".tars-baseline", "__pycache__"}


def git(*args: str, input_data: bytes | None = None) -> bytes:
    result = subprocess.run(["git", "--no-optional-locks", "-c", "core.quotePath=false", *args],
                            cwd=ROOT, input=input_data, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError("Git public-content inspection failed: " + " ".join(args[:2]))
    return result.stdout


def prohibited(path: str) -> bool:
    item = Path(path)
    return bool(set(item.parts) & PRIVATE_PARTS) or item.suffix in {".db", ".sqlite", ".sqlite3", ".log"}


def normalize_verified_metadata(data: bytes, path: str) -> bytes:
    """Only a proven public Git commit scalar is exempted from entropy detection."""
    if path != "docs/experimental/OVERLAY_APPLIED.json":
        return data
    record = json.loads(data)
    if not isinstance(record, dict):
        raise ValueError("Invalid overlay application metadata")
    commit = record.get("base_main_commit")
    if commit is None:
        return data  # Uncommitted preview; no commit value is being exempted.
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise ValueError("Invalid overlay base commit metadata")
    if git("cat-file", "-t", commit).strip() != b"commit":
        raise ValueError("Overlay base metadata is not a real Git commit")
    git("merge-base", "--is-ancestor", commit, "refs/heads/main")
    record["base_main_commit"] = "verified_public_git_commit"
    # Every other field remains in the scanner input and still needs normal review.
    return json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")


def inspect_blob(data: bytes, identity: str, path: str,
                 reviewed: set[str]) -> list[dict[str, Any]]:
    if not sys.flags.utf8_mode:
        raise RuntimeError("Public-content scanning requires Python UTF-8 mode (-X utf8)")
    findings: list[dict[str, Any]] = []
    if prohibited(path):
        return [{"object": identity, "path": path, "type": "prohibited_private_path"}]
    try:
        data.decode("utf-8")
    except UnicodeError:
        return [{"object": identity, "path": path, "type": "unscannable_encoding"}]
    with tempfile.TemporaryDirectory(prefix="tars-public-scan-") as directory:
        snapshot = Path(directory) / Path(path).name
        data = normalize_verified_metadata(data, path)
        if path == "docs/baseline/secret-scan-reviews.json":
            records = json.loads(data)
            if not isinstance(records, list):
                raise ValueError("Invalid review records")
            for record in records:
                if not re.fullmatch(r"[0-9a-f]{64}", record.get("fingerprint", "")):
                    raise ValueError("Invalid generated review fingerprint")
                record["fingerprint"] = "generated_sha256"
            data = json.dumps(records).encode("utf-8")
        snapshot.write_bytes(data)
        with default_settings():
            for secret in scan_file(str(snapshot)):
                fingerprint = hashlib.sha256(
                    (path + "\0" + secret.type + "\0" + secret.secret_hash).encode()
                ).hexdigest()
                if fingerprint not in reviewed:
                    findings.append({"object": identity, "path": path,
                                     "line": secret.line_number,
                                     "type": secret.type, "fingerprint": fingerprint})
    return findings


def resolve_delivery_refs(refs: list[str]) -> dict[str, str]:
    """Resolve every requested commit-ish; never silently omit an invalid ref."""
    resolved: dict[str, str] = {}
    for ref in refs:
        try:
            commit = git("rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}")
        except RuntimeError as exc:
            raise ValueError(f"Requested delivery ref does not resolve to a commit: {ref}") from exc
        resolved[ref] = commit.decode("ascii").strip()
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--refs", nargs="+", help="Commit-ish values to scan; defaults to HEAD")
    args = parser.parse_args()
    if args.refs is not None and not args.history:
        parser.error("--refs requires --history")
    resolved_refs = resolve_delivery_refs(args.refs or ["HEAD"]) if args.history else {}
    reviewed_path = ROOT / "docs/baseline/secret-scan-reviews.json"
    reviews = json.loads(reviewed_path.read_text(encoding="utf-8")) if reviewed_path.exists() else []
    if any(not item.get("reason") for item in reviews):
        raise ValueError("Every secret-scan exception needs an explicit review reason")
    reviewed = {item["fingerprint"] for item in reviews}
    findings: list[dict[str, Any]] = []
    paths = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
    for relative in sorted(set(paths)):
        if not relative:
            continue
        file = ROOT / relative
        if file.is_file():
            if not file.resolve().is_relative_to(ROOT.resolve()):
                findings.append({"path": relative, "type": "link_escapes_public_root"})
            else:
                findings.extend(inspect_blob(file.read_bytes(), "worktree", relative, reviewed))
    blob_count = 0
    if args.history:
        objects = git("rev-list", "--objects", *resolved_refs.values()).decode().splitlines()
        path_by_id = dict(line.split(" ", 1) for line in objects if " " in line)
        metadata = git("cat-file", "--batch-check", input_data=("\n".join(path_by_id) + "\n").encode())
        for line in metadata.decode().splitlines():
            oid, kind, _size = line.split()
            if kind != "blob":
                continue
            blob_count += 1
            findings.extend(inspect_blob(git("cat-file", "blob", oid), oid, path_by_id[oid], reviewed))
    output = ROOT / "build/qa/public-content.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"refs": list(resolved_refs), "resolved_refs": resolved_refs,
                                  "history_blobs": blob_count,
                                  "reviewed_exceptions": reviews, "findings": findings},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Public scan: {len(findings)} unresolved findings; {blob_count} history blobs")
    print("Details contain only paths, detector types and review fingerprints; no secret values.")
    return 1 if findings else 0


if __name__ == "__main__":
    # detect-secrets uses open() without encoding and skips UnicodeDecodeError.
    # Keep its complete scanner behavior, but make decoding independent of locale.
    if not sys.flags.utf8_mode:
        child = subprocess.run(
            [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), *sys.argv[1:]],
            check=False,
        )
        raise SystemExit(child.returncode)
    raise SystemExit(main())
