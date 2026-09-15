from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from scripts import check_public_content as scanner

PATH = "docs/experimental/OVERLAY_APPLIED.json"


def test_only_real_ancestor_commit_scalar_is_normalized(monkeypatch) -> None:
    calls = []
    def git(*args, **kwargs):
        calls.append(args)
        return b"commit\n" if args[0] == "cat-file" else b""
    monkeypatch.setattr(scanner, "git", git)
    data = json.dumps({"base_main_commit": "a" * 40, "other": "unchanged"}).encode()
    result = json.loads(scanner.normalize_verified_metadata(data, PATH))
    assert result == {"base_main_commit": "verified_public_git_commit", "other": "unchanged"}
    assert calls == [("cat-file", "-t", "a" * 40), ("merge-base", "--is-ancestor", "a" * 40, "refs/heads/main")]


@pytest.mark.parametrize("failure", ["forged_object", "not_ancestor"])
def test_forged_hex_or_non_ancestor_is_not_exempted(monkeypatch, failure) -> None:
    def git(*args, **kwargs):
        if failure == "forged_object" or args[0] == "merge-base":
            raise RuntimeError("Git validation failed")
        return b"commit\n"
    monkeypatch.setattr(scanner, "git", git)
    with pytest.raises(RuntimeError):
        scanner.normalize_verified_metadata(json.dumps({"base_main_commit": "f" * 40}).encode(), PATH)


def test_other_fields_still_reach_normal_secret_detection(monkeypatch) -> None:
    monkeypatch.setattr(scanner, "git", lambda *args, **kwargs: b"commit\n" if args[0] == "cat-file" else b"")
    generated = hashlib.sha256(b"unreviewed metadata test field").hexdigest()
    data = json.dumps({"base_main_commit": "a" * 40, "unreviewed": generated}).encode()
    normalized = scanner.normalize_verified_metadata(data, PATH)
    assert json.loads(normalized)["unreviewed"] == generated
    program = r"""import json, sys
from scripts import check_public_content as scanner
scanner.git = lambda *args, **kwargs: b"commit\n" if args[0] == "cat-file" else b""
print(json.dumps(scanner.inspect_blob(sys.stdin.buffer.read(), "test", sys.argv[1], set())))
"""
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", program, PATH],
        input=data, capture_output=True, check=True, cwd=scanner.ROOT,
    )
    findings = json.loads(result.stdout)
    assert any(item["type"] == "Hex High Entropy String" for item in findings)


def test_other_paths_do_not_get_commit_metadata_exemption(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("must not validate or normalize an unrelated path")
    monkeypatch.setattr(scanner, "git", forbidden)
    data = b'{"base_main_commit":"not-a-commit","other":"untouched"}'
    assert scanner.normalize_verified_metadata(data, "docs/unrelated.json") == data


def test_cli_detects_utf8_canary_equally_from_default_and_utf8_python(tmp_path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / "check_public_content.py"
    shutil.copyfile(Path(scanner.__file__), script)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    canary = hashlib.sha256(b"synthetic unicode scan regression fixture").hexdigest()
    (tmp_path / "canary.py").write_text(
        '# 中文说明：本值仅为合成扫描样本\napi_key = "' + canary + '"\n', encoding="utf-8"
    )
    environment = dict(os.environ)
    environment.pop("PYTHONUTF8", None)
    environment["PYTHONIOENCODING"] = "utf-8"
    reports = []
    for flags in ([], ["-X", "utf8"], ["-X", "utf8=0"]):
        completed = subprocess.run(
            [sys.executable, *flags, str(script)], cwd=tmp_path,
            env=environment, capture_output=True, check=False,
        )
        assert completed.returncode == 1
        report = json.loads((tmp_path / "build/qa/public-content.json").read_text(encoding="utf-8"))
        canary_findings = [item for item in report["findings"] if item["path"] == "canary.py"]
        assert {item["type"] for item in canary_findings} == {
            "Secret Keyword", "Hex High Entropy String"
        }
        reports.append(canary_findings)
    assert reports[0] == reports[1] == reports[2]


def test_direct_inspect_fails_closed_without_utf8_mode() -> None:
    program = """from scripts import check_public_content as scanner
try:
    scanner.inspect_blob(b"public content", "test", "example.py", set())
except RuntimeError as error:
    assert "UTF-8 mode" in str(error)
    print("rejected")
else:
    raise AssertionError("Non-UTF8 direct inspection silently returned")
"""
    result = subprocess.run(
        [sys.executable, "-X", "utf8=0", "-c", program],
        capture_output=True, check=True, cwd=scanner.ROOT,
    )
    assert result.stdout.strip() == b"rejected"


def test_non_utf8_file_is_a_gate_failure_in_utf8_process() -> None:
    program = """import json, sys
from scripts import check_public_content as scanner
print(json.dumps(scanner.inspect_blob(sys.stdin.buffer.read(), "test", "canary.py", set())))
"""
    canary = hashlib.sha256(b"synthetic GBK scan regression fixture").hexdigest()
    payload = ('# 中文说明\napi_key = "' + canary + '"\n').encode("gbk")
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", program], input=payload,
        capture_output=True, check=True, cwd=scanner.ROOT,
    )
    assert json.loads(result.stdout) == [
        {"object": "test", "path": "canary.py", "type": "unscannable_encoding"}
    ]
