from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src" / "tars_agent"

# Web is an HTTP/SSE adapter, so its Core dependency surface is intentionally
# smaller than the public Python package surface.  Symbols not listed here must
# be exposed through Wire Protocol V2 instead of imported from a business layer.
_WEB_CORE_IMPORT_ALLOWLIST: dict[str, frozenset[str]] = {
    "tars_agent.core.transport.socket_client": frozenset({"SocketClient"}),
    "tars_agent.core.bus.envelope": frozenset(
        {"EventPushEnvelope", "EventOverflowEnvelope"}
    ),
}

# The process launcher needs the Core endpoint from shared configuration.  Keep
# this exception file-scoped so adapter modules cannot grow a config dependency.
_WEB_CORE_FILE_IMPORT_ALLOWLIST: dict[
    str,
    dict[str, frozenset[str]],
] = {
    "web/cli.py": {
        "tars_agent.core.config": frozenset({"get_config"}),
    }
}


def _imports(module: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _web_core_import_errors(module: ast.AST, relative: Path) -> list[str]:
    allowed = dict(_WEB_CORE_IMPORT_ALLOWLIST)
    allowed.update(_WEB_CORE_FILE_IMPORT_ALLOWLIST.get(relative.as_posix(), {}))
    errors: list[str] = []

    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "tars_agent.core" or alias.name.startswith(
                    "tars_agent.core."
                ):
                    errors.append(
                        f"{relative}: Web adapter Core module import is not allowlisted: "
                        f"{alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom) and node.module:
            core_module = node.module
            if core_module != "tars_agent.core" and not core_module.startswith(
                "tars_agent.core."
            ):
                continue
            allowed_symbols = allowed.get(core_module)
            imported_symbols = {alias.name for alias in node.names}
            forbidden_symbols = (
                imported_symbols
                if allowed_symbols is None
                else imported_symbols - allowed_symbols
            )
            if forbidden_symbols:
                rendered = ", ".join(
                    f"{core_module}.{symbol}" for symbol in sorted(forbidden_symbols)
                )
                errors.append(
                    f"{relative}: Web adapter Core import is not allowlisted: {rendered}"
                )
    return errors


def check(source_root: Path = SOURCE) -> list[str]:
    errors: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root)
        source_text = path.read_text(encoding="utf-8")
        module = ast.parse(source_text, filename=str(relative))
        imports = _imports(module)

        # Migration/bootstrap code may use SQLite mechanics. Runtime and adapters
        # must go through persistence services and Repository APIs.
        if relative.parts[0] != "core" or "persistence" not in relative.parts:
            if "sqlite3" in imports:
                errors.append(f"{relative}: direct sqlite3 import outside persistence")

        if relative.parts[0] == "web":
            errors.extend(_web_core_import_errors(module, relative))
            forbidden = [
                name
                for name in imports
                if name.startswith("sqlalchemy")
                or name == "sqlite3"
            ]
            if forbidden:
                errors.append(
                    f"{relative}: Web adapter imports persistence layer: "
                    + ", ".join(sorted(forbidden))
                )
    return errors


def main() -> None:
    errors = check()
    if errors:
        raise SystemExit("Architecture boundary violations:\n" + "\n".join(errors))
    print("Architecture boundaries OK")


if __name__ == "__main__":
    main()
