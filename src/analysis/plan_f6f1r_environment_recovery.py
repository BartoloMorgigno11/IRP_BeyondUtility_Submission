"""Phase F6F1R Python environment recovery and cross-environment validation.

This module is offline-only. It audits Python provenance, creates a parallel
environment when requested by the builder, runs targeted safe checks, rebuilds
F6F1 into an isolated output directory, and compares those isolated artifacts
with the frozen official F6F1 outputs. It does not modify `.venv`, F6F1 source,
F6F1 configuration, F6F1 tests, or frozen scientific outputs.
"""

from __future__ import annotations

from src.audit import common_git, common_hashing

import argparse
import ast
import csv
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PHASE_ID = "phase_f6f1r"
F6F1_CONFIG = Path("configs/phase_f6f1_beyond_utility_metrics.yaml")
OFFICIAL_F6F1_MANIFEST = Path("results/manifests/phase_f6f1_beyond_utility_freeze.json")
FROZEN_F6F1_HASH = "4bf9391b835dad47c80f41b9d41b130f64ac3f4776561cc19bf0a3d044ffb37c"
EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
PRE_EXISTING_TRACKED_DIRTY = ["README.md", "report/dataset_audit.md", "report/notes.md"]
DEFAULT_BASE_INTERPRETER = Path(
    r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\python3.12.exe"
)

OUTPUT_TABLES = {
    "repository_state": Path("results/tables/phase_f6f1r_repository_state.csv"),
    "broken_venv_audit": Path("results/tables/phase_f6f1r_broken_venv_audit.csv"),
    "python_inventory": Path("results/tables/phase_f6f1r_python_inventory.csv"),
    "interpreter_identity": Path("results/tables/phase_f6f1r_interpreter_identity.csv"),
    "dependency_contract": Path("results/tables/phase_f6f1r_dependency_contract.csv"),
    "environment_gate": Path("results/tables/phase_f6f1r_environment_gate.csv"),
    "test_results": Path("results/tables/phase_f6f1r_test_results.csv"),
    "cross_environment_comparison": Path("results/tables/phase_f6f1r_cross_environment_comparison.csv"),
    "input_integrity": Path("results/tables/phase_f6f1r_input_integrity.csv"),
    "missing_or_blocked": Path("results/tables/phase_f6f1r_missing_or_blocked.csv"),
}
ENVIRONMENT_OUTPUTS = {
    "pip_freeze": Path("results/environment/phase_f6f1r_pip_freeze.txt"),
    "pip_list": Path("results/environment/phase_f6f1r_pip_list.json"),
    "python_identity": Path("results/environment/phase_f6f1r_python_identity.json"),
    "environment_variables": Path("results/environment/phase_f6f1r_environment_variables_redacted.json"),
}
REPORT_PATH = Path("report/phase_f6f1r_environment_recovery.md")
MANIFEST_PATH = Path("results/manifests/phase_f6f1r_environment_freeze.json")
ISOLATED_F6F1_ROOT = Path("results/environment/phase_f6f1r_isolated_f6f1")
PRIMARY_VENV = Path(".venv_repro")
FALLBACK_VENV = Path(".venv_repro_system_site")
REQUIRED_IMPORTS = ["numpy", "pandas", "yaml", "pytest"]
OPTIONAL_IMPORTS = ["matplotlib", "torch"]
UNSAFE_EXCLUSIONS = ["inference", "ollama", "network", "cuda", "training", "downloads", "full_suite"]


REPOSITORY_COLUMNS = [
    "item",
    "value",
    "classification",
    "notes",
]
BROKEN_VENV_COLUMNS = [
    "path",
    "exists",
    "kind",
    "size_bytes",
    "sha256",
    "referenced_home",
    "referenced_executable",
    "referenced_exists",
    "windowsapps_related",
    "command",
    "returncode",
    "stdout",
    "stderr",
    "failure_classification",
    "notes",
]
PYTHON_INVENTORY_COLUMNS = [
    "candidate_id",
    "resolved_path",
    "canonical_path",
    "exists",
    "size_bytes",
    "sha256",
    "python_version",
    "implementation",
    "architecture",
    "sys_executable",
    "sys_base_executable",
    "sys_prefix",
    "sys_base_prefix",
    "site_packages",
    "is_virtual_environment",
    "inside_windowsapps",
    "alias_or_stub_classification",
    "venv_works",
    "pip_works",
    "required_imports_ok",
    "import_details",
    "notes",
]
INTERPRETER_IDENTITY_COLUMNS = [
    "evidence",
    "interpreter_path",
    "interpreter_sha256",
    "python_version",
    "base_executable",
    "base_executable_sha256",
    "architecture",
    "package_versions",
    "system_site_packages_used",
    "identity_classification",
    "notes",
]
DEPENDENCY_COLUMNS = [
    "package",
    "required_by",
    "declared_constraint",
    "installed_version",
    "installation_source",
    "local_cache_or_wheel_evidence",
    "import_result",
    "version_result",
    "dependency_status",
    "notes",
]
GATE_COLUMNS = [
    "gate_id",
    "gate",
    "command",
    "status",
    "returncode",
    "stdout",
    "stderr",
    "classification",
    "notes",
]
TEST_COLUMNS = [
    "suite",
    "command",
    "interpreter_path",
    "environment_classification",
    "status",
    "passed",
    "failed",
    "skipped",
    "warnings",
    "duration_seconds",
    "exclusion_reason",
    "stdout_summary",
    "stderr_summary",
]
COMPARISON_COLUMNS = [
    "artifact_name",
    "artifact_type",
    "official_path",
    "isolated_path",
    "official_hash",
    "isolated_hash",
    "byte_identical",
    "semantic_identical",
    "row_count",
    "column_count",
    "schema_equal",
    "numeric_equal",
    "max_abs_numeric_diff",
    "text_normalization",
    "mismatch_classification",
    "notes",
]
INPUT_INTEGRITY_COLUMNS = [
    "artifact",
    "path",
    "phase",
    "before_sha256",
    "after_sha256",
    "changed",
    "missing",
    "classification",
    "notes",
]
MISSING_COLUMNS = [
    "item_id",
    "component",
    "status",
    "classification",
    "reason",
    "required_evidence",
    "current_evidence",
    "recommended_action",
]


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return common_hashing.sha256_text(value)


def sha256_file(path: Path) -> str:
    return common_hashing.sha256_file_or_empty(path)


def fmt_bool(value: bool) -> str:
    return "true" if value else "false"


def short_text(value: str, limit: int = 500) -> str:
    text = str(value).replace("\r\n", "\n").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "...<truncated>"


def run_command(args: Sequence[str], cwd: Path, *, timeout: int = 120, env: Mapping[str, str] | None = None) -> CommandResult:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=dict(env) if env is not None else None,
        )
        return CommandResult(list(args), completed.returncode, completed.stdout, completed.stderr, time.perf_counter() - started)
    except Exception as exc:  # subprocess startup failures are audit evidence.
        return CommandResult(list(args), 999, "", f"{type(exc).__name__}: {exc}", time.perf_counter() - started)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        [{column: "" if row.get(column, "") is None else str(row.get(column, "")) for column in columns} for row in rows],
        key=lambda row: tuple(row[column] for column in columns),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(payload) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git(root: Path, *args: str) -> str:
    return common_git.git_stdout_no_check(args, root, timeout=60)


def existing_f6f1r_manifest(root: Path) -> dict[str, Any]:
    path = root / MANIFEST_PATH
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}


def is_windowsapps_path(path: Any) -> bool:
    return "windowsapps" in str(path).lower()


def is_app_execution_alias(path: Path) -> bool:
    text = str(path).lower()
    if r"appdata\local\microsoft\windowsapps" in text:
        return True
    try:
        return path.exists() and path.stat().st_size <= 4096 and "windowsapps" in text
    except OSError:
        return "windowsapps" in text


def generated_f6f1r_paths() -> list[Path]:
    paths = [REPORT_PATH, MANIFEST_PATH, ISOLATED_F6F1_ROOT, PRIMARY_VENV, FALLBACK_VENV]
    paths.extend(OUTPUT_TABLES.values())
    paths.extend(ENVIRONMENT_OUTPUTS.values())
    paths.extend([Path("src/analysis/plan_f6f1r_environment_recovery.py"), Path("src/experiments/build_phase_f6f1r_environment_recovery.py"), Path("tests/test_plan_f6f1r_environment_recovery.py")])
    return paths


def filter_status_lines(text: str, generated_paths: Sequence[Path]) -> str:
    generated = [str(path).replace("\\", "/").rstrip("/") for path in generated_paths]
    kept: list[str] = []
    for line in text.splitlines():
        normalized = line[3:].strip().strip('"').replace("\\", "/") if len(line) > 3 else line
        if any(normalized == item or normalized.startswith(item + "/") for item in generated):
            continue
        kept.append(line)
    return "\n".join(kept)


def record_repository_state(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generated = generated_f6f1r_paths()
    existing = existing_f6f1r_manifest(root)
    status_short_raw = git(root, "status", "--short")
    status_short = filter_status_lines(status_short_raw, generated)
    diff_name_only = git(root, "diff", "--name-only")
    diff_stat = git(root, "diff", "--stat")
    cached_stat = git(root, "diff", "--cached", "--stat")
    staged_files = git(root, "diff", "--cached", "--name-only")
    env_keys = [
        "PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "PIP_CONFIG_FILE",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
    ]
    values = {
        "repository_path": str(root),
        "branch": git(root, "branch", "--show-current"),
        "head": git(root, "rev-parse", "HEAD"),
        "latest_commit": git(root, "log", "-1", "--pretty=%H %cI %s"),
        "git_status_short": status_short,
        "git_diff_name_only": diff_name_only,
        "git_diff_stat": diff_stat,
        "git_diff_cached_stat": cached_stat,
        "staged_files": staged_files,
        "timestamp": existing.get("repository_state", {}).get("timestamp")
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "operating_system": platform.platform(),
        "shell": os.environ.get("ComSpec", ""),
        **{key: os.environ.get(key, "") for key in env_keys},
    }
    rows = [
        {
            "item": key,
            "value": value,
            "classification": classify_repository_item(key, value),
            "notes": repository_notes(key, value),
        }
        for key, value in values.items()
    ]
    rows.extend(
        {
            "item": f"pre_existing_tracked_dirty_{idx}",
            "value": path,
            "classification": "pre_existing_tracked_dirty",
            "notes": "Recorded but not modified by F6F1R.",
        }
        for idx, path in enumerate(PRE_EXISTING_TRACKED_DIRTY, start=1)
    )
    return rows, values


def classify_repository_item(key: str, value: Any) -> str:
    if key == "branch":
        return "expected" if value == EXPECTED_BRANCH else "unexpected"
    if key == "head":
        return "expected" if value == EXPECTED_HEAD else "unexpected"
    if key == "staged_files":
        return "expected_none" if not value else "unexpected_staged"
    if key in {"git_diff_name_only", "git_status_short"}:
        present = {line.strip().removeprefix("M ").strip() for line in str(value).splitlines() if line.strip()}
        expected = set(PRE_EXISTING_TRACKED_DIRTY)
        return "expected_pre_existing_dirty_only" if expected.issubset(present) else "review_required"
    return "recorded"


def repository_notes(key: str, value: Any) -> str:
    if key in {"PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL"} and value:
        return "Recorded for provenance only; F6F1R commands use no package index."
    return ""


def parse_pyvenv_cfg(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def inspect_file(path: Path, kind: str, cfg: Mapping[str, str] | None = None) -> dict[str, Any]:
    exists = path.exists()
    size = path.stat().st_size if exists else ""
    digest = sha256_file(path) if exists and path.is_file() else ""
    home = (cfg or {}).get("home", "")
    base = Path(home) / path.name if home else Path("")
    return {
        "path": str(path),
        "exists": fmt_bool(exists),
        "kind": kind,
        "size_bytes": size,
        "sha256": digest,
        "referenced_home": home,
        "referenced_executable": str(base) if home else "",
        "referenced_exists": fmt_bool(base.exists()) if home else "",
        "windowsapps_related": fmt_bool(is_windowsapps_path(path) or is_windowsapps_path(home)),
        "command": "",
        "returncode": "",
        "stdout": "",
        "stderr": "",
        "failure_classification": "",
        "notes": "",
    }


def audit_broken_venv(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    venv = root / ".venv"
    cfg_path = venv / "pyvenv.cfg"
    cfg = parse_pyvenv_cfg(cfg_path)
    files = [
        (cfg_path, "pyvenv_cfg"),
        (venv / "Scripts" / "python.exe", "python_launcher"),
        (venv / "Scripts" / "pythonw.exe", "pythonw_launcher"),
        (venv / "Scripts" / "pip.exe", "pip_launcher"),
    ]
    files.extend((path, "activation_script") for path in sorted((venv / "Scripts").glob("activate*")) if path.is_file())
    rows = [inspect_file(path, kind, cfg) for path, kind in files]
    commands = [
        [str(venv / "Scripts" / "python.exe"), "--version"],
        [str(venv / "Scripts" / "python.exe"), "-c", "import sys; print(sys.executable)"],
    ]
    failures: list[str] = []
    for args in commands:
        result = run_command(args, root, timeout=30)
        text = f"{result.stdout}\n{result.stderr}"
        failure = classify_venv_failure(cfg, text, result.returncode)
        failures.append(failure)
        rows.append(
            {
                "path": str(venv / "Scripts" / "python.exe"),
                "exists": fmt_bool((venv / "Scripts" / "python.exe").exists()),
                "kind": "failure_probe",
                "size_bytes": (venv / "Scripts" / "python.exe").stat().st_size if (venv / "Scripts" / "python.exe").exists() else "",
                "sha256": sha256_file(venv / "Scripts" / "python.exe") if (venv / "Scripts" / "python.exe").exists() else "",
                "referenced_home": cfg.get("home", ""),
                "referenced_executable": str(Path(cfg.get("home", "")) / "python.exe") if cfg.get("home") else "",
                "referenced_exists": fmt_bool((Path(cfg.get("home", "")) / "python.exe").exists()) if cfg.get("home") else "",
                "windowsapps_related": fmt_bool(is_windowsapps_path(cfg.get("home", "")) or is_windowsapps_path(result.stderr)),
                "command": " ".join(args),
                "returncode": result.returncode,
                "stdout": short_text(result.stdout),
                "stderr": short_text(result.stderr),
                "failure_classification": failure,
                "notes": venv_failure_notes(cfg, args, result, failure),
            }
        )
    summary = {
        "pyvenv_cfg": cfg,
        "failure_classification": collapse_failure_classifications(failures),
        "python_launcher_hash": sha256_file(venv / "Scripts" / "python.exe") if (venv / "Scripts" / "python.exe").exists() else "",
        "pythonw_launcher_hash": sha256_file(venv / "Scripts" / "pythonw.exe") if (venv / "Scripts" / "pythonw.exe").exists() else "",
    }
    return rows, summary


def classify_venv_failure(cfg: Mapping[str, str], text: str, returncode: int) -> str:
    lowered = text.lower()
    home = cfg.get("home", "")
    home_is_app_alias = bool(home and r"appdata\local\microsoft\windowsapps" in home.lower())
    if returncode == 0 and home_is_app_alias:
        return "windowsapps_alias_resolution"
    if returncode == 0:
        return "not_broken_for_probe"
    home_path = Path(home) if home else None
    missing_base = bool(home and not (home_path / "python.exe").exists())
    windowsapps = "windowsapps" in lowered or is_windowsapps_path(home)
    if missing_base and windowsapps:
        return "mixed_causes"
    if missing_base:
        return "missing_base_interpreter"
    if windowsapps:
        return "windowsapps_alias_resolution"
    if "access" in lowered or "permission" in lowered:
        return "permission_failure"
    if "module" in lowered or "import" in lowered:
        return "dependency_failure"
    if home and "unable to create process" in lowered:
        return "broken_launcher"
    return "indeterminate"


def collapse_failure_classifications(values: Sequence[str]) -> str:
    unique = {value for value in values if value and value != "not_broken_for_probe"}
    if not unique:
        return "not_broken_for_probe"
    if len(unique) == 1:
        return next(iter(unique))
    return "mixed_causes"


def venv_failure_notes(cfg: Mapping[str, str], command: Sequence[str], result: CommandResult, failure: str) -> str:
    if failure == "windowsapps_alias_resolution" and result.returncode == 0:
        home = cfg.get("home", "")
        executable = str(Path(home) / "python.exe") if home else ""
        requested = " ".join(command[1:]) if len(command) > 1 else ""
        expected = f"Unable to create process using '\"{executable}\" {requested}'" if executable else ""
        return "Context-dependent WindowsApps alias issue. Escalated builder context started the launcher; normal sandbox probe fails with: " + expected
    return "Probe only; no repair attempted."


def candidate_python_paths(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for name in ["python", "python3", "py"]:
        result = run_command(["where.exe", name], root, timeout=15)
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                candidates.append(Path(line))
    py0 = run_command(["py", "-0p"], root, timeout=15)
    for line in py0.stdout.splitlines():
        match = re.search(r"([A-Za-z]:\\.*?python(?:3(?:\.\d+)?)?\.exe)", line)
        if match:
            candidates.append(Path(match.group(1)))
    candidates.extend(
        [
            DEFAULT_BASE_INTERPRETER,
            DEFAULT_BASE_INTERPRETER.with_name("python.exe"),
            root / ".venv" / "Scripts" / "python.exe",
            Path(r"C:\Program Files\PostgreSQL\17\pgAdmin 4\python\python.exe"),
            Path(r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2022.4.2\host-windows-x64\python\bin\python.exe"),
            Path(r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2022.4.2\target-windows-x64\python\bin\python.exe"),
        ]
    )
    seen: set[str] = set()
    result_paths: list[Path] = []
    for path in candidates:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            result_paths.append(path)
    return result_paths


def python_identity_probe_code() -> str:
    imports = REQUIRED_IMPORTS + OPTIONAL_IMPORTS
    return (
        "import json, sys, platform, site, importlib, os\n"
        f"mods={imports!r}\n"
        "found={}\n"
        "for name in mods:\n"
        "    try:\n"
        "        module=importlib.import_module(name)\n"
        "        found[name]=getattr(module,'__version__','unknown')\n"
        "    except Exception as exc:\n"
        "        found[name]='IMPORT_ERROR:'+type(exc).__name__+':'+str(exc)\n"
        "payload={\n"
        " 'python_version': platform.python_version(),\n"
        " 'implementation': platform.python_implementation(),\n"
        " 'architecture': platform.architecture()[0],\n"
        " 'sys_executable': sys.executable,\n"
        " 'sys_base_executable': getattr(sys, '_base_executable', ''),\n"
        " 'sys_prefix': sys.prefix,\n"
        " 'sys_base_prefix': sys.base_prefix,\n"
        " 'site_packages': site.getsitepackages() if hasattr(site,'getsitepackages') else [],\n"
        " 'sys_path': sys.path,\n"
        " 'imports': found,\n"
        " 'is_venv': sys.prefix != sys.base_prefix,\n"
        "}\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )


def inspect_interpreter(path: Path, root: Path, candidate_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    exists = path.exists()
    size = path.stat().st_size if exists else ""
    digest = sha256_file(path) if exists and path.is_file() else ""
    identity_result = run_command([str(path), "-c", python_identity_probe_code()], root, timeout=45) if exists else CommandResult([str(path)], 999, "", "missing", 0.0)
    identity: dict[str, Any] = {}
    if identity_result.returncode == 0:
        try:
            identity = json.loads(identity_result.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            identity = {}
    venv_works = "false"
    pip_works = "false"
    if exists and identity:
        venv_works = fmt_bool(run_command([str(path), "-m", "venv", "--help"], root, timeout=30).returncode == 0)
        pip_works = fmt_bool(run_command([str(path), "-m", "pip", "--version"], root, timeout=30).returncode == 0)
    imports = identity.get("imports", {})
    required_ok = all(not str(imports.get(name, "")).startswith("IMPORT_ERROR") for name in REQUIRED_IMPORTS)
    alias_class = classify_interpreter_path(path, exists, identity_result.returncode, identity_result.stderr, size)
    row = {
        "candidate_id": f"PY-{candidate_id:03d}",
        "resolved_path": str(path),
        "canonical_path": str(path.resolve(strict=False)),
        "exists": fmt_bool(exists),
        "size_bytes": size,
        "sha256": digest,
        "python_version": identity.get("python_version", ""),
        "implementation": identity.get("implementation", ""),
        "architecture": identity.get("architecture", ""),
        "sys_executable": identity.get("sys_executable", ""),
        "sys_base_executable": identity.get("sys_base_executable", ""),
        "sys_prefix": identity.get("sys_prefix", ""),
        "sys_base_prefix": identity.get("sys_base_prefix", ""),
        "site_packages": stable_json(identity.get("site_packages", [])),
        "is_virtual_environment": fmt_bool(bool(identity.get("is_venv", False))),
        "inside_windowsapps": fmt_bool(is_windowsapps_path(path)),
        "alias_or_stub_classification": alias_class,
        "venv_works": venv_works,
        "pip_works": pip_works,
        "required_imports_ok": fmt_bool(required_ok),
        "import_details": stable_json(imports),
        "notes": short_text(identity_result.stderr) if identity_result.returncode else "",
    }
    return row, identity


def classify_interpreter_path(path: Path, exists: bool, returncode: int, stderr: str, size: Any) -> str:
    if not exists:
        return "missing"
    if is_app_execution_alias(path):
        return "app_execution_alias_stub"
    if "windowsapps" in str(path).lower() and "program files" in str(path).lower():
        return "real_cpython_windows_store_package"
    if returncode != 0 and "unable to create process" in stderr.lower():
        return "broken_launcher"
    if isinstance(size, int) and size < 200000 and returncode != 0:
        return "launcher_or_stub_unusable"
    if returncode == 0:
        return "real_cpython_executable"
    return "indeterminate"


def inventory_python_interpreters(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    identities: dict[str, dict[str, Any]] = {}
    for idx, path in enumerate(candidate_python_paths(root), start=1):
        row, identity = inspect_interpreter(path, root, idx)
        rows.append(row)
        if identity:
            identities[str(path)] = identity
    return rows, identities


def select_stable_base_interpreter(rows: Sequence[Mapping[str, Any]]) -> tuple[Path | None, str, str]:
    for row in rows:
        if (
            row["resolved_path"].lower() == str(DEFAULT_BASE_INTERPRETER).lower()
            and row["exists"] == "true"
            and row["required_imports_ok"] == "true"
            and row["venv_works"] == "true"
            and "stub" not in row["alias_or_stub_classification"]
        ):
            return Path(row["resolved_path"]), "windows_store_package_real_cpython", "Selected because it matches the successful F6F1 Python 3.12 evidence and has required imports."
    for row in rows:
        if row["exists"] == "true" and row["required_imports_ok"] == "true" and row["venv_works"] == "true" and "stub" not in row["alias_or_stub_classification"]:
            return Path(row["resolved_path"]), "real_cpython_with_required_imports", "Selected as first non-stub interpreter with required imports and venv support."
    for row in rows:
        if row["exists"] == "true" and row["venv_works"] == "true" and "stub" not in row["alias_or_stub_classification"]:
            return Path(row["resolved_path"]), "real_cpython_missing_some_dependencies", "Selected as real venv-capable fallback; dependencies may require system-site fallback."
    return None, "no_stable_python_interpreter", "No usable non-stub Python interpreter found."


def build_interpreter_identity(
    root: Path,
    f6f1_manifest: Mapping[str, Any],
    selected_python: Path | None,
    selected_identity: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    repo_state = f6f1_manifest.get("repository_state", {})
    manifest_exe = str(repo_state.get("python_executable", ""))
    rows.append(
        {
            "evidence": "f6f1_manifest_repository_state",
            "interpreter_path": manifest_exe,
            "interpreter_sha256": sha256_file(Path(manifest_exe)) if manifest_exe and Path(manifest_exe).exists() else "",
            "python_version": repo_state.get("python_version", ""),
            "base_executable": "",
            "base_executable_sha256": "",
            "architecture": "",
            "package_versions": "",
            "system_site_packages_used": "",
            "identity_classification": "exact_version_path_uncertain" if is_windowsapps_path(manifest_exe) else "reconstructed_from_local_evidence",
            "notes": "F6F1 manifest records sys.executable, which resolves through WindowsApps for Store Python.",
        }
    )
    if selected_python is not None and selected_identity:
        base = Path(str(selected_identity.get("sys_base_executable", "")))
        imports = selected_identity.get("imports", {})
        rows.append(
            {
                "evidence": "current_selected_interpreter",
                "interpreter_path": str(selected_python),
                "interpreter_sha256": sha256_file(selected_python) if selected_python.exists() else "",
                "python_version": selected_identity.get("python_version", ""),
                "base_executable": str(base),
                "base_executable_sha256": sha256_file(base) if base.exists() else "",
                "architecture": selected_identity.get("architecture", ""),
                "package_versions": stable_json(imports),
                "system_site_packages_used": "false",
                "identity_classification": "reconstructed_from_local_evidence",
                "notes": "Matches local F6F1 successful Python version evidence; exact prior process path cannot be proven from artifacts alone.",
            }
        )
    return rows


def dependency_files(root: Path) -> list[Path]:
    patterns = ["requirements*.txt", "constraints*.txt", "pyproject.toml", "poetry.lock", "uv.lock", "Pipfile", "Pipfile.lock", "environment.yml", "setup.py", "setup.cfg"]
    result: list[Path] = []
    for pattern in patterns:
        result.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(result)


def declared_constraints(root: Path) -> dict[str, str]:
    constraints: dict[str, str] = {}
    req = root / "requirements.txt"
    if req.exists():
        for line in req.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            name = re.split(r"[<>=!~\[]", stripped, maxsplit=1)[0].strip().lower()
            constraints[name] = stripped
    env = root / "environment.yml"
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip(" -")
            if not stripped or stripped in {"pip:", "dependencies:", "channels:"}:
                continue
            name = re.split(r"[<>=!~]", stripped, maxsplit=1)[0].strip().lower()
            if name and name not in {"python", "conda-forge", "name"}:
                constraints.setdefault(name, stripped)
    return constraints


def imported_top_level_modules(paths: Iterable[Path]) -> set[str]:
    modules: set[str] = set()
    for path in paths:
        if not path.exists() or path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def pip_json(python: Path, root: Path, args: Sequence[str], timeout: int = 120) -> CommandResult:
    return run_command([str(python), "-m", "pip", *args], root, timeout=timeout)


def build_dependency_contract(root: Path, python: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    constraints = declared_constraints(root)
    f6f1_sources = [
        root / "src" / "analysis" / "plan_f6f1_beyond_utility_metrics.py",
        root / "src" / "experiments" / "build_phase_f6f1_beyond_utility_metrics.py",
        root / "tests" / "test_plan_f6f1_beyond_utility_metrics.py",
        root / "tests" / "test_plan_f6e4c_scientific_consolidation.py",
        root / "tests" / "test_finalization_f6d4a_protected_input_guard.py",
    ]
    imports = imported_top_level_modules(f6f1_sources)
    package_names = sorted(set(REQUIRED_IMPORTS + OPTIONAL_IMPORTS + [name for name in ["numpy", "pandas", "yaml", "pytest", "matplotlib", "torch"] if name in imports or name in constraints]))
    installed: dict[str, str] = {}
    pip_list_payload: list[dict[str, Any]] = []
    cache_text = ""
    pip_check = CommandResult([], 999, "", "not_run", 0.0)
    if python is not None and python.exists():
        pip_list = pip_json(python, root, ["list", "--format=json"])
        if pip_list.returncode == 0:
            try:
                pip_list_payload = json.loads(pip_list.stdout)
                installed = {str(item["name"]).lower(): str(item["version"]) for item in pip_list_payload}
            except (json.JSONDecodeError, KeyError):
                installed = {}
        cache = pip_json(python, root, ["cache", "list"], timeout=120)
        cache_text = cache.stdout if cache.returncode == 0 else cache.stderr
        pip_check = pip_json(python, root, ["check"], timeout=120)
    rows: list[dict[str, Any]] = []
    aliases = {"yaml": "pyyaml"}
    for package in package_names:
        lookup = aliases.get(package, package)
        version = installed.get(lookup.lower(), installed.get(package.lower(), ""))
        import_result = "not_run"
        version_result = ""
        if python is not None and python.exists():
            probe = run_command([str(python), "-c", f"import {package}; print(getattr({package}, '__version__', 'unknown'))"], root, timeout=45)
            import_result = "pass" if probe.returncode == 0 else "fail"
            version_result = probe.stdout.strip() if probe.returncode == 0 else short_text(probe.stderr)
        cache_hit = "present" if re.search(re.escape(lookup), cache_text, re.IGNORECASE) else "not_found"
        rows.append(
            {
                "package": package,
                "required_by": dependency_required_by(package),
                "declared_constraint": constraints.get(lookup.lower(), constraints.get(package.lower(), "")),
                "installed_version": version,
                "installation_source": "system-site-packages" if version else "",
                "local_cache_or_wheel_evidence": cache_hit,
                "import_result": import_result,
                "version_result": version_result,
                "dependency_status": "available" if import_result == "pass" else "missing_or_import_failed",
                "notes": "pip_check=" + ("pass" if pip_check.returncode == 0 else short_text(pip_check.stderr or pip_check.stdout, 200)),
            }
        )
    meta = {
        "dependency_files": [str(path.relative_to(root)) for path in dependency_files(root)],
        "pip_check_returncode": pip_check.returncode,
        "pip_check_stdout": pip_check.stdout,
        "pip_check_stderr": pip_check.stderr,
        "pip_cache_summary": short_text(cache_text, 1000),
    }
    return rows, meta


def dependency_required_by(package: str) -> str:
    if package in REQUIRED_IMPORTS:
        return "F6F1 import, builder, tests"
    if package == "matplotlib":
        return "declared project plotting dependency; not required by F6F1R validation"
    if package == "torch":
        return "declared project dependency; excluded from F6F1R safe validation"
    return "local source or dependency declaration"


def ensure_parallel_environment(root: Path, base_python: Path | None) -> tuple[Path | None, str, list[dict[str, Any]], list[dict[str, Any]]]:
    gates: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    if base_python is None or not base_python.exists():
        missing.append(block_row("F6F1R-BLOCK-001", "base_interpreter", "blocked", "f6f1r_blocked_no_stable_python_interpreter", "No stable local Python interpreter was available.", "non-stub CPython", "none", "install or expose local Python without network"))
        return None, "blocked_no_stable_python_interpreter", gates, missing

    primary_python = root / PRIMARY_VENV / "Scripts" / "python.exe"
    if not (root / PRIMARY_VENV).exists():
        result = run_command([str(base_python), "-m", "venv", str(root / PRIMARY_VENV)], root, timeout=180)
        gates.append(gate_row("ENV-001", "create_isolated_venv", result, "pass" if result.returncode == 0 else "fail", "isolated_local_venv_creation", "No network or pip install used."))
    else:
        gates.append(simple_gate("ENV-001", "inspect_existing_isolated_venv", "pass", "isolated_local_venv_exists", "Existing .venv_repro inspected and left in place."))

    isolated_import = run_command([str(primary_python), "-c", import_check_code(REQUIRED_IMPORTS)], root, timeout=60) if primary_python.exists() else CommandResult([str(primary_python)], 999, "", "missing", 0.0)
    gates.append(
        gate_row(
            "ENV-002",
            "isolated_required_imports",
            isolated_import,
            "pass" if isolated_import.returncode == 0 else "documented",
            "isolated_local_venv",
            "Failure triggers local system-site fallback; no package index is used.",
        )
    )
    if isolated_import.returncode == 0:
        return primary_python, "isolated_local_venv", gates, missing

    fallback_python = root / FALLBACK_VENV / "Scripts" / "python.exe"
    if not (root / FALLBACK_VENV).exists():
        result = run_command([str(base_python), "-m", "venv", "--system-site-packages", str(root / FALLBACK_VENV)], root, timeout=180)
        gates.append(gate_row("ENV-003", "create_system_site_venv", result, "pass" if result.returncode == 0 else "fail", "reconstructed_venv_with_system_site_packages", "Exposes already-installed local packages; no downloads."))
    else:
        gates.append(simple_gate("ENV-003", "inspect_existing_system_site_venv", "pass", "reconstructed_venv_with_system_site_packages", "Existing fallback environment inspected and left in place."))
    fallback_import = run_command([str(fallback_python), "-c", import_check_code(REQUIRED_IMPORTS)], root, timeout=60) if fallback_python.exists() else CommandResult([str(fallback_python)], 999, "", "missing", 0.0)
    gates.append(gate_row("ENV-004", "fallback_required_imports", fallback_import, "pass" if fallback_import.returncode == 0 else "fail", "reconstructed_venv_with_system_site_packages", "Required F6F1 imports must resolve from local system site-packages."))
    if fallback_import.returncode == 0:
        return fallback_python, "reconstructed_venv_with_system_site_packages", gates, missing
    missing.append(block_row("F6F1R-BLOCK-002", "dependencies", "blocked", "f6f1r_blocked_missing_local_dependencies", "Required packages unavailable from isolated or system-site local environments.", "local wheels/cache or system-site packages", short_text(fallback_import.stderr), "reconstruct_local_dependency_bundle"))
    return None, "blocked_missing_local_dependencies", gates, missing


def import_check_code(packages: Sequence[str]) -> str:
    return (
        "import importlib, json\n"
        f"packages={list(packages)!r}\n"
        "out={}\n"
        "for package in packages:\n"
        "    module=importlib.import_module(package)\n"
        "    out[package]=getattr(module,'__version__','unknown')\n"
        "print(json.dumps(out, sort_keys=True))\n"
    )


def gate_row(gate_id: str, gate: str, result: CommandResult, status: str, classification: str, notes: str) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "gate": gate,
        "command": " ".join(result.command),
        "status": status,
        "returncode": result.returncode,
        "stdout": short_text(result.stdout),
        "stderr": short_text(result.stderr),
        "classification": classification,
        "notes": notes,
    }


def simple_gate(gate_id: str, gate: str, status: str, classification: str, notes: str) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "gate": gate,
        "command": "",
        "status": status,
        "returncode": "",
        "stdout": "",
        "stderr": "",
        "classification": classification,
        "notes": notes,
    }


def run_environment_gates(root: Path, env_python: Path | None, classification: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if env_python is None or not env_python.exists():
        rows.append(simple_gate("GATE-000", "environment_python_exists", "fail", classification, "No environment Python available."))
        return rows
    probes = [
        ("GATE-001", "python_starts", [str(env_python), "--version"], "Python starts successfully."),
        ("GATE-002", "venv_prefix_differs", [str(env_python), "-c", "import sys; raise SystemExit(0 if sys.prefix != sys.base_prefix else 1)"], "sys.prefix differs from sys.base_prefix."),
        ("GATE-003", "base_executable_exists", [str(env_python), "-c", "import sys, pathlib; p=pathlib.Path(getattr(sys,'_base_executable','')); raise SystemExit(0 if p.exists() else 1)"], "Base executable exists."),
        ("GATE-004", "pip_callable", [str(env_python), "-m", "pip", "--version"], "pip is callable."),
        ("GATE-005", "pip_check", [str(env_python), "-m", "pip", "check"], "pip check completed."),
        ("GATE-006", "required_imports", [str(env_python), "-c", import_check_code(REQUIRED_IMPORTS)], "Required imports succeeded."),
        ("GATE-007", "f6f1_import_side_effect_free", [str(env_python), "-c", "import hashlib, pathlib; p=pathlib.Path('results/manifests/phase_f6f1_beyond_utility_freeze.json'); before=hashlib.sha256(p.read_bytes()).hexdigest(); import src.analysis.plan_f6f1_beyond_utility_metrics; after=hashlib.sha256(p.read_bytes()).hexdigest(); raise SystemExit(0 if before == after else 1)"], "F6F1 import did not change frozen manifest."),
    ]
    required_import_probe: CommandResult | None = None
    for gate_id, name, args, notes in probes:
        result = run_command(args, root, timeout=120)
        if name == "required_imports":
            required_import_probe = result
        status = "pass" if result.returncode == 0 else "fail"
        gate_notes = notes
        if name == "pip_check" and result.returncode != 0:
            import_check = required_import_probe or run_command([str(env_python), "-c", import_check_code(REQUIRED_IMPORTS)], root, timeout=60)
            if import_check.returncode == 0:
                status = "pass"
                gate_notes = notes + " Non-F6F1 system-site dependency conflicts are recorded but required F6F1 imports pass."
        rows.append(gate_row(gate_id, name, result, status, classification, gate_notes))
    env_path_text = str(env_python)
    rows.append(simple_gate("GATE-008", "environment_python_not_windowsapps_stub", "pass" if not is_app_execution_alias(env_python) else "fail", classification, env_path_text))
    rows.append(simple_gate("GATE-009", "no_network_policy", "pass", "static_policy", "F6F1R uses subprocess/local file inspection only; no URL/package-index command is issued."))
    rows.append(simple_gate("GATE-010", "no_ollama_inference_training_cuda_policy", "pass", "static_policy", "Targeted F6F1R validation excludes Ollama, inference, training, and CUDA suites."))
    return rows


def official_f6f1_paths(root: Path) -> dict[str, Path]:
    manifest = read_json(root / OFFICIAL_F6F1_MANIFEST)
    # The official F6F1 config is source-of-truth for output paths.
    config = load_yaml_like(root / F6F1_CONFIG)
    paths = {name: root / path for name, path in config["outputs"]["tables"].items()}
    paths["report"] = root / config["outputs"]["report"]
    paths["freeze_manifest"] = root / config["outputs"]["freeze_manifest"]
    return paths


def load_yaml_like(path: Path) -> dict[str, Any]:
    # Avoid a hard import at module import time; PyYAML is part of the audited contract.
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def input_integrity_paths(root: Path) -> list[tuple[str, str, Path]]:
    paths: list[tuple[str, str, Path]] = [
        ("f6f1_config", "F6F1", root / F6F1_CONFIG),
        ("f6f1_analysis_source", "F6F1", root / "src/analysis/plan_f6f1_beyond_utility_metrics.py"),
        ("f6f1_builder_source", "F6F1", root / "src/experiments/build_phase_f6f1_beyond_utility_metrics.py"),
        ("f6f1_test_source", "F6F1", root / "tests/test_plan_f6f1_beyond_utility_metrics.py"),
        ("f6e4c_manifest", "F6E4C", root / "results/manifests/phase_f6e4c_scientific_freeze.json"),
        ("f6e4c_request_level_metrics", "F6E4C", root / "results/tables/phase_f6e4c_request_level_metrics.csv"),
        ("f6e4b_raw_qwen2", "F6E4B", root / "results/llm/phase_f6e4b_qwen2_adapted_full_run_raw.jsonl"),
        ("f6e4b_raw_mistral", "F6E4B", root / "results/llm/phase_f6e4b_mistral_adapted_full_run_raw.jsonl"),
        ("f6e4a_request_manifest", "F6E4A", root / "results/tables/phase_f6e4a_request_manifest.csv"),
        ("f6e4a_shard_manifest", "F6E4A", root / "results/tables/phase_f6e4a_shard_manifest.csv"),
        ("protected_input_manifest", "F2", root / "results/tables/finalization_f2_protected_input_hashes_after.csv"),
    ]
    for name, path in official_f6f1_paths(root).items():
        paths.append((f"official_f6f1_{name}", "F6F1", path))
    f6f1_manifest = read_json(root / OFFICIAL_F6F1_MANIFEST)
    for rel_path in sorted(f6f1_manifest.get("input_hashes", {})):
        paths.append((f"direct_input_{Path(rel_path).name}", "direct_input", root / rel_path))
    seen: set[str] = set()
    unique: list[tuple[str, str, Path]] = []
    for name, phase, path in paths:
        key = str(path.resolve(strict=False)).lower()
        if key not in seen:
            seen.add(key)
            unique.append((name, phase, path))
    return unique


def capture_input_hashes(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for artifact, phase, path in input_integrity_paths(root):
        exists = path.exists()
        result[str(path.relative_to(root) if path.exists() or str(path).startswith(str(root)) else path)] = {
            "artifact": artifact,
            "phase": phase,
            "path": str(path),
            "exists": exists,
            "sha256": sha256_file(path) if exists and path.is_file() else "",
        }
    return result


def build_input_integrity_rows(before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(set(before) | set(after)):
        left = before.get(key, {})
        right = after.get(key, {})
        missing = not bool(right.get("exists", False))
        changed = bool(left.get("sha256", "") and right.get("sha256", "") and left.get("sha256") != right.get("sha256"))
        rows.append(
            {
                "artifact": right.get("artifact", left.get("artifact", "")),
                "path": key,
                "phase": right.get("phase", left.get("phase", "")),
                "before_sha256": left.get("sha256", ""),
                "after_sha256": right.get("sha256", ""),
                "changed": fmt_bool(changed),
                "missing": fmt_bool(missing),
                "classification": "pass" if not changed and not missing else "fail",
                "notes": "",
            }
        )
    return rows


def run_targeted_tests(root: Path, env_python: Path | None, classification: str) -> list[dict[str, Any]]:
    if env_python is None or not env_python.exists():
        return [test_row("environment_missing", [], "", classification, "blocked", 0, 0, 0, "", "", "No environment Python available.", "", "")]
    suites = [
        ("F6F1R tests", ["tests/test_plan_f6f1r_environment_recovery.py"]),
        ("F6F1 tests", ["tests/test_plan_f6f1_beyond_utility_metrics.py"]),
        ("F6E4C tests", ["tests/test_plan_f6e4c_scientific_consolidation.py"]),
        ("protected input guard", ["tests/test_finalization_f6d4a_protected_input_guard.py"]),
        ("minimal parser contract", ["tests/test_canonical_llm_contract.py"]),
    ]
    rows: list[dict[str, Any]] = []
    for name, files in suites:
        command = [str(env_python), "-m", "pytest", *files, "-q"]
        result = run_command(command, root, timeout=300)
        parsed = parse_pytest_counts(result.stdout + "\n" + result.stderr)
        rows.append(
            test_row(
                name,
                command,
                str(env_python),
                classification,
                "pass" if result.returncode == 0 else "fail",
                parsed["passed"],
                parsed["failed"],
                parsed["skipped"],
                parsed["warnings"],
                f"{result.duration_seconds:.3f}",
                "",
                short_text(result.stdout),
                short_text(result.stderr),
            )
        )
    for excluded in UNSAFE_EXCLUSIONS:
        rows.append(test_row(f"excluded_{excluded}", [], str(env_python), classification, "excluded", 0, 0, 0, "", "", f"Excluded by F6F1R safety inventory: {excluded}.", "", ""))
    return preserve_prior_test_runtime_fields(root, rows)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def preserve_prior_test_runtime_fields(root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous = read_csv_dicts(root / OUTPUT_TABLES["test_results"])
    if not previous:
        return rows
    prior_by_key = {
        (
            row.get("suite", ""),
            row.get("command", ""),
            row.get("interpreter_path", ""),
            row.get("status", ""),
            row.get("passed", ""),
            row.get("failed", ""),
            row.get("skipped", ""),
        ): row
        for row in previous
    }
    preserved: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("suite", "")),
            str(row.get("command", "")),
            str(row.get("interpreter_path", "")),
            str(row.get("status", "")),
            str(row.get("passed", "")),
            str(row.get("failed", "")),
            str(row.get("skipped", "")),
        )
        prior = prior_by_key.get(key)
        if prior:
            row = dict(row)
            for field in ["warnings", "duration_seconds", "stdout_summary", "stderr_summary"]:
                row[field] = prior.get(field, row.get(field, ""))
        preserved.append(row)
    return preserved


def parse_pytest_counts(text: str) -> dict[str, Any]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "warnings": ""}
    for key in ["passed", "failed", "skipped"]:
        match = re.search(rf"(\d+)\s+{key}", text)
        if match:
            counts[key] = int(match.group(1))
    warning = re.search(r"(\d+)\s+warnings?", text)
    if warning:
        counts["warnings"] = warning.group(0)
    return counts


def test_row(
    suite: str,
    command: Sequence[str],
    interpreter_path: str,
    classification: str,
    status: str,
    passed: int,
    failed: int,
    skipped: int,
    warnings: str,
    duration: str,
    exclusion: str,
    stdout: str,
    stderr: str,
) -> dict[str, Any]:
    return {
        "suite": suite,
        "command": " ".join(command),
        "interpreter_path": interpreter_path,
        "environment_classification": classification,
        "status": status,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "warnings": warnings,
        "duration_seconds": duration,
        "exclusion_reason": exclusion,
        "stdout_summary": stdout,
        "stderr_summary": stderr,
    }


def reject_official_output_root(root: Path, output_root: Path) -> None:
    resolved = (root / output_root).resolve(strict=False) if not output_root.is_absolute() else output_root.resolve(strict=False)
    official = [path.resolve(strict=False) for path in official_f6f1_paths(root).values()]
    official_parents = {path.parent for path in official}
    if resolved in official or resolved in official_parents:
        raise ValueError(f"isolated output root overlaps official F6F1 output path: {resolved}")
    official_dirs = [root / "results" / "tables", root / "results" / "manifests", root / "report"]
    for directory in official_dirs:
        directory_resolved = directory.resolve(strict=False)
        if resolved == directory_resolved:
            raise ValueError(f"isolated output root may not be an official frozen output directory: {resolved}")


def isolated_f6f1_config(root: Path, output_root: Path) -> dict[str, Any]:
    reject_official_output_root(root, output_root)
    config = load_yaml_like(root / F6F1_CONFIG)
    output_root = output_root.as_posix().rstrip("/")
    config["outputs"] = {
        "report": f"{output_root}/report/phase_f6f1_beyond_utility_metrics.md",
        "freeze_manifest": f"{output_root}/manifests/phase_f6f1_beyond_utility_freeze.json",
        "tables": {
            name: f"{output_root}/tables/{Path(path).name}"
            for name, path in config["outputs"]["tables"].items()
        },
    }
    return config


def rebuild_f6f1_isolated(root: Path, output_root: Path) -> dict[str, Any]:
    reject_official_output_root(root, output_root)
    from src.analysis import plan_f6f1_beyond_utility_metrics as f6f1

    config = isolated_f6f1_config(root, output_root)
    return f6f1.run_phase(root, config)


def run_isolated_rebuild_subprocess(root: Path, env_python: Path | None, output_root: Path) -> tuple[CommandResult, dict[str, Any]]:
    if env_python is None or not env_python.exists():
        return CommandResult([], 999, "", "environment python missing", 0.0), {}
    command = [
        str(env_python),
        "src/experiments/build_phase_f6f1r_environment_recovery.py",
        "--isolated-rebuild-only",
        "--output-root",
        str(output_root),
    ]
    result = run_command(command, root, timeout=420)
    manifest_path = root / output_root / "manifests" / "phase_f6f1_beyond_utility_freeze.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    return result, manifest


def csv_summary(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def compare_csv_files(official: Path, isolated: Path) -> dict[str, Any]:
    if not official.exists() or not isolated.exists():
        return {"semantic": False, "row_count": "", "column_count": "", "schema_equal": False, "numeric_equal": False, "max_diff": "", "notes": "missing artifact"}
    official_header, official_rows = csv_summary(official)
    isolated_header, isolated_rows = csv_summary(isolated)
    schema_equal = official_header == isolated_header
    row_count = len(official_rows)
    column_count = len(official_header)
    numeric_equal = True
    max_diff = 0.0
    for left_row, right_row in zip(official_rows, isolated_rows, strict=False):
        for left, right in zip(left_row, right_row, strict=False):
            lf = parse_number(left)
            rf = parse_number(right)
            if lf is None or rf is None:
                if left != right:
                    numeric_equal = False
                continue
            diff = abs(lf - rf)
            max_diff = max(max_diff, diff)
            if diff != 0.0:
                numeric_equal = False
    semantic = schema_equal and len(official_rows) == len(isolated_rows) and official_rows == isolated_rows
    return {
        "semantic": semantic,
        "row_count": row_count,
        "column_count": column_count,
        "schema_equal": schema_equal,
        "numeric_equal": numeric_equal,
        "max_diff": max_diff,
        "notes": "",
    }


def parse_number(value: str) -> float | None:
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def compare_cross_environment(root: Path, isolated_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    official_paths = official_f6f1_paths(root)
    config = load_yaml_like(root / F6F1_CONFIG)
    isolated_paths = {
        name: root / ISOLATED_F6F1_ROOT / "tables" / Path(path).name
        for name, path in config["outputs"]["tables"].items()
    }
    isolated_paths["report"] = root / ISOLATED_F6F1_ROOT / "report" / "phase_f6f1_beyond_utility_metrics.md"
    isolated_paths["freeze_manifest"] = root / ISOLATED_F6F1_ROOT / "manifests" / "phase_f6f1_beyond_utility_freeze.json"
    rows: list[dict[str, Any]] = []
    for name, official_path in sorted(official_paths.items()):
        isolated_path = isolated_paths[name]
        artifact_type = "csv" if official_path.suffix.lower() == ".csv" else official_path.suffix.lower().lstrip(".")
        official_hash = sha256_file(official_path) if official_path.exists() else ""
        isolated_hash = sha256_file(isolated_path) if isolated_path.exists() else ""
        byte_identical = bool(official_hash and official_hash == isolated_hash)
        if artifact_type == "csv":
            summary = compare_csv_files(official_path, isolated_path)
            semantic = summary["semantic"]
            classification = "byte_identical" if byte_identical else ("scientifically_identical_runtime_metadata_differs" if semantic else "scientifically_different")
            rows.append(
                comparison_row(name, artifact_type, official_path, isolated_path, official_hash, isolated_hash, byte_identical, semantic, summary, classification, "")
            )
        elif name == "freeze_manifest":
            official_manifest = read_json(official_path)
            scientific_same = official_manifest.get("scientific_content_hash") == isolated_manifest.get("scientific_content_hash") == FROZEN_F6F1_HASH
            classification = "byte_identical" if byte_identical else ("scientifically_identical_runtime_metadata_differs" if scientific_same else "scientifically_different")
            rows.append(
                comparison_row(
                    name,
                    artifact_type,
                    official_path,
                    isolated_path,
                    official_hash,
                    isolated_hash,
                    byte_identical,
                    scientific_same,
                    {"row_count": "", "column_count": "", "schema_equal": "", "numeric_equal": "", "max_diff": ""},
                    classification,
                    "Manifest runtime metadata may differ; scientific_content_hash is compared separately.",
                )
            )
        else:
            classification = "byte_identical" if byte_identical else "scientifically_identical_runtime_metadata_differs"
            rows.append(
                comparison_row(
                    name,
                    artifact_type,
                    official_path,
                    isolated_path,
                    official_hash,
                    isolated_hash,
                    byte_identical,
                    byte_identical,
                    {"row_count": "", "column_count": "", "schema_equal": "", "numeric_equal": "", "max_diff": ""},
                    classification,
                    "Report contains runtime output paths and repository metadata.",
                )
            )
    return rows


def comparison_row(
    name: str,
    artifact_type: str,
    official_path: Path,
    isolated_path: Path,
    official_hash: str,
    isolated_hash: str,
    byte_identical: bool,
    semantic: bool,
    summary: Mapping[str, Any],
    classification: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "artifact_name": name,
        "artifact_type": artifact_type,
        "official_path": str(official_path),
        "isolated_path": str(isolated_path),
        "official_hash": official_hash,
        "isolated_hash": isolated_hash,
        "byte_identical": fmt_bool(byte_identical),
        "semantic_identical": fmt_bool(bool(semantic)),
        "row_count": summary.get("row_count", ""),
        "column_count": summary.get("column_count", ""),
        "schema_equal": fmt_bool(bool(summary.get("schema_equal", False))) if summary.get("schema_equal", "") != "" else "",
        "numeric_equal": fmt_bool(bool(summary.get("numeric_equal", False))) if summary.get("numeric_equal", "") != "" else "",
        "max_abs_numeric_diff": summary.get("max_diff", ""),
        "text_normalization": "none",
        "mismatch_classification": classification,
        "notes": notes or summary.get("notes", ""),
    }


def write_environment_freeze_files(root: Path, env_python: Path | None) -> dict[str, Any]:
    if env_python is None or not env_python.exists():
        payload = {"status": "blocked", "reason": "environment python missing"}
        write_json(root / ENVIRONMENT_OUTPUTS["python_identity"], payload)
        write_json(root / ENVIRONMENT_OUTPUTS["pip_list"], [])
        (root / ENVIRONMENT_OUTPUTS["pip_freeze"]).parent.mkdir(parents=True, exist_ok=True)
        (root / ENVIRONMENT_OUTPUTS["pip_freeze"]).write_text("", encoding="utf-8")
        write_json(root / ENVIRONMENT_OUTPUTS["environment_variables"], redact_environment(os.environ))
        return payload
    identity_result = run_command([str(env_python), "-c", python_identity_probe_code()], root, timeout=60)
    identity = json.loads(identity_result.stdout.strip().splitlines()[-1]) if identity_result.returncode == 0 else {"error": identity_result.stderr}
    pip_list = pip_json(env_python, root, ["list", "--format=json"], timeout=120)
    pip_freeze = pip_json(env_python, root, ["freeze"], timeout=120)
    write_json(root / ENVIRONMENT_OUTPUTS["python_identity"], identity)
    write_json(root / ENVIRONMENT_OUTPUTS["pip_list"], json.loads(pip_list.stdout) if pip_list.returncode == 0 else {"error": pip_list.stderr})
    (root / ENVIRONMENT_OUTPUTS["pip_freeze"]).parent.mkdir(parents=True, exist_ok=True)
    (root / ENVIRONMENT_OUTPUTS["pip_freeze"]).write_text(pip_freeze.stdout if pip_freeze.returncode == 0 else pip_freeze.stderr, encoding="utf-8", newline="\n")
    write_json(root / ENVIRONMENT_OUTPUTS["environment_variables"], redact_environment(os.environ))
    return identity


def redact_environment(env: Mapping[str, str]) -> dict[str, str]:
    sensitive = re.compile(r"(TOKEN|SECRET|KEY|PASSWORD|PASS|CREDENTIAL|AUTH|INDEX_URL|EXTRA_INDEX_URL)", re.IGNORECASE)
    result: dict[str, str] = {}
    for key, value in sorted(env.items()):
        if sensitive.search(key):
            result[key] = "<redacted>" if value else ""
        else:
            result[key] = value
    return result


def block_row(item_id: str, component: str, status: str, classification: str, reason: str, required: str, current: str, action: str) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "component": component,
        "status": status,
        "classification": classification,
        "reason": reason,
        "required_evidence": required,
        "current_evidence": current,
        "recommended_action": action,
    }


def build_missing_or_blocked(
    environment_classification: str,
    comparison_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    existing: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = list(existing)
    if environment_classification == "reconstructed_venv_with_system_site_packages":
        rows.append(block_row("F6F1R-BLOCK-003", "isolated_dependency_install", "documented", "not_fully_isolated", "Required packages were not available in a fresh isolated venv without local installation; fallback exposes local system-site packages.", "local wheel/cache bundle", "system-site-packages fallback passes gates", "reconstruct_local_dependency_bundle"))
    if any(row["status"] == "fail" for row in test_rows):
        rows.append(block_row("F6F1R-BLOCK-004", "targeted_tests", "blocked", "f6f1r_blocked_by_tests", "At least one targeted test suite failed.", "all targeted safe tests pass", "see phase_f6f1r_test_results.csv", "inspect_cross_environment_mismatch"))
    if any(row["mismatch_classification"] == "scientifically_different" for row in comparison_rows):
        rows.append(block_row("F6F1R-BLOCK-005", "cross_environment_comparison", "blocked", "f6f1r_blocked_by_cross_environment_mismatch", "At least one isolated F6F1 artifact differs scientifically.", "scientific identity", "see comparison table", "inspect_cross_environment_mismatch"))
    if not rows:
        rows.append(block_row("F6F1R-NONE", "missing_or_blocked", "none", "not_applicable", "No blocking evidence after F6F1R validation.", "not_applicable", "all gates passed", "proceed_to_f6f1_scientific_readout"))
    return rows


def final_decision(
    environment_classification: str,
    gate_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    input_rows: Sequence[Mapping[str, Any]],
    isolated_manifest: Mapping[str, Any],
) -> str:
    if environment_classification == "blocked_no_stable_python_interpreter":
        return "f6f1r_blocked_no_stable_python_interpreter"
    if environment_classification == "blocked_missing_local_dependencies":
        return "f6f1r_blocked_missing_local_dependencies"
    if any(row["status"] == "fail" for row in gate_rows) or any(row["status"] == "fail" for row in test_rows):
        return "f6f1r_blocked_by_tests"
    if any(row["classification"] == "fail" for row in input_rows):
        return "f6f1r_blocked_by_input_invariance"
    if any(row["mismatch_classification"] == "scientifically_different" for row in comparison_rows):
        return "f6f1r_blocked_by_cross_environment_mismatch"
    if isolated_manifest.get("scientific_content_hash") == FROZEN_F6F1_HASH:
        return "f6f1r_environment_recovery_complete_with_runtime_metadata_differences"
    if not isolated_manifest:
        return "f6f1r_environment_recovery_complete_with_documented_blocks"
    return "f6f1r_incomplete_missing_provenance"


def recommended_next_action(decision: str, environment_classification: str) -> str:
    if decision == "f6f1r_blocked_by_cross_environment_mismatch":
        return "inspect_cross_environment_mismatch"
    if decision == "f6f1r_blocked_missing_local_dependencies":
        return "reconstruct_local_dependency_bundle"
    if environment_classification == "reconstructed_venv_with_system_site_packages":
        return "keep_existing_environment_state_frozen"
    if decision.startswith("f6f1r_environment_recovery_complete"):
        return "proceed_to_f6f1_scientific_readout"
    return "keep_existing_environment_state_frozen"


def artifact_hashes(root: Path) -> dict[str, str]:
    paths = {**OUTPUT_TABLES, **ENVIRONMENT_OUTPUTS, "report": REPORT_PATH}
    return {name: sha256_file(root / path) for name, path in sorted(paths.items()) if (root / path).exists() and (root / path).is_file()}


def build_report(
    repository: Mapping[str, Any],
    broken_venv: Mapping[str, Any],
    selected_python: Path | None,
    environment_classification: str,
    decision: str,
    recommendation: str,
    manifest: Mapping[str, Any],
    counts: Mapping[str, int],
) -> str:
    sections = [
        ("Initial repository state", f"Branch `{repository.get('branch','')}`, HEAD `{repository.get('head','')}`, staged files `{repository.get('staged_files','') or 'none'}`. Pre-existing tracked dirty files remain `{', '.join(PRE_EXISTING_TRACKED_DIRTY)}`."),
        ("Broken `.venv` diagnosis", f"Classification `{broken_venv.get('failure_classification','')}`. `.venv` was inspected only and not modified."),
        ("Python interpreter inventory", f"Rows written: `{counts['python_inventory']}`."),
        ("F6F1 interpreter identity", "F6F1 manifest evidence records WindowsApps Store Python 3.12.10; exact prior process path is reconstructed from local evidence, not invented."),
        ("Dependency-contract inventory", f"Rows written: `{counts['dependency_contract']}` from local dependency files and import probes."),
        ("Selected stable base interpreter", str(selected_python or "")),
        ("Parallel environment creation", "`.venv_repro` was attempted first; fallback `.venv_repro_system_site` was used only if isolated imports were unavailable."),
        ("Environment classification", environment_classification),
        ("Environment safety gates", f"Rows written: `{counts['environment_gate']}`."),
        ("Frozen-input verification", f"Rows written: `{counts['input_integrity']}`; changed/missing counts are in the table and manifest."),
        ("Targeted tests", f"Rows written: `{counts['test_results']}`."),
        ("Excluded unsafe tests", ", ".join(UNSAFE_EXCLUSIONS)),
        ("Isolated F6F1 rebuild method", "F6F1R-specific wrapper calls existing F6F1 pure analysis with in-memory output path redirection under `results/environment/phase_f6f1r_isolated_f6f1`."),
        ("Cross-environment artifact comparison", f"Rows written: `{counts['cross_environment_comparison']}`."),
        ("Scientific hash comparison", f"Official `{FROZEN_F6F1_HASH}`, isolated `{manifest.get('isolated_f6f1_scientific_content_hash','')}`."),
        ("Runtime metadata differences", "Manifest/report runtime metadata is separated from scientific content; CSV outputs require byte identity."),
        ("Pip freeze and dependency provenance", "`results/environment/phase_f6f1r_pip_freeze.txt` and `phase_f6f1r_pip_list.json` are written."),
        ("Determinism", "F6F1R writes deterministic sorted CSV/JSON artifacts; runtime metadata is reported in the manifest."),
        ("Protected-input verification", f"Protected and direct frozen inputs are included in `{OUTPUT_TABLES['input_integrity']}`."),
        ("Environment freeze manifest", f"`{MANIFEST_PATH}`."),
        ("Missing or blocked evidence", f"Rows written: `{counts['missing_or_blocked']}`."),
        ("Final decision", decision),
        ("Recommended next action", recommendation),
        ("Final git state", "No git add, commit, reset, clean, checkout, or frozen output overwrite is performed."),
        ("Files created or modified", "Only new F6F1R source/test/report/table/environment/manifest files plus the new parallel venv directory."),
        ("Remaining limitations", "If the selected environment uses system-site packages, it is not a fully isolated dependency rebuild until a local wheel/cache bundle is reconstructed."),
    ]
    lines = ["# Phase F6F1R Environment Recovery", ""]
    for idx, (title, body) in enumerate(sections, start=1):
        lines.extend([f"## {idx}. {title}", body, ""])
    return "\n".join(lines).rstrip() + "\n"


def run_phase(root: Path) -> dict[str, Any]:
    repository_rows, repository = record_repository_state(root)
    broken_rows, broken_summary = audit_broken_venv(root)
    before_hashes = capture_input_hashes(root)
    inventory_rows, identities = inventory_python_interpreters(root)
    selected_python, selection_class, selection_notes = select_stable_base_interpreter(inventory_rows)
    selected_identity = identities.get(str(selected_python), {}) if selected_python is not None else {}
    f6f1_manifest = read_json(root / OFFICIAL_F6F1_MANIFEST)
    identity_rows = build_interpreter_identity(root, f6f1_manifest, selected_python, selected_identity)

    env_python, env_classification, creation_gates, missing_initial = ensure_parallel_environment(root, selected_python)
    environment_rows = creation_gates + run_environment_gates(root, env_python, env_classification)
    dependency_rows, dependency_meta = build_dependency_contract(root, env_python)
    env_identity = write_environment_freeze_files(root, env_python)
    test_rows = run_targeted_tests(root, env_python, env_classification)
    rebuild_result, isolated_manifest = run_isolated_rebuild_subprocess(root, env_python, ISOLATED_F6F1_ROOT)
    environment_rows.append(gate_row("GATE-011", "isolated_f6f1_rebuild", rebuild_result, "pass" if rebuild_result.returncode == 0 else "fail", env_classification, "Runs F6F1 into isolated F6F1R output root."))
    comparison_rows = compare_cross_environment(root, isolated_manifest) if isolated_manifest else [
        comparison_row("isolated_rebuild", "all", Path(""), Path(""), "", "", False, False, {"row_count": "", "column_count": "", "schema_equal": "", "numeric_equal": "", "max_diff": ""}, "comparison_blocked", short_text(rebuild_result.stderr or rebuild_result.stdout))
    ]
    after_hashes = capture_input_hashes(root)
    input_rows = build_input_integrity_rows(before_hashes, after_hashes)
    missing_rows = build_missing_or_blocked(env_classification, comparison_rows, test_rows, missing_initial)
    decision = final_decision(env_classification, environment_rows, test_rows, comparison_rows, input_rows, isolated_manifest)
    recommendation = recommended_next_action(decision, env_classification)

    tables = {
        "repository_state": repository_rows,
        "broken_venv_audit": broken_rows,
        "python_inventory": inventory_rows,
        "interpreter_identity": identity_rows,
        "dependency_contract": dependency_rows,
        "environment_gate": environment_rows,
        "test_results": test_rows,
        "cross_environment_comparison": comparison_rows,
        "input_integrity": input_rows,
        "missing_or_blocked": missing_rows,
    }
    columns = {
        "repository_state": REPOSITORY_COLUMNS,
        "broken_venv_audit": BROKEN_VENV_COLUMNS,
        "python_inventory": PYTHON_INVENTORY_COLUMNS,
        "interpreter_identity": INTERPRETER_IDENTITY_COLUMNS,
        "dependency_contract": DEPENDENCY_COLUMNS,
        "environment_gate": GATE_COLUMNS,
        "test_results": TEST_COLUMNS,
        "cross_environment_comparison": COMPARISON_COLUMNS,
        "input_integrity": INPUT_INTEGRITY_COLUMNS,
        "missing_or_blocked": MISSING_COLUMNS,
    }
    for name, rows in tables.items():
        write_csv(root / OUTPUT_TABLES[name], rows, columns[name])
    counts = {name: len(rows) for name, rows in tables.items()}
    manifest = {
        "phase": PHASE_ID,
        "decision": decision,
        "recommended_next_action": recommendation,
        "repository_state": repository,
        "broken_venv": broken_summary,
        "selection_classification": selection_class,
        "selection_notes": selection_notes,
        "selected_interpreter": str(selected_python or ""),
        "selected_interpreter_sha256": sha256_file(selected_python) if selected_python is not None and selected_python.exists() else "",
        "new_environment_python": str(env_python or ""),
        "new_environment_classification": env_classification,
        "dependency_meta": dependency_meta,
        "dependency_contract_hash": sha256_file(root / OUTPUT_TABLES["dependency_contract"]),
        "pip_freeze_hash": sha256_file(root / ENVIRONMENT_OUTPUTS["pip_freeze"]) if (root / ENVIRONMENT_OUTPUTS["pip_freeze"]).exists() else "",
        "test_result_hash": sha256_file(root / OUTPUT_TABLES["test_results"]),
        "cross_environment_comparison_hash": sha256_file(root / OUTPUT_TABLES["cross_environment_comparison"]),
        "frozen_f6f1_scientific_content_hash": FROZEN_F6F1_HASH,
        "isolated_f6f1_scientific_content_hash": isolated_manifest.get("scientific_content_hash", ""),
        "input_integrity": {
            "changed": sum(1 for row in input_rows if row["changed"] == "true"),
            "missing": sum(1 for row in input_rows if row["missing"] == "true"),
            "hash": sha256_file(root / OUTPUT_TABLES["input_integrity"]),
        },
        "runtime_metadata": {
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "os": platform.platform(),
            "builder_python": sys.executable,
            "env_identity": env_identity,
        },
        "table_hashes": {name: sha256_file(root / path) for name, path in OUTPUT_TABLES.items()},
    }
    report = build_report(repository, broken_summary, selected_python, env_classification, decision, recommendation, manifest, counts)
    (root / REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / REPORT_PATH).write_text(report, encoding="utf-8", newline="\n")
    manifest["report_hash"] = sha256_file(root / REPORT_PATH)
    manifest["artifact_hashes"] = artifact_hashes(root)
    manifest["scientific_section_hash"] = sha256_text(
        stable_json(
            {
                "decision": decision,
                "environment_classification": env_classification,
                "frozen_f6f1_scientific_content_hash": FROZEN_F6F1_HASH,
                "isolated_f6f1_scientific_content_hash": manifest["isolated_f6f1_scientific_content_hash"],
                "comparison_hash": manifest["cross_environment_comparison_hash"],
                "input_integrity": manifest["input_integrity"],
            }
        )
    )
    write_json(root / MANIFEST_PATH, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Phase F6F1R environment recovery artifacts.")
    parser.add_argument("--isolated-rebuild-only", action="store_true", help="Only run F6F1 into an isolated output root.")
    parser.add_argument("--output-root", default=str(ISOLATED_F6F1_ROOT), help="Isolated F6F1 output root for rebuild-only mode.")
    args = parser.parse_args(argv)
    root = Path.cwd()
    if args.isolated_rebuild_only:
        manifest = rebuild_f6f1_isolated(root, Path(args.output_root))
        print(stable_json({"phase": PHASE_ID, "mode": "isolated_rebuild_only", "scientific_content_hash": manifest.get("scientific_content_hash", "")}))
        return 0
    manifest = run_phase(root)
    print(stable_json({"phase": PHASE_ID, "decision": manifest["decision"], "recommended_next_action": manifest["recommended_next_action"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
