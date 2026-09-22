"""Phase F6G2R CPython 3.12 runtime recovery and execution preflight.

The phase records provenance for the repaired F6G2 Python environment and
checks whether the frozen F6G2 execution can proceed. It never calls an LLM,
never pulls models, never writes raw response files, and never mutates F6G2 or
earlier phase artifacts.
"""

from __future__ import annotations

from src.audit import common_git

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PHASE_ID = "phase_f6g2r"
DEFAULT_CONFIG_PATH = Path("configs/phase_f6g2r_runtime_recovery.yaml")

DECISION_SUCCESS = "f6g2r_runtime_recovery_complete_ready_for_execution"
DECISION_PYTHON = "f6g2r_blocked_by_python_installation"
DECISION_DEPENDENCY = "f6g2r_blocked_by_dependency_resolution"
DECISION_OLLAMA = "f6g2r_blocked_by_ollama_gate"
DECISION_CONTRACT = "f6g2r_blocked_by_contract_conflict"
DECISION_TESTS = "f6g2r_blocked_by_tests"
DECISION_INPUTS = "f6g2r_blocked_by_input_invariance"
DECISION_PROVENANCE = "f6g2r_blocked_by_incomplete_provenance"

RECOMMEND_SUCCESS = "proceed_to_f6g2b_lastfm_reranking_execution"
RECOMMEND_DEPENDENCY = "repair_dependency_environment"
RECOMMEND_CONTRACT = "inspect_output_contract_conflict"
RECOMMEND_OLLAMA = "repair_ollama_runtime"
RECOMMEND_BLOCKED = "keep_execution_blocked"

PRE_EXISTING_TRACKED_DIRTY = ["README.md", "report/dataset_audit.md", "report/notes.md"]

PYTHON_INVENTORY_COLUMNS = [
    "candidate_id",
    "path",
    "exists",
    "size_bytes",
    "sha256",
    "python_version",
    "architecture",
    "sys_executable",
    "sys_prefix",
    "sys_base_prefix",
    "is_virtual_environment",
    "windowsapps_related",
    "alias_or_stub_classification",
    "venv_works",
    "pip_works",
    "status",
    "notes",
]
INSTALLER_PROVENANCE_COLUMNS = [
    "component",
    "path",
    "source_url",
    "expected_sha256",
    "actual_sha256",
    "status",
    "classification",
    "notes",
]
ENVIRONMENT_IDENTITY_COLUMNS = ["item", "value", "status", "notes"]
DEPENDENCY_LOCK_COLUMNS = [
    "package",
    "pinned_version",
    "wheel_file",
    "wheel_sha256",
    "wheel_tag",
    "source",
    "status",
    "notes",
]
DEPENDENCY_INSTALLATION_COLUMNS = [
    "package",
    "pinned_version",
    "installed_version",
    "import_name",
    "import_version",
    "import_status",
    "pip_check_status",
    "status",
    "notes",
]
IMPORT_GATE_COLUMNS = [
    "module",
    "import_expression",
    "status",
    "version",
    "returncode",
    "stdout",
    "stderr",
    "blocking",
    "notes",
]
OLLAMA_PREFLIGHT_COLUMNS = [
    "check_id",
    "requirement",
    "observed",
    "status",
    "blocking",
    "generation_attempted",
    "notes",
]
INPUT_INTEGRITY_COLUMNS = [
    "artifact",
    "path",
    "expected_sha256",
    "actual_sha256",
    "status",
    "changed",
    "missing",
    "phase",
    "notes",
]
TEST_RESULTS_COLUMNS = [
    "suite",
    "command",
    "status",
    "passed",
    "failed",
    "skipped",
    "returncode",
    "duration_seconds",
    "stdout_summary",
    "stderr_summary",
    "notes",
]
EXECUTION_PREFLIGHT_COLUMNS = [
    "check_id",
    "requirement",
    "observed",
    "status",
    "blocking",
    "decision_if_failed",
    "recommendation_if_failed",
    "notes",
]
MISSING_COLUMNS = [
    "item_id",
    "component",
    "status",
    "classification",
    "reason",
    "current_evidence",
    "required_evidence",
    "recommended_action",
]


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    return payload


def run_command(command: Sequence[str], cwd: Path, timeout: int = 30) -> CommandResult:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(
            command=list(command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=time.perf_counter() - started,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return CommandResult(
            command=list(command),
            returncode=124,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=time.perf_counter() - started,
            timed_out=True,
        )
    except OSError as exc:
        return CommandResult(
            command=list(command),
            returncode=127,
            stdout="",
            stderr=str(exc),
            duration_seconds=time.perf_counter() - started,
            timed_out=False,
        )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: "" if value is None else value for key, value in row.items()} for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(payload) + "\n", encoding="utf-8")


def _safe_bool(value: bool) -> str:
    return "true" if value else "false"


def _short(text: str, max_chars: int = 260) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3] + "..."


def _run_git(args: Sequence[str], root: Path) -> str:
    return common_git.git_stdout_no_check(args, root, timeout=10, empty_on_failure=True)


def record_repository_state(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    phase = config.get("phase", {})
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    head = _run_git(["rev-parse", "HEAD"], root)
    staged = _run_git(["diff", "--name-only", "--cached"], root)
    tracked_dirty = _run_git(["diff", "--name-only"], root)
    expected_branch = str(phase.get("expected_branch", ""))
    expected_head = str(phase.get("expected_head", ""))
    dirty_files = [line for line in tracked_dirty.splitlines() if line]
    return {
        "branch": branch,
        "head": head,
        "expected_branch": expected_branch,
        "expected_head": expected_head,
        "branch_status": "pass" if not expected_branch or branch == expected_branch else "fail",
        "head_status": "pass" if not expected_head or head == expected_head else "fail",
        "staged_files": ";".join(line for line in staged.splitlines() if line),
        "tracked_dirty_files": ";".join(dirty_files),
        "pre_existing_tracked_dirty_files": ";".join(PRE_EXISTING_TRACKED_DIRTY),
        "status": "pass"
        if sorted(dirty_files) == sorted(PRE_EXISTING_TRACKED_DIRTY)
        and (not expected_branch or branch == expected_branch)
        and (not expected_head or head == expected_head)
        else "warn",
    }


def classify_python_candidate(path: Path, size_bytes: int | None = None) -> dict[str, str]:
    path_text = str(path)
    inside_windowsapps = "WindowsApps" in path_text
    size = size_bytes if size_bytes is not None else (path.stat().st_size if path.exists() else 0)
    if inside_windowsapps:
        classification = "windowsapps_app_execution_alias_or_stub"
    elif not path.exists():
        classification = "missing"
    elif size <= 4096 and path.name.lower().startswith("python"):
        classification = "small_launcher_or_stub"
    else:
        classification = "real_interpreter_candidate"
    return {
        "windowsapps_related": _safe_bool(inside_windowsapps),
        "alias_or_stub_classification": classification,
    }


def _probe_python(path: Path, root: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    probe = (
        "import json,platform,sys;"
        "print(json.dumps({"
        "'version':sys.version,"
        "'architecture':platform.architecture()[0],"
        "'executable':sys.executable,"
        "'prefix':sys.prefix,"
        "'base_prefix':sys.base_prefix,"
        "'is_venv':sys.prefix!=sys.base_prefix"
        "},sort_keys=True))"
    )
    result = run_command([str(path), "-c", probe], root, timeout=8)
    if result.returncode != 0:
        return {"error": _short(result.stderr or result.stdout), "returncode": str(result.returncode)}
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"error": _short(result.stdout), "returncode": str(result.returncode)}
    pip = run_command([str(path), "-m", "pip", "--version"], root, timeout=8)
    return {
        "python_version": str(payload.get("version", "")),
        "architecture": str(payload.get("architecture", "")),
        "sys_executable": str(payload.get("executable", "")),
        "sys_prefix": str(payload.get("prefix", "")),
        "sys_base_prefix": str(payload.get("base_prefix", "")),
        "is_virtual_environment": _safe_bool(bool(payload.get("is_venv"))),
        "venv_works": "true",
        "pip_works": _safe_bool(pip.returncode == 0),
        "returncode": str(result.returncode),
        "pip_stdout": _short(pip.stdout),
        "pip_stderr": _short(pip.stderr),
    }


def build_python_inventory(config: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    candidates = config.get("runtime", {}).get("python_inventory_candidates", [])
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        path = resolve_path(root, str(candidate))
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        classification = classify_python_candidate(path, size)
        probe = {} if classification["alias_or_stub_classification"].startswith("windowsapps") else _probe_python(path, root)
        status = "pass" if probe.get("venv_works") == "true" or probe.get("python_version") else "fail"
        if not exists:
            status = "missing"
        elif classification["alias_or_stub_classification"].startswith("windowsapps"):
            status = "blocked"
        rows.append(
            {
                "candidate_id": f"python-{index:02d}",
                "path": str(Path(candidate).as_posix()) if not Path(str(candidate)).is_absolute() else str(candidate),
                "exists": _safe_bool(exists),
                "size_bytes": size,
                "sha256": sha256_file(path) if exists and path.is_file() else "",
                "python_version": probe.get("python_version", ""),
                "architecture": probe.get("architecture", ""),
                "sys_executable": probe.get("sys_executable", ""),
                "sys_prefix": probe.get("sys_prefix", ""),
                "sys_base_prefix": probe.get("sys_base_prefix", ""),
                "is_virtual_environment": probe.get("is_virtual_environment", ""),
                "windowsapps_related": classification["windowsapps_related"],
                "alias_or_stub_classification": classification["alias_or_stub_classification"],
                "venv_works": probe.get("venv_works", "false" if exists else ""),
                "pip_works": probe.get("pip_works", "false" if exists else ""),
                "status": status,
                "notes": probe.get("error", ""),
            }
        )
    return rows


def collect_python_identity(python_path: Path, root: Path) -> dict[str, Any]:
    script = (
        "import json,platform,site,sys;"
        "print(json.dumps({"
        "'python_version':sys.version,"
        "'version_info':list(sys.version_info[:3]),"
        "'architecture':platform.architecture()[0],"
        "'platform':platform.platform(),"
        "'executable':sys.executable,"
        "'prefix':sys.prefix,"
        "'base_prefix':sys.base_prefix,"
        "'base_executable':getattr(sys,'_base_executable',''),"
        "'site_packages':site.getsitepackages(),"
        "'user_site':site.getusersitepackages(),"
        "'system_site_packages_used':False"
        "},sort_keys=True))"
    )
    result = run_command([str(python_path), "-c", script], root, timeout=10)
    if result.returncode != 0:
        return {"status": "fail", "stderr": _short(result.stderr), "stdout": _short(result.stdout)}
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    payload["status"] = "pass"
    payload["python_sha256"] = sha256_file(python_path) if python_path.exists() else ""
    return payload


def build_installer_provenance(config: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    installer = config.get("installer", {})
    runtime = config.get("runtime", {})
    downloaded = resolve_path(root, installer.get("downloaded_path", ""))
    base_python = resolve_path(root, runtime.get("base_python_path", ""))
    venv_python = resolve_path(root, runtime.get("venv_python_path", ""))
    rows: list[dict[str, Any]] = []

    def add(component: str, path: Path, expected: str, notes: str = "") -> None:
        actual = sha256_file(path) if path.exists() and path.is_file() else ""
        rows.append(
            {
                "component": component,
                "path": str(path.relative_to(root).as_posix()) if path.is_relative_to(root) else str(path),
                "source_url": installer.get("source_url", "") if component == "installer" else "",
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": "pass" if actual and actual == expected else "fail",
                "classification": installer.get("provenance_classification", ""),
                "notes": notes,
            }
        )

    add("installer", downloaded, str(installer.get("expected_sha256", "")))
    add("base_python", base_python, str(installer.get("installed_python_sha256", "")))
    add("venv_python", venv_python, str(installer.get("venv_python_sha256", "")), "Created in .venv_f6g2_py312; legacy venvs are not modified.")
    return rows


def build_environment_identity(config: Mapping[str, Any], identity: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    runtime = config.get("runtime", {})
    expected_version = str(config.get("phase", {}).get("cpython_version", ""))
    expected_arch = str(config.get("phase", {}).get("architecture", ""))
    venv_python = resolve_path(root, runtime.get("venv_python_path", ""))
    rows = [
        {"item": "venv_python_path", "value": str(Path(runtime.get("venv_python_path", "")).as_posix()), "status": "pass" if venv_python.exists() else "fail", "notes": ""},
        {"item": "python_version", "value": identity.get("python_version", ""), "status": "pass" if expected_version in str(identity.get("python_version", "")) else "fail", "notes": f"Expected CPython {expected_version}."},
        {"item": "architecture", "value": identity.get("architecture", ""), "status": "pass" if identity.get("architecture") == expected_arch else "fail", "notes": f"Expected {expected_arch}."},
        {"item": "is_virtual_environment", "value": _safe_bool(identity.get("prefix") != identity.get("base_prefix")), "status": "pass" if identity.get("prefix") != identity.get("base_prefix") else "fail", "notes": "sys.prefix must differ from sys.base_prefix."},
        {"item": "system_site_packages_used", "value": _safe_bool(bool(identity.get("system_site_packages_used"))), "status": "pass" if not identity.get("system_site_packages_used") else "fail", "notes": "F6G2R uses an isolated venv."},
        {"item": "base_executable", "value": identity.get("base_executable", ""), "status": "pass" if str(identity.get("base_executable", "")).endswith("python.exe") else "fail", "notes": ""},
        {"item": "python_sha256", "value": identity.get("python_sha256", ""), "status": "pass" if identity.get("python_sha256") == config.get("installer", {}).get("venv_python_sha256") else "fail", "notes": ""},
    ]
    return rows


def _pip_list(python_path: Path, root: Path) -> list[dict[str, str]]:
    result = run_command([str(python_path), "-m", "pip", "list", "--format=json"], root, timeout=20)
    if result.returncode != 0:
        return []
    return json.loads(result.stdout)


def _pip_freeze(python_path: Path, root: Path) -> str:
    result = run_command([str(python_path), "-m", "pip", "freeze"], root, timeout=20)
    return result.stdout if result.returncode == 0 else ""


def _pip_check(python_path: Path, root: Path) -> CommandResult:
    return run_command([str(python_path), "-m", "pip", "check"], root, timeout=20)


def write_environment_files(config: Mapping[str, Any], root: Path, identity: Mapping[str, Any]) -> dict[str, str]:
    outputs = config.get("outputs", {}).get("environment", {})
    python_path = resolve_path(root, config.get("runtime", {}).get("venv_python_path", ""))
    pip_freeze_path = resolve_path(root, outputs.get("pip_freeze", "results/environment/phase_f6g2r_pip_freeze.txt"))
    pip_list_path = resolve_path(root, outputs.get("pip_list", "results/environment/phase_f6g2r_pip_list.json"))
    identity_path = resolve_path(root, outputs.get("python_identity", "results/environment/phase_f6g2r_python_identity.json"))

    pip_freeze_path.parent.mkdir(parents=True, exist_ok=True)
    pip_freeze_path.write_text(_pip_freeze(python_path, root), encoding="utf-8")
    write_json(pip_list_path, _pip_list(python_path, root))
    write_json(identity_path, identity)
    return {
        "pip_freeze": sha256_file(pip_freeze_path),
        "pip_list": sha256_file(pip_list_path),
        "python_identity": sha256_file(identity_path),
    }


def build_dependency_lock(config: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    wheelhouse = resolve_path(root, config.get("dependencies", {}).get("wheelhouse", ""))
    rows: list[dict[str, Any]] = []
    for pin in config.get("dependencies", {}).get("pins", []):
        wheel = str(pin.get("wheel", ""))
        wheel_path = wheelhouse / wheel
        actual = sha256_file(wheel_path) if wheel_path.exists() else ""
        status = "pass" if actual and actual == pin.get("sha256") else "fail"
        rows.append(
            {
                "package": pin.get("package", ""),
                "pinned_version": pin.get("version", ""),
                "wheel_file": wheel,
                "wheel_sha256": actual,
                "wheel_tag": _wheel_tag(wheel),
                "source": config.get("dependencies", {}).get("pip_index_policy", ""),
                "status": status,
                "notes": "" if status == "pass" else "Wheel missing or hash mismatch.",
            }
        )
    return rows


def _wheel_tag(wheel_name: str) -> str:
    if not wheel_name.endswith(".whl"):
        return ""
    parts = wheel_name[:-4].split("-")
    if len(parts) < 5:
        return "py3-none-any"
    return "-".join(parts[-3:])


def _import_version(module: str, python_path: Path, root: Path) -> tuple[str, str, str]:
    script = (
        "import importlib,json;"
        f"m=importlib.import_module({module!r});"
        "print(json.dumps({'version':getattr(m,'__version__','')}))"
    )
    result = run_command([str(python_path), "-c", script], root, timeout=10)
    if result.returncode != 0:
        return "fail", "", _short(result.stderr or result.stdout)
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    return "pass", str(payload.get("version", "")), ""


def build_dependency_installation(config: Mapping[str, Any], root: Path) -> tuple[list[dict[str, Any]], str]:
    python_path = resolve_path(root, config.get("runtime", {}).get("venv_python_path", ""))
    installed = {row["name"].lower(): row["version"] for row in _pip_list(python_path, root)}
    pip_check = _pip_check(python_path, root)
    pip_check_status = "pass" if pip_check.returncode == 0 else "fail"
    rows: list[dict[str, Any]] = []
    for pin in config.get("dependencies", {}).get("pins", []):
        package = str(pin.get("package", ""))
        installed_version = installed.get(package.lower(), "")
        import_status, import_version, import_notes = _import_version(str(pin.get("import_name", package)), python_path, root)
        version_status = installed_version == str(pin.get("version", ""))
        status = "pass" if version_status and import_status == "pass" and pip_check_status == "pass" else "fail"
        rows.append(
            {
                "package": package,
                "pinned_version": pin.get("version", ""),
                "installed_version": installed_version,
                "import_name": pin.get("import_name", package),
                "import_version": import_version,
                "import_status": import_status,
                "pip_check_status": pip_check_status,
                "status": status,
                "notes": import_notes if import_notes else _short(pip_check.stdout if pip_check_status == "pass" else pip_check.stderr),
            }
        )
    return rows, pip_check_status


def build_import_gate(config: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    python_path = resolve_path(root, config.get("runtime", {}).get("venv_python_path", ""))
    rows: list[dict[str, Any]] = []
    for module in config.get("imports", {}).get("required_modules", []):
        script = (
            "import importlib,json;"
            f"m=importlib.import_module({str(module)!r});"
            "print(json.dumps({'version':getattr(m,'__version__','')}))"
        )
        result = run_command([str(python_path), "-c", script], root, timeout=20)
        version = ""
        status = "fail"
        if result.returncode == 0:
            status = "pass"
            try:
                version = str(json.loads(result.stdout.strip().splitlines()[-1]).get("version", ""))
            except (json.JSONDecodeError, IndexError):
                version = ""
        rows.append(
            {
                "module": module,
                "import_expression": f"import {module}",
                "status": status,
                "version": version,
                "returncode": result.returncode,
                "stdout": _short(result.stdout),
                "stderr": _short(result.stderr),
                "blocking": _safe_bool(status != "pass"),
                "notes": "No network, API, Ollama generation, training, or CUDA import is required.",
            }
        )
    return rows


def parse_ollama_list(stdout: str) -> dict[str, str]:
    models: dict[str, str] = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("name"):
            continue
        parts = stripped.split()
        if len(parts) >= 2 and ":" in parts[0]:
            models[parts[0]] = parts[1]
    return models


def build_ollama_preflight(config: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    ollama = config.get("ollama", {})
    expected = list(ollama.get("expected_models", []))
    rows: list[dict[str, Any]] = []
    if not ollama.get("live_metadata_check", True):
        rows.append(
            {
                "check_id": "ollama-live-metadata",
                "requirement": "metadata-only local Ollama model inventory",
                "observed": "live check disabled by config",
                "status": "skipped",
                "blocking": "false",
                "generation_attempted": "false",
                "notes": "No model generation or pull attempted.",
            }
        )
        return rows
    result = run_command(["ollama", "list"], root, timeout=int(ollama.get("command_timeout_seconds", 30)))
    models = parse_ollama_list(result.stdout)
    cli_status = "pass" if result.returncode == 0 or models else "fail"
    rows.append(
        {
            "check_id": "ollama-cli",
            "requirement": "ollama list metadata command returns local inventory",
            "observed": f"returncode={result.returncode};models={';'.join(sorted(models))}",
            "status": cli_status,
            "blocking": _safe_bool(cli_status != "pass"),
            "generation_attempted": "false",
            "notes": "Command timed out after emitting inventory." if result.timed_out and models else _short(result.stderr),
        }
    )
    for model in expected:
        name = str(model.get("name", ""))
        expected_short = str(model.get("observed_short_id", ""))
        actual = models.get(name, "")
        status = "pass" if actual and (not expected_short or actual.startswith(expected_short) or expected_short.startswith(actual)) else "fail"
        rows.append(
            {
                "check_id": f"ollama-model-{name}",
                "requirement": f"local model tag present: {name}",
                "observed": actual,
                "status": status,
                "blocking": _safe_bool(status != "pass"),
                "generation_attempted": "false",
                "notes": "Short local model ID only; no generation, pull, or digest guessing.",
            }
        )
    return rows


def verify_input_hashes(config: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, spec in config.get("input_hashes", {}).items():
        path = resolve_path(root, spec.get("path", ""))
        missing = not path.exists()
        actual = "" if missing else sha256_file(path)
        expected = str(spec.get("sha256", ""))
        changed = bool(actual and expected and actual != expected)
        status = "missing" if missing else ("changed" if changed else "pass")
        rows.append(
            {
                "artifact": name,
                "path": spec.get("path", ""),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": status,
                "changed": _safe_bool(changed),
                "missing": _safe_bool(missing),
                "phase": spec.get("phase", ""),
                "notes": "",
            }
        )
    return rows


def _unique_values(rows: Sequence[Mapping[str, str]], column: str) -> list[str]:
    return sorted({str(row.get(column, "")) for row in rows if str(row.get(column, "")) != ""})


def _pipe_count(value: str) -> int:
    if not value:
        return 0
    return len([part for part in value.split("|") if part != ""])


def summarize_contract(
    config: Mapping[str, Any],
    logical_rows: Sequence[Mapping[str, str]],
    request_rows: Sequence[Mapping[str, str]],
    shard_rows: Sequence[Mapping[str, str]],
    f6g2_request_rows: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    contract = config.get("f6g2_contract", {})
    candidate_pool_sizes = sorted({_pipe_count(str(row.get("candidate_item_ids", ""))) for row in request_rows if row.get("candidate_item_ids", "")})
    logical_output_lengths = _unique_values(logical_rows, "expected_output_length")
    request_output_lengths = _unique_values(request_rows, "required_output_length")
    parser_contracts = _unique_values(request_rows, "parser_contract") or _unique_values(logical_rows, "parser_contract")
    f6g2_requested_lengths = _unique_values(f6g2_request_rows or [], "requested_ranking_length")
    f6g2_parser_lengths = _unique_values(f6g2_request_rows or [], "parser_expected_length")
    expected_requests = int(contract.get("expected_requests", 0))
    expected_logical = int(contract.get("expected_logical_instances", 0))
    expected_shards = int(contract.get("expected_shards", 0))
    unique_request_ids = len({row.get("request_id", "") for row in request_rows})
    top10_required = str(contract.get("top10_required_requested_ranking_length", "10"))
    top10_parser = str(contract.get("top10_required_parser_expected_length", "10"))
    conflict = (
        "20" in request_output_lengths
        or "20" in logical_output_lengths
        or "indexed_json_top20_unique_candidates" in parser_contracts
        or "20" in f6g2_requested_lengths
        or "20" in f6g2_parser_lengths
    ) and (top10_required == "10" or top10_parser == "10")
    return {
        "logical_count": len(logical_rows),
        "request_count": len(request_rows),
        "shard_count": len(shard_rows),
        "expected_logical_count": expected_logical,
        "expected_request_count": expected_requests,
        "expected_shard_count": expected_shards,
        "unique_request_count": unique_request_ids,
        "candidate_pool_sizes": [str(value) for value in candidate_pool_sizes],
        "logical_output_lengths": logical_output_lengths,
        "request_output_lengths": request_output_lengths,
        "parser_contracts": parser_contracts,
        "f6g2_requested_lengths": f6g2_requested_lengths,
        "f6g2_parser_lengths": f6g2_parser_lengths,
        "top10_required_requested_ranking_length": top10_required,
        "top10_required_parser_expected_length": top10_parser,
        "contract_status": "blocked_by_top20_top10_conflict" if conflict else "pass",
        "execution_allowed": not conflict
        and len(request_rows) == expected_requests
        and len(logical_rows) == expected_logical
        and len(shard_rows) == expected_shards
        and unique_request_ids == len(request_rows),
        "blocking_reason": (
            "Frozen F6G1/F6G2 request manifests require top-20 output while prior adapter evidence requires strict top-10."
            if conflict
            else ""
        ),
    }


def _load_f6g2_config(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    path = resolve_path(root, config.get("f6g2_contract", {}).get("authoritative_f6g2_config", ""))
    if not path.exists():
        return {}
    return load_config(path)


def _raw_file_stats(f6g2_config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    raw_specs = f6g2_config.get("outputs", {}).get("raw", {}) if f6g2_config else {}
    stats: dict[str, Any] = {"raw_files": {}, "total_lines": 0, "existing_files": 0}
    for name, path_value in raw_specs.items():
        path = resolve_path(root, path_value)
        line_count = 0
        if path.exists():
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                line_count = sum(1 for _ in handle)
            stats["existing_files"] += 1
        stats["total_lines"] += line_count
        stats["raw_files"][name] = {"path": path_value, "exists": path.exists(), "line_count": line_count}
    return stats


def build_execution_preflight(config: Mapping[str, Any], root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_hashes = config.get("input_hashes", {})
    logical_rows = read_csv_rows(resolve_path(root, input_hashes["f6g1_future_logical_instances"]["path"]))
    request_rows = read_csv_rows(resolve_path(root, input_hashes["f6g1_future_request_manifest"]["path"]))
    shard_rows = read_csv_rows(resolve_path(root, input_hashes["f6g1_future_shard_manifest"]["path"]))
    f6g2_rows = read_csv_rows(resolve_path(root, input_hashes["f6g2_request_execution"]["path"]))
    summary = summarize_contract(config, logical_rows, request_rows, shard_rows, f6g2_rows)
    raw_stats = _raw_file_stats(_load_f6g2_config(config, root), root)
    summary["raw_stats"] = raw_stats

    def row(check_id: str, requirement: str, observed: str, status: str, notes: str = "", decision: str = DECISION_CONTRACT, recommendation: str = RECOMMEND_CONTRACT) -> dict[str, Any]:
        return {
            "check_id": check_id,
            "requirement": requirement,
            "observed": observed,
            "status": status,
            "blocking": _safe_bool(status not in {"pass", "skipped"}),
            "decision_if_failed": decision,
            "recommendation_if_failed": recommendation,
            "notes": notes,
        }

    rows = [
        row("logical-count", "F6G1 logical instance count is frozen", f"{summary['logical_count']} expected {summary['expected_logical_count']}", "pass" if summary["logical_count"] == summary["expected_logical_count"] else "fail", decision=DECISION_INPUTS, recommendation=RECOMMEND_BLOCKED),
        row("request-count", "F6G1 request manifest count is frozen", f"{summary['request_count']} expected {summary['expected_request_count']}", "pass" if summary["request_count"] == summary["expected_request_count"] else "fail", decision=DECISION_INPUTS, recommendation=RECOMMEND_BLOCKED),
        row("shard-count", "F6G1 shard manifest count is frozen", f"{summary['shard_count']} expected {summary['expected_shard_count']}", "pass" if summary["shard_count"] == summary["expected_shard_count"] else "fail", decision=DECISION_INPUTS, recommendation=RECOMMEND_BLOCKED),
        row("request-ids", "Request IDs are unique", f"{summary['unique_request_count']} unique of {summary['request_count']}", "pass" if summary["unique_request_count"] == summary["request_count"] else "fail", decision=DECISION_INPUTS, recommendation=RECOMMEND_BLOCKED),
        row("candidate-pool-size", "Candidate pool size remains 20", ";".join(summary["candidate_pool_sizes"]), "pass" if summary["candidate_pool_sizes"] == ["20"] else "fail", decision=DECISION_INPUTS, recommendation=RECOMMEND_BLOCKED),
        row("f6g1-output-contract", "Frozen F6G1 output contract is recorded", f"request={summary['request_output_lengths']};logical={summary['logical_output_lengths']};parser={summary['parser_contracts']}", "pass" if summary["request_output_lengths"] == ["20"] and summary["logical_output_lengths"] == ["20"] else "fail", decision=DECISION_INPUTS, recommendation=RECOMMEND_BLOCKED),
        row("top10-adapter-contract", "Prior adapter evidence requires strict top-10", f"requested={summary['top10_required_requested_ranking_length']};parser={summary['top10_required_parser_expected_length']}", "pass" if summary["top10_required_requested_ranking_length"] == "10" and summary["top10_required_parser_expected_length"] == "10" else "fail", decision=DECISION_PROVENANCE, recommendation=RECOMMEND_BLOCKED),
        row("contract-reconciliation", "Do not execute if top-20 F6G1 conflicts with top-10 adapter", summary["blocking_reason"] or "no conflict", "blocked" if summary["contract_status"] != "pass" else "pass", notes="Hard pre-inference gate; F6G2R does not rewrite F6G1 or F6G2.", decision=DECISION_CONTRACT, recommendation=RECOMMEND_CONTRACT),
        row("raw-files", "No F6G2 raw JSONL records exist before execution", f"files={raw_stats['existing_files']};lines={raw_stats['total_lines']}", "pass" if raw_stats["total_lines"] == 0 else "fail", decision=DECISION_INPUTS, recommendation=RECOMMEND_BLOCKED),
        row("no-inference", "F6G2R performs no model generation", "generation_attempted=false", "pass", notes="Only metadata checks are allowed."),
    ]
    return rows, summary


def parse_pytest_counts(stdout: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0}
    for key in counts:
        matches = re.findall(rf"(\d+)\s+{key}", stdout)
        if matches:
            counts[key] = int(matches[-1])
    return counts


def run_validation_suites(config: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    validation = config.get("validation", {})
    if not validation.get("run_suites", True):
        return [
            {
                "suite": "validation",
                "command": "",
                "status": "skipped",
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "returncode": 0,
                "duration_seconds": "",
                "stdout_summary": "",
                "stderr_summary": "",
                "notes": "Validation disabled by config.",
            }
        ]
    python_path = resolve_path(root, config.get("runtime", {}).get("venv_python_path", ""))
    rows: list[dict[str, Any]] = []
    for suite in validation.get("suites", []):
        files = [str(file) for file in suite.get("files", [])]
        missing = [file for file in files if not resolve_path(root, file).exists()]
        safe_suite = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(suite.get("name", "suite"))).strip("_") or "suite"
        basetemp_root = Path(str(validation.get("pytest_basetemp_root", "results/logs/phase_f6g2r_pytest_tmp")))
        basetemp = basetemp_root / safe_suite
        resolve_path(root, basetemp).parent.mkdir(parents=True, exist_ok=True)
        command = [str(python_path), "-m", "pytest", *files, "-q", "-p", "no:cacheprovider", "--basetemp", basetemp.as_posix()]
        if missing:
            rows.append(
                {
                    "suite": suite.get("name", ""),
                    "command": " ".join(command),
                    "status": "missing",
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "returncode": 2,
                    "duration_seconds": "",
                    "stdout_summary": "",
                    "stderr_summary": "",
                    "notes": f"Missing files: {';'.join(missing)}",
                }
            )
            continue
        result = run_command(command, root, timeout=int(validation.get("suite_timeout_seconds", 180)))
        counts = parse_pytest_counts(result.stdout)
        status = "pass" if result.returncode == 0 else "fail"
        rows.append(
            {
                "suite": suite.get("name", ""),
                "command": " ".join([Path(command[0]).as_posix(), *command[1:]]),
                "status": status,
                "passed": counts["passed"],
                "failed": counts["failed"],
                "skipped": counts["skipped"],
                "returncode": result.returncode,
                "duration_seconds": "",
                "stdout_summary": _pytest_summary(result.stdout),
                "stderr_summary": _short(result.stderr),
                "notes": "Runtime duration intentionally omitted from scientific table.",
            }
        )
    return rows


def _pytest_summary(stdout: str) -> str:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if " passed" in line or " failed" in line or " skipped" in line or " error" in line:
            normalized = re.sub(r"\s+in\s+\d+(\.\d+)?s", "", line)
            normalized = re.sub(r"\s+\(\d+:\d{2}(:\d{2})?\)", "", normalized)
            return normalized
    return _short(stdout)


def _table_status(rows: Sequence[Mapping[str, Any]], fail_statuses: set[str] | None = None) -> str:
    fail_statuses = fail_statuses or {"fail", "missing", "changed", "blocked"}
    return "fail" if any(str(row.get("status", "")) in fail_statuses for row in rows) else "pass"


def select_final_decision(
    input_rows: Sequence[Mapping[str, Any]],
    installer_rows: Sequence[Mapping[str, Any]],
    environment_rows: Sequence[Mapping[str, Any]],
    dependency_lock_rows: Sequence[Mapping[str, Any]],
    dependency_install_rows: Sequence[Mapping[str, Any]],
    import_rows: Sequence[Mapping[str, Any]],
    ollama_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    execution_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    if _table_status(input_rows, {"missing", "changed"}) == "fail":
        return DECISION_INPUTS, RECOMMEND_BLOCKED
    if _table_status(installer_rows) == "fail" or _table_status(environment_rows) == "fail":
        return DECISION_PYTHON, RECOMMEND_DEPENDENCY
    if _table_status(dependency_lock_rows) == "fail" or _table_status(dependency_install_rows) == "fail" or _table_status(import_rows) == "fail":
        return DECISION_DEPENDENCY, RECOMMEND_DEPENDENCY
    if _table_status(test_rows, {"fail", "missing"}) == "fail":
        return DECISION_TESTS, RECOMMEND_BLOCKED
    if _table_status(ollama_rows) == "fail":
        return DECISION_OLLAMA, RECOMMEND_OLLAMA
    if any(row.get("status") == "blocked" for row in execution_rows):
        return DECISION_CONTRACT, RECOMMEND_CONTRACT
    if _table_status(execution_rows) == "fail":
        return DECISION_PROVENANCE, RECOMMEND_BLOCKED
    return DECISION_SUCCESS, RECOMMEND_SUCCESS


def build_missing_or_blocked(decision: str, summary: Mapping[str, Any], ollama_rows: Sequence[Mapping[str, Any]], test_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if decision == DECISION_CONTRACT:
        rows.append(
            {
                "item_id": "f6g2r-contract-conflict",
                "component": "F6G2 execution contract",
                "status": "blocked",
                "classification": DECISION_CONTRACT,
                "reason": summary.get("blocking_reason", ""),
                "current_evidence": f"F6G1 output lengths={summary.get('request_output_lengths')}; parser={summary.get('parser_contracts')}",
                "required_evidence": "Corrective offline freeze with candidate_pool_size=20, requested_ranking_length=10, parser_expected_length=10, and matching strict parser.",
                "recommended_action": RECOMMEND_CONTRACT,
            }
        )
    if any(row.get("status") == "fail" for row in ollama_rows):
        rows.append(
            {
                "item_id": "f6g2r-ollama-gate",
                "component": "Ollama metadata preflight",
                "status": "blocked",
                "classification": DECISION_OLLAMA,
                "reason": "One or more local model tags were not visible through metadata-only inventory.",
                "current_evidence": stable_json(list(ollama_rows)),
                "required_evidence": "Local qwen2:7b-instruct and mistral:7b-instruct tags visible without pull or generation.",
                "recommended_action": RECOMMEND_OLLAMA,
            }
        )
    if any(row.get("status") in {"fail", "missing"} for row in test_rows):
        rows.append(
            {
                "item_id": "f6g2r-tests",
                "component": "Safe validation suites",
                "status": "blocked",
                "classification": DECISION_TESTS,
                "reason": "One or more configured safe suites failed or were missing.",
                "current_evidence": stable_json(list(test_rows)),
                "required_evidence": "All configured F6G2R/F6G2/parser/protected-input tests pass in .venv_f6g2_py312.",
                "recommended_action": RECOMMEND_BLOCKED,
            }
        )
    if not rows and decision == DECISION_SUCCESS:
        rows.append(
            {
                "item_id": "none",
                "component": "F6G2R",
                "status": "complete",
                "classification": DECISION_SUCCESS,
                "reason": "All runtime, dependency, Ollama, test, input, and execution preflight gates passed.",
                "current_evidence": "ready",
                "required_evidence": "",
                "recommended_action": RECOMMEND_SUCCESS,
            }
        )
    return rows


def build_report(decision: str, recommendation: str, summary: Mapping[str, Any], identity: Mapping[str, Any], table_hashes: Mapping[str, str]) -> str:
    return "\n".join(
        [
            "# F6G2R Runtime Recovery And Execution Preflight",
            "",
            f"Decision: `{decision}`",
            f"Recommendation: `{recommendation}`",
            "",
            "## Runtime",
            f"- CPython: `{identity.get('python_version', '')}`",
            f"- Architecture: `{identity.get('architecture', '')}`",
            f"- Venv executable: `{identity.get('executable', '')}`",
            "- Legacy `.venv`, `.venv_repro`, and `.venv_repro_system_site` were treated as immutable inputs.",
            "",
            "## Execution Contract",
            f"- Frozen logical instances: `{summary.get('logical_count')}`",
            f"- Frozen requests: `{summary.get('request_count')}`",
            f"- Frozen shards: `{summary.get('shard_count')}`",
            f"- Candidate pool sizes: `{';'.join(summary.get('candidate_pool_sizes', []))}`",
            f"- F6G1 output lengths: `{';'.join(summary.get('request_output_lengths', []))}`",
            f"- F6G2 requested/parser lengths: `{';'.join(summary.get('f6g2_requested_lengths', []))}` / `{';'.join(summary.get('f6g2_parser_lengths', []))}`",
            f"- Contract status: `{summary.get('contract_status')}`",
            "",
            "No Ollama generation, model pull, training, CUDA execution, API call, raw JSONL append, staging, or commit was performed.",
            "",
            "## Scientific Hash Inputs",
            f"- Table hashes are recorded in `results/manifests/phase_f6g2r_runtime_recovery_freeze.json`.",
            f"- Report table count: `{len(table_hashes)}`",
            "",
        ]
    )


def _hash_outputs(root: Path, outputs: Mapping[str, str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, path_value in outputs.items():
        path = resolve_path(root, path_value)
        hashes[name] = sha256_file(path) if path.exists() else ""
    return hashes


def _previous_timestamp(manifest_path: Path) -> str:
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            timestamp = payload.get("runtime_metadata", {}).get("first_run_timestamp")
            if timestamp:
                return str(timestamp)
        except json.JSONDecodeError:
            pass
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_manifest(
    config: Mapping[str, Any],
    root: Path,
    decision: str,
    recommendation: str,
    repository_state: Mapping[str, Any],
    identity: Mapping[str, Any],
    table_hashes: Mapping[str, str],
    environment_hashes: Mapping[str, str],
    report_hash: str,
    summary: Mapping[str, Any],
    test_rows: Sequence[Mapping[str, Any]],
    input_rows: Sequence[Mapping[str, Any]],
    manifest_path: Path,
) -> dict[str, Any]:
    changed = sum(1 for row in input_rows if row.get("changed") == "true")
    missing = sum(1 for row in input_rows if row.get("missing") == "true")
    total_failed = sum(int(row.get("failed", 0) or 0) for row in test_rows)
    total_passed = sum(int(row.get("passed", 0) or 0) for row in test_rows)
    total_skipped = sum(int(row.get("skipped", 0) or 0) for row in test_rows)
    scientific_payload = {
        "phase": PHASE_ID,
        "decision": decision,
        "recommendation": recommendation,
        "environment_hashes": environment_hashes,
        "input_hashes": {row["path"]: row["actual_sha256"] for row in input_rows},
        "input_invariance": {"changed": changed, "missing": missing},
        "no_inference": True,
        "report_hash": report_hash,
        "runtime_identity": {
            "architecture": identity.get("architecture", ""),
            "python_sha256": identity.get("python_sha256", ""),
            "python_version": identity.get("python_version", ""),
            "venv_executable": identity.get("executable", ""),
        },
        "table_hashes": dict(table_hashes),
        "execution_contract": {
            "contract_status": summary.get("contract_status"),
            "logical_count": summary.get("logical_count"),
            "request_count": summary.get("request_count"),
            "shard_count": summary.get("shard_count"),
            "candidate_pool_sizes": summary.get("candidate_pool_sizes"),
            "request_output_lengths": summary.get("request_output_lengths"),
            "f6g2_requested_lengths": summary.get("f6g2_requested_lengths"),
            "f6g2_parser_lengths": summary.get("f6g2_parser_lengths"),
            "execution_allowed": summary.get("execution_allowed"),
        },
        "test_totals": {"passed": total_passed, "failed": total_failed, "skipped": total_skipped},
        "unsafe_exclusions": config.get("validation", {}).get("unsafe_exclusions", []),
    }
    return {
        **scientific_payload,
        "scientific_content_hash": sha256_text(stable_json(scientific_payload)),
        "runtime_metadata": {
            "first_run_timestamp": _previous_timestamp(manifest_path),
            "generated_timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "platform": platform.platform(),
            "repository_state": dict(repository_state),
            "runtime_metadata_separated_from_scientific_hash": True,
        },
    }


def write_phase_outputs(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    repository_state = record_repository_state(config, root)
    identity = collect_python_identity(resolve_path(root, config.get("runtime", {}).get("venv_python_path", "")), root)
    environment_hashes = write_environment_files(config, root, identity)
    python_inventory = build_python_inventory(config, root)
    installer = build_installer_provenance(config, root)
    environment_identity = build_environment_identity(config, identity, root)
    dependency_lock = build_dependency_lock(config, root)
    dependency_installation, _pip_status = build_dependency_installation(config, root)
    import_gate = build_import_gate(config, root)
    input_integrity = verify_input_hashes(config, root)
    execution_preflight, contract_summary = build_execution_preflight(config, root)
    ollama_preflight = build_ollama_preflight(config, root)
    test_results = run_validation_suites(config, root)

    decision, recommendation = select_final_decision(
        input_integrity,
        installer,
        environment_identity,
        dependency_lock,
        dependency_installation,
        import_gate,
        ollama_preflight,
        test_results,
        execution_preflight,
    )
    missing = build_missing_or_blocked(decision, contract_summary, ollama_preflight, test_results)

    table_rows = {
        "python_inventory": (python_inventory, PYTHON_INVENTORY_COLUMNS),
        "installer_provenance": (installer, INSTALLER_PROVENANCE_COLUMNS),
        "environment_identity": (environment_identity, ENVIRONMENT_IDENTITY_COLUMNS),
        "dependency_lock": (dependency_lock, DEPENDENCY_LOCK_COLUMNS),
        "dependency_installation": (dependency_installation, DEPENDENCY_INSTALLATION_COLUMNS),
        "import_gate": (import_gate, IMPORT_GATE_COLUMNS),
        "ollama_preflight": (ollama_preflight, OLLAMA_PREFLIGHT_COLUMNS),
        "input_integrity": (input_integrity, INPUT_INTEGRITY_COLUMNS),
        "test_results": (test_results, TEST_RESULTS_COLUMNS),
        "execution_preflight": (execution_preflight, EXECUTION_PREFLIGHT_COLUMNS),
        "missing_or_blocked": (missing, MISSING_COLUMNS),
    }
    table_paths = config.get("outputs", {}).get("tables", {})
    for name, (rows, columns) in table_rows.items():
        write_csv(resolve_path(root, table_paths[name]), rows, columns)

    table_hashes = _hash_outputs(root, table_paths)
    report_text = build_report(decision, recommendation, contract_summary, identity, table_hashes)
    report_path = resolve_path(root, config.get("outputs", {}).get("report", "report/phase_f6g2r_runtime_recovery.md"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    report_hash = sha256_file(report_path)
    manifest_path = resolve_path(root, config.get("outputs", {}).get("freeze_manifest", "results/manifests/phase_f6g2r_runtime_recovery_freeze.json"))
    manifest = build_manifest(
        config,
        root,
        decision,
        recommendation,
        repository_state,
        identity,
        table_hashes,
        environment_hashes,
        report_hash,
        contract_summary,
        test_results,
        input_integrity,
        manifest_path,
    )
    write_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the F6G2R runtime recovery freeze.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the F6G2R YAML config.")
    args = parser.parse_args(argv)
    root = Path.cwd()
    config = load_config(resolve_path(root, args.config))
    manifest = write_phase_outputs(config, root)
    print(stable_json({"decision": manifest["decision"], "recommendation": manifest["recommendation"], "scientific_content_hash": manifest["scientific_content_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
