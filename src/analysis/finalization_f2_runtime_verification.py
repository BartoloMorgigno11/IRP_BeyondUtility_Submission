"""Finalization F2 runtime verification and canonical consolidation.

This module builds an offline verification package for the Beyond Utility
reproducibility wrapper. It inspects the local environment, hashes protected
inputs, runs only explicitly allow-listed safe tests when requested, reconciles
metric evidence by cohort and source, and writes deterministic F2 artifacts.

It never downloads data, calls APIs, calls Ollama, trains models, stages Git
changes, commits, checks out branches, or modifies upstream artifacts.
"""

from __future__ import annotations

from src.audit import common_git

import ast
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

try:  # pragma: no cover - dependency is expected but inventory must be robust.
    import yaml
except Exception:  # pragma: no cover
    yaml = None


EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"

REPORT_PATH = Path("report/finalization_f2_runtime_verification.md")
OUTPUT_TABLES = {
    "environment_inventory": Path("results/tables/finalization_f2_environment_inventory.csv"),
    "test_inventory": Path("results/tables/finalization_f2_test_inventory.csv"),
    "test_results": Path("results/tables/finalization_f2_test_results.csv"),
    "hashes_before": Path("results/tables/finalization_f2_protected_input_hashes_before.csv"),
    "hashes_after": Path("results/tables/finalization_f2_protected_input_hashes_after.csv"),
    "hash_validation": Path("results/tables/finalization_f2_protected_input_validation.csv"),
    "result_conflicts": Path("results/tables/finalization_f2_result_conflicts.csv"),
    "authoritative_sources": Path("results/tables/finalization_f2_authoritative_sources.csv"),
    "artifact_classification": Path("results/tables/finalization_f2_artifact_classification.csv"),
    "canonical_registry": Path("results/tables/finalization_f2_canonical_result_registry.csv"),
}
ALL_OUTPUT_PATHS = [REPORT_PATH, *OUTPUT_TABLES.values()]

PHASE6C_HASH_SOURCE = Path("results/tables/plan_d_phase_6c_protected_input_hashes_initial.csv")
PHASE6C_HASH_FINAL = Path("results/tables/plan_d_phase_6c_protected_input_hashes_final.csv")

LEVEL0_FILES = [
    "tests/test_utility_metrics.py",
    "tests/test_hallucination.py",
    "tests/test_output_parser.py",
    "tests/test_schemas.py",
    "tests/test_cost_metrics.py",
    "tests/test_canonical_metric_contract.py",
    "tests/test_canonical_metric_computation.py",
    "tests/test_canonical_local_llm_beyond_utility.py",
    "tests/test_item_title_map_reconciliation.py",
]
LEVEL1_FILES = [
    "tests/test_plan_d_llm_prompt_templates.py",
    "tests/test_plan_d_llm_model_registry.py",
    "tests/test_plan_d_exam_readiness_review.py",
    "tests/test_canonical_local_llm_full_evaluation.py",
    "tests/test_finalization_f2_runtime_verification.py",
]
LEVEL2_FILES = [
    "tests/test_experiment_cohort_policy.py",
    "tests/test_plan_d_phase_5e_paper_coverage_audit.py",
    "tests/test_plan_d_reproducibility_matrix.py",
    "tests/test_plan_d_bprmf.py",
    "tests/test_plan_d_gru4rec.py",
    "tests/test_plan_d_sasrec.py",
    "tests/test_plan_d_lightgcn.py",
    "tests/test_plan_d_amazon_extended_summary.py",
    "tests/test_plan_d_extended_neural_summary.py",
    "tests/test_plan_d_global_extended_neural_summary.py",
    "tests/test_scaffold.py",
]
LEVEL3_EXAMPLES = {
    "tests/test_mostpop.py",
    "tests/test_prepare_ml1m.py",
    "tests/test_run_pipeline.py",
    "tests/test_run_plan_c_baselines.py",
    "tests/test_make_plots.py",
}
LEVEL4_EXAMPLES = {
    "tests/test_check_plan_c_source_urls.py",
    "tests/test_plan_e_lightgcn_cuda_first_step_diagnostic.py",
}

ENVIRONMENT_COLUMNS = [
    "inventory_id",
    "category",
    "check_name",
    "expected_value",
    "observed_value",
    "status",
    "severity",
    "source_path",
    "command",
    "notes",
]
TEST_INVENTORY_COLUMNS = [
    "test_node_id",
    "file",
    "test_name",
    "level",
    "estimated_level",
    "modules_covered",
    "uses_tmp_path",
    "writes_files",
    "persistent_path_literal",
    "network_reference",
    "ollama_reference",
    "torch_or_cuda_reference",
    "risk",
    "recommended_command_group",
    "prerequisites",
    "estimated_time_seconds",
    "success_criterion",
    "notes",
]
TEST_RESULT_COLUMNS = [
    "test_run_id",
    "level",
    "command",
    "static_test_count",
    "return_code",
    "duration_seconds",
    "status",
    "stdout_tail",
    "stderr_tail",
    "notes",
]
HASH_COLUMNS = [
    "relative_path",
    "exists",
    "size_bytes",
    "sha256",
    "hash_policy",
    "source_hash_table",
    "notes",
]
HASH_VALIDATION_COLUMNS = [
    "relative_path",
    "exists_before",
    "exists_after",
    "sha256_before",
    "sha256_after",
    "status",
    "severity",
    "source_hash_table",
    "notes",
]
REGISTRY_COLUMNS = [
    "evidence_id",
    "phase",
    "experiment_family",
    "dataset",
    "task",
    "model",
    "model_role",
    "paper_equivalence",
    "prompt_contract",
    "cohort_name",
    "cohort_type",
    "candidate_set_name",
    "candidate_count",
    "positive_count",
    "evaluation_population",
    "strict_or_full",
    "seed",
    "run_id",
    "metric",
    "cutoff",
    "value",
    "valid_rows",
    "total_rows",
    "status",
    "result_scope",
    "source_file",
    "source_row",
    "config_file",
    "code_entrypoint",
    "test_evidence",
    "report_reference",
    "notes",
    "source_sha256",
    "metric_definition_id",
    "evaluation_policy_id",
    "denominator_policy",
    "artifact_category",
    "source_value_column",
    "source_table_kind",
    "git_commit",
    "created_at_utc",
    "comparability_group",
    "conflict_group",
]
CONFLICT_COLUMNS = [
    "conflict_id",
    "model",
    "dataset",
    "task",
    "metric",
    "cutoff",
    "values",
    "source_files",
    "phase",
    "cohort",
    "candidate_set",
    "seed",
    "possible_explanation",
    "is_true_conflict",
    "evidence_needed",
]
AUTHORITATIVE_COLUMNS = [
    "authority_rank",
    "source_file",
    "source_kind",
    "exists",
    "row_count",
    "key_columns",
    "authority_class",
    "preferred_use",
    "exceptions",
    "sha256",
    "notes",
]
ARTIFACT_COLUMNS = [
    "source_file",
    "path",
    "status",
    "git_status",
    "artifact_category",
    "importance",
    "dependencies",
    "loss_risk",
    "regenerability",
    "backup_need",
    "tracking_recommendation",
    "gitignore_recommendation",
    "phase_relation",
    "notes",
]


class F2OutputExistsError(RuntimeError):
    """Raised when F2 output paths already exist."""


@dataclass(frozen=True)
class CommandResult:
    command: str
    return_code: int
    stdout: str
    stderr: str
    duration_seconds: float


def safe_test_files() -> dict[str, list[str]]:
    return {
        "LEVEL0": list(LEVEL0_FILES),
        "LEVEL1": list(LEVEL1_FILES),
        "LEVEL2": list(LEVEL2_FILES),
    }


def all_f2_output_paths() -> list[Path]:
    return list(ALL_OUTPUT_PATHS)


def ensure_no_existing_outputs(root: Path) -> None:
    existing = [path.as_posix() for path in all_f2_output_paths() if (root / path).exists()]
    if existing:
        raise F2OutputExistsError("F2 output paths already exist: " + "; ".join(existing))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, values: Iterable[Any], width: int = 12) -> str:
    payload = json.dumps([str(value) for value in values], ensure_ascii=True, sort_keys=False)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:width]}"


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def rel_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def run_command(args: Sequence[str], *, cwd: Path, timeout: int = 30) -> CommandResult:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(
            command=" ".join(args),
            return_code=int(completed.returncode),
            stdout=completed.stdout.strip(),
            stderr=completed.stderr.strip(),
            duration_seconds=round(time.perf_counter() - start, 3),
        )
    except Exception as exc:  # pragma: no cover - environment dependent.
        return CommandResult(
            command=" ".join(args),
            return_code=999,
            stdout="",
            stderr=repr(exc),
            duration_seconds=round(time.perf_counter() - start, 3),
        )


def table_row_count(path: Path) -> int | str:
    if not path.exists():
        return "missing"
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except UnicodeDecodeError:
        return "binary_or_unreadable"


def table_columns(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return handle.readline().strip()
    except UnicodeDecodeError:
        return ""


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


def environment_inventory(root: Path, config_path: Path | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(category: str, check_name: str, expected: Any, observed: Any, status: str, severity: str, source: str = "", command: str = "", notes: str = "") -> None:
        rows.append(
            {
                "inventory_id": f"ENV-{len(rows) + 1:04d}",
                "category": category,
                "check_name": check_name,
                "expected_value": clean_cell(expected),
                "observed_value": clean_cell(observed),
                "status": status,
                "severity": severity,
                "source_path": source,
                "command": command,
                "notes": notes,
            }
        )

    git_branch = run_command(["git", "branch", "--show-current"], cwd=root)
    git_head = run_command(["git", "rev-parse", "HEAD"], cwd=root)
    git_status = run_command(["git", "status", "--short", "--branch"], cwd=root)
    git_diff = run_command(["git", "diff", "--stat", "--", "README.md", "report/dataset_audit.md", "report/notes.md"], cwd=root)
    add("git", "branch", EXPECTED_BRANCH, git_branch.stdout, "pass" if git_branch.stdout == EXPECTED_BRANCH else "warning", "blocking", command=git_branch.command)
    add("git", "head", EXPECTED_HEAD, git_head.stdout, "pass" if git_head.stdout == EXPECTED_HEAD else "warning", "blocking", command=git_head.command)
    add("git", "short_status", "dirty worktree documented", git_status.stdout, "info", "warning", command=git_status.command)
    add("git", "tracked_diff_stat", "README/report tracked diffs only", git_diff.stdout or git_diff.stderr, "info", "warning", command=git_diff.command)

    pyvenv_path = root / ".venv/pyvenv.cfg"
    pyvenv_text = pyvenv_path.read_text(encoding="utf-8") if pyvenv_path.exists() else ""
    pyvenv = parse_key_value_text(pyvenv_text)
    add("python", "pyvenv_cfg_exists", "true", pyvenv_path.exists(), "pass" if pyvenv_path.exists() else "fail", "blocking", " .venv/pyvenv.cfg".strip())
    add("python", "pyvenv_version", "3.12.10 from prior audit", pyvenv.get("version", ""), "pass" if pyvenv.get("version") == "3.12.10" else "warning", "warning", ".venv/pyvenv.cfg")
    add("python", "pyvenv_home", "WindowsApps Python base recorded", pyvenv.get("home", ""), "warning" if "WindowsApps" in pyvenv.get("home", "") else "info", "warning", ".venv/pyvenv.cfg")
    add("python", "pyvenv_command", "venv creation command recorded", pyvenv.get("command", ""), "info", "info", ".venv/pyvenv.cfg")

    exe = root / ".venv/Scripts/python.exe"
    add("python", "venv_python_exe_exists", "true", exe.exists(), "pass" if exe.exists() else "fail", "blocking", ".venv/Scripts/python.exe")
    if exe.exists():
        stat = exe.stat()
        add("python", "venv_python_exe_metadata", "size and mtime readable", f"size={stat.st_size};mtime={int(stat.st_mtime)}", "pass", "info", ".venv/Scripts/python.exe")

    for command_name, args in [
        ("where_python", ["where.exe", "python"]),
        ("py_launcher", ["py", "-0p"]),
        ("powershell_get_command_python", ["powershell", "-NoProfile", "-Command", "Get-Command python -All | Select-Object -ExpandProperty Source"]),
        ("venv_python_isolated_version", [str(exe), "-I", "-S", "-VV"]),
        ("venv_python_introspection", [str(exe), "-I", "-S", "-c", "import sys; print(sys.executable); print(sys.version)"]),
    ]:
        result = run_command(args, cwd=root)
        status = "pass" if result.return_code == 0 else "fail"
        add("python", command_name, "return_code=0", result.stdout or result.stderr, status, "blocking" if command_name.startswith("venv_python") else "warning", command=result.command)

    req_path = root / "requirements.txt"
    env_path = root / "environment.yml"
    requirements = req_path.read_text(encoding="utf-8").splitlines() if req_path.exists() else []
    env_text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    add("dependencies", "requirements_txt_exists", "true", req_path.exists(), "pass" if req_path.exists() else "fail", "blocking", "requirements.txt")
    add("dependencies", "environment_yml_exists", "true", env_path.exists(), "pass" if env_path.exists() else "warning", "warning", "environment.yml")
    add("dependencies", "requirements_declared", "numpy,pandas,PyYAML,matplotlib,pytest,torch", ";".join(line.strip() for line in requirements if line.strip()), "info", "info", "requirements.txt")
    add("dependencies", "environment_python_version", "python=3.10", "python=3.10" if "python=3.10" in env_text else "", "warning", "warning", "environment.yml", notes="Diverges from .venv Python 3.12.10")
    add("dependencies", "lock_files", "none expected by audit", ";".join(existing_optional_files(root, ["pyproject.toml", "setup.cfg", "setup.py", "poetry.lock", "Pipfile.lock", "conda-lock.yml"])) or "none", "info", "info")
    add("ide", "vscode_dir", "not required", (root / ".vscode").exists(), "info", "info", ".vscode")

    pip = run_command([str(exe), "-m", "pip", "list", "--format=json"], cwd=root, timeout=60) if exe.exists() else CommandResult("", 1, "", "missing python", 0)
    if pip.return_code == 0:
        packages = parse_pip_json(pip.stdout)
        add("dependencies", "pip_package_count", "packages listed", len(packages), "pass", "info", command=pip.command)
        for name in ["numpy", "pandas", "PyYAML", "matplotlib", "pytest", "torch", "rapidfuzz"]:
            version = package_version(packages, name)
            declared = dependency_declared(name, requirements, env_text)
            status = "pass" if version else ("info" if name == "rapidfuzz" else "warning")
            notes = "optional undeclared fuzzy dependency" if name == "rapidfuzz" and not declared else ""
            add("dependencies", f"package_{name}", "installed if required", version or "not installed", status, "warning" if not version and name != "rapidfuzz" else "info", "pip list", notes=notes)
    else:
        add("dependencies", "pip_list", "return_code=0", pip.stderr, "fail", "blocking", command=pip.command)

    used = used_important_imports(root / "src")
    declared_text = "\n".join(requirements) + "\n" + env_text
    for package in sorted(used):
        declared = dependency_declared(package, requirements, declared_text)
        status = "pass" if declared or package == "rapidfuzz" else "warning"
        notes = "optional import guarded in output parser" if package == "rapidfuzz" else ""
        add("dependencies", f"used_import_{package}", "declared or optional", f"declared={declared}", status, "warning", "src/**/*.py", notes=notes)

    if config_path:
        add("config", "requested_config", "readable", config_path.as_posix(), "pass" if (root / config_path).exists() else "warning", "info", config_path.as_posix())
        if yaml and (root / config_path).exists():
            try:
                data = yaml.safe_load((root / config_path).read_text(encoding="utf-8"))
                add("config", "requested_config_dataset", "dataset name readable", data.get("dataset", {}).get("name", ""), "pass", "info", config_path.as_posix())
            except Exception as exc:  # pragma: no cover
                add("config", "requested_config_parse", "yaml parse", repr(exc), "warning", "warning", config_path.as_posix())

    return pd.DataFrame(rows, columns=ENVIRONMENT_COLUMNS)


def parse_key_value_text(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def existing_optional_files(root: Path, names: Sequence[str]) -> list[str]:
    return [name for name in names if (root / name).exists()]


def parse_pip_json(text: str) -> list[dict[str, str]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def package_version(packages: Sequence[Mapping[str, Any]], name: str) -> str:
    normalized = normalize_name(name)
    for package in packages:
        if normalize_name(package.get("name", "")) == normalized:
            return clean_cell(package.get("version", ""))
    return ""


def normalize_name(value: str) -> str:
    return value.lower().replace("_", "-")


def dependency_declared(name: str, requirements: Sequence[str], env_text: str) -> bool:
    aliases = {normalize_name(name)}
    if name.lower() == "pyyaml":
        aliases.add("pyyaml")
        aliases.add("yaml")
    text = "\n".join(requirements).lower() + "\n" + env_text.lower()
    return any(alias in text for alias in aliases)


def used_important_imports(src_dir: Path) -> set[str]:
    important = {"numpy", "pandas", "yaml", "matplotlib", "pytest", "torch", "rapidfuzz"}
    used: set[str] = set()
    if not src_dir.exists():
        return used
    for path in src_dir.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in important:
                        used.add("PyYAML" if top == "yaml" else top)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if top in important:
                    used.add("PyYAML" if top == "yaml" else top)
    return used


def test_inventory(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    safe_map = {file: "LEVEL0" for file in LEVEL0_FILES}
    safe_map.update({file: "LEVEL1" for file in LEVEL1_FILES})
    safe_map.update({file: "LEVEL2" for file in LEVEL2_FILES})
    for file_path in sorted((root / "tests").glob("test_*.py")):
        rel = rel_path(root, file_path)
        text = file_path.read_text(encoding="utf-8")
        functions = test_functions(text)
        if not functions:
            continue
        file_level = safe_map.get(rel)
        for name, segment in functions:
            body = segment or text
            flags = classify_test_text(body + "\n" + text[:500])
            estimated = file_level or estimate_level(rel, flags)
            rows.append(
                {
                    "test_node_id": f"{rel}::{name}",
                    "file": rel,
                    "test_name": name,
                    "level": estimated,
                    "estimated_level": estimated,
                    "modules_covered": modules_covered_from_text(text),
                    "uses_tmp_path": str(flags["tmp"]).lower(),
                    "writes_files": str(flags["write"]).lower(),
                    "persistent_path_literal": str(flags["persistent"]).lower(),
                    "network_reference": str(flags["network"]).lower(),
                    "ollama_reference": str(flags["ollama"]).lower(),
                    "torch_or_cuda_reference": str(flags["torch"]).lower(),
                    "risk": risk_label(estimated, flags),
                    "recommended_command_group": estimated if estimated in {"LEVEL0", "LEVEL1", "LEVEL2"} else "excluded_from_f2_safe_run",
                    "prerequisites": "python starts; dependencies import" if estimated in {"LEVEL0", "LEVEL1", "LEVEL2"} else "separate authorization",
                    "estimated_time_seconds": estimated_seconds(estimated),
                    "success_criterion": "pytest return_code=0" if estimated in {"LEVEL0", "LEVEL1", "LEVEL2"} else "not executed in F2",
                    "notes": notes_for_test(rel, estimated),
                }
            )
    return pd.DataFrame(rows, columns=TEST_INVENTORY_COLUMNS)


def test_functions(text: str) -> list[tuple[str, str]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    result: list[tuple[str, str]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            start = max(node.lineno - 1, 0)
            end = getattr(node, "end_lineno", node.lineno)
            result.append((node.name, "\n".join(lines[start:end])))
    return result


def classify_test_text(text: str) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "tmp": "tmp_path" in lowered or "tmpdir" in lowered or "temporarydirectory" in lowered,
        "write": any(token in lowered for token in ["write_text", "to_csv", "to_json", ".mkdir", "open(\"w", "open('w", "gzip.open"]),
        "persistent": any(token in lowered for token in ["results/", "results\\", "report/", "report\\", "data/", "data\\"]),
        "network": any(token in lowered for token in ["urllib", "requests", "socket", "urlopen", "http"]),
        "ollama": "ollama" in lowered,
        "torch": any(token in lowered for token in ["torch", "cuda", "gpu"]),
    }


def estimate_level(rel: str, flags: Mapping[str, bool]) -> str:
    if rel in LEVEL3_EXAMPLES:
        return "LEVEL3"
    if rel in LEVEL4_EXAMPLES:
        return "LEVEL4"
    if flags["network"] or flags["ollama"]:
        return "LEVEL4"
    if flags["torch"] and ("training" in rel or "cuda" in rel or "run_" in rel):
        return "LEVEL4"
    if flags["write"] and flags["tmp"]:
        return "LEVEL3"
    if flags["persistent"]:
        return "LEVEL2"
    return "LEVEL0"


def modules_covered_from_text(text: str) -> str:
    modules: list[str] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        module = ""
        if isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src."):
                    modules.append(alias.name)
            continue
        if module.startswith("src."):
            modules.append(module)
    return ";".join(sorted(set(modules)))


def risk_label(level: str, flags: Mapping[str, bool]) -> str:
    if level == "LEVEL4":
        return "high_excluded"
    if flags["write"] and not flags["tmp"]:
        return "medium_review_write_pattern"
    if level == "LEVEL3":
        return "medium_tmp_only"
    if level == "LEVEL2":
        return "medium_read_existing_artifacts"
    return "low"


def estimated_seconds(level: str) -> int:
    return {"LEVEL0": 30, "LEVEL1": 45, "LEVEL2": 90, "LEVEL3": 180, "LEVEL4": 0}.get(level, 0)


def notes_for_test(rel: str, level: str) -> str:
    if rel in LEVEL0_FILES + LEVEL1_FILES + LEVEL2_FILES:
        return "explicitly allow-listed for F2 safe execution"
    if level == "LEVEL4":
        return "excluded because it may require network, Ollama, CUDA, training, or pipeline execution"
    if level == "LEVEL3":
        return "may be safe with tmp_path and external pytest cache but excluded from required F2 tiers"
    return "static heuristic classification"


def protected_input_paths(root: Path) -> list[str]:
    paths: set[str] = set()
    if (root / PHASE6C_HASH_SOURCE).exists():
        frame = read_csv(root / PHASE6C_HASH_SOURCE)
        for value in frame.get("relative_path", []):
            rel = clean_cell(value).replace("\\", "/")
            if rel and not is_f2_output(rel) and (root / rel).is_file():
                paths.add(rel)
    for pattern in [
        "configs/plan_e_*.yaml",
        "results/tables/plan_e_*_validation.csv",
        "results/tables/plan_e_*_run_manifest.csv",
        "results/tables/plan_e_*_all_datasets_metrics.csv",
        "results/models/**/metadata.json",
    ]:
        for path in root.glob(pattern):
            if path.is_file():
                paths.add(rel_path(root, path))
    return sorted(paths)


def is_f2_output(relative: str) -> bool:
    normalized = relative.replace("\\", "/")
    return normalized == REPORT_PATH.as_posix() or normalized.startswith("results/tables/finalization_f2_")


def hash_table(root: Path, paths: Sequence[str], label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for relative in sorted(dict.fromkeys(paths)):
        path = root / relative
        exists = path.exists()
        rows.append(
            {
                "relative_path": relative,
                "exists": str(exists).lower(),
                "size_bytes": str(path.stat().st_size) if exists and path.is_file() else "",
                "sha256": file_sha256(path) if exists and path.is_file() else "",
                "hash_policy": "sha256",
                "source_hash_table": label,
                "notes": "Phase 6C protected input plus Plan E referenced artifacts",
            }
        )
    return pd.DataFrame(rows, columns=HASH_COLUMNS)


def validate_hashes(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    before_map = {row["relative_path"]: row for row in before.to_dict("records")}
    after_map = {row["relative_path"]: row for row in after.to_dict("records")}
    rows: list[dict[str, Any]] = []
    for relative in sorted(set(before_map) | set(after_map)):
        b = before_map.get(relative, {})
        a = after_map.get(relative, {})
        exists_before = clean_cell(b.get("exists"))
        exists_after = clean_cell(a.get("exists"))
        sha_before = clean_cell(b.get("sha256"))
        sha_after = clean_cell(a.get("sha256"))
        if exists_before == exists_after and sha_before == sha_after:
            status = "unchanged"
            severity = "info"
            notes = "protected input unchanged"
        elif not b:
            status = "new_after"
            severity = "blocking"
            notes = "unexpected protected input added after test execution"
        elif not a:
            status = "missing_after"
            severity = "blocking"
            notes = "protected input missing after test execution"
        else:
            status = "changed"
            severity = "blocking"
            notes = "protected input hash changed"
        rows.append(
            {
                "relative_path": relative,
                "exists_before": exists_before,
                "exists_after": exists_after,
                "sha256_before": sha_before,
                "sha256_after": sha_after,
                "status": status,
                "severity": severity,
                "source_hash_table": clean_cell(b.get("source_hash_table") or a.get("source_hash_table")),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=HASH_VALIDATION_COLUMNS)


def run_safe_tests(root: Path, inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cache_dir = Path("C:/tmp/irp_f2_pytest_cache")
    base_temp = Path("C:/tmp/irp_f2_pytest_tmp")
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_temp.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for level, files in safe_test_files().items():
        existing = [file for file in files if (root / file).exists()]
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            f"cache_dir={cache_dir}",
            "--basetemp",
            str(base_temp),
            *existing,
        ]
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False, env=env)
        duration = round(time.perf_counter() - start, 3)
        static_count = int(inventory.loc[inventory["level"].eq(level) & inventory["file"].isin(existing), "test_node_id"].count())
        rows.append(
            {
                "test_run_id": f"F2-TEST-{len(rows) + 1:03d}",
                "level": level,
                "command": " ".join(command),
                "static_test_count": str(static_count),
                "return_code": str(completed.returncode),
                "duration_seconds": str(duration),
                "status": "pass" if completed.returncode == 0 else "fail",
                "stdout_tail": tail_text(completed.stdout),
                "stderr_tail": tail_text(completed.stderr),
                "notes": "safe allow-listed F2 test tier",
            }
        )
    return pd.DataFrame(rows, columns=TEST_RESULT_COLUMNS)


def tail_text(value: str, max_chars: int = 1200) -> str:
    value = value.strip()
    return value[-max_chars:] if len(value) > max_chars else value


def canonical_result_registry(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    head = git_head(root)
    add_ml1m_mostpop_metrics(root, rows, head)
    add_traditional_baselines(root, rows, head, Path("results/tables/traditional_baselines_metrics.csv"), "Plan E3A/E3B", "traditional_baselines")
    add_traditional_baselines(root, rows, head, Path("results/tables/plan_e_same_cohort_traditional_baselines_metrics.csv"), "Plan E same cohort", "same_cohort_traditional_baselines")
    for model in ["bprmf", "gru4rec", "sasrec", "lightgcn"]:
        add_plan_e_neural_metrics(root, rows, head, model)
    add_llm_cell_metrics(root, rows, head)
    add_llm_utility_metrics(root, rows, head)
    for index, row in enumerate(rows, start=1):
        row["evidence_id"] = f"F2-EVID-{index:06d}"
        row["comparability_group"] = comparability_group(row, include_policy=True)
        row["conflict_group"] = comparability_group(row, include_policy=False)
    return pd.DataFrame(rows, columns=REGISTRY_COLUMNS)


def git_head(root: Path) -> str:
    return common_git.git_stdout_no_check(["rev-parse", "HEAD"], root, strip=False, empty_on_failure=True)


def base_registry_row(root: Path, source: Path, head: str) -> dict[str, Any]:
    return {
        "evidence_id": "",
        "phase": "",
        "experiment_family": "",
        "dataset": "",
        "task": "",
        "model": "",
        "model_role": "",
        "paper_equivalence": "",
        "prompt_contract": "",
        "cohort_name": "",
        "cohort_type": "",
        "candidate_set_name": "",
        "candidate_count": "",
        "positive_count": "",
        "evaluation_population": "",
        "strict_or_full": "",
        "seed": "",
        "run_id": "",
        "metric": "",
        "cutoff": "",
        "value": "",
        "valid_rows": "",
        "total_rows": "",
        "status": "",
        "result_scope": "",
        "source_file": source.as_posix(),
        "source_row": "",
        "config_file": "",
        "code_entrypoint": "",
        "test_evidence": "",
        "report_reference": "",
        "notes": "",
        "source_sha256": file_sha256(root / source) if (root / source).exists() else "",
        "metric_definition_id": "",
        "evaluation_policy_id": "",
        "denominator_policy": "",
        "artifact_category": "",
        "source_value_column": "",
        "source_table_kind": "",
        "git_commit": head,
        "created_at_utc": "",
        "comparability_group": "",
        "conflict_group": "",
    }


def add_ml1m_mostpop_metrics(root: Path, rows: list[dict[str, Any]], head: str) -> None:
    source = Path("results/metrics/ml1m_mostpop_metrics.csv")
    if not (root / source).exists():
        return
    frame = read_csv(root / source)
    for idx, record in frame.iterrows():
        row = base_registry_row(root, source, head)
        metric = clean_cell(record.get("metric"))
        row.update(
            {
                "phase": "Plan B",
                "experiment_family": "ml1m_mostpop_candidate_baseline",
                "dataset": "ML-1M",
                "task": "ranking",
                "model": "MostPop",
                "model_role": "baseline",
                "paper_equivalence": "legacy_reconstructed",
                "cohort_name": "plan_b_ml1m_full_processed",
                "cohort_type": "full_processed_ml1m",
                "candidate_set_name": "data/processed/ml1m/candidates.csv",
                "candidate_count": infer_candidate_count(record),
                "evaluation_population": clean_cell(record.get("num_users")),
                "strict_or_full": "full",
                "seed": "",
                "run_id": clean_cell(record.get("run_id")),
                "metric": "hr_at_5" if metric == "hit_rate" else "ndcg_at_5",
                "cutoff": clean_cell(record.get("k")),
                "value": clean_cell(record.get("value")),
                "valid_rows": clean_cell(record.get("num_users")),
                "total_rows": clean_cell(record.get("num_users")),
                "status": clean_cell(record.get("status")) or "computed",
                "result_scope": "legacy_plan_b_full_ml1m",
                "source_row": str(idx + 2),
                "config_file": clean_cell(record.get("config_path")),
                "code_entrypoint": "src/baselines/ml1m_mostpop_baseline.py",
                "test_evidence": "tests/test_ml1m_mostpop_baseline.py",
                "report_reference": "report/notes.md",
                "notes": "Legacy Plan B full ML-1M MostPop; not directly comparable with Plan E2 1000-row cohort",
                "metric_definition_id": "hit_rate_at_5" if metric == "hit_rate" else "ndcg_at_5",
                "evaluation_policy_id": "legacy_full_processed_candidate_evaluation",
                "denominator_policy": "num_users",
                "artifact_category": "legacy",
                "source_value_column": "value",
                "source_table_kind": "metrics_csv",
            }
        )
        rows.append(row)


def infer_candidate_count(record: Mapping[str, Any]) -> str:
    try:
        users = int(float(clean_cell(record.get("num_users"))))
        recs = int(float(clean_cell(record.get("num_recommendation_rows"))))
        if users:
            return str(recs // users)
    except ValueError:
        return ""
    return ""


def add_traditional_baselines(root: Path, rows: list[dict[str, Any]], head: str, source: Path, phase: str, family: str) -> None:
    if not (root / source).exists():
        return
    frame = read_csv(root / source)
    for idx, record in frame.iterrows():
        for metric, column in [("hr_at_5", "hr_at_5"), ("ndcg_at_5", "ndcg_at_5")]:
            if column not in record or clean_cell(record.get(column)) == "":
                continue
            baseline = clean_cell(record.get("baseline_id"))
            row = base_registry_row(root, source, head)
            row.update(
                {
                    "phase": "Plan E3B" if "bm25" in baseline else phase,
                    "experiment_family": family,
                    "dataset": clean_cell(record.get("dataset")),
                    "task": clean_cell(record.get("task_type")),
                    "model": model_label(baseline),
                    "model_role": "baseline",
                    "paper_equivalence": clean_cell(record.get("paper_alignment")),
                    "cohort_name": clean_cell(record.get("cohort_id")),
                    "cohort_type": "paper_like_full_or_matched",
                    "candidate_set_name": clean_cell(record.get("candidate_set_id")),
                    "candidate_count": clean_cell(record.get("candidate_count")),
                    "evaluation_population": clean_cell(record.get("evaluated_rows")),
                    "strict_or_full": strict_or_full(clean_cell(record.get("cohort_id"))),
                    "seed": "42" if "random" in baseline else "",
                    "run_id": clean_cell(record.get("run_id")),
                    "metric": metric,
                    "cutoff": "5",
                    "value": clean_cell(record.get(column)),
                    "valid_rows": clean_cell(record.get("evaluated_rows")),
                    "total_rows": clean_cell(record.get("evaluated_rows")),
                    "status": "computed",
                    "result_scope": clean_cell(record.get("cohort_id")),
                    "source_row": str(idx + 2),
                    "config_file": "",
                    "code_entrypoint": "src/baselines/content_bm25_baseline.py;src/baselines/random_baseline.py;src/baselines/mostpop.py",
                    "test_evidence": "tests/test_plan_e_baseline_alignment.py;tests/test_plan_e_bm25_scoring.py",
                    "report_reference": "report/plan_e_baseline_alignment.md;report/plan_e_bm25_scoring.md",
                    "notes": clean_cell(record.get("notes")),
                    "metric_definition_id": metric,
                    "evaluation_policy_id": "plan_e2_candidate_pool_evaluation",
                    "denominator_policy": "evaluated_rows",
                    "artifact_category": "reconstructed" if "reconstructed" in clean_cell(record.get("paper_alignment")) else "adapted",
                    "source_value_column": column,
                    "source_table_kind": "normalized_result_table",
                }
            )
            rows.append(row)


def model_label(model_id: str) -> str:
    mapping = {
        "mostpop_train_only": "MostPop",
        "random_seeded": "Random",
        "bm25_like_e3b": "BM25-like",
    }
    return mapping.get(model_id, model_id)


def strict_or_full(cohort: str) -> str:
    lowered = cohort.lower()
    if "matched" in lowered or "strict" in lowered:
        return "strict_or_matched"
    if "full" in lowered:
        return "full"
    return cohort


def add_plan_e_neural_metrics(root: Path, rows: list[dict[str, Any]], head: str, model: str) -> None:
    source = Path(f"results/tables/plan_e_{model}_all_datasets_metrics.csv")
    if not (root / source).exists():
        return
    frame = read_csv(root / source)
    for idx, record in frame.iterrows():
        for metric, column in [("hr_at_5", "hr_at_5"), ("ndcg_at_5", "ndcg_at_5")]:
            if column not in record or clean_cell(record.get(column)) == "":
                continue
            row = base_registry_row(root, source, head)
            row.update(
                {
                    "phase": "Plan E",
                    "experiment_family": f"plan_e_{model}",
                    "dataset": clean_cell(record.get("dataset")),
                    "task": clean_cell(record.get("task_type")),
                    "model": model.upper() if model == "bprmf" else model.capitalize() if model != "lightgcn" else "LightGCN",
                    "model_role": "neural_baseline",
                    "paper_equivalence": clean_cell(record.get("paper_alignment")),
                    "cohort_name": clean_cell(record.get("cohort_id")),
                    "cohort_type": "strict_train_known_or_matched",
                    "candidate_set_name": clean_cell(record.get("candidate_set_id")),
                    "candidate_count": clean_cell(record.get("candidate_count")),
                    "evaluation_population": clean_cell(record.get("evaluated_rows")),
                    "strict_or_full": strict_or_full(clean_cell(record.get("cohort_id"))),
                    "seed": "42",
                    "run_id": clean_cell(record.get("model_id")),
                    "metric": metric,
                    "cutoff": "5",
                    "value": clean_cell(record.get(column)),
                    "valid_rows": clean_cell(record.get("evaluated_rows")),
                    "total_rows": clean_cell(record.get("evaluated_rows")),
                    "status": "computed",
                    "result_scope": "final_extension",
                    "source_row": str(idx + 2),
                    "config_file": f"configs/plan_e_{model}_*.yaml",
                    "code_entrypoint": f"src/baselines/plan_e_{model}_training.py",
                    "test_evidence": f"tests/test_plan_e_{model}_training.py",
                    "report_reference": f"report/plan_e_{model}_beauty_sports_and_baseline_comparison.md",
                    "notes": clean_cell(record.get("notes")),
                    "metric_definition_id": metric,
                    "evaluation_policy_id": "plan_e_strict_candidate_pool_evaluation",
                    "denominator_policy": "evaluated_rows",
                    "artifact_category": "final_extension",
                    "source_value_column": column,
                    "source_table_kind": "normalized_result_table",
                }
            )
            rows.append(row)


def add_llm_cell_metrics(root: Path, rows: list[dict[str, Any]], head: str) -> None:
    source = Path("results/tables/canonical_local_llm_full_matrix_cell_metrics.csv")
    if not (root / source).exists():
        return
    frame = read_csv(root / source)
    for idx, record in frame.iterrows():
        for metric, column in [("hr_at_5", "hr_at_5"), ("ndcg_at_5", "ndcg_at_5")]:
            if column not in record or clean_cell(record.get(column)) == "":
                continue
            row = base_registry_row(root, source, head)
            row.update(
                {
                    "phase": "Plan D Phase 5C",
                    "experiment_family": "canonical_local_llm_full_matrix",
                    "dataset": clean_cell(record.get("dataset")),
                    "task": clean_cell(record.get("task_type")),
                    "model": clean_cell(record.get("model_family")),
                    "model_role": "local_llm",
                    "paper_equivalence": "adapted_project_policy",
                    "prompt_contract": "canonical_index_json",
                    "cohort_name": "canonical_local_llm_full_matrix_cell",
                    "cohort_type": "strict_valid_outputs",
                    "candidate_set_name": "plan_e2_llm_strict_title",
                    "candidate_count": "20",
                    "evaluation_population": clean_cell(record.get("strict_valid_rows")),
                    "strict_or_full": "strict_valid",
                    "seed": "",
                    "run_id": clean_cell(record.get("model_tag")),
                    "metric": metric,
                    "cutoff": "5",
                    "value": clean_cell(record.get(column)),
                    "valid_rows": clean_cell(record.get("strict_valid_rows")),
                    "total_rows": clean_cell(record.get("planned_rows")),
                    "status": "computed_valid_only_diagnostic",
                    "result_scope": "diagnostic_valid_only",
                    "source_row": str(idx + 2),
                    "config_file": "results/tables/canonical_llm_execution_matrix.csv",
                    "code_entrypoint": "src/llm/canonical_local_llm_full_matrix_runner.py",
                    "test_evidence": "tests/test_canonical_local_llm_full_matrix_runner.py",
                    "report_reference": "report/canonical_local_llm_full_evaluation.md",
                    "notes": clean_cell(record.get("notes")),
                    "metric_definition_id": metric,
                    "evaluation_policy_id": "conditional_strict_valid_output_utility",
                    "denominator_policy": "strict_valid_rows",
                    "artifact_category": "diagnostic",
                    "source_value_column": column,
                    "source_table_kind": "canonical_cell_metrics",
                }
            )
            rows.append(row)


def add_llm_utility_metrics(root: Path, rows: list[dict[str, Any]], head: str) -> None:
    source = Path("results/tables/canonical_local_llm_utility_metrics.csv")
    if not (root / source).exists():
        return
    frame = read_csv(root / source)
    for idx, record in frame.iterrows():
        for metric, column in [("hr_at_5", "hr_at_5"), ("ndcg_at_5", "ndcg_at_5")]:
            if column not in record or clean_cell(record.get(column)) == "":
                continue
            policy = clean_cell(record.get("evaluation_policy_id"))
            row = base_registry_row(root, source, head)
            row.update(
                {
                    "phase": "Plan D Phase 5C",
                    "experiment_family": "canonical_local_llm_utility",
                    "dataset": clean_cell(record.get("dataset")),
                    "task": clean_cell(record.get("task_type")),
                    "model": clean_cell(record.get("model_family")),
                    "model_role": "local_llm",
                    "paper_equivalence": "adapted_project_policy",
                    "prompt_contract": "canonical_index_json",
                    "cohort_name": "canonical_local_llm_utility",
                    "cohort_type": policy,
                    "candidate_set_name": "plan_e2_llm_strict_title",
                    "candidate_count": "20",
                    "evaluation_population": clean_cell(record.get("evaluated_rows")),
                    "strict_or_full": "operational" if "operational" in policy else "strict_valid",
                    "seed": "",
                    "run_id": clean_cell(record.get("model_tag")),
                    "metric": metric,
                    "cutoff": "5",
                    "value": clean_cell(record.get(column)),
                    "valid_rows": clean_cell(record.get("evaluated_rows")),
                    "total_rows": clean_cell(record.get("planned_rows")),
                    "status": "computed",
                    "result_scope": "final_canonical" if "operational" in policy else "diagnostic",
                    "source_row": str(idx + 2),
                    "config_file": "results/tables/canonical_local_llm_evaluation_policy_contract.csv",
                    "code_entrypoint": "src/analysis/canonical_local_llm_full_evaluation.py",
                    "test_evidence": "tests/test_canonical_local_llm_full_evaluation.py",
                    "report_reference": "report/canonical_local_llm_full_evaluation.md",
                    "notes": clean_cell(record.get("notes")),
                    "metric_definition_id": metric,
                    "evaluation_policy_id": policy,
                    "denominator_policy": clean_cell(record.get("denominator_policy")),
                    "artifact_category": "final_canonical" if "operational" in policy else "diagnostic",
                    "source_value_column": column,
                    "source_table_kind": "canonical_utility_metrics",
                }
            )
            rows.append(row)


def comparability_group(row: Mapping[str, Any], *, include_policy: bool) -> str:
    parts = [
        row.get("dataset", ""),
        row.get("task", ""),
        row.get("model", ""),
        row.get("metric", ""),
        row.get("cutoff", ""),
    ]
    if include_policy:
        parts.extend(
            [
                row.get("cohort_name", ""),
                row.get("candidate_set_name", ""),
                row.get("evaluation_policy_id", ""),
                row.get("denominator_policy", ""),
            ]
        )
    return stable_id("CMP" if include_policy else "CONFLICT", parts, width=16)


def result_conflicts(registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if registry.empty:
        return pd.DataFrame(rows, columns=CONFLICT_COLUMNS)
    grouped = registry.groupby(["model", "dataset", "task", "metric", "cutoff"], dropna=False)
    for key, group in grouped:
        values = sorted({clean_cell(value) for value in group["value"] if clean_cell(value) != ""})
        if len(values) <= 1:
            continue
        cohorts = sorted(set(group["cohort_name"].astype(str)))
        candidates = sorted(set(group["candidate_set_name"].astype(str)))
        policies = sorted(set(group["evaluation_policy_id"].astype(str)))
        true_conflict = len(cohorts) == 1 and len(candidates) == 1 and len(policies) == 1
        explanation = "same natural comparison fields but values differ" if true_conflict else "different cohort, candidate set, phase, or denominator policy"
        if key[0] == "MostPop" and key[1] == "ML-1M" and key[3] == "hr_at_5":
            explanation = "Plan B full ML-1M has 6040 users/candidate rows; Plan E2 uses 1000-row 20-candidate manifests"
        rows.append(
            {
                "conflict_id": f"F2-CONFLICT-{len(rows) + 1:04d}",
                "model": key[0],
                "dataset": key[1],
                "task": key[2],
                "metric": key[3],
                "cutoff": key[4],
                "values": ";".join(values),
                "source_files": ";".join(sorted(set(group["source_file"].astype(str)))),
                "phase": ";".join(sorted(set(group["phase"].astype(str)))),
                "cohort": ";".join(cohorts),
                "candidate_set": ";".join(candidates),
                "seed": ";".join(sorted(set(group["seed"].astype(str)))),
                "possible_explanation": explanation,
                "is_true_conflict": str(true_conflict).lower(),
                "evidence_needed": "manual review" if true_conflict else "none; registry fields distinguish the values",
            }
        )
    return pd.DataFrame(rows, columns=CONFLICT_COLUMNS)


def authoritative_sources(root: Path) -> pd.DataFrame:
    source_specs = [
        (1, "results/llm/canonical_local_full_matrix/requests.jsonl", "raw_execution_output", "raw evidence for LLM requests"),
        (1, "results/llm/canonical_local_full_matrix/responses.jsonl", "raw_execution_output", "raw evidence for LLM responses"),
        (2, "results/tables/canonical_local_llm_full_matrix_cell_metrics.csv", "normalized_result_table", "LLM cell utility metrics"),
        (2, "results/tables/canonical_local_llm_utility_metrics.csv", "normalized_result_table", "LLM utility policy table"),
        (2, "results/tables/traditional_baselines_metrics.csv", "normalized_result_table", "traditional baseline metrics on Plan E2 manifests"),
        (2, "results/tables/plan_e_bprmf_all_datasets_metrics.csv", "normalized_result_table", "Plan E BPRMF metrics"),
        (2, "results/tables/plan_e_gru4rec_all_datasets_metrics.csv", "normalized_result_table", "Plan E GRU4Rec metrics"),
        (2, "results/tables/plan_e_sasrec_all_datasets_metrics.csv", "normalized_result_table", "Plan E SASRec metrics"),
        (2, "results/tables/plan_e_lightgcn_all_datasets_metrics.csv", "normalized_result_table", "Plan E LightGCN metrics"),
        (3, "results/tables/canonical_local_llm_full_matrix_validation.csv", "validation_table", "LLM matrix validation"),
        (3, "results/tables/canonical_metric_results_validation.csv", "validation_table", "canonical metric validation"),
        (3, "results/tables/plan_d_phase_6c_validation.csv", "validation_table", "final report validation"),
        (4, "results/tables/plan_d_phase_5e_coverage_metrics.csv", "coverage_table", "paper component coverage"),
        (4, "results/tables/experiment_cohort_summary.csv", "cohort_table", "cohort definitions"),
        (4, "results/tables/dataset_statistics.csv", "dataset_table", "dataset lineage"),
        (5, "report/final_scientific_report.md", "scientific_report", "narrative synthesis"),
        (6, "README.md", "readme", "user-facing summary"),
        (7, "report/notes.md", "notes", "chronological notes"),
    ]
    rows = []
    for rank, source, kind, use in source_specs:
        path = root / source
        rows.append(
            {
                "authority_rank": str(rank),
                "source_file": source,
                "source_kind": kind,
                "exists": str(path.exists()).lower(),
                "row_count": str(table_row_count(path)) if path.suffix.lower() == ".csv" else "",
                "key_columns": table_columns(path),
                "authority_class": authority_class(rank),
                "preferred_use": use,
                "exceptions": "legacy/superseded rows require artifact taxonomy before citation" if "README" not in source else "not authoritative over CSV tables",
                "sha256": file_sha256(path) if path.exists() and path.is_file() else "",
                "notes": "ordered by proximity to execution and normalization",
            }
        )
    return pd.DataFrame(rows, columns=AUTHORITATIVE_COLUMNS)


def authority_class(rank: int) -> str:
    return {
        1: "highest_raw_execution",
        2: "primary_normalized_results",
        3: "validation_evidence",
        4: "canonical_scope_tables",
        5: "scientific_interpretation",
        6: "portable_summary",
        7: "chronology",
    }[rank]


def artifact_classification(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for status, path in git_status_entries(root):
        rows.append(classify_artifact(path, status))
    for path in all_f2_output_paths():
        rows.append(
            {
                "source_file": path.as_posix(),
                "path": path.as_posix(),
                "status": "planned",
                "git_status": "planned_f2_output",
                "artifact_category": "finalization_output",
                "importance": "high",
                "dependencies": "F2 builder",
                "loss_risk": "low after regeneration if inputs unchanged",
                "regenerability": "regenerable by F2 builder if no output guard collision",
                "backup_need": "normal",
                "tracking_recommendation": "track after review",
                "gitignore_recommendation": "do not ignore canonical F2 tables/report",
                "phase_relation": "Finalization F2",
                "notes": "created only after all in-memory checks complete",
            }
        )
    return pd.DataFrame(rows, columns=ARTIFACT_COLUMNS)


def git_status_entries(root: Path) -> list[tuple[str, str]]:
    return common_git.git_status_entries_z(root)


def classify_artifact(path: str, status: str) -> dict[str, str]:
    normalized = path.replace("\\", "/")
    category = "unknown"
    importance = "medium"
    regenerability = "unknown"
    tracking = "review"
    gitignore = "review"
    phase = infer_phase(normalized)
    if normalized.startswith("src/"):
        category = "source implementation"
        importance = "high"
        regenerability = "hard to regenerate automatically"
        tracking = "track after tests pass"
    elif normalized.startswith("tests/"):
        category = "tests"
        importance = "high"
        regenerability = "hard to regenerate automatically"
        tracking = "track after tests pass"
    elif normalized.startswith("configs/"):
        category = "config"
        importance = "high"
        regenerability = "manual or phase-specific"
        tracking = "track if used by cited outputs"
    elif normalized.startswith("results/llm/"):
        category = "raw model output"
        importance = "high"
        regenerability = "not regenerable without LLM inference"
        tracking = "review size and privacy; preserve"
        gitignore = "possibly keep ignored if too large but catalog"
    elif normalized.startswith("results/models/"):
        category = "model artifact"
        importance = "medium"
        regenerability = "not regenerable without training"
        tracking = "review size; preserve metadata/checkpoints if cited"
    elif normalized.startswith("results/tables/"):
        category = result_table_category(normalized)
        importance = "high" if "final" in category or "validation" in category else "medium"
        regenerability = "regenerable only by phase builder if inputs unchanged"
        tracking = "track canonical/validation tables after review"
    elif normalized.startswith("report/"):
        category = "report"
        importance = "high"
        regenerability = "manual or builder-generated"
        tracking = "track final reports after review"
    elif normalized.startswith("docs/") or normalized.endswith(".md"):
        category = "documentation"
        importance = "medium"
        regenerability = "manual"
        tracking = "track if part of final package"
    elif normalized.startswith("paper/") or normalized.startswith("original_repo/"):
        category = "upstream artifact"
        importance = "high"
        regenerability = "external source required"
        tracking = "preserve, do not modify"
    return {
        "source_file": normalized,
        "path": normalized,
        "status": "tracked_modified" if "M" in status else "untracked" if status == "??" else "git_status",
        "git_status": status,
        "artifact_category": category,
        "importance": importance,
        "dependencies": infer_dependencies(normalized),
        "loss_risk": "high" if "not regenerable" in regenerability else "medium",
        "regenerability": regenerability,
        "backup_need": "yes" if category in {"raw model output", "model artifact", "upstream artifact"} else "normal",
        "tracking_recommendation": tracking,
        "gitignore_recommendation": gitignore,
        "phase_relation": phase,
        "notes": "classification is non-mutating; no deletion proposed",
    }


def infer_phase(path: str) -> str:
    for token in ["phase_6c", "phase_6b", "phase_6a", "phase_5e", "phase_5d", "phase_6clean"]:
        if token in path:
            return token
    if "plan_e" in path:
        return "Plan E"
    if "canonical_local_llm" in path:
        return "Canonical local LLM"
    return ""


def infer_dependencies(path: str) -> str:
    if path.startswith("results/tables/"):
        return "source data/config/code from producing phase"
    if path.startswith("report/"):
        return "CSV evidence and source inventory"
    if path.startswith("src/"):
        return "Python runtime and local CSV artifacts"
    if path.startswith("tests/"):
        return "pytest and referenced code/artifacts"
    return ""


def result_table_category(path: str) -> str:
    lowered = path.lower()
    if "validation" in lowered:
        return "final result validation" if "phase_6c" in lowered or "canonical" in lowered else "validation"
    if "pilot" in lowered:
        return "pilot result"
    if "diagnostic" in lowered:
        return "diagnostic result"
    if "legacy" in lowered or "phase_6clean" in lowered:
        return "legacy or cleanup artifact"
    if "metrics" in lowered or "summary" in lowered:
        return "final or extension result"
    return "intermediate result"


def build_report(
    env: pd.DataFrame,
    tests: pd.DataFrame,
    test_results: pd.DataFrame,
    hash_validation: pd.DataFrame,
    registry: pd.DataFrame,
    conflicts: pd.DataFrame,
    authoritative: pd.DataFrame,
    artifacts: pd.DataFrame,
) -> str:
    python_rows = env.loc[env["category"].eq("python")]
    py_status = "pass" if (python_rows["status"] == "pass").any() else "blocked"
    hash_failures = int(hash_validation["status"].ne("unchanged").sum()) if not hash_validation.empty else 0
    test_failures = int(test_results["status"].ne("pass").sum()) if not test_results.empty else 0
    decision = f2_decision(py_status, hash_failures, test_failures)
    lines = [
        "# Finalization F2 Runtime Verification",
        "",
        "## Executive Summary",
        "",
        f"- Decision: `{decision}`.",
        f"- Python gate: `{py_status}`.",
        f"- Safe test tiers executed: `{len(test_results)}`.",
        f"- Safe test tier failures: `{test_failures}`.",
        f"- Protected input hash failures: `{hash_failures}`.",
        f"- Canonical registry rows: `{len(registry)}`.",
        f"- Apparent result conflict groups: `{len(conflicts)}`.",
        "",
        "## Environment",
        "",
        "- Official interpreter candidate: `.venv/Scripts/python.exe`.",
        "- `.venv/pyvenv.cfg` records Python `3.12.10`; `environment.yml` records `python=3.10`.",
        "- The mismatch is documented as an environment divergence, not corrected in F2.",
        "",
        "## Test Execution",
        "",
        *render_test_result_lines(test_results),
        "",
        "## Protected Inputs",
        "",
        f"- Protected paths hashed before tests: `{hash_validation['relative_path'].nunique() if not hash_validation.empty else 0}`.",
        f"- All unchanged: `{str(hash_failures == 0).lower()}`.",
        "- Hash algorithm: `sha256`.",
        "",
        "## Canonical Registry And Conflicts",
        "",
        "- Registry fields include cohort, candidate set, evaluation policy and denominator policy.",
        "- MostPop ML-1M Plan B and Plan E values are classified as non-conflicting when the cohort differs.",
        "- A conflict is true only when the natural comparison fields match and the values differ.",
        "",
        "## Authoritative Sources",
        "",
        *render_authority_lines(authoritative),
        "",
        "## Worktree",
        "",
        f"- Classified worktree/artifact rows: `{len(artifacts)}`.",
        "- F2 did not stage, commit, checkout, reset, clean, download, call APIs, call Ollama, or train models.",
        "",
        "## Completion Criteria",
        "",
        "- Interpreter identified: yes if Python gate is pass.",
        "- Dependencies inventoried: yes.",
        "- LEVEL 0-2 safe tests: pass only if every tier return code is zero.",
        "- Protected inputs unchanged: pass only if every hash validation row is unchanged.",
        "- Existing outputs overwritten: no, enforced by F2 output guard.",
        "",
        "## Next Action",
        "",
        "Review the generated registry, conflicts and artifact classification before staging any F2 files.",
    ]
    return "\n".join(lines) + "\n"


def f2_decision(python_status: str, hash_failures: int, test_failures: int) -> str:
    if python_status != "pass":
        return "blocked_python_startup"
    if hash_failures:
        return "blocked_hash_changed"
    if test_failures:
        return "blocked_safe_tests_failed"
    return "finalization_f2_complete"


def render_test_result_lines(test_results: pd.DataFrame) -> list[str]:
    if test_results.empty:
        return ["- No tests were run by the builder."]
    return [
        f"- `{row['level']}`: status `{row['status']}`, return code `{row['return_code']}`, static tests `{row['static_test_count']}`."
        for row in test_results.to_dict("records")
    ]


def render_authority_lines(authoritative: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for row in authoritative.head(8).to_dict("records"):
        lines.append(f"- Rank `{row['authority_rank']}` `{row['source_file']}`: {row['preferred_use']}.")
    return lines


def write_outputs(root: Path, frames: Mapping[str, pd.DataFrame], report: str) -> None:
    for key, frame in frames.items():
        path = root / OUTPUT_TABLES[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, lineterminator="\n")
    report_path = root / REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")


def build_all(root: Path, *, config_path: Path | None = None, run_tests: bool = True) -> dict[str, Any]:
    ensure_no_existing_outputs(root)
    env = environment_inventory(root, config_path)
    protected = protected_input_paths(root)
    before = hash_table(root, protected, "finalization_f2_before")
    tests = test_inventory(root)
    test_results = run_safe_tests(root, tests) if run_tests else empty_test_results("not_run")
    after = hash_table(root, protected, "finalization_f2_after")
    validation = validate_hashes(before, after)
    registry = canonical_result_registry(root)
    conflicts = result_conflicts(registry)
    authoritative = authoritative_sources(root)
    artifacts = artifact_classification(root)
    report = build_report(env, tests, test_results, validation, registry, conflicts, authoritative, artifacts)
    frames = {
        "environment_inventory": env,
        "test_inventory": tests,
        "test_results": test_results,
        "hashes_before": before,
        "hashes_after": after,
        "hash_validation": validation,
        "result_conflicts": conflicts,
        "authoritative_sources": authoritative,
        "artifact_classification": artifacts,
        "canonical_registry": registry,
    }
    write_outputs(root, frames, report)
    return {
        "frames": frames,
        "report": report,
        "decision": f2_decision(
            "pass" if env.loc[env["check_name"].eq("venv_python_introspection"), "status"].eq("pass").any() else "blocked",
            int(validation["status"].ne("unchanged").sum()),
            int(test_results["status"].ne("pass").sum()) if not test_results.empty else 0,
        ),
    }


def empty_test_results(status: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "test_run_id": "F2-TEST-000",
                "level": "not_run",
                "command": "",
                "static_test_count": "0",
                "return_code": "",
                "duration_seconds": "0",
                "status": status,
                "stdout_tail": "",
                "stderr_tail": "",
                "notes": "safe tests were disabled by CLI flag",
            }
        ],
        columns=TEST_RESULT_COLUMNS,
    )


def validate_schema(frame: pd.DataFrame, columns: Sequence[str]) -> bool:
    return list(frame.columns) == list(columns)
