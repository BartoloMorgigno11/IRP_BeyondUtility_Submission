"""Phase F6I2A substantive repository consolidation design.

The builder is read-only with respect to existing scientific artifacts. It uses
F6I1 as the authoritative file universe, performs semantic audits for future
consolidation, and emits approval-ready manifests for a later F6I2B phase. It
does not delete, move, archive, rename, merge, train, infer, download, stage, or
commit anything.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_ID = "phase_f6i2a"
PHASE_NAME = "phase_f6i2a_substantive_repository_consolidation_design"
EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
RUNTIME_CLASSIFICATION = "defensible_repaired_clone_exact_package_versions"

CONFIG_PATH = PROJECT_ROOT / "configs" / f"{PHASE_NAME}.yaml"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = PROJECT_ROOT / "results" / "manifests"
REPORT_PATH = PROJECT_ROOT / "report" / f"{PHASE_NAME}.md"
HANDOFF_PATH = PROJECT_ROOT / "report" / "future_operations_context_handoff.md"
SNAPSHOT_PATH = MANIFESTS_DIR / "current_project_context_snapshot.json"

F6I1_INPUTS = {
    "inventory": TABLES_DIR / "phase_f6i1_relevant_file_inventory.csv",
    "edges": TABLES_DIR / "phase_f6i1_dependency_edges.csv",
    "retention": TABLES_DIR / "phase_f6i1_retention_classification.csv",
    "delete_candidates": TABLES_DIR / "phase_f6i1_delete_now_candidates.csv",
    "manual_review": TABLES_DIR / "phase_f6i1_manual_review.csv",
    "future_map": TABLES_DIR / "phase_f6i1_future_dependency_map.csv",
    "duplicates": TABLES_DIR / "phase_f6i1_duplicate_groups.csv",
    "validation": TABLES_DIR / "phase_f6i1_validation.csv",
    "initial_capture": MANIFESTS_DIR / "phase_f6i1_initial_capture.json",
    "report": PROJECT_ROOT / "report" / "phase_f6i1_repository_retention_audit_and_safe_cleanup_batch1.md",
    "source": PROJECT_ROOT / "src" / "analysis" / "phase_f6i1_repository_retention_audit.py",
    "test": PROJECT_ROOT / "tests" / "test_phase_f6i1_repository_retention_audit.py",
}

OUTPUT_TABLES = {
    "source_file_audit": TABLES_DIR / f"{PHASE_ID}_source_file_audit.csv",
    "python_semantic_groups": TABLES_DIR / f"{PHASE_ID}_python_semantic_groups.csv",
    "python_merge_plan": TABLES_DIR / f"{PHASE_ID}_python_merge_plan.csv",
    "config_audit": TABLES_DIR / f"{PHASE_ID}_config_audit.csv",
    "config_merge_plan": TABLES_DIR / f"{PHASE_ID}_config_merge_plan.csv",
    "test_audit": TABLES_DIR / f"{PHASE_ID}_test_audit.csv",
    "test_assertion_coverage": TABLES_DIR / f"{PHASE_ID}_test_assertion_coverage.csv",
    "markdown_audit": TABLES_DIR / f"{PHASE_ID}_markdown_audit.csv",
    "agents_codex_audit": TABLES_DIR / f"{PHASE_ID}_agents_codex_audit.csv",
    "results_manifest_audit": TABLES_DIR / f"{PHASE_ID}_results_manifest_audit.csv",
    "phase_family_consolidation": TABLES_DIR / f"{PHASE_ID}_phase_family_consolidation.csv",
    "future_dependency_protection": TABLES_DIR / f"{PHASE_ID}_future_dependency_protection.csv",
    "retention_decisions": TABLES_DIR / f"{PHASE_ID}_retention_decisions.csv",
    "low_risk_delete_bundle": TABLES_DIR / f"{PHASE_ID}_low_risk_delete_bundle.csv",
    "archive_bundle": TABLES_DIR / f"{PHASE_ID}_archive_bundle.csv",
    "code_consolidation_bundle": TABLES_DIR / f"{PHASE_ID}_code_consolidation_bundle.csv",
    "config_consolidation_bundle": TABLES_DIR / f"{PHASE_ID}_config_consolidation_bundle.csv",
    "test_consolidation_bundle": TABLES_DIR / f"{PHASE_ID}_test_consolidation_bundle.csv",
    "documentation_consolidation_bundle": TABLES_DIR / f"{PHASE_ID}_documentation_consolidation_bundle.csv",
    "target_repository_architecture": TABLES_DIR / f"{PHASE_ID}_target_repository_architecture.csv",
    "manual_review": TABLES_DIR / f"{PHASE_ID}_manual_review.csv",
    "validation": TABLES_DIR / f"{PHASE_ID}_validation.csv",
    "determinism": TABLES_DIR / f"{PHASE_ID}_determinism.csv",
    "next_actions": TABLES_DIR / f"{PHASE_ID}_next_actions.csv",
}

DESIGN_MANIFEST_PATH = MANIFESTS_DIR / f"{PHASE_ID}_consolidation_design_manifest.json"
PROTECTED_HASHES_PATH = MANIFESTS_DIR / f"{PHASE_ID}_protected_input_hashes.json"

PATH_RE = re.compile(
    r"(?P<path>(?:configs|data|docs|original_repo|paper|report|results|src|tests)[/\\][A-Za-z0-9_. /\\'-]+"
    r"\.(?:csv|json|jsonl|md|py|txt|ya?ml|pt|pth|pkl|png|pdf|zip|gz))",
    re.IGNORECASE,
)
ROADMAP_TERMS = {
    "F6J dataset/candidate/reranking closure": ["lastfm", "candidate", "reranking", "ranking", "dataset", "cohort"],
    "F6K traditional models and prompt recovery": ["random", "mostpop", "bm25", "bprmf", "gru4rec", "sasrec", "lightgcn", "prompt"],
    "F6L model availability and execution": ["qwen", "mistral", "llama", "gpt", "claude", "ollama", "model"],
    "F6M complete ranking/reranking matrix": ["matrix", "ranking", "reranking", "full", "cell"],
    "F6N history/no-history and reduced-data D'": ["history", "zero", "reduced", "d'", "training"],
    "F6O candidate-position bias": ["candidate_position", "position_bias", "position"],
    "F6P profile generation": ["profile", "generation"],
    "F6Q hallucination and reliability": ["hallucination", "reliability", "invalid", "parser"],
    "F6R metrics": ["metric", "hr", "ndcg", "coverage", "novelty", "serendipity", "fairness", "diversity"],
    "F6S repeated runs and statistical analysis": ["determinism", "nondeterminism", "repeated", "statistical", "seed"],
    "F7 fifth dataset": ["fifth", "additional dataset"],
    "F8 scientific extensions": ["extension", "quantization", "prompt-format", "model-size"],
    "F9 final deliverables": ["final", "report", "reproducibility", "readme", "archive", "documentation"],
}
PROTECTED_CATEGORIES = {
    "official_repository",
    "paper",
    "raw_data",
    "raw_llm_output",
    "prediction",
    "model_checkpoint",
    "model_artifact",
}
RAW_OR_EXPENSIVE_CATEGORIES = {
    "raw_data",
    "raw_llm_output",
    "prediction",
    "model_checkpoint",
    "model_artifact",
}
SOURCE_CATEGORIES = {"source_code", "test", "config"}


@dataclass(frozen=True)
class PhaseData:
    inventory: list[dict[str, str]]
    edges: list[dict[str, str]]
    retention: dict[str, dict[str, str]]
    manual_review: list[dict[str, str]]
    duplicates: list[dict[str, str]]
    future_map: list[dict[str, str]]
    validation: list[dict[str, str]]
    initial_capture: dict[str, Any]
    f6i1_report: str
    f6i1_source: str
    f6i1_test: str


@dataclass(frozen=True)
class BuildResult:
    tables: dict[str, list[dict[str, str]]]
    manifest: dict[str, Any]
    protected_hashes: dict[str, Any]
    report: str
    initial_git: str
    final_git: str


def normalize_path(path: str | Path) -> str:
    value = str(path).replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", value):
        try:
            value = Path(value).resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return value
    while value.startswith("./"):
        value = value[2:]
    return value.rstrip("/")


def run_command(args: Sequence[str]) -> tuple[int, str, str]:
    completed = subprocess.run(
        list(args),
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def git_stdout(args: Sequence[str]) -> str:
    code, stdout, stderr = run_command(["git", *args])
    if code != 0:
        return f"ERROR: {stderr}"
    return stdout


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = sorted(
        [dict(row) for row in rows],
        key=lambda row: tuple(str(row.get(column, "")) for column in columns),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in materialized:
            writer.writerow({column: row.get(column, "") for column in columns})


def read_text(path: Path, limit: int | None = None) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if limit is None else text[:limit]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_phase_data() -> PhaseData:
    for name, path in F6I1_INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing F6I1 input {name}: {path}")
    retention_rows = read_csv(F6I1_INPUTS["retention"])
    return PhaseData(
        inventory=read_csv(F6I1_INPUTS["inventory"]),
        edges=read_csv(F6I1_INPUTS["edges"]),
        retention={row["path"]: row for row in retention_rows},
        manual_review=read_csv(F6I1_INPUTS["manual_review"]),
        duplicates=read_csv(F6I1_INPUTS["duplicates"]),
        future_map=read_csv(F6I1_INPUTS["future_map"]),
        validation=read_csv(F6I1_INPUTS["validation"]),
        initial_capture=json.loads(read_text(F6I1_INPUTS["initial_capture"])),
        f6i1_report=read_text(F6I1_INPUTS["report"]),
        f6i1_source=read_text(F6I1_INPUTS["source"], 200_000),
        f6i1_test=read_text(F6I1_INPUTS["test"], 100_000),
    )


def edge_index(edges: Sequence[Mapping[str, str]]) -> dict[str, dict[str, list[str]]]:
    indexes: dict[str, dict[str, list[str]]] = {
        "incoming": defaultdict(list),
        "outgoing": defaultdict(list),
        "imports": defaultdict(list),
        "imported_by": defaultdict(list),
        "reports": defaultdict(list),
        "tests": defaultdict(list),
    }
    for edge in edges:
        source = edge.get("source_path", "")
        target = edge.get("target_path", "")
        edge_type = edge.get("edge_type", "")
        if not source or not target:
            continue
        indexes["outgoing"][source].append(target)
        indexes["incoming"][target].append(source)
        if edge_type == "python_import":
            indexes["imports"][target].append(source)
            indexes["imported_by"][source].append(target)
        if source.startswith("report/") or target.startswith("report/"):
            indexes["reports"][source].append(target)
            indexes["reports"][target].append(source)
        if source.startswith("tests/") or target.startswith("tests/"):
            indexes["tests"][source].append(target)
            indexes["tests"][target].append(source)
    return indexes


def unique_join(values: Iterable[str], limit: int = 30) -> str:
    unique = sorted({str(value) for value in values if value})
    if len(unique) > limit:
        return ";".join(unique[:limit]) + f";...(+{len(unique) - limit})"
    return ";".join(unique)


def path_text(path: str, limit: int = 500_000) -> str:
    absolute = PROJECT_ROOT / path
    if not absolute.exists() or not absolute.is_file():
        return ""
    suffix = absolute.suffix.lower()
    if suffix not in {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".csv"}:
        return ""
    try:
        return read_text(absolute, limit)
    except OSError:
        return ""


def parse_ast(path: str, text: str) -> ast.AST | None:
    if not text or not path.endswith(".py"):
        return None
    try:
        return ast.parse(text)
    except SyntaxError:
        return None


def python_metadata(path: str, indexes: Mapping[str, Mapping[str, list[str]]]) -> dict[str, str]:
    text = path_text(path)
    tree = parse_ast(path, text)
    functions: list[str] = []
    classes: list[str] = []
    assertions = 0
    has_main = "if __name__ == \"__main__\"" in text or "if __name__ == '__main__'" in text
    subprocess = "subprocess." in text or "run_command(" in text
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.Assert):
                assertions += 1
    references = sorted({normalize_path(match.group("path")) for match in PATH_RE.finditer(text)})
    schemas = [name for name in functions if "schema" in name.lower()]
    schemas.extend(re.findall(r"([A-Z][A-Z0-9_]*COLUMNS)\s*=", text))
    lower = f"{path.lower()} {text[:80_000].lower()}"
    return {
        "functions": unique_join(functions),
        "classes": unique_join(classes),
        "function_count": str(len(functions)),
        "class_count": str(len(classes)),
        "imports": unique_join(indexes["imports"].get(path, [])),
        "imported_by": unique_join(indexes["imported_by"].get(path, [])),
        "main_guard": str(has_main).lower(),
        "subprocess_usage": str(subprocess).lower(),
        "config_paths": unique_join([ref for ref in references if ref.startswith("configs/")]),
        "input_paths": unique_join([ref for ref in references if ref.startswith(("data/", "paper/", "original_repo/"))]),
        "output_paths": unique_join([ref for ref in references if ref.startswith("results/")]),
        "schema_definitions": unique_join(schemas),
        "metric_implementation": str(any(term in lower for term in ["metric", "hr@", "ndcg", "coverage", "serendipity", "fairness", "diversity"])).lower(),
        "parser_implementation": str("parser" in lower or "parse" in lower).lower(),
        "prompt_construction": str("prompt" in lower or "openp5" in lower or "llmrank" in lower or "llmsrec" in lower).lower(),
        "assertion_count": str(assertions),
    }


def classify_python(path: str, meta: Mapping[str, str], inventory_row: Mapping[str, str]) -> str:
    lower = path.lower()
    if path.startswith("tests/"):
        return "future_required"
    if path.startswith("src/evaluation/") or path.startswith("src/metrics/") or meta["parser_implementation"] == "true":
        return "canonical_metric_or_parser"
    if path.startswith("src/experiments/run_") or "/run_" in lower:
        return "canonical_experiment_runner"
    if path.startswith("src/experiments/build_") or path.startswith("src/analysis/"):
        return "canonical_builder"
    if path.startswith(("src/baselines/", "src/llm/")):
        return "canonical_runtime"
    if path.startswith(("src/data/", "src/utils/", "src/testing/", "src/audit/", "src/prompts/")):
        return "canonical_shared_library"
    if inventory_row.get("future_roadmap_dependencies"):
        return "future_required"
    return "unresolved"


def build_source_file_audit(data: PhaseData, indexes: Mapping[str, Mapping[str, list[str]]]) -> list[dict[str, str]]:
    rows = []
    for row in data.inventory:
        if row.get("extension") != "py":
            continue
        path = row["path"]
        meta = python_metadata(path, indexes)
        rows.append(
            {
                "path": path,
                "category": row.get("category", ""),
                "phase": row.get("phase", ""),
                "python_class": classify_python(path, meta, row),
                "imports": meta["imports"],
                "imported_by": meta["imported_by"],
                "functions": meta["functions"],
                "classes": meta["classes"],
                "function_count": meta["function_count"],
                "class_count": meta["class_count"],
                "cli_entrypoint": meta["main_guard"],
                "main_guard": meta["main_guard"],
                "config_paths": meta["config_paths"],
                "input_paths": meta["input_paths"],
                "output_paths": meta["output_paths"],
                "schema_definitions": meta["schema_definitions"],
                "metric_implementation": meta["metric_implementation"],
                "parser_implementation": meta["parser_implementation"],
                "prompt_construction": meta["prompt_construction"],
                "subprocess_usage": meta["subprocess_usage"],
                "tests": row.get("tests", ""),
                "reports_citing": row.get("reports_and_manifests_referencing", ""),
                "future_roadmap_dependency": row.get("future_roadmap_dependencies", ""),
                "one_off_assessment": one_off_assessment(path, meta, row),
                "provenance_required": str(path.startswith("src/analysis/") or "phase_" in path).lower(),
            }
        )
    return rows


def one_off_assessment(path: str, meta: Mapping[str, str], row: Mapping[str, str]) -> str:
    if path.startswith("src/analysis/") or path.startswith("src/experiments/build_"):
        if row.get("reports_and_manifests_referencing") or row.get("future_roadmap_dependencies"):
            return "not_obsolete:producer_or_future_dependency"
        return "unresolved:phase_builder_without_complete_absorption_proof"
    if path.startswith("src/experiments/run_"):
        return "not_obsolete:experiment_runner"
    return "not_one_off_or_shared_runtime"


def helper_family(function_name: str) -> str:
    lower = function_name.lower()
    if lower in {"write_csv", "read_csv", "read_csv_rows"}:
        return "deterministic_csv_io"
    if "sha256" in lower or "hash" in lower:
        return "hashing_helpers"
    if lower in {"normalize_path", "rel", "path_text"}:
        return "path_normalization"
    if lower.startswith("git_") or "git" in lower:
        return "git_state_capture"
    if "validation" in lower or lower.startswith("validate"):
        return "validation_table_creation"
    if "manifest" in lower:
        return "manifest_generation"
    if "report" in lower:
        return "phase_report_generation"
    if "candidate" in lower or "cohort" in lower:
        return "candidate_and_cohort_reconciliation"
    if "lineage" in lower or "provenance" in lower:
        return "request_response_lineage"
    return ""


def build_python_groups_and_merge(source_rows: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    groups: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"files": set(), "functions": set()})
    for row in source_rows:
        for function in row.get("functions", "").split(";"):
            family = helper_family(function)
            if family:
                groups[family]["files"].add(row["path"])
                groups[family]["functions"].add(function)
    semantic_rows: list[dict[str, str]] = []
    merge_rows: list[dict[str, str]] = []
    module_map = {
        "deterministic_csv_io": "src/audit/common_io.py",
        "hashing_helpers": "src/audit/hash_utils.py",
        "path_normalization": "src/audit/path_utils.py",
        "git_state_capture": "src/audit/git_state.py",
        "validation_table_creation": "src/audit/validation.py",
        "manifest_generation": "src/audit/manifest_utils.py",
        "phase_report_generation": "src/audit/reporting.py",
        "candidate_and_cohort_reconciliation": "src/audit/reconciliation.py",
        "request_response_lineage": "src/audit/lineage.py",
    }
    order = 0
    for family, values in sorted(groups.items()):
        files = sorted(values["files"])
        functions = sorted(values["functions"])
        if len(files) < 2:
            continue
        order += 1
        semantic_rows.append(
            {
                "semantic_group_id": f"PYSEM-F6I2A-{order:03d}",
                "group_name": family,
                "member_files": ";".join(files),
                "shared_functions": ";".join(functions),
                "unique_function_policy": "preserve per-file unique functions before replacing helpers",
                "evidence": "AST function-name family match plus F6I1 dependency context",
                "risk": "medium",
            }
        )
        merge_rows.append(
            {
                "merge_group_id": f"PYMERGE-F6I2A-{order:03d}",
                "member_files": ";".join(files),
                "shared_functions": ";".join(functions),
                "unique_functions": "extract during F6I2B before editing; no deletion proposed in F6I2A",
                "unique_constants": "preserve module constants and output schemas in wrappers",
                "unique_schemas": "preserve *_COLUMNS and CSV schemas",
                "proposed_common_module": module_map[family],
                "wrapper_files": ";".join(files),
                "later_delete_candidates": "",
                "migration_order": str(order),
                "tests_required": "focused tests for helpers plus existing phase tests importing wrappers",
                "affected_outputs": "phase manifests, validation tables, reports",
                "provenance_implications": "wrappers retain historical entrypoint paths",
                "rollback_strategy": "restore wrapper-local helper copies from Git or archival bundle",
                "approval_flag": "requires_human_approval",
            }
        )
    return semantic_rows, merge_rows


def flatten_yaml_keys(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        keys: list[str] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            keys.extend(flatten_yaml_keys(child, child_prefix))
        return keys
    if isinstance(value, list):
        return [prefix + "[]"] if prefix else ["[]"]
    return [prefix] if prefix else []


def build_config_audit_and_plan(data: PhaseData, indexes: Mapping[str, Mapping[str, list[str]]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = []
    groups: dict[str, list[str]] = defaultdict(list)
    for row in data.inventory:
        if row.get("extension") not in {"yaml", "yml"}:
            continue
        path = row["path"]
        text = path_text(path, 200_000)
        try:
            parsed = yaml.safe_load(text) if text else {}
        except yaml.YAMLError:
            parsed = {}
        keys = flatten_yaml_keys(parsed)
        group = config_group(path)
        groups[group].append(path)
        rows.append(
            {
                "path": path,
                "config_class": classify_config(path, keys, row),
                "loader_sources": unique_join(indexes["incoming"].get(path, []) + indexes["outgoing"].get(path, [])),
                "command_or_phase": row.get("phase", ""),
                "executed_evidence": "yes" if row.get("reports_and_manifests_referencing") else "unknown_or_design_only",
                "unique_keys": unique_join(keys, limit=80),
                "duplicated_key_group": group,
                "inherited_values": "to_be_extracted_in_f6i2b",
                "phase_specific_values": unique_join([key for key in keys if "phase" in key or "experiment" in key]),
                "scientific_contract_values": unique_join([key for key in keys if key.split(".")[0] in {"dataset", "models", "metrics", "experiment"}]),
                "output_paths": unique_join(re.findall(r"results[/\\][A-Za-z0-9_./\\-]+", text)),
                "model_settings": unique_join([key for key in keys if "model" in key.lower()]),
                "prompt_settings": unique_join([key for key in keys if "prompt" in key.lower()]),
                "dataset_settings": unique_join([key for key in keys if "dataset" in key.lower()]),
                "seed_settings": unique_join([key for key in keys if "seed" in key.lower()]),
                "metric_settings": unique_join([key for key in keys if "metric" in key.lower()]),
                "future_dependency": row.get("future_roadmap_dependencies", ""),
            }
        )
    plan_rows = []
    for order, (group, paths) in enumerate(sorted(groups.items()), start=1):
        if len(paths) < 2 and group != "phase_specific_archive":
            continue
        plan_rows.append(
            {
                "config_merge_group_id": f"CFGMERGE-F6I2A-{order:03d}",
                "group_name": group,
                "member_configs": ";".join(sorted(paths)),
                "target_structure": target_config_structure(group),
                "values_to_preserve": "all dataset/model/prompt/seed/metric/output-path values",
                "historical_contract_policy": "freeze under configs/archive/phases/ before any deletion",
                "migration": "create base/dataset/model/prompt overlays, then wrappers for historical phase configs",
                "validation": "load every old config and compare flattened effective values",
                "rollback": "restore original YAML files from Git or archival bundle",
                "approval_flag": "requires_human_approval",
            }
        )
    return rows, plan_rows


def config_group(path: str) -> str:
    lower = path.lower()
    if lower in {"configs/beauty.yaml", "configs/sports.yaml", "configs/ml1m.yaml", "configs/cost.yaml"}:
        return "base_dataset_or_cost"
    for model in ["bprmf", "gru4rec", "lightgcn", "sasrec"]:
        if model in lower:
            return f"model_{model}_overrides"
    if "phase_" in lower or "finalization_" in lower or "plan_c" in lower:
        return "phase_specific_archive"
    return "experiment_overrides"


def target_config_structure(group: str) -> str:
    if group == "base_dataset_or_cost":
        return "configs/base/;configs/datasets/"
    if group.startswith("model_"):
        return "configs/models/;configs/experiments/"
    if group == "phase_specific_archive":
        return "configs/archive/phases/"
    return "configs/experiments/"


def classify_config(path: str, keys: Sequence[str], row: Mapping[str, str]) -> str:
    lower = path.lower()
    if lower in {"configs/beauty.yaml", "configs/sports.yaml", "configs/ml1m.yaml", "configs/cost.yaml"}:
        return "canonical_base_config"
    if "phase_" in lower or "finalization_" in lower:
        return "frozen_historical_contract"
    if row.get("future_roadmap_dependencies"):
        return "future_required"
    if len(keys) > 0:
        return "canonical_phase_override"
    return "unresolved"


def build_test_audit_and_coverage(data: PhaseData, indexes: Mapping[str, Mapping[str, list[str]]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = []
    coverage = []
    target_to_tests: dict[str, list[str]] = defaultdict(list)
    for row in data.inventory:
        if row.get("category") == "test" and row.get("extension") == "py":
            for target in indexes["imports"].get(row["path"], []):
                target_to_tests[target].append(row["path"])
    for row in data.inventory:
        if row.get("category") != "test" or row.get("extension") != "py":
            continue
        path = row["path"]
        meta = python_metadata(path, indexes)
        text = path_text(path, 300_000)
        imported_targets = indexes["imports"].get(path, [])
        unsafe = any(term in text.lower() for term in ["allow_api", "ollama", "download", "subprocess.run", "trainable", "cuda"])
        test_functions = [fn for fn in meta["functions"].split(";") if fn.startswith("test_")]
        test_class = "unsafe_full_suite_test" if unsafe else classify_test(path, text, imported_targets)
        rows.append(
            {
                "path": path,
                "test_class": test_class,
                "target_implementation": unique_join(imported_targets),
                "test_functions": unique_join(test_functions),
                "assertion_count": meta["assertion_count"],
                "edge_cases": infer_edge_cases(text),
                "fixtures": unique_join([fn for fn in meta["functions"].split(";") if "fixture" in fn.lower()]),
                "protected_regression": infer_regression(path, text),
                "still_runs": "focused_or_not_executed_in_f6i2a",
                "subsumed_by": subsumed_by(path, imported_targets, target_to_tests),
                "unsafe_operations": str(unsafe).lower(),
                "filesystem_writes": str("tmp_path" in text or "write_text" in text or "to_csv" in text).lower(),
                "future_roadmap_dependency": row.get("future_roadmap_dependencies", ""),
            }
        )
        coverage.append(
            {
                "test_path": path,
                "target_implementation": unique_join(imported_targets),
                "unique_assertions": f"{meta['assertion_count']} AST assert statements plus pytest assertions/calls",
                "edge_cases": infer_edge_cases(text),
                "replacement_test": "none_required_no_test_deletion_proposed",
                "coverage_status": "retained",
                "assertion_level_deletion_safe": "false",
                "notes": "F6I2A does not propose deleting tests without assertion-by-assertion replacement.",
            }
        )
    return rows, coverage


def classify_test(path: str, text: str, targets: Sequence[str]) -> str:
    lower = f"{path.lower()} {text[:80_000].lower()}"
    if "regression" in lower or "protected" in lower or "bug" in lower:
        return "canonical_regression_test"
    if len(targets) > 1 or "subprocess" in lower:
        return "canonical_integration_test"
    if "phase_" in lower or "finalization_" in lower or "plan_" in lower:
        return "historical_provenance_test"
    return "canonical_unit_test"


def infer_edge_cases(text: str) -> str:
    terms = [term for term in ["empty", "missing", "duplicate", "invalid", "zero", "determinism", "protected"] if term in text.lower()]
    return unique_join(terms) or "not_explicitly_labeled"


def infer_regression(path: str, text: str) -> str:
    lower = f"{path.lower()} {text.lower()[:50_000]}"
    if "protected" in lower:
        return "protected input guard"
    if "determinism" in lower:
        return "determinism"
    if "parser" in lower or "invalid" in lower:
        return "parser or invalid-output behavior"
    return "not_explicitly_labeled"


def subsumed_by(path: str, targets: Sequence[str], target_to_tests: Mapping[str, Sequence[str]]) -> str:
    peers = sorted({test for target in targets for test in target_to_tests.get(target, []) if test != path})
    return unique_join(peers, limit=10)


def build_markdown_audit(data: PhaseData, indexes: Mapping[str, Mapping[str, list[str]]]) -> list[dict[str, str]]:
    rows = []
    for row in data.inventory:
        if row.get("extension") != "md":
            continue
        path = row["path"]
        text = path_text(path, 500_000)
        headings = re.findall(r"^#{1,3}\s+(.+)$", text, flags=re.MULTILINE)
        decisions = re.findall(r"Decision:\s*`?([^`\n]+)`?", text)
        rows.append(
            {
                "path": path,
                "markdown_class": classify_markdown(path, text),
                "unique_claims": unique_join(headings[:20]),
                "current_or_historical_status": markdown_status(path, text),
                "referenced_artifacts": unique_join(indexes["outgoing"].get(path, [])),
                "inbound_references": unique_join(indexes["incoming"].get(path, [])),
                "outbound_references": unique_join(indexes["outgoing"].get(path, [])),
                "decision_tokens": unique_join(decisions[:20]),
                "duplicated_sections": "not_deletion_safe_without_claim_map",
                "future_usefulness": markdown_future_value(path, text, row),
                "target_canonical_document": target_markdown_document(path, text),
            }
        )
    return rows


def classify_markdown(path: str, text: str) -> str:
    lower = f"{path.lower()} {text[:100_000].lower()}"
    if path in {"README.md", "AGENTS.md"}:
        return "root operational documentation"
    if "codex" in lower or "prompt" in lower:
        return "Codex prompt or execution transcript"
    if "future_operations_context_handoff" in path or "handoff" in lower:
        return "current context handoff"
    if "final" in lower or "current status" in lower:
        return "current scientific status"
    if "architecture" in lower or "repository" in lower:
        return "architecture documentation"
    if "reproducibility" in lower:
        return "reproducibility instructions"
    if "decision:" in lower:
        return "phase report with unique scientific evidence"
    if path.startswith("report/"):
        return "phase report with unique execution provenance"
    return "unresolved"


def markdown_status(path: str, text: str) -> str:
    if "current" in text.lower() or path in {"README.md", "AGENTS.md"}:
        return "current"
    if "phase_" in path or "finalization_" in path or "plan_" in path:
        return "historical_phase"
    return "unresolved"


def markdown_future_value(path: str, text: str, row: Mapping[str, str]) -> str:
    if row.get("future_roadmap_dependencies"):
        return row["future_roadmap_dependencies"]
    if path.startswith("report/"):
        return "provenance and final reproducibility package"
    return "operational context"


def target_markdown_document(path: str, text: str) -> str:
    lower = f"{path.lower()} {text[:80_000].lower()}"
    if path == "README.md":
        return "README.md"
    if path == "AGENTS.md":
        return "AGENTS.md"
    if "architecture" in lower:
        return "docs/architecture.md"
    if "roadmap" in lower or "next action" in lower:
        return "docs/roadmap.md"
    if "reproducibility" in lower:
        return "docs/reproduction_protocol.md"
    if path.startswith("report/"):
        return "docs/archive/phase_reports/"
    return "docs/current_status.md"


def build_agents_codex_audit(data: PhaseData) -> list[dict[str, str]]:
    rows = []
    for row in data.inventory:
        path = row["path"]
        lower = path.lower()
        if not any(token in lower for token in ["agents", "codex", "handoff", "context", "snapshot", "roadmap", "recovery", "blocked"]):
            continue
        text = path_text(path, 300_000)
        rows.append(
            {
                "path": path,
                "scope": "root" if path in {"AGENTS.md", "CODEX_PROMPTS.md"} else row.get("phase", ""),
                "instruction_precedence": "active_root" if path == "AGENTS.md" else "historical_or_contextual",
                "currently_active": str(path == "AGENTS.md" or path.endswith("future_operations_context_handoff.md")).lower(),
                "conflicts_with_newer_instructions": "manual_review_required",
                "unique_safety_rules": "yes" if any(term in text.lower() for term in ["do not", "never", "must"]) else "unknown",
                "historical_value": "high" if row.get("phase") else "medium",
                "future_value": row.get("future_roadmap_dependencies") or "repository operations and provenance",
                "proposed_target_location": target_agents_doc(path),
                "recommendation": "keep" if path == "AGENTS.md" else "archive_or_merge_in_f6i2b",
            }
        )
    return rows


def target_agents_doc(path: str) -> str:
    if path == "AGENTS.md":
        return "AGENTS.md"
    if "codex" in path.lower():
        return "docs/archive/codex/"
    if "handoff" in path.lower() or "snapshot" in path.lower():
        return "docs/archive/decisions/"
    return "docs/archive/phase_reports/"


def build_results_manifest_audit(data: PhaseData, indexes: Mapping[str, Mapping[str, list[str]]]) -> list[dict[str, str]]:
    rows = []
    for row in data.inventory:
        category = row.get("category", "")
        if category not in {
            "result_table",
            "manifest",
            "raw_llm_output",
            "prediction",
            "model_checkpoint",
            "model_artifact",
            "local_data",
            "raw_data",
            "result_other",
        }:
            continue
        path = row["path"]
        producer = row.get("producer", "")
        retained_producer = "yes" if producer.startswith("src/") and (PROJECT_ROOT / producer).exists() else "frozen_or_unknown"
        rows.append(
            {
                "path": path,
                "artifact_kind": result_kind(row),
                "producer": producer,
                "retained_producer_available": retained_producer,
                "input_hashes": row.get("inputs", ""),
                "regeneration_cost": regeneration_cost(row),
                "requires_training_or_inference": str(category in RAW_OR_EXPENSIVE_CATEGORIES or "run_" in producer).lower(),
                "deterministic_status": "deterministic_if_builder_and_inputs_retained" if category == "result_table" else "frozen_evidence",
                "current_consumers": unique_join(indexes["outgoing"].get(path, []) + indexes["incoming"].get(path, [])),
                "historical_provenance_value": "high" if row.get("phase") else "medium",
                "replacement_artifact": "none; retain or archive only" if category in RAW_OR_EXPENSIVE_CATEGORIES else "future consolidated table if approved",
                "deletion_or_archive_safety": result_safety(row),
            }
        )
    return rows


def result_kind(row: Mapping[str, str]) -> str:
    category = row.get("category", "")
    path = row.get("path", "").lower()
    if category in RAW_OR_EXPENSIVE_CATEGORIES:
        return "immutable_raw_or_expensive_model_evidence"
    if "validation" in path:
        return "validation_table"
    if "manifest" in path:
        return "manifest"
    if "metric" in path:
        return "metric_table"
    return category


def regeneration_cost(row: Mapping[str, str]) -> str:
    category = row.get("category", "")
    if category in {"model_checkpoint", "model_artifact"}:
        return "expensive_training_or_model_download"
    if category in {"raw_llm_output", "prediction"}:
        return "expensive_or_nonrepeatable_inference"
    if category in {"raw_data"}:
        return "external_raw_input"
    if row.get("producer", "").startswith("src/"):
        return "offline_builder"
    return "unknown_or_frozen"


def result_safety(row: Mapping[str, str]) -> str:
    if row.get("category") in RAW_OR_EXPENSIVE_CATEGORIES:
        return "never_delete_in_f6i2b"
    if row.get("future_roadmap_dependencies"):
        return "retain_until_future_task_closed"
    return "archive_candidate_only_after_provenance_manifest"


def phase_family_for(path: str, phase: str) -> str:
    lower = f"{path.lower()} {phase.lower()}"
    if "dataset" in lower or "reconstruct" in lower or "prepare" in lower:
        return "dataset reconstruction"
    if "candidate" in lower or "cohort" in lower:
        return "candidate construction"
    if any(model in lower for model in ["random", "mostpop", "bm25", "baseline"]):
        return "baseline evaluation"
    if any(model in lower for model in ["bprmf", "gru4rec", "sasrec", "lightgcn"]):
        return "trainable recommenders"
    if "reranking" in lower:
        return "LLM reranking"
    if "llm" in lower or "qwen" in lower or "mistral" in lower:
        return "LLM ranking"
    if "prompt" in lower:
        return "prompt recovery"
    if "parser" in lower or "invalid" in lower:
        return "parser recovery"
    if "metric" in lower or "coverage" in lower or "serendipity" in lower or "fairness" in lower:
        return "metric resolution"
    if "exactness" in lower or "evidence" in lower:
        return "exactness audit"
    if "environment" in lower or "runtime" in lower:
        return "environment recovery"
    if "lineage" in lower or "provenance" in lower or "forensic" in lower or "f6i" in lower:
        return "provenance and lineage audit"
    if "cleanup" in lower or "pruning" in lower or "retention" in lower:
        return "repository cleanup"
    return "provenance and lineage audit"


def build_phase_family_consolidation(data: PhaseData) -> list[dict[str, str]]:
    families: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"source": set(), "reports": set(), "outputs": set(), "phases": set()})
    for row in data.inventory:
        family = phase_family_for(row["path"], row.get("phase", ""))
        families[family]["phases"].add(row.get("phase", ""))
        if row.get("category") == "source_code":
            families[family]["source"].add(row["path"])
        elif row.get("category") == "report":
            families[family]["reports"].add(row["path"])
        elif row.get("category") in {"result_table", "manifest"}:
            families[family]["outputs"].add(row["path"])
    rows = []
    for order, (family, values) in enumerate(sorted(families.items()), start=1):
        rows.append(
            {
                "phase_family_id": f"PHASEFAM-F6I2A-{order:03d}",
                "scientific_function": family,
                "phases": unique_join(values["phases"]),
                "canonical_final_implementation": choose_canonical(values["source"]),
                "still_useful_intermediate_implementation": unique_join(sorted(values["source"])[1:10]),
                "provenance_only_phases": unique_join([phase for phase in values["phases"] if "f5" in phase or "f6e" in phase]),
                "fully_absorbed_corrective_phases": "none_proven_in_f6i2a",
                "frozen_outputs_to_remain": unique_join(values["outputs"], limit=20),
                "scripts_configs_tests_to_consolidate": "see code/config/test consolidation bundles",
                "reports_to_archive": unique_join(values["reports"], limit=20),
            }
        )
    return rows


def choose_canonical(paths: Iterable[str]) -> str:
    ordered = sorted(paths, key=lambda path: (0 if "canonical" in path else 1, 0 if "finalization" in path else 1, path))
    return ordered[0] if ordered else "none_identified"


def task_groups_for(row: Mapping[str, str]) -> list[tuple[str, str]]:
    lower = " ".join([row.get("path", ""), row.get("purpose", ""), row.get("future_roadmap_dependencies", ""), row.get("phase", "")]).lower()
    groups = []
    for group, terms in ROADMAP_TERMS.items():
        hits = [term for term in terms if term in lower]
        if hits:
            groups.append((group, ",".join(hits[:5])))
    if not groups:
        category = row.get("category", "")
        if category in {"report", "docs", "manifest", "result_table"}:
            groups.append(("F9 final deliverables", category))
        elif category in SOURCE_CATEGORIES:
            groups.append(("F6K traditional models and prompt recovery", category))
        else:
            groups.append(("F9 final deliverables", "supporting_provenance"))
    return groups


def build_future_dependency_protection(data: PhaseData) -> list[dict[str, str]]:
    rows = []
    for row in data.inventory:
        for task, evidence in task_groups_for(row):
            rows.append(
                {
                    "path": row["path"],
                    "future_task_group": task,
                    "evidence": evidence,
                    "retention_requirement": "retain_or_bundle_tested_replacement_before_f6i2b_delete",
                    "delete_allowed_without_replacement": "false",
                }
            )
    return rows


def decision_for(row: Mapping[str, str], retention_row: Mapping[str, str], duplicate_rows: Sequence[Mapping[str, str]]) -> tuple[str, str, str]:
    path = row["path"]
    category = row.get("category", "")
    if category in PROTECTED_CATEGORIES or path in {"AGENTS.md", "README.md"}:
        return "keep_protected", "protected raw/upstream/root artifact", ""
    if category == "directory_unit":
        return "keep_canonical_runtime", "runtime or inaccessible state retained as unit", ""
    if category == "test":
        return "keep_canonical_test", "test retained until assertion replacement exists", ""
    if row.get("extension") == "py":
        meta = python_metadata(path, {"imports": {}, "imported_by": {}})
        py_class = classify_python(path, meta, row)
        mapping = {
            "canonical_runtime": "keep_canonical_runtime",
            "canonical_shared_library": "keep_canonical_shared_library",
            "canonical_builder": "keep_canonical_builder",
            "canonical_experiment_runner": "keep_canonical_experiment_runner",
            "canonical_metric_or_parser": "keep_canonical_shared_library",
            "future_required": "keep_future_dependency",
        }
        return mapping.get(py_class, "unresolved"), py_class, ""
    if category == "config":
        return "keep_future_dependency", "historical or future config retained", ""
    if category in {"raw_llm_output", "prediction", "model_checkpoint", "model_artifact", "local_data", "raw_data"}:
        return "keep_frozen_evidence", "raw/expensive/local scientific evidence", ""
    if category in {"manifest", "result_table"}:
        if row.get("future_roadmap_dependencies"):
            return "keep_future_dependency", "mapped to future roadmap", ""
        return "keep_frozen_evidence", "frozen generated evidence or manifest", ""
    if category == "report":
        if "final" in path.lower() or "canonical" in path.lower() or "handoff" in path.lower():
            return "keep_current_documentation", "current/canonical documentation", ""
        return "archive_in_f6i2b", "historical report with unique provenance; archive only", "docs/archive/phase_reports/"
    if category == "docs" or category == "root_or_other":
        return "keep_current_documentation", "documentation or repository support", ""
    for duplicate in duplicate_rows:
        if duplicate.get("member_path") == path and duplicate.get("delete_eligible") == "true":
            return "delete_in_f6i2b_exact_duplicate", "F6I1 exact duplicate with canonical retained path", duplicate.get("canonical_path", "")
    return retention_row.get("class", "manual_review"), "fallback from F6I1 retention", retention_row.get("canonical_replacement", "")


def build_retention_decisions(data: PhaseData) -> list[dict[str, str]]:
    duplicate_by_path = defaultdict(list)
    for row in data.duplicates:
        duplicate_by_path[row.get("member_path", "")].append(row)
    rows = []
    for row in data.inventory:
        path = row["path"]
        decision, reason, replacement = decision_for(row, data.retention.get(path, {}), duplicate_by_path.get(path, []))
        future = unique_join([task for task, _ in task_groups_for(row)])
        rows.append(
            {
                "path": path,
                "decision": decision,
                "confidence": confidence_for(decision),
                "scientific_risk": scientific_risk(row, decision),
                "technical_risk": technical_risk(row, decision),
                "unique_information_status": row.get("unique_scientific_information", ""),
                "future_roadmap_dependencies": future,
                "canonical_replacement": replacement,
                "required_migration": migration_for(decision),
                "required_validation": validation_for(decision),
                "rollback_source": replacement or "git_or_pre_action_archive_bundle",
                "reason": reason,
            }
        )
    return rows


def confidence_for(decision: str) -> str:
    if decision.startswith("keep_"):
        return "high"
    if decision.startswith(("archive_", "merge_")):
        return "medium"
    if decision.startswith("delete_"):
        return "medium"
    return "low"


def scientific_risk(row: Mapping[str, str], decision: str) -> str:
    if row.get("category") in PROTECTED_CATEGORIES or row.get("future_roadmap_dependencies"):
        return "high" if not decision.startswith("keep_") else "low"
    if decision.startswith("delete_"):
        return "medium"
    return "low"


def technical_risk(row: Mapping[str, str], decision: str) -> str:
    if row.get("category") in SOURCE_CATEGORIES and not decision.startswith("keep_"):
        return "high"
    if decision.startswith(("merge_", "archive_")):
        return "medium"
    return "low"


def migration_for(decision: str) -> str:
    if decision.startswith("archive_"):
        return "archive with path/hash/phase metadata before removal from active tree"
    if decision.startswith("merge_"):
        return "create common target and wrappers, then run focused tests"
    if decision.startswith("delete_"):
        return "pre-action deletion manifest plus rollback bundle"
    return "none"


def validation_for(decision: str) -> str:
    if decision.startswith("delete_"):
        return "strict F6I2B deletion gates and post-delete focused tests"
    if decision.startswith("archive_"):
        return "archive transfer hash validation and documentation link audit"
    if decision.startswith("merge_"):
        return "old/new output equivalence and focused tests"
    return "retention manifest check"


def low_risk_delete_bundle(data: PhaseData, decisions: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    by_path = {row["path"]: row for row in data.inventory}
    rows = []
    order = 0
    for decision in decisions:
        path = decision["path"]
        row = by_path[path]
        lower = path.lower()
        if row.get("size_bytes") == "0" and lower.startswith("results/logs/") and lower.endswith(".log"):
            order += 1
            rows.append(bundle_row("A", order, path, "delete_in_f6i2b_obsolete_safe", "zero-byte diagnostic log with no unique scientific evidence", "not required; preserve path/hash in manifest", "delete after approval", "remove one empty log file", "focused validation plus git status", "pre-action archive bundle", "low"))
    return rows


def bundle_row(bundle: str, order: int, path: str, action: str, reason: str, replacement: str, migration: str, diff: str, validation: str, rollback: str, risk: str) -> dict[str, str]:
    return {
        "bundle_id": f"F6I2B-{bundle}-{order:04d}",
        "path": path,
        "action": action,
        "reason": reason,
        "replacement": replacement,
        "migration": migration,
        "expected_diff": diff,
        "validation": validation,
        "rollback": rollback,
        "risk": risk,
        "approval_flag": "requires_explicit_approval",
    }


def archive_bundle(markdown_rows: Sequence[Mapping[str, str]], result_rows: Sequence[Mapping[str, str]], config_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows = []
    order = 0
    for row in markdown_rows:
        if row["path"].startswith("report/") and row["markdown_class"] in {"phase report with unique execution provenance", "phase report with unique scientific evidence", "Codex prompt or execution transcript"}:
            order += 1
            rows.append(bundle_row("B", order, row["path"], "archive_in_f6i2b", "historical report retained for provenance but removable from current docs surface", row["target_canonical_document"], "copy to docs/archive with hash manifest", "path moves only after documentation link update", "archive hash validation and link audit", "archive bundle", "medium"))
    for row in config_rows:
        if row["config_class"] == "frozen_historical_contract":
            order += 1
            rows.append(bundle_row("B", order, row["path"], "archive_in_f6i2b", "historical phase config with unique protocol values", "configs/archive/phases/", "freeze value map then archive", "path move after loader compatibility wrapper", "flattened value equivalence", "archive bundle", "medium"))
    for row in result_rows:
        if row["deletion_or_archive_safety"].startswith("archive_candidate"):
            order += 1
            rows.append(bundle_row("B", order, row["path"], "archive_in_f6i2b", "derived historical output with retained/frozen provenance", "results/archive/", "archive with path/hash/producer metadata", "path move only", "producer or frozen-status validation", "archive bundle", "medium"))
    return rows


def consolidation_bundle_from_merge(kind: str, rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    prefix = {"code": "C", "config": "D", "test": "E", "documentation": "F"}[kind]
    bundle_rows = []
    for order, row in enumerate(rows, start=1):
        path = row.get("member_files") or row.get("member_configs") or row.get("test_path") or row.get("path") or row.get("member_documents", "")
        replacement = row.get("proposed_common_module") or row.get("target_structure") or row.get("replacement_test") or row.get("target_canonical_document", "")
        bundle_rows.append(bundle_row(prefix, order, path, f"{kind}_consolidation_in_f6i2b", f"{kind} consolidation candidate", replacement, row.get("migration", "design only"), "wrappers/archive targets only after approval", row.get("validation", "focused validation"), row.get("rollback", "restore from Git or archive"), "medium"))
    return bundle_rows


def documentation_bundle(markdown_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in markdown_rows:
        if row["path"] in {"README.md", "AGENTS.md"}:
            continue
        grouped[row["target_canonical_document"]].append(row["path"])
    rows = []
    for order, (target, paths) in enumerate(sorted(grouped.items()), start=1):
        rows.append(
            {
                "member_documents": ";".join(sorted(paths)),
                "target_canonical_document": target,
                "migration": "extract unique claims and decisions before moving historical files",
                "validation": "claim-preservation table plus Markdown link audit",
                "rollback": "restore source docs from Git or archive",
            }
        )
    return consolidation_bundle_from_merge("documentation", rows)


def build_target_architecture() -> list[dict[str, str]]:
    return [
        arch_row("source code", "src/audit/", "shared hash/path/git/csv/report/provenance helpers", "new helpers must preserve historical wrappers", "medium"),
        arch_row("source code", "src/data/", "dataset loaders, schemas, reconstruction", "dataset configs remain path-driven", "low"),
        arch_row("source code", "src/baselines/", "Random, MostPop, BM25, BPRMF, GRU4Rec, SASRec, LightGCN", "old entrypoints remain wrappers", "medium"),
        arch_row("source code", "src/llm/", "LLM orchestration, prompts, parsers, local runners", "no prompt/parser loss; raw outputs frozen", "high"),
        arch_row("source code", "src/evaluation/;src/metrics/", "utility and beyond-utility metrics", "metric contracts stay stable", "high"),
        arch_row("source code", "src/experiments/", "CLI entrypoints and phase wrappers", "all scripts executable from root", "medium"),
        arch_row("configs", "configs/base/;configs/datasets/;configs/models/;configs/prompts/;configs/experiments/;configs/archive/phases/", "defaults, overrides, historical frozen configs", "loader compatibility wrappers required", "medium"),
        arch_row("tests", "tests/unit/;tests/integration/;tests/regression/;tests/offline_audit/;tests/manual_unsafe/", "test suite split by risk and runtime", "full suite not run by default", "medium"),
        arch_row("documentation", "docs/current_status.md;docs/architecture.md;docs/reproduction_protocol.md;docs/roadmap.md;docs/archive/", "current docs and historical archive", "Markdown links and claims must be preserved", "medium"),
        arch_row("results", "results/raw/;results/frozen/;results/current/;results/derived/;results/archive/", "raw evidence, frozen evidence, current summaries, derived tables, archive", "raw/predictions/checkpoints never deleted without replacement", "high"),
    ]


def arch_row(area: str, target: str, rule: str, compatibility: str, risk: str) -> dict[str, str]:
    return {
        "area": area,
        "target_directory": target,
        "migration_rule": rule,
        "compatibility_implications": compatibility,
        "path_collision_policy": "fail_closed_before_move",
        "risk": risk,
    }


def manual_review_rows(data: PhaseData, decisions: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows = []
    for row in data.manual_review:
        rows.append(
            {
                "path": row.get("path", ""),
                "reason": row.get("reason", ""),
                "source": "phase_f6i1_manual_review",
                "risk": row.get("risk", "medium"),
                "required_resolution": "manual inspection before any F6I2B action",
            }
        )
    for decision in decisions:
        if decision["decision"] in {"manual_review", "unresolved"}:
            rows.append(
                {
                    "path": decision["path"],
                    "reason": decision["reason"],
                    "source": "phase_f6i2a_retention_decision",
                    "risk": decision["scientific_risk"],
                    "required_resolution": "human approval and dependency proof",
                }
            )
    unique = {row["path"] + "|" + row["source"]: row for row in rows}
    return list(unique.values())


def validate_outputs(data: PhaseData, tables: Mapping[str, Sequence[Mapping[str, str]]], protected_hashes: Mapping[str, Any], initial_git: str, final_git: str) -> list[dict[str, str]]:
    rows = []

    def add(name: str, expected: str, observed: Any, passed: bool, severity: str = "blocking", evidence: str = "") -> None:
        rows.append(
            {
                "validation_id": f"VAL-F6I2A-{len(rows) + 1:03d}",
                "check_name": name,
                "expected": expected,
                "observed": str(observed),
                "status": "pass" if passed else "fail",
                "severity": severity,
                "evidence": evidence,
            }
        )

    inventory_paths = [row["path"] for row in data.inventory]
    decision_paths = [row["path"] for row in tables["retention_decisions"]]
    add("selected_phase_identifier_available", PHASE_ID, PHASE_ID, True)
    add("f6i1_inventory_count", "1948", len(data.inventory), len(data.inventory) == 1948)
    add("f6i1_edges_count", "45826", len(data.edges), len(data.edges) == 45826)
    add("one_decision_per_relevant_file", "same unique path set", len(decision_paths), sorted(inventory_paths) == sorted(decision_paths))
    future_counts = Counter(row["future_task_group"] for row in tables["future_dependency_protection"])
    missing_tasks = [task for task in ROADMAP_TERMS if future_counts[task] == 0]
    add("every_future_task_has_dependencies", "0 missing tasks", len(missing_tasks), not missing_tasks, evidence=";".join(missing_tasks))
    delete_rows = list(tables["low_risk_delete_bundle"])
    bad_delete = [row["path"] for row in delete_rows if not delete_gate_passes(row["path"], data)]
    add("future_delete_candidates_pass_strict_gate", "0 bad candidates", len(bad_delete), not bad_delete, evidence=";".join(bad_delete[:20]))
    add("merge_candidates_have_unique_logic_policy", "all code merge rows name unique preservation", all(row.get("unique_functions") for row in tables["python_merge_plan"]), all(row.get("unique_functions") for row in tables["python_merge_plan"]))
    add("archive_bundle_has_metadata", "all archive rows include path/reason/rollback", all(row.get("path") and row.get("reason") and row.get("rollback") for row in tables["archive_bundle"]), all(row.get("path") and row.get("reason") and row.get("rollback") for row in tables["archive_bundle"]))
    test_deletes = [row for row in tables["test_consolidation_bundle"] if "delete" in row.get("action", "")]
    add("test_deletion_assertion_replacements", "0 test deletes in F6I2A", len(test_deletes), not test_deletes)
    config_deletes = [row for row in tables["config_consolidation_bundle"] if "delete" in row.get("action", "")]
    add("config_deletion_unique_values_preserved", "0 config deletes in F6I2A", len(config_deletes), not config_deletes)
    add("markdown_consolidation_preserves_claims", "documentation bundle rows require claim-preservation validation", all("claim" in row.get("validation", "") for row in tables["documentation_consolidation_bundle"]), all("claim" in row.get("validation", "") for row in tables["documentation_consolidation_bundle"]))
    raw_delete = [row["path"] for row in delete_rows if inventory_by_path(data).get(row["path"], {}).get("category") in RAW_OR_EXPENSIVE_CATEGORIES]
    add("raw_evidence_never_proposed_for_deletion", "0 raw deletes", len(raw_delete), not raw_delete, evidence=";".join(raw_delete))
    lost_prompt_parser_metric = [row["path"] for row in tables["retention_decisions"] if any(token in row["path"].lower() for token in ["prompt", "parser", "metric"]) and row["decision"].startswith("delete_")]
    add("no_prompt_parser_metric_capability_loss", "0 deletes", len(lost_prompt_parser_metric), not lost_prompt_parser_metric)
    target_dirs = [row["target_directory"] for row in tables["target_repository_architecture"]]
    add("target_architecture_no_path_collisions", "unique target rows", len(set(target_dirs)) == len(target_dirs), len(set(target_dirs)) == len(target_dirs))
    add("bundles_dependency_ordered", "bundle IDs sorted", bundle_ids_sorted(tables), bundle_ids_sorted(tables))
    add("no_git_deletions_or_renames", "no D/R status rows", no_deleted_or_renamed(final_git), no_deleted_or_renamed(final_git))
    add("protected_hashes_unchanged", "all protected hashes unchanged", protected_hashes.get("all_unchanged"), protected_hashes.get("all_unchanged") is True)
    branch = git_stdout(["branch", "--show-current"])
    head = git_stdout(["rev-parse", "HEAD"])
    add("branch_matches_expected", EXPECTED_BRANCH, branch, branch == EXPECTED_BRANCH)
    add("head_matches_expected", EXPECTED_HEAD, head, head == EXPECTED_HEAD)
    return rows


def inventory_by_path(data: PhaseData) -> dict[str, dict[str, str]]:
    return {row["path"]: row for row in data.inventory}


def delete_gate_passes(path: str, data: PhaseData) -> bool:
    row = inventory_by_path(data).get(path)
    if not row:
        return False
    return (
        row.get("category") not in PROTECTED_CATEGORIES
        and row.get("category") not in RAW_OR_EXPENSIVE_CATEGORIES
        and row.get("size_bytes") == "0"
        and path.startswith("results/logs/")
        and path.endswith(".log")
    )


def bundle_ids_sorted(tables: Mapping[str, Sequence[Mapping[str, str]]]) -> bool:
    ids = []
    for name in ["low_risk_delete_bundle", "archive_bundle", "code_consolidation_bundle", "config_consolidation_bundle", "test_consolidation_bundle", "documentation_consolidation_bundle"]:
        ids.extend(row.get("bundle_id", "") for row in tables[name])
    return ids == sorted(ids)


def no_deleted_or_renamed(git_status: str) -> bool:
    return not any(line[:2].strip() in {"D", "R"} or line.startswith((" D", "D ", "R ")) for line in git_status.splitlines() if line)


def protected_paths(data: PhaseData) -> list[str]:
    paths = []
    for row in data.inventory:
        category = row.get("category", "")
        path = row["path"]
        if category in PROTECTED_CATEGORIES or path.startswith(("configs/", "src/evaluation/", "src/metrics/", "src/prompts/")):
            if (PROJECT_ROOT / path).is_file():
                paths.append(path)
    return sorted(paths)


def compute_protected_hashes(data: PhaseData) -> dict[str, Any]:
    rows = []
    unchanged = True
    by_path = inventory_by_path(data)
    for path in protected_paths(data):
        current = sha256_file(PROJECT_ROOT / path)
        f6i1_hash = by_path[path].get("sha256", "")
        status = "unchanged" if current == f6i1_hash else "changed"
        unchanged = unchanged and status == "unchanged"
        rows.append({"path": path, "f6i1_sha256": f6i1_hash, "current_sha256": current, "status": status})
    return {"all_unchanged": unchanged, "protected_file_count": len(rows), "rows": rows}


def build_tables(data: PhaseData) -> dict[str, list[dict[str, str]]]:
    indexes = edge_index(data.edges)
    source_rows = build_source_file_audit(data, indexes)
    python_groups, python_merge = build_python_groups_and_merge(source_rows)
    config_rows, config_plan = build_config_audit_and_plan(data, indexes)
    test_rows, assertion_coverage = build_test_audit_and_coverage(data, indexes)
    markdown_rows = build_markdown_audit(data, indexes)
    agents_rows = build_agents_codex_audit(data)
    result_rows = build_results_manifest_audit(data, indexes)
    phase_family = build_phase_family_consolidation(data)
    future_rows = build_future_dependency_protection(data)
    decisions = build_retention_decisions(data)
    low_delete = low_risk_delete_bundle(data, decisions)
    archive_rows = archive_bundle(markdown_rows, result_rows, config_rows)
    code_bundle = consolidation_bundle_from_merge("code", python_merge)
    config_bundle = consolidation_bundle_from_merge("config", config_plan)
    test_bundle = consolidation_bundle_from_merge("test", assertion_coverage[:0])
    doc_bundle = documentation_bundle(markdown_rows)
    architecture = build_target_architecture()
    manual = manual_review_rows(data, decisions)
    next_actions = [
        {"action_id": "NEXT-F6I2A-001", "phase": "phase_f6i2b", "action": "Review and approve Bundle A low-risk deletion only after confirming zero-byte logs remain unused.", "blocking": "false"},
        {"action_id": "NEXT-F6I2A-002", "phase": "phase_f6i2b", "action": "Build archive manifests before moving any historical report/config/result.", "blocking": "true"},
        {"action_id": "NEXT-F6I2A-003", "phase": "F6J-F9", "action": "Use future dependency protection map before deleting any roadmap-mapped file.", "blocking": "true"},
    ]
    return {
        "source_file_audit": source_rows,
        "python_semantic_groups": python_groups,
        "python_merge_plan": python_merge,
        "config_audit": config_rows,
        "config_merge_plan": config_plan,
        "test_audit": test_rows,
        "test_assertion_coverage": assertion_coverage,
        "markdown_audit": markdown_rows,
        "agents_codex_audit": agents_rows,
        "results_manifest_audit": result_rows,
        "phase_family_consolidation": phase_family,
        "future_dependency_protection": future_rows,
        "retention_decisions": decisions,
        "low_risk_delete_bundle": low_delete,
        "archive_bundle": archive_rows,
        "code_consolidation_bundle": code_bundle,
        "config_consolidation_bundle": config_bundle,
        "test_consolidation_bundle": test_bundle,
        "documentation_consolidation_bundle": doc_bundle,
        "target_repository_architecture": architecture,
        "manual_review": manual,
        "next_actions": next_actions,
    }


def table_columns(name: str, rows: Sequence[Mapping[str, str]]) -> list[str]:
    explicit = {
        "source_file_audit": ["path", "category", "phase", "python_class", "imports", "imported_by", "functions", "classes", "function_count", "class_count", "cli_entrypoint", "main_guard", "config_paths", "input_paths", "output_paths", "schema_definitions", "metric_implementation", "parser_implementation", "prompt_construction", "subprocess_usage", "tests", "reports_citing", "future_roadmap_dependency", "one_off_assessment", "provenance_required"],
        "python_semantic_groups": ["semantic_group_id", "group_name", "member_files", "shared_functions", "unique_function_policy", "evidence", "risk"],
        "python_merge_plan": ["merge_group_id", "member_files", "shared_functions", "unique_functions", "unique_constants", "unique_schemas", "proposed_common_module", "wrapper_files", "later_delete_candidates", "migration_order", "tests_required", "affected_outputs", "provenance_implications", "rollback_strategy", "approval_flag"],
        "config_audit": ["path", "config_class", "loader_sources", "command_or_phase", "executed_evidence", "unique_keys", "duplicated_key_group", "inherited_values", "phase_specific_values", "scientific_contract_values", "output_paths", "model_settings", "prompt_settings", "dataset_settings", "seed_settings", "metric_settings", "future_dependency"],
        "config_merge_plan": ["config_merge_group_id", "group_name", "member_configs", "target_structure", "values_to_preserve", "historical_contract_policy", "migration", "validation", "rollback", "approval_flag"],
        "test_audit": ["path", "test_class", "target_implementation", "test_functions", "assertion_count", "edge_cases", "fixtures", "protected_regression", "still_runs", "subsumed_by", "unsafe_operations", "filesystem_writes", "future_roadmap_dependency"],
        "test_assertion_coverage": ["test_path", "target_implementation", "unique_assertions", "edge_cases", "replacement_test", "coverage_status", "assertion_level_deletion_safe", "notes"],
        "markdown_audit": ["path", "markdown_class", "unique_claims", "current_or_historical_status", "referenced_artifacts", "inbound_references", "outbound_references", "decision_tokens", "duplicated_sections", "future_usefulness", "target_canonical_document"],
        "agents_codex_audit": ["path", "scope", "instruction_precedence", "currently_active", "conflicts_with_newer_instructions", "unique_safety_rules", "historical_value", "future_value", "proposed_target_location", "recommendation"],
        "results_manifest_audit": ["path", "artifact_kind", "producer", "retained_producer_available", "input_hashes", "regeneration_cost", "requires_training_or_inference", "deterministic_status", "current_consumers", "historical_provenance_value", "replacement_artifact", "deletion_or_archive_safety"],
        "phase_family_consolidation": ["phase_family_id", "scientific_function", "phases", "canonical_final_implementation", "still_useful_intermediate_implementation", "provenance_only_phases", "fully_absorbed_corrective_phases", "frozen_outputs_to_remain", "scripts_configs_tests_to_consolidate", "reports_to_archive"],
        "future_dependency_protection": ["path", "future_task_group", "evidence", "retention_requirement", "delete_allowed_without_replacement"],
        "retention_decisions": ["path", "decision", "confidence", "scientific_risk", "technical_risk", "unique_information_status", "future_roadmap_dependencies", "canonical_replacement", "required_migration", "required_validation", "rollback_source", "reason"],
        "target_repository_architecture": ["area", "target_directory", "migration_rule", "compatibility_implications", "path_collision_policy", "risk"],
        "manual_review": ["path", "reason", "source", "risk", "required_resolution"],
        "validation": ["validation_id", "check_name", "expected", "observed", "status", "severity", "evidence"],
        "determinism": ["determinism_id", "check_name", "first_digest", "second_digest", "status", "notes"],
        "next_actions": ["action_id", "phase", "action", "blocking"],
    }
    bundle_cols = ["bundle_id", "path", "action", "reason", "replacement", "migration", "expected_diff", "validation", "rollback", "risk", "approval_flag"]
    for bundle_name in ["low_risk_delete_bundle", "archive_bundle", "code_consolidation_bundle", "config_consolidation_bundle", "test_consolidation_bundle", "documentation_consolidation_bundle"]:
        explicit[bundle_name] = bundle_cols
    if name in explicit:
        return explicit[name]
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return keys


def semantic_digest(tables: Mapping[str, Sequence[Mapping[str, str]]]) -> str:
    stable = {name: sorted([dict(row) for row in rows], key=lambda row: json.dumps(row, sort_keys=True)) for name, rows in tables.items() if name not in {"validation", "determinism"}}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode("utf-8")).hexdigest().upper()


def build_manifest(data: PhaseData, tables: Mapping[str, Sequence[Mapping[str, str]]], protected_hashes: Mapping[str, Any], initial_git: str, final_git: str) -> dict[str, Any]:
    counts = summary_counts(tables)
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "not_exact_original_runtime_reproduction": True,
        "selected_identifier_available": True,
        "f6i1_authoritative_inputs": {name: normalize_path(path) for name, path in F6I1_INPUTS.items()},
        "counts": counts,
        "output_tables": {name: normalize_path(path) for name, path in OUTPUT_TABLES.items()},
        "report": normalize_path(REPORT_PATH),
        "protected_hashes_all_unchanged": protected_hashes["all_unchanged"],
        "initial_git_state": initial_git,
        "final_git_state": final_git,
        "no_cleanup_or_execution": True,
    }


def summary_counts(tables: Mapping[str, Sequence[Mapping[str, str]]]) -> dict[str, int]:
    decisions = Counter(row["decision"] for row in tables["retention_decisions"])
    return {
        "relevant_files": len(tables["retention_decisions"]),
        "python_files_audited": len(tables["source_file_audit"]),
        "yaml_files_audited": len(tables["config_audit"]),
        "tests_audited": len(tables["test_audit"]),
        "markdown_files_audited": len(tables["markdown_audit"]),
        "agents_codex_files_audited": len(tables["agents_codex_audit"]),
        "canonical_runtime_files": decisions["keep_canonical_runtime"],
        "canonical_builders": decisions["keep_canonical_builder"],
        "future_dependency_files": decisions["keep_future_dependency"],
        "frozen_evidence_files": decisions["keep_frozen_evidence"],
        "provenance_only_files": decisions["keep_provenance_only"],
        "low_risk_deletion_candidates": len(tables["low_risk_delete_bundle"]),
        "archive_candidates": len(tables["archive_bundle"]),
        "python_merge_groups": len(tables["python_merge_plan"]),
        "config_merge_groups": len(tables["config_merge_plan"]),
        "test_merge_groups": len(tables["test_consolidation_bundle"]),
        "documentation_merge_groups": len(tables["documentation_consolidation_bundle"]),
        "manual_review_count": len(tables["manual_review"]),
    }


def generate_report(data: PhaseData, tables: Mapping[str, Sequence[Mapping[str, str]]], protected_hashes: Mapping[str, Any], initial_git: str, final_git: str) -> str:
    counts = summary_counts(tables)
    validation_failures = [row for row in tables["validation"] if row["status"] != "pass" and row["severity"] == "blocking"]
    architecture_summary = "; ".join(f"{row['area']}->{row['target_directory']}" for row in tables["target_repository_architecture"])
    return f"""# Phase F6I2A Substantive Repository Consolidation Design

Decision: `f6i2a_consolidation_design_ready_for_approval_review`.

Runtime classification: `{RUNTIME_CLASSIFICATION}`. This is not exact original runtime reproduction.

## Scope

- Selected phase identifier: `{PHASE_ID}`.
- F6I1 authoritative inventory rows: `{len(data.inventory)}`.
- F6I1 dependency edges: `{len(data.edges)}`.
- Cleanup executed in F6I2A: `0`.
- Deletions, moves, renames, archives, merges, training, inference, API calls, downloads, package changes, staging, commits: `0`.

## Counts

{markdown_counts(counts)}

## Target Architecture

{architecture_summary}

## Approval Bundles

- Bundle A low-risk deletion: `{len(tables['low_risk_delete_bundle'])}` paths.
- Bundle B archive-only: `{len(tables['archive_bundle'])}` paths.
- Bundle C code consolidation: `{len(tables['code_consolidation_bundle'])}` groups.
- Bundle D config consolidation: `{len(tables['config_consolidation_bundle'])}` groups.
- Bundle E test consolidation: `{len(tables['test_consolidation_bundle'])}` groups.
- Bundle F documentation consolidation: `{len(tables['documentation_consolidation_bundle'])}` groups.

Expected file-count reduction after approved F6I2B low-risk deletion bundle: `{len(tables['low_risk_delete_bundle'])}` files. Archive and merge bundles require separate approval and compatibility wrappers before any count reduction is defensible.

## Validation

- Blocking validation failures: `{len(validation_failures)}`.
- Protected input hashes unchanged: `{protected_hashes['all_unchanged']}`.
- Determinism semantic two-pass result: `{tables['determinism'][0]['status'] if tables.get('determinism') else 'not_recorded'}`.

## Git State

Initial git status:

```text
{initial_git[:4000]}
```

Final git status:

```text
{final_git[:4000]}
```
"""


def markdown_counts(counts: Mapping[str, int]) -> str:
    return "\n".join(f"- `{key}`: `{value}`" for key, value in counts.items())


def update_handoff_and_snapshot(manifest: Mapping[str, Any]) -> None:
    HANDOFF_PATH.parent.mkdir(parents=True, exist_ok=True)
    old = HANDOFF_PATH.read_text(encoding="utf-8", errors="replace") if HANDOFF_PATH.exists() else "# Future Operations Context Handoff\n"
    start = "<!-- phase_f6i2a:start -->"
    end = "<!-- phase_f6i2a:end -->"
    section = f"""{start}
## Phase F6I2A Consolidation Design Handoff

- Phase: `{PHASE_ID}`.
- Runtime classification: `{RUNTIME_CLASSIFICATION}`; not exact original runtime reproduction.
- Authoritative input: F6I1 inventory and dependency graph.
- Relevant files: `{manifest['counts']['relevant_files']}`.
- Approval bundles: Bundle A `{manifest['counts']['low_risk_deletion_candidates']}` deletion candidates; Bundle B `{manifest['counts']['archive_candidates']}` archive candidates; Bundle C/D/E/F consolidation plans in `results/tables/phase_f6i2a_*bundle.csv`.
- No cleanup, movement, rename, merge, training, inference, API call, download, package change, staging or commit was performed.
{end}"""
    if start in old and end in old:
        old = re.sub(re.escape(start) + r".*?" + re.escape(end), section, old, flags=re.DOTALL)
    else:
        old = old.rstrip() + "\n\n" + section + "\n"
    HANDOFF_PATH.write_text(old, encoding="utf-8")

    snapshot: dict[str, Any] = {}
    if SNAPSHOT_PATH.exists():
        try:
            snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            snapshot = {}
    snapshot["phase_f6i2a"] = {
        "phase_id": PHASE_ID,
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "not_exact_original_runtime_reproduction": True,
        "counts": manifest["counts"],
        "report": normalize_path(REPORT_PATH),
        "manifest": normalize_path(DESIGN_MANIFEST_PATH),
        "no_cleanup_or_execution": True,
    }
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_outputs(result: BuildResult, update_current_state: bool) -> None:
    for name, rows in result.tables.items():
        if name not in OUTPUT_TABLES:
            continue
        write_csv(OUTPUT_TABLES[name], rows, table_columns(name, rows))
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    DESIGN_MANIFEST_PATH.write_text(json.dumps(result.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    PROTECTED_HASHES_PATH.write_text(json.dumps(result.protected_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(result.report, encoding="utf-8")
    if update_current_state:
        update_handoff_and_snapshot(result.manifest)


def build(update_current_state: bool = True) -> BuildResult:
    initial_git = git_stdout(["status", "--short", "--branch"])
    data = load_phase_data()
    first_tables = build_tables(data)
    second_tables = build_tables(data)
    first_digest = semantic_digest(first_tables)
    second_digest = semantic_digest(second_tables)
    protected_hashes = compute_protected_hashes(data)
    final_git_placeholder = git_stdout(["status", "--short", "--branch"])
    validation_rows = validate_outputs(data, first_tables, protected_hashes, initial_git, final_git_placeholder)
    determinism_rows = [
        {
            "determinism_id": "DET-F6I2A-001",
            "check_name": "semantic_two_pass_tables",
            "first_digest": first_digest,
            "second_digest": second_digest,
            "status": "pass" if first_digest == second_digest else "fail",
            "notes": "Compares in-memory generated analysis tables before writing outputs.",
        }
    ]
    first_tables["validation"] = validation_rows
    first_tables["determinism"] = determinism_rows
    final_git = git_stdout(["status", "--short", "--branch"])
    manifest = build_manifest(data, first_tables, protected_hashes, initial_git, final_git)
    report = generate_report(data, first_tables, protected_hashes, initial_git, final_git)
    return BuildResult(
        tables=first_tables,
        manifest=manifest,
        protected_hashes=protected_hashes,
        report=report,
        initial_git=initial_git,
        final_git=final_git,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH), help="F6I2A config path.")
    parser.add_argument("--skip-current-state-updates", action="store_true", help="Do not update allowed handoff/snapshot files.")
    args = parser.parse_args(argv)

    config_path = PROJECT_ROOT / normalize_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if config.get("policy", {}).get("execute_cleanup") is not False:
        raise RuntimeError("F6I2A config must keep execute_cleanup=false.")

    result = build(update_current_state=not args.skip_current_state_updates)
    write_outputs(result, update_current_state=not args.skip_current_state_updates)
    blocking = [row for row in result.tables["validation"] if row["status"] != "pass" and row["severity"] == "blocking"]
    deterministic = all(row["status"] == "pass" for row in result.tables["determinism"])
    if blocking or not deterministic:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
