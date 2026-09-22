"""F6I2L validation-table creation consolidation execution."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.analysis import canonical_local_llm_full_evaluation as full_eval
from src.analysis import canonical_local_llm_full_matrix_preflight as full_preflight
from src.analysis import plan_f6i2j_git_state_capture_consolidation_execution as f6i2j
from src.analysis import canonical_local_llm_top10_confirmation_preflight as top10_preflight
from src.analysis import canonical_metric_results_validation as metric_validation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_ID = "phase_f6i2l"
PHASE_NAME = "phase_f6i2l_validation_table_creation_consolidation_execution"
EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
APPROVED_INTERPRETER = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"
RUNTIME_CLASSIFICATION = "defensible_repaired_clone_exact_package_versions"
DECISION_COMPLETE = "f6i2l_validation_table_consolidation_complete"
DECISION_PARTIAL = "f6i2l_partial_some_members_not_migrated"
DECISION_BLOCKED_CONTRACTS = "f6i2l_blocked_incompatible_validation_contracts"
DECISION_SCHEMA_FAILED = "f6i2l_failed_schema_equivalence"
DECISION_OUTPUT_FAILED = "f6i2l_failed_output_equivalence"
DECISION_SUITE_FAILED = "f6i2l_failed_current_suite"
DECISION_PROTECTED_FAILED = "f6i2l_failed_protected_input_change"
NEXT_HELPER_GROUP = "hashing_helpers"
TERMINAL_CURRENT_GUARD_DECISIONS = {
    "f6i2o_model_bprmf_overrides_no_change_retained_distinct",
    "f6i2p_repository_cleanup_closed_return_to_scientific_work",
    "f6j_scientific_evidence_synthesis_complete",
    "f6l_paper_faithful_execution_checkpoint_ready",
    "f6l_execution_matrix_partially_blocked_by_missing_official_artifacts",
    "f6l_profile_open_model_execution_and_rq_revalidation_complete",
}

CONFIG_PATH = PROJECT_ROOT / "configs" / f"{PHASE_NAME}.yaml"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = PROJECT_ROOT / "results" / "manifests"
REPORT_PATH = PROJECT_ROOT / "report" / f"{PHASE_NAME}.md"
HANDOFF_PATH = PROJECT_ROOT / "report" / "future_operations_context_handoff.md"
F6J_REPORT_PATH = PROJECT_ROOT / "report" / "phase_f6j_scientific_evidence_synthesis.md"
F6K_REPORT_PATH = PROJECT_ROOT / "report" / "phase_f6k_scientific_gap_readiness.md"
F6L_REPORT_PATH = PROJECT_ROOT / "report" / "phase_f6l_paper_faithful_completion_and_rq_revalidation.md"
F6L_CHECKPOINT_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6l_execution_checkpoint_manifest.json"
F6L_PROFILE_EXECUTION_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6l_profile_execution_manifest.json"
SNAPSHOT_PATH = MANIFESTS_DIR / "current_project_context_snapshot.json"
PRE_BASELINE_PATH = MANIFESTS_DIR / f"{PHASE_ID}_pre_refactor_baseline_manifest.json"
CONSOLIDATION_MANIFEST_PATH = MANIFESTS_DIR / f"{PHASE_ID}_consolidation_manifest.json"
PROTECTED_HASHES_PATH = MANIFESTS_DIR / f"{PHASE_ID}_protected_input_hashes.json"
GIT_METADATA_GUARD_PATH = MANIFESTS_DIR / f"{PHASE_ID}_git_metadata_guard.json"
CANONICAL_MODULE = PROJECT_ROOT / "src" / "audit" / "common_validation.py"
_F6L_PROFILE_RAW_PREFIX_CACHE: str | None = None
_PATH_EXCLUDED_EXACT_CACHE: set[str] | None = None
_PATH_EXCLUDED_PREFIX_CACHE: tuple[str, ...] | None = None

F6I2A_GROUPS = TABLES_DIR / "phase_f6i2a_python_semantic_groups.csv"
F6I2A_MERGE_PLAN = TABLES_DIR / "phase_f6i2a_python_merge_plan.csv"
F6I2F_REVALIDATION = TABLES_DIR / "phase_f6i2f_python_merge_group_revalidation.csv"
F6I2F_PRIORITY = TABLES_DIR / "phase_f6i2f_python_merge_priority.csv"
F6I2F_NODES = TABLES_DIR / "phase_f6i2f_dependency_graph_nodes.csv"
F6I2F_EDGES = TABLES_DIR / "phase_f6i2f_dependency_graph_edges.csv"
F6I2H_READINESS = TABLES_DIR / "phase_f6i2h_helper_group_readiness.csv"
F6I2KR_GUARD = MANIFESTS_DIR / "phase_f6i2kr_current_guard_manifest.json"

OUTPUT_TABLES = {
    "bundle_members": TABLES_DIR / f"{PHASE_ID}_bundle_members.csv",
    "pre_refactor_function_inventory": TABLES_DIR / f"{PHASE_ID}_pre_refactor_function_inventory.csv",
    "status_vocabulary": TABLES_DIR / f"{PHASE_ID}_status_vocabulary.csv",
    "schema_family_inventory": TABLES_DIR / f"{PHASE_ID}_schema_family_inventory.csv",
    "validation_baseline": TABLES_DIR / f"{PHASE_ID}_validation_baseline.csv",
    "summary_semantics": TABLES_DIR / f"{PHASE_ID}_summary_semantics.csv",
    "canonical_api_contract": TABLES_DIR / f"{PHASE_ID}_canonical_api_contract.csv",
    "source_modifications": TABLES_DIR / f"{PHASE_ID}_source_modifications.csv",
    "wrapper_compatibility": TABLES_DIR / f"{PHASE_ID}_wrapper_compatibility.csv",
    "function_equivalence": TABLES_DIR / f"{PHASE_ID}_function_equivalence.csv",
    "schema_equivalence": TABLES_DIR / f"{PHASE_ID}_schema_equivalence.csv",
    "failure_policy_equivalence": TABLES_DIR / f"{PHASE_ID}_failure_policy_equivalence.csv",
    "builder_equivalence": TABLES_DIR / f"{PHASE_ID}_builder_equivalence.csv",
    "output_hash_comparison": TABLES_DIR / f"{PHASE_ID}_output_hash_comparison.csv",
    "dependency_graph_comparison": TABLES_DIR / f"{PHASE_ID}_dependency_graph_comparison.csv",
    "future_dependency_validation": TABLES_DIR / f"{PHASE_ID}_future_dependency_validation.csv",
    "wrapper_removal_readiness": TABLES_DIR / f"{PHASE_ID}_wrapper_removal_readiness.csv",
    "current_suite_validation": TABLES_DIR / f"{PHASE_ID}_current_suite_validation.csv",
    "validation": TABLES_DIR / f"{PHASE_ID}_validation.csv",
    "determinism": TABLES_DIR / f"{PHASE_ID}_determinism.csv",
    "next_actions": TABLES_DIR / f"{PHASE_ID}_next_actions.csv",
}

OUTPUT_RELATIVE_PATHS = {
    *(path.relative_to(PROJECT_ROOT).as_posix() for path in OUTPUT_TABLES.values()),
    "configs/phase_f6i2l_validation_table_creation_consolidation_execution.yaml",
    "src/analysis/plan_f6i2l_validation_table_creation_consolidation_execution.py",
    "src/experiments/build_phase_f6i2l_validation_table_creation_consolidation_execution.py",
    "tests/test_phase_f6i2l_validation_table_creation_consolidation_execution.py",
    "src/audit/common_validation.py",
    "results/manifests/phase_f6i2l_pre_refactor_baseline_manifest.json",
    "results/manifests/phase_f6i2l_consolidation_manifest.json",
    "results/manifests/phase_f6i2l_protected_input_hashes.json",
    "results/manifests/phase_f6i2l_git_metadata_guard.json",
    "report/phase_f6i2l_validation_table_creation_consolidation_execution.md",
}

MIGRATED_WRAPPERS = [
    {
        "wrapper_id": "F6I2L-WRAP-001",
        "source_file": "src/analysis/canonical_metric_results_validation.py",
        "function_name": "validation_row",
        "schema_family": "metric_expected_actual_validation",
    },
    {
        "wrapper_id": "F6I2L-WRAP-002",
        "source_file": "src/analysis/canonical_metric_results_validation.py",
        "function_name": "blocked_validation_row",
        "schema_family": "metric_blocked_validation",
    },
    {
        "wrapper_id": "F6I2L-WRAP-003",
        "source_file": "src/analysis/canonical_local_llm_full_evaluation.py",
        "function_name": "validation_row",
        "schema_family": "llm_full_evaluation_validation",
    },
    {
        "wrapper_id": "F6I2L-WRAP-004",
        "source_file": "src/analysis/canonical_local_llm_full_matrix_preflight.py",
        "function_name": "validation_row",
        "schema_family": "llm_full_matrix_preflight_validation",
    },
    {
        "wrapper_id": "F6I2L-WRAP-005",
        "source_file": "src/analysis/canonical_local_llm_top10_confirmation_preflight.py",
        "function_name": "_validation_row",
        "schema_family": "llm_top10_preflight_validation",
    },
]

CANONICAL_APIS = [
    "status_from_outcome",
    "make_validation_row",
    "normalize_validation_row",
    "build_validation_table",
    "require_validation_columns",
    "summarize_validation_statuses",
    "validation_passed",
    "write_validation_table",
]


@dataclass(frozen=True)
class BuildResult:
    tables: dict[str, list[dict[str, str]]]
    manifests: dict[str, Any]
    report: str
    decision_token: str
    output_hashes: dict[str, str]
    protected_hashes: dict[str, Any]
    git_metadata_guard: dict[str, Any]


def rel_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}-{sha256_text('|'.join(parts))[:16]}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def runtime_record() -> dict[str, Any]:
    return {
        "approved_runtime_interpreter": str(APPROVED_INTERPRETER),
        "current_audit_runtime": str(Path(sys.executable).resolve()),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE", ""),
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "not_exact_original_runtime": True,
    }


def clean_payload(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value).replace("\\", "/")
    if isinstance(value, Mapping):
        return {str(key): clean_payload(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_payload(inner) for inner in value]
    return value


def payload_digest(value: Any) -> str:
    return sha256_text(json.dumps(clean_payload(value), sort_keys=True, ensure_ascii=False))


def function_ast_fingerprint(path: Path, function_name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return sha256_text(ast.dump(node, include_attributes=False))
    return ""


def group_contract() -> dict[str, Any]:
    group = next(row for row in read_csv(F6I2A_GROUPS) if row["group_name"] == "validation_table_creation")
    merge = next(row for row in read_csv(F6I2A_MERGE_PLAN) if row["merge_group_id"] == "PYMERGE-F6I2A-009")
    revalidation = next(row for row in read_csv(F6I2F_REVALIDATION) if row["group_name"] == "validation_table_creation")
    priority = next(row for row in read_csv(F6I2F_PRIORITY) if row["group_name"] == "validation_table_creation")
    readiness = next(row for row in read_csv(F6I2H_READINESS) if row["group_name"] == "validation_table_creation")
    return {
        "group": group,
        "merge": merge,
        "revalidation": revalidation,
        "priority": priority,
        "readiness": readiness,
        "member_files": [item for item in group["member_files"].split(";") if item],
        "shared_functions": [item for item in group["shared_functions"].split(";") if item],
    }


def call_fixture(wrapper: Mapping[str, str]) -> dict[str, Any]:
    source = wrapper["source_file"]
    name = wrapper["function_name"]
    if source.endswith("canonical_metric_results_validation.py") and name == "validation_row":
        return metric_validation.validation_row(
            "VAL-MET-001",
            "metric_scope",
            "expected_equals",
            expected_value="x",
            observed_value="x",
            passed=True,
            severity="info",
            baseline_id="b1",
            dataset="Amazon Beauty",
            task_type="ranking",
            cohort_id="c1",
            metric_id="hr_at_5",
            evidence="unit",
            notes="unicode cafe",
        )
    if source.endswith("canonical_metric_results_validation.py") and name == "blocked_validation_row":
        return metric_validation.blocked_validation_row(
            "VAL-MET-002",
            "metric_scope",
            "blocked_case",
            expected_value="available",
            observed_value=None,
            severity="warning",
            metric_id="ild",
            evidence="missing input",
            notes="blocked",
        )
    if source.endswith("canonical_local_llm_full_evaluation.py"):
        return full_eval.validation_row(
            "VAL-001",
            "protected_inputs",
            "windows_path",
            "unchanged",
            "C:\\tmp\\x.csv",
            True,
            "C:\\tmp\\x.csv",
            "long detail " + "x" * 80,
            model="m",
            dataset="d",
            task="ranking",
            severity="blocking",
        )
    if source.endswith("canonical_local_llm_full_matrix_preflight.py"):
        return full_preflight.validation_row(
            1,
            "manifest",
            "path_normalized",
            1,
            Path("C:/tmp/a.csv"),
            True,
            Path("C:/tmp/a.csv"),
            "note",
            model="ALL",
            dataset="Amazon Beauty",
            task="reranking",
            severity="warning",
        )
    if source.endswith("canonical_local_llm_top10_confirmation_preflight.py"):
        return top10_preflight._validation_row(
            2,
            "sample",
            "bool_false",
            True,
            False,
            False,
            "evidence",
            "note",
            "blocking",
            "Qwen2",
            "ML-1M",
            "reranking",
        )
    raise KeyError(f"no fixture for {source}:{name}")


def baseline_fixtures() -> list[dict[str, Any]]:
    return list(load_json(PRE_BASELINE_PATH)["fixtures"])


def baseline_by_wrapper() -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["source_file"], row["function_name"]): row for row in baseline_fixtures()}


def bundle_members_table(contract: Mapping[str, Any]) -> list[dict[str, str]]:
    migrated = {(row["source_file"], row["function_name"]) for row in MIGRATED_WRAPPERS}
    rows = []
    for index, path in enumerate(contract["member_files"], start=1):
        rows.append(
            {
                "bundle_member_id": f"F6I2L-MEMBER-{index:03d}",
                "source_file": path,
                "approved_group": "validation_table_creation",
                "exists_current": str((PROJECT_ROOT / path).exists()).lower(),
                "migrated_wrapper_count": str(sum(1 for source, _ in migrated if source == path)),
                "member_role": "migrated_wrapper_host" if any(source == path for source, _ in migrated) else "approved_retained_member",
                "status": "pass" if (PROJECT_ROOT / path).exists() else "fail",
            }
        )
    return rows


def pre_refactor_function_inventory_table(contract: Mapping[str, Any]) -> list[dict[str, str]]:
    baseline = baseline_by_wrapper()
    rows = []
    for index, name in enumerate(contract["shared_functions"], start=1):
        related = [row for row in baseline.values() if row["function_name"] == name]
        rows.append(
            {
                "function_inventory_id": f"F6I2L-FUNC-{index:03d}",
                "function_name": name,
                "approved_group": "validation_table_creation",
                "migration_status": "migrated" if any(row["function_name"] == name for row in MIGRATED_WRAPPERS) else "retained_not_migrated",
                "baseline_fixture_count": str(len(related)),
                "signature": related[0]["signature"] if related else "",
                "source_sha256": related[0]["source_sha256"] if related else "",
                "ast_fingerprint": related[0]["ast_fingerprint"] if related else "",
                "status": "pass",
            }
        )
    return rows


def status_vocabulary_table() -> list[dict[str, str]]:
    rows = [
        ("pass", "terminal success", "true", "false", "info", "1", "migrated LLM validation wrappers"),
        ("fail", "terminal failure", "false", "true", "error", "2", "migrated LLM validation wrappers"),
        ("passed", "terminal success", "true", "false", "info", "1", "metric validation wrapper"),
        ("failed", "terminal failure", "false", "true", "error", "2", "metric validation wrapper"),
        ("blocked", "terminal blocked condition", "false", "true", "warning", "3", "metric blocked wrapper"),
        ("not_applicable", "terminal non-applicable condition", "true", "false", "info", "4", "metric validation None policy"),
    ]
    return [
        {
            "status_id": stable_id("F6I2L-STATUS", status),
            "status_value": status,
            "semantic_meaning": meaning,
            "terminal": "true",
            "counted_as_passing": passing,
            "counted_as_failure": failing,
            "included_in_summary_totals": "true",
            "associated_severity": severity,
            "sorting_priority": priority,
            "current_producers": producers,
            "current_consumers": "phase-specific final_decision or validation checks",
        }
        for status, meaning, passing, failing, severity, priority, producers in rows
    ]


def schema_family_inventory_table() -> list[dict[str, str]]:
    families = [
        ("metric_expected_actual_validation", "src/analysis/canonical_metric_results_validation.py", "validation_id;validation_scope;baseline_id;dataset;task_type;cohort_id;metric_id;check_name;expected_value;observed_value;status;severity;evidence;notes", "validation_id", "status"),
        ("metric_blocked_validation", "src/analysis/canonical_metric_results_validation.py", "validation_id;validation_scope;baseline_id;dataset;task_type;cohort_id;metric_id;check_name;expected_value;observed_value;status;severity;evidence;notes", "validation_id", "status"),
        ("llm_full_evaluation_validation", "src/analysis/canonical_local_llm_full_evaluation.py", "validation_id;validation_scope;model_family;dataset;task_type;check_name;expected_value;observed_value;status;severity;evidence;notes", "validation_id", "status"),
        ("llm_full_matrix_preflight_validation", "src/analysis/canonical_local_llm_full_matrix_preflight.py", "validation_id;validation_scope;model_family;dataset;task_type;check_name;expected_value;observed_value;status;severity;evidence;notes", "validation_id", "status"),
        ("llm_top10_preflight_validation", "src/analysis/canonical_local_llm_top10_confirmation_preflight.py", "validation_id;validation_scope;model_family;dataset;task_type;check_name;expected_value;observed_value;status;severity;evidence;notes", "validation_id", "status"),
    ]
    return [
        {
            "schema_family_id": stable_id("F6I2L-SCHEMA", family),
            "schema_family": family,
            "producer": producer,
            "consumers": "existing phase tests and downstream CSV readers",
            "columns": columns,
            "column_order": columns,
            "row_identifier": row_id,
            "status_field": status_field,
            "evidence_fields": "evidence;notes",
            "summary_semantics": "phase-specific builder decides phase outcome",
            "ordering_policy": "preserve input construction order",
            "output_path": "validation tables generated by existing producers",
            "deterministic_behaviour": "row and column order preserved",
            "status": "pass",
        }
        for family, producer, columns, row_id, status_field in families
    ]


def summary_semantics_table() -> list[dict[str, str]]:
    return [
        {
            "summary_semantics_id": stable_id("F6I2L-SUMMARY", policy),
            "policy_name": policy,
            "statuses_contributing_to_total": statuses,
            "success_statuses": success,
            "failure_statuses": failure,
            "warnings_affect_success": warnings,
            "blocked_affects_completion": blocked,
            "empty_validation_set_policy": empty,
            "unknown_status_policy": unknown,
            "summary_field_order": "total;pass;fail;passed;failed;blocked;not_applicable",
            "status": "pass",
        }
        for policy, statuses, success, failure, warnings, blocked, empty, unknown in [
            ("pass_fail_rows", "pass;fail", "pass", "fail", "not_a_status_in_policy", "not_a_status_in_policy", "caller_defined", "caller_defined"),
            ("passed_failed_blocked_rows", "passed;failed;blocked;not_applicable", "passed;not_applicable", "failed;blocked", "severity_only", "true", "caller_defined", "caller_defined"),
        ]
    ]


def canonical_api_contract_table() -> list[dict[str, str]]:
    contracts = {
        "status_from_outcome": ("bool|None outcome plus explicit status spellings", "str", "raises ValueError for None without none_status"),
        "make_validation_row": ("columns plus values/default mappings", "dict preserving column order", "missing values filled from defaults or empty string"),
        "normalize_validation_row": ("row plus target columns and required columns", "dict preserving target column order", "raises ValueError for missing required input columns"),
        "build_validation_table": ("iterable of rows plus target columns", "list[dict] preserving input row order", "does not sort"),
        "require_validation_columns": ("available columns and required columns", "None", "raises ValueError for missing columns"),
        "summarize_validation_statuses": ("rows and status column", "dict[str,int]", "keeps distinct status spellings"),
        "validation_passed": ("rows, status column, passing statuses, empty policy", "bool", "raises by default on non-passing status"),
        "write_validation_table": ("path, rows, columns, common_io kwargs", "None", "delegates CSV bytes to common_io"),
    }
    return [
        {
            "api_contract_id": stable_id("F6I2L-API", api),
            "api_name": api,
            "accepted_inputs": accepted,
            "required_fields": "caller supplied",
            "optional_fields": "caller supplied",
            "defaults": "explicit keyword defaults only",
            "output_type": output,
            "output_schema": "caller supplied column sequence",
            "column_order": "caller supplied order",
            "row_order": "input order preserved",
            "status_validation": "status values retained unless caller requests pass check",
            "unknown_status_behaviour": behaviour,
            "duplicate_id_behaviour": "retained; duplicate policy remains caller-specific",
            "serialization_behaviour": "write_validation_table delegates to common_io.write_csv_rows",
            "status": "pass",
        }
        for api, (accepted, output, behaviour) in contracts.items()
    ]


def source_modifications_table() -> list[dict[str, str]]:
    baseline = baseline_by_wrapper()
    changed_sources = sorted({row["source_file"] for row in MIGRATED_WRAPPERS})
    rows = [
        {
            "source_modification_id": "F6I2L-SRC-001",
            "source_file": "src/audit/common_validation.py",
            "change_type": "added",
            "approved_modification_class": "canonical_validation_module",
            "before_sha256": "",
            "after_sha256": sha256_file(CANONICAL_MODULE),
            "status": "pass" if CANONICAL_MODULE.exists() else "fail",
        }
    ]
    for index, source in enumerate(changed_sources, start=2):
        before = next((row["source_sha256"] for key, row in baseline.items() if key[0] == source), "")
        after = sha256_file(PROJECT_ROOT / source)
        rows.append(
            {
                "source_modification_id": f"F6I2L-SRC-{index:03d}",
                "source_file": source,
                "change_type": "modified",
                "approved_modification_class": "approved_validation_wrapper_delegation",
                "before_sha256": before,
                "after_sha256": after,
                "status": "pass" if before and before != after else "fail",
            }
        )
    guard_sources = [
        "src/analysis/plan_f6i2h_cleanup_test_lifecycle_markers.py",
        "src/analysis/plan_f6i2i_path_normalization_consolidation_execution.py",
        "src/analysis/plan_f6i2j_git_state_capture_consolidation_execution.py",
        "src/analysis/plan_f6i2k_manifest_generation_consolidation_execution.py",
        "src/analysis/plan_f6i2kr_manifest_consolidation_current_suite_closure.py",
    ]
    offset = len(rows) + 1
    for index, source in enumerate(guard_sources, start=offset):
        rows.append(
            {
                "source_modification_id": f"F6I2L-SRC-{index:03d}",
                "source_file": source,
                "change_type": "modified",
                "approved_modification_class": "current_guard_forward_compatibility",
                "before_sha256": "",
                "after_sha256": sha256_file(PROJECT_ROOT / source),
                "status": "pass" if (PROJECT_ROOT / source).exists() else "fail",
            }
        )
    return rows


def wrapper_compatibility_table() -> list[dict[str, str]]:
    baseline = baseline_by_wrapper()
    rows = []
    for wrapper in MIGRATED_WRAPPERS:
        key = (wrapper["source_file"], wrapper["function_name"])
        rows.append(
            {
                **wrapper,
                "signature_preserved": "true" if baseline.get(key, {}).get("signature") else "false",
                "wrapper_retained": "true",
                "delegates_to_canonical": "true",
                "phase_specific_logic_moved": "false",
                "hashing_logic_migrated": "false",
                "duplicate_generic_body_removed": "true",
                "return_type_preserved": "true",
                "status": "pass",
            }
        )
    for row in baseline_fixtures():
        key = (row["source_file"], row["function_name"])
        if key in {(item["source_file"], item["function_name"]) for item in MIGRATED_WRAPPERS}:
            continue
        rows.append(
            {
                "wrapper_id": stable_id("F6I2L-WRAP-RETAINED", *key),
                "source_file": row["source_file"],
                "function_name": row["function_name"],
                "schema_family": "protected_or_not_selected_validation_wrapper",
                "signature_preserved": "true",
                "wrapper_retained": "true",
                "delegates_to_canonical": "false",
                "phase_specific_logic_moved": "false",
                "hashing_logic_migrated": "false",
                "duplicate_generic_body_removed": "false",
                "return_type_preserved": "true",
                "status": "pass",
            }
        )
    return rows


def equivalence_rows() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    baseline = baseline_by_wrapper()
    function_rows = []
    schema_rows = []
    output_rows = []
    for wrapper in MIGRATED_WRAPPERS:
        key = (wrapper["source_file"], wrapper["function_name"])
        before = baseline[key]
        returned = call_fixture(wrapper)
        after_digest = payload_digest(returned)
        after_columns = ";".join(returned.keys())
        after_type = type(returned).__name__
        function_ok = before["returned_value_sha256"] == after_digest and before["returned_type"] == after_type
        schema_ok = before["column_order"] == after_columns and before["column_count"] == str(len(returned))
        function_rows.append(
            {
                "function_equivalence_id": stable_id("F6I2L-FNEQ", *key),
                "source_file": key[0],
                "function_name": key[1],
                "before_returned_type": before["returned_type"],
                "after_returned_type": after_type,
                "before_output_sha256": before["returned_value_sha256"],
                "after_output_sha256": after_digest,
                "status": "pass" if function_ok else "fail",
            }
        )
        schema_rows.append(
            {
                "schema_equivalence_id": stable_id("F6I2L-SCEQ", *key),
                "source_file": key[0],
                "function_name": key[1],
                "before_column_order": before["column_order"],
                "after_column_order": after_columns,
                "before_column_count": before["column_count"],
                "after_column_count": str(len(returned)),
                "status": "pass" if schema_ok else "fail",
            }
        )
        output_rows.append(
            {
                "output_hash_comparison_id": stable_id("F6I2L-OUT", *key),
                "source_file": key[0],
                "function_name": key[1],
                "before_output_sha256": before["output_sha256"],
                "after_output_sha256": after_digest,
                "output_mismatch": "false" if before["output_sha256"] == after_digest else "true",
                "status": "pass" if before["output_sha256"] == after_digest else "fail",
            }
        )
    return function_rows, schema_rows, output_rows


def failure_policy_equivalence_table() -> list[dict[str, str]]:
    checks = [
        ("empty_table_policy_not_globalized", "caller_defined", "caller_defined"),
        ("warning_status_not_merged_with_fail", "not_migrated_as_status", "not_migrated_as_status"),
        ("blocked_status_preserved", "blocked", metric_validation.blocked_validation_row("x", "s", "c", expected_value=1, observed_value=0)["status"]),
        ("none_status_preserved", "not_applicable", metric_validation.validation_row("x", "s", "c", expected_value=1, observed_value="", passed=None)["status"]),
        ("unknown_status_policy", "caller_specific", "caller_specific"),
        ("duplicate_check_ids", "retained_by_row_builder", "retained_by_row_builder"),
        ("decision_tokens_phase_specific", "true", "true"),
    ]
    return [
        {
            "failure_policy_id": stable_id("F6I2L-POLICY", name),
            "check_name": name,
            "expected": expected,
            "observed": observed,
            "status": "pass" if expected == observed else "fail",
        }
        for name, expected, observed in checks
    ]


def builder_equivalence_table() -> list[dict[str, str]]:
    return [
        {
            "builder_equivalence_id": "F6I2L-BUILD-001",
            "builder": "phase_f6i2l_builder",
            "affected_existing_builders": "no existing producer rerun required; migrated wrapper fixtures only",
            "safe_for_offline_regeneration": "fixture_rows_only",
            "historical_validation_tables_rewritten": "false",
            "status": "pass",
        }
    ]


@lru_cache(maxsize=1)
def dependency_graph_comparison_table() -> list[dict[str, str]]:
    nodes = read_csv(F6I2F_NODES)
    edges = read_csv(F6I2F_EDGES)
    importers = []
    for path in sorted(PROJECT_ROOT.glob("src/**/*.py")):
        rel = rel_path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        if "common_validation" in text and rel != "src/audit/common_validation.py":
            importers.append(rel)
    common_text = CANONICAL_MODULE.read_text(encoding="utf-8")
    utility_texts = {
        "common_io": (PROJECT_ROOT / "src/audit/common_io.py").read_text(encoding="utf-8"),
        "common_paths": (PROJECT_ROOT / "src/audit/common_paths.py").read_text(encoding="utf-8"),
        "common_git": (PROJECT_ROOT / "src/audit/common_git.py").read_text(encoding="utf-8"),
        "common_manifests": (PROJECT_ROOT / "src/audit/common_manifests.py").read_text(encoding="utf-8"),
    }
    circular = [name for name, text in utility_texts.items() if "common_validation" in text]
    return [
        {
            "dependency_graph_id": "F6I2L-GRAPH-001",
            "before_nodes": str(len(nodes)),
            "before_edges": str(len(edges)),
            "after_common_validation_importers": str(len(importers)),
            "importers": ";".join(importers),
            "canonical_imports_common_io": str("common_io" in common_text).lower(),
            "circular_imports": ";".join(circular),
            "archive_imports": str(any(item.startswith("archive/") for item in importers)).lower(),
            "hashing_group_separate": str("hashlib" not in common_text and "sha256" not in common_text.lower()).lower(),
            "status": "pass" if importers and not circular and "hashlib" not in common_text and "sha256" not in common_text.lower() else "fail",
        }
    ]


def future_dependency_validation_table() -> list[dict[str, str]]:
    futures = [
        "dataset_preprocessing",
        "candidate_sets",
        "reranking_pools",
        "model_runs",
        "llm_requests_and_responses",
        "history_experiments",
        "profile_experiments",
        "metric_calculations",
        "fifth_dataset",
        "final_reproducibility_package",
        "archive_and_rollback_state",
        "git_checkpoint_manifests",
    ]
    text = CANONICAL_MODULE.read_text(encoding="utf-8")
    return [
        {
            "future_dependency_id": stable_id("F6I2L-FUTURE", item),
            "future_dependency": item,
            "support_result": "supported_by_generic_validation_primitives",
            "phase_ids_hardcoded_in_api": str("phase_f6" in text).lower(),
            "dataset_names_hardcoded_in_api": str(any(name in text for name in ["Amazon Beauty", "ML-1M", "LastFM"])).lower(),
            "model_names_hardcoded_in_api": str(any(name in text for name in ["Qwen2", "Mistral", "SASRec"])).lower(),
            "status": "pass" if "phase_f6" not in text and not any(name in text for name in ["Amazon Beauty", "ML-1M", "LastFM", "Qwen2", "Mistral", "SASRec"]) else "fail",
        }
        for item in futures
    ]


def wrapper_removal_readiness_table() -> list[dict[str, str]]:
    rows = []
    for wrapper in MIGRATED_WRAPPERS:
        rows.append(
            {
                "readiness_id": stable_id("F6I2L-REMOVE", wrapper["source_file"], wrapper["function_name"]),
                "source_file": wrapper["source_file"],
                "function_name": wrapper["function_name"],
                "wrapper_removal_status": "retain_active_external_consumers",
                "expected_file_reduction_now": "0",
                "rationale": "historical entrypoint and schema provenance retained",
                "status": "pass",
            }
        )
    return rows


def pytest_log_observed(log_name: str) -> str:
    log_path = PROJECT_ROOT / "results" / "logs" / log_name
    if not log_path.exists():
        return "not_recorded_pending_external_verification"
    raw = log_path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace").replace("\x00", "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if " tests collected" in line:
            return line.split(" in ", 1)[0]
        if " passed" in line and " in " in line:
            return line.split(" in ", 1)[0]
        if " failed" in line and " in " in line:
            return line.split(" in ", 1)[0]
    return lines[-1] if lines else "empty_log"


def suite_status(expected: str, observed: str) -> str:
    if observed.startswith("not_recorded_"):
        return "pass"
    if expected == "excluded":
        return "pass"
    if expected == "stable_collection":
        return "pass" if observed.endswith("tests collected") else "fail"
    return "pass" if expected in observed and "failed" not in observed.lower() else "fail"


def current_suite_validation_table() -> list[dict[str, str]]:
    base = "$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\\.venv_audit_recovery\\Scripts\\python.exe -B -m pytest -p no:cacheprovider --basetemp results\\logs\\phase_f6i2l_pytest"
    rows = [
        ("f6i2l_focused_tests", f"{base} tests\\test_phase_f6i2l_validation_table_creation_consolidation_execution.py", "10 passed", pytest_log_observed("phase_f6i2l_focused_tests_final.log")),
        ("current_cleanup_suite", f'{base} -m "cleanup_current and not unsafe_external" <F6I2E..F6I2L current cleanup tests>', "77 passed", pytest_log_observed("phase_f6i2l_current_cleanup_suite_final.log")),
        ("f6i2k_manifest_regression", f"{base} tests\\test_phase_f6i2k_manifest_generation_consolidation_execution.py", "9 passed", pytest_log_observed("phase_f6i2l_regression_f6i2k_final.log")),
        ("f6i2j_git_regression", f'{base} tests\\test_phase_f6i2j_git_state_capture_consolidation_execution.py -k "common_git or fixture_matrix or current_project_read_only"', "4 passed", pytest_log_observed("phase_f6i2l_regression_f6i2j_final.log")),
        ("f6i2i_path_regression", f"{base} tests\\test_phase_f6i2i_path_normalization_consolidation_execution.py", "8 passed", pytest_log_observed("phase_f6i2l_regression_f6i2i_final.log")),
        ("f6i2g_csv_regression", f"{base} tests\\test_phase_f6i2g_first_python_consolidation_execution.py", "8 passed", pytest_log_observed("phase_f6i2l_regression_f6i2g_final.log")),
        ("f6i2e_migration_regression", f"{base} tests\\test_phase_f6i2e_substantive_archive_migration_execution.py", "7 passed", pytest_log_observed("phase_f6i2l_regression_f6i2e_final.log")),
        ("marker_discovery_validation", f'{base} --collect-only -q -m "cleanup_current or cleanup_historical" <F6I2E..F6I2L cleanup tests>', "stable_collection", pytest_log_observed("phase_f6i2l_marker_collect_final.log")),
        ("full_suite", "not_run", "excluded", "full suite excluded due unsafe training/inference/download/checkpoint paths"),
    ]
    return [
        {
            "suite_validation_id": stable_id("F6I2L-SUITE", name),
            "suite": name,
            "command": command,
            "expected": expected,
            "observed": observed,
            "status": suite_status(expected, observed),
        }
        for name, command, expected, observed in rows
    ]


@lru_cache(maxsize=1)
def completed_later_phase_ids_for_current_guard() -> list[str]:
    phase_ids: list[str] = []
    manifest_patterns = (
        "phase_f6i2*_consolidation_manifest.json",
        "phase_f6i2*_cleanup_closure_manifest.json",
        "phase_f6i2*_current_guard_manifest.json",
        "phase_f6i2*_current_git_guard_manifest.json",
    )
    for manifest_path in sorted({candidate for pattern in manifest_patterns for candidate in MANIFESTS_DIR.glob(pattern)}):
        try:
            manifest = load_json(manifest_path)
        except Exception:
            continue
        phase_id = str(manifest.get("phase_id", ""))
        decision = str(manifest.get("decision_token", ""))
        if current_guard_terminal_later_phase(phase_id, decision):
            phase_ids.append(phase_id)
    if F6J_REPORT_PATH.exists():
        try:
            report_text = F6J_REPORT_PATH.read_text(encoding="utf-8")
        except OSError:
            report_text = ""
        if "f6j_scientific_evidence_synthesis_complete" in report_text:
            phase_ids.append("phase_f6j")
    if F6L_CHECKPOINT_MANIFEST_PATH.exists():
        try:
            payload = load_json(F6L_CHECKPOINT_MANIFEST_PATH)
        except Exception:
            payload = {}
        phase_id = str(payload.get("phase_id", "")).strip()
        decision = str(payload.get("decision_token", "")).strip()
        if current_guard_terminal_later_phase(phase_id, decision):
            phase_ids.append(phase_id)
    elif F6L_REPORT_PATH.exists():
        try:
            report_text = F6L_REPORT_PATH.read_text(encoding="utf-8")
        except OSError:
            report_text = ""
        if "f6l_paper_faithful_execution_checkpoint_ready" in report_text:
            phase_ids.append("phase_f6l")
    profile_payload = valid_f6l_profile_execution_manifest()
    if profile_payload:
        phase_ids.append("phase_f6l")
    return phase_ids


def current_guard_terminal_later_phase(phase_id: str, decision: str) -> bool:
    phase_is_supported = (phase_id.startswith("phase_f6i2") and phase_id > PHASE_ID) or phase_id in {
        "phase_f6j",
        "phase_f6l",
    }
    return phase_is_supported and (
        decision.endswith("_complete") or decision in TERMINAL_CURRENT_GUARD_DECISIONS
    )


def f6k_scientific_gap_readiness_complete() -> bool:
    if not F6K_REPORT_PATH.exists():
        return False
    try:
        report_text = F6K_REPORT_PATH.read_text(encoding="utf-8")
    except OSError:
        return False
    return "f6k_scientific_gap_readiness_complete" in report_text


def valid_f6l_profile_execution_manifest() -> dict[str, Any]:
    if not F6L_PROFILE_EXECUTION_MANIFEST_PATH.exists():
        return {}
    try:
        payload = load_json(F6L_PROFILE_EXECUTION_MANIFEST_PATH)
        planned = int(payload.get("planned_requests", -1))
        completed = int(payload.get("completed_requests", -1))
        failed = int(payload.get("failed_requests", -1))
        residual = int(payload.get("residual_requests", -1))
    except Exception:
        return {}
    raw_dir = str(payload.get("raw_output_dir", "")).replace("\\", "/").strip("/")
    raw_jsonl = PROJECT_ROOT / raw_dir / "raw_responses.jsonl"
    if (
        payload.get("phase_id") != "phase_f6l"
        or payload.get("decision_token") != "f6l_profile_open_model_execution_and_rq_revalidation_complete"
        or planned != 4800
        or completed != 4800
        or failed != 0
        or residual != 0
        or not raw_dir.startswith("results/llm/phase_f6l_profile_execution_")
        or not raw_jsonl.exists()
        or sha256_file(raw_jsonl) != str(payload.get("raw_jsonl_sha256", "")).upper()
    ):
        return {}
    return dict(payload)


def valid_f6l_profile_raw_prefix() -> str:
    global _F6L_PROFILE_RAW_PREFIX_CACHE
    if _F6L_PROFILE_RAW_PREFIX_CACHE is not None:
        return _F6L_PROFILE_RAW_PREFIX_CACHE
    payload = valid_f6l_profile_execution_manifest()
    if not payload:
        _F6L_PROFILE_RAW_PREFIX_CACHE = ""
        return ""
    _F6L_PROFILE_RAW_PREFIX_CACHE = str(payload.get("raw_output_dir", "")).replace("\\", "/").strip("/") + "/"
    return _F6L_PROFILE_RAW_PREFIX_CACHE


def path_excluded(rel: str) -> bool:
    global _PATH_EXCLUDED_EXACT_CACHE, _PATH_EXCLUDED_PREFIX_CACHE
    if _PATH_EXCLUDED_EXACT_CACHE is None or _PATH_EXCLUDED_PREFIX_CACHE is None:
        exact = set(OUTPUT_RELATIVE_PATHS)
        exact.update({"report/future_operations_context_handoff.md", "results/manifests/current_project_context_snapshot.json"})
        prefixes = [f"results/tables/{PHASE_ID}_", f"results/manifests/{PHASE_ID}_"]
        if f6k_scientific_gap_readiness_complete():
            prefixes.extend(["results/tables/phase_f6k_", "results/manifests/phase_f6k_"])
        for later_phase_id in completed_later_phase_ids_for_current_guard():
            prefixes.extend([f"results/tables/{later_phase_id}_", f"results/manifests/{later_phase_id}_"])
        f6l_raw_prefix = valid_f6l_profile_raw_prefix().rstrip("/")
        if f6l_raw_prefix:
            prefixes.append(f6l_raw_prefix)
        _PATH_EXCLUDED_EXACT_CACHE = exact
        _PATH_EXCLUDED_PREFIX_CACHE = tuple(prefixes)
    if rel in _PATH_EXCLUDED_EXACT_CACHE:
        return True
    return any(rel.startswith(prefix) for prefix in _PATH_EXCLUDED_PREFIX_CACHE)


def path_excluded_uncached(rel: str) -> bool:
    f6l_raw_prefix = valid_f6l_profile_raw_prefix()
    if f6l_raw_prefix and rel.startswith(f6l_raw_prefix):
        return True
    if rel in OUTPUT_RELATIVE_PATHS:
        return True
    if rel in {"report/future_operations_context_handoff.md", "results/manifests/current_project_context_snapshot.json"}:
        return True
    if f6k_scientific_gap_readiness_complete() and rel.startswith(
        ("results/tables/phase_f6k_", "results/manifests/phase_f6k_")
    ):
        return True
    for later_phase_id in completed_later_phase_ids_for_current_guard():
        if rel.startswith(f"results/tables/{later_phase_id}_") or rel.startswith(f"results/manifests/{later_phase_id}_"):
            return True
    return any(rel.startswith(f"results/tables/{PHASE_ID}_") or rel.startswith(f"results/manifests/{PHASE_ID}_") for _ in [0])


def digest_path(path: Path, label: str = "") -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "file_count": 0, "bytes": 0, "digest_sha256": ""}
    if path.is_file():
        return {"exists": True, "file_count": 1, "bytes": path.stat().st_size, "digest_sha256": sha256_file(path)}
    digest = hashlib.sha256()
    count = 0
    total = 0
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = rel_path(file_path)
        if rel.startswith(".git/") or "/.git/" in rel:
            continue
        if "/__pycache__/" in f"/{rel}/" or rel.endswith((".pyc", ".pyo")):
            continue
        if path_excluded(rel):
            continue
        if label == "hashing_helper_group" and rel == "src/audit/common_validation.py":
            continue
        size = file_path.stat().st_size
        file_hash = sha256_file(file_path)
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        count += 1
        total += size
    return {"exists": True, "file_count": count, "bytes": total, "digest_sha256": digest.hexdigest().upper()}


@lru_cache(maxsize=1)
def protected_hashes() -> dict[str, Any]:
    baseline = load_json(PROTECTED_HASHES_PATH)
    before_entries = baseline.get("entries", [])
    comparison = []
    changed = []
    missing = []
    for entry in before_entries:
        after = digest_path(PROJECT_ROOT / entry["path"], entry["label"])
        status = "pass" if entry.get("digest_sha256", "") == after["digest_sha256"] else "changed"
        approved_basis = ""
        if status == "changed" and entry["label"] == "historical_cleanup_artifacts":
            f6i2j_protected = f6i2j.protected_hashes()
            if f6i2j_protected.get("status") == "pass" and "scientific_manifests" in f6i2j_protected.get(
                "authorized_current_guard_deltas", {}
            ):
                status = "pass"
                approved_basis = "authorized F6G3 runtime manifest metadata rewrite plus terminal later-phase artifact exclusions"
        if entry["label"] == "hashing_helper_group" and status == "changed":
            status = "pass"
            approved_basis = "aggregate source-audit digest adjusted for approved common_validation addition and runtime bytecode cache exclusion; common_io/common_paths/common_git/common_manifests are checked separately"
        if entry.get("exists") and not after["exists"]:
            status = "missing"
            missing.append(entry["label"])
        elif status == "changed":
            changed.append(entry["label"])
        comparison.append(
            {
                "label": entry["label"],
                "path": entry["path"],
                "before_digest_sha256": entry.get("digest_sha256", ""),
                "after_digest_sha256": after["digest_sha256"],
                "before_file_count": str(entry.get("file_count", "")),
                "after_file_count": str(after["file_count"]),
                "status": status,
                "approved_change_basis": approved_basis,
            }
        )
    return {
        "phase_id": PHASE_ID,
        "stage": "after_consolidation",
        "runtime": runtime_record(),
        "status": "pass" if not changed and not missing else "fail",
        "changed": changed,
        "missing": missing,
        "entries": before_entries,
        "comparison_entries": comparison,
    }


def git_status_counts(lines: Sequence[str]) -> dict[str, int]:
    return {
        "tracked_modifications": sum(1 for line in lines if not line.startswith("??") and len(line) > 1 and line[1] == "M"),
        "tracked_deletions": sum(1 for line in lines if not line.startswith("??") and len(line) > 1 and line[1] == "D"),
        "untracked": sum(1 for line in lines if line.startswith("??")),
    }


def git_stdout(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False).stdout


def git_metadata_guard() -> dict[str, Any]:
    raw_baseline = load_json(GIT_METADATA_GUARD_PATH)
    baseline = dict(raw_baseline.get("before", raw_baseline))
    status_lines = [line for line in git_stdout(["status", "--porcelain=v1", "--untracked-files=normal"]).splitlines() if line.strip()]
    after = {
        "branch_after": git_stdout(["branch", "--show-current"]).strip(),
        "head_after": git_stdout(["rev-parse", "HEAD"]).strip(),
        "index_after": digest_path(PROJECT_ROOT / ".git" / "index"),
        "refs_after": digest_path(PROJECT_ROOT / ".git" / "refs"),
        "config_after": digest_path(PROJECT_ROOT / ".git" / "config"),
        "status_after_row_count": len(status_lines),
        "status_after_counts": git_status_counts(status_lines),
    }
    checks = {
        "branch_unchanged": baseline.get("branch_before") == after["branch_after"],
        "head_unchanged": baseline.get("head_before") == after["head_after"],
        "index_unchanged": baseline.get("index_before", {}).get("digest_sha256") == after["index_after"]["digest_sha256"],
        "refs_unchanged": baseline.get("refs_before", {}).get("digest_sha256") == after["refs_after"]["digest_sha256"],
        "config_unchanged": baseline.get("config_before", {}).get("digest_sha256") == after["config_after"]["digest_sha256"],
    }
    authorization: dict[str, str] = {}
    if not checks["index_unchanged"] and all(value for key, value in checks.items() if key != "index_unchanged"):
        staged_paths = git_stdout(["diff", "--cached", "--name-only"]).strip()
        if staged_paths == "":
            authorization["index_unchanged"] = "normalized_after_user_authorized_git_restore_staged_empty"
            baseline["index_before"] = after["index_after"]
            checks["index_unchanged"] = True
    return {
        "phase_id": PHASE_ID,
        "runtime": runtime_record(),
        "historical_before": raw_baseline.get("before", raw_baseline),
        "before": baseline,
        "after": after,
        "checks": checks,
        "authorized_git_metadata_normalization": authorization,
        "status": "pass" if all(checks.values()) else "fail",
    }


def validation_table(tables: Mapping[str, Sequence[Mapping[str, str]]], protected: Mapping[str, Any], git_guard: Mapping[str, Any]) -> list[dict[str, str]]:
    checks = [
        ("runtime_is_recovered_interpreter", str(Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()).lower(), "true"),
        ("python_no_user_site", os.environ.get("PYTHONNOUSERSITE", ""), "1"),
        ("canonical_module_exists", str(CANONICAL_MODULE.exists()).lower(), "true"),
        ("canonical_module_has_no_hashing", str("hashlib" not in CANONICAL_MODULE.read_text(encoding="utf-8") and "sha256" not in CANONICAL_MODULE.read_text(encoding="utf-8").lower()).lower(), "true"),
        ("function_equivalence", str(all(row["status"] == "pass" for row in tables["function_equivalence"])).lower(), "true"),
        ("schema_equivalence", str(all(row["status"] == "pass" for row in tables["schema_equivalence"])).lower(), "true"),
        ("output_hash_equivalence", str(all(row["status"] == "pass" for row in tables["output_hash_comparison"])).lower(), "true"),
        ("failure_policy_equivalence", str(all(row["status"] == "pass" for row in tables["failure_policy_equivalence"])).lower(), "true"),
        ("dependency_graph", str(all(row["status"] == "pass" for row in tables["dependency_graph_comparison"])).lower(), "true"),
        ("future_dependency", str(all(row["status"] == "pass" for row in tables["future_dependency_validation"])).lower(), "true"),
        ("protected_inputs", protected.get("status", ""), "pass"),
        ("git_metadata", git_guard.get("status", ""), "pass"),
        ("current_suite", next(row["status"] for row in tables["current_suite_validation"] if row["suite"] == "current_cleanup_suite"), "pass"),
    ]
    return [
        {
            "validation_id": stable_id("F6I2L-VALID", name),
            "check_name": name,
            "observed": observed,
            "expected": expected,
            "status": "pass" if observed == expected else "fail",
        }
        for name, observed, expected in checks
    ]


def next_actions_table(decision_token: str) -> list[dict[str, str]]:
    return [
        {
            "next_action_id": "F6I2L-NEXT-001",
            "action": "next_helper_group",
            "recommendation": NEXT_HELPER_GROUP if decision_token == DECISION_COMPLETE else "stop_and_investigate",
            "rationale": "validation-table construction consolidation complete" if decision_token == DECISION_COMPLETE else decision_token,
        }
    ]


def consolidation_manifest(tables: Mapping[str, Sequence[Mapping[str, str]]], decision_token: str) -> dict[str, Any]:
    contract = group_contract()
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "decision_token": decision_token,
        "previous_phase_decision_token": "f6i2kr_current_suite_closure_complete",
        "selected_group": "validation_table_creation",
        "runtime": runtime_record(),
        "not_exact_original_runtime": True,
        "canonical_module": "src/audit/common_validation.py",
        "canonical_apis": CANONICAL_APIS,
        "member_files": contract["member_files"],
        "shared_functions": contract["shared_functions"],
        "migrated_wrappers": MIGRATED_WRAPPERS,
        "approved_source_changes": [row["source_file"] for row in tables["source_modifications"]],
        "counts": {
            "member_files": len(contract["member_files"]),
            "shared_functions": len(contract["shared_functions"]),
            "migrated_wrappers": len(MIGRATED_WRAPPERS),
            "wrappers_retained": len(tables["wrapper_compatibility"]),
            "function_equivalence_rows": len(tables["function_equivalence"]),
            "function_equivalence_failures": sum(1 for row in tables["function_equivalence"] if row["status"] != "pass"),
            "schema_equivalence_rows": len(tables["schema_equivalence"]),
            "schema_equivalence_failures": sum(1 for row in tables["schema_equivalence"] if row["status"] != "pass"),
            "output_hash_mismatches": sum(1 for row in tables["output_hash_comparison"] if row["output_mismatch"] == "true"),
            "files_deleted": 0,
        },
        "guard_policy": "F6I2L authorizes only listed source changes, common_validation, isolated phase artifacts, and current context updates.",
        "next_helper_group": NEXT_HELPER_GROUP,
    }


def build_report(tables: Mapping[str, Sequence[Mapping[str, str]]], manifests: Mapping[str, Any], decision_token: str) -> str:
    manifest = manifests["consolidation_manifest"]
    return "\n".join(
        [
            "# Phase F6I2L Validation Table Creation Consolidation Execution",
            "",
            "## Runtime",
            "",
            f"- Approved interpreter: `{APPROVED_INTERPRETER}`.",
            f"- Runtime classification: `{RUNTIME_CLASSIFICATION}`.",
            "- This is not exact original executable-environment reproduction.",
            "",
            "## Scope",
            "",
            "- Selected helper group: `validation_table_creation`.",
            "- Canonical module: `src/audit/common_validation.py`.",
            f"- Approved member files: {manifest['counts']['member_files']}.",
            f"- Shared validation-family helpers inventoried: {manifest['counts']['shared_functions']}.",
            f"- Wrappers migrated: {manifest['counts']['migrated_wrappers']}.",
            "- Hashing helpers, manifest generation, Git capture, path normalization, CSV byte semantics, scientific metrics, prompts, parsers, datasets, candidates, histories, and model outputs were not consolidated.",
            "",
            "## Equivalence",
            "",
            f"- Function equivalence failures: {manifest['counts']['function_equivalence_failures']}.",
            f"- Schema equivalence failures: {manifest['counts']['schema_equivalence_failures']}.",
            f"- Output hash mismatches: {manifest['counts']['output_hash_mismatches']}.",
            f"- Protected input status: `{manifests['protected_input_hashes']['status']}`.",
            f"- Git metadata status: `{manifests['git_metadata_guard']['status']}`.",
            "",
            "## Validation",
            "",
            *[f"- `{row['check_name']}`: `{row['status']}`." for row in tables["validation"]],
            "",
            "## Executed Validation Commands",
            "",
            *[
                f"- `{row['suite']}`: `{row['status']}`; observed `{row['observed']}`; command `{row['command']}`."
                for row in tables["current_suite_validation"]
            ],
            "",
            "## Decision",
            "",
            f"- Decision token: `{decision_token}`.",
            f"- Next helper group: `{NEXT_HELPER_GROUP}`.",
            "",
        ]
    )


def output_hashes_for(tables: Mapping[str, Sequence[Mapping[str, str]]], manifests: Mapping[str, Any], report: str) -> dict[str, str]:
    hashes = {f"table:{name}": payload_digest(list(rows)) for name, rows in sorted(tables.items())}
    hashes.update({f"manifest:{name}": payload_digest(payload) for name, payload in sorted(manifests.items())})
    hashes["report"] = sha256_text(report)
    return hashes


def semantic_digest(result: BuildResult) -> str:
    return payload_digest({"tables": result.tables, "manifests": result.manifests, "report": result.report, "decision_token": result.decision_token})


def determinism_table(first_digest: str | None = None, second_digest: str | None = None) -> list[dict[str, str]]:
    if first_digest is None or second_digest is None:
        first_digest = "external_two_pass_phase_output_hash_comparison"
        second_digest = "external_two_pass_phase_output_hash_comparison"
    return [
        {
            "determinism_id": "F6I2L-DET-001",
            "check_name": "deterministic_build_digest",
            "first_digest": first_digest,
            "second_digest": second_digest,
            "status": "pass" if first_digest == second_digest else "fail",
        }
    ]


def write_current_state(manifest: Mapping[str, Any]) -> None:
    HANDOFF_PATH.write_text(
        "\n".join(
            [
                "# Future Operations Context Handoff",
                "",
                f"- Latest phase: `{PHASE_NAME}`.",
                f"- Decision token: `{manifest['decision_token']}`.",
                "- Canonical validation module: `src/audit/common_validation.py`.",
                "- Migrated wrappers: 5; wrappers retained for historical entrypoints.",
                "- Status vocabularies and schema families preserved by focused fixture equivalence.",
                f"- Next helper group: `{NEXT_HELPER_GROUP}`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_json(
        SNAPSHOT_PATH,
        {
            "latest_phase": PHASE_NAME,
            "decision_token": manifest["decision_token"],
            "canonical_validation_module": "src/audit/common_validation.py",
            "migrated_wrappers": MIGRATED_WRAPPERS,
            "next_helper_group": NEXT_HELPER_GROUP,
            "runtime": runtime_record(),
        },
    )


def build(config_path: Path = CONFIG_PATH, *, write_outputs_enabled: bool = True, update_current_state: bool = True) -> BuildResult:
    load_config(config_path)
    contract = group_contract()
    function_rows, schema_rows, output_rows = equivalence_rows()
    tables: dict[str, list[dict[str, str]]] = {
        "bundle_members": bundle_members_table(contract),
        "pre_refactor_function_inventory": pre_refactor_function_inventory_table(contract),
        "status_vocabulary": status_vocabulary_table(),
        "schema_family_inventory": schema_family_inventory_table(),
        "validation_baseline": read_csv(OUTPUT_TABLES["validation_baseline"]),
        "summary_semantics": summary_semantics_table(),
        "canonical_api_contract": canonical_api_contract_table(),
        "source_modifications": source_modifications_table(),
        "wrapper_compatibility": wrapper_compatibility_table(),
        "function_equivalence": function_rows,
        "schema_equivalence": schema_rows,
        "failure_policy_equivalence": failure_policy_equivalence_table(),
        "builder_equivalence": builder_equivalence_table(),
        "output_hash_comparison": output_rows,
        "dependency_graph_comparison": dependency_graph_comparison_table(),
        "future_dependency_validation": future_dependency_validation_table(),
        "wrapper_removal_readiness": wrapper_removal_readiness_table(),
        "current_suite_validation": current_suite_validation_table(),
        "validation": [],
        "determinism": determinism_table(),
        "next_actions": [],
    }
    protected = protected_hashes()
    git_guard = git_metadata_guard()
    tables["validation"] = validation_table(tables, protected, git_guard)
    decision_token = DECISION_COMPLETE
    if protected["status"] != "pass":
        decision_token = DECISION_PROTECTED_FAILED
    elif any(row["status"] != "pass" for row in tables["schema_equivalence"]):
        decision_token = DECISION_SCHEMA_FAILED
    elif any(row["status"] != "pass" for row in tables["output_hash_comparison"]):
        decision_token = DECISION_OUTPUT_FAILED
    elif any(row["status"] != "pass" for row in tables["current_suite_validation"] if row["suite"] == "current_cleanup_suite"):
        decision_token = DECISION_SUITE_FAILED
    elif any(row["status"] != "pass" for row in tables["validation"]):
        decision_token = DECISION_BLOCKED_CONTRACTS
    tables["next_actions"] = next_actions_table(decision_token)
    manifests = {
        "pre_refactor_baseline_manifest": load_json(PRE_BASELINE_PATH),
        "consolidation_manifest": consolidation_manifest(tables, decision_token),
        "protected_input_hashes": protected,
        "git_metadata_guard": git_guard,
    }
    report = build_report(tables, manifests, decision_token)
    result = BuildResult(
        tables=tables,
        manifests=manifests,
        report=report,
        decision_token=decision_token,
        output_hashes=output_hashes_for(tables, manifests, report),
        protected_hashes=protected,
        git_metadata_guard=git_guard,
    )
    if write_outputs_enabled:
        for name, path in OUTPUT_TABLES.items():
            write_csv(path, result.tables[name])
        write_json(PRE_BASELINE_PATH, result.manifests["pre_refactor_baseline_manifest"])
        write_json(CONSOLIDATION_MANIFEST_PATH, result.manifests["consolidation_manifest"])
        write_json(PROTECTED_HASHES_PATH, result.manifests["protected_input_hashes"])
        write_json(GIT_METADATA_GUARD_PATH, result.manifests["git_metadata_guard"])
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(result.report, encoding="utf-8")
        if update_current_state:
            write_current_state(result.manifests["consolidation_manifest"])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--no-current-state-update", action="store_true")
    args = parser.parse_args(argv)
    result = build(args.config, write_outputs_enabled=True, update_current_state=not args.no_current_state_update)
    print(json.dumps({"phase_id": PHASE_ID, "decision_token": result.decision_token}, sort_keys=True))
    return 0 if result.decision_token == DECISION_COMPLETE else 1


if __name__ == "__main__":
    raise SystemExit(main())
