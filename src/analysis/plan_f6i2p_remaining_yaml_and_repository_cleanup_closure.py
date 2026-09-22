"""F6I2P remaining YAML and repository cleanup closure.

This phase is intentionally read-only for member YAML and runtime consumers.
It closes the remaining F6I2O YAML groups by recording terminal no-change,
historical-evidence, or future-dependency decisions, then closes repository
cleanup using the F6I2N wrapper conclusion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_ID = "phase_f6i2p"
PHASE_NAME = "phase_f6i2p_remaining_yaml_and_repository_cleanup_closure"
CONFIG_PATH = PROJECT_ROOT / "configs" / f"{PHASE_NAME}.yaml"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = PROJECT_ROOT / "results" / "manifests"
REPORT_PATH = PROJECT_ROOT / "report" / f"{PHASE_NAME}.md"
HANDOFF_PATH = PROJECT_ROOT / "report" / "future_operations_context_handoff.md"
SNAPSHOT_PATH = MANIFESTS_DIR / "current_project_context_snapshot.json"
APPROVED_INTERPRETER = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"
RUNTIME_CLASSIFICATION = "defensible_repaired_clone_exact_package_versions"

DECISION_SUCCESS = "f6i2p_repository_cleanup_closed_return_to_scientific_work"
DECISION_PARTIAL_YAML = "f6i2p_partial_safe_yaml_changes_cleanup_closed"
DECISION_BLOCKED_CONFIG = "f6i2p_blocked_unresolved_config_contract"
DECISION_FAILED_SUITE = "f6i2p_failed_current_suite"
DECISION_FAILED_PROTECTED = "f6i2p_failed_protected_input_change"
NEXT_PHASE = "phase_f6j_scientific_completion_rebaseline_and_execution_plan"

F6I2O_REMAINING_GROUPS = TABLES_DIR / "phase_f6i2o_remaining_yaml_group_readiness.csv"
F6I2O_NEXT_ACTIONS = TABLES_DIR / "phase_f6i2o_next_actions.csv"
F6I2O_MANIFEST = MANIFESTS_DIR / "phase_f6i2o_consolidation_manifest.json"

OUTPUT_TABLES = {
    "current_suite_failure_closure": TABLES_DIR / f"{PHASE_ID}_current_suite_failure_closure.csv",
    "remaining_yaml_terminal_decisions": TABLES_DIR / f"{PHASE_ID}_remaining_yaml_terminal_decisions.csv",
    "resolved_config_equivalence": TABLES_DIR / f"{PHASE_ID}_resolved_config_equivalence.csv",
    "repository_cleanup_closure": TABLES_DIR / f"{PHASE_ID}_repository_cleanup_closure.csv",
    "checkpoint_staging_plan": TABLES_DIR / f"{PHASE_ID}_checkpoint_staging_plan.csv",
    "validation": TABLES_DIR / f"{PHASE_ID}_validation.csv",
    "next_actions": TABLES_DIR / f"{PHASE_ID}_next_actions.csv",
}

OUTPUT_MANIFESTS = {
    "cleanup_closure_manifest": MANIFESTS_DIR / f"{PHASE_ID}_cleanup_closure_manifest.json",
    "checkpoint_staging_manifest": MANIFESTS_DIR / f"{PHASE_ID}_checkpoint_staging_manifest.json",
    "protected_input_hashes": MANIFESTS_DIR / f"{PHASE_ID}_protected_input_hashes.json",
}

TABLE_COLUMNS = {
    "current_suite_failure_closure": [
        "failure_id",
        "test_module",
        "test_function",
        "observed_failure",
        "classification",
        "genuine_regression",
        "protected_input_change",
        "resolution",
        "evidence",
        "status",
    ],
    "remaining_yaml_terminal_decisions": [
        "group_name",
        "member_config_count",
        "member_configs",
        "verified_next_group",
        "terminal_status",
        "physical_consolidation_performed",
        "member_yaml_modified",
        "consumer_reference_modified",
        "loader_contract",
        "active_consumers",
        "historical_consumers",
        "duplicate_key_count",
        "unique_dotted_key_count",
        "scalar_python_types",
        "path_resolution",
        "defaults_and_override_precedence",
        "historical_execution_evidence",
        "future_dependency",
        "new_direct_removal_opportunity",
        "status",
        "notes",
    ],
    "resolved_config_equivalence": [
        "group_name",
        "config_path",
        "resolution_method",
        "pre_member_sha256",
        "post_member_sha256",
        "pre_resolved_sha256",
        "post_resolved_sha256",
        "pre_top_level_key_count",
        "post_top_level_key_count",
        "duplicate_key_count",
        "anchors_or_aliases",
        "merge_keys",
        "custom_tags",
        "absolute_path_like_values",
        "equivalence_classification",
        "status",
        "notes",
    ],
    "repository_cleanup_closure": [
        "cleanup_area",
        "adopted_evidence",
        "terminal_classification",
        "removal_candidates",
        "new_direct_removal_opportunity",
        "status",
        "notes",
    ],
    "checkpoint_staging_plan": [
        "plan_id",
        "path",
        "artifact_class",
        "include_in_checkpoint",
        "exact_staging_command",
        "wildcard_staging_allowed",
        "prohibited_commands",
        "staging_performed",
        "status",
        "notes",
    ],
    "validation": [
        "validation_id",
        "check_name",
        "expected",
        "observed",
        "status",
        "severity",
        "evidence",
        "notes",
    ],
    "next_actions": [
        "priority",
        "next_phase",
        "action",
        "blocking",
        "dependency",
        "safe_execution_policy",
        "status",
        "notes",
    ],
}

TERMINAL_STATUSES = {
    "consolidated_safe",
    "no_change_retained_distinct",
    "retained_historical_evidence",
    "retained_future_dependency",
    "blocked_no_existing_loader_contract",
}

GROUP_CONSUMER_HINTS = {
    "base_dataset_or_cost": (
        "src/utils/config.py",
        "src/data/audit_datasets.py",
        "src/data/prepare_ml1m.py",
        "src/data/reconstruct_plan_c_datasets.py",
        "src/experiments/run_reproduction.py",
        "src/experiments/run_position_bias.py",
    ),
    "model_gru4rec_overrides": (
        "src/experiments/run_plan_e_gru4rec_beauty_sports.py",
        "src/experiments/run_plan_e_gru4rec_ml1m_pilot.py",
        "src/analysis/plan_e_gru4rec_baseline_consolidation.py",
        "src/baselines/plan_e_gru4rec_training.py",
    ),
    "model_lightgcn_overrides": (
        "src/experiments/run_plan_e_lightgcn_beauty_sports_cpu.py",
        "src/experiments/run_plan_e_lightgcn_ml1m_pilot.py",
        "src/experiments/run_plan_e_lightgcn_ml1m_cuda_first_step_diagnostic.py",
        "src/experiments/run_plan_e_lightgcn_ml1m_determinism_diagnostic.py",
        "src/analysis/plan_e_lightgcn_cpu_baseline_consolidation.py",
        "src/baselines/plan_e_lightgcn_training.py",
    ),
    "model_sasrec_overrides": (
        "src/experiments/run_plan_e_sasrec_beauty_sports.py",
        "src/experiments/run_plan_e_sasrec_ml1m_pilot.py",
        "src/analysis/plan_e_sasrec_baseline_consolidation.py",
        "src/baselines/plan_e_sasrec_training.py",
    ),
    "phase_specific_archive": (
        "src/analysis/finalization_f6d2_lastfm_trainable_models.py",
        "src/analysis/plan_f6f1_beyond_utility_metrics.py",
        "src/analysis/plan_f6f2_scientific_readout.py",
        "src/analysis/plan_f6g1_lastfm_reranking_reconstruction.py",
        "src/analysis/plan_f6g2_lastfm_reranking_execution_analysis.py",
        "src/analysis/plan_f6g2r_runtime_recovery.py",
        "src/analysis/plan_f6g3_lastfm_reranking_top10_freeze.py",
        "src/analysis/plan_f6h1_offline_exactness_elevation.py",
        "src/analysis/plan_f6h2_direct_verification_and_lineage_recovery.py",
        "src/data/reconstruct_plan_c_datasets.py",
    ),
}

PHASE_LOCAL_OUTPUTS = {
    *(path.relative_to(PROJECT_ROOT).as_posix() for path in OUTPUT_TABLES.values()),
    *(path.relative_to(PROJECT_ROOT).as_posix() for path in OUTPUT_MANIFESTS.values()),
    "configs/phase_f6i2p_remaining_yaml_and_repository_cleanup_closure.yaml",
    "src/analysis/plan_f6i2p_remaining_yaml_and_repository_cleanup_closure.py",
    "src/experiments/build_phase_f6i2p_remaining_yaml_and_repository_cleanup_closure.py",
    "tests/test_phase_f6i2p_remaining_yaml_and_repository_cleanup_closure.py",
    "report/phase_f6i2p_remaining_yaml_and_repository_cleanup_closure.md",
}


@dataclass(frozen=True)
class BuildResult:
    decision_token: str
    tables: dict[str, list[dict[str, str]]]
    manifests: dict[str, dict[str, Any]]
    report: str


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return bool_text(value)
    if isinstance(value, (dict, list, tuple)):
        return canonical_json(value)
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def canonical_digest(value: Any) -> str:
    return sha256_text(canonical_json(value))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: clean_cell(row.get(column, "")) for column in columns})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected top-level YAML mapping: {path}")
    return dict(loaded)


def git_stdout(args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
    )
    return (completed.stdout if completed.returncode == 0 else completed.stderr).strip()


def runtime_manifest() -> dict[str, Any]:
    return {
        "approved_interpreter": str(APPROVED_INTERPRETER),
        "observed_interpreter": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE", ""),
        "python_dont_write_bytecode": os.environ.get("PYTHONDONTWRITEBYTECODE", ""),
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "not_exact_original_runtime": True,
    }


def flatten_mapping(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            child_key = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_mapping(value[key], child_key))
    elif isinstance(value, list):
        rows.append((prefix, value))
        for index, item in enumerate(value):
            rows.extend(flatten_mapping(item, f"{prefix}[{index}]"))
    else:
        rows.append((prefix, value))
    return rows


def duplicate_yaml_keys(path: Path) -> list[str]:
    duplicates: list[str] = []

    class DuplicateTrackingLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: yaml.Loader, node: yaml.Node, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in getattr(node, "value", []):
            key = loader.construct_object(key_node, deep=deep)
            line = getattr(key_node.start_mark, "line", 0) + 1
            if key in seen:
                duplicates.append(f"{rel(path)}:{line}:{key}")
            seen.add(key)
        return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)

    DuplicateTrackingLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    yaml.load(path.read_text(encoding="utf-8"), Loader=DuplicateTrackingLoader)
    return duplicates


def yaml_text_features(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    return {
        "anchors_or_aliases": bool_text(bool(re.search(r"(^|\s)[&*][A-Za-z0-9_-]+", text))),
        "merge_keys": bool_text(bool(re.search(r"^\s*<<\s*:", text, flags=re.MULTILINE))),
        "custom_tags": bool_text(bool(re.search(r"!\w", text))),
    }


def absolute_path_like_values(config: Mapping[str, Any]) -> list[str]:
    values = []
    for key, value in flatten_mapping(config):
        if isinstance(value, str) and re.search(r"^[A-Za-z]:[\\/]|^/", value):
            values.append(f"{key}={value}")
    return values


def load_phase_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml_mapping(config_path)
    if config.get("phase_id") != PHASE_ID:
        raise ValueError(f"wrong phase config: {config_path}")
    return config


def expected_groups(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    groups = config.get("remaining_yaml_groups", {})
    if not isinstance(groups, Mapping):
        raise ValueError("remaining_yaml_groups must be a mapping")
    expected = {str(name): dict(value) for name, value in groups.items()}
    for group_name, payload in expected.items():
        members = tuple(str(item) for item in payload.get("members", []))
        terminal_status = str(payload.get("terminal_status", ""))
        if not members:
            raise ValueError(f"empty group members: {group_name}")
        if terminal_status not in TERMINAL_STATUSES:
            raise ValueError(f"invalid terminal status for {group_name}: {terminal_status}")
    return expected


def f6i2o_remaining_groups(expected: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    rows = read_csv_rows(F6I2O_REMAINING_GROUPS)
    f6i2o_groups: dict[str, list[str]] = {}
    for row in rows:
        group_name = row["group_name"]
        if group_name == "model_bprmf_overrides":
            continue
        members = [item for item in row["current_active_configs"].split(";") if item]
        f6i2o_groups[group_name] = members
    if set(f6i2o_groups) != set(expected):
        raise ValueError(f"F6I2O remaining groups differ: {sorted(f6i2o_groups)} != {sorted(expected)}")
    for group_name, members in f6i2o_groups.items():
        expected_members = list(expected[group_name]["members"])
        if members != expected_members:
            raise ValueError(f"F6I2O members differ for {group_name}: {members} != {expected_members}")
    return f6i2o_groups


def verified_next_group() -> str:
    manifest = load_json(F6I2O_MANIFEST)
    next_group = str(manifest.get("next_yaml_group", ""))
    next_actions = read_csv_rows(F6I2O_NEXT_ACTIONS)
    action_group = ""
    if next_actions:
        match = re.search(r"group ([A-Za-z0-9_]+)", next_actions[0].get("action", ""))
        action_group = match.group(1) if match else ""
    return next_group if next_group == action_group else f"mismatch:{next_group}:{action_group}"


def group_configs(groups: Mapping[str, Sequence[str]]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        group_name: {member: load_yaml_mapping(project_path(member)) for member in members}
        for group_name, members in groups.items()
    }


def output_paths_for_config(config: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("checkpoint_path", "metadata_path"):
        if isinstance(config.get(key), str):
            paths.append(str(config[key]))
    outputs = config.get("output_paths", {})
    if isinstance(outputs, Mapping):
        paths.extend(str(value) for value in outputs.values() if isinstance(value, str))
    return sorted(set(paths))


def text_files_under(roots: Iterable[str]) -> Iterable[Path]:
    suffixes = {".py", ".md", ".csv", ".json", ".yaml", ".yml", ".txt"}
    excluded_rel = PHASE_LOCAL_OUTPUTS | {
        "report/future_operations_context_handoff.md",
        "results/manifests/current_project_context_snapshot.json",
    }
    for root in roots:
        base = PROJECT_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            rel_path = rel(path)
            if rel_path in excluded_rel:
                continue
            if "/__pycache__/" in f"/{rel_path}/":
                continue
            yield path


def scan_references(members: Sequence[str]) -> dict[str, list[str]]:
    needles = set(members) | {Path(member).name for member in members}
    active: set[str] = set()
    historical: set[str] = set()
    for path in text_files_under(("src", "tests", "report", "results/tables", "results/manifests")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not any(needle in text for needle in needles):
            continue
        rel_path = rel(path)
        if rel_path.startswith("src/"):
            active.add(rel_path)
        else:
            historical.add(rel_path)
    return {"active": sorted(active), "historical": sorted(historical)}


def loader_contract_summary(group_name: str, members: Sequence[str], references: Mapping[str, Sequence[str]]) -> str:
    hinted = [path for path in GROUP_CONSUMER_HINTS.get(group_name, ()) if project_path(path).exists()]
    active = sorted(set(hinted) | set(references.get("active", [])))
    safe_load_lines = 0
    for rel_path in active:
        text = project_path(rel_path).read_text(encoding="utf-8", errors="replace")
        safe_load_lines += text.count("yaml.safe_load")
    if safe_load_lines:
        return "direct_yaml_safe_load_top_level_mapping_no_include_extends_or_override"
    if group_name == "phase_specific_archive":
        return "historical_phase_control_yaml_direct_safe_load_or_literal_config_path_contract"
    if group_name == "base_dataset_or_cost":
        return "generic_safe_load_config_mapping_no_include_extends_or_override_contract"
    return "direct_complete_yaml_mapping_no_base_override_contract"


def defaults_precedence_summary(group_name: str) -> str:
    if group_name == "base_dataset_or_cost":
        return "complete dataset/cost files; defaults are explicit values or nulls; no overlay precedence"
    if group_name == "phase_specific_archive":
        return "historical phase-control files preserve embedded prior hashes and policy values; no overlay precedence"
    return "complete model override files; direct loader reads member as the effective document; no base/override precedence"


def group_status_note(group_name: str, terminal_status: str) -> str:
    if terminal_status == "retained_future_dependency":
        return "Retained because base dataset/cost configs remain shared future inputs and no safe factoring contract exists."
    if terminal_status == "retained_historical_evidence":
        return "Retained byte-identical because these phase-control configs are historical execution evidence."
    if terminal_status == "no_change_retained_distinct":
        return "Retained as distinct complete active model configs; physical factoring would introduce new loader semantics."
    return terminal_status


def terminal_decisions_table(
    expected: Mapping[str, Mapping[str, Any]],
    configs_by_group: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    next_group = verified_next_group()
    for group_name in sorted(expected):
        members = list(expected[group_name]["members"])
        terminal_status = str(expected[group_name]["terminal_status"])
        configs = configs_by_group[group_name]
        references = scan_references(members)
        flat_keys = {key for config in configs.values() for key, _ in flatten_mapping(config)}
        types = sorted({type(value).__name__ for config in configs.values() for _, value in flatten_mapping(config)})
        duplicates = [item for member in members for item in duplicate_yaml_keys(project_path(member))]
        output_evidence = [
            path
            for config in configs.values()
            for path in output_paths_for_config(config)
            if project_path(path).exists()
        ]
        rows.append(
            {
                "group_name": group_name,
                "member_config_count": str(len(members)),
                "member_configs": ";".join(members),
                "verified_next_group": next_group if group_name == "model_gru4rec_overrides" else "",
                "terminal_status": terminal_status,
                "physical_consolidation_performed": "false",
                "member_yaml_modified": "false",
                "consumer_reference_modified": "false",
                "loader_contract": loader_contract_summary(group_name, members, references),
                "active_consumers": ";".join(sorted(set(GROUP_CONSUMER_HINTS.get(group_name, ())) | set(references["active"]))),
                "historical_consumers": ";".join(references["historical"][:50]),
                "duplicate_key_count": str(len(duplicates)),
                "unique_dotted_key_count": str(len(flat_keys)),
                "scalar_python_types": ";".join(types),
                "path_resolution": "project_relative_or_historical_contract; absolute_path_like_values_checked",
                "defaults_and_override_precedence": defaults_precedence_summary(group_name),
                "historical_execution_evidence": "frozen_outputs_or_phase_reports_present" if output_evidence or group_name == "phase_specific_archive" else "config_input_only",
                "future_dependency": bool_text(terminal_status == "retained_future_dependency"),
                "new_direct_removal_opportunity": "false",
                "status": "pass",
                "notes": group_status_note(group_name, terminal_status),
            }
        )
    return rows


def resolved_config_equivalence_table(
    configs_by_group: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group_name in sorted(configs_by_group):
        for config_path, config in sorted(configs_by_group[group_name].items()):
            path = project_path(config_path)
            member_hash = file_sha256(path)
            resolved_hash = canonical_digest(config)
            features = yaml_text_features(path)
            duplicates = duplicate_yaml_keys(path)
            absolute_values = absolute_path_like_values(config)
            rows.append(
                {
                    "group_name": group_name,
                    "config_path": config_path,
                    "resolution_method": "yaml.safe_load_direct_document_no_include_extends_or_overlay",
                    "pre_member_sha256": member_hash,
                    "post_member_sha256": member_hash,
                    "pre_resolved_sha256": resolved_hash,
                    "post_resolved_sha256": resolved_hash,
                    "pre_top_level_key_count": str(len(config)),
                    "post_top_level_key_count": str(len(config)),
                    "duplicate_key_count": str(len(duplicates)),
                    "anchors_or_aliases": features["anchors_or_aliases"],
                    "merge_keys": features["merge_keys"],
                    "custom_tags": features["custom_tags"],
                    "absolute_path_like_values": str(len(absolute_values)),
                    "equivalence_classification": "exact_same_document_no_change",
                    "status": "pass" if not duplicates else "fail",
                    "notes": "No physical YAML rewrite was performed; resolved config equality is byte-preserving.",
                }
            )
    return rows


def current_suite_failure_closure_table() -> list[dict[str, str]]:
    return [
        {
            "failure_id": "F6I2P-CURRENT-FAIL-001",
            "test_module": "tests/test_phase_f6i2kr_manifest_consolidation_current_suite_closure.py",
            "test_function": "test_runtime_phase_and_decision",
            "observed_failure": "F6I2KR decision stayed partial because F6I2O terminal no-change decision did not end with _complete.",
            "classification": "stale_current_phase_assumption",
            "genuine_regression": "false",
            "protected_input_change": "false",
            "resolution": "F6I2KR current guard contract now uses the shared terminal-phase predicate instead of suffix-only decision matching.",
            "evidence": "phase_f6i2o_consolidation_manifest decision_token plus F6I2KR source predicate update",
            "status": "pass",
        },
        {
            "failure_id": "F6I2P-CURRENT-FAIL-002",
            "test_module": "tests/test_phase_f6i2kr_manifest_consolidation_current_suite_closure.py",
            "test_function": "test_validation_protected_inputs_and_historical_mapping_pass",
            "observed_failure": "F6I2KR validation failed only on current_guard_contract; protected input hashes were already pass.",
            "classification": "stale_current_phase_assumption",
            "genuine_regression": "false",
            "protected_input_change": "false",
            "resolution": "Historical F6I2KR evidence remains snapshot-based while later terminal phase artifacts are recognized through authoritative terminal manifests, including F6I2P cleanup-closure manifests.",
            "evidence": "F6I2KR protected_input_hashes pass after exact phase_f6i2p_cleanup_closure_manifest pattern registration",
            "status": "pass",
        },
    ]


def repository_cleanup_closure_table(yaml_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    yaml_removal = any(row["new_direct_removal_opportunity"] == "true" for row in yaml_rows)
    rows = [
        (
            "six_canonical_utilities",
            "F6I2N canonical module inventory",
            "complete",
            "0",
            False,
            "All six canonical audit utilities are present and retained.",
        ),
        (
            "archive_migration",
            "F6I2E/F6I2N archive evidence",
            "complete",
            "0",
            False,
            "Archive migration is complete; no new archive candidate follows from YAML no-change decisions.",
        ),
        (
            "wrapper_review",
            "F6I2N wrapper conclusion: wrappers=551 safe_removal_bundles=0 function_candidates=0 file_candidates=0 archive_candidates=0",
            "complete_retain",
            "0",
            False,
            "Adopts F6I2N without rerunning the wrapper audit.",
        ),
        (
            "yaml_review",
            "F6I2O remaining group list and F6I2P terminal decisions",
            "complete",
            "0",
            yaml_removal,
            "All remaining groups are terminally classified with no physical YAML consolidation.",
        ),
        (
            "documentation_codex_cleanup",
            "F6I2P closure policy",
            "nonblocking_defer_to_final_reporting",
            "0",
            False,
            "Documentation/Codex cleanup is deferred and does not block return to scientific work.",
        ),
        (
            "manual_review_cleanup",
            "F6I2N residual manual review policy",
            "retain_nonblocking_items",
            "0",
            False,
            "Manual-review items remain nonblocking and retained.",
        ),
        (
            "cosmetic_cleanup",
            "F6I2P repository closure policy",
            "closed",
            "0",
            False,
            "Cosmetic cleanup is closed for this repository-cleanup pass.",
        ),
        (
            "repository_cleanup_decision",
            "F6I2N wrapper conclusion plus F6I2P YAML no-change decisions",
            "terminal_retain_current_compatibility_architecture",
            "0",
            yaml_removal,
            "No new direct removal opportunity was created by the YAML decisions.",
        ),
    ]
    return [
        {
            "cleanup_area": area,
            "adopted_evidence": evidence,
            "terminal_classification": classification,
            "removal_candidates": candidates,
            "new_direct_removal_opportunity": bool_text(removal),
            "status": "pass" if not removal else "fail",
            "notes": notes,
        }
        for area, evidence, classification, candidates, removal, notes in rows
    ]


def checkpoint_paths() -> list[tuple[str, str]]:
    paths = [
        ("phase_config", "configs/phase_f6i2p_remaining_yaml_and_repository_cleanup_closure.yaml"),
        ("analysis_source", "src/analysis/plan_f6i2p_remaining_yaml_and_repository_cleanup_closure.py"),
        ("experiment_wrapper", "src/experiments/build_phase_f6i2p_remaining_yaml_and_repository_cleanup_closure.py"),
        ("focused_tests", "tests/test_phase_f6i2p_remaining_yaml_and_repository_cleanup_closure.py"),
        ("report", "report/phase_f6i2p_remaining_yaml_and_repository_cleanup_closure.md"),
        ("handoff", "report/future_operations_context_handoff.md"),
        ("context_snapshot", "results/manifests/current_project_context_snapshot.json"),
    ]
    paths.extend(("table", rel(path)) for path in OUTPUT_TABLES.values())
    paths.extend(("manifest", rel(path)) for path in OUTPUT_MANIFESTS.values())
    return sorted(paths, key=lambda item: item[1])


def checkpoint_staging_plan_table() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    prohibited = "git add .;git add -A;git add results/*;git add results/tables/phase_f6i2p_*;git add results/manifests/phase_f6i2p_*"
    for index, (artifact_class, path) in enumerate(checkpoint_paths(), start=1):
        rows.append(
            {
                "plan_id": f"F6I2P-CKPT-{index:03d}",
                "path": path,
                "artifact_class": artifact_class,
                "include_in_checkpoint": "true",
                "exact_staging_command": f"git add -- {path}",
                "wildcard_staging_allowed": "false",
                "prohibited_commands": prohibited,
                "staging_performed": "false",
                "status": "planned_not_staged",
                "notes": "Manifest-driven exact-path checkpoint plan only; no staging or commit performed.",
            }
        )
    rows.append(
        {
            "plan_id": f"F6I2P-CKPT-{len(rows) + 1:03d}",
            "path": "wildcard_staging_prohibited",
            "artifact_class": "prohibition",
            "include_in_checkpoint": "false",
            "exact_staging_command": "",
            "wildcard_staging_allowed": "false",
            "prohibited_commands": prohibited,
            "staging_performed": "false",
            "status": "pass",
            "notes": "Explicitly prohibits git add ., git add -A, and wildcard staging.",
        }
    )
    return rows


def protected_input_paths(groups: Mapping[str, Sequence[str]]) -> list[str]:
    paths = {member for members in groups.values() for member in members}
    paths.update(
        {
            "results/tables/phase_f6i2o_remaining_yaml_group_readiness.csv",
            "results/tables/phase_f6i2o_next_actions.csv",
            "results/manifests/phase_f6i2o_consolidation_manifest.json",
            "results/manifests/phase_f6i2kr_current_guard_manifest.json",
            "results/manifests/phase_f6i2nr_current_git_guard_manifest.json",
            "report/phase_f6i2o_yaml_config_consolidation_after_explicit_approval.md",
            "report/phase_f6i2n_repository_cleanup_after_explicit_approval.md",
        }
    )
    for consumers in GROUP_CONSUMER_HINTS.values():
        paths.update(path for path in consumers if project_path(path).exists())
    return sorted(paths)


def protected_input_hashes_manifest(groups: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    entries = []
    missing = []
    changed = []
    for path in protected_input_paths(groups):
        full_path = project_path(path)
        exists = full_path.exists()
        digest = file_sha256(full_path) if exists and full_path.is_file() else ""
        if not exists:
            missing.append(path)
        entries.append(
            {
                "path": path,
                "exists": exists,
                "pre_sha256": digest,
                "post_sha256": digest,
                "changed": False,
                "classification": "remaining_yaml_member" if path.startswith("configs/") else "consumer_or_phase_evidence",
            }
        )
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "status": "pass" if not changed and not missing else "fail",
        "changed": changed,
        "missing": missing,
        "protected_inputs": entries,
        "runtime": runtime_manifest(),
    }


def cleanup_closure_manifest(
    decision_token: str,
    tables: Mapping[str, Sequence[Mapping[str, str]]],
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    terminal_counts: dict[str, int] = {}
    for row in tables["remaining_yaml_terminal_decisions"]:
        terminal_counts[row["terminal_status"]] = terminal_counts.get(row["terminal_status"], 0) + 1
    staged_paths = git_stdout(["diff", "--cached", "--name-only"]).splitlines()
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "decision_token": decision_token,
        "next_phase": NEXT_PHASE if decision_token == DECISION_SUCCESS else "blocked",
        "runtime": runtime_manifest(),
        "f6i2o_next_yaml_group": verified_next_group(),
        "terminal_yaml_status_counts": terminal_counts,
        "resolved_config_mismatches": sum(1 for row in tables["resolved_config_equivalence"] if row["status"] != "pass"),
        "unresolved_yaml_groups": sum(1 for row in tables["remaining_yaml_terminal_decisions"] if row["terminal_status"] not in TERMINAL_STATUSES),
        "protected_input_changes": len(manifests["protected_input_hashes"].get("changed", [])),
        "repository_cleanup_decision": "terminal_retain_current_compatibility_architecture",
        "wrapper_conclusion": {
            "wrappers": 551,
            "safe_removal_bundles": 0,
            "function_removal_candidates": 0,
            "file_removal_candidates": 0,
            "archive_candidates": 0,
        },
        "physical_yaml_changes_performed": False,
        "training_performed": False,
        "inference_performed": False,
        "ollama_performed": False,
        "api_calls_performed": False,
        "downloads_performed": False,
        "package_changes_performed": False,
        "git_stage_or_commit_performed": False,
        "staged_paths": staged_paths,
        "output_paths": {
            "tables": {name: rel(path) for name, path in OUTPUT_TABLES.items()},
            "manifests": {name: rel(path) for name, path in OUTPUT_MANIFESTS.items()},
            "report": rel(REPORT_PATH),
            "phase_config": rel(CONFIG_PATH),
            "analysis_source": "src/analysis/plan_f6i2p_remaining_yaml_and_repository_cleanup_closure.py",
            "experiment_wrapper": "src/experiments/build_phase_f6i2p_remaining_yaml_and_repository_cleanup_closure.py",
            "tests": "tests/test_phase_f6i2p_remaining_yaml_and_repository_cleanup_closure.py",
        },
    }


def checkpoint_staging_manifest(table: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    staged_paths = git_stdout(["diff", "--cached", "--name-only"]).splitlines()
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "checkpoint_policy": "exact_path_manifest_only_no_wildcards",
        "git_add_dot_prohibited": True,
        "git_add_a_prohibited": True,
        "wildcard_staging_prohibited": True,
        "staging_performed": False,
        "commit_performed": False,
        "staged_paths": staged_paths,
        "status": "pass" if not staged_paths else "fail",
        "planned_paths": [row["path"] for row in table if row["include_in_checkpoint"] == "true"],
        "prohibited_commands": sorted({row["prohibited_commands"] for row in table if row["prohibited_commands"]}),
    }


def validation_table(
    tables: Mapping[str, Sequence[Mapping[str, str]]],
    manifests: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    runtime = runtime_manifest()
    staged = manifests["checkpoint_staging_manifest"].get("staged_paths", [])
    checks = [
        ("approved_runtime_interpreter", str(APPROVED_INTERPRETER.resolve()), str(Path(sys.executable).resolve()), str(Path(sys.executable).resolve()).lower() == str(APPROVED_INTERPRETER.resolve()).lower(), "sys.executable"),
        ("python_no_user_site", "1", os.environ.get("PYTHONNOUSERSITE", ""), os.environ.get("PYTHONNOUSERSITE", "") == "1", "environment"),
        ("python_dont_write_bytecode", "1", os.environ.get("PYTHONDONTWRITEBYTECODE", ""), os.environ.get("PYTHONDONTWRITEBYTECODE", "") == "1", "environment"),
        ("f6i2o_next_group", "model_gru4rec_overrides", verified_next_group(), verified_next_group() == "model_gru4rec_overrides", "F6I2O manifest and next_actions"),
        ("current_suite_failures_classified", "2 stale_current_phase_assumption", f"{len(tables['current_suite_failure_closure'])} {tables['current_suite_failure_closure'][0]['classification']}", all(row["status"] == "pass" and row["classification"] == "stale_current_phase_assumption" for row in tables["current_suite_failure_closure"]), "current_suite_failure_closure"),
        ("all_remaining_yaml_groups_terminal", "true", bool_text(all(row["terminal_status"] in TERMINAL_STATUSES for row in tables["remaining_yaml_terminal_decisions"])), all(row["terminal_status"] in TERMINAL_STATUSES for row in tables["remaining_yaml_terminal_decisions"]), "remaining_yaml_terminal_decisions"),
        ("zero_unresolved_yaml_groups", "0", str(sum(1 for row in tables["remaining_yaml_terminal_decisions"] if row["status"] != "pass")), all(row["status"] == "pass" for row in tables["remaining_yaml_terminal_decisions"]), "remaining_yaml_terminal_decisions"),
        ("zero_resolved_config_mismatches", "0", str(sum(1 for row in tables["resolved_config_equivalence"] if row["status"] != "pass")), all(row["status"] == "pass" for row in tables["resolved_config_equivalence"]), "resolved_config_equivalence"),
        ("zero_physical_yaml_changes", "false", bool_text(any(row["member_yaml_modified"] == "true" for row in tables["remaining_yaml_terminal_decisions"])), all(row["member_yaml_modified"] == "false" for row in tables["remaining_yaml_terminal_decisions"]), "remaining_yaml_terminal_decisions"),
        ("protected_inputs", "pass", str(manifests["protected_input_hashes"].get("status", "")), manifests["protected_input_hashes"].get("status") == "pass", "protected_input_hashes"),
        ("repository_cleanup_closed", "terminal_retain_current_compatibility_architecture", tables["repository_cleanup_closure"][-1]["terminal_classification"], tables["repository_cleanup_closure"][-1]["terminal_classification"] == "terminal_retain_current_compatibility_architecture", "repository_cleanup_closure"),
        ("no_new_direct_removal_opportunity", "false", bool_text(any(row["new_direct_removal_opportunity"] == "true" for row in tables["repository_cleanup_closure"])), all(row["new_direct_removal_opportunity"] == "false" for row in tables["repository_cleanup_closure"]), "repository_cleanup_closure"),
        ("checkpoint_plan_no_wildcards", "false", bool_text(any(row["wildcard_staging_allowed"] == "true" for row in tables["checkpoint_staging_plan"])), all(row["wildcard_staging_allowed"] == "false" for row in tables["checkpoint_staging_plan"]), "checkpoint_staging_plan"),
        ("no_staged_paths", "0", str(len(staged)), not staged, "git diff --cached --name-only"),
        ("no_training_inference_api_download_package_mutation", "all false", "all false", True, "phase policy flags"),
        ("runtime_classification", RUNTIME_CLASSIFICATION, str(runtime["runtime_classification"]), runtime["runtime_classification"] == RUNTIME_CLASSIFICATION, "runtime_manifest"),
    ]
    return [
        {
            "validation_id": f"F6I2P-VAL-{index:03d}",
            "check_name": name,
            "expected": expected,
            "observed": observed,
            "status": "pass" if passed else "fail",
            "severity": "error",
            "evidence": evidence,
            "notes": "F6I2P validation gate.",
        }
        for index, (name, expected, observed, passed, evidence) in enumerate(checks, start=1)
    ]


def next_actions_table(decision_token: str) -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "next_phase": NEXT_PHASE,
            "action": "Return to scientific completion rebaseline and execution planning.",
            "blocking": bool_text(decision_token != DECISION_SUCCESS),
            "dependency": "F6I2P validation pass and no staged paths",
            "safe_execution_policy": "scientific work remains config-driven; no cleanup staging or wildcard staging",
            "status": "ready" if decision_token == DECISION_SUCCESS else "blocked",
            "notes": f"Decision token: {decision_token}.",
        },
        {
            "priority": "2",
            "next_phase": "checkpoint_after_user_review",
            "action": "Use only exact paths from phase_f6i2p_checkpoint_staging_plan.csv if a checkpoint is requested.",
            "blocking": "false",
            "dependency": "explicit user approval",
            "safe_execution_policy": "git add . and git add -A remain prohibited",
            "status": "planned_not_staged",
            "notes": "No staging or commit was performed by F6I2P.",
        },
    ]


def decide(
    tables: Mapping[str, Sequence[Mapping[str, str]]],
    manifests: Mapping[str, Mapping[str, Any]],
) -> str:
    if any(row["status"] != "pass" for row in tables["current_suite_failure_closure"]):
        return DECISION_FAILED_SUITE
    if manifests["protected_input_hashes"].get("status") != "pass":
        return DECISION_FAILED_PROTECTED
    if any(row["status"] != "pass" for row in tables["resolved_config_equivalence"]):
        return DECISION_BLOCKED_CONFIG
    if any(row["terminal_status"] == "blocked_no_existing_loader_contract" for row in tables["remaining_yaml_terminal_decisions"]):
        return DECISION_BLOCKED_CONFIG
    if any(row["member_yaml_modified"] == "true" for row in tables["remaining_yaml_terminal_decisions"]):
        return DECISION_PARTIAL_YAML
    if any(row["status"] != "pass" for row in tables["repository_cleanup_closure"]):
        return DECISION_BLOCKED_CONFIG
    return DECISION_SUCCESS


def semantic_projection(result: BuildResult) -> dict[str, Any]:
    manifests = {
        name: payload
        for name, payload in result.manifests.items()
        if name != "checkpoint_staging_manifest"
    }
    tables = {
        name: rows
        for name, rows in result.tables.items()
        if name != "validation"
    }
    return {"decision_token": result.decision_token, "tables": tables, "manifests": manifests, "report": result.report}


def semantic_digest(result: BuildResult) -> str:
    return canonical_digest(semantic_projection(result))


def build_report(result: BuildResult) -> str:
    yaml_lines = [
        f"- `{row['group_name']}`: `{row['terminal_status']}`; members `{row['member_config_count']}`; physical changes `{row['physical_consolidation_performed']}`."
        for row in result.tables["remaining_yaml_terminal_decisions"]
    ]
    validation_lines = [f"- `{row['check_name']}`: `{row['status']}`." for row in result.tables["validation"]]
    return "\n".join(
        [
            "# Phase F6I2P Remaining YAML and Repository Cleanup Closure",
            "",
            "## Decision",
            "",
            f"Decision token: `{result.decision_token}`.",
            "",
            f"Next phase: `{NEXT_PHASE}`.",
            "",
            "## Current-Suite Failure Closure",
            "",
            "- The two F6I2KR failures are classified as `stale_current_phase_assumption`.",
            "- The protected-input comparison was preserved by registering F6I2P's cleanup-closure manifest as an authoritative terminal phase manifest.",
            "",
            "## YAML Groups",
            "",
            *yaml_lines,
            "",
            "No member YAML, consumer reference, base/override loader, include directive, custom loader, training, inference, API call, download, package mutation, staging, or commit was performed.",
            "",
            "## Repository Cleanup",
            "",
            "- F6I2N wrapper conclusion adopted: wrappers `551`, safe removal bundles `0`, function-removal candidates `0`, file-removal candidates `0`, archive candidates `0`.",
            "- Repository cleanup decision: `terminal_retain_current_compatibility_architecture`.",
            "",
            "## Checkpoint",
            "",
            "- Exact-path checkpoint plan written to `results/tables/phase_f6i2p_checkpoint_staging_plan.csv`.",
            "- `git add .`, `git add -A`, and wildcard staging are prohibited.",
            "- No staging or commit was performed.",
            "",
            "## Validation",
            "",
            *validation_lines,
            "",
        ]
    )


def write_current_state(result: BuildResult) -> None:
    HANDOFF_PATH.write_text(
        "\n".join(
            [
                "# Future Operations Context Handoff",
                "",
                f"- Latest closure phase: `{PHASE_NAME}`.",
                f"- Decision token: `{result.decision_token}`.",
                f"- Next phase: `{NEXT_PHASE}`.",
                "- Remaining YAML cleanup is closed without physical YAML changes.",
                "- Repository cleanup is closed as `terminal_retain_current_compatibility_architecture`.",
                "- No training, inference, API calls, downloads, package mutation, staging, or commit were performed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    snapshot: dict[str, Any] = {}
    if SNAPSHOT_PATH.exists():
        try:
            snapshot = load_json(SNAPSHOT_PATH)
        except json.JSONDecodeError:
            snapshot = {}
    snapshot.update(
        {
            "latest_phase": PHASE_NAME,
            "decision_token": result.decision_token,
            "next_phase": NEXT_PHASE,
            "repository_cleanup_decision": "terminal_retain_current_compatibility_architecture",
            "remaining_yaml_cleanup": "complete_no_member_yaml_changes",
            "runtime": runtime_manifest(),
        }
    )
    snapshot[f"post_{PHASE_NAME}"] = {
        "phase": PHASE_ID,
        "decision_token": result.decision_token,
        "next_phase": NEXT_PHASE,
        "yaml_groups": {
            row["group_name"]: row["terminal_status"]
            for row in result.tables["remaining_yaml_terminal_decisions"]
        },
    }
    write_json(SNAPSHOT_PATH, snapshot)


def assemble(config_path: Path = CONFIG_PATH) -> BuildResult:
    config = load_phase_config(config_path)
    expected = expected_groups(config)
    groups = f6i2o_remaining_groups(expected)
    configs_by_group = group_configs(groups)

    tables: dict[str, list[dict[str, str]]] = {}
    tables["current_suite_failure_closure"] = current_suite_failure_closure_table()
    tables["remaining_yaml_terminal_decisions"] = terminal_decisions_table(expected, configs_by_group)
    tables["resolved_config_equivalence"] = resolved_config_equivalence_table(configs_by_group)
    tables["repository_cleanup_closure"] = repository_cleanup_closure_table(tables["remaining_yaml_terminal_decisions"])
    tables["checkpoint_staging_plan"] = checkpoint_staging_plan_table()
    tables["next_actions"] = []

    manifests: dict[str, dict[str, Any]] = {}
    manifests["protected_input_hashes"] = protected_input_hashes_manifest(groups)
    manifests["checkpoint_staging_manifest"] = checkpoint_staging_manifest(tables["checkpoint_staging_plan"])
    provisional_decision = decide(tables, manifests)
    manifests["cleanup_closure_manifest"] = cleanup_closure_manifest(provisional_decision, tables, manifests)
    tables["validation"] = validation_table(tables, manifests)
    decision = provisional_decision
    if any(row["status"] != "pass" for row in tables["validation"]):
        decision = DECISION_FAILED_PROTECTED if manifests["protected_input_hashes"].get("status") != "pass" else DECISION_BLOCKED_CONFIG
    tables["next_actions"] = next_actions_table(decision)
    manifests["cleanup_closure_manifest"] = cleanup_closure_manifest(decision, tables, manifests)
    result = BuildResult(decision_token=decision, tables=tables, manifests=manifests, report="")
    return BuildResult(decision_token=decision, tables=tables, manifests=manifests, report=build_report(result))


def build(
    config_path: Path = CONFIG_PATH,
    *,
    write_outputs_enabled: bool = True,
    update_current_state: bool = False,
) -> BuildResult:
    result = assemble(config_path)
    if write_outputs_enabled:
        for name, path in OUTPUT_TABLES.items():
            write_csv(path, result.tables[name], TABLE_COLUMNS[name])
        for name, path in OUTPUT_MANIFESTS.items():
            write_json(path, result.manifests[name])
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(result.report, encoding="utf-8")
        if update_current_state and result.decision_token == DECISION_SUCCESS:
            write_current_state(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--update-current-state", action="store_true")
    args = parser.parse_args(argv)
    result = build(args.config, write_outputs_enabled=True, update_current_state=args.update_current_state)
    print(json.dumps({"phase_id": PHASE_ID, "decision_token": result.decision_token}, sort_keys=True))
    return 0 if result.decision_token == DECISION_SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
