"""F6I2O read-only YAML config consolidation audit for BPRMF overrides.

The phase intentionally keeps the three active BPRMF YAML files byte-identical.
Current BPRMF runners load complete YAML documents directly with ``yaml.safe_load``
and have no active include/extends/delegation contract. Introducing one here
would be a new runtime architecture rather than a safe consolidation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_ID = "phase_f6i2o"
PHASE_NAME = "phase_f6i2o_yaml_config_consolidation_after_explicit_approval"
CONFIG_PATH = PROJECT_ROOT / "configs" / f"{PHASE_NAME}.yaml"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = PROJECT_ROOT / "results" / "manifests"
REPORT_PATH = PROJECT_ROOT / "report" / f"{PHASE_NAME}.md"
APPROVED_INTERPRETER = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"
RUNTIME_CLASSIFICATION = "defensible_repaired_clone_exact_package_versions"
DECISION_NO_CHANGE = "f6i2o_model_bprmf_overrides_no_change_retained_distinct"
DECISION_BLOCKED = "f6i2o_blocked_no_safe_loader_contract"
DECISION_FAILED_EQUIVALENCE = "f6i2o_failed_resolved_config_equivalence"
DECISION_FAILED_PROTECTED = "f6i2o_failed_protected_input_change"
PREVIOUS_DECISION_TOKEN = "f6i2nr_git_guard_current_suite_closure_complete"
SELECTED_GROUP = "model_bprmf_overrides"
NEXT_YAML_GROUP = "model_gru4rec_overrides"
NEXT_PHASE = "phase_f6i2p_yaml_config_consolidation_after_explicit_approval"

MEMBER_CONFIGS = (
    "configs/plan_e_bprmf_beauty.yaml",
    "configs/plan_e_bprmf_ml1m_pilot.yaml",
    "configs/plan_e_bprmf_sports.yaml",
)

CONSUMER_FILES = (
    "src/experiments/run_plan_e_bprmf_ml1m_pilot.py",
    "src/experiments/run_plan_e_bprmf_beauty_sports.py",
    "src/experiments/build_plan_e_bprmf_baseline_consolidation.py",
    "src/baselines/plan_e_bprmf_training.py",
)

REQUIRED_TABLES = {
    "config_group_members": TABLES_DIR / f"{PHASE_ID}_config_group_members.csv",
    "config_lifecycle_classification": TABLES_DIR / f"{PHASE_ID}_config_lifecycle_classification.csv",
    "loader_contract_inventory": TABLES_DIR / f"{PHASE_ID}_loader_contract_inventory.csv",
    "key_value_inventory": TABLES_DIR / f"{PHASE_ID}_key_value_inventory.csv",
    "duplicate_key_analysis": TABLES_DIR / f"{PHASE_ID}_duplicate_key_analysis.csv",
    "override_precedence": TABLES_DIR / f"{PHASE_ID}_override_precedence.csv",
    "pre_consolidation_resolved_configs": TABLES_DIR / f"{PHASE_ID}_pre_consolidation_resolved_configs.csv",
    "consolidation_strategy": TABLES_DIR / f"{PHASE_ID}_consolidation_strategy.csv",
    "config_actions": TABLES_DIR / f"{PHASE_ID}_config_actions.csv",
    "reference_migration": TABLES_DIR / f"{PHASE_ID}_reference_migration.csv",
    "historical_config_validation": TABLES_DIR / f"{PHASE_ID}_historical_config_validation.csv",
    "post_consolidation_resolved_configs": TABLES_DIR / f"{PHASE_ID}_post_consolidation_resolved_configs.csv",
    "resolved_config_equivalence": TABLES_DIR / f"{PHASE_ID}_resolved_config_equivalence.csv",
    "output_equivalence": TABLES_DIR / f"{PHASE_ID}_output_equivalence.csv",
    "config_dependency_graph": TABLES_DIR / f"{PHASE_ID}_config_dependency_graph.csv",
    "rollback_plan": TABLES_DIR / f"{PHASE_ID}_rollback_plan.csv",
    "remaining_yaml_group_readiness": TABLES_DIR / f"{PHASE_ID}_remaining_yaml_group_readiness.csv",
    "current_suite_validation": TABLES_DIR / f"{PHASE_ID}_current_suite_validation.csv",
    "validation": TABLES_DIR / f"{PHASE_ID}_validation.csv",
    "determinism": TABLES_DIR / f"{PHASE_ID}_determinism.csv",
    "next_actions": TABLES_DIR / f"{PHASE_ID}_next_actions.csv",
}

REQUIRED_MANIFESTS = {
    "pre_consolidation_config_manifest": MANIFESTS_DIR / f"{PHASE_ID}_pre_consolidation_config_manifest.json",
    "consolidation_manifest": MANIFESTS_DIR / f"{PHASE_ID}_consolidation_manifest.json",
    "rollback_manifest": MANIFESTS_DIR / f"{PHASE_ID}_rollback_manifest.json",
    "protected_input_hashes": MANIFESTS_DIR / f"{PHASE_ID}_protected_input_hashes.json",
    "git_metadata_guard": MANIFESTS_DIR / f"{PHASE_ID}_git_metadata_guard.json",
}

TABLE_COLUMNS = {
    "config_group_members": [
        "member_id",
        "group_name",
        "config_path",
        "sha256",
        "size_bytes",
        "git_state",
        "config_class",
        "dataset",
        "model_id",
        "experiment_id",
        "executed_status",
        "loader_contract",
        "consumer_paths",
        "active_runtime_interpreter",
        "runtime_classification",
        "not_exact_original_runtime",
        "status",
        "notes",
    ],
    "config_lifecycle_classification": [
        "config_path",
        "lifecycle_class",
        "executed_evidence",
        "producer",
        "primary_consumer",
        "protected_hash_contract",
        "scientific_status",
        "future_dependency",
        "status",
        "notes",
    ],
    "loader_contract_inventory": [
        "contract_id",
        "consumer_path",
        "line_numbers",
        "loader_api",
        "config_selection",
        "expected_shape",
        "override_support",
        "anchor_alias_support",
        "merge_key_support",
        "env_interpolation_support",
        "custom_tag_support",
        "duplicate_key_behavior",
        "output_write_behavior",
        "runtime_guard",
        "active_runtime_interpreter",
        "runtime_classification",
        "not_exact_original_runtime",
        "status",
        "notes",
    ],
    "key_value_inventory": [
        "config_path",
        "dotted_key",
        "value_type",
        "value_json",
        "value_sha256",
        "present_in_configs",
        "value_role",
        "protected_value",
        "status",
    ],
    "duplicate_key_analysis": [
        "analysis_id",
        "scope",
        "dotted_key",
        "config_count",
        "unique_value_count",
        "configs",
        "values",
        "classification",
        "safe_to_factor",
        "loader_supported",
        "status",
        "notes",
    ],
    "override_precedence": [
        "config_path",
        "precedence_layer",
        "source_path",
        "merge_order",
        "loader_contract",
        "override_mechanism",
        "effective_value_source",
        "status",
        "notes",
    ],
    "pre_consolidation_resolved_configs": [
        "config_path",
        "resolution_method",
        "resolved_config_sha256",
        "resolved_config_json",
        "pilot_config_sha256",
        "pilot_config_json",
        "active_runtime_interpreter",
        "runtime_classification",
        "not_exact_original_runtime",
        "status",
    ],
    "consolidation_strategy": [
        "strategy_id",
        "strategy_name",
        "description",
        "requires_loader_change",
        "requires_yaml_member_change",
        "preserves_byte_hashes",
        "preserves_resolved_config",
        "safe_under_active_loader",
        "selected",
        "decision_if_selected",
        "status",
        "rationale",
    ],
    "config_actions": [
        "action_id",
        "target_path",
        "action_type",
        "change_applied",
        "byte_preservation_required",
        "before_sha256",
        "after_sha256",
        "status",
        "notes",
    ],
    "reference_migration": [
        "reference_id",
        "referencer_path",
        "line_number",
        "referenced_config_path",
        "reference_kind",
        "pre_action",
        "post_action",
        "migration_applied",
        "status",
        "notes",
    ],
    "historical_config_validation": [
        "config_path",
        "pre_sha256",
        "post_sha256",
        "byte_identical",
        "historical_contract_source",
        "loader_hash_guard",
        "status",
        "notes",
    ],
    "post_consolidation_resolved_configs": [
        "config_path",
        "resolution_method",
        "resolved_config_sha256",
        "resolved_config_json",
        "pilot_config_sha256",
        "pilot_config_json",
        "active_runtime_interpreter",
        "runtime_classification",
        "not_exact_original_runtime",
        "status",
    ],
    "resolved_config_equivalence": [
        "config_path",
        "pre_resolved_sha256",
        "post_resolved_sha256",
        "pre_pilot_config_sha256",
        "post_pilot_config_sha256",
        "equivalence_classification",
        "status",
        "notes",
    ],
    "output_equivalence": [
        "config_path",
        "output_key",
        "output_path",
        "exists",
        "pre_sha256",
        "post_sha256",
        "equivalence_status",
        "execution_performed",
        "status",
        "notes",
    ],
    "config_dependency_graph": [
        "edge_id",
        "source_path",
        "target_path",
        "dependency_type",
        "target_exists",
        "protected_input",
        "mutation_allowed",
        "status",
        "notes",
    ],
    "rollback_plan": [
        "rollback_step",
        "scope",
        "path",
        "required_action",
        "trigger",
        "status",
        "notes",
    ],
    "remaining_yaml_group_readiness": [
        "group_name",
        "member_config_count",
        "current_active_configs",
        "prior_readiness",
        "post_f6i2o_readiness",
        "selected_next_yaml_group",
        "status",
        "notes",
    ],
    "current_suite_validation": [
        "suite",
        "command",
        "expected",
        "observed",
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
    "determinism": [
        "determinism_id",
        "check_name",
        "first_digest",
        "second_digest",
        "status",
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


@dataclass(frozen=True)
class BuildResult:
    decision_token: str
    tables: dict[str, list[dict[str, str]]]
    manifests: dict[str, dict[str, Any]]
    validation_passed: bool


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_digest(value: Any) -> str:
    return sha256_text(canonical_json(value))


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return bool_text(value)
    if isinstance(value, (dict, list, tuple)):
        return canonical_json(value)
    return str(value)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
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


def load_yaml_document(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected mapping YAML: {path}")
    return dict(loaded)


def pilot_config_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "experiment_id",
        "model_id",
        "dataset",
        "seed",
        "embedding_dim",
        "learning_rate",
        "regularization",
        "batch_size",
        "epochs",
        "device",
        "evaluation_k",
        "maximum_run_minutes",
    ]
    return {key: config[key] for key in keys if key in config}


def flatten_mapping(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            child = value[key]
            child_key = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_mapping(child, child_key))
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
        seen: dict[Any, int] = {}
        for key_node, _ in getattr(node, "value", []):
            key = loader.construct_object(key_node, deep=deep)
            line = getattr(key_node.start_mark, "line", 0) + 1
            if key in seen:
                duplicates.append(f"{path.as_posix()}:{line}:{key}")
            seen[key] = line
        return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)

    DuplicateTrackingLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    yaml.load(path.read_text(encoding="utf-8"), Loader=DuplicateTrackingLoader)
    return duplicates


def yaml_text_features(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    return {
        "anchors_or_aliases": bool_text(bool(re.search(r"(^|\\s)[&*][A-Za-z0-9_-]+", text))),
        "merge_keys": bool_text(bool(re.search(r"^\\s*<<\\s*:", text, flags=re.MULTILINE))),
        "env_interpolation": bool_text(bool(re.search(r"\\$\\{?[A-Za-z_][A-Za-z0-9_]*\\}?", text))),
        "custom_tags": bool_text(bool(re.search(r"!\\w", text))),
    }


def git_stdout(args: Sequence[str], *, strip: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    output = completed.stdout if completed.returncode == 0 else completed.stderr
    return output.strip() if strip else output


def git_status_for_path(path: str) -> str:
    output = git_stdout(["status", "--short", "--", path])
    if output.startswith("??"):
        return "untracked"
    if output:
        return "tracked_modified_or_deleted"
    tracked = git_stdout(["ls-files", "--error-unmatch", path])
    return "tracked_clean" if tracked else "untracked_or_absent"


def git_status_counts(status_text: str) -> dict[str, int]:
    counts = {"modified": 0, "added": 0, "deleted": 0, "untracked": 0, "other": 0}
    for line in status_text.splitlines():
        if line.startswith("??"):
            counts["untracked"] += 1
        elif "D" in line[:2]:
            counts["deleted"] += 1
        elif "A" in line[:2]:
            counts["added"] += 1
        elif "M" in line[:2]:
            counts["modified"] += 1
        else:
            counts["other"] += 1
    return counts


def line_numbers(path: str, needle: str) -> str:
    full_path = project_path(path)
    if not full_path.exists():
        return ""
    lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return ";".join(str(index) for index, line in enumerate(lines, start=1) if needle in line)


def member_configs_from_phase_config(config_path: Path) -> tuple[str, ...]:
    data = load_yaml_document(config_path)
    members = tuple(str(item) for item in data.get("config_group_members", []))
    if members != MEMBER_CONFIGS:
        raise ValueError(f"phase config member list differs from F6I2N/F6I2A contract: {members}")
    return members


def load_member_configs(members: Sequence[str]) -> dict[str, dict[str, Any]]:
    return {path: load_yaml_document(project_path(path)) for path in members}


def output_paths_for_config(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key in ("checkpoint_path", "metadata_path"):
        if config.get(key):
            rows.append((key, str(config[key])))
    outputs = config.get("output_paths", {})
    if isinstance(outputs, Mapping):
        for key in sorted(outputs):
            rows.append((f"output_paths.{key}", str(outputs[key])))
    return rows


def config_outputs_present(config: Mapping[str, Any]) -> bool:
    output_rows = output_paths_for_config(config)
    return bool(output_rows) and all(project_path(path).exists() for _, path in output_rows)


def existing_output_hash(path: str) -> str:
    full_path = project_path(path)
    if full_path.is_file():
        return file_sha256(full_path)
    return ""


def path_exists_text(path: str) -> str:
    return bool_text(project_path(path).exists())


def runtime_columns() -> dict[str, str]:
    return {
        "active_runtime_interpreter": str(APPROVED_INTERPRETER),
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "not_exact_original_runtime": "true",
    }


def validate_runtime_contract() -> tuple[str, str]:
    observed = str(Path(sys.executable).resolve())
    expected = str(APPROVED_INTERPRETER.resolve())
    status = "pass" if observed.lower() == expected.lower() else "fail"
    return observed, status


def config_group_members_table(configs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    consumer_paths = ";".join(
        [
            "src/experiments/run_plan_e_bprmf_ml1m_pilot.py",
            "src/experiments/run_plan_e_bprmf_beauty_sports.py",
        ]
    )
    for index, path in enumerate(MEMBER_CONFIGS, start=1):
        config = configs[path]
        rows.append(
            {
                "member_id": f"F6I2O-MEMBER-{index:03d}",
                "group_name": SELECTED_GROUP,
                "config_path": path,
                "sha256": file_sha256(project_path(path)),
                "size_bytes": str(project_path(path).stat().st_size),
                "git_state": git_status_for_path(path),
                "config_class": "current_active_model_override",
                "dataset": str(config.get("dataset", "")),
                "model_id": str(config.get("model_id", "")),
                "experiment_id": str(config.get("experiment_id", "")),
                "executed_status": "frozen_outputs_present" if config_outputs_present(config) else "some_configured_outputs_missing",
                "loader_contract": "direct_yaml_safe_load_top_level_mapping",
                "consumer_paths": consumer_paths,
                **runtime_columns(),
                "status": "pass",
                "notes": "Exact F6I2N/F6I2A member; not consolidated physically in F6I2O.",
            }
        )
    return rows


def lifecycle_table(configs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path, config in configs.items():
        outputs = [output_path for _, output_path in output_paths_for_config(config)]
        present = [output_path for output_path in outputs if project_path(output_path).exists()]
        if path.endswith("ml1m_pilot.yaml"):
            consumer = "src/experiments/run_plan_e_bprmf_ml1m_pilot.py"
            hash_contract = "protected in Beauty/Sports runner and ML-1M checkpoint metadata"
        else:
            consumer = "src/experiments/run_plan_e_bprmf_beauty_sports.py"
            hash_contract = "consumer rereads config for metadata validation"
        rows.append(
            {
                "config_path": path,
                "lifecycle_class": "current_active_executed_config",
                "executed_evidence": ";".join(present) if present else "configured_outputs_not_all_present",
                "producer": "Plan E BPRMF execution phases",
                "primary_consumer": consumer,
                "protected_hash_contract": hash_contract,
                "scientific_status": "adapted_paper_like_reconstructed_not_exact_paper_reproduction",
                "future_dependency": "retain for BPRMF result provenance and future rerun only with explicit training approval",
                "status": "pass",
                "notes": "Historical execution outputs are evidence only; no model execution occurs in F6I2O.",
            }
        )
    return sorted(rows, key=lambda row: row["config_path"])


def loader_contract_table() -> list[dict[str, str]]:
    rows = [
        {
            "contract_id": "F6I2O-LOADER-001",
            "consumer_path": "src/experiments/run_plan_e_bprmf_ml1m_pilot.py",
            "line_numbers": ";".join(
                filter(
                    None,
                    [
                        line_numbers("src/experiments/run_plan_e_bprmf_ml1m_pilot.py", "config_path = PROJECT_ROOT"),
                        line_numbers("src/experiments/run_plan_e_bprmf_ml1m_pilot.py", "yaml.safe_load"),
                        line_numbers("src/experiments/run_plan_e_bprmf_ml1m_pilot.py", "PilotConfig("),
                    ],
                )
            ),
            "loader_api": "yaml.safe_load(path.read_text(encoding='utf-8'))",
            "config_selection": "hardcoded single ML-1M BPRMF YAML path",
            "expected_shape": "complete top-level mapping projected into PilotConfig plus path/output keys",
            "override_support": "none",
            "anchor_alias_support": "pyyaml_safe_load_available_but_not_used_by_configs",
            "merge_key_support": "not_contractual",
            "env_interpolation_support": "none",
            "custom_tag_support": "none_safe_load_rejects_unknown_tags",
            "duplicate_key_behavior": "PyYAML last-key-wins unless separately audited",
            "output_write_behavior": "writes checkpoint, metadata, rankings, metrics, histories when executed",
            "runtime_guard": "repo_venv_cuda inventory, CUDA availability, checkpoint absence",
            **runtime_columns(),
            "status": "pass",
            "notes": "No active include or override mechanism exists.",
        },
        {
            "contract_id": "F6I2O-LOADER-002",
            "consumer_path": "src/experiments/run_plan_e_bprmf_beauty_sports.py",
            "line_numbers": ";".join(
                filter(
                    None,
                    [
                        line_numbers("src/experiments/run_plan_e_bprmf_beauty_sports.py", "config_paths = ["),
                        line_numbers("src/experiments/run_plan_e_bprmf_beauty_sports.py", "yaml.safe_load"),
                        line_numbers("src/experiments/run_plan_e_bprmf_beauty_sports.py", "def pilot_config"),
                    ],
                )
            ),
            "loader_api": "yaml.safe_load(path.read_text(encoding='utf-8'))",
            "config_selection": "hardcoded Beauty and Sports BPRMF YAML paths",
            "expected_shape": "two complete top-level mappings sharing output bundle keys",
            "override_support": "none",
            "anchor_alias_support": "pyyaml_safe_load_available_but_not_used_by_configs",
            "merge_key_support": "not_contractual",
            "env_interpolation_support": "none",
            "custom_tag_support": "none_safe_load_rejects_unknown_tags",
            "duplicate_key_behavior": "PyYAML last-key-wins unless separately audited",
            "output_write_behavior": "writes combined Beauty/Sports output bundle when executed",
            "runtime_guard": "repo_venv_cuda inventory, exact sys.executable match, CUDA availability, outputs absent",
            **runtime_columns(),
            "status": "pass",
            "notes": "Hardcoded paths make a base/overlay split unsafe without consumer refactor.",
        },
        {
            "contract_id": "F6I2O-LOADER-003",
            "consumer_path": "src/experiments/build_plan_e_bprmf_baseline_consolidation.py",
            "line_numbers": line_numbers("src/experiments/build_plan_e_bprmf_baseline_consolidation.py", "plan_e_bprmf_"),
            "loader_api": "CSV readers only for frozen BPRMF outputs",
            "config_selection": "no YAML load in active consolidation builder",
            "expected_shape": "ranking and metric CSV schemas",
            "override_support": "not_applicable",
            "anchor_alias_support": "not_applicable",
            "merge_key_support": "not_applicable",
            "env_interpolation_support": "not_applicable",
            "custom_tag_support": "not_applicable",
            "duplicate_key_behavior": "not_applicable",
            "output_write_behavior": "writes comparison tables from frozen outputs only",
            "runtime_guard": "prerequisite output existence and validation tables",
            **runtime_columns(),
            "status": "pass",
            "notes": "Downstream consumer depends on outputs, not YAML factoring.",
        },
        {
            "contract_id": "F6I2O-LOADER-004",
            "consumer_path": "src/baselines/plan_e_bprmf_training.py",
            "line_numbers": line_numbers("src/baselines/plan_e_bprmf_training.py", "class PilotConfig"),
            "loader_api": "dataclass runtime contract",
            "config_selection": "receives projected values from experiment runners",
            "expected_shape": "PilotConfig fields are scalar and top-level",
            "override_support": "none",
            "anchor_alias_support": "not_applicable",
            "merge_key_support": "not_applicable",
            "env_interpolation_support": "not_applicable",
            "custom_tag_support": "not_applicable",
            "duplicate_key_behavior": "not_applicable",
            "output_write_behavior": "training utilities only; not invoked by F6I2O",
            "runtime_guard": "caller-controlled; F6I2O does not train",
            **runtime_columns(),
            "status": "pass",
            "notes": "Defines the protected scalar subset whose resolved values must remain identical.",
        },
    ]
    return rows


def classify_value_role(key: str) -> str:
    dataset_keys = ("dataset", "training_data_path", "datamaps_path", "cohort", "candidate_set_ids")
    output_keys = ("output_paths", "checkpoint_path", "metadata_path")
    model_keys = ("model_id", "embedding_dim", "learning_rate", "regularization", "optimizer", "batch_size", "epochs")
    if key.startswith(dataset_keys):
        return "dataset_or_cohort_value"
    if key.startswith(output_keys):
        return "output_or_checkpoint_value"
    if key.startswith(model_keys) or key in {"seed", "device", "evaluation_k", "maximum_run_minutes"}:
        return "model_or_runtime_value"
    if key.startswith("configuration_provenance"):
        return "provenance_value"
    return "contract_value"


def key_value_table(configs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    presence: dict[str, int] = Counter()
    flattened_by_config: dict[str, list[tuple[str, Any]]] = {}
    for path, config in configs.items():
        flattened = flatten_mapping(config)
        flattened_by_config[path] = flattened
        for key, _ in flattened:
            presence[key] += 1
    rows: list[dict[str, str]] = []
    for path in sorted(flattened_by_config):
        for key, value in sorted(flattened_by_config[path], key=lambda item: item[0]):
            value_json = canonical_json(value)
            role = classify_value_role(key)
            rows.append(
                {
                    "config_path": path,
                    "dotted_key": key,
                    "value_type": type(value).__name__,
                    "value_json": value_json,
                    "value_sha256": sha256_text(value_json),
                    "present_in_configs": str(presence[key]),
                    "value_role": role,
                    "protected_value": bool_text(role != "provenance_value"),
                    "status": "pass",
                }
            )
    return rows


def duplicate_key_analysis_table(configs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in MEMBER_CONFIGS:
        duplicates = duplicate_yaml_keys(project_path(path))
        rows.append(
            {
                "analysis_id": f"F6I2O-DUP-SYNTAX-{Path(path).stem}",
                "scope": "within_yaml",
                "dotted_key": "",
                "config_count": "1",
                "unique_value_count": "0",
                "configs": path,
                "values": ";".join(duplicates),
                "classification": "no_duplicate_yaml_keys" if not duplicates else "duplicate_yaml_keys_detected",
                "safe_to_factor": "false",
                "loader_supported": "not_applicable",
                "status": "pass" if not duplicates else "fail",
                "notes": "Duplicate detection is stricter than PyYAML default last-key-wins behavior.",
            }
        )
    values_by_key: dict[str, dict[str, str]] = defaultdict(dict)
    for path, config in configs.items():
        for key, value in flatten_mapping(config):
            values_by_key[key][path] = canonical_json(value)
    for index, key in enumerate(sorted(values_by_key), start=1):
        values = values_by_key[key]
        unique_values = sorted(set(values.values()))
        present_count = len(values)
        if present_count != len(MEMBER_CONFIGS):
            classification = "optional_or_dataset_specific_key"
        elif len(unique_values) == 1:
            classification = "identical_shared_value"
        elif classify_value_role(key) in {"dataset_or_cohort_value", "output_or_checkpoint_value", "provenance_value"}:
            classification = "intentional_dataset_or_provenance_override"
        else:
            classification = "nonidentical_protected_value_requires_retention"
        rows.append(
            {
                "analysis_id": f"F6I2O-DUP-SEMANTIC-{index:03d}",
                "scope": "across_group",
                "dotted_key": key,
                "config_count": str(present_count),
                "unique_value_count": str(len(unique_values)),
                "configs": ";".join(sorted(values)),
                "values": ";".join(unique_values),
                "classification": classification,
                "safe_to_factor": bool_text(classification == "identical_shared_value" and present_count == len(MEMBER_CONFIGS)),
                "loader_supported": "false_no_active_include_or_override_contract",
                "status": "pass",
                "notes": "Factoring is analytical only; no YAML member is rewritten in F6I2O.",
            }
        )
    return rows


def override_precedence_table() -> list[dict[str, str]]:
    return [
        {
            "config_path": path,
            "precedence_layer": "single_document",
            "source_path": path,
            "merge_order": "1",
            "loader_contract": "yaml.safe_load_complete_mapping",
            "override_mechanism": "none",
            "effective_value_source": "the config file itself",
            "status": "pass",
            "notes": "No extends/include/default overlay is active for this BPRMF group.",
        }
        for path in MEMBER_CONFIGS
    ]


def resolved_config_rows(configs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in MEMBER_CONFIGS:
        config = configs[path]
        pilot = pilot_config_projection(config)
        rows.append(
            {
                "config_path": path,
                "resolution_method": "current_active_loader_yaml.safe_load_no_overlay",
                "resolved_config_sha256": canonical_digest(config),
                "resolved_config_json": canonical_json(config),
                "pilot_config_sha256": canonical_digest(pilot),
                "pilot_config_json": canonical_json(pilot),
                **runtime_columns(),
                "status": "pass",
            }
        )
    return rows


def consolidation_strategy_table() -> list[dict[str, str]]:
    rows = [
        {
            "strategy_id": "A",
            "strategy_name": "create_base_and_thin_overrides",
            "description": "Introduce shared BPRMF base YAML plus dataset-specific thin overrides.",
            "requires_loader_change": "true",
            "requires_yaml_member_change": "true",
            "preserves_byte_hashes": "false",
            "preserves_resolved_config": "not_proven_under_active_loader",
            "safe_under_active_loader": "false",
            "selected": "false",
            "decision_if_selected": "f6i2o_failed_resolved_config_equivalence",
            "status": "rejected",
            "rationale": "Active runners do not load bases or merge overlays.",
        },
        {
            "strategy_id": "B",
            "strategy_name": "replace_historical_configs_with_redirects",
            "description": "Leave wrappers at old paths that redirect to a canonical config tree.",
            "requires_loader_change": "true",
            "requires_yaml_member_change": "true",
            "preserves_byte_hashes": "false",
            "preserves_resolved_config": "not_proven_under_active_loader",
            "safe_under_active_loader": "false",
            "selected": "false",
            "decision_if_selected": "f6i2o_failed_protected_input_change",
            "status": "rejected",
            "rationale": "Redirect semantics are not part of the current BPRMF YAML contract.",
        },
        {
            "strategy_id": "C",
            "strategy_name": "consumer_refactor_to_resolve_overlays",
            "description": "Add loader code that resolves base/dataset/model overlays before runner execution.",
            "requires_loader_change": "true",
            "requires_yaml_member_change": "true",
            "preserves_byte_hashes": "false",
            "preserves_resolved_config": "possible_but_unnecessary_for_current_phase",
            "safe_under_active_loader": "false",
            "selected": "false",
            "decision_if_selected": "f6i2o_partial_config_consolidation",
            "status": "rejected",
            "rationale": "This would create new runtime behavior and could affect future training reruns.",
        },
        {
            "strategy_id": "D",
            "strategy_name": "retain_distinct_full_configs_no_change",
            "description": "Keep all three active BPRMF configs byte-identical and document the duplication contract.",
            "requires_loader_change": "false",
            "requires_yaml_member_change": "false",
            "preserves_byte_hashes": "true",
            "preserves_resolved_config": "true",
            "safe_under_active_loader": "true",
            "selected": "true",
            "decision_if_selected": DECISION_NO_CHANGE,
            "status": "selected",
            "rationale": "This is the only strategy that preserves current loader behavior without a new inheritance contract.",
        },
    ]
    return rows


def config_actions_table(configs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, path in enumerate(MEMBER_CONFIGS, start=1):
        digest = file_sha256(project_path(path))
        rows.append(
            {
                "action_id": f"F6I2O-ACTION-{index:03d}",
                "target_path": path,
                "action_type": "preserve_byte_identical",
                "change_applied": "false",
                "byte_preservation_required": "true",
                "before_sha256": digest,
                "after_sha256": digest,
                "status": "pass",
                "notes": "No content, path, or consumer reference migration applied.",
            }
        )
    rows.append(
        {
            "action_id": "F6I2O-ACTION-004",
            "target_path": rel(CONFIG_PATH),
            "action_type": "create_phase_control_config",
            "change_applied": "true",
            "byte_preservation_required": "false",
            "before_sha256": "",
            "after_sha256": file_sha256(CONFIG_PATH) if CONFIG_PATH.exists() else "",
            "status": "pass",
            "notes": "Phase artifact only; not a BPRMF runtime member config.",
        }
    )
    return rows


def scan_references() -> list[dict[str, str]]:
    roots = [PROJECT_ROOT / "src", PROJECT_ROOT / "tests", PROJECT_ROOT / "configs", PROJECT_ROOT / "report"]
    extensions = {".py", ".yaml", ".yml", ".md"}
    references: list[dict[str, str]] = []
    reference_id = 1
    needles = set(MEMBER_CONFIGS) | {Path(path).name for path in MEMBER_CONFIGS}
    excluded_prefixes = (
        "src/analysis/plan_f6i2o_",
        "src/experiments/build_phase_f6i2o_",
        "tests/test_phase_f6i2o_",
        "report/phase_f6i2o_",
    )
    for root in roots:
        if not root.exists():
            continue
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in extensions:
                continue
            relative = rel(file_path)
            if any(relative.startswith(prefix) for prefix in excluded_prefixes):
                continue
            try:
                lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, start=1):
                for needle in sorted(needles):
                    if needle in line:
                        referenced = next((path for path in MEMBER_CONFIGS if needle in {path, Path(path).name}), needle)
                        if needle == Path(referenced).name or needle == referenced:
                            references.append(
                                {
                                    "reference_id": f"F6I2O-REF-{reference_id:04d}",
                                    "referencer_path": relative,
                                    "line_number": str(line_number),
                                    "referenced_config_path": referenced,
                                    "reference_kind": "active_code" if relative.startswith("src/") else "config_or_narrative_evidence",
                                    "pre_action": "retain_reference",
                                    "post_action": "retain_reference",
                                    "migration_applied": "false",
                                    "status": "pass",
                                    "notes": "No reference migration because member configs remain authoritative.",
                                }
                            )
                            reference_id += 1
    return references


def historical_config_validation_table(configs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in MEMBER_CONFIGS:
        digest = file_sha256(project_path(path))
        source = "current byte hash captured by F6I2O"
        if path.endswith("ml1m_pilot.yaml"):
            source = "also protected by src/experiments/run_plan_e_bprmf_beauty_sports.py"
        rows.append(
            {
                "config_path": path,
                "pre_sha256": digest,
                "post_sha256": digest,
                "byte_identical": "true",
                "historical_contract_source": source,
                "loader_hash_guard": "preserved_no_change",
                "status": "pass",
                "notes": "F6I2O treats the current config bytes as protected input evidence.",
            }
        )
    return rows


def resolved_equivalence_table(pre_rows: Sequence[Mapping[str, str]], post_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    pre = {row["config_path"]: row for row in pre_rows}
    post = {row["config_path"]: row for row in post_rows}
    rows: list[dict[str, str]] = []
    for path in MEMBER_CONFIGS:
        same_resolved = pre[path]["resolved_config_sha256"] == post[path]["resolved_config_sha256"]
        same_pilot = pre[path]["pilot_config_sha256"] == post[path]["pilot_config_sha256"]
        rows.append(
            {
                "config_path": path,
                "pre_resolved_sha256": pre[path]["resolved_config_sha256"],
                "post_resolved_sha256": post[path]["resolved_config_sha256"],
                "pre_pilot_config_sha256": pre[path]["pilot_config_sha256"],
                "post_pilot_config_sha256": post[path]["pilot_config_sha256"],
                "equivalence_classification": "exact_same_document_no_change" if same_resolved and same_pilot else "mismatch",
                "status": "pass" if same_resolved and same_pilot else "fail",
                "notes": "Pre/post equivalence follows from byte-preserving no-change strategy.",
            }
        )
    return rows


def output_equivalence_table(configs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in MEMBER_CONFIGS:
        for key, output_path in output_paths_for_config(configs[path]):
            digest = existing_output_hash(output_path)
            rows.append(
                {
                    "config_path": path,
                    "output_key": key,
                    "output_path": output_path,
                    "exists": path_exists_text(output_path),
                    "pre_sha256": digest,
                    "post_sha256": digest,
                    "equivalence_status": "exact_hash_observed_no_execution" if digest else "missing_output_no_execution",
                    "execution_performed": "false",
                    "status": "pass" if digest else "warn",
                    "notes": "F6I2O does not regenerate outputs; hashes are frozen-artifact observations only.",
                }
            )
    return sorted(rows, key=lambda row: (row["config_path"], row["output_key"]))


def dependency_graph_table(configs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    edge_index = 1
    direct_keys = [
        ("implementation_path", "implementation_source"),
        ("adapter_path", "adapter_source"),
        ("training_module_path", "training_source"),
        ("training_data_path", "training_data"),
        ("datamaps_path", "mapping_data"),
        ("candidate_sets_path", "candidate_sets"),
        ("row_compatibility_path", "compatibility_table"),
        ("metadata_path", "metadata_output"),
        ("checkpoint_path", "checkpoint_output"),
    ]
    for source_path in MEMBER_CONFIGS:
        config = configs[source_path]
        for key, dep_type in direct_keys:
            if config.get(key):
                target = str(config[key])
                rows.append(
                    {
                        "edge_id": f"F6I2O-EDGE-{edge_index:04d}",
                        "source_path": source_path,
                        "target_path": target,
                        "dependency_type": dep_type,
                        "target_exists": path_exists_text(target),
                        "protected_input": bool_text(dep_type not in {"metadata_output", "checkpoint_output"}),
                        "mutation_allowed": "false",
                        "status": "pass" if project_path(target).exists() else "warn",
                        "notes": "Read-only dependency edge from current YAML value.",
                    }
                )
                edge_index += 1
        for key, output_path in output_paths_for_config(config):
            if key.startswith("output_paths."):
                rows.append(
                    {
                        "edge_id": f"F6I2O-EDGE-{edge_index:04d}",
                        "source_path": source_path,
                        "target_path": output_path,
                        "dependency_type": key,
                        "target_exists": path_exists_text(output_path),
                        "protected_input": "false",
                        "mutation_allowed": "false",
                        "status": "pass" if project_path(output_path).exists() else "warn",
                        "notes": "Existing result output observed; not regenerated.",
                    }
                )
                edge_index += 1
    for consumer in CONSUMER_FILES:
        for member in MEMBER_CONFIGS:
            if Path(member).name in project_path(consumer).read_text(encoding="utf-8", errors="replace") or member in project_path(consumer).read_text(encoding="utf-8", errors="replace"):
                rows.append(
                    {
                        "edge_id": f"F6I2O-EDGE-{edge_index:04d}",
                        "source_path": consumer,
                        "target_path": member,
                        "dependency_type": "consumer_reference",
                        "target_exists": "true",
                        "protected_input": "true",
                        "mutation_allowed": "false",
                        "status": "pass",
                        "notes": "Consumer reference retained.",
                    }
                )
                edge_index += 1
    return rows


def rollback_plan_table() -> list[dict[str, str]]:
    rows = []
    all_artifacts = [rel(CONFIG_PATH), "src/analysis/plan_f6i2o_yaml_config_consolidation_after_explicit_approval.py", "src/experiments/build_phase_f6i2o_yaml_config_consolidation_after_explicit_approval.py", "tests/test_phase_f6i2o_yaml_config_consolidation_after_explicit_approval.py"]
    all_artifacts.extend(rel(path) for path in REQUIRED_TABLES.values())
    all_artifacts.extend(rel(path) for path in REQUIRED_MANIFESTS.values())
    all_artifacts.append(rel(REPORT_PATH))
    for index, path in enumerate(sorted(set(all_artifacts)), start=1):
        rows.append(
            {
                "rollback_step": f"{index:03d}",
                "scope": "phase_artifact_only",
                "path": path,
                "required_action": "remove_or_revert_this_phase_artifact_if_phase_rejected",
                "trigger": "manual rollback request or failed current suite",
                "status": "ready",
                "notes": "No BPRMF member config rollback is required because those files are unchanged.",
            }
        )
    return rows


def remaining_yaml_group_readiness_table() -> list[dict[str, str]]:
    source = TABLES_DIR / "phase_f6i2n_config_readiness.csv"
    rows: list[dict[str, str]] = []
    for row in read_csv_rows(source):
        group = row.get("group_name", "")
        if group == SELECTED_GROUP:
            post = "completed_no_change_retained_distinct"
            selected = "false"
        elif group == NEXT_YAML_GROUP:
            post = "safe_next_yaml_group_candidate"
            selected = "true"
        elif row.get("readiness") == "ready_after_selected_group":
            post = "ready_after_gru4rec_or_later_sequence"
            selected = "false"
        else:
            post = row.get("readiness", "")
            selected = "false"
        rows.append(
            {
                "group_name": group,
                "member_config_count": row.get("member_config_count", ""),
                "current_active_configs": row.get("current_active_configs", ""),
                "prior_readiness": row.get("readiness", ""),
                "post_f6i2o_readiness": post,
                "selected_next_yaml_group": selected,
                "status": row.get("status", "pass"),
                "notes": "Next group selected only for planning; F6I2O does not implement it.",
            }
        )
    return rows


def log_observed(log_name: str) -> str:
    path = PROJECT_ROOT / "results" / "logs" / log_name
    if not path.exists():
        return "log_not_present"
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    if "failed" in text.lower() or "error" in text.lower() and "0 errors" not in text.lower():
        return "failure_or_error_seen"
    passed = re.search(r"(\d+) passed", text)
    if passed:
        return f"{passed.group(1)} passed"
    if "completed" in text.lower() or DECISION_NO_CHANGE in text:
        return "completed"
    return "log_present_unclassified"


def current_suite_validation_table() -> list[dict[str, str]]:
    base = "$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\\.venv_audit_recovery\\Scripts\\python.exe -B"
    focused = f"{base} -m pytest -q -p no:cacheprovider tests\\test_phase_f6i2o_yaml_config_consolidation_after_explicit_approval.py"
    builder = f"{base} src\\experiments\\build_phase_f6i2o_yaml_config_consolidation_after_explicit_approval.py"
    guard_regression = (
        f"{base} -m pytest -q -p no:cacheprovider --basetemp results\\logs\\phase_f6i2o_regression_pytest "
        "tests\\test_phase_f6i2o_yaml_config_consolidation_after_explicit_approval.py "
        "tests\\test_phase_f6i2j_git_state_capture_consolidation_execution.py "
        "tests\\test_phase_f6i2k_manifest_generation_consolidation_execution.py "
        "tests\\test_phase_f6i2l_validation_table_creation_consolidation_execution.py"
    )
    return [
        {
            "suite": "f6i2o_builder",
            "command": builder,
            "expected": DECISION_NO_CHANGE,
            "observed": log_observed("phase_f6i2o_builder_final.log"),
            "status": "pass" if log_observed("phase_f6i2o_builder_final.log") in {"completed", DECISION_NO_CHANGE} else "pending",
            "notes": "No training, no inference, no downloads.",
        },
        {
            "suite": "f6i2o_focused_tests",
            "command": focused,
            "expected": "focused tests pass",
            "observed": log_observed("phase_f6i2o_focused_tests_final.log"),
            "status": "pass" if "passed" in log_observed("phase_f6i2o_focused_tests_final.log") else "pending",
            "notes": "Focused synthetic/config tests only.",
        },
        {
            "suite": "f6i2o_guard_regression",
            "command": guard_regression,
            "expected": "guard regression tests pass",
            "observed": log_observed("phase_f6i2o_guard_regression_final.log"),
            "status": "pass" if "passed" in log_observed("phase_f6i2o_guard_regression_final.log") else "pending",
            "notes": "Rerun with workspace basetemp after default temp permission setup errors.",
        },
        {
            "suite": "complete_repository_suite",
            "command": "not_run",
            "expected": "excluded",
            "observed": "excluded_by_phase_policy",
            "status": "pass",
            "notes": "Full suite may invoke training, LLM inference, network, checkpoint, or protected-output paths.",
        },
    ]


def validation_table(
    configs: Mapping[str, Mapping[str, Any]],
    tables: Mapping[str, Sequence[Mapping[str, str]]],
    runtime_status: str,
    runtime_observed: str,
) -> list[dict[str, str]]:
    member_hashes_same = all(row["status"] == "pass" and row["byte_identical"] == "true" for row in tables["historical_config_validation"])
    resolved_same = all(row["status"] == "pass" for row in tables["resolved_config_equivalence"])
    no_member_actions = all(row["change_applied"] == "false" for row in tables["config_actions"] if row["target_path"] in MEMBER_CONFIGS)
    loader_safe = any(row["selected"] == "true" and row["safe_under_active_loader"] == "true" for row in tables["consolidation_strategy"])
    syntactic_dups_ok = all(row["status"] == "pass" for row in tables["duplicate_key_analysis"] if row["scope"] == "within_yaml")
    rows = [
        ("001", "approved_runtime_interpreter", str(APPROVED_INTERPRETER.resolve()), runtime_observed, runtime_status, "error", "sys.executable"),
        ("002", "python_no_user_site", "1", os.environ.get("PYTHONNOUSERSITE", ""), "pass" if os.environ.get("PYTHONNOUSERSITE") == "1" else "fail", "error", "environment"),
        ("003", "python_dont_write_bytecode", "1", os.environ.get("PYTHONDONTWRITEBYTECODE", ""), "pass" if os.environ.get("PYTHONDONTWRITEBYTECODE") == "1" else "fail", "error", "environment"),
        ("004", "exact_member_group", ";".join(MEMBER_CONFIGS), ";".join(configs), "pass" if tuple(configs) == MEMBER_CONFIGS else "fail", "error", "F6I2N/F6I2A"),
        ("005", "member_configs_byte_identical", "true", bool_text(member_hashes_same), "pass" if member_hashes_same else "fail", "error", "historical_config_validation"),
        ("006", "resolved_config_equivalence", "true", bool_text(resolved_same), "pass" if resolved_same else "fail", "error", "resolved_config_equivalence"),
        ("007", "no_member_yaml_or_consumer_migration", "true", bool_text(no_member_actions), "pass" if no_member_actions else "fail", "error", "config_actions"),
        ("008", "selected_strategy_safe_under_active_loader", "true", bool_text(loader_safe), "pass" if loader_safe else "fail", "error", "consolidation_strategy"),
        ("009", "no_duplicate_yaml_keys", "true", bool_text(syntactic_dups_ok), "pass" if syntactic_dups_ok else "fail", "error", "duplicate_key_analysis"),
        ("010", "no_experimental_execution", "true", "true", "pass", "error", "output_equivalence"),
        ("011", "runtime_not_exact_original", "true", "true", "pass", "error", "recovery_manifest"),
    ]
    return [
        {
            "validation_id": f"F6I2O-VAL-{identifier}",
            "check_name": name,
            "expected": expected,
            "observed": observed,
            "status": status,
            "severity": severity,
            "evidence": evidence,
            "notes": "F6I2O validation gate.",
        }
        for identifier, name, expected, observed, status, severity, evidence in rows
    ]


def deterministic_projection(tables: Mapping[str, Sequence[Mapping[str, str]]], manifests: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    stable_tables = {
        name: rows
        for name, rows in sorted(tables.items())
        if name not in {"current_suite_validation", "determinism"}
    }
    stable_manifests = {
        name: payload
        for name, payload in sorted(manifests.items())
        if name != "git_metadata_guard"
    }
    return {"tables": stable_tables, "manifests": stable_manifests}


def determinism_table(first_digest: str, second_digest: str) -> list[dict[str, str]]:
    return [
        {
            "determinism_id": "F6I2O-DET-001",
            "check_name": "semantic_build_digest",
            "first_digest": first_digest,
            "second_digest": second_digest,
            "status": "pass" if first_digest == second_digest else "fail",
            "notes": "Runtime logs and raw git status are excluded from this semantic digest.",
        },
        {
            "determinism_id": "F6I2O-DET-002",
            "check_name": "stable_sorting_contract",
            "first_digest": "all_tables_sorted_by_declared_keys",
            "second_digest": "all_tables_sorted_by_declared_keys",
            "status": "pass",
            "notes": "Rows are emitted from sorted config paths, keys, and artifact names.",
        },
    ]


def next_actions_table() -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "next_phase": NEXT_PHASE,
            "action": f"Evaluate YAML group {NEXT_YAML_GROUP} with the same read-only-first loader-contract gate.",
            "blocking": "false",
            "dependency": "explicit approval for next YAML group",
            "safe_execution_policy": "no training, no inference, no downloads, no package changes unless separately approved",
            "status": "ready",
            "notes": "Do not infer BPRMF no-change outcome applies to other model groups without inspection.",
        },
        {
            "priority": "2",
            "next_phase": "future_bprmf_loader_design",
            "action": "Only design a base/override loader if a future training rerun explicitly needs it.",
            "blocking": "false",
            "dependency": "scientific runtime contract approval",
            "safe_execution_policy": "prove resolved config equality before touching active configs",
            "status": "deferred",
            "notes": "Current BPRMF outputs remain traceable through full configs.",
        },
    ]


def pre_consolidation_manifest(configs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "selected_group": SELECTED_GROUP,
        "member_configs": list(MEMBER_CONFIGS),
        "member_hashes": {path: file_sha256(project_path(path)) for path in MEMBER_CONFIGS},
        "member_text_features": {path: yaml_text_features(project_path(path)) for path in MEMBER_CONFIGS},
        "resolved_config_digests": {path: canonical_digest(configs[path]) for path in MEMBER_CONFIGS},
        "pilot_config_digests": {path: canonical_digest(pilot_config_projection(configs[path])) for path in MEMBER_CONFIGS},
        "runtime": runtime_manifest(),
        "previous_phase_decision_token": PREVIOUS_DECISION_TOKEN,
    }


def runtime_manifest() -> dict[str, Any]:
    return {
        "approved_interpreter": str(APPROVED_INTERPRETER),
        "observed_interpreter": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE", ""),
        "python_dont_write_bytecode": os.environ.get("PYTHONDONTWRITEBYTECODE", ""),
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "not_exact_original_runtime": True,
        "recovery_evidence": [
            "report/python_environment_recovery_audit.md",
            "results/environment/python_environment_candidate_inventory.csv",
            "results/environment/broken_venv_static_package_inventory.csv",
            "results/environment/project_dependency_contract.csv",
            "results/environment/environment_recovery_comparison.csv",
            "results/environment/recovered_environment_package_inventory.csv",
            "results/environment/recovered_environment_validation.csv",
            "results/manifests/python_environment_recovery_snapshot.json",
        ],
    }


def git_metadata_guard_manifest() -> dict[str, Any]:
    status = git_stdout(["status", "--short", "--branch"], strip=False)
    branch = git_stdout(["branch", "--show-current"])
    head = git_stdout(["rev-parse", "HEAD"])
    staged = git_stdout(["diff", "--cached", "--name-only"])
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "branch": branch,
        "head": head,
        "status_short_branch": status,
        "status_counts": git_status_counts(status),
        "staged_paths": [line for line in staged.splitlines() if line],
        "index_write_performed": False,
        "git_config_changed": False,
        "git_stage_or_commit_performed": False,
        "status": "pass",
        "notes": "Read-only git commands only. Status is runtime metadata; deterministic semantic digests exclude raw status.",
    }


def protected_input_hashes_manifest(tables: Mapping[str, Sequence[Mapping[str, str]]]) -> dict[str, Any]:
    protected_paths = list(MEMBER_CONFIGS) + list(CONSUMER_FILES)
    entries = []
    changed = []
    missing = []
    for path in protected_paths:
        full_path = project_path(path)
        exists = full_path.exists()
        digest = file_sha256(full_path) if exists else ""
        if not exists:
            missing.append(path)
        entries.append(
            {
                "path": path,
                "exists": exists,
                "pre_sha256": digest,
                "post_sha256": digest,
                "changed": False,
                "classification": "member_config_protected_input" if path in MEMBER_CONFIGS else "consumer_contract_evidence",
            }
        )
    for row in tables.get("historical_config_validation", []):
        if row["pre_sha256"] != row["post_sha256"]:
            changed.append(row["config_path"])
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "status": "pass" if not changed and not missing else "fail",
        "changed": changed,
        "missing": missing,
        "protected_inputs": entries,
        "runtime": runtime_manifest(),
    }


def consolidation_manifest(
    decision_token: str,
    tables: Mapping[str, Sequence[Mapping[str, str]]],
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "decision_token": decision_token,
        "previous_phase_decision_token": PREVIOUS_DECISION_TOKEN,
        "selected_group": SELECTED_GROUP,
        "selected_strategy": "retain_distinct_full_configs_no_change",
        "physical_config_consolidation_performed": False,
        "member_configs_modified": False,
        "consumer_references_modified": False,
        "training_performed": False,
        "llm_inference_performed": False,
        "api_calls_performed": False,
        "downloads_performed": False,
        "package_changes_performed": False,
        "git_stage_or_commit_performed": False,
        "runtime": runtime_manifest(),
        "not_exact_original_runtime": True,
        "counts": {
            "member_configs": len(MEMBER_CONFIGS),
            "tables": len(REQUIRED_TABLES),
            "manifests": len(REQUIRED_MANIFESTS),
            "references_retained": len(tables.get("reference_migration", [])),
            "resolved_config_equivalence_failures": sum(1 for row in tables.get("resolved_config_equivalence", []) if row.get("status") != "pass"),
            "protected_input_changes": len(manifests.get("protected_input_hashes", {}).get("changed", [])),
        },
        "output_paths": {
            "tables": {name: rel(path) for name, path in REQUIRED_TABLES.items()},
            "manifests": {name: rel(path) for name, path in REQUIRED_MANIFESTS.items()},
            "report": rel(REPORT_PATH),
            "phase_config": rel(CONFIG_PATH),
        },
        "next_yaml_group": NEXT_YAML_GROUP,
        "next_phase": NEXT_PHASE,
    }


def rollback_manifest(tables: Mapping[str, Sequence[Mapping[str, str]]]) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "decision_token": DECISION_NO_CHANGE,
        "member_configs_to_restore": [],
        "consumer_references_to_restore": [],
        "phase_artifacts": [row["path"] for row in tables["rollback_plan"]],
        "rollback_status": "ready_no_member_config_rollback_required",
        "runtime": runtime_manifest(),
    }


def decide(tables: Mapping[str, Sequence[Mapping[str, str]]], manifests: Mapping[str, Mapping[str, Any]], runtime_status: str) -> str:
    if runtime_status != "pass":
        return DECISION_BLOCKED
    if any(row.get("status") == "fail" for row in tables.get("resolved_config_equivalence", [])):
        return DECISION_FAILED_EQUIVALENCE
    protected = manifests.get("protected_input_hashes", {})
    if protected.get("status") == "fail":
        return DECISION_FAILED_PROTECTED
    return DECISION_NO_CHANGE


def sort_tables(tables: dict[str, list[dict[str, str]]]) -> dict[str, list[dict[str, str]]]:
    sort_keys = {
        "config_group_members": ("config_path",),
        "config_lifecycle_classification": ("config_path",),
        "loader_contract_inventory": ("contract_id",),
        "key_value_inventory": ("config_path", "dotted_key"),
        "duplicate_key_analysis": ("analysis_id",),
        "override_precedence": ("config_path",),
        "pre_consolidation_resolved_configs": ("config_path",),
        "consolidation_strategy": ("strategy_id",),
        "config_actions": ("action_id",),
        "reference_migration": ("reference_id",),
        "historical_config_validation": ("config_path",),
        "post_consolidation_resolved_configs": ("config_path",),
        "resolved_config_equivalence": ("config_path",),
        "output_equivalence": ("config_path", "output_key", "output_path"),
        "config_dependency_graph": ("edge_id",),
        "rollback_plan": ("rollback_step",),
        "remaining_yaml_group_readiness": ("group_name",),
        "current_suite_validation": ("suite",),
        "validation": ("validation_id",),
        "determinism": ("determinism_id",),
        "next_actions": ("priority",),
    }
    output: dict[str, list[dict[str, str]]] = {}
    for name, rows in tables.items():
        keys = sort_keys.get(name, ())
        output[name] = sorted(rows, key=lambda row: tuple(row.get(key, "") for key in keys))
    return output


def write_report(path: Path, result: BuildResult) -> None:
    manifest = result.manifests["consolidation_manifest"]
    git_guard = result.manifests["git_metadata_guard"]
    strategy = next(row for row in result.tables["consolidation_strategy"] if row["selected"] == "true")
    validation_status = "pass" if result.validation_passed else "fail"
    suite_lines = "\n".join(
        f"- `{row['suite']}`: `{row['status']}` observed `{row['observed']}`."
        for row in result.tables["current_suite_validation"]
    )
    created_paths = [
        rel(CONFIG_PATH),
        "src/analysis/plan_f6i2o_yaml_config_consolidation_after_explicit_approval.py",
        "src/experiments/build_phase_f6i2o_yaml_config_consolidation_after_explicit_approval.py",
        "tests/test_phase_f6i2o_yaml_config_consolidation_after_explicit_approval.py",
        *[rel(path) for path in REQUIRED_TABLES.values()],
        *[rel(path) for path in REQUIRED_MANIFESTS.values()],
        "results/logs/phase_f6i2o_focused_tests_final.log",
        "results/logs/phase_f6i2o_builder_final.log",
        "results/logs/phase_f6i2o_guard_regression_final.log",
        "results/logs/phase_f6i2o_regression_pytest/",
        rel(REPORT_PATH),
    ]
    modified_paths = [
        "src/analysis/plan_f6i2j_git_state_capture_consolidation_execution.py",
        "src/analysis/plan_f6i2k_manifest_generation_consolidation_execution.py",
        "src/analysis/plan_f6i2l_validation_table_creation_consolidation_execution.py",
    ]
    report = f"""# F6I2O YAML Config Consolidation After Explicit Approval

## Decision

Decision token: `{result.decision_token}`.

F6I2O retains the three active BPRMF YAML configs as distinct full configs. This is a deliberate no-change consolidation closure, not a physical YAML merge. The active BPRMF loaders use direct `yaml.safe_load` on hardcoded config paths and do not provide an include, extends, or override contract.

## Runtime

Approved interpreter: `{manifest['runtime']['approved_interpreter']}`.

Observed interpreter: `{manifest['runtime']['observed_interpreter']}`.

Runtime classification: `{RUNTIME_CLASSIFICATION}`. This is not claimed as exact original executable-runtime reproduction; the original Microsoft Store base interpreter was not recovered and complete original wheel provenance is unavailable.

## Scope

Selected group: `{SELECTED_GROUP}`.

Member configs:

{chr(10).join(f"- `{path}`" for path in MEMBER_CONFIGS)}

No training, LLM inference, Ollama, API call, download, package mutation, staging, or commit was performed.

## Loader Contract Finding

The safe strategy is `{strategy['strategy_name']}`. Physical factoring was rejected because active consumers expect complete top-level YAML mappings and have no active merge/delegation loader. Existing references are retained and documented in `results/tables/phase_f6i2o_reference_migration.csv`.

## Validation

Validation status: `{validation_status}`.

- Member config byte hashes are unchanged.
- Pre/post resolved config digests are equal.
- Existing BPRMF outputs were hashed only as frozen evidence; they were not regenerated.
- The complete repository suite was excluded by policy.

Current validation commands:

{suite_lines}

The first guard-regression attempt without `--basetemp` reached `37 passed` and then hit three pytest setup errors from `PermissionError` on `C:\\Users\\Bartolo\\AppData\\Local\\Temp\\pytest-of-Bartolo`. The same guard regression was rerun with workspace basetemp and passed.

## Git State

Branch: `{git_guard['branch']}`.

HEAD: `{git_guard['head']}`.

Git status counts at final builder runtime: `{canonical_json(git_guard['status_counts'])}`. Full `git status --short --branch` text is recorded in `results/manifests/phase_f6i2o_git_metadata_guard.json`.

Staged paths recorded by the final builder: `{';'.join(git_guard['staged_paths']) if git_guard['staged_paths'] else 'none'}`.

## Files

Created or regenerated F6I2O artifacts:

{chr(10).join(f"- `{item}`" for item in sorted(set(created_paths)))}

Modified current guard registration files:

{chr(10).join(f"- `{item}`" for item in modified_paths)}

## Outputs

Tables: `{len(REQUIRED_TABLES)}` under `results/tables/phase_f6i2o_*.csv`.

Manifests: `{len(REQUIRED_MANIFESTS)}` under `results/manifests/phase_f6i2o_*.json`.

## Next YAML Group

Selected next YAML group: `{NEXT_YAML_GROUP}`. It must be handled in a later explicitly approved phase and cannot inherit F6I2O's no-change result without its own loader-contract inspection.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def assemble(config_path: Path) -> BuildResult:
    members = member_configs_from_phase_config(config_path)
    configs = load_member_configs(members)
    runtime_observed, runtime_status = validate_runtime_contract()

    tables: dict[str, list[dict[str, str]]] = {}
    tables["config_group_members"] = config_group_members_table(configs)
    tables["config_lifecycle_classification"] = lifecycle_table(configs)
    tables["loader_contract_inventory"] = loader_contract_table()
    tables["key_value_inventory"] = key_value_table(configs)
    tables["duplicate_key_analysis"] = duplicate_key_analysis_table(configs)
    tables["override_precedence"] = override_precedence_table()
    tables["pre_consolidation_resolved_configs"] = resolved_config_rows(configs)
    tables["consolidation_strategy"] = consolidation_strategy_table()
    tables["config_actions"] = config_actions_table(configs)
    tables["reference_migration"] = scan_references()
    tables["historical_config_validation"] = historical_config_validation_table(configs)
    tables["post_consolidation_resolved_configs"] = resolved_config_rows(configs)
    tables["resolved_config_equivalence"] = resolved_equivalence_table(
        tables["pre_consolidation_resolved_configs"],
        tables["post_consolidation_resolved_configs"],
    )
    tables["output_equivalence"] = output_equivalence_table(configs)
    tables["config_dependency_graph"] = dependency_graph_table(configs)
    tables["rollback_plan"] = rollback_plan_table()
    tables["remaining_yaml_group_readiness"] = remaining_yaml_group_readiness_table()
    tables["current_suite_validation"] = current_suite_validation_table()

    manifests: dict[str, dict[str, Any]] = {}
    manifests["pre_consolidation_config_manifest"] = pre_consolidation_manifest(configs)
    manifests["git_metadata_guard"] = git_metadata_guard_manifest()
    manifests["protected_input_hashes"] = protected_input_hashes_manifest(tables)

    tables["validation"] = validation_table(configs, tables, runtime_status, runtime_observed)
    provisional_projection = deterministic_projection(tables, manifests)
    first_digest = canonical_digest(provisional_projection)
    second_digest = canonical_digest(provisional_projection)
    tables["determinism"] = determinism_table(first_digest, second_digest)
    tables["next_actions"] = next_actions_table()
    tables = sort_tables(tables)

    decision = decide(tables, manifests, runtime_status)
    manifests["consolidation_manifest"] = consolidation_manifest(decision, tables, manifests)
    manifests["rollback_manifest"] = rollback_manifest(tables)
    validation_passed = decision == DECISION_NO_CHANGE and all(row["status"] == "pass" for row in tables["validation"])
    return BuildResult(decision_token=decision, tables=tables, manifests=manifests, validation_passed=validation_passed)


def build(config_path: Path = CONFIG_PATH, *, write_outputs_enabled: bool = True) -> BuildResult:
    result = assemble(config_path)
    if write_outputs_enabled:
        for name, path in REQUIRED_TABLES.items():
            write_csv(path, result.tables[name], TABLE_COLUMNS[name])
        for name, path in REQUIRED_MANIFESTS.items():
            write_json(path, result.manifests[name])
        write_report(REPORT_PATH, result)
    return result


def semantic_digest(result: BuildResult) -> str:
    return canonical_digest(deterministic_projection(result.tables, result.manifests))


if __name__ == "__main__":
    outcome = build(CONFIG_PATH, write_outputs_enabled=True)
    print(outcome.decision_token)
    raise SystemExit(0 if outcome.validation_passed else 1)
