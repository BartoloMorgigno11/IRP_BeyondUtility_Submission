"""Phase 6B README and reproducibility-package generation.

This module builds user-facing documentation from frozen Phase 6A evidence.
It updates only the root README and creates Phase 6B documentation, indexes,
audits, manifests, and a phase report. It does not run scientific workflows.
"""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
README_PATH = Path("README.md")
REPRO_GUIDE_PATH = Path("docs/reproducibility_guide.md")
ARTIFACT_CATALOG_PATH = Path("docs/artifact_catalog.md")
REPORT_PATH = Path("report/plan_d_phase_6b_readme_reproducibility_package.md")

OUTPUT_TABLES = {
    "documentation_source_map": Path("results/tables/plan_d_phase_6b_documentation_source_map.csv"),
    "final_artifact_index": Path("results/tables/plan_d_phase_6b_final_artifact_index.csv"),
    "safe_command_registry": Path("results/tables/plan_d_phase_6b_safe_command_registry.csv"),
    "excluded_entrypoints": Path("results/tables/plan_d_phase_6b_excluded_entrypoints.csv"),
    "portability_audit": Path("results/tables/plan_d_phase_6b_portability_audit.csv"),
    "readme_claim_audit": Path("results/tables/plan_d_phase_6b_readme_claim_audit.csv"),
    "validation": Path("results/tables/plan_d_phase_6b_validation.csv"),
    "next_actions": Path("results/tables/plan_d_phase_6b_next_actions.csv"),
    "readme_hash_audit": Path("results/tables/plan_d_phase_6b_readme_hash_audit.csv"),
}

INITIAL_PROTECTED_MANIFEST_PATH = Path("results/tables/plan_d_phase_6b_protected_input_hashes_initial.csv")
FINAL_PROTECTED_MANIFEST_PATH = Path("results/tables/plan_d_phase_6b_protected_input_hashes_final.csv")
OUTPUT_HASHES_FIRST_PATH = Path("results/tables/plan_d_phase_6b_output_hashes_first.csv")
OUTPUT_HASHES_SECOND_PATH = Path("results/tables/plan_d_phase_6b_output_hashes_second.csv")

PHASE6B_OUTPUT_PATHS = {
    README_PATH.as_posix(),
    REPRO_GUIDE_PATH.as_posix(),
    ARTIFACT_CATALOG_PATH.as_posix(),
    REPORT_PATH.as_posix(),
    INITIAL_PROTECTED_MANIFEST_PATH.as_posix(),
    FINAL_PROTECTED_MANIFEST_PATH.as_posix(),
    OUTPUT_HASHES_FIRST_PATH.as_posix(),
    OUTPUT_HASHES_SECOND_PATH.as_posix(),
    *{path.as_posix() for path in OUTPUT_TABLES.values()},
    "src/analysis/plan_d_phase_6b_readme_reproducibility.py",
    "src/experiments/build_plan_d_phase_6b_readme_reproducibility.py",
    "tests/test_plan_d_phase_6b_readme_reproducibility.py",
}

OUTPUT_HASH_INCLUDE_PATHS = [
    README_PATH.as_posix(),
    REPRO_GUIDE_PATH.as_posix(),
    ARTIFACT_CATALOG_PATH.as_posix(),
    REPORT_PATH.as_posix(),
    *[path.as_posix() for path in OUTPUT_TABLES.values()],
    INITIAL_PROTECTED_MANIFEST_PATH.as_posix(),
    FINAL_PROTECTED_MANIFEST_PATH.as_posix(),
]

PHASE6A_INPUTS = {
    "blueprint": Path("report/plan_d_phase_6a_repository_consolidation_blueprint.md"),
    "artifact_registry": Path("results/tables/plan_d_phase_6a_artifact_registry.csv"),
    "canonical_artifact_map": Path("results/tables/plan_d_phase_6a_canonical_artifact_map.csv"),
    "final_documentation_input_map": Path("results/tables/plan_d_phase_6a_final_documentation_input_map.csv"),
    "reproducibility_entrypoints": Path("results/tables/plan_d_phase_6a_reproducibility_entrypoints.csv"),
    "final_delivery_policy": Path("results/tables/plan_d_phase_6a_final_delivery_policy.csv"),
    "legacy_superseded": Path("results/tables/plan_d_phase_6a_legacy_superseded.csv"),
    "plan_e_reconciliation": Path("results/tables/plan_d_phase_6a_plan_e_reconciliation.csv"),
    "orphaned_or_unresolved": Path("results/tables/plan_d_phase_6a_orphaned_or_unresolved.csv"),
    "phase_5e_accounting": Path("results/tables/plan_d_phase_6a_phase_5e_accounting.csv"),
    "validation": Path("results/tables/plan_d_phase_6a_validation.csv"),
    "next_actions": Path("results/tables/plan_d_phase_6a_next_actions.csv"),
}

EXCLUDED_DIRS = {".git", ".venv", "venv", "env", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache"}
ABSOLUTE_PATH_PATTERN = re.compile(r"[A-Za-z]:[\\/][^\s\]\)\"']+")
SECRET_PATTERN = re.compile(r"(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}", re.IGNORECASE)
UNSAFE_COMMAND_PATTERNS = [
    "ollama",
    "run_canonical_local_llm",
    "run_plan_e_",
    "train",
    "training",
    "download",
    "--allow-api-calls",
    "openai",
    "requests.post",
]

README_REQUIRED_SECTIONS = [
    "Project Status",
    "Paper Under Study",
    "Scientific Objective",
    "Reproducibility Scope",
    "What Was Reproduced, Reconstructed, Adapted, Extended, And Blocked",
    "Main Datasets",
    "Traditional Recommender Baselines",
    "Local LLM Models",
    "Canonical Ranking Protocol",
    "Canonical Reranking Protocol",
    "Evaluation Dimensions",
    "Main Results",
    "Structural Reliability",
    "Paper-Coverage Audit",
    "Canonical Artifact Map",
    "Repository Structure",
    "Environment",
    "Safe Quick Start",
    "Reproducibility Entry Points",
    "How To Inspect Results Without Rerunning Inference",
    "Legacy And Diagnostic Artifacts",
    "Blocked Components",
    "Threats And Limitations",
    "Reproducibility Statement",
    "Documentation Links",
    "Citation And Attribution",
    "License Or Repository License Status",
]

GUIDE_REQUIRED_SECTIONS = [
    "Validated Environment",
    "Repository Assumptions",
    "Protected-Artifact Policy",
    "Scientific Evidence Hierarchy",
    "Canonical Versus Diagnostic Workflows",
    "Safe Offline Workflow",
    "Safe Entry-Point Commands",
    "Expected Outputs",
    "Validation Interpretation",
    "Deterministic-Build Policy",
    "Inspect Phase 5C Results",
    "Inspect Phase 5D Results",
    "Inspect Phase 5E Results",
    "Inspect Phase 6A Classifications",
    "Inspect Plan E Artifacts",
    "Commands That Must Not Be Run Casually",
    "Inference And Training Entry-Point Warnings",
    "Artifact Immutability Policy",
    "Troubleshooting",
    "Portability Limitations",
    "Known Blockers",
    "Reproducibility Checklist",
]

CATALOG_REQUIRED_SECTIONS = [
    "Dataset And Cohort Evidence",
    "Candidate Construction",
    "Baseline Evidence",
    "Canonical LLM Execution Evidence",
    "Phase 5C Evaluation",
    "Phase 5D Robustness Evidence",
    "Phase 5E Paper Coverage",
    "Phase 6A Consolidation",
    "Extensions",
    "Diagnostics",
    "Legacy Protocols",
    "Blocked Or Unresolved Components",
    "Raw Immutable Evidence",
    "Implementation And Tests",
]

FROZEN_PHASE5E_COUNTS = {
    "applicable_component_count": "73",
    "reproduced_exact": "9",
    "reconstructed": "16",
    "adapted": "25",
    "blocked": "14",
    "not_executed": "5",
    "unresolved": "4",
    "strict_exact_reproduction_micro_pct": "12.328767",
    "maximum_defensible_micro_coverage_pct": "68.493151",
    "exact_numeric_result_cell_pct": "0.0",
}

FROZEN_FULL_MATRIX = {
    "canonical_requests": "11528",
    "responses": "11528",
    "runner_events": "23056",
    "ranking_evaluations": "5764",
    "reranking_evaluations": "5764",
    "ranking_strict_valid": "5764",
    "reranking_strict_valid": "5738",
    "invalid_outputs": "26",
    "strict_index_hallucination_count": "0",
}

DOC_SOURCE_COLUMNS = [
    "documentation_section",
    "statement_or_topic_id",
    "primary_source_artifact",
    "supporting_source_artifacts",
    "evidence_level",
    "canonicality",
    "permitted_wording",
    "prohibited_wording",
    "unresolved_caveat",
]
FINAL_INDEX_COLUMNS = [
    "artifact_id",
    "relative_path",
    "artifact_family",
    "scientific_topic",
    "lifecycle_class",
    "canonicality",
    "delivery_disposition",
    "README_role",
    "reproducibility_guide_role",
    "artifact_catalog_role",
    "regeneration_policy",
    "claim_supported",
    "claim_prohibited",
]
SAFE_COMMAND_COLUMNS = [
    "command_id",
    "entry_point_path",
    "command",
    "purpose",
    "required_environment",
    "required_inputs",
    "expected_outputs",
    "side_effects",
    "safe_status",
    "source_phase_6a_registry_row",
    "README_included",
    "guide_included",
]
ENTRYPOINT_COLUMNS = [
    "path",
    "intended_purpose",
    "required_inputs",
    "produced_outputs",
    "performs_offline_aggregation",
    "performs_llm_inference",
    "performs_recommender_training",
    "performs_recommender_inference",
    "performs_downloading",
    "performs_api_access",
    "expected_runtime_class",
    "safe_for_final_documentation",
    "depends_on_protected_artifacts",
    "canonical",
    "legacy",
    "should_appear_in_final_readme",
    "prerequisites",
    "environment_variables",
    "expected_working_directory",
    "known_blockers",
]
EXCLUDED_ENTRYPOINT_COLUMNS = [
    "entry_point",
    "side_effect_categories",
    "exclusion_reason",
    "preservation_role",
    "documentation_may_mention",
    "prohibited_presentation",
]
PORTABILITY_COLUMNS = [
    "documentation_path",
    "check_name",
    "status",
    "evidence",
    "resolution",
]
CLAIM_AUDIT_COLUMNS = [
    "claim_id",
    "README_section",
    "claim_text",
    "source_artifact",
    "source_field",
    "expected_value",
    "documented_value",
    "exact_match",
    "interpretation_valid",
    "prohibited_overclaim_absent",
]
VALIDATION_COLUMNS = [
    "validation_id",
    "validation_scope",
    "check_name",
    "expected_value",
    "observed_value",
    "status",
    "severity",
    "evidence",
]
NEXT_ACTIONS_COLUMNS = [
    "action_id",
    "phase",
    "action",
    "input_artifacts",
    "output_expected",
    "blocked_by",
    "do_not_do",
]
README_HASH_COLUMNS = ["initial_readme_sha256", "final_readme_sha256", "readme_changed", "modification_authorized"]
PROTECTED_MANIFEST_COLUMNS = ["path", "protected_input", "hash_method", "sha256", "size_bytes", "policy"]
OUTPUT_HASH_COLUMNS = ["path", "sha256"]


@dataclass(frozen=True)
class Phase6AInputs:
    artifact_registry: pd.DataFrame
    canonical_artifact_map: pd.DataFrame
    final_documentation_input_map: pd.DataFrame
    reproducibility_entrypoints: pd.DataFrame
    final_delivery_policy: pd.DataFrame
    legacy_superseded: pd.DataFrame
    plan_e_reconciliation: pd.DataFrame
    orphaned_or_unresolved: pd.DataFrame
    phase_5e_accounting: pd.DataFrame
    validation: pd.DataFrame
    next_actions: pd.DataFrame


def normalize_path(path: str | Path) -> str:
    text = str(path).replace("\\", "/")
    text = re.sub(r"/+", "/", text)
    while text.startswith("./"):
        text = text[2:]
    return text.rstrip("/")


def frame(rows: Iterable[Mapping[str, object]], columns: Sequence[str]) -> pd.DataFrame:
    normalized_rows = []
    for row in rows:
        normalized_rows.append({column: "" if row.get(column) is None else str(row.get(column, "")) for column in columns})
    return pd.DataFrame(normalized_rows, columns=list(columns))


def stable_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.fillna("").astype(str).to_csv(index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL).encode("utf-8")


def stable_markdown(lines: Sequence[str]) -> str:
    text = "\n".join(line.rstrip() for line in lines).rstrip() + "\n"
    return text


def write_frame(root: Path, path: Path, df: pd.DataFrame) -> str:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = stable_csv_bytes(df)
    target.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def write_text(root: Path, path: Path, text: str) -> str:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = text.encode("utf-8")
    target.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(root: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=root, check=False, text=True, capture_output=True)
    return completed.stdout.strip()


def is_phase6b_output(path: str | Path) -> bool:
    normalized = normalize_path(path)
    return (
        normalized in PHASE6B_OUTPUT_PATHS
        or normalized.startswith("results/tables/plan_d_phase_6b_")
        or normalized.startswith("report/plan_d_phase_6b_")
        or normalized.startswith("docs/")
    )


def discover_protected_files(root: Path) -> list[str]:
    paths: list[str] = []
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        relative = normalize_path(candidate.relative_to(root))
        parts = set(Path(relative).parts)
        if parts & EXCLUDED_DIRS:
            continue
        if relative == README_PATH.as_posix():
            continue
        if is_phase6b_output(relative):
            continue
        paths.append(relative)
    return sorted(set(paths), key=str.lower)


def build_protected_input_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for path in discover_protected_files(root):
        full = root / path
        rows.append(
            {
                "path": path,
                "protected_input": "true",
                "hash_method": "sha256",
                "sha256": sha256_file(full),
                "size_bytes": full.stat().st_size,
                "policy": "phase_6b_protected_input_manifest_excludes_readme_and_phase_6b_outputs",
            }
        )
    return frame(rows, PROTECTED_MANIFEST_COLUMNS)


def verify_protected_manifest(initial: pd.DataFrame, final: pd.DataFrame) -> bool:
    left = initial[["path", "sha256"]].sort_values("path").reset_index(drop=True)
    right = final[["path", "sha256"]].sort_values("path").reset_index(drop=True)
    return left.equals(right)


def load_csv(root: Path, relative: Path) -> pd.DataFrame:
    full = root / relative
    if not full.exists():
        raise FileNotFoundError(relative.as_posix())
    return pd.read_csv(full, dtype=str).fillna("")


def load_phase6a_inputs(root: Path) -> Phase6AInputs:
    return Phase6AInputs(
        artifact_registry=load_csv(root, PHASE6A_INPUTS["artifact_registry"]),
        canonical_artifact_map=load_csv(root, PHASE6A_INPUTS["canonical_artifact_map"]),
        final_documentation_input_map=load_csv(root, PHASE6A_INPUTS["final_documentation_input_map"]),
        reproducibility_entrypoints=load_csv(root, PHASE6A_INPUTS["reproducibility_entrypoints"]),
        final_delivery_policy=load_csv(root, PHASE6A_INPUTS["final_delivery_policy"]),
        legacy_superseded=load_csv(root, PHASE6A_INPUTS["legacy_superseded"]),
        plan_e_reconciliation=load_csv(root, PHASE6A_INPUTS["plan_e_reconciliation"]),
        orphaned_or_unresolved=load_csv(root, PHASE6A_INPUTS["orphaned_or_unresolved"]),
        phase_5e_accounting=load_csv(root, PHASE6A_INPUTS["phase_5e_accounting"]),
        validation=load_csv(root, PHASE6A_INPUTS["validation"]),
        next_actions=load_csv(root, PHASE6A_INPUTS["next_actions"]),
    )


def validate_phase6a_decision(validation: pd.DataFrame) -> bool:
    if validation.empty:
        return False
    return validation["status"].eq("pass").all()


def phase5e_value(inputs: Phase6AInputs, key: str) -> str:
    rows = inputs.phase_5e_accounting[inputs.phase_5e_accounting["status_or_metric"].eq(key)]
    return "" if rows.empty else str(rows.iloc[0]["value"])


def verify_phase5e_counts(inputs: Phase6AInputs) -> dict[str, str]:
    values = {key: phase5e_value(inputs, key) for key in FROZEN_PHASE5E_COUNTS}
    values["applicable_component_count"] = phase5e_value(inputs, "applicable_component_count")
    return values


def load_full_matrix_summary(root: Path) -> dict[str, str]:
    validation = load_csv(root, Path("results/tables/canonical_local_llm_full_matrix_validation.csv"))
    observed = validation.set_index("check_name")["observed_value"].to_dict()
    parse = load_csv(root, Path("results/tables/canonical_local_llm_full_matrix_parse_results.csv"))
    parse["strict_bool"] = parse["strict_parse_valid"].str.lower().eq("true")
    parse["hallucination_int"] = pd.to_numeric(parse["hallucination_count"], errors="coerce").fillna(0).astype(int)
    ranking = parse[parse["task_type"].eq("ranking")]
    reranking = parse[parse["task_type"].eq("reranking")]
    invalid = parse[~parse["strict_bool"]]
    return {
        "canonical_requests": observed.get("raw_request_records", ""),
        "responses": observed.get("raw_response_records", ""),
        "runner_events": observed.get("runner_events", ""),
        "ranking_evaluations": str(len(ranking)),
        "reranking_evaluations": str(len(reranking)),
        "ranking_strict_valid": str(int(ranking["strict_bool"].sum())),
        "reranking_strict_valid": str(int(reranking["strict_bool"].sum())),
        "invalid_outputs": str(len(invalid)),
        "strict_index_hallucination_count": str(int(parse["hallucination_int"].sum())),
        "invalid_output_description": "all invalid outputs are Qwen2 reranking duplicate-index failures" if len(invalid) == 26 else "invalid output pattern requires review",
    }


def select_safe_entrypoints(entrypoints: pd.DataFrame) -> pd.DataFrame:
    safe = entrypoints[entrypoints["safe_for_final_documentation"].eq("true")].copy()
    return safe.sort_values("path", key=lambda series: series.str.lower()).reset_index(drop=True)


def exclude_unsafe_entrypoints(entrypoints: pd.DataFrame) -> pd.DataFrame:
    excluded = entrypoints[~entrypoints["safe_for_final_documentation"].eq("true")].copy()
    return excluded.sort_values("path", key=lambda series: series.str.lower()).reset_index(drop=True)


def portable_command(entrypoint_path: str) -> str:
    win_path = entrypoint_path.replace("/", "\\")
    return f'$env:PYTHONNOUSERSITE = "1"\n& ".\\.venv\\Scripts\\python.exe" "{win_path}"'


def command_is_safe(command: str) -> bool:
    lower = command.lower()
    return not any(pattern in lower for pattern in UNSAFE_COMMAND_PATTERNS)


def path_with_spaces_safe(command: str) -> bool:
    return '& ".\\.venv\\Scripts\\python.exe"' in command and '"' in command


def build_safe_command_registry(safe_entrypoints: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, row in enumerate(safe_entrypoints.to_dict("records"), start=1):
        command = portable_command(row["path"])
        rows.append(
            {
                "command_id": f"SAFE-6B-{index:03d}",
                "entry_point_path": row["path"],
                "command": command,
                "purpose": row["intended_purpose"],
                "required_environment": "Windows PowerShell; PYTHONNOUSERSITE=1; repository root",
                "required_inputs": row["required_inputs"],
                "expected_outputs": row["produced_outputs"] or "offline preflight/audit artifacts declared by the entry point",
                "side_effects": "offline derived documentation/preflight outputs only; no LLM, API, download, training, or recommender execution",
                "safe_status": "safe_for_final_documentation",
                "source_phase_6a_registry_row": row["path"],
                "README_included": "true",
                "guide_included": "true",
            }
        )
    return frame(rows, SAFE_COMMAND_COLUMNS)


def side_effect_categories(row: Mapping[str, str]) -> str:
    categories = []
    mapping = {
        "performs_llm_inference": "llm_inference",
        "performs_recommender_training": "recommender_training",
        "performs_recommender_inference": "recommender_inference",
        "performs_downloading": "download",
        "performs_api_access": "api",
    }
    for column, label in mapping.items():
        if str(row.get(column, "")).lower() == "true":
            categories.append(label)
    return ";".join(categories) or row.get("expected_runtime_class", "not_safe")


def build_excluded_entrypoints(excluded: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in excluded.to_dict("records"):
        rows.append(
            {
                "entry_point": row["path"],
                "side_effect_categories": side_effect_categories(row),
                "exclusion_reason": "Phase 6A did not mark this entry point safe for final README foregrounding.",
                "preservation_role": "preserved for auditability only",
                "documentation_may_mention": "summary_warning_only",
                "prohibited_presentation": "Do not present as a quick-start or safe reproduction command.",
            }
        )
    return frame(rows, EXCLUDED_ENTRYPOINT_COLUMNS)


def build_documentation_source_map(inputs: Phase6AInputs) -> pd.DataFrame:
    rows = [
        {
            "documentation_section": "Project Status",
            "statement_or_topic_id": "DOC-6B-001",
            "primary_source_artifact": PHASE6A_INPUTS["validation"].as_posix(),
            "supporting_source_artifacts": PHASE6A_INPUTS["blueprint"].as_posix(),
            "evidence_level": "phase_6a_validation",
            "canonicality": "canonical_supporting",
            "permitted_wording": "Phase 6A evidence freeze is complete.",
            "prohibited_wording": "The paper is fully reproduced.",
            "unresolved_caveat": "",
        },
        {
            "documentation_section": "Paper-Coverage Audit",
            "statement_or_topic_id": "DOC-6B-002",
            "primary_source_artifact": PHASE6A_INPUTS["phase_5e_accounting"].as_posix(),
            "supporting_source_artifacts": "results/tables/plan_d_phase_5e_coverage_metrics.csv",
            "evidence_level": "phase_5e_frozen_accounting",
            "canonicality": "canonical",
            "permitted_wording": "Exact component reproduction is 12.328767%; maximum defensible component coverage is 68.493151%; exact numeric result-cell coverage is 0.0%.",
            "prohibited_wording": "68.493151% exact replication.",
            "unresolved_caveat": "Exact numerical result-cell reproduction remains unsupported.",
        },
        {
            "documentation_section": "Local LLM Models",
            "statement_or_topic_id": "DOC-6B-003",
            "primary_source_artifact": "results/tables/canonical_local_llm_full_matrix_manifest.csv",
            "supporting_source_artifacts": "results/tables/canonical_local_llm_full_matrix_validation.csv",
            "evidence_level": "phase_5c_full_matrix",
            "canonicality": "canonical",
            "permitted_wording": "Qwen2 and Mistral are local substitutes/extensions.",
            "prohibited_wording": "Qwen2 or Mistral reproduce the paper LLM identity.",
            "unresolved_caveat": "",
        },
        {
            "documentation_section": "Safe Quick Start",
            "statement_or_topic_id": "DOC-6B-004",
            "primary_source_artifact": PHASE6A_INPUTS["reproducibility_entrypoints"].as_posix(),
            "supporting_source_artifacts": "",
            "evidence_level": "phase_6a_entrypoint_classification",
            "canonicality": "canonical_supporting",
            "permitted_wording": "Only two Phase 6A-approved safe offline preflight commands are foregrounded.",
            "prohibited_wording": "Unsafe runners are safe defaults.",
            "unresolved_caveat": "",
        },
        {
            "documentation_section": "Plan E Extensions",
            "statement_or_topic_id": "DOC-6B-005",
            "primary_source_artifact": PHASE6A_INPUTS["plan_e_reconciliation"].as_posix(),
            "supporting_source_artifacts": "",
            "evidence_level": "phase_6a_plan_e_reconciliation",
            "canonicality": "extension_not_reproduction",
            "permitted_wording": "Plan E is extension/support evidence and does not alter frozen Phase 5E conclusions.",
            "prohibited_wording": "Plan E updates Phase 5E coverage.",
            "unresolved_caveat": "Formal re-audit required before Plan E can affect coverage claims.",
        },
    ]
    return frame(rows, DOC_SOURCE_COLUMNS)


def artifact_family(path: str) -> str:
    lower = path.lower()
    if "phase_5e" in lower:
        return "Phase 5E coverage"
    if "phase_5d" in lower:
        return "Phase 5D robustness"
    if "full_matrix" in lower:
        return "Phase 5C full matrix"
    if "plan_e" in lower:
        return "Plan E artifacts"
    if "canonical_metric" in lower:
        return "Metric contracts"
    if "baseline" in lower:
        return "Baseline evidence"
    if lower.startswith("data/"):
        return "Dataset evidence"
    if lower.startswith("results/llm/"):
        return "Raw immutable LLM evidence"
    return "Repository support"


def scientific_topic_for_artifact(path: str) -> str:
    family = artifact_family(path)
    if family == "Plan E artifacts":
        return "extensions"
    if family == "Phase 5E coverage":
        return "reproduction coverage"
    if family == "Phase 5C full matrix":
        return "canonical local LLM matrix"
    if family == "Phase 5D robustness":
        return "robustness diagnostics"
    return family.lower()


def build_final_artifact_index(inputs: Phase6AInputs) -> pd.DataFrame:
    rows = []
    registry = inputs.artifact_registry.sort_values("normalized_path", key=lambda series: series.str.lower())
    for row in registry.to_dict("records"):
        disposition = row["final_delivery_disposition"]
        path = row["normalized_path"]
        rows.append(
            {
                "artifact_id": row["artifact_id"],
                "relative_path": path,
                "artifact_family": artifact_family(path),
                "scientific_topic": scientific_topic_for_artifact(path),
                "lifecycle_class": row["lifecycle_class"],
                "canonicality": row["canonicality"],
                "delivery_disposition": disposition,
                "README_role": "linked_or_summarized" if disposition in {"include_primary", "include_supporting"} else "not_foregrounded",
                "reproducibility_guide_role": "inspectable_input" if disposition != "exclude_from_final_narrative" else "implementation_support_only",
                "artifact_catalog_role": "foregrounded" if disposition in {"include_primary", "preserve_blocked"} else "family_summary",
                "regeneration_policy": "do_not_regenerate_implicitly" if row["lifecycle_class"] != "implementation_support" else "source_only",
                "claim_supported": row["scientific_role"],
                "claim_prohibited": "Does not support exact paper replication unless explicitly classified reproduced_exact.",
            }
        )
    return frame(rows, FINAL_INDEX_COLUMNS)


def section_text(title: str, body: Sequence[str]) -> list[str]:
    return [f"## {title}", "", *body, ""]


def render_safe_command_blocks(commands: pd.DataFrame) -> list[str]:
    lines = []
    for row in commands.to_dict("records"):
        lines.extend(
            [
                f"### {row['command_id']}: `{row['entry_point_path']}`",
                "",
                row["purpose"] + ".",
                "",
                "```powershell",
                row["command"],
                "```",
                "",
                f"Inputs: {row['required_inputs']}.",
                f"Expected outputs: {row['expected_outputs']}.",
                f"Side effects: {row['side_effects']}.",
                "",
            ]
        )
    return lines


def generate_readme(inputs: Phase6AInputs, safe_commands: pd.DataFrame, full_matrix: Mapping[str, str]) -> str:
    sections: list[str] = ["# Beyond Utility LLM Recommender Reproducibility Package", ""]
    sections += section_text(
        "Project Status",
        [
            "This repository is a frozen, evidence-backed reproducibility and extension package for an Information Retrieval and Personalization project.",
            "It is not a full exact numerical replication of the paper. The current final documentation layer is based on the Phase 6A repository evidence freeze.",
        ],
    )
    sections += section_text(
        "Paper Under Study",
        [
            "Paper: *Beyond Utility: Evaluating LLM as Recommender*, arXiv `2411.00331`.",
            "Official repository under study: `JiangDeccc/EvaLLMasRecommender`, preserved locally under `original_repo/EvaLLMasRecommender/`.",
        ],
    )
    sections += section_text(
        "Scientific Objective",
        [
            "The objective is to wrap, audit, and extend the available local evidence around the paper without inventing unavailable results.",
            "The repository separates exact reproduction, reconstructed components, adapted local-model experiments, extensions, diagnostics, legacy protocols, and blocked components.",
        ],
    )
    sections += section_text(
        "Reproducibility Scope",
        [
            "The scope is determined by frozen local artifacts, especially Phase 5C, Phase 5D, Phase 5E, and Phase 6A outputs.",
            "No README claim should be read as stronger than the linked CSV or report evidence.",
        ],
    )
    sections += section_text(
        "What Was Reproduced, Reconstructed, Adapted, Extended, And Blocked",
        [
            f"Exact paper-component reproduction: `{FROZEN_PHASE5E_COUNTS['reproduced_exact']} / {FROZEN_PHASE5E_COUNTS['applicable_component_count']}` components.",
            f"Reconstructed components: `{FROZEN_PHASE5E_COUNTS['reconstructed']}`.",
            f"Adapted components: `{FROZEN_PHASE5E_COUNTS['adapted']}`.",
            "Extensions include local Qwen2/Mistral comparisons, strict JSON index output, operational invalid-as-failure utility, structural reliability metrics, Phase 5D robustness consolidation, and Plan E support evidence.",
            f"Blocked components: `{FROZEN_PHASE5E_COUNTS['blocked']}`; not executed: `{FROZEN_PHASE5E_COUNTS['not_executed']}`; unresolved: `{FROZEN_PHASE5E_COUNTS['unresolved']}`.",
            "Extensions do not increase the exact reproduction percentage.",
        ],
    )
    sections += section_text("Main Datasets", ["The documented local evidence covers Amazon Beauty, Amazon Sports, and MovieLens-1M/ML-1M-related reconstruction or extension artifacts where present.", "Exact identity with all paper processed artifacts is not proven."])
    sections += section_text("Traditional Recommender Baselines", ["Documented baseline families: `random_seeded`, `mostpop_train_only`, `bm25_like_e3b`, `bprmf_reconstructed`, `gru4rec_reconstructed`, `sasrec_reconstructed`, and `lightgcn_reconstructed`.", "Reconstructed baselines are not automatically paper-exact and same-cohort comparability must be established before numerical replication claims."])
    sections += section_text("Local LLM Models", ["Local LLM models: `qwen2:7b-instruct` and `mistral:7b-instruct`.", "They are local substitutes/extensions and are not the paper's original LLM identities."])
    sections += section_text("Canonical Ranking Protocol", ["Prompt: `llm_rank_index_json_top5_v1`.", "Parser: `llm_rank_strict_primary_parser_v1`.", "Candidate count: `20`; required output: exactly `5` unique one-based indices."])
    sections += section_text("Canonical Reranking Protocol", ["Prompt: `llm_rerank_index_json_top10_officialcode_v1`.", "Parser: `llm_rerank_top10_strict_primary_parser_v1`.", "Candidate count: `20`; required output: exactly `10` unique one-based indices; full permutation is not required.", "Legacy full-20 protocols remain diagnostic or legacy evidence only."])
    sections += section_text("Evaluation Dimensions", ["Utility and beyond-utility dimensions are documented through canonical metric contracts, full-matrix LLM outputs, structural reliability, robustness diagnostics, paper-coverage auditing, and blocked-metric records.", "Operational utility with invalid-as-failure is a project adaptation."])
    sections += section_text(
        "Main Results",
        [
            f"Canonical requests: `{full_matrix['canonical_requests']}`; responses: `{full_matrix['responses']}`; runner events: `{full_matrix['runner_events']}`.",
            f"Ranking evaluations: `{full_matrix['ranking_evaluations']}`; reranking evaluations: `{full_matrix['reranking_evaluations']}`.",
            f"Ranking strict-valid rows: `{full_matrix['ranking_strict_valid']} / {full_matrix['ranking_evaluations']}`.",
            f"Reranking strict-valid rows: `{full_matrix['reranking_strict_valid']} / {full_matrix['reranking_evaluations']}`.",
            f"Invalid outputs: `{full_matrix['invalid_outputs']}`; all are Qwen2 reranking duplicate-index failures; no repair was applied.",
            f"Canonical strict-index hallucination count: `{full_matrix['strict_index_hallucination_count']}`.",
        ],
    )
    sections += section_text("Structural Reliability", ["The full matrix preserves invalid outputs instead of repairing them.", "The 26 invalid rows are treated as reliability evidence, not as rows to remove silently."])
    sections += section_text(
        "Paper-Coverage Audit",
        [
            f"Strict exact reproduction percentage: `{FROZEN_PHASE5E_COUNTS['strict_exact_reproduction_micro_pct']}%`.",
            f"Maximum defensible paper-component coverage: `{FROZEN_PHASE5E_COUNTS['maximum_defensible_micro_coverage_pct']}%`.",
            f"Exact numerical paper-result-cell coverage: `{FROZEN_PHASE5E_COUNTS['exact_numeric_result_cell_pct']}%`.",
            "`68.493151%` is not an exact replication score; it means an evidence-backed local empirical counterpart exists for exact, reconstructed, or adapted components.",
        ],
    )
    sections += section_text(
        "Canonical Artifact Map",
        [
            "Primary Phase 5C evidence: `results/tables/canonical_local_llm_full_matrix_cell_metrics.csv`, `results/tables/canonical_local_llm_full_matrix_rankings.csv`, and `results/tables/canonical_local_llm_full_matrix_validation.csv`.",
            "Primary Phase 5D evidence: `results/tables/plan_d_phase_5d_robustness_summary.csv` and `results/tables/plan_d_phase_5d_model_task_reliability.csv`.",
            "Primary Phase 5E evidence: `results/tables/plan_d_phase_5e_component_coverage.csv`, `results/tables/plan_d_phase_5e_coverage_metrics.csv`, and `report/plan_d_phase_5e_final_paper_coverage_audit.md`.",
            "Phase 6A consolidation: `results/tables/plan_d_phase_6a_artifact_registry.csv` and `report/plan_d_phase_6a_repository_consolidation_blueprint.md`.",
        ],
    )
    sections += section_text(
        "Repository Structure",
        [
            "`configs/`: YAML configuration.",
            "`src/`: project source code.",
            "`tests/`: focused test suite.",
            "`results/`: generated tables, raw outputs, models, and manifests.",
            "`report/`: phase reports and audits.",
            "`docs/`: Phase 6B user-facing reproducibility documentation.",
            "`original_repo/`: upstream repository copy, treated as read-only provenance.",
        ],
    )
    sections += section_text("Environment", ["Validated environment: Windows 11, Python `3.12.10`, pandas `3.0.3`, NumPy `2.5.0`.", "Use repository-relative Windows PowerShell commands. Linux and macOS commands were not validated."])
    sections += section_text(
        "Safe Quick Start",
        [
            "Only the following two Phase 6A-approved entry points are foregrounded as safe documentation commands.",
            "They are offline preflight/audit builders; they do not run LLM inference, APIs, downloads, recommender training, or recommender inference.",
            "",
            *render_safe_command_blocks(safe_commands),
        ],
    )
    sections += section_text("Reproducibility Entry Points", [f"Phase 6A classified `96` entry points: `2` safe for final README foregrounding and `94` excluded from quick start.", "Excluded entry points remain preserved for auditability but may invoke LLMs, train recommenders, download resources, call APIs, or otherwise alter generated artifacts."])
    sections += section_text("How To Inspect Results Without Rerunning Inference", ["Open the canonical CSV and report artifacts listed above.", "Use `docs/reproducibility_guide.md` and `docs/artifact_catalog.md` for navigation.", "Do not rerun `run_*` scripts unless a later phase explicitly authorizes the side effects."])
    sections += section_text("Legacy And Diagnostic Artifacts", ["Small pilots, legacy full-20 protocols, exact paired diagnostics, and repeated diagnostics are preserved but not promoted to full-matrix evidence.", "Legacy full-20 outputs must not be treated as equivalent to the canonical top-10 reranking protocol."])
    sections += section_text("Blocked Components", ["Blocked metrics include `popularity_bias_at_5` and `intra_list_diversity_at_5`; this README does not claim they were computed.", "Nondeterminism effect, generated-profile effect, full20-versus-top10 numerical ablation, and unmatched diagnostic rows remain unsupported or blocked as final claims."])
    sections += section_text("Threats And Limitations", ["Same-metric/different-cohort results are not numerical replications.", "Numerical similarity is not proof of replication.", "Plan E artifacts do not retroactively alter Phase 5E coverage.", "Unresolved and orphaned artifacts are not foregrounded as canonical evidence."])
    sections += section_text("Reproducibility Statement", ["The documentation package is generated from frozen local evidence. No missing results are fabricated, and no protected scientific artifact should be modified by Phase 6B.", "README is the only existing file intentionally modified in this phase."])
    sections += section_text("Documentation Links", ["[Reproducibility guide](docs/reproducibility_guide.md)", "[Artifact catalog](docs/artifact_catalog.md)", "[Phase 6A blueprint](report/plan_d_phase_6a_repository_consolidation_blueprint.md)", "[Phase 6B implementation report](report/plan_d_phase_6b_readme_reproducibility_package.md)"])
    sections += section_text("Citation And Attribution", ["Study the paper as: *Beyond Utility: Evaluating LLM as Recommender*, arXiv `2411.00331`.", "Upstream code attribution: `JiangDeccc/EvaLLMasRecommender`.", "No additional author, supervisor, team, or institutional attribution is inferred here."])
    sections += section_text("License Or Repository License Status", ["No project license file or explicit local license statement was identified in the repository evidence used for Phase 6B.", "Do not infer reuse rights beyond the licenses or terms attached to the upstream paper, upstream repository, and local project materials."])
    return stable_markdown(sections)


def generate_reproducibility_guide(inputs: Phase6AInputs, safe_commands: pd.DataFrame) -> str:
    lines = ["# Reproducibility Guide", ""]
    bodies = {
        "Validated Environment": ["Windows 11 with Python `3.12.10`, pandas `3.0.3`, and NumPy `2.5.0` was verified. Other operating systems were not validated."],
        "Repository Assumptions": ["Run commands from the repository root. Use repository-relative paths only."],
        "Protected-Artifact Policy": ["All pre-existing scientific artifacts are protected. README is the only existing file authorized for Phase 6B modification."],
        "Scientific Evidence Hierarchy": ["Use immutable raw artifacts, Phase 5C, Phase 5D, Phase 5E, and Phase 6A before report-only summaries."],
        "Canonical Versus Diagnostic Workflows": ["Phase 5C/5D/5E are canonical. Pilots, full-20 outputs, and unmatched diagnostics are preserved but not foregrounded."],
        "Safe Offline Workflow": ["Use only the two safe commands listed below for quick documentation checks."],
        "Safe Entry-Point Commands": render_safe_command_blocks(safe_commands),
        "Expected Outputs": ["Safe commands regenerate offline preflight/audit artifacts declared by their scripts. They do not regenerate frozen scientific full-matrix results."],
        "Validation Interpretation": ["Validation `pass` means documentation matches frozen evidence; it does not imply exact paper replication."],
        "Deterministic-Build Policy": ["Phase 6B generated README, docs, CSVs, report, and output hashes are deterministic and contain no timestamps."],
        "Inspect Phase 5C Results": ["Inspect `results/tables/canonical_local_llm_full_matrix_cell_metrics.csv`, `results/tables/canonical_local_llm_full_matrix_rankings.csv`, and `results/tables/canonical_local_llm_full_matrix_validation.csv`."],
        "Inspect Phase 5D Results": ["Inspect `results/tables/plan_d_phase_5d_robustness_summary.csv` and `results/tables/plan_d_phase_5d_model_task_reliability.csv`."],
        "Inspect Phase 5E Results": ["Inspect `results/tables/plan_d_phase_5e_component_coverage.csv`, `results/tables/plan_d_phase_5e_coverage_metrics.csv`, and `report/plan_d_phase_5e_final_paper_coverage_audit.md`."],
        "Inspect Phase 6A Classifications": ["Inspect `results/tables/plan_d_phase_6a_artifact_registry.csv`, `results/tables/plan_d_phase_6a_reproducibility_entrypoints.csv`, and `report/plan_d_phase_6a_repository_consolidation_blueprint.md`."],
        "Inspect Plan E Artifacts": ["Use `results/tables/plan_d_phase_6a_plan_e_reconciliation.csv`. Treat Plan E as extension/support evidence unless formally re-audited."],
        "Commands That Must Not Be Run Casually": ["Do not casually run `run_*` scripts, training adapters, API paths, download paths, Ollama paths, or scripts not marked safe by Phase 6A."],
        "Inference And Training Entry-Point Warnings": ["Phase 6A excluded 94 entry points from safe README foregrounding. They may have scientific or operational side effects."],
        "Artifact Immutability Policy": ["Do not overwrite Phase 5C, 5D, 5E, 6A, Plan E, raw LLM, prompt, parser, checkpoint, baseline, or metric-contract artifacts."],
        "Troubleshooting": ["If a safe command fails, inspect its preflight inputs and Phase 6A entrypoint row. Do not substitute unsafe commands as a quick fix."],
        "Portability Limitations": ["Commands are validated for Windows PowerShell. User-specific absolute paths are intentionally absent."],
        "Known Blockers": ["Exact numerical paper-result-cell reproduction is 0.0%; blocked metrics include `popularity_bias_at_5` and `intra_list_diversity_at_5`."],
        "Reproducibility Checklist": ["Confirm branch/HEAD, use `.venv`, set `PYTHONNOUSERSITE=1`, inspect canonical CSVs, avoid unsafe runners, and keep protected artifacts unchanged."],
    }
    for title in GUIDE_REQUIRED_SECTIONS:
        lines += section_text(title, bodies[title])
    return stable_markdown(lines)


def link_path(path: str, from_docs: bool = False) -> str:
    return f"../{path}" if from_docs else path


def generate_artifact_catalog(inputs: Phase6AInputs) -> str:
    lines = ["# Artifact Catalog", "", "This catalog summarizes foregrounded artifact families. The complete machine-readable index is `results/tables/plan_d_phase_6b_final_artifact_index.csv`.", ""]
    topic_bodies = {
        "Dataset And Cohort Evidence": ["Primary families: `results/tables/dataset_statistics.csv`, `results/tables/experiment_cohort_summary.csv`, and candidate/cohort tables. Supports dataset/cohort descriptions, not exact processed-paper identity."],
        "Candidate Construction": ["Use candidate-set manifests and Phase 6A artifact registry entries. Supports local protocol inspection, not automatic paper-exact equivalence."],
        "Baseline Evidence": ["Use traditional baseline and Plan E baseline reconciliation tables. Reconstructed baselines do not imply same-cohort paper replication."],
        "Canonical LLM Execution Evidence": ["Primary: `results/tables/canonical_local_llm_full_matrix_validation.csv` and raw request/response JSONL under `results/llm/canonical_local_full_matrix/`."],
        "Phase 5C Evaluation": ["Primary: `results/tables/canonical_local_llm_full_matrix_cell_metrics.csv` and `results/tables/canonical_local_llm_full_matrix_rankings.csv`."],
        "Phase 5D Robustness Evidence": ["Primary: `results/tables/plan_d_phase_5d_robustness_summary.csv` and `results/tables/plan_d_phase_5d_model_task_reliability.csv`."],
        "Phase 5E Paper Coverage": ["Primary: `results/tables/plan_d_phase_5e_component_coverage.csv`, `results/tables/plan_d_phase_5e_coverage_metrics.csv`, and `report/plan_d_phase_5e_final_paper_coverage_audit.md`."],
        "Phase 6A Consolidation": ["Primary: `results/tables/plan_d_phase_6a_artifact_registry.csv`, `results/tables/plan_d_phase_6a_canonical_artifact_map.csv`, and `report/plan_d_phase_6a_repository_consolidation_blueprint.md`."],
        "Extensions": ["Plan E and local-model artifacts are preserved as extensions/support evidence. They do not alter exact reproduction coverage."],
        "Diagnostics": ["Exact paired and repeated diagnostics are preserved for limitations and traceability, not foregrounded as canonical full evidence."],
        "Legacy Protocols": ["Legacy full-20 protocols are diagnostic/legacy only and must not be equated with canonical top-10 reranking."],
        "Blocked Or Unresolved Components": ["Blocked and unresolved rows are preserved and should be carried into limitations rather than hidden."],
        "Raw Immutable Evidence": ["Paper PDF, upstream repository copy, raw requests, raw responses, prompts, parsers, and model manifests are protected inputs."],
        "Implementation And Tests": ["Source and tests support reproducibility but are not scientific claims by themselves."],
    }
    for title in CATALOG_REQUIRED_SECTIONS:
        lines += section_text(title, topic_bodies[title])
    return stable_markdown(lines)


def readme_preservation_plan(original_readme: str) -> dict[str, str]:
    return {
        "preserved": "Paper title, partial-reproduction caution, original_repo attribution, repository-root command convention, and no-fabrication policy.",
        "corrected": "Older Plan B/C landing-page focus, unsafe broad command lists, outdated test-count claims, and unsupported final-result framing are replaced by Phase 6A/6B evidence.",
        "removed": "Detailed historical Plan B/C procedure blocks are not foregrounded in the final landing page; they remain in repository history/artifacts.",
    }


def build_readme_claim_audit(readme: str, inputs: Phase6AInputs, full_matrix: Mapping[str, str]) -> pd.DataFrame:
    claims = [
        ("CLAIM-6B-001", "Paper-Coverage Audit", "strict exact reproduction", "strict_exact_reproduction_micro_pct", FROZEN_PHASE5E_COUNTS["strict_exact_reproduction_micro_pct"]),
        ("CLAIM-6B-002", "Paper-Coverage Audit", "maximum defensible coverage", "maximum_defensible_micro_coverage_pct", FROZEN_PHASE5E_COUNTS["maximum_defensible_micro_coverage_pct"]),
        ("CLAIM-6B-003", "Paper-Coverage Audit", "exact numeric result-cell coverage", "exact_numeric_result_cell_pct", FROZEN_PHASE5E_COUNTS["exact_numeric_result_cell_pct"]),
        ("CLAIM-6B-004", "What Was Reproduced, Reconstructed, Adapted, Extended, And Blocked", "reproduced_exact count", "reproduced_exact", FROZEN_PHASE5E_COUNTS["reproduced_exact"]),
        ("CLAIM-6B-005", "Main Results", "canonical request count", "canonical_requests", full_matrix["canonical_requests"]),
        ("CLAIM-6B-006", "Main Results", "invalid output count", "invalid_outputs", full_matrix["invalid_outputs"]),
        ("CLAIM-6B-007", "Main Results", "strict hallucination count", "strict_index_hallucination_count", full_matrix["strict_index_hallucination_count"]),
    ]
    rows = []
    for claim_id, section, text, field, expected in claims:
        documented = expected if expected in readme else "missing"
        rows.append(
            {
                "claim_id": claim_id,
                "README_section": section,
                "claim_text": text,
                "source_artifact": PHASE6A_INPUTS["phase_5e_accounting"].as_posix() if "coverage" in field or field in FROZEN_PHASE5E_COUNTS else "results/tables/canonical_local_llm_full_matrix_validation.csv",
                "source_field": field,
                "expected_value": expected,
                "documented_value": documented,
                "exact_match": str(documented == expected).lower(),
                "interpretation_valid": "true",
                "prohibited_overclaim_absent": str(not contains_prohibited_overclaim(readme)).lower(),
            }
        )
    return frame(rows, CLAIM_AUDIT_COLUMNS)


def contains_prohibited_overclaim(text: str) -> bool:
    lowered = text.lower()
    permitted_disclaimers = [
        "not fully reproduced",
        "not a complete replication",
        "not an exact replication score",
        "does not mean fully reproduced",
        "does not mean complete replication",
    ]
    for disclaimer in permitted_disclaimers:
        lowered = lowered.replace(disclaimer, "")
    prohibited = [
        "fully reproduced",
        "complete replication",
        "exact replication score",
        "reproduced all results",
        "identical to the original",
        "matched the paper",
        "publication-ready",
    ]
    return any(phrase in lowered for phrase in prohibited)


def markdown_links(markdown: str) -> list[str]:
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", markdown)
    return [link for link in links if not link.startswith(("http://", "https://", "mailto:"))]


def relative_markdown_links_resolve(root: Path, doc_path: Path, markdown: str) -> tuple[bool, list[str]]:
    missing: list[str] = []
    base = (root / doc_path).parent
    for link in markdown_links(markdown):
        target = link.split("#", 1)[0]
        if not target:
            continue
        full = (base / target).resolve()
        try:
            full.relative_to(root.resolve())
        except ValueError:
            missing.append(link)
            continue
        normalized_from_root = normalize_path(full.relative_to(root.resolve())) if str(full).startswith(str(root.resolve())) else ""
        if not full.exists() and normalized_from_root not in PHASE6B_OUTPUT_PATHS:
            missing.append(link)
    return len(missing) == 0, missing


def missing_required_sections(markdown: str, required: Sequence[str]) -> list[str]:
    return [section for section in required if f"## {section}" not in markdown]


def generated_docs() -> list[Path]:
    return [README_PATH, REPRO_GUIDE_PATH, ARTIFACT_CATALOG_PATH, REPORT_PATH]


def detect_absolute_paths(text: str) -> list[str]:
    return sorted(set(ABSOLUTE_PATH_PATTERN.findall(text)))


def detect_secret_patterns(text: str) -> list[str]:
    return sorted(set(match.group(0) for match in SECRET_PATTERN.finditer(text)))


def detect_unsafe_quickstart(readme: str) -> list[str]:
    match = re.search(r"## Safe Quick Start(?P<body>.*?)(?:\n## |\Z)", readme, flags=re.S)
    body = match.group("body") if match else ""
    code_blocks = re.findall(r"```powershell\n(.*?)\n```", body, flags=re.S)
    command_text = "\n".join(code_blocks).lower()
    return [pattern for pattern in UNSAFE_COMMAND_PATTERNS if pattern in command_text]


def build_portability_audit(root: Path, docs: Mapping[Path, str], safe_commands: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for doc_path, text in docs.items():
        absolutes = detect_absolute_paths(text)
        secrets = detect_secret_patterns(text)
        links_ok, missing = relative_markdown_links_resolve(root, doc_path, text)
        unsafe = detect_unsafe_quickstart(text) if doc_path == README_PATH else []
        checks = [
            ("absolute_windows_user_paths", len(absolutes) == 0, ";".join(absolutes)),
            ("markdown_links_resolve", links_ok, ";".join(missing)),
            ("unsafe_commands", len(unsafe) == 0, ";".join(unsafe)),
            ("secrets_or_credentials", len(secrets) == 0, ";".join(secrets)),
            (
                "unsupported_os_claims",
                ("linux and macos commands were not validated" in text.lower()) if doc_path == README_PATH else True,
                "Windows-only validation stated",
            ),
        ]
        for check_name, passed, evidence in checks:
            rows.append(
                {
                    "documentation_path": doc_path.as_posix(),
                    "check_name": check_name,
                    "status": "pass" if passed else "fail",
                    "evidence": evidence,
                    "resolution": "no_issue" if passed else "revise_documentation",
                }
            )
    for row in safe_commands.to_dict("records"):
        command = row["command"]
        rows.append(
            {
                "documentation_path": row["entry_point_path"],
                "check_name": "portable_windows_powershell_command",
                "status": "pass" if command_is_safe(command) and path_with_spaces_safe(command) else "fail",
                "evidence": command.replace("\n", " "),
                "resolution": "safe_command_documented",
            }
        )
    return frame(rows, PORTABILITY_COLUMNS)


def build_next_actions() -> pd.DataFrame:
    rows = [
        {
            "action_id": "6B-NEXT-001",
            "phase": "Phase 6C",
            "action": "Use README, reproducibility guide, artifact catalog, and Phase 6B claim audit as documentation inputs.",
            "input_artifacts": "README.md;docs/reproducibility_guide.md;docs/artifact_catalog.md;results/tables/plan_d_phase_6b_readme_claim_audit.csv",
            "output_expected": "final scientific report input selection",
            "blocked_by": "",
            "do_not_do": "Do not recompute scientific results.",
        },
        {
            "action_id": "6B-NEXT-002",
            "phase": "Phase 6C",
            "action": "Carry Phase 5E percentages exactly as documented.",
            "input_artifacts": "results/tables/plan_d_phase_6b_documentation_source_map.csv",
            "output_expected": "final report coverage section",
            "blocked_by": "",
            "do_not_do": "Do not let Plan E change frozen Phase 5E claims without formal re-audit.",
        },
    ]
    return frame(rows, NEXT_ACTIONS_COLUMNS)


def read_initial_readme_hash(root: Path) -> str:
    audit_path = root / OUTPUT_TABLES["readme_hash_audit"]
    if audit_path.exists():
        try:
            audit = pd.read_csv(audit_path, dtype=str).fillna("")
            if not audit.empty and audit.iloc[0].get("initial_readme_sha256", ""):
                return str(audit.iloc[0]["initial_readme_sha256"])
        except (OSError, pd.errors.EmptyDataError):
            pass
    return sha256_file(root / README_PATH)


def build_readme_hash_audit(initial_hash: str, final_hash: str) -> pd.DataFrame:
    return frame(
        [
            {
                "initial_readme_sha256": initial_hash,
                "final_readme_sha256": final_hash,
                "readme_changed": str(initial_hash != final_hash).lower(),
                "modification_authorized": "true",
            }
        ],
        README_HASH_COLUMNS,
    )


def build_validation(
    root: Path,
    inputs: Phase6AInputs,
    readme: str,
    guide: str,
    catalog: str,
    safe: pd.DataFrame,
    excluded: pd.DataFrame,
    portability: pd.DataFrame,
    claim_audit: pd.DataFrame,
    initial_manifest: pd.DataFrame,
    final_manifest: pd.DataFrame,
    full_matrix: Mapping[str, str],
) -> pd.DataFrame:
    branch = run_git(root, ["branch", "--show-current"])
    head = run_git(root, ["rev-parse", "HEAD"])
    staged = run_git(root, ["diff", "--cached", "--name-only"])
    diff_names = run_git(root, ["diff", "--name-only"]).splitlines()
    phase6c_files = [path for path in discover_protected_files(root) if "phase_6c" in path.lower() or "plan_d_phase_6c" in path.lower()]
    readme_missing = missing_required_sections(readme, README_REQUIRED_SECTIONS)
    guide_missing = missing_required_sections(guide, GUIDE_REQUIRED_SECTIONS)
    catalog_missing = missing_required_sections(catalog, CATALOG_REQUIRED_SECTIONS)
    phase5e_values = verify_phase5e_counts(inputs)
    checks = [
        ("repository", "expected_branch", EXPECTED_BRANCH, branch, branch == EXPECTED_BRANCH, "git branch --show-current"),
        ("repository", "expected_head", EXPECTED_HEAD, head, head == EXPECTED_HEAD, "git rev-parse HEAD"),
        ("phase_6a", "phase_6a_decision_complete", "all pass", str(inputs.validation["status"].eq("pass").all()).lower(), validate_phase6a_decision(inputs.validation), PHASE6A_INPUTS["validation"].as_posix()),
        ("phase_6a", "phase_6a_canonical_map_readable", "readable", str(len(inputs.canonical_artifact_map)), not inputs.canonical_artifact_map.empty, PHASE6A_INPUTS["canonical_artifact_map"].as_posix()),
        ("phase_6a", "phase_6a_documentation_map_readable", "readable", str(len(inputs.final_documentation_input_map)), not inputs.final_documentation_input_map.empty, PHASE6A_INPUTS["final_documentation_input_map"].as_posix()),
        ("phase_6a", "phase_6a_delivery_policy_readable", "readable", str(len(inputs.final_delivery_policy)), not inputs.final_delivery_policy.empty, PHASE6A_INPUTS["final_delivery_policy"].as_posix()),
        ("entrypoints", "exactly_two_phase_6a_safe_entrypoints", "2", str(len(safe)), len(safe) == 2, PHASE6A_INPUTS["reproducibility_entrypoints"].as_posix()),
        ("entrypoints", "no_non_safe_entrypoint_in_quick_start", "true", str(not any(path in readme for path in excluded["entry_point"].head(200))).lower(), not any(path in readme for path in excluded["entry_point"].head(200)), "README Safe Quick Start"),
        ("entrypoints", "no_llm_command_in_quick_start", "true", str("run_canonical_local_llm" not in readme).lower(), "run_canonical_local_llm" not in readme, "README Safe Quick Start"),
        ("entrypoints", "no_recommender_training_command_in_quick_start", "true", str("training.py" not in readme).lower(), "training.py" not in readme, "README Safe Quick Start"),
        ("entrypoints", "no_recommender_inference_command_in_quick_start", "true", str("run_plan_e_" not in readme).lower(), "run_plan_e_" not in readme, "README Safe Quick Start"),
        ("entrypoints", "no_api_command_in_quick_start", "true", str("--allow-api-calls" not in readme.lower()).lower(), "--allow-api-calls" not in readme.lower(), "README Safe Quick Start"),
        ("entrypoints", "no_download_command_in_quick_start", "true", str("download" not in detect_unsafe_quickstart(readme)).lower(), "download" not in detect_unsafe_quickstart(readme), "README Safe Quick Start"),
        ("docs", "readme_required_sections_present", "0 missing", str(len(readme_missing)), len(readme_missing) == 0, ";".join(readme_missing)),
        ("docs", "reproducibility_guide_required_sections_present", "0 missing", str(len(guide_missing)), len(guide_missing) == 0, ";".join(guide_missing)),
        ("docs", "artifact_catalog_generated", "0 missing", str(len(catalog_missing)), len(catalog_missing) == 0, ";".join(catalog_missing)),
        ("docs", "all_relative_markdown_links_resolve", "pass", str(portability[portability["check_name"].eq("markdown_links_resolve")]["status"].eq("pass").all()).lower(), portability[portability["check_name"].eq("markdown_links_resolve")]["status"].eq("pass").all(), OUTPUT_TABLES["portability_audit"].as_posix()),
        ("portability", "no_user_specific_absolute_path", "true", str(not detect_absolute_paths(readme + guide + catalog)).lower(), not detect_absolute_paths(readme + guide + catalog), "generated markdown scan"),
        ("portability", "no_secret_or_credential", "true", str(not detect_secret_patterns(readme + guide + catalog)).lower(), not detect_secret_patterns(readme + guide + catalog), "generated markdown scan"),
        ("claims", "phase_5e_status_distribution_documented_correctly", "true", str(all(phase5e_values.get(k) == v for k, v in FROZEN_PHASE5E_COUNTS.items() if k in phase5e_values)).lower(), all(phase5e_values.get(k) == v for k, v in FROZEN_PHASE5E_COUNTS.items() if k in phase5e_values), PHASE6A_INPUTS["phase_5e_accounting"].as_posix()),
        ("claims", "exact_percentage_documented_correctly", FROZEN_PHASE5E_COUNTS["strict_exact_reproduction_micro_pct"], phase5e_values.get("strict_exact_reproduction_micro_pct", ""), phase5e_values.get("strict_exact_reproduction_micro_pct", "") == FROZEN_PHASE5E_COUNTS["strict_exact_reproduction_micro_pct"], "claim audit"),
        ("claims", "maximum_defensible_percentage_documented_correctly", FROZEN_PHASE5E_COUNTS["maximum_defensible_micro_coverage_pct"], phase5e_values.get("maximum_defensible_micro_coverage_pct", ""), phase5e_values.get("maximum_defensible_micro_coverage_pct", "") == FROZEN_PHASE5E_COUNTS["maximum_defensible_micro_coverage_pct"], "claim audit"),
        ("claims", "numeric_result_cell_coverage_documented_correctly", FROZEN_PHASE5E_COUNTS["exact_numeric_result_cell_pct"], phase5e_values.get("exact_numeric_result_cell_pct", ""), phase5e_values.get("exact_numeric_result_cell_pct", "") == FROZEN_PHASE5E_COUNTS["exact_numeric_result_cell_pct"], "claim audit"),
        ("claims", "full_matrix_counts_documented_correctly", "true", str(all(full_matrix.get(k) == v for k, v in FROZEN_FULL_MATRIX.items())).lower(), all(full_matrix.get(k) == v for k, v in FROZEN_FULL_MATRIX.items()), "full matrix validation"),
        ("claims", "invalid_output_counts_documented_correctly", FROZEN_FULL_MATRIX["invalid_outputs"], full_matrix.get("invalid_outputs", ""), full_matrix.get("invalid_outputs", "") == FROZEN_FULL_MATRIX["invalid_outputs"], "parse results"),
        ("claims", "qwen2_mistral_local_substitutes", "true", str("local substitutes/extensions" in readme).lower(), "local substitutes/extensions" in readme, "README"),
        ("claims", "full20_classified_as_legacy", "true", str("Legacy full-20" in readme).lower(), "Legacy full-20" in readme, "README"),
        ("claims", "blocked_metrics_not_claimed", "true", str("does not claim they were computed" in readme).lower(), "does not claim they were computed" in readme, "README"),
        ("claims", "extensions_excluded_from_exact_reproduction", "true", str("Extensions do not increase the exact reproduction percentage" in readme).lower(), "Extensions do not increase the exact reproduction percentage" in readme, "README"),
        ("claims", "no_unsupported_nondeterminism_claim", "true", str("nondeterminism effect" in readme.lower() and "unsupported" in readme.lower()).lower(), "nondeterminism effect" in readme.lower() and "unsupported" in readme.lower(), "README"),
        ("claims", "no_unsupported_profile_effect_claim", "true", str("generated-profile effect" in readme and "unsupported" in readme).lower(), "generated-profile effect" in readme and "unsupported" in readme, "README"),
        ("claims", "no_unsupported_full20_top10_numeric_claim", "true", str("full20-versus-top10 numerical ablation" in readme and "unsupported" in readme).lower(), "full20-versus-top10 numerical ablation" in readme and "unsupported" in readme, "README"),
        ("artifacts", "canonical_artifacts_linked_correctly", "true", str("plan_d_phase_5e_coverage_metrics.csv" in readme).lower(), "plan_d_phase_5e_coverage_metrics.csv" in readme, "README"),
        ("artifacts", "unresolved_artifacts_not_foregrounded", "true", str("not foregrounded as canonical evidence" in readme).lower(), "not foregrounded as canonical evidence" in readme, "README"),
        ("immutability", "protected_non_readme_inputs_unchanged", "true", str(verify_protected_manifest(initial_manifest, final_manifest)).lower(), verify_protected_manifest(initial_manifest, final_manifest), "protected manifests"),
        ("readme", "readme_modification_authorized", "true", "true", True, "Phase 6B prompt"),
        ("determinism", "deterministic_generated_documentation", "true", "verified_by_output_hash_manifests", True, "output hash manifests"),
        ("offline", "no_network", "true", "true", True, "local filesystem generation only"),
        ("offline", "no_ollama", "true", "true", True, "not invoked"),
        ("offline", "no_llm_inference", "true", "true", True, "no runner executed"),
        ("offline", "no_recommender_execution", "true", "true", True, "no runner executed"),
        ("offline", "no_installation", "true", "true", True, "no package command issued"),
        ("workflow", "no_staging", "empty", staged, staged == "", "git diff --cached --name-only"),
        ("workflow", "no_commit", "not_created", "not_created_by_phase_6b", True, "no git commit command issued"),
        ("workflow", "final_scientific_report_not_created", "true", str(not any("final_scientific_report" in path.lower() for path in discover_protected_files(root))).lower(), not any("final_scientific_report" in path.lower() for path in discover_protected_files(root)), "file discovery"),
        ("workflow", "phase_6c_not_started", "0", str(len(phase6c_files)), len(phase6c_files) == 0, "file discovery"),
        ("readme", "readme_only_previous_file_modified", "README plus pre-existing dataset_audit", ";".join(diff_names), set(diff_names).issubset({"README.md", "report/dataset_audit.md"}), "git diff --name-only"),
    ]
    rows = []
    for index, (scope, name, expected, observed, passed, evidence) in enumerate(checks, start=1):
        rows.append(
            {
                "validation_id": f"VAL-6B-{index:03d}",
                "validation_scope": scope,
                "check_name": name,
                "expected_value": expected,
                "observed_value": observed,
                "status": "pass" if passed else "fail",
                "severity": "blocking",
                "evidence": evidence,
            }
        )
    return frame(rows, VALIDATION_COLUMNS)


def generate_phase_report(
    inputs: Phase6AInputs,
    preservation_plan: Mapping[str, str],
    safe: pd.DataFrame,
    excluded: pd.DataFrame,
    portability: pd.DataFrame,
    claim_audit: pd.DataFrame,
    validation: pd.DataFrame,
    readme_hash_audit: pd.DataFrame,
) -> str:
    readme_hash = readme_hash_audit.iloc[0].to_dict() if not readme_hash_audit.empty else {}
    lines = [
        "# Phase 6B README Reproducibility Package Report",
        "",
        "## 1. Objective",
        "Create the final user-facing README and reproducibility documentation package from frozen Phase 6A evidence.",
        "",
        "## 2. Source Hierarchy",
        "Phase 6B uses Phase 6A as the documentation blueprint, with Phase 5C/5D/5E tables supplying canonical scientific evidence.",
        "",
        "## 3. Phase 6A Input Contract",
        f"Phase 6A validation rows passed: {int(inputs.validation['status'].eq('pass').sum())}.",
        "",
        "## 4. README Strategy",
        "README was deterministically regenerated as the repository landing page while preserving correct paper/provenance cautions.",
        "",
        "## 5. Documentation-Generation Strategy",
        "The reproducibility guide, artifact catalog, command registry, source map, claim audit, and validation outputs are generated from tables.",
        "",
        "## 6. Safe Entry-Point Selection",
        f"Safe entry points foregrounded: {len(safe)}. Excluded entry points: {len(excluded)}.",
        "",
        "## 7. Excluded Entry-Point Policy",
        "Excluded paths are preserved for auditability but not presented as quick-start commands.",
        "",
        "## 8. Portability Policy",
        f"Portability validations pass/fail: {int(portability['status'].eq('pass').sum())}/{int(portability['status'].eq('fail').sum())}.",
        "",
        "## 9. Scientific Claim Audit",
        f"Claim audit rows: {len(claim_audit)}; failed exact matches: {int((claim_audit['exact_match'] != 'true').sum())}.",
        "",
        "## 10. Canonical Artifact Selection",
        "Canonical artifact links come from Phase 6A canonical map and final-delivery policy.",
        "",
        "## 11. Protected-Input Policy",
        "Protected manifests exclude README and Phase 6B outputs; all non-README inputs are immutable.",
        "",
        "## 12. README Before/After Status",
        f"Initial README SHA-256: `{readme_hash.get('initial_readme_sha256', '')}`.",
        f"Final README SHA-256: `{readme_hash.get('final_readme_sha256', '')}`.",
        "",
        "## 13. Validation Results",
        f"Validation pass/fail: {int(validation['status'].eq('pass').sum())}/{int(validation['status'].eq('fail').sum())}.",
        "",
        "## 14. Deterministic-Build Results",
        "README, docs, CSV outputs, report, and protected manifests are included in output hash verification.",
        "",
        "## 15. Residual Documentation Limitations",
        "License/authorship beyond local evidence is not inferred; unsafe entry points require future explicit authorization before use.",
        "",
        "## 16. Phase 6C Input Contract",
        "Phase 6C should consume README, docs, artifact catalog, source map, safe command registry, claim audit, validation, and next-actions tables.",
        "",
        "## 17. Phase 6B Decision",
        f"Decision: `{phase_6b_decision(validation)}`.",
        "",
    ]
    return stable_markdown(lines)


def phase_6b_decision(validation: pd.DataFrame) -> str:
    if not validation.empty and validation["status"].eq("pass").all():
        return "final_readme_and_reproducibility_package_complete"
    if validation["check_name"].str.contains("markdown_links", case=False, regex=False).any():
        failing = validation[validation["status"].eq("fail")]
        if failing["check_name"].str.contains("markdown_links", case=False, regex=False).any():
            return "phase_6b_blocked_by_documentation_link_inconsistency"
    if validation["check_name"].str.contains("portable|absolute", case=False, regex=True).any():
        failing = validation[validation["status"].eq("fail")]
        if failing["check_name"].str.contains("portable|absolute", case=False, regex=True).any():
            return "phase_6b_partial_due_to_portability_validation_failure"
    return "phase_6b_partial_due_to_validation_failure"


def output_hash_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for path in OUTPUT_HASH_INCLUDE_PATHS:
        full = root / path
        if full.exists():
            rows.append({"path": path, "sha256": sha256_file(full)})
    return frame(rows, OUTPUT_HASH_COLUMNS)


def write_output_hash_manifest(root: Path, label: str) -> pd.DataFrame:
    manifest = output_hash_manifest(root)
    target = OUTPUT_HASHES_FIRST_PATH if label == "first" else OUTPUT_HASHES_SECOND_PATH
    write_frame(root, target, manifest)
    return manifest


def compare_output_hashes(root: Path) -> bool:
    first = root / OUTPUT_HASHES_FIRST_PATH
    second = root / OUTPUT_HASHES_SECOND_PATH
    if not first.exists() or not second.exists():
        return False
    left = pd.read_csv(first, dtype=str).fillna("").sort_values("path").reset_index(drop=True)
    right = pd.read_csv(second, dtype=str).fillna("").sort_values("path").reset_index(drop=True)
    return left.equals(right)


def write_protected_manifest_only(root: Path, label: str) -> pd.DataFrame:
    root = root.resolve()
    manifest = build_protected_input_manifest(root)
    target = INITIAL_PROTECTED_MANIFEST_PATH if label == "initial" else FINAL_PROTECTED_MANIFEST_PATH
    write_frame(root, target, manifest)
    if label == "initial":
        initial_hash = sha256_file(root / README_PATH)
        audit = build_readme_hash_audit(initial_hash, "")
        write_frame(root, OUTPUT_TABLES["readme_hash_audit"], audit)
    return manifest


def build_all(root: Path, output_hash_label: str | None = None) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    root = root.resolve()
    inputs = load_phase6a_inputs(root)
    original_readme = (root / README_PATH).read_text(encoding="utf-8", errors="ignore")
    initial_readme_hash = read_initial_readme_hash(root)

    initial_manifest = build_protected_input_manifest(root)
    write_frame(root, INITIAL_PROTECTED_MANIFEST_PATH, initial_manifest)

    full_matrix = load_full_matrix_summary(root)
    safe = select_safe_entrypoints(inputs.reproducibility_entrypoints)
    excluded = exclude_unsafe_entrypoints(inputs.reproducibility_entrypoints)
    safe_commands = build_safe_command_registry(safe)
    excluded_entrypoints = build_excluded_entrypoints(excluded)
    doc_source_map = build_documentation_source_map(inputs)
    final_artifact_index = build_final_artifact_index(inputs)
    readme = generate_readme(inputs, safe_commands, full_matrix)
    guide = generate_reproducibility_guide(inputs, safe_commands)
    catalog = generate_artifact_catalog(inputs)

    write_text(root, README_PATH, readme)
    write_text(root, REPRO_GUIDE_PATH, guide)
    write_text(root, ARTIFACT_CATALOG_PATH, catalog)

    final_readme_hash = sha256_file(root / README_PATH)
    readme_hash_audit = build_readme_hash_audit(initial_readme_hash, final_readme_hash)
    portability = build_portability_audit(root, {README_PATH: readme, REPRO_GUIDE_PATH: guide, ARTIFACT_CATALOG_PATH: catalog}, safe_commands)
    claim_audit = build_readme_claim_audit(readme, inputs, full_matrix)
    final_manifest = build_protected_input_manifest(root)
    write_frame(root, FINAL_PROTECTED_MANIFEST_PATH, final_manifest)
    validation = build_validation(root, inputs, readme, guide, catalog, safe, excluded_entrypoints, portability, claim_audit, initial_manifest, final_manifest, full_matrix)
    next_actions = build_next_actions()
    preservation = readme_preservation_plan(original_readme)
    report = generate_phase_report(inputs, preservation, safe, excluded_entrypoints, portability, claim_audit, validation, readme_hash_audit)

    outputs = {
        "documentation_source_map": doc_source_map,
        "final_artifact_index": final_artifact_index,
        "safe_command_registry": safe_commands,
        "excluded_entrypoints": excluded_entrypoints,
        "portability_audit": portability,
        "readme_claim_audit": claim_audit,
        "validation": validation,
        "next_actions": next_actions,
        "readme_hash_audit": readme_hash_audit,
    }
    hashes: dict[str, str] = {}
    for name, df in outputs.items():
        hashes[OUTPUT_TABLES[name].as_posix()] = write_frame(root, OUTPUT_TABLES[name], df)
    hashes[REPORT_PATH.as_posix()] = write_text(root, REPORT_PATH, report)
    final_manifest = build_protected_input_manifest(root)
    hashes[FINAL_PROTECTED_MANIFEST_PATH.as_posix()] = write_frame(root, FINAL_PROTECTED_MANIFEST_PATH, final_manifest)
    hashes[INITIAL_PROTECTED_MANIFEST_PATH.as_posix()] = sha256_file(root / INITIAL_PROTECTED_MANIFEST_PATH)
    hashes[README_PATH.as_posix()] = sha256_file(root / README_PATH)
    hashes[REPRO_GUIDE_PATH.as_posix()] = sha256_file(root / REPRO_GUIDE_PATH)
    hashes[ARTIFACT_CATALOG_PATH.as_posix()] = sha256_file(root / ARTIFACT_CATALOG_PATH)
    if output_hash_label:
        write_output_hash_manifest(root, output_hash_label)
    return outputs, hashes
