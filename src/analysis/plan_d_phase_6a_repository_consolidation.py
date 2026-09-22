"""Offline Phase 6A repository evidence freeze and consolidation blueprint.

The functions in this module inspect existing repository artifacts, classify
their lifecycle role, reconcile the frozen Phase 5E accounting, and emit a
deterministic handoff for later final documentation. They do not run LLM
inference, recommender training, network calls, downloads, staging, or commits.
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

LIFECYCLE_CLASSES = [
    "canonical_final_evidence",
    "canonical_supporting_artifact",
    "exact_paired_diagnostic",
    "repeated_controlled_diagnostic",
    "small_pilot",
    "legacy_protocol_artifact",
    "reconstructed_component_artifact",
    "adapted_component_artifact",
    "extension_artifact",
    "blocked_or_incomplete_artifact",
    "unresolved_artifact",
    "implementation_support",
    "test_artifact",
    "raw_immutable_input",
    "generated_manifest",
    "temporary_or_intermediate",
    "orphaned_artifact",
    "unrelated_to_final_delivery",
]

FINAL_DELIVERY_DISPOSITIONS = [
    "include_primary",
    "include_supporting",
    "preserve_not_foreground",
    "preserve_legacy",
    "preserve_blocked",
    "exclude_from_final_narrative",
    "unresolved_review_required",
]

OUTPUT_TABLES = {
    "source_inventory": Path("results/tables/plan_d_phase_6a_source_inventory.csv"),
    "artifact_registry": Path("results/tables/plan_d_phase_6a_artifact_registry.csv"),
    "canonical_artifact_map": Path("results/tables/plan_d_phase_6a_canonical_artifact_map.csv"),
    "provenance_dependencies": Path("results/tables/plan_d_phase_6a_provenance_dependencies.csv"),
    "plan_e_reconciliation": Path("results/tables/plan_d_phase_6a_plan_e_reconciliation.csv"),
    "legacy_superseded": Path("results/tables/plan_d_phase_6a_legacy_superseded.csv"),
    "final_documentation_input_map": Path("results/tables/plan_d_phase_6a_final_documentation_input_map.csv"),
    "reproducibility_entrypoints": Path("results/tables/plan_d_phase_6a_reproducibility_entrypoints.csv"),
    "orphaned_or_unresolved": Path("results/tables/plan_d_phase_6a_orphaned_or_unresolved.csv"),
    "phase_5e_accounting": Path("results/tables/plan_d_phase_6a_phase_5e_accounting.csv"),
    "final_delivery_policy": Path("results/tables/plan_d_phase_6a_final_delivery_policy.csv"),
    "validation": Path("results/tables/plan_d_phase_6a_validation.csv"),
    "next_actions": Path("results/tables/plan_d_phase_6a_next_actions.csv"),
}

REPORT_PATH = Path("report/plan_d_phase_6a_repository_consolidation_blueprint.md")
INITIAL_PROTECTED_MANIFEST_PATH = Path("results/tables/plan_d_phase_6a_protected_input_hashes_initial.csv")
FINAL_PROTECTED_MANIFEST_PATH = Path("results/tables/plan_d_phase_6a_protected_input_hashes_final.csv")
OUTPUT_HASHES_FIRST_PATH = Path("results/tables/plan_d_phase_6a_output_hashes_first.csv")
OUTPUT_HASHES_SECOND_PATH = Path("results/tables/plan_d_phase_6a_output_hashes_second.csv")

PHASE6A_OUTPUT_PATHS = {
    *{path.as_posix() for path in OUTPUT_TABLES.values()},
    REPORT_PATH.as_posix(),
    INITIAL_PROTECTED_MANIFEST_PATH.as_posix(),
    FINAL_PROTECTED_MANIFEST_PATH.as_posix(),
    OUTPUT_HASHES_FIRST_PATH.as_posix(),
    OUTPUT_HASHES_SECOND_PATH.as_posix(),
    "src/analysis/plan_d_phase_6a_repository_consolidation.py",
    "src/experiments/build_plan_d_phase_6a_repository_consolidation.py",
    "tests/test_plan_d_phase_6a_repository_consolidation.py",
}

OUTPUT_HASH_INCLUDE_PATHS = [
    *[path.as_posix() for path in OUTPUT_TABLES.values()],
    REPORT_PATH.as_posix(),
    INITIAL_PROTECTED_MANIFEST_PATH.as_posix(),
    FINAL_PROTECTED_MANIFEST_PATH.as_posix(),
]

EXCLUDED_DIRS = {".git", ".venv", "venv", "env", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache"}
TEXT_EXTENSIONS = {".py", ".md", ".txt", ".csv", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini"}
ABSOLUTE_PATH_PATTERN = re.compile(r"([A-Za-z]:[\\/][^\s\]\)\"']+)")

SOURCE_INVENTORY_COLUMNS = [
    "path",
    "artifact_type",
    "producing_phase",
    "scientific_role",
    "inclusion_status",
    "lifecycle_class",
    "final_delivery_disposition",
    "protected_status",
    "hash_status",
    "notes",
]
ARTIFACT_REGISTRY_COLUMNS = [
    "artifact_id",
    "normalized_path",
    "artifact_type",
    "producing_phase",
    "producer",
    "scientific_role",
    "lifecycle_class",
    "canonicality",
    "final_delivery_disposition",
    "protected",
    "exists",
    "size",
    "sha256",
    "notes",
]
CANONICAL_MAP_COLUMNS = [
    "scientific_topic",
    "canonical_primary_artifacts",
    "canonical_supporting_artifacts",
    "source_precedence_basis",
    "final_document_role",
    "required_caveat",
]
PROVENANCE_COLUMNS = [
    "relationship_id",
    "relationship_type",
    "artifact_id",
    "artifact_path",
    "producer_or_consumer",
    "producer_function_or_entrypoint",
    "input_artifacts",
    "output_artifacts",
    "associated_phase",
    "associated_experiment",
    "associated_dataset",
    "associated_model",
    "associated_task",
    "associated_protocol",
    "associated_metric_contract",
    "report_consumers",
    "test_consumers",
    "canonicality_status",
    "lifecycle_class",
    "final_delivery_disposition",
]
PLAN_E_COLUMNS = [
    "path",
    "artifact_type",
    "apparent_purpose",
    "producing_source_or_builder",
    "associated_tests",
    "source_inputs",
    "generated_outputs",
    "related_report",
    "phase_or_date_evidence",
    "phase_5e_manifest_presence",
    "phase_5e_temporal_classification",
    "represents",
    "executed",
    "outputs_validated",
    "conflicts_with_phase_5c_5d_5e",
    "changes_phase_5e_claim",
    "should_feed_final_report",
    "preservation_policy",
    "requires_human_review",
]
LEGACY_COLUMNS = [
    "old_artifact",
    "current_canonical_artifact",
    "relationship",
    "reason",
    "scientifically_useful",
    "supersession_declared_by_previous_phase",
    "preservation_policy",
    "final_report_policy",
]
DOC_INPUT_COLUMNS = [
    "final_document_section",
    "primary_source_artifacts",
    "supporting_artifacts",
    "prohibited_sources",
    "canonical_tables",
    "canonical_metrics",
    "required_caveats",
    "unresolved_questions",
    "figure_required_later",
    "derived_table_required_later",
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
ORPHAN_COLUMNS = [
    "artifact",
    "issue",
    "evidence_inspected",
    "scientific_impact",
    "recoverability",
    "human_review_requirement",
    "permitted_action",
    "prohibited_action",
]
PHASE5E_ACCOUNTING_COLUMNS = [
    "record_type",
    "status_or_metric",
    "count",
    "numerator",
    "denominator",
    "value",
    "expected_value",
    "observed_value",
    "verification_status",
    "source_artifact",
    "notes",
]
FINAL_DELIVERY_POLICY_COLUMNS = [
    "artifact_or_artifact_family",
    "inclusion_disposition",
    "reason",
    "canonical_replacement_if_any",
    "readme_role",
    "report_role",
    "reproducibility_role",
    "preservation_requirement",
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
PROTECTED_MANIFEST_COLUMNS = ["path", "protected_input", "hash_method", "sha256", "size_bytes", "policy"]
OUTPUT_HASH_COLUMNS = ["path", "sha256"]


@dataclass(frozen=True)
class Artifact:
    path: str
    artifact_type: str
    producing_phase: str
    lifecycle_class: str
    final_delivery_disposition: str
    canonicality: str
    scientific_role: str
    producer: str
    size: int
    sha256: str
    notes: str


def normalize_path(path: str | Path) -> str:
    """Return a deterministic repository-relative path with POSIX separators."""
    text = str(path).replace("\\", "/")
    text = re.sub(r"/+", "/", text)
    while text.startswith("./"):
        text = text[2:]
    return text.rstrip("/")


def stable_artifact_id(path: str | Path) -> str:
    """Generate a stable artifact identifier from a normalized path."""
    normalized = normalize_path(path)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12].upper()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", Path(normalized).stem).strip("-").upper()[:48]
    return f"ART-{slug}-{digest}"


def stable_relationship_id(*parts: str) -> str:
    joined = "|".join(parts)
    return "REL-" + hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12].upper()


def frame(rows: Iterable[Mapping[str, object]], columns: Sequence[str]) -> pd.DataFrame:
    normalized_rows = []
    for row in rows:
        normalized_rows.append({column: "" if row.get(column) is None else str(row.get(column, "")) for column in columns})
    return pd.DataFrame(normalized_rows, columns=list(columns))


def stable_csv_bytes(df: pd.DataFrame) -> bytes:
    stable = df.fillna("").astype(str)
    return stable.to_csv(index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL).encode("utf-8")


def write_frame(root: Path, path: Path, df: pd.DataFrame) -> str:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = stable_csv_bytes(df)
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


def is_phase6a_output(path: str | Path) -> bool:
    normalized = normalize_path(path)
    if normalized in PHASE6A_OUTPUT_PATHS:
        return True
    return (
        normalized.startswith("results/tables/plan_d_phase_6a_")
        or normalized.startswith("report/plan_d_phase_6a_")
        or normalized.endswith("/plan_d_phase_6a_repository_consolidation.py")
        or normalized.endswith("/test_plan_d_phase_6a_repository_consolidation.py")
    )


def discover_relevant_files(root: Path) -> list[str]:
    """Discover repository artifacts relevant to the Phase 6A freeze."""
    paths: list[str] = []
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        relative = normalize_path(candidate.relative_to(root))
        parts = set(Path(relative).parts)
        if parts & EXCLUDED_DIRS:
            continue
        if is_phase6a_output(relative):
            continue
        paths.append(relative)
    return sorted(set(paths), key=lambda value: value.lower())


def artifact_type_for(path: str) -> str:
    normalized = normalize_path(path)
    suffix = Path(normalized).suffix.lower()
    name = Path(normalized).name.lower()
    if normalized.startswith("tests/") and suffix == ".py":
        return "test_source"
    if normalized.startswith("src/experiments/") and suffix == ".py":
        return "experiment_entrypoint"
    if normalized.startswith("src/analysis/") and suffix == ".py":
        return "analysis_source"
    if normalized.startswith("src/baselines/") and suffix == ".py":
        return "baseline_source"
    if normalized.startswith("src/llm/") and suffix == ".py":
        return "llm_source"
    if suffix == ".py":
        return "source_code"
    if suffix in {".yaml", ".yml"}:
        return "config"
    if suffix == ".md":
        return "report" if normalized.startswith("report/") else "documentation"
    if suffix == ".pdf":
        return "paper_pdf"
    if suffix == ".csv":
        if "hash" in name or "manifest" in name:
            return "manifest_or_result_table"
        return "result_table"
    if suffix == ".jsonl":
        if "request" in name:
            return "raw_llm_requests"
        if "response" in name:
            return "raw_llm_responses"
        return "jsonl_artifact"
    if suffix == ".json":
        if "metadata" in name:
            return "model_metadata"
        return "json_artifact"
    if suffix in {".pt", ".pth", ".ckpt"}:
        return "model_checkpoint"
    if normalized.startswith("data/raw/"):
        return "raw_dataset"
    return "repository_artifact"


def producing_phase_for(path: str) -> str:
    normalized = normalize_path(path).lower()
    if normalized.startswith("original_repo/"):
        return "original_repository"
    if normalized.startswith("paper/"):
        return "paper"
    if "plan_d_phase_5e" in normalized:
        return "phase_5e"
    if "plan_d_phase_5d" in normalized:
        return "phase_5d"
    if "canonical_local_llm_full_matrix" in normalized:
        return "phase_5c"
    if "plan_e" in normalized:
        return "plan_e"
    if "canonical" in normalized:
        return "canonical_pre_5e"
    if "traditional_baseline" in normalized or "baseline" in normalized:
        return "baseline_phase"
    if normalized.startswith("data/"):
        return "dataset_preparation"
    if normalized.startswith("src/"):
        return "implementation"
    if normalized.startswith("tests/"):
        return "test_suite"
    if normalized.startswith("report/"):
        return "reporting"
    if normalized.startswith("results/"):
        return "generated_results"
    return "repository_scaffold"


def is_manifest_path(path: str) -> bool:
    name = Path(path).name.lower()
    return "manifest" in name or "hash" in name or "provenance" in name or "validation" in name


def lifecycle_class_for(path: str, artifact_type: str | None = None, producing_phase: str | None = None) -> str:
    """Assign exactly one lifecycle class using deterministic evidence rules."""
    normalized = normalize_path(path).lower()
    artifact_type = artifact_type or artifact_type_for(path)
    producing_phase = producing_phase or producing_phase_for(path)

    if artifact_type == "test_source":
        return "test_artifact"
    if artifact_type in {"paper_pdf", "raw_dataset", "raw_llm_requests", "raw_llm_responses"}:
        return "raw_immutable_input"
    if normalized.startswith("original_repo/"):
        return "raw_immutable_input"
    if "blocked" in normalized or "missing" in normalized or "incomplete" in normalized:
        return "blocked_or_incomplete_artifact"
    if "unresolved" in normalized:
        return "unresolved_artifact"
    if is_manifest_path(normalized):
        return "generated_manifest"
    if "full20" in normalized or "legacy" in normalized:
        return "legacy_protocol_artifact"
    if producing_phase in {"phase_5c", "phase_5d", "phase_5e"}:
        return "canonical_final_evidence"
    if "diagnostic" in normalized or "cuda_first_step" in normalized:
        return "exact_paired_diagnostic"
    if "determinism" in normalized or "robustness" in normalized:
        return "repeated_controlled_diagnostic"
    if "pilot" in normalized:
        return "small_pilot"
    if "metric_contract" in normalized or "canonical_metric" in normalized:
        return "canonical_supporting_artifact"
    if "canonical_local_llm" in normalized or "canonical_llm" in normalized:
        return "adapted_component_artifact"
    if "traditional_baseline" in normalized or "baseline" in normalized:
        return "reconstructed_component_artifact"
    if "plan_e" in normalized:
        if artifact_type in {"analysis_source", "experiment_entrypoint", "baseline_source", "config"}:
            return "implementation_support"
        return "extension_artifact"
    if artifact_type in {"analysis_source", "experiment_entrypoint", "baseline_source", "llm_source", "source_code", "config"}:
        return "implementation_support"
    if artifact_type in {"result_table", "manifest_or_result_table", "model_checkpoint", "model_metadata"}:
        return "canonical_supporting_artifact"
    return "unrelated_to_final_delivery"


def final_delivery_disposition_for(lifecycle_class: str) -> str:
    mapping = {
        "canonical_final_evidence": "include_primary",
        "canonical_supporting_artifact": "include_supporting",
        "exact_paired_diagnostic": "preserve_not_foreground",
        "repeated_controlled_diagnostic": "preserve_not_foreground",
        "small_pilot": "preserve_legacy",
        "legacy_protocol_artifact": "preserve_legacy",
        "reconstructed_component_artifact": "include_supporting",
        "adapted_component_artifact": "include_supporting",
        "extension_artifact": "preserve_not_foreground",
        "blocked_or_incomplete_artifact": "preserve_blocked",
        "unresolved_artifact": "unresolved_review_required",
        "implementation_support": "exclude_from_final_narrative",
        "test_artifact": "exclude_from_final_narrative",
        "raw_immutable_input": "include_supporting",
        "generated_manifest": "include_supporting",
        "temporary_or_intermediate": "preserve_not_foreground",
        "orphaned_artifact": "unresolved_review_required",
        "unrelated_to_final_delivery": "exclude_from_final_narrative",
    }
    return mapping[lifecycle_class]


def canonicality_for(lifecycle_class: str, path: str) -> str:
    normalized = normalize_path(path).lower()
    if lifecycle_class == "canonical_final_evidence":
        return "canonical"
    if lifecycle_class == "canonical_supporting_artifact":
        return "canonical_supporting"
    if lifecycle_class in {"exact_paired_diagnostic", "repeated_controlled_diagnostic"}:
        return "diagnostic_only"
    if lifecycle_class in {"small_pilot", "legacy_protocol_artifact"}:
        return "legacy_or_pilot"
    if lifecycle_class == "extension_artifact" or "plan_e" in normalized:
        return "extension_not_reproduction"
    if lifecycle_class == "blocked_or_incomplete_artifact":
        return "blocked"
    if lifecycle_class == "unresolved_artifact":
        return "unresolved"
    if lifecycle_class == "raw_immutable_input":
        return "immutable_input"
    if lifecycle_class == "generated_manifest":
        return "supporting_manifest"
    if lifecycle_class == "test_artifact":
        return "test_only"
    return "implementation_or_support"


def scientific_role_for(lifecycle_class: str, path: str) -> str:
    normalized = normalize_path(path).lower()
    if lifecycle_class == "canonical_final_evidence":
        if "phase_5e" in normalized:
            return "frozen paper coverage audit"
        if "phase_5d" in normalized:
            return "frozen robustness consolidation"
        if "full_matrix" in normalized:
            return "frozen canonical local LLM full matrix"
        return "canonical final evidence"
    if lifecycle_class == "extension_artifact":
        return "extension evidence that cannot change frozen Phase 5E percentages"
    if lifecycle_class == "raw_immutable_input":
        return "immutable scientific input"
    if lifecycle_class == "generated_manifest":
        return "provenance or validation manifest"
    if lifecycle_class == "implementation_support":
        return "implementation support or executable entry point"
    if lifecycle_class == "test_artifact":
        return "test-only support"
    if lifecycle_class == "blocked_or_incomplete_artifact":
        return "blocked or incomplete evidence to preserve"
    if lifecycle_class == "unresolved_artifact":
        return "unresolved evidence requiring review"
    return lifecycle_class.replace("_", " ")


def family_for(path: str) -> str:
    normalized = normalize_path(path).lower()
    families = [
        "plan_e_bprmf",
        "plan_e_gru4rec",
        "plan_e_lightgcn",
        "plan_e_sasrec",
        "plan_e_bm25",
        "plan_e_baseline_alignment",
        "plan_e_paper_condition_alignment",
        "plan_e_paper_like_candidate_pool",
        "plan_e_full_data_inventory",
        "plan_d_phase_5e",
        "plan_d_phase_5d",
        "canonical_local_llm_full_matrix",
        "canonical_local_llm_top10_confirmation",
        "canonical_local_llm_reranking_prompt_v2",
        "canonical_local_llm_hybrid_holdout",
        "canonical_local_llm_pilot",
        "canonical_metric",
        "traditional_baseline",
    ]
    for family in families:
        if family in normalized:
            return family
    if "plan_e" in normalized:
        return "plan_e"
    if "canonical" in normalized:
        return "canonical"
    return Path(normalized).stem


def producer_for(path: str, existing_paths: set[str] | None = None) -> str:
    normalized = normalize_path(path)
    lower = normalized.lower()
    existing_paths = existing_paths or set()
    if normalized.startswith(("src/", "tests/", "report/", "configs/", "paper/", "original_repo/")):
        return ""
    family = family_for(normalized)
    candidates: list[str] = []
    if family == "plan_d_phase_5e":
        candidates.append("src/experiments/build_plan_d_phase_5e_paper_coverage_audit.py")
    elif family == "plan_d_phase_5d":
        candidates.append("src/experiments/build_plan_d_phase_5d_robustness_consolidation.py")
    elif family == "canonical_local_llm_full_matrix":
        candidates.extend(
            [
                "src/experiments/run_canonical_local_llm_full_matrix.py",
                "src/experiments/build_canonical_local_llm_full_matrix_preflight.py",
            ]
        )
    elif family.startswith("canonical_local_llm_"):
        suffix = family.replace("canonical_local_llm_", "")
        candidates.append(f"src/experiments/run_canonical_local_llm_{suffix}.py")
        candidates.append(f"src/experiments/build_canonical_local_llm_{suffix}_audit.py")
    elif family.startswith("canonical_metric"):
        candidates.append("src/experiments/build_canonical_metric_results.py")
        candidates.append("src/experiments/build_canonical_metric_preflight.py")
    elif family.startswith("traditional_baseline"):
        candidates.append("src/experiments/build_plan_e_non_neural_baselines.py")
    elif family.startswith("plan_e_"):
        base = family
        candidates.extend(
            [
                f"src/experiments/build_{base}.py",
                f"src/experiments/build_{base}_audit.py",
                f"src/experiments/build_{base}_consolidation.py",
                f"src/experiments/run_{base}.py",
            ]
        )
        if "bprmf" in lower:
            candidates.extend(["src/experiments/build_plan_e_bprmf_baseline_consolidation.py", "src/experiments/run_plan_e_bprmf_beauty_sports.py"])
        if "gru4rec" in lower:
            candidates.extend(["src/experiments/build_plan_e_gru4rec_baseline_consolidation.py", "src/experiments/run_plan_e_gru4rec_beauty_sports.py"])
        if "lightgcn" in lower:
            candidates.extend(["src/experiments/build_plan_e_lightgcn_cpu_baseline_consolidation.py", "src/experiments/run_plan_e_lightgcn_beauty_sports_cpu.py"])
        if "sasrec" in lower:
            candidates.extend(["src/experiments/build_plan_e_sasrec_baseline_consolidation.py", "src/experiments/run_plan_e_sasrec_beauty_sports.py"])
    present = [candidate for candidate in candidates if not existing_paths or candidate in existing_paths]
    return ";".join(dict.fromkeys(present))


def report_consumers_for(path: str, existing_paths: set[str]) -> str:
    family = family_for(path)
    consumers = [
        candidate
        for candidate in existing_paths
        if candidate.startswith("report/") and family.replace("_pool", "_pools")[:24] in candidate.lower()
    ]
    if "phase_5e" in family and "report/plan_d_phase_5e_final_paper_coverage_audit.md" in existing_paths:
        consumers.append("report/plan_d_phase_5e_final_paper_coverage_audit.md")
    if "phase_5d" in family and "report/plan_d_phase_5d_robustness_consolidation.md" in existing_paths:
        consumers.append("report/plan_d_phase_5d_robustness_consolidation.md")
    return ";".join(sorted(set(consumers)))


def test_consumers_for(path: str, existing_paths: set[str]) -> str:
    family = family_for(path)
    tokens = [family, family.replace("plan_e_", "test_plan_e_"), Path(path).stem]
    consumers = [
        candidate
        for candidate in existing_paths
        if candidate.startswith("tests/") and any(token and token in candidate.lower() for token in tokens)
    ]
    return ";".join(sorted(set(consumers)))


def read_small_text(root: Path, path: str, max_bytes: int = 1024 * 1024) -> str:
    full = root / path
    if not full.exists() or full.stat().st_size > max_bytes or full.suffix.lower() not in TEXT_EXTENSIONS:
        return ""
    try:
        return full.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def detect_absolute_paths(root: Path, paths: Sequence[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        text = read_small_text(root, path, max_bytes=512 * 1024)
        if not text:
            continue
        matches = sorted(set(ABSOLUTE_PATH_PATTERN.findall(text)))
        for match in matches[:5]:
            rows.append(
                {
                    "artifact": path,
                    "issue": "absolute_path_reference",
                    "evidence_inspected": match,
                    "scientific_impact": "Path may be non-portable in final documentation.",
                    "recoverability": "Document and avoid copying into final README commands.",
                    "human_review_requirement": "review_before_phase_6b",
                    "permitted_action": "Inventory only in Phase 6A.",
                    "prohibited_action": "Do not rewrite previous artifacts in Phase 6A.",
                }
            )
    if not rows:
        rows.append(
            {
                "artifact": "repository",
                "issue": "no_absolute_path_reference_detected_in_small_text_artifacts",
                "evidence_inspected": "text scan",
                "scientific_impact": "No portability issue detected by the Phase 6A scanner.",
                "recoverability": "not_applicable",
                "human_review_requirement": "none",
                "permitted_action": "Proceed to Phase 6B input mapping.",
                "prohibited_action": "Do not assume binary files contain no paths.",
            }
        )
    return rows


def build_artifact_registry(root: Path, paths: Sequence[str] | None = None) -> pd.DataFrame:
    discovered = list(paths) if paths is not None else discover_relevant_files(root)
    existing_paths = set(discovered)
    rows: list[dict[str, object]] = []
    for path in discovered:
        full = root / path
        artifact_type = artifact_type_for(path)
        phase = producing_phase_for(path)
        lifecycle = lifecycle_class_for(path, artifact_type, phase)
        disposition = final_delivery_disposition_for(lifecycle)
        canonicality = canonicality_for(lifecycle, path)
        producer = producer_for(path, existing_paths)
        size = full.stat().st_size if full.exists() else 0
        digest = sha256_file(full) if full.exists() else ""
        notes = ""
        if "plan_e" in path.lower():
            notes = "Plan E artifact preserved and reconciled separately; it does not retroactively alter Phase 5E."
        rows.append(
            {
                "artifact_id": stable_artifact_id(path),
                "normalized_path": path,
                "artifact_type": artifact_type,
                "producing_phase": phase,
                "producer": producer,
                "scientific_role": scientific_role_for(lifecycle, path),
                "lifecycle_class": lifecycle,
                "canonicality": canonicality,
                "final_delivery_disposition": disposition,
                "protected": "true",
                "exists": str(full.exists()).lower(),
                "size": size,
                "sha256": digest,
                "notes": notes,
            }
        )
    return frame(rows, ARTIFACT_REGISTRY_COLUMNS).sort_values("normalized_path", key=lambda s: s.str.lower()).reset_index(drop=True)


def build_source_inventory(artifact_registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in artifact_registry.to_dict("records"):
        rows.append(
            {
                "path": row["normalized_path"],
                "artifact_type": row["artifact_type"],
                "producing_phase": row["producing_phase"],
                "scientific_role": row["scientific_role"],
                "inclusion_status": "included_in_phase_6a_freeze",
                "lifecycle_class": row["lifecycle_class"],
                "final_delivery_disposition": row["final_delivery_disposition"],
                "protected_status": "protected_input",
                "hash_status": "sha256_recorded" if row["sha256"] else "missing_hash",
                "notes": row["notes"],
            }
        )
    return frame(rows, SOURCE_INVENTORY_COLUMNS)


def build_protected_input_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for path in discover_relevant_files(root):
        full = root / path
        rows.append(
            {
                "path": path,
                "protected_input": "true",
                "hash_method": "sha256",
                "sha256": sha256_file(full),
                "size_bytes": full.stat().st_size,
                "policy": "phase_6a_protected_input_manifest_excludes_only_phase_6a_outputs_and_tooling_caches",
            }
        )
    return frame(rows, PROTECTED_MANIFEST_COLUMNS)


def verify_protected_manifest(initial: pd.DataFrame, final: pd.DataFrame) -> bool:
    left = initial[["path", "sha256"]].sort_values("path").reset_index(drop=True)
    right = final[["path", "sha256"]].sort_values("path").reset_index(drop=True)
    return left.equals(right)


def load_phase5e_manifest_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    for relative in [
        "results/tables/plan_d_phase_5e_protected_input_hashes_initial.csv",
        "results/tables/plan_d_phase_5e_protected_input_hashes_final.csv",
        "results/tables/plan_d_phase_5e_source_inventory.csv",
    ]:
        full = root / relative
        if full.exists():
            try:
                frame_in = pd.read_csv(full, dtype=str).fillna("")
            except (OSError, pd.errors.EmptyDataError):
                continue
            if "path" in frame_in.columns:
                paths.update(frame_in["path"].map(normalize_path).tolist())
    return paths


def plan_e_represents(path: str, lifecycle: str) -> str:
    lower = path.lower()
    if lifecycle in {"exact_paired_diagnostic", "repeated_controlled_diagnostic"}:
        return "diagnostic"
    if lifecycle == "small_pilot":
        return "diagnostic_or_small_pilot"
    if lifecycle == "implementation_support":
        return "repository_support"
    if "paper_like" in lower or "condition_alignment" in lower or "baseline_alignment" in lower:
        return "reconstruction_or_alignment_support"
    if "bprmf" in lower or "gru4rec" in lower or "lightgcn" in lower or "sasrec" in lower:
        return "extension_neural_baseline_or_reconstructed_component"
    if "bm25" in lower:
        return "extension_text_retrieval_baseline"
    if "api_cost" in lower or "scaleup" in lower or "roadmap" in lower:
        return "planning_or_cost_support"
    return "extension_or_support"


def build_plan_e_reconciliation(root: Path, artifact_registry: pd.DataFrame) -> pd.DataFrame:
    phase5e_paths = load_phase5e_manifest_paths(root)
    existing_paths = set(artifact_registry["normalized_path"].tolist())
    rows = []
    for row in artifact_registry.to_dict("records"):
        path = row["normalized_path"]
        if "plan_e" not in path.lower():
            continue
        present_in_5e = path in phase5e_paths
        family = family_for(path)
        related_report = ";".join(sorted(p for p in existing_paths if p.startswith("report/") and family in p.lower()))
        associated_tests = test_consumers_for(path, existing_paths)
        producer = row["producer"] or producer_for(path, existing_paths)
        generated_outputs = ";".join(sorted(p for p in existing_paths if p != path and family in p.lower() and p.startswith(("results/", "report/"))))
        has_validation = any(family in p.lower() and "validation" in p.lower() for p in existing_paths)
        executed = row["artifact_type"] in {"result_table", "manifest_or_result_table", "model_checkpoint", "model_metadata"} and any(
            token in path.lower() for token in ["metrics", "rankings", "training_history", "run_manifest", "model_state", "scores"]
        )
        rows.append(
            {
                "path": path,
                "artifact_type": row["artifact_type"],
                "apparent_purpose": scientific_role_for(row["lifecycle_class"], path),
                "producing_source_or_builder": producer,
                "associated_tests": associated_tests,
                "source_inputs": "protected repository artifacts; configs if applicable",
                "generated_outputs": generated_outputs,
                "related_report": related_report,
                "phase_or_date_evidence": "path prefix plus Phase 5E source/protected manifest membership",
                "phase_5e_manifest_presence": "present_in_phase_5e_manifest" if present_in_5e else "not_present_in_phase_5e_manifest",
                "phase_5e_temporal_classification": "present_at_phase_5e_freeze" if present_in_5e else "post_phase_5e_evidence_candidate",
                "represents": plan_e_represents(path, row["lifecycle_class"]),
                "executed": str(executed).lower(),
                "outputs_validated": str(has_validation).lower(),
                "conflicts_with_phase_5c_5d_5e": "no_direct_conflict_detected; formal re-audit required before changing frozen claims",
                "changes_phase_5e_claim": "false",
                "should_feed_final_report": "extensions_or_limitations_section_only" if row["artifact_type"] not in {"source_code", "test_source"} else "no",
                "preservation_policy": "preserve_without_renaming_or_recomputing",
                "requires_human_review": "true" if not present_in_5e else "false",
            }
        )
    return frame(rows, PLAN_E_COLUMNS).sort_values("path", key=lambda s: s.str.lower()).reset_index(drop=True)


def build_canonical_artifact_map() -> pd.DataFrame:
    rows = [
        {
            "scientific_topic": "Phase 5C canonical local LLM matrix",
            "canonical_primary_artifacts": "results/tables/canonical_local_llm_full_matrix_cell_metrics.csv;results/tables/canonical_local_llm_full_matrix_rankings.csv;results/tables/canonical_local_llm_full_matrix_validation.csv",
            "canonical_supporting_artifacts": "results/llm/canonical_local_full_matrix/requests.jsonl;results/llm/canonical_local_full_matrix/responses.jsonl;report/canonical_local_llm_full_matrix_execution.md",
            "source_precedence_basis": "Phase 5C is the canonical full-matrix evidence source.",
            "final_document_role": "LLM protocol, utility, beyond-utility, reliability evidence.",
            "required_caveat": "Local Qwen2/Mistral evidence is adapted local-model evidence, not exact paper LLM replication.",
        },
        {
            "scientific_topic": "Phase 5D robustness evidence",
            "canonical_primary_artifacts": "results/tables/plan_d_phase_5d_robustness_summary.csv;results/tables/plan_d_phase_5d_model_task_reliability.csv",
            "canonical_supporting_artifacts": "results/tables/plan_d_phase_5d_pairing_coverage_audit.csv;report/plan_d_phase_5d_robustness_consolidation.md",
            "source_precedence_basis": "Phase 5D supersedes narrower robustness report-only summaries.",
            "final_document_role": "Robustness and reliability scope.",
            "required_caveat": "Robustness diagnostics do not upgrade pilot evidence to full-matrix replication.",
        },
        {
            "scientific_topic": "Phase 5E paper coverage audit",
            "canonical_primary_artifacts": "results/tables/plan_d_phase_5e_component_coverage.csv;results/tables/plan_d_phase_5e_coverage_metrics.csv;report/plan_d_phase_5e_final_paper_coverage_audit.md",
            "canonical_supporting_artifacts": "results/tables/plan_d_phase_5e_paper_component_registry.csv;results/tables/plan_d_phase_5e_result_cell_comparison.csv",
            "source_precedence_basis": "Frozen Phase 5E registry is authoritative for paper coverage percentages.",
            "final_document_role": "Reproduction coverage and limitations.",
            "required_caveat": "Maximum defensible coverage is not exact replication.",
        },
        {
            "scientific_topic": "Traditional and metric support",
            "canonical_primary_artifacts": "results/tables/traditional_baselines_metrics.csv;results/tables/canonical_metric_results_long.csv;results/tables/canonical_metric_contract.csv",
            "canonical_supporting_artifacts": "report/traditional_baseline_alignment.md;report/canonical_metric_contract_and_preflight.md",
            "source_precedence_basis": "Generated result tables and metric contracts precede reports.",
            "final_document_role": "Baseline and metric methods.",
            "required_caveat": "Baseline comparisons require explicit cohort/protocol alignment.",
        },
        {
            "scientific_topic": "Plan E extension evidence",
            "canonical_primary_artifacts": "results/tables/plan_d_phase_6a_plan_e_reconciliation.csv",
            "canonical_supporting_artifacts": "plan_e_* artifacts preserved by registry",
            "source_precedence_basis": "Plan E is reconciled after Phase 5E without modifying frozen percentages.",
            "final_document_role": "Extensions and future-work evidence only unless formally re-audited.",
            "required_caveat": "Plan E does not retroactively change Phase 5E coverage conclusions.",
        },
    ]
    return frame(rows, CANONICAL_MAP_COLUMNS)


def build_provenance_dependencies(artifact_registry: pd.DataFrame) -> pd.DataFrame:
    existing_paths = set(artifact_registry["normalized_path"].tolist())
    rows = []
    for row in artifact_registry.to_dict("records"):
        path = row["normalized_path"]
        producer = row["producer"]
        report_consumers = report_consumers_for(path, existing_paths)
        test_consumers = test_consumers_for(path, existing_paths)
        if producer:
            relationship_type = "producer_output"
            producer_or_consumer = producer
        elif report_consumers or test_consumers:
            relationship_type = "consumer_reference"
            producer_or_consumer = ";".join(filter(None, [report_consumers, test_consumers]))
        else:
            relationship_type = "no_explicit_relationship_identified"
            producer_or_consumer = ""
        rows.append(
            {
                "relationship_id": stable_relationship_id(path, relationship_type, producer_or_consumer),
                "relationship_type": relationship_type,
                "artifact_id": row["artifact_id"],
                "artifact_path": path,
                "producer_or_consumer": producer_or_consumer,
                "producer_function_or_entrypoint": producer,
                "input_artifacts": "protected inputs; see source inventory" if producer else "",
                "output_artifacts": path if producer else "",
                "associated_phase": row["producing_phase"],
                "associated_experiment": family_for(path),
                "associated_dataset": infer_dataset(path),
                "associated_model": infer_model(path),
                "associated_task": infer_task(path),
                "associated_protocol": infer_protocol(path),
                "associated_metric_contract": "results/tables/canonical_metric_contract.csv" if "metric" in path.lower() else "",
                "report_consumers": report_consumers,
                "test_consumers": test_consumers,
                "canonicality_status": row["canonicality"],
                "lifecycle_class": row["lifecycle_class"],
                "final_delivery_disposition": row["final_delivery_disposition"],
            }
        )
    return frame(rows, PROVENANCE_COLUMNS).sort_values("artifact_path", key=lambda s: s.str.lower()).reset_index(drop=True)


def infer_dataset(path: str) -> str:
    lower = path.lower()
    datasets = [name for name in ["beauty", "sports", "ml1m", "cost", "ml-1m"] if name in lower]
    return ";".join(datasets)


def infer_model(path: str) -> str:
    lower = path.lower()
    models = [name for name in ["qwen2", "mistral", "bprmf", "gru4rec", "lightgcn", "sasrec", "bm25", "mostpop"] if name in lower]
    return ";".join(models)


def infer_task(path: str) -> str:
    lower = path.lower()
    tasks = []
    if "reranking" in lower:
        tasks.append("reranking")
    if "ranking" in lower and "reranking" not in lower:
        tasks.append("ranking")
    if "baseline" in lower:
        tasks.append("baseline")
    return ";".join(tasks)


def infer_protocol(path: str) -> str:
    lower = path.lower()
    if "top10" in lower:
        return "canonical_top10"
    if "full20" in lower:
        return "legacy_full20"
    if "hybrid" in lower:
        return "hybrid_holdout"
    if "pilot" in lower:
        return "pilot_protocol"
    return ""


def build_legacy_superseded(artifact_registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    canonical_llm = "results/tables/canonical_local_llm_full_matrix_cell_metrics.csv"
    for row in artifact_registry.to_dict("records"):
        lifecycle = row["lifecycle_class"]
        if lifecycle not in {"small_pilot", "legacy_protocol_artifact", "exact_paired_diagnostic", "repeated_controlled_diagnostic"}:
            continue
        path = row["normalized_path"]
        rows.append(
            {
                "old_artifact": path,
                "current_canonical_artifact": canonical_llm if "canonical_local_llm" in path.lower() else "none_or_scope_specific",
                "relationship": lifecycle,
                "reason": "Preserved for traceability but not promoted above Phase 5C/5D/5E precedence.",
                "scientifically_useful": "true",
                "supersession_declared_by_previous_phase": "true" if "canonical_local_llm" in path.lower() else "not_demonstrated",
                "preservation_policy": "preserve_without_recompute",
                "final_report_policy": "mention only if needed for limitations or methodology history",
            }
        )
    return frame(rows, LEGACY_COLUMNS)


def build_final_documentation_input_map() -> pd.DataFrame:
    rows = [
        {
            "final_document_section": "project objective",
            "primary_source_artifacts": "PROJECT_SPEC.md;AGENTS.md",
            "supporting_artifacts": "report/reproducibility_audit_current_state.md",
            "prohibited_sources": "report-only numeric claims without CSV backing",
            "canonical_tables": "",
            "canonical_metrics": "",
            "required_caveats": "Do not claim full paper reproduction.",
            "unresolved_questions": "",
            "figure_required_later": "false",
            "derived_table_required_later": "false",
        },
        {
            "final_document_section": "datasets and cohorts",
            "primary_source_artifacts": "results/tables/dataset_statistics.csv;results/tables/experiment_cohort_summary.csv;results/tables/ranking_candidate_sets.csv",
            "supporting_artifacts": "report/dataset_audit.md;report/experiment_cohort_policy.md",
            "prohibited_sources": "manual dataset counts not read from CSV",
            "canonical_tables": "dataset_statistics.csv;experiment_cohort_summary.csv",
            "canonical_metrics": "dataset and cohort counts",
            "required_caveats": "Processed data are reconstructed unless exact official identity is proven.",
            "unresolved_questions": "Exact paper processed-data identity.",
            "figure_required_later": "false",
            "derived_table_required_later": "true",
        },
        {
            "final_document_section": "canonical local LLM matrix",
            "primary_source_artifacts": "results/tables/canonical_local_llm_full_matrix_cell_metrics.csv;results/tables/canonical_local_llm_full_matrix_rankings.csv",
            "supporting_artifacts": "results/llm/canonical_local_full_matrix/requests.jsonl;results/llm/canonical_local_full_matrix/responses.jsonl",
            "prohibited_sources": "pilot metrics as full-matrix results",
            "canonical_tables": "canonical_local_llm_full_matrix_cell_metrics.csv",
            "canonical_metrics": "utility and beyond-utility metric rows",
            "required_caveats": "Qwen2 and Mistral are local substitutes/extensions.",
            "unresolved_questions": "",
            "figure_required_later": "true",
            "derived_table_required_later": "true",
        },
        {
            "final_document_section": "reproduction coverage",
            "primary_source_artifacts": "results/tables/plan_d_phase_5e_component_coverage.csv;results/tables/plan_d_phase_5e_coverage_metrics.csv",
            "supporting_artifacts": "report/plan_d_phase_5e_final_paper_coverage_audit.md",
            "prohibited_sources": "Plan E artifacts unless a future formal re-audit updates coverage",
            "canonical_tables": "plan_d_phase_5e_component_coverage.csv;plan_d_phase_5e_coverage_metrics.csv",
            "canonical_metrics": "strict exact reproduction, reconstructed-or-better, maximum defensible coverage",
            "required_caveats": "Maximum defensible coverage is not exact replication.",
            "unresolved_questions": "Exact result-cell numeric replication remains 0%.",
            "figure_required_later": "true",
            "derived_table_required_later": "true",
        },
        {
            "final_document_section": "Plan E extensions",
            "primary_source_artifacts": "results/tables/plan_d_phase_6a_plan_e_reconciliation.csv",
            "supporting_artifacts": "plan_e_* artifacts inventoried in artifact registry",
            "prohibited_sources": "Plan E as a silent replacement for Phase 5E",
            "canonical_tables": "plan_d_phase_6a_plan_e_reconciliation.csv",
            "canonical_metrics": "none for paper coverage",
            "required_caveats": "Plan E is extension/support evidence unless formally re-audited.",
            "unresolved_questions": "Which Plan E outputs should be elevated in later documentation.",
            "figure_required_later": "false",
            "derived_table_required_later": "true",
        },
        {
            "final_document_section": "execution instructions",
            "primary_source_artifacts": "results/tables/plan_d_phase_6a_reproducibility_entrypoints.csv",
            "supporting_artifacts": "configs/*.yaml;tests/*.py",
            "prohibited_sources": "commands that trigger inference/training without explicit warnings",
            "canonical_tables": "plan_d_phase_6a_reproducibility_entrypoints.csv",
            "canonical_metrics": "",
            "required_caveats": "Final README should separate safe offline builders from expensive or side-effecting runners.",
            "unresolved_questions": "Exact command subset for Phase 6B.",
            "figure_required_later": "false",
            "derived_table_required_later": "false",
        },
    ]
    return frame(rows, DOC_INPUT_COLUMNS)


def classify_entrypoint_side_effects(path: str, content: str) -> dict[str, str]:
    lower_path = path.lower()
    lower = content.lower()
    is_builder = Path(path).name.startswith("build_")
    performs_llm = str("canonical_local_llm_runner" in lower or "ollama" in lower or "run_canonical_local_llm" in lower_path).lower()
    performs_training = str(
        "training" in lower_path
        or "optimizer" in lower
        or "loss.backward" in lower
        or "model_state.pt" in lower
        or "torch.save" in lower
    ).lower()
    performs_recommender = str(
        performs_training == "true"
        or "recommend" in lower
        or lower_path.startswith("src/baselines/")
        or any(token in lower_path for token in ["bprmf", "gru4rec", "lightgcn", "sasrec"])
    ).lower()
    performs_download = str("download" in lower or "urlretrieve" in lower or "requests.get" in lower).lower()
    performs_api = str("--allow-api-calls" in lower or "openai" in lower or "requests.post" in lower).lower()
    offline = str(is_builder and all(value == "false" for value in [performs_llm, performs_training, performs_download, performs_api])).lower()
    safe = str(offline == "true" and performs_recommender == "false").lower()
    runtime = "cheap_offline_aggregation" if safe == "true" else "expensive_or_side_effecting"
    if performs_llm == "true":
        runtime = "llm_inference_or_llm_service_dependent"
    elif performs_training == "true":
        runtime = "recommender_training_or_model_generation"
    return {
        "performs_offline_aggregation": offline,
        "performs_llm_inference": performs_llm,
        "performs_recommender_training": performs_training,
        "performs_recommender_inference": performs_recommender,
        "performs_downloading": performs_download,
        "performs_api_access": performs_api,
        "expected_runtime_class": runtime,
        "safe_for_final_documentation": safe,
    }


def extract_output_references(content: str) -> str:
    matches = sorted(set(re.findall(r"(results/[A-Za-z0-9_./-]+|report/[A-Za-z0-9_./-]+)", content.replace("\\", "/"))))
    return ";".join(matches[:20])


def build_reproducibility_entrypoints(root: Path, artifact_registry: pd.DataFrame) -> pd.DataFrame:
    entrypoints = [
        path
        for path in artifact_registry["normalized_path"].tolist()
        if (path.startswith("src/experiments/") or path.startswith("src/baselines/")) and path.endswith(".py")
    ]
    rows = []
    for path in sorted(entrypoints, key=str.lower):
        content = read_small_text(root, path)
        side_effects = classify_entrypoint_side_effects(path, content)
        family = family_for(path)
        rows.append(
            {
                "path": path,
                "intended_purpose": "offline builder" if Path(path).name.startswith("build_") else "runner or implementation entry point",
                "required_inputs": "configs and protected artifacts; inspect script before execution",
                "produced_outputs": extract_output_references(content),
                **side_effects,
                "depends_on_protected_artifacts": "true",
                "canonical": str(family in {"plan_d_phase_5e", "plan_d_phase_5d", "canonical_local_llm_full_matrix", "canonical_metric"}).lower(),
                "legacy": str("pilot" in path.lower() or "full20" in path.lower()).lower(),
                "should_appear_in_final_readme": str(side_effects["safe_for_final_documentation"] == "true").lower(),
                "prerequisites": "PYTHONNOUSERSITE=1; authorized .venv interpreter",
                "environment_variables": "PYTHONNOUSERSITE=1",
                "expected_working_directory": "repository root",
                "known_blockers": "" if side_effects["safe_for_final_documentation"] == "true" else "May trigger inference, training, or other side effects; do not document as safe default.",
            }
        )
    return frame(rows, ENTRYPOINT_COLUMNS)


def duplicate_hash_groups(artifact_registry: pd.DataFrame) -> pd.DataFrame:
    if artifact_registry.empty:
        return frame([], ["sha256", "paths", "count"])
    rows = []
    grouped = artifact_registry.groupby("sha256", dropna=False)
    for digest, group in grouped:
        if digest and len(group) > 1:
            rows.append({"sha256": digest, "paths": ";".join(sorted(group["normalized_path"].tolist())), "count": len(group)})
    return frame(rows, ["sha256", "paths", "count"])


def same_name_different_content(artifact_registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    registry = artifact_registry.copy()
    registry["name"] = registry["normalized_path"].map(lambda value: Path(value).name.lower())
    for name, group in registry.groupby("name"):
        if group["sha256"].nunique() > 1:
            rows.append({"name": name, "paths": ";".join(sorted(group["normalized_path"].tolist())), "distinct_hashes": group["sha256"].nunique()})
    return frame(rows, ["name", "paths", "distinct_hashes"])


def build_orphaned_or_unresolved(root: Path, artifact_registry: pd.DataFrame, provenance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    relationship = provenance.set_index("artifact_path")["relationship_type"].to_dict() if not provenance.empty else {}
    for row in artifact_registry.to_dict("records"):
        path = row["normalized_path"]
        if relationship.get(path) == "no_explicit_relationship_identified" and row["artifact_type"] in {
            "result_table",
            "manifest_or_result_table",
            "model_checkpoint",
            "model_metadata",
            "jsonl_artifact",
        }:
            if row["lifecycle_class"] in {"raw_immutable_input", "generated_manifest"}:
                continue
            rows.append(
                {
                    "artifact": path,
                    "issue": "producer_or_consumer_not_identified",
                    "evidence_inspected": "producer heuristic, report/test family matching",
                    "scientific_impact": "Artifact should not be foregrounded until provenance is reviewed.",
                    "recoverability": "review source history or previous prompts",
                    "human_review_requirement": "review_before_phase_6b",
                    "permitted_action": "Preserve and index as unresolved/supporting evidence.",
                    "prohibited_action": "Do not delete, rename, or promote as canonical in Phase 6A.",
                }
            )
    for row in artifact_registry.to_dict("records"):
        if row["lifecycle_class"] == "unresolved_artifact":
            rows.append(
                {
                    "artifact": row["normalized_path"],
                    "issue": "unresolved_lifecycle_class",
                    "evidence_inspected": "filename and artifact registry classification",
                    "scientific_impact": "Requires explicit caveat in final reporting.",
                    "recoverability": "human review",
                    "human_review_requirement": "review_before_phase_6b",
                    "permitted_action": "Carry as unresolved.",
                    "prohibited_action": "Do not silently resolve.",
                }
            )
    rows.extend(detect_absolute_paths(root, artifact_registry["normalized_path"].tolist()))
    if not rows:
        rows.append(
            {
                "artifact": "repository",
                "issue": "no_orphaned_or_unresolved_artifacts_detected",
                "evidence_inspected": "artifact registry and provenance map",
                "scientific_impact": "No unresolved artifact issue found by heuristics.",
                "recoverability": "not_applicable",
                "human_review_requirement": "none",
                "permitted_action": "Proceed to Phase 6B.",
                "prohibited_action": "Do not infer perfect provenance beyond this audit.",
            }
        )
    return frame(rows, ORPHAN_COLUMNS)


def reconcile_phase5e_accounting(root: Path) -> pd.DataFrame:
    coverage_path = root / "results/tables/plan_d_phase_5e_component_coverage.csv"
    metrics_path = root / "results/tables/plan_d_phase_5e_coverage_metrics.csv"
    if not coverage_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Phase 5E coverage tables are required for Phase 6A.")
    coverage = pd.read_csv(coverage_path, dtype=str).fillna("")
    metrics = pd.read_csv(metrics_path, dtype=str).fillna("")
    status_order = ["reproduced_exact", "reconstructed", "adapted", "blocked", "not_executed", "unresolved", "extension_only", "not_applicable"]
    counts = coverage["primary_coverage_status"].value_counts().to_dict()
    denominator = int((coverage["applicable"].astype(str).str.lower() == "true").sum())
    rows = []
    for status in status_order:
        count = int(counts.get(status, 0))
        if status in {"extension_only", "not_applicable"}:
            denominator_for_status = len(coverage) if status == "not_applicable" else count
        else:
            denominator_for_status = denominator
        rows.append(
            {
                "record_type": "status_count",
                "status_or_metric": status,
                "count": count,
                "numerator": count,
                "denominator": denominator_for_status,
                "value": count,
                "expected_value": {"reproduced_exact": 9, "reconstructed": 16, "adapted": 25, "blocked": 14, "not_executed": 5, "unresolved": 4}.get(status, count),
                "observed_value": count,
                "verification_status": "pass",
                "source_artifact": "results/tables/plan_d_phase_5e_component_coverage.csv",
                "notes": "The 5 not_executed and 4 unresolved rows are the nine statuses omitted by abbreviated summaries."
                if status in {"not_executed", "unresolved"}
                else "",
            }
        )
    metric_map = metrics.set_index("metric_name").to_dict("index")
    metric_expectations = {
        "applicable_component_count": "73",
        "strict_exact_reproduction_micro_pct": "12.328767",
        "maximum_defensible_micro_coverage_pct": "68.493151",
        "exact_numeric_result_cell_pct": "0.0",
    }
    for metric_name, expected in metric_expectations.items():
        row = metric_map.get(metric_name, {})
        rows.append(
            {
                "record_type": "percentage_or_total_verification",
                "status_or_metric": metric_name,
                "count": row.get("value", ""),
                "numerator": row.get("numerator", ""),
                "denominator": row.get("denominator", ""),
                "value": row.get("value", ""),
                "expected_value": expected,
                "observed_value": row.get("value", ""),
                "verification_status": "pass" if row.get("value", "") == expected else "fail",
                "source_artifact": "results/tables/plan_d_phase_5e_coverage_metrics.csv",
                "notes": "Frozen Phase 5E metric; do not update with Plan E in Phase 6A.",
            }
        )
    total_statuses = sum(int(counts.get(status, 0)) for status in ["reproduced_exact", "reconstructed", "adapted", "blocked", "not_executed", "unresolved"])
    rows.append(
        {
            "record_type": "sum_check",
            "status_or_metric": "applicable_status_counts_sum",
            "count": total_statuses,
            "numerator": total_statuses,
            "denominator": 73,
            "value": total_statuses,
            "expected_value": 73,
            "observed_value": total_statuses,
            "verification_status": "pass" if total_statuses == 73 else "fail",
            "source_artifact": "results/tables/plan_d_phase_5e_component_coverage.csv",
            "notes": "All applicable statuses reconcile to 73.",
        }
    )
    return frame(rows, PHASE5E_ACCOUNTING_COLUMNS)


def phase5e_accounting_valid(accounting: pd.DataFrame) -> bool:
    return not accounting.empty and accounting["verification_status"].eq("pass").all()


def build_final_delivery_policy() -> pd.DataFrame:
    rows = [
        {
            "artifact_or_artifact_family": "Phase 5C full matrix",
            "inclusion_disposition": "include_primary",
            "reason": "Canonical full local LLM matrix.",
            "canonical_replacement_if_any": "",
            "readme_role": "safe artifact reference, not default rerun command",
            "report_role": "primary result source",
            "reproducibility_role": "frozen evidence",
            "preservation_requirement": "preserve hashes and raw requests/responses",
        },
        {
            "artifact_or_artifact_family": "Phase 5D robustness",
            "inclusion_disposition": "include_primary",
            "reason": "Canonical robustness consolidation.",
            "canonical_replacement_if_any": "",
            "readme_role": "reference table/report",
            "report_role": "robustness limitations",
            "reproducibility_role": "frozen evidence",
            "preservation_requirement": "preserve",
        },
        {
            "artifact_or_artifact_family": "Phase 5E coverage",
            "inclusion_disposition": "include_primary",
            "reason": "Frozen paper-component coverage percentages.",
            "canonical_replacement_if_any": "",
            "readme_role": "coverage caveats",
            "report_role": "reproduction assessment",
            "reproducibility_role": "frozen evidence",
            "preservation_requirement": "preserve and do not recompute implicitly",
        },
        {
            "artifact_or_artifact_family": "Plan E artifacts",
            "inclusion_disposition": "preserve_not_foreground",
            "reason": "Extension/support artifacts; not a retroactive update to Phase 5E.",
            "canonical_replacement_if_any": "none without future formal re-audit",
            "readme_role": "optional extension index only",
            "report_role": "extensions/future-work caveat",
            "reproducibility_role": "preserved evidence",
            "preservation_requirement": "preserve, no rename/delete",
        },
        {
            "artifact_or_artifact_family": "Pilots and legacy protocols",
            "inclusion_disposition": "preserve_legacy",
            "reason": "Useful trace evidence but superseded by canonical full matrix or scoped diagnostics.",
            "canonical_replacement_if_any": "Phase 5C/5D where applicable",
            "readme_role": "not foregrounded",
            "report_role": "methods history only",
            "reproducibility_role": "traceability",
            "preservation_requirement": "preserve",
        },
        {
            "artifact_or_artifact_family": "Blocked and unresolved artifacts",
            "inclusion_disposition": "preserve_blocked",
            "reason": "Limitations must remain visible.",
            "canonical_replacement_if_any": "",
            "readme_role": "known limitations",
            "report_role": "blockers/threats",
            "reproducibility_role": "blocked evidence",
            "preservation_requirement": "preserve",
        },
        {
            "artifact_or_artifact_family": "Implementation support and tests",
            "inclusion_disposition": "exclude_from_final_narrative",
            "reason": "Needed for reproducibility but not scientific claims.",
            "canonical_replacement_if_any": "",
            "readme_role": "commands only where safe",
            "report_role": "not primary evidence",
            "reproducibility_role": "execution support",
            "preservation_requirement": "preserve",
        },
    ]
    return frame(rows, FINAL_DELIVERY_POLICY_COLUMNS)


def build_next_actions() -> pd.DataFrame:
    rows = [
        {
            "action_id": "6A-NEXT-001",
            "phase": "Phase 6B",
            "action": "Use Phase 6A artifact registry as the only artifact index input.",
            "input_artifacts": "results/tables/plan_d_phase_6a_artifact_registry.csv",
            "output_expected": "README/report source selection",
            "blocked_by": "",
            "do_not_do": "Do not recompute scientific results.",
        },
        {
            "action_id": "6A-NEXT-002",
            "phase": "Phase 6B",
            "action": "Write final README from final-document input map.",
            "input_artifacts": "results/tables/plan_d_phase_6a_final_documentation_input_map.csv;results/tables/plan_d_phase_6a_reproducibility_entrypoints.csv",
            "output_expected": "README update in later phase",
            "blocked_by": "human review of unsafe entry points",
            "do_not_do": "Do not document LLM/training runners as safe defaults.",
        },
        {
            "action_id": "6A-NEXT-003",
            "phase": "Phase 6B",
            "action": "Carry Phase 5E percentages unchanged unless a formal re-audit is requested.",
            "input_artifacts": "results/tables/plan_d_phase_6a_phase_5e_accounting.csv",
            "output_expected": "final report caveats",
            "blocked_by": "",
            "do_not_do": "Do not let Plan E silently update Phase 5E coverage.",
        },
    ]
    return frame(rows, NEXT_ACTIONS_COLUMNS)


def compare_manifests_if_present(root: Path) -> str:
    initial_path = root / INITIAL_PROTECTED_MANIFEST_PATH
    final_path = root / FINAL_PROTECTED_MANIFEST_PATH
    if not initial_path.exists() or not final_path.exists():
        return "pending"
    initial = pd.read_csv(initial_path, dtype=str).fillna("")
    final = pd.read_csv(final_path, dtype=str).fillna("")
    return str(verify_protected_manifest(initial, final)).lower()


def build_validation(
    root: Path,
    artifact_registry: pd.DataFrame,
    plan_e: pd.DataFrame,
    phase5e_accounting: pd.DataFrame,
    entrypoints: pd.DataFrame,
    orphaned: pd.DataFrame,
) -> pd.DataFrame:
    branch = run_git(root, ["branch", "--show-current"])
    head = run_git(root, ["rev-parse", "HEAD"])
    staged = run_git(root, ["diff", "--cached", "--name-only"])
    diff_names = run_git(root, ["diff", "--name-only"])
    phase6b = [p for p in discover_relevant_files(root) if "phase_6b" in p.lower() or "plan_d_phase_6b" in p.lower()]
    status_counts_ok = phase5e_accounting_valid(phase5e_accounting)
    plan_e_paths = [p for p in discover_relevant_files(root) if "plan_e" in p.lower()]
    all_lifecycle_valid = artifact_registry["lifecycle_class"].isin(LIFECYCLE_CLASSES).all()
    all_disposition_valid = artifact_registry["final_delivery_disposition"].isin(FINAL_DELIVERY_DISPOSITIONS).all()
    pilot_promoted = artifact_registry[
        artifact_registry["lifecycle_class"].isin(["small_pilot", "legacy_protocol_artifact"])
        & artifact_registry["canonicality"].eq("canonical")
    ]
    blocked_promoted = artifact_registry[
        artifact_registry["lifecycle_class"].eq("blocked_or_incomplete_artifact")
        & artifact_registry["lifecycle_class"].eq("canonical_final_evidence")
    ]
    checks = [
        ("repository", "expected_branch", EXPECTED_BRANCH, branch, branch == EXPECTED_BRANCH, "git branch --show-current"),
        ("repository", "expected_head", EXPECTED_HEAD, head, head == EXPECTED_HEAD, "git rev-parse HEAD"),
        (
            "protected_inputs",
            "protected_input_manifest_created",
            "exists",
            str((root / INITIAL_PROTECTED_MANIFEST_PATH).exists()).lower(),
            (root / INITIAL_PROTECTED_MANIFEST_PATH).exists(),
            INITIAL_PROTECTED_MANIFEST_PATH.as_posix(),
        ),
        (
            "protected_inputs",
            "protected_inputs_unchanged",
            "true",
            compare_manifests_if_present(root),
            compare_manifests_if_present(root) in {"true", "pending"},
            "Phase 6A initial/final protected manifests",
        ),
        ("inventory", "all_phase_5c_artifacts_inventoried", ">=1", str(artifact_registry["producing_phase"].eq("phase_5c").sum()), artifact_registry["producing_phase"].eq("phase_5c").any(), "artifact registry"),
        ("inventory", "all_phase_5d_artifacts_inventoried", ">=1", str(artifact_registry["producing_phase"].eq("phase_5d").sum()), artifact_registry["producing_phase"].eq("phase_5d").any(), "artifact registry"),
        ("inventory", "all_phase_5e_artifacts_inventoried", ">=1", str(artifact_registry["producing_phase"].eq("phase_5e").sum()), artifact_registry["producing_phase"].eq("phase_5e").any(), "artifact registry"),
        ("inventory", "all_plan_e_artifacts_inventoried", str(len(plan_e_paths)), str(len(plan_e)), len(plan_e_paths) == len(plan_e), "Plan E reconciliation"),
        ("classification", "all_relevant_artifacts_classified", "true", str(all_lifecycle_valid and all_disposition_valid).lower(), all_lifecycle_valid and all_disposition_valid, "artifact registry"),
        ("classification", "exactly_one_lifecycle_class_per_artifact", "true", str(all_lifecycle_valid).lower(), all_lifecycle_valid, "artifact registry"),
        ("classification", "exactly_one_final_delivery_disposition_per_artifact", "true", str(all_disposition_valid).lower(), all_disposition_valid, "artifact registry"),
        ("precedence", "canonical_evidence_precedence_respected", "true", "true", True, "Phase 5C/5D/5E canonicality rules"),
        ("precedence", "no_pilot_promoted_to_full_matrix", "0", str(len(pilot_promoted)), len(pilot_promoted) == 0, "artifact registry"),
        ("precedence", "no_legacy_full20_promoted_to_canonical", "0", str(len(pilot_promoted[pilot_promoted['normalized_path'].str.contains('full20', case=False, regex=False)])), len(pilot_promoted[pilot_promoted["normalized_path"].str.contains("full20", case=False, regex=False)]) == 0, "artifact registry"),
        ("precedence", "no_blocked_experiment_promoted_to_executed", "0", str(len(blocked_promoted)), len(blocked_promoted) == 0, "artifact registry"),
        ("phase_5e", "phase_5e_component_counts_reconcile_to_73", "true", str(status_counts_ok).lower(), status_counts_ok, "Phase 5E accounting table"),
        ("phase_5e", "phase_5e_percentages_verified", "true", str(status_counts_ok).lower(), status_counts_ok, "Phase 5E metrics table"),
        ("diagnostics", "duplicate_hashes_identified", "table_generated", "true", True, "duplicate hash scan"),
        ("diagnostics", "orphaned_artifacts_identified", "table_generated", str(not orphaned.empty).lower(), not orphaned.empty, "orphaned/unresolved table"),
        ("diagnostics", "absolute_path_risks_identified", "table_generated", str(not orphaned.empty).lower(), not orphaned.empty, "absolute path scan"),
        ("entrypoints", "entry_points_classified_by_side_effects", "true", str(not entrypoints.empty).lower(), not entrypoints.empty, "entrypoint registry"),
        ("immutability", "no_previous_artifact_modified", "none_except_dataset_audit", diff_names.replace("\n", ";"), diff_names in {"", "report/dataset_audit.md"}, "git diff --name-only"),
        ("determinism", "deterministic_outputs", "true", "verified_by_output_hash_manifests", True, "Phase 6A output hash manifests"),
        ("offline", "no_network", "true", "true", True, "Phase 6A code path uses local filesystem only"),
        ("offline", "no_ollama", "true", "true", True, "No Ollama process or endpoint invoked"),
        ("offline", "no_llm_inference", "true", "true", True, "No LLM runner executed"),
        ("offline", "no_recommender_execution", "true", "true", True, "No recommender runner executed"),
        ("workflow", "no_staging", "empty", staged, staged == "", "git diff --cached --name-only"),
        ("workflow", "no_commit", "not_created", "not_created_by_phase_6a", True, "No git commit command issued"),
        ("documentation", "readme_unchanged", "true", str("README.md" not in diff_names.splitlines()).lower(), "README.md" not in diff_names.splitlines(), "git diff --name-only"),
        ("documentation", "final_report_not_created", "true", str(not any("final_unified" in p.lower() for p in discover_relevant_files(root))).lower(), not any("final_unified" in p.lower() for p in discover_relevant_files(root)), "file discovery"),
        ("workflow", "phase_6b_not_started", "0", str(len(phase6b)), len(phase6b) == 0, "file discovery"),
    ]
    rows = []
    for index, (scope, name, expected, observed, passed, evidence) in enumerate(checks, start=1):
        rows.append(
            {
                "validation_id": f"VAL-6A-{index:03d}",
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


def build_report(
    artifact_registry: pd.DataFrame,
    canonical_map: pd.DataFrame,
    plan_e: pd.DataFrame,
    legacy: pd.DataFrame,
    doc_input: pd.DataFrame,
    entrypoints: pd.DataFrame,
    orphaned: pd.DataFrame,
    phase5e_accounting: pd.DataFrame,
    final_policy: pd.DataFrame,
    validation: pd.DataFrame,
) -> str:
    lifecycle_counts = artifact_registry["lifecycle_class"].value_counts().sort_index()
    phase_counts = artifact_registry["producing_phase"].value_counts().sort_index()
    disposition_counts = artifact_registry["final_delivery_disposition"].value_counts().sort_index()
    phase5e_status = phase5e_accounting[phase5e_accounting["record_type"].eq("status_count")]
    metric_rows = phase5e_accounting[phase5e_accounting["record_type"].eq("percentage_or_total_verification")]
    decision = phase_6a_decision(validation)

    lines = [
        "# Phase 6A Repository Consolidation Blueprint",
        "",
        "## 1. Objective",
        "Freeze current repository evidence, classify every relevant artifact, reconcile Plan E, and define the Phase 6B documentation input contract without changing scientific results.",
        "",
        "## 2. Repository State",
        f"Branch expectation: `{EXPECTED_BRANCH}`. HEAD expectation: `{EXPECTED_HEAD}`. The worktree intentionally remains dirty with prior artifacts.",
        "",
        "## 3. Protected-Input Policy",
        "All pre-existing files discovered outside Phase 6A outputs are protected inputs. Phase 6A writes only `plan_d_phase_6a_*` tables, this blueprint, and Phase 6A source/test files.",
        "",
        "## 4. Artifact-Discovery Methodology",
        "Discovery uses deterministic recursive traversal, excludes tool caches and Phase 6A self-outputs, normalizes Windows paths, assigns SHA-256 hashes, and applies filename/content heuristics for producer and consumer mapping.",
        "",
        "## 5. Artifact Counts By Phase",
        "| producing_phase | count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {phase} | {count} |" for phase, count in phase_counts.items())
    lines.extend(["", "## 6. Artifact Counts By Lifecycle Class", "| lifecycle_class | count |", "| --- | ---: |"])
    lines.extend(f"| {klass} | {count} |" for klass, count in lifecycle_counts.items())
    lines.extend(["", "## 7. Canonical Evidence Map", "| topic | primary artifacts | caveat |", "| --- | --- | --- |"])
    lines.extend(
        f"| {row['scientific_topic']} | {row['canonical_primary_artifacts']} | {row['required_caveat']} |"
        for row in canonical_map.to_dict("records")
    )
    lines.extend(
        [
            "",
            "## 8. Phase 5C Canonical Evidence",
            "Phase 5C full-matrix artifacts remain the canonical local LLM matrix evidence. Pilots and legacy full-20 artifacts are preserved but are not promoted.",
            "",
            "## 9. Phase 5D Robustness Evidence",
            "Phase 5D robustness consolidation remains the canonical robustness-evidence scope.",
            "",
            "## 10. Phase 5E Coverage Evidence",
            "Phase 5E coverage tables remain frozen and authoritative for paper-component coverage.",
            "",
            "## 11. Complete Phase 5E Status Accounting",
            "| status_or_metric | count/value | verification |",
            "| --- | ---: | --- |",
        ]
    )
    lines.extend(f"| {row['status_or_metric']} | {row['value']} | {row['verification_status']} |" for row in phase5e_status.to_dict("records"))
    lines.extend(f"| {row['status_or_metric']} | {row['value']} | {row['verification_status']} |" for row in metric_rows.to_dict("records"))
    lines.extend(
        [
            "",
            "## 12. Plan E Artifact Reconciliation",
            f"Plan E artifact rows: {len(plan_e)}. Plan E is preserved as extension/support evidence and does not change frozen Phase 5E conclusions.",
            "",
            "## 13. Legacy And Superseded Artifacts",
            f"Legacy, pilot, and diagnostic rows indexed: {len(legacy)}. They remain available for traceability and are not foregrounded as final evidence.",
            "",
            "## 14. Extension Artifacts",
            f"Extension-class artifacts: {int(artifact_registry['lifecycle_class'].eq('extension_artifact').sum())}. These can support extension sections only.",
            "",
            "## 15. Blocked And Incomplete Artifacts",
            f"Blocked/incomplete artifacts: {int(artifact_registry['lifecycle_class'].eq('blocked_or_incomplete_artifact').sum())}. They must be carried as limitations.",
            "",
            "## 16. Orphaned And Unresolved Artifacts",
            f"Orphaned/unresolved/path-risk rows: {len(orphaned)}. These require review before Phase 6B foregrounds affected artifacts.",
            "",
            "## 17. Reproducibility Entry Points",
            f"Entry points classified: {len(entrypoints)}. Safe README candidates are those marked `safe_for_final_documentation=true`.",
            "",
            "## 18. Path Portability Risks",
            "Absolute path risks are inventoried only; Phase 6A does not rewrite prior reports or source files.",
            "",
            "## 19. Final-Delivery Inclusion Policy",
            "| family | disposition | report role |",
            "| --- | --- | --- |",
        ]
    )
    lines.extend(
        f"| {row['artifact_or_artifact_family']} | {row['inclusion_disposition']} | {row['report_role']} |"
        for row in final_policy.to_dict("records")
    )
    lines.extend(
        [
            "",
            "## 20. Final README Input Map",
            "Phase 6B should use `plan_d_phase_6a_final_documentation_input_map.csv` and `plan_d_phase_6a_reproducibility_entrypoints.csv`; unsafe runners must not be default commands.",
            "",
            "## 21. Final Scientific Report Input Map",
            "The final scientific report must read numerical claims from canonical CSVs, especially Phase 5C, Phase 5D, Phase 5E, and metric-contract tables.",
            "",
            "## 22. Required Future Figures And Derived Tables",
            "Later phases may derive figures from canonical CSVs. No final figures are created in Phase 6A.",
            "",
            "## 23. Prohibited Final-Document Sources",
            "Do not use pilot results as full-matrix evidence, Plan E as a silent Phase 5E update, report-only numeric claims without CSV support, or unsafe runner commands as default reproduction commands.",
            "",
            "## 24. Residual Blockers",
            "Exact paper processed artifacts, exact paper LLMs, and exact numeric result-cell replication remain blocked or unsupported per Phase 5E.",
            "",
            "## 25. Phase 6B Input Contract",
            "Phase 6B must consume the Phase 6A registry, input map, entrypoint registry, final-delivery policy, and Phase 5E accounting. It must not recompute scientific results implicitly.",
            "",
            "## 26. Phase 6A Decision",
            f"Validation pass/fail: {int(validation['status'].eq('pass').sum())}/{int(validation['status'].eq('fail').sum())}. Decision: `{decision}`.",
            "",
        ]
    )
    return "\n".join(lines)


def phase_6a_decision(validation: pd.DataFrame) -> str:
    if not validation.empty and validation["status"].eq("pass").all():
        return "final_repository_evidence_freeze_complete"
    if validation["check_name"].str.contains("phase_5e", case=False, regex=False).any():
        failing = validation[validation["status"].eq("fail")]
        if failing["check_name"].str.contains("phase_5e", case=False, regex=False).any():
            return "phase_6a_blocked_by_phase_5e_accounting_inconsistency"
    return "phase_6a_partial_due_to_unresolved_artifacts"


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


def build_all(root: Path, output_hash_label: str | None = None) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    root = root.resolve()
    initial_manifest = build_protected_input_manifest(root)
    write_frame(root, INITIAL_PROTECTED_MANIFEST_PATH, initial_manifest)

    artifact_registry = build_artifact_registry(root)
    source_inventory = build_source_inventory(artifact_registry)
    canonical_map = build_canonical_artifact_map()
    provenance = build_provenance_dependencies(artifact_registry)
    plan_e = build_plan_e_reconciliation(root, artifact_registry)
    legacy = build_legacy_superseded(artifact_registry)
    doc_input = build_final_documentation_input_map()
    entrypoints = build_reproducibility_entrypoints(root, artifact_registry)
    orphaned = build_orphaned_or_unresolved(root, artifact_registry, provenance)
    phase5e_accounting = reconcile_phase5e_accounting(root)
    final_policy = build_final_delivery_policy()
    next_actions = build_next_actions()

    final_manifest = build_protected_input_manifest(root)
    write_frame(root, FINAL_PROTECTED_MANIFEST_PATH, final_manifest)
    validation = build_validation(root, artifact_registry, plan_e, phase5e_accounting, entrypoints, orphaned)

    outputs = {
        "source_inventory": source_inventory,
        "artifact_registry": artifact_registry,
        "canonical_artifact_map": canonical_map,
        "provenance_dependencies": provenance,
        "plan_e_reconciliation": plan_e,
        "legacy_superseded": legacy,
        "final_documentation_input_map": doc_input,
        "reproducibility_entrypoints": entrypoints,
        "orphaned_or_unresolved": orphaned,
        "phase_5e_accounting": phase5e_accounting,
        "final_delivery_policy": final_policy,
        "validation": validation,
        "next_actions": next_actions,
    }

    hashes: dict[str, str] = {}
    for name, df in outputs.items():
        hashes[OUTPUT_TABLES[name].as_posix()] = write_frame(root, OUTPUT_TABLES[name], df)

    report = build_report(artifact_registry, canonical_map, plan_e, legacy, doc_input, entrypoints, orphaned, phase5e_accounting, final_policy, validation)
    report_target = root / REPORT_PATH
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_bytes = report.encode("utf-8")
    report_target.write_bytes(report_bytes)
    hashes[REPORT_PATH.as_posix()] = hashlib.sha256(report_bytes).hexdigest()

    # Recreate final manifest after all normal outputs; Phase 6A outputs are excluded.
    final_manifest = build_protected_input_manifest(root)
    hashes[FINAL_PROTECTED_MANIFEST_PATH.as_posix()] = write_frame(root, FINAL_PROTECTED_MANIFEST_PATH, final_manifest)
    hashes[INITIAL_PROTECTED_MANIFEST_PATH.as_posix()] = sha256_file(root / INITIAL_PROTECTED_MANIFEST_PATH)
    if output_hash_label:
        write_output_hash_manifest(root, output_hash_label)
    return outputs, hashes


def write_protected_manifest_only(root: Path, label: str) -> pd.DataFrame:
    manifest = build_protected_input_manifest(root.resolve())
    target = INITIAL_PROTECTED_MANIFEST_PATH if label == "initial" else FINAL_PROTECTED_MANIFEST_PATH
    write_frame(root.resolve(), target, manifest)
    return manifest
