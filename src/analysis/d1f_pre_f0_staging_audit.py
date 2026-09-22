"""Build the D1F pre-F0 staging classification evidence.

This script is intentionally offline. It reads Git status and diffs, classifies
the pending workspace paths, and writes audit CSVs for selective staging.
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSIFICATION_PATH = PROJECT_ROOT / "results" / "tables" / "d1f_staging_classification.csv"
FINALIZATION_AUDIT_PATH = PROJECT_ROOT / "results" / "tables" / "d1f_finalization_diff_audit.csv"

CLASSIFICATION_COLUMNS = [
    "path",
    "git_status",
    "classification",
    "origin_phase",
    "reason",
    "scientific_role",
    "historical_or_current",
    "action",
    "expected_after_action",
]

FINALIZATION_COLUMNS = [
    "path",
    "git_status",
    "origin_phase",
    "diff_pattern",
    "protected_input_compatibility_only",
    "metric_formula_change",
    "cohort_change",
    "head_or_token_change",
    "decision",
    "reason",
]

FINALIZATION_FILES = {
    "src/analysis/finalization_f3_canonical_scientific_consolidation.py",
    "src/analysis/finalization_f4_metric_resolution.py",
    "src/analysis/finalization_f5a_paper_numeric_evidence.py",
    "src/analysis/finalization_f5af_atomic_paper_cells.py",
    "src/analysis/finalization_f5ar_structured_numeric_recovery.py",
    "src/analysis/finalization_f5b_ild_policy.py",
    "src/analysis/finalization_f5c_popularity_bias.py",
    "src/analysis/finalization_f5d_serendipity.py",
    "src/analysis/finalization_f5e_calibration.py",
    "src/analysis/finalization_f6a_paper_scope_rebaseline.py",
    "src/analysis/finalization_f6b_atomic_scope_registry.py",
    "src/analysis/finalization_f6c_lastfm_readiness.py",
    "src/analysis/finalization_f6d1_lastfm_baselines.py",
    "src/analysis/finalization_f6d2_lastfm_trainable_models.py",
    "src/analysis/finalization_f6d3_lastfm_llm_preflight.py",
    "src/analysis/finalization_f6d4a_lastfm_llm_ranking.py",
}

INFRA_FIX_FILES = FINALIZATION_FILES | {
    "src/analysis/protected_input_compatibility.py",
    "src/analysis/build_exactness_frontier_and_evidence_closure.py",
    "src/analysis/build_workspace_forensic_state_audit.py",
}

HISTORICAL_RESTORE_FILES = {
    "report/future_operations_context_handoff.md",
    "report/notes.md",
    "report/reproduction_r4_three_way_beyond_utility_reconciliation.md",
    "report/reproduction_r5a_controlled_recovery_manifest.md",
    "report/reproduction_r5b_canddif_recovery.md",
    "results/tables/reproduction_r4_postcondition_git_state.csv",
    "results/tables/reproduction_r4_precondition_git_state.csv",
    "results/tables/reproduction_r4_protected_artifact_hash_check.csv",
    "results/tables/reproduction_r4_validation.csv",
    "results/tables/reproduction_r5a_validation.csv",
    "results/tables/reproduction_r5b_runtime_provenance.csv",
    "results/tables/reproduction_r5b_validation.csv",
}

HISTORICAL_SNAPSHOT_OUTPUTS = {
    "report/exactness_frontier_and_evidence_closure.md",
    "report/workspace_forensic_state_audit.md",
    "results/manifests/current_project_context_snapshot.json",
    "results/manifests/exactness_frontier_snapshot.json",
    "results/tables/artifact_lineage_and_supersession.csv",
    "results/tables/blockers_gaps_and_next_actions.csv",
    "results/tables/current_reproduction_status_matrix.csv",
    "results/tables/exactness_elevation_candidates.csv",
    "results/tables/exactness_frontier_by_component.csv",
    "results/tables/fifth_dataset_extension_requirements.csv",
    "results/tables/irreducible_exactness_blockers.csv",
    "results/tables/numeric_cell_exactness_frontier.csv",
    "results/tables/numeric_result_reconciliation.csv",
    "results/tables/paper_atomic_contract.csv",
    "results/tables/paper_repository_contract_conflicts.csv",
    "results/tables/paper_to_local_traceability.csv",
    "results/tables/project_claim_verification.csv",
    "results/tables/workspace_forensic_audit_determinism.csv",
    "results/tables/workspace_forensic_audit_test_results.csv",
    "results/tables/workspace_phase_and_supersession_registry.csv",
    "results/tables/workspace_relevant_file_inventory.csv",
}

TRANSIENT_GENERATED_FILES = HISTORICAL_SNAPSHOT_OUTPUTS | {
    "results/tables/d1r3_current_suite_junit.xml",
}

RAW_RUNTIME_FILES = {
    "data/raw/foodcom/source/food-com-recipes-and-user-interactions.zip",
    "data/raw/foodcom/d0a_extracted/e7a43e623c61/RAW_interactions.csv",
    "data/raw/foodcom/d0a_extracted/e7a43e623c61/RAW_recipes.csv",
}

VIDEO_GAMES_PURGE_PREFIXES = (
    "Espansione_progetto/video_games_tutti_esperimenti/",
    "data/processed/video_games/",
    "results/extension/e1_pilot/",
)

VIDEO_GAMES_PURGE_FILES = {
    "ISTRUZIONI_AMICO.md",
    "configs/extension_e1_pilot.yaml",
    "configs/video_games.yaml",
    "report/dataset_audit.md",
    "report/extension_e0_protocol_freeze.md",
    "report/extension_e1_pilot.md",
    "report/results_analysis.md",
    "results/extension/extension_e0_video_games_mostpop_candidate_dry_run.csv",
    "results/extension/extension_e0_video_games_random_candidate_dry_run.csv",
    "results/manifests/extension_e0_prompt_samples.csv",
    "results/manifests/extension_e0_reproducibility.json",
    "results/manifests/extension_e0_video_games_primary_candidates.csv",
    "results/manifests/extension_e1_pilot_manifest.csv",
    "results/manifests/extension_e1_reproducibility.json",
    "results/manifests/extension_e1_response_manifest.csv",
    "results/manifests/extension_e1_runtime_provenance.json",
    "results/manifests/extension_e1_selected_users.csv",
    "results/metrics/video_games_coverage_metrics.csv",
    "results/metrics/video_games_final_results.csv",
    "results/metrics/video_games_position_bias_metrics.csv",
    "results/metrics/video_games_reproduction_metrics.csv",
    "results/raw_outputs/video_games_history_length_prompts.csv",
    "results/raw_outputs/video_games_mostpop_recommendations.csv",
    "results/tables/extension_e0_video_games_dry_run_summary.csv",
    "results/tables/extension_e1_canary_results.csv",
    "results/tables/extension_e1_early_health_results.csv",
    "results/tables/extension_e1_full_run_projection.csv",
    "results/tables/extension_e1_history_validation.csv",
    "results/tables/extension_e1_model_readiness.csv",
    "results/tables/extension_e1_pilot_utility_by_cell.csv",
    "results/tables/extension_e1_preflight.csv",
    "results/tables/extension_e1_reliability_by_cell.csv",
    "results/tables/extension_e1_reliability_by_model.csv",
    "results/tables/extension_e1_runtime_by_model.csv",
    "src/data/prepare_video_games.py",
    "src/experiments/run_video_games_gemma2.py",
    "src/extension/e0_dry_run.py",
    "src/extension/e1_pilot.py",
    "tests/test_extension_e1_pilot.py",
}


@dataclass(frozen=True)
class Classification:
    classification: str
    origin_phase: str
    reason: str
    scientific_role: str
    historical_or_current: str
    action: str
    expected_after_action: str


def run_git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def parse_status() -> dict[str, str]:
    output = run_git(["status", "--short", "--untracked-files=all"])
    paths: dict[str, str] = {}
    for line in output.splitlines():
        if not line:
            continue
        status = line[:2]
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths[path.replace("\\", "/")] = status
    return paths


def exists(path: str) -> bool:
    return (PROJECT_ROOT / path).exists()


def is_video_games_purge(path: str) -> bool:
    lowered = path.lower()
    return (
        path in VIDEO_GAMES_PURGE_FILES
        or lowered.startswith(VIDEO_GAMES_PURGE_PREFIXES)
        or "video_games" in lowered
        or "/extension_e1_" in lowered
        or path.startswith("results/extension/e1_pilot/")
    )


def is_current_extension(path: str) -> bool:
    if path in {
        ".gitignore",
        "configs/extension_e0_protocol.yaml",
        "Espansione_progetto/gemma2_su_vecchi_dataset/0_prepara_beauty_sports.py",
        "src/extension/e0_candidates.py",
        "src/extension/e0_protocol.py",
        "tests/test_extension_e0_protocol.py",
        "src/analysis/d1f_pre_f0_staging_audit.py",
        "results/tables/d1f_staging_classification.csv",
        "results/tables/d1f_finalization_diff_audit.csv",
    }:
        return True
    prefixes = (
        "report/d0",
        "report/d1_",
        "report/d1r",
        "report/extension_cell_universe",
        "report/extension_h0_",
        "results/manifests/d0",
        "results/manifests/d1",
        "results/manifests/d1r",
        "results/manifests/extension_cell_",
        "results/manifests/extension_h0_",
        "results/tables/d0",
        "results/tables/d1_",
        "results/tables/d1r",
        "results/tables/extension_cell_",
        "results/tables/extension_h0_",
        "results/tables/extension_metric_",
        "results/tables/extension_parity_",
        "results/tables/extension_raw_",
        "src/analysis/d1r",
        "src/extension/d0",
        "src/extension/d1_",
        "src/extension/e2a_",
        "src/extension/h0_",
        "tests/test_extension_d0",
        "tests/test_extension_d1_",
        "tests/test_extension_e2a_",
        "tests/test_extension_h0_",
    )
    return path.startswith(prefixes)


def origin_for_current(path: str) -> str:
    if "/d0a_" in path or path.startswith(("report/d0a", "src/extension/d0a", "tests/test_extension_d0a")):
        return "D0A"
    if "/d0b_" in path or path.startswith(("report/d0b", "src/extension/d0b", "tests/test_extension_d0b")):
        return "D0B"
    if "/d1r3_" in path or path.startswith(("report/d1r3", "src/analysis/d1r3")):
        return "D1R3"
    if "/d1r2_" in path or path.startswith(("report/d1r2", "src/analysis/d1r2")):
        return "D1R2"
    if "/d1r_" in path or path.startswith("report/d1r_"):
        return "D1R"
    if "/d1_" in path or path.startswith(("report/d1_", "src/extension/d1_", "tests/test_extension_d1_")):
        return "D1"
    if "extension_h0" in path or "h0_" in path:
        return "H0"
    if "extension_cell" in path or "e2a_" in path or "extension_metric_" in path or "extension_raw_" in path or "extension_parity_" in path:
        return "E2A"
    if "d1f_" in path:
        return "D1F"
    if path in {".gitignore", "configs/extension_e0_protocol.yaml"}:
        return "D1"
    return "CURRENT_EXTENSION"


def classify(path: str, git_status: str) -> Classification:
    if path in HISTORICAL_RESTORE_FILES:
        return Classification(
            "D_RESTORE_HISTORICAL_TEST_SIDE_EFFECT",
            "R4_R5_HISTORY",
            "Historical builders rewrote frozen R4/R5/F6M handoff evidence on the current tree.",
            "preserve frozen historical record",
            "historical",
            "restore_tracked_head_bytes",
            "tracked_file_matches_HEAD",
        )
    if path in TRANSIENT_GENERATED_FILES:
        return Classification(
            "E_EXCLUDE_TRANSIENT_GENERATED_ARTIFACT",
            "HISTORICAL_BUILDER_OR_TEST_RUNTIME",
            "Generated by historical snapshot builders or pytest runtime and not part of the current extension checkpoint.",
            "transient diagnostic output",
            "transient",
            "remove_untracked_artifact",
            "absent_from_worktree_status",
        )
    if path.startswith("data/raw/foodcom/"):
        return Classification(
            "F_EXCLUDE_RAW_OR_RUNTIME",
            "D0A_RAW_INTAKE",
            "Food.com archive and extracted source bytes must remain local-only.",
            "raw material input",
            "raw_runtime",
            "leave_unstaged_ignored",
            "ignored_and_unstaged",
        )
    if path in INFRA_FIX_FILES:
        return Classification(
            "B_INCLUDE_INTENTIONAL_INFRASTRUCTURE_FIX",
            "D1R_COMPATIBILITY",
            "Compatibility-only import-path/protected-input helper change for current validation separation.",
            "test and protected-input compatibility",
            "current",
            "stage_explicitly",
            "cached_in_commit",
        )
    if is_video_games_purge(path):
        return Classification(
            "C_INCLUDE_VIDEO_GAMES_PURGE",
            "D1",
            "Obsolete Video Games or E1 pilot operational artifact removed after Food.com final selection.",
            "remove loser dataset and obsolete pilot runtime",
            "current_purge",
            "stage_explicitly",
            "cached_deletion_in_commit",
        )
    if is_current_extension(path):
        return Classification(
            "A_INCLUDE_CURRENT_EXTENSION",
            origin_for_current(path),
            "Current Food.com selection, cell-universe, H0 parity, D1 cleanup, or D1R validation evidence.",
            "current extension evidence",
            "current",
            "stage_explicitly",
            "cached_in_commit",
        )
    return Classification(
        "G_REVIEW_BLOCKING",
        "UNKNOWN",
        "No D1F rule matched this path.",
        "unknown",
        "unknown",
        "manual_review_required",
        "do_not_stage",
    )


def supplemental_existing_paths() -> dict[str, str]:
    status = {}
    for path in sorted(TRANSIENT_GENERATED_FILES | RAW_RUNTIME_FILES):
        if exists(path):
            status[path] = "IGNORED_PRESENT"
    for path in sorted(
        {
            str(CLASSIFICATION_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            str(FINALIZATION_AUDIT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        }
    ):
        status.setdefault(path, "GENERATED_BY_D1F")
    return status


def build_classification_rows() -> list[dict[str, str]]:
    statuses = parse_status()
    for path, status in supplemental_existing_paths().items():
        statuses.setdefault(path, status)
    rows = []
    for path in sorted(statuses):
        classification = classify(path, statuses[path])
        rows.append(
            {
                "path": path,
                "git_status": statuses[path],
                "classification": classification.classification,
                "origin_phase": classification.origin_phase,
                "reason": classification.reason,
                "scientific_role": classification.scientific_role,
                "historical_or_current": classification.historical_or_current,
                "action": classification.action,
                "expected_after_action": classification.expected_after_action,
            }
        )
    blockers = [row for row in rows if row["classification"] == "G_REVIEW_BLOCKING"]
    if blockers:
        blocked = "\n".join(row["path"] for row in blockers)
        raise RuntimeError(f"D1F classification has REVIEW_BLOCKING rows:\n{blocked}")
    return rows


def status_for(path: str) -> str:
    return parse_status().get(path, "CLEAN_OR_UNTRACKED_IGNORED")


def changed_diff_lines(diff_text: str) -> list[str]:
    lines = []
    for line in diff_text.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(("+", "-")):
            lines.append(line)
    return lines


def finalization_diff_pattern(path: str) -> tuple[str, bool, bool, bool, bool, str]:
    diff_text = run_git(["diff", "--", path])
    changed = changed_diff_lines(diff_text)
    joined = "\n".join(changed)
    protected_terms = (
        "protected_input_compatibility",
        "_d1r_protected_hash_counts",
        "_d1r_protected_hash_status",
        "protected_hash_counts",
        "protected_hash_status",
        "checked=",
        "changed=",
        "PROTECTED_HASHES",
        "F2_PROTECTED_HASHES",
        "relative_path",
        "sha256",
        "file_sha256",
        "if protected_after.empty",
        "if protected.empty",
        "return 0, 0",
        "return False",
        "return True",
        "return checked, changed",
        'return "missing f2 protected hash table"',
    )
    allowed_line = True
    for line in changed:
        body = line[1:].strip()
        if not body or body in {"(", ")", "}"}:
            continue
        if any(term in body for term in protected_terms):
            continue
        if body.startswith(("from src.analysis.protected_input_compatibility import", "path =", "target =", "table =", "table_path =", "checked", "changed", "expected_", "observed_", "relative =", "rel =", "continue", "for ", "if ", "elif ", "return f")):
            continue
        allowed_line = False
    metric_change = any(term in joined.lower() for term in ("hr@", "ndcg", "metric_value", "metric_name", "utility", "coverage_value"))
    cohort_change = any(term in joined.lower() for term in ("cohort", "sample_size", "candidate_count", "history_length", "dataset ="))
    head_token_change = any(term in joined.lower() for term in ("expected_head", "head_sha", "token", "decision token"))
    compatibility_only = allowed_line and not metric_change and not cohort_change and not head_token_change
    pattern = "protected_input_compatibility" if compatibility_only else "requires_review"
    decision = "INCLUDE_INTENTIONAL_INFRASTRUCTURE_FIX" if compatibility_only else "REVIEW_BLOCKING"
    return pattern, compatibility_only, metric_change, cohort_change, head_token_change, decision


def build_finalization_rows() -> list[dict[str, str]]:
    rows = []
    for path in sorted(FINALIZATION_FILES):
        if not run_git(["diff", "--name-only", "--", path]).strip():
            continue
        pattern, compatibility_only, metric_change, cohort_change, head_token_change, decision = finalization_diff_pattern(path)
        rows.append(
            {
                "path": path,
                "git_status": status_for(path),
                "origin_phase": "D1R_COMPATIBILITY",
                "diff_pattern": pattern,
                "protected_input_compatibility_only": str(compatibility_only).lower(),
                "metric_formula_change": str(metric_change).lower(),
                "cohort_change": str(cohort_change).lower(),
                "head_or_token_change": str(head_token_change).lower(),
                "decision": decision,
                "reason": "Shared protected-input compatibility helper replaces duplicated local hash guards.",
            }
        )
    blockers = [row for row in rows if row["decision"] == "REVIEW_BLOCKING"]
    if blockers:
        blocked = "\n".join(row["path"] for row in blockers)
        raise RuntimeError(f"D1F finalization audit has REVIEW_BLOCKING rows:\n{blocked}")
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    write_csv(CLASSIFICATION_PATH, build_classification_rows(), CLASSIFICATION_COLUMNS)
    write_csv(FINALIZATION_AUDIT_PATH, build_finalization_rows(), FINALIZATION_COLUMNS)


if __name__ == "__main__":
    main()
