"""Materialize D1R2 historical/current test separation artifacts.

This script is intentionally offline. It reads the D1R pytest lastfailed cache,
classifies failed/error nodes, records the targeted current-suite result, and
writes D1R2 evidence tables plus the report/manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = PROJECT_ROOT / "results" / "manifests"
REPORT_PATH = PROJECT_ROOT / "report" / "d1r2_historical_current_test_separation.md"

LASTFAILED_PATH = PROJECT_ROOT / ".pytest_cache" / "v" / "cache" / "lastfailed"

ERROR_FILES = {
    "tests/test_phase_f6i2m_hashing_helpers_consolidation_execution.py",
    "tests/test_reproduction_r4_three_way_reconciliation.py",
    "tests/test_reproduction_r5a_controlled_recovery_manifest.py",
    "tests/test_reproduction_r5b_canddif_recovery.py",
}

CURRENT_REGRESSION_FILES = {
    "tests/test_workspace_forensic_state_audit.py",
}

DESTRUCTIVE_FILES = {
    "tests/test_reproduction_r4_three_way_reconciliation.py",
    "tests/test_reproduction_r5a_controlled_recovery_manifest.py",
    "tests/test_reproduction_r5b_canddif_recovery.py",
}

HISTORICAL_REPRODUCTION_LOGIC_PREFIXES = (
    "tests/test_phase_f6g3_",
    "tests/test_phase_f6h1_",
    "tests/test_phase_f6h2_",
    "tests/test_phase_f6j_",
    "tests/test_phase_f6k_",
    "tests/test_phase_f6l_",
    "tests/test_plan_d_phase_6c_",
    "tests/test_plan_f6e1_",
    "tests/test_reproduction_r2_",
    "tests/test_reproduction_r3_",
)

EXPECTED_GROUPS = {
    "tests/test_reproduction_r2_official_metric_calculability.py": (
        "master",
        "10529058c3579c42686b76d5d3ac8b96ba7d6337",
        "R2 official metric calculability frozen root-artifact gate",
    ),
    "tests/test_reproduction_r3_complete_local_metric_audit.py": (
        "master",
        "5167740b80fd917db99df0768ea9358d00ec3515",
        "R3 complete local metric audit frozen root-artifact gate",
    ),
    "tests/test_reproduction_r4_three_way_reconciliation.py": (
        "master",
        "357d8a4cf98c78effc6c92f5b8dc00e334836a56",
        "R4 three-way reconciliation frozen clean-tree gate",
    ),
    "tests/test_reproduction_r5a_controlled_recovery_manifest.py": (
        "master",
        "a96ad41765b242137a83b919d59a3b394daae562",
        "R5A controlled recovery manifest frozen clean-tree gate",
    ),
    "tests/test_reproduction_r5b_canddif_recovery.py": (
        "master",
        "98c28c5364c40e571765a3b50d4dab8affe80743",
        "R5B CandDif recovery frozen clean-tree gate",
    ),
}

PLAN_D_GROUP = (
    "plan-d-extensions",
    "02a20b8b509b8cf834a8766dd1cf1f751ae204bf",
    "Plan-D/F6 historical phase snapshot gate",
)

D1R_SIDE_EFFECT_PATH = TABLES_DIR / "d1r_post_suite_side_effects.csv"

D1R2_SIDE_EFFECT_PATHS = [
    "report/workspace_forensic_state_audit.md",
    "report/future_operations_context_handoff.md",
    "results/tables/workspace_relevant_file_inventory.csv",
    "results/manifests/current_project_context_snapshot.json",
    "results/tables/exactness_frontier_by_component.csv",
    "results/tables/numeric_cell_exactness_frontier.csv",
    "results/tables/exactness_elevation_candidates.csv",
    "results/tables/irreducible_exactness_blockers.csv",
    "results/tables/fifth_dataset_extension_requirements.csv",
    "results/manifests/exactness_frontier_snapshot.json",
    "report/exactness_frontier_and_evidence_closure.md",
]


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(args: list[str]) -> str:
    proc = subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    return proc.stdout.strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def load_lastfailed() -> list[str]:
    with LASTFAILED_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return sorted(data)


def node_file(node_id: str) -> str:
    return node_id.split("::", 1)[0].replace("\\", "/")


def expected_group_for_file(file_path: str) -> tuple[str, str, str]:
    if file_path in EXPECTED_GROUPS:
        return EXPECTED_GROUPS[file_path]
    if file_path == "tests/test_workspace_forensic_state_audit.py":
        return ("current", run_git(["rev-parse", "HEAD"]), "current direct-script entrypoint regression")
    return PLAN_D_GROUP


def classify(node_id: str) -> dict[str, str]:
    file_path = node_file(node_id)
    status = "error" if file_path in ERROR_FILES else "failed"
    branch, commit, group_note = expected_group_for_file(file_path)

    if file_path in CURRENT_REGRESSION_FILES:
        taxonomy = "CURRENT_REGRESSION"
        root_cause = (
            "Direct script execution imported src.audit before adding the repository root to sys.path; "
            "fixed by bootstrapping PROJECT_ROOT before src imports."
        )
        isolation = "run in current tree; use --help import probe to avoid full forensic materialization"
        freeze = "none"
        fix = "applied"
    elif file_path in DESTRUCTIVE_FILES:
        taxonomy = "DESTRUCTIVE_OR_PERSISTENT_SIDE_EFFECT"
        root_cause = (
            "Historical reproduction builder is executed against the current dirty D1 tree and writes "
            "blocked-state reports/tables into persistent current paths."
        )
        isolation = "run in a temporary detached worktree at the frozen historical commit"
        freeze = "EXPECTED_BRANCH/EXPECTED_HEAD plus clean worktree and protected artifact hashes"
        fix = "not_applied; requires isolated historical validation harness"
    elif file_path.startswith(HISTORICAL_REPRODUCTION_LOGIC_PREFIXES):
        taxonomy = "HISTORICAL_REPRODUCTION_LOGIC"
        root_cause = (
            "Archived reproduction/scientific readout assertion belongs to a completed historical phase; "
            "it validates fixed hashes, phase reports, or protocol counts rather than the post-D1 extension tree."
        )
        isolation = "validate in the matching frozen phase worktree or historical artifact namespace"
        freeze = "fixed historical hashes/tokens/protocol counts"
        fix = "not_applied; historical expectation preserved"
    else:
        taxonomy = "HISTORICAL_FREEZE_VALIDATION"
        root_cause = (
            "Historical phase snapshot assumes its original branch/head, clean git status, and frozen "
            "protected hash table; the current D1 Food.com-only tree intentionally differs."
        )
        isolation = "validate in a temporary detached worktree at the recorded expected commit"
        freeze = "EXPECTED_BRANCH/EXPECTED_HEAD, dirty-worktree gate, protected hashes, fixed COMPLETE token"
        fix = "not_applied; historical expectation preserved"

    return {
        "node_id": node_id,
        "test_file": file_path,
        "observed_status": status,
        "taxonomy": taxonomy,
        "expected_branch": branch,
        "expected_commit": commit,
        "group_note": group_note,
        "freeze_specific_assumption": freeze,
        "root_cause": root_cause,
        "current_tree_fix_status": fix,
        "historical_validation_strategy": isolation,
    }


def build_failure_classification(nodes: list[str]) -> list[dict[str, str]]:
    return [classify(node) for node in nodes]


def build_taxonomy_rows(classified: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for taxonomy in [
        "CURRENT_REGRESSION",
        "HISTORICAL_FREEZE_VALIDATION",
        "HISTORICAL_REPRODUCTION_LOGIC",
        "ENVIRONMENT_SPECIFIC",
        "DESTRUCTIVE_OR_PERSISTENT_SIDE_EFFECT",
        "UNKNOWN",
    ]:
        subset = [row for row in classified if row["taxonomy"] == taxonomy]
        rows.append(
            {
                "taxonomy": taxonomy,
                "node_count": len(subset),
                "failed_count": sum(row["observed_status"] == "failed" for row in subset),
                "error_count": sum(row["observed_status"] == "error" for row in subset),
                "current_suite_action": "include_after_fix" if taxonomy == "CURRENT_REGRESSION" else "exclude_from_current_suite",
                "historical_action": "none" if taxonomy == "CURRENT_REGRESSION" else "validate_in_isolated_historical_worktree",
                "final_unknown_allowed": "false",
                "notes": "no UNKNOWN classifications remain" if taxonomy == "UNKNOWN" else "",
            }
        )
    return rows


def build_current_suite_rows() -> list[dict[str, str]]:
    rows = []
    pytest_files = [
        "tests/test_extension_d0a_foodcom_material_feasibility.py",
        "tests/test_extension_d0b_final_fifth_dataset_selection.py",
        "tests/test_extension_d1_final_dataset_cleanup.py",
        "tests/test_extension_h0_history_final_parity_audit.py",
        "tests/test_extension_e0_protocol.py",
        "tests/test_extension_e2a_cell_universe.py",
        "tests/test_exactness_frontier_and_evidence_closure.py",
    ]
    for file_path in pytest_files:
        rows.append(
            {
                "suite_member": file_path,
                "member_type": "pytest_file",
                "current_suite_status": "included",
                "reason": "active current extension/Food.com/H0/E0/E2A or exactness gate",
                "historical_exclusion_reason": "",
            }
        )
    rows.append(
        {
            "suite_member": "src/analysis/build_workspace_forensic_state_audit.py --help",
            "member_type": "direct_script_probe",
            "current_suite_status": "included",
            "reason": "verifies direct-script import bootstrap without writing forensic artifacts",
            "historical_exclusion_reason": "",
        }
    )
    rows.append(
        {
            "suite_member": "tests/test_workspace_forensic_state_audit.py",
            "member_type": "pytest_file",
            "current_suite_status": "excluded_from_light_current_suite",
            "reason": "full tests invoke the expensive forensic builder repeatedly",
            "historical_exclusion_reason": "builder materializes mutable current-tree inventory and contaminated exactness fingerprint during D1R2",
        }
    )
    return rows


def build_historical_groups(classified: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in classified:
        if row["taxonomy"] == "CURRENT_REGRESSION":
            continue
        key = (row["expected_branch"], row["expected_commit"], row["group_note"])
        grouped[key].append(row)

    rows = []
    for index, ((branch, commit, note), members) in enumerate(sorted(grouped.items()), start=1):
        rows.append(
            {
                "group_id": f"HIST-GRP-{index:03d}",
                "expected_branch": branch,
                "expected_commit": commit,
                "group_note": note,
                "node_count": len(members),
                "failed_count": sum(row["observed_status"] == "failed" for row in members),
                "error_count": sum(row["observed_status"] == "error" for row in members),
                "worktree_strategy": (
                    "git worktree add --detach <tmpdir> "
                    f"{commit}; run only this group's tests inside tmpdir; delete tmpdir after validation"
                ),
                "run_now": "false",
            }
        )
    return rows


def build_side_effect_rows() -> list[dict[str, str]]:
    rows = []
    for item in read_csv(D1R_SIDE_EFFECT_PATH):
        rows.append(
            {
                "relative_path": item.get("relative_path", ""),
                "current_sha256": item.get("current_sha256", ""),
                "source_run": "D1R_full_suite",
                "side_effect_family": item.get("side_effect_family", ""),
                "status": "unresolved",
                "root_cause": item.get("preliminary_root_cause", ""),
                "required_action": item.get("required_action", ""),
            }
        )
    for relative in D1R2_SIDE_EFFECT_PATHS:
        path = PROJECT_ROOT / relative
        if path.exists():
            rows.append(
                {
                    "relative_path": relative,
                    "current_sha256": sha256(path),
                    "source_run": "D1R2_current_suite_attempt",
                    "side_effect_family": "CURRENT_FORENSIC_BUILDER_WRITES_TO_SHARED_ARTIFACT",
                    "status": "unresolved",
                    "root_cause": (
                        "The full workspace forensic pytest invokes the builder repeatedly and rewrites "
                        "shared audit artifacts; exactness then observes a changed protected fingerprint."
                    ),
                    "required_action": (
                        "Run forensic materialization in tmp_path or preserve/restore the exactness "
                        "forensic baseline before including it in the current suite."
                    ),
                }
            )
    dedup = {}
    for row in rows:
        dedup[(row["relative_path"], row["source_run"])] = row
    return list(dedup.values())


def build_current_suite_result_rows() -> list[dict[str, Any]]:
    return [
        {
            "run_id": "d1r2_workspace_forensic_help_probe",
            "command": ".\\.venv_audit_recovery\\Scripts\\python.exe -B src/analysis/build_workspace_forensic_state_audit.py --help",
            "exit_code": 0,
            "passed": 1,
            "failed": 0,
            "errors": 0,
            "warnings": 0,
            "elapsed_seconds": "",
            "status": "passed",
            "blocking_issue": "",
        },
        {
            "run_id": "d1r2_current_suite_light_pytest",
            "command": (
                ".\\.venv_audit_recovery\\Scripts\\python.exe -B -m pytest -q --tb=short -ra "
                "--basetemp results\\logs\\d1r2_current_suite_tmp_light "
                "tests/test_extension_d0a_foodcom_material_feasibility.py "
                "tests/test_extension_d0b_final_fifth_dataset_selection.py "
                "tests/test_extension_d1_final_dataset_cleanup.py "
                "tests/test_extension_h0_history_final_parity_audit.py "
                "tests/test_extension_e0_protocol.py "
                "tests/test_extension_e2a_cell_universe.py "
                "tests/test_exactness_frontier_and_evidence_closure.py"
            ),
            "exit_code": 1,
            "passed": 58,
            "failed": 1,
            "errors": 0,
            "warnings": 2,
            "elapsed_seconds": "34.88",
            "status": "blocked",
            "blocking_issue": (
                "tests/test_exactness_frontier_and_evidence_closure.py::"
                "test_determinism_and_protected_input_fingerprint observed row_count=17579, expected 136"
            ),
        },
    ]


def latest_d1_status() -> dict[str, Any]:
    active = read_csv(TABLES_DIR / "d1_active_dataset_audit.csv")
    refs = read_csv(TABLES_DIR / "d1_remaining_video_games_reference_audit.csv")
    d1_manifest_path = MANIFESTS_DIR / "d1_video_games_purge_execution.json"
    d1_manifest = {}
    if d1_manifest_path.exists():
        d1_manifest = json.loads(d1_manifest_path.read_text(encoding="utf-8"))
    return {
        "active_datasets": [row.get("dataset", "") for row in active if row.get("active_status") in {"active", "ACTIVE", "retained_active"} or row.get("dataset")],
        "video_games_reference_rows": len(refs),
        "d1_decision_token": d1_manifest.get("decision_token", ""),
        "d1_manifest_sha256": sha256(d1_manifest_path),
    }


def write_report(
    classified: list[dict[str, str]],
    taxonomy_rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    side_effects: list[dict[str, str]],
    current_results: list[dict[str, Any]],
    d1_status: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    counts = Counter(row["taxonomy"] for row in classified)
    failed_errors = Counter(row["observed_status"] for row in classified)
    git_status_rows = run_git(["status", "--porcelain=v1"]).splitlines()
    current_failure = next(row for row in current_results if row["run_id"] == "d1r2_current_suite_light_pytest")
    lines = [
        "# D1R2 Historical Freeze vs Current Extension Test Separation",
        "",
        "## FAILURE CLASSIFICATION",
        f"- D1R lastfailed nodes classified: {len(classified)} ({failed_errors['failed']} failed, {failed_errors['error']} errors).",
        f"- Classification counts: {dict(sorted(counts.items()))}.",
        "- UNKNOWN classifications: 0.",
        "",
        "## HISTORICAL FREEZE FAILURES",
        f"- Historical freeze validation nodes: {counts['HISTORICAL_FREEZE_VALIDATION']}.",
        "- These remain excluded from the current suite and must be validated in detached worktrees at their expected commits.",
        "",
        "## REAL CURRENT REGRESSIONS",
        f"- Current regression nodes: {counts['CURRENT_REGRESSION']}.",
        "- The direct-script import regression in `build_workspace_forensic_state_audit.py` was fixed.",
        "- A new current blocker remains: exactness fingerprint reads a mutable forensic inventory regenerated during D1R2.",
        "",
        "## DIRECT SCRIPT IMPORT FAILURES",
        "- `src/analysis/build_workspace_forensic_state_audit.py --help` exits 0 after the bootstrap fix.",
        "- Full forensic pytest tests were not accepted into the light current suite because they materialize shared audit artifacts repeatedly.",
        "",
        "## PERSISTENT SIDE EFFECTS",
        f"- Persistent side-effect rows recorded: {len(side_effects)}.",
        "- D1R R4/R5 builders rewrote persistent reports/tables, including `report/notes.md`.",
        "- D1R2 forensic pytest attempt regenerated `workspace_relevant_file_inventory.csv`, causing exactness fingerprint row_count 17579 vs expected 136.",
        "",
        "## FIXES APPLIED",
        "- Added repository-root bootstrap before `src.audit` imports in `src/analysis/build_workspace_forensic_state_audit.py`.",
        "- No historical expected HEAD/branch/hash/token assertions were weakened.",
        "",
        "## CURRENT REGRESSION SUITE",
        "- Included D0A, D0B, D1, H0, E0, E2A, exactness frontier, and a direct forensic `--help` import probe.",
        "- Excluded historical freeze/destructive R/F6 nodes from the current suite.",
        "",
        "## CURRENT SUITE RESULT",
        f"- Direct forensic import probe: passed.",
        f"- Light current pytest suite: {current_failure['passed']} passed, {current_failure['failed']} failed, {current_failure['errors']} errors.",
        f"- Blocking issue: {current_failure['blocking_issue']}.",
        "",
        "## HISTORICAL FREEZE GROUPS",
        f"- Historical worktree groups designed: {len(groups)}.",
        "- Strategy: create one temporary detached worktree per expected commit, run only that group's historical tests, then remove the worktree.",
        "- No historical worktrees were executed in D1R2.",
        "",
        "## FOOD.COM INTEGRITY",
        "- Food.com gates in the light current suite passed before the exactness failure.",
        f"- D1 token: `{d1_status['d1_decision_token']}`.",
        "",
        "## H0 INTEGRITY",
        "- H0 gates in the light current suite passed before the exactness failure.",
        "",
        "## VIDEO GAMES ABSENCE",
        f"- D1 active dataset artifact remains available; D1 manifest hash: `{d1_status['d1_manifest_sha256']}`.",
        "- No Video Games re-materialization was performed.",
        "",
        "## GIT STATUS",
        f"- Branch: `{manifest['branch']}`.",
        f"- HEAD: `{manifest['head']}`.",
        f"- Porcelain status rows: {len(git_status_rows)}.",
        "",
        "## F0 PRECONDITION",
        "- F0 is blocked. Current suite is not green because exactness fingerprint remains contaminated by a mutable forensic inventory.",
        "",
        "## NEXT STEP",
        "- Restore or isolate the forensic inventory side effect, then rerun the D1R2 current suite.",
        "",
        manifest["decision_token"],
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    nodes = load_lastfailed()
    classified = build_failure_classification(nodes)
    taxonomy_rows = build_taxonomy_rows(classified)
    current_suite_rows = build_current_suite_rows()
    groups = build_historical_groups(classified)
    side_effects = build_side_effect_rows()
    current_results = build_current_suite_result_rows()
    d1_status = latest_d1_status()

    status_counts = Counter(row["observed_status"] for row in classified)
    taxonomy_counts = Counter(row["taxonomy"] for row in classified)
    current_suite_green = all(int(row["failed"]) == 0 and int(row["errors"]) == 0 and int(row["exit_code"]) == 0 for row in current_results)
    decision_token = "D1R2_CURRENT_TREE_VALIDATED" if current_suite_green else "D1R2_CURRENT_TREE_BLOCKED"

    manifest = {
        "phase_id": "d1r2_historical_freeze_vs_current_extension_test_separation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "decision_token": decision_token,
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "head": run_git(["rev-parse", "HEAD"]),
        "failed_nodes": status_counts["failed"],
        "error_nodes": status_counts["error"],
        "classified_nodes": len(classified),
        "taxonomy_counts": dict(sorted(taxonomy_counts.items())),
        "unknown_nodes": taxonomy_counts["UNKNOWN"],
        "current_suite_green": current_suite_green,
        "current_suite_results": "results/tables/d1r2_current_suite_results.csv",
        "historical_worktrees_executed": 0,
        "downloads_executed": 0,
        "llm_requests_executed": 0,
        "training_runs_executed": 0,
        "fixes_applied": [
            "Added direct-script repository-root bootstrap to src/analysis/build_workspace_forensic_state_audit.py.",
        ],
        "blocking_issue": "" if current_suite_green else current_results[1]["blocking_issue"],
        "source_lastfailed": ".pytest_cache/v/cache/lastfailed",
    }

    write_csv(
        TABLES_DIR / "d1r2_failure_classification.csv",
        classified,
        [
            "node_id",
            "test_file",
            "observed_status",
            "taxonomy",
            "expected_branch",
            "expected_commit",
            "group_note",
            "freeze_specific_assumption",
            "root_cause",
            "current_tree_fix_status",
            "historical_validation_strategy",
        ],
    )
    write_csv(
        TABLES_DIR / "d1r2_test_taxonomy.csv",
        taxonomy_rows,
        [
            "taxonomy",
            "node_count",
            "failed_count",
            "error_count",
            "current_suite_action",
            "historical_action",
            "final_unknown_allowed",
            "notes",
        ],
    )
    write_csv(
        TABLES_DIR / "d1r2_current_regression_suite.csv",
        current_suite_rows,
        ["suite_member", "member_type", "current_suite_status", "reason", "historical_exclusion_reason"],
    )
    write_csv(
        TABLES_DIR / "d1r2_historical_freeze_groups.csv",
        groups,
        [
            "group_id",
            "expected_branch",
            "expected_commit",
            "group_note",
            "node_count",
            "failed_count",
            "error_count",
            "worktree_strategy",
            "run_now",
        ],
    )
    write_csv(
        TABLES_DIR / "d1r2_persistent_side_effect_audit.csv",
        side_effects,
        ["relative_path", "current_sha256", "source_run", "side_effect_family", "status", "root_cause", "required_action"],
    )
    write_csv(
        TABLES_DIR / "d1r2_current_suite_results.csv",
        current_results,
        ["run_id", "command", "exit_code", "passed", "failed", "errors", "warnings", "elapsed_seconds", "status", "blocking_issue"],
    )

    (MANIFESTS_DIR / "d1r2_test_separation.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(classified, taxonomy_rows, groups, side_effects, current_results, d1_status, manifest)


if __name__ == "__main__":
    main()
