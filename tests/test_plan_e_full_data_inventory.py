from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.plan_e_full_data_inventory import (
    ARTIFACT_SUMMARY_COLUMNS,
    CALL_ESTIMATE_COLUMNS,
    DATASET_SUMMARY_COLUMNS,
    MANIFEST_SUMMARY_COLUMNS,
    METRIC_SUMMARY_COLUMNS,
    NEXT_ACTION_COLUMNS,
    RAW_OUTPUT_HYGIENE_COLUMNS,
    RUNTIME_ESTIMATE_COLUMNS,
    SCALEUP_READINESS_COLUMNS,
    PlanEInventoryPaths,
    build_all_tables,
    build_markdown_report,
    write_inventory_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "src/experiments/build_plan_e_full_data_inventory.py"
RAW_OUTPUTS_DIR = PROJECT_ROOT / "results/raw_outputs"


@pytest.fixture(scope="module")
def inventory_tables() -> dict[str, pd.DataFrame]:
    return build_all_tables(PROJECT_ROOT)


def test_all_required_output_files_can_be_generated(tmp_path: Path, inventory_tables: dict[str, pd.DataFrame]) -> None:
    paths = _tmp_paths(tmp_path)
    write_inventory_outputs(inventory_tables, paths)

    for output_path in paths.required_output_paths():
        assert output_path.exists(), output_path

    assert set(DATASET_SUMMARY_COLUMNS).issubset(pd.read_csv(paths.dataset_summary).columns)
    assert set(ARTIFACT_SUMMARY_COLUMNS).issubset(pd.read_csv(paths.artifact_summary).columns)
    assert set(MANIFEST_SUMMARY_COLUMNS).issubset(pd.read_csv(paths.manifest_summary).columns)
    assert set(METRIC_SUMMARY_COLUMNS).issubset(pd.read_csv(paths.metric_summary).columns)
    assert set(RAW_OUTPUT_HYGIENE_COLUMNS).issubset(pd.read_csv(paths.raw_output_hygiene).columns)
    assert set(SCALEUP_READINESS_COLUMNS).issubset(pd.read_csv(paths.scaleup_readiness).columns)
    assert set(CALL_ESTIMATE_COLUMNS).issubset(pd.read_csv(paths.call_estimates).columns)
    assert set(RUNTIME_ESTIMATE_COLUMNS).issubset(pd.read_csv(paths.runtime_estimates).columns)
    assert set(NEXT_ACTION_COLUMNS).issubset(pd.read_csv(paths.next_actions).columns)


def test_generated_csvs_are_non_empty_where_expected(inventory_tables: dict[str, pd.DataFrame]) -> None:
    for key in (
        "dataset_summary",
        "artifact_summary",
        "manifest_summary",
        "metric_summary",
        "raw_output_hygiene",
        "scaleup_readiness",
        "call_estimates",
        "runtime_estimates",
        "next_actions",
    ):
        assert not inventory_tables[key].empty, key


def test_dataset_summary_includes_all_paper_datasets(inventory_tables: dict[str, pd.DataFrame]) -> None:
    datasets = set(inventory_tables["dataset_summary"]["dataset"])

    assert {"ML-1M", "Amazon Beauty", "Amazon Sports", "LastFM"}.issubset(datasets)


def test_manifest_summary_represents_candidate_pool_20_or_missing_status(inventory_tables: dict[str, pd.DataFrame]) -> None:
    manifest_summary = inventory_tables["manifest_summary"]
    readiness = inventory_tables["scaleup_readiness"]

    if (manifest_summary["candidate_pool_size_if_available"].astype(str) == "20").any():
        matches = manifest_summary.loc[manifest_summary["candidate_pool_size_if_available"].astype(str) == "20"]
        assert (matches["matches_paper_candidate_pool_20"] == "true").any()
        assert (matches["matches_1_positive_19_negative"] == "true").any()
    else:
        assert (readiness["current_manifest_status"] == "missing").any()


def test_scaleup_readiness_includes_paper_candidate_requirements(inventory_tables: dict[str, pd.DataFrame]) -> None:
    readiness = inventory_tables["scaleup_readiness"]

    assert 1000 in set(readiness["target_users"].astype(int))
    assert set(readiness["candidate_pool_size"].astype(int)) == {20}
    assert set(readiness["required_positive_per_user"].astype(int)) == {1}
    assert set(readiness["required_negative_per_user"].astype(int)) == {19}


def test_call_estimates_include_local_and_paper_matrix_scenarios(inventory_tables: dict[str, pd.DataFrame]) -> None:
    calls = inventory_tables["call_estimates"]
    scenario_ids = set(calls["scenario_id"])
    text = _frame_text(calls)

    assert "local_Qwen2_Mistral_20_users_ranking" in scenario_ids
    assert "local_Qwen2_Mistral_100_users_ranking" in scenario_ids
    assert "local_Qwen2_Mistral_1000_users_ranking_ML1M" in scenario_ids
    assert "local_Qwen2_Mistral_1000_users_ranking_3_valid_datasets" in scenario_ids
    assert "paper_7LLM_1000_users_3_valid_datasets_ranking" in scenario_ids
    assert "paper_7LLM_3_prompt_strategies_1000_users_3_valid_datasets_ranking_plus_reranking" in scenario_ids
    assert "LastFM".lower() in text
    assert "conditional" in text


def test_runtime_estimates_are_assumed_not_measured(inventory_tables: dict[str, pd.DataFrame]) -> None:
    runtime = inventory_tables["runtime_estimates"]

    assert set(runtime["assumed_seconds_per_call"].astype(int)) == {10, 20, 45, 90}
    assert set(runtime["measured_or_assumed"]) == {"assumed_planning_estimate"}
    assert (runtime["estimated_total_calls"].astype(int) > 0).all()


def test_metric_summary_represents_plan_c_plan_d_d5_and_e0(inventory_tables: dict[str, pd.DataFrame]) -> None:
    metrics = inventory_tables["metric_summary"]
    text = _frame_text(metrics)

    assert "plan c" in text
    assert "plan d/d5" in text
    assert "diagnostic_subset" in text
    assert "extended_subset" in text
    assert "e0" in text
    assert "reconstructed reproduction" in text


def test_raw_output_hygiene_table_exists_and_checks_git_boundaries(inventory_tables: dict[str, pd.DataFrame]) -> None:
    hygiene = inventory_tables["raw_output_hygiene"]
    check_names = set(hygiene["check_name"])

    assert "results_raw_outputs_exists" in check_names
    assert "tracked_jsonl_files_under_results_raw_outputs" in check_names
    assert "untracked_forbidden_file_patterns" in check_names
    assert "original_repo_git_status" in check_names
    assert "plan_c_git_status" in check_names
    assert hygiene.loc[hygiene["check_name"] == "tracked_jsonl_files_under_results_raw_outputs"].iloc[0]["status"] == "passed"


def test_report_contains_required_boundaries_and_claim_safety(inventory_tables: dict[str, pd.DataFrame]) -> None:
    report = build_markdown_report(inventory_tables).lower()

    for required in (
        "e1 performed no inference",
        "e1 made no ollama calls",
        "e1 made no qwen2/mistral calls",
        "e1 made no api calls",
        "e1 performed no downloads",
        "e1 performed no model pulls",
        "e1 performed no training",
        "e1 performed no lastfm run",
        "e1 performed no plan c modification",
        "e1 performed no original_repo modification",
        "e1 is an inventory and estimation phase only",
        "e1 does not prove that 1000-user paper-like evaluation is complete",
        "e1 prepares e2 by identifying what manifests need to be built",
        "lastfm remains conditional/blocked unless valid utility-evaluation artifacts are found",
        "local qwen2/mistral status remains local diagnostic, not paper-exact",
        "closed-source model costs are not verified in e1",
        "api pricing verification belongs to e6",
        "no full exact reproduction is claimed",
    ):
        assert required in report

    for unsafe_claim in (
        "we achieved exact reproduction",
        "full exact reproduction is complete",
        "we completed 1000-user paper-like evaluation",
        "the 1000-user paper-like evaluation is complete",
        "lastfm is reproduced",
    ):
        assert unsafe_claim not in report


def test_outputs_are_deterministic_across_repeated_builds(tmp_path: Path, inventory_tables: dict[str, pd.DataFrame]) -> None:
    first = _tmp_paths(tmp_path / "first")
    second = _tmp_paths(tmp_path / "second")

    write_inventory_outputs(inventory_tables, first)
    write_inventory_outputs(inventory_tables, second)

    for first_path, second_path in zip(first.required_output_paths(), second.required_output_paths()):
        assert first_path.read_bytes() == second_path.read_bytes(), first_path.name


def test_build_script_does_not_create_raw_jsonl_or_modify_plan_c_files() -> None:
    before_raw_jsonl = _raw_jsonl_files()
    before_plan_c_hashes = _plan_c_hashes()

    completed = subprocess.run(
        [sys.executable, str(RUNNER_PATH)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert "no inference" in completed.stdout.lower()
    assert _raw_jsonl_files() == before_raw_jsonl
    assert _plan_c_hashes() == before_plan_c_hashes


def test_no_output_path_is_under_original_repo() -> None:
    paths = PlanEInventoryPaths()
    for output_path in paths.required_output_paths():
        relative = output_path.relative_to(PROJECT_ROOT).as_posix()
        assert not relative.startswith("original_repo/")


def test_plan_e_inventory_sources_do_not_invoke_model_api_download_or_training_code() -> None:
    source_text = (
        (PROJECT_ROOT / "src/analysis/plan_e_full_data_inventory.py").read_text(encoding="utf-8")
        + "\n"
        + RUNNER_PATH.read_text(encoding="utf-8")
    ).lower()

    for forbidden in (
        "ollama run",
        "ollama.chat",
        "ollama.generate",
        "openai.",
        "anthropic.",
        "requests.get",
        "requests.post",
        "urlopen(",
        "hf_hub_download",
        "snapshot_download",
        ".fit(",
        ".train(",
    ):
        assert forbidden not in source_text


def _tmp_paths(root: Path) -> PlanEInventoryPaths:
    tables = root / "results" / "tables"
    report = root / "report"
    return PlanEInventoryPaths(
        dataset_summary=tables / "plan_e_full_data_inventory_dataset_summary.csv",
        artifact_summary=tables / "plan_e_full_data_inventory_artifact_summary.csv",
        manifest_summary=tables / "plan_e_full_data_inventory_manifest_summary.csv",
        metric_summary=tables / "plan_e_full_data_inventory_metric_summary.csv",
        raw_output_hygiene=tables / "plan_e_full_data_inventory_raw_output_hygiene.csv",
        scaleup_readiness=tables / "plan_e_full_data_inventory_scaleup_readiness.csv",
        call_estimates=tables / "plan_e_full_data_inventory_call_estimates.csv",
        runtime_estimates=tables / "plan_e_full_data_inventory_runtime_estimates.csv",
        next_actions=tables / "plan_e_full_data_inventory_next_actions.csv",
        report=report / "plan_e_full_data_inventory.md",
    )


def _frame_text(frame: pd.DataFrame) -> str:
    return " ".join(frame.astype(str).to_numpy().ravel()).lower()


def _raw_jsonl_files() -> set[str]:
    if not RAW_OUTPUTS_DIR.exists():
        return set()
    return {path.relative_to(PROJECT_ROOT).as_posix() for path in RAW_OUTPUTS_DIR.glob("*.jsonl")}


def _plan_c_hashes() -> dict[str, str]:
    paths: list[Path] = []
    for root in ("results", "report", "configs", "src", "tests"):
        base = PROJECT_ROOT / root
        if not base.exists():
            continue
        paths.extend(path for path in base.rglob("*plan_c*") if path.is_file())
    return {path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path) for path in sorted(paths)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
