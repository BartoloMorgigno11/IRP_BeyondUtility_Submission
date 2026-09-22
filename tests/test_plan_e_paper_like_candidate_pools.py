from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.plan_e_paper_like_candidate_pools import (
    BLOCKED_COLUMNS,
    DATASET_ELIGIBILITY_COLUMNS,
    MANIFEST_SCHEMA_COLUMNS,
    MANIFEST_SUMMARY_COLUMNS,
    NEXT_ACTION_COLUMNS,
    SOURCE_AUDIT_COLUMNS,
    VALIDATION_COLUMNS,
    PlanECandidatePoolPaths,
    build_candidate_pool_outputs,
    build_markdown_report,
    write_candidate_pool_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "src/experiments/build_plan_e_paper_like_candidate_pools.py"
RAW_OUTPUTS_DIR = PROJECT_ROOT / "results/raw_outputs"

REQUIRED_VALIDATION_CHECKS = {
    "file_exists",
    "row_count_is_20000_when_complete",
    "user_count_is_1000_when_complete",
    "candidate_pool_size_is_20",
    "one_positive_per_user",
    "nineteen_negatives_per_user",
    "no_duplicate_items_per_user",
    "candidate_positions_are_1_to_20",
    "no_missing_user_ids",
    "no_missing_item_ids",
    "deterministic_seed_recorded",
    "no_LastFM_manifest_unless_valid",
    "no_raw_outputs_created",
    "no_original_repo_output",
    "no_plan_c_modification",
}


@pytest.fixture(scope="module")
def candidate_pool_tables() -> dict[str, object]:
    return build_candidate_pool_outputs(PROJECT_ROOT, PlanECandidatePoolPaths())


def test_all_required_output_files_can_be_generated(tmp_path: Path, candidate_pool_tables: dict[str, object]) -> None:
    paths = _tmp_paths(tmp_path)
    write_candidate_pool_outputs(candidate_pool_tables, paths)

    for output_path in paths.required_output_paths():
        assert output_path.exists(), output_path

    assert set(DATASET_ELIGIBILITY_COLUMNS).issubset(pd.read_csv(paths.dataset_eligibility).columns)
    assert set(MANIFEST_SUMMARY_COLUMNS).issubset(pd.read_csv(paths.manifest_summary).columns)
    assert set(VALIDATION_COLUMNS).issubset(pd.read_csv(paths.validation).columns)
    assert set(SOURCE_AUDIT_COLUMNS).issubset(pd.read_csv(paths.source_audit).columns)
    assert set(BLOCKED_COLUMNS).issubset(pd.read_csv(paths.blocked_datasets).columns)
    assert set(NEXT_ACTION_COLUMNS).issubset(pd.read_csv(paths.next_actions).columns)
    for manifest_path in paths.manifest_paths().values():
        assert set(MANIFEST_SCHEMA_COLUMNS).issubset(pd.read_csv(manifest_path, nrows=1).columns)


def test_dataset_eligibility_includes_all_required_datasets(candidate_pool_tables: dict[str, object]) -> None:
    eligibility = _frame(candidate_pool_tables["dataset_eligibility"])
    datasets = set(eligibility["dataset"])

    assert {"ML-1M", "Amazon Beauty", "Amazon Sports", "LastFM"}.issubset(datasets)
    lastfm = eligibility.loc[eligibility["dataset"] == "LastFM"].iloc[0]
    assert lastfm["eligibility_class"] == "blocked/conditional"
    assert lastfm["can_build_1000_user_manifest"] == "false"


def test_valid_dataset_manifests_exist_when_enough_eligible_users(candidate_pool_tables: dict[str, object]) -> None:
    eligibility = _frame(candidate_pool_tables["dataset_eligibility"])
    summary = _frame(candidate_pool_tables["manifest_summary"])

    for dataset in ("ML-1M", "Amazon Beauty", "Amazon Sports"):
        row = eligibility.loc[eligibility["dataset"] == dataset].iloc[0]
        assert int(row["eligible_user_count"]) >= 1000
        dataset_summary = summary.loc[summary["dataset"] == dataset]
        assert set(dataset_summary["task"]) == {"ranking", "reranking"}
        assert (dataset_summary["exact_1000_user_target_met"] == "true").all()


def test_complete_manifests_have_1000_users_and_20000_rows(candidate_pool_tables: dict[str, object]) -> None:
    summary = _frame(candidate_pool_tables["manifest_summary"])

    assert len(summary) == 6
    assert set(summary["rows"].astype(int)) == {20000}
    assert set(summary["user_count"].astype(int)) == {1000}


def test_each_selected_user_has_20_candidates_1_positive_19_negatives(candidate_pool_tables: dict[str, object]) -> None:
    manifests = candidate_pool_tables["manifests"]
    assert isinstance(manifests, dict)
    for frame in manifests.values():
        assert set(frame["candidate_pool_size"].astype(int)) == {20}
        per_user = frame.groupby("user_id").size()
        positives = frame.loc[frame["is_positive"] == "true"].groupby("user_id").size()
        negatives = frame.loc[frame["is_positive"] == "false"].groupby("user_id").size()
        assert set(per_user.astype(int)) == {20}
        assert set(positives.astype(int)) == {1}
        assert set(negatives.astype(int)) == {19}


def test_candidate_positions_are_1_to_20_and_no_duplicate_items(candidate_pool_tables: dict[str, object]) -> None:
    manifests = candidate_pool_tables["manifests"]
    assert isinstance(manifests, dict)
    expected_positions = list(range(1, 21))
    for frame in manifests.values():
        positions = frame.groupby("user_id")["candidate_position"].apply(lambda values: sorted(values.astype(int).tolist()))
        unique_items = frame.groupby("user_id")["item_id"].nunique()
        assert all(value == expected_positions for value in positions)
        assert set(unique_items.astype(int)) == {20}


def test_candidate_pool_size_and_positive_negative_requirements_are_present(candidate_pool_tables: dict[str, object]) -> None:
    summary = _frame(candidate_pool_tables["manifest_summary"])
    eligibility = _frame(candidate_pool_tables["dataset_eligibility"])

    assert set(summary["candidate_pool_size"].astype(int)) == {20}
    assert set(summary["positive_items_per_user"].astype(int)) == {1}
    assert set(summary["negative_items_per_user"].astype(int)) == {19}
    assert set(eligibility["candidate_pool_size"].astype(int)) == {20}
    assert set(eligibility["required_positive_per_user"].astype(int)) == {1}
    assert set(eligibility["required_negative_per_user"].astype(int)) == {19}


def test_lastfm_is_not_generated_unless_explicitly_valid(candidate_pool_tables: dict[str, object]) -> None:
    summary = _frame(candidate_pool_tables["manifest_summary"])
    blocked = _frame(candidate_pool_tables["blocked_datasets"])
    manifests = candidate_pool_tables["manifests"]

    assert isinstance(manifests, dict)
    assert "LastFM" not in set(summary["dataset"])
    assert all(key[0] != "lastfm" for key in manifests)
    assert not blocked.empty
    assert set(blocked["dataset"]) == {"LastFM"}
    assert (blocked["blocked_or_conditional"] == "blocked/conditional").all()


def test_validation_table_contains_required_checks_and_passes(candidate_pool_tables: dict[str, object]) -> None:
    validation = _frame(candidate_pool_tables["validation"])

    assert REQUIRED_VALIDATION_CHECKS.issubset(set(validation["check_name"]))
    assert set(validation["passed"]) == {"true"}


def test_report_contains_required_boundaries_and_claim_safety(candidate_pool_tables: dict[str, object]) -> None:
    report = build_markdown_report(candidate_pool_tables).lower()

    for required in (
        "e2 constructs paper-like candidate manifests, not llm results",
        "e2 performed no inference",
        "e2 made no ollama calls",
        "e2 made no qwen2/mistral calls",
        "e2 made no api calls",
        "e2 performed no downloads",
        "e2 performed no model pulls",
        "e2 performed no training",
        "e2 performed no lastfm run",
        "e2 performed no plan c modification",
        "e2 performed no original_repo modification",
        "e2 does not run recommendation training",
        "e2 does not run llm inference",
        "e2 does not complete full paper reproduction",
        "generated manifests are reconstructed paper-like artifacts unless exact original paper candidate files are verified",
        "strict prompt remains a plan d/e extension, not a paper-exact prompt strategy",
        "lastfm remains conditional/blocked unless valid utility artifacts are found",
        "no full exact reproduction is claimed",
    ):
        assert required in report

    for unsafe in (
        "we achieved exact reproduction",
        "full exact reproduction is complete",
        "llm results were generated",
        "lastfm is reproduced",
    ):
        assert unsafe not in report


def test_outputs_are_deterministic_across_repeated_writes(tmp_path: Path, candidate_pool_tables: dict[str, object]) -> None:
    first = _tmp_paths(tmp_path / "first")
    second = _tmp_paths(tmp_path / "second")

    write_candidate_pool_outputs(candidate_pool_tables, first)
    write_candidate_pool_outputs(candidate_pool_tables, second)

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
        timeout=300,
    )

    assert "no inference" in completed.stdout.lower()
    assert _raw_jsonl_files() == before_raw_jsonl
    assert _plan_c_hashes() == before_plan_c_hashes


def test_no_output_path_is_under_original_repo() -> None:
    paths = PlanECandidatePoolPaths()
    for output_path in paths.required_output_paths():
        relative = output_path.relative_to(PROJECT_ROOT).as_posix()
        assert not relative.startswith("original_repo/")


def test_plan_e_candidate_pool_sources_do_not_invoke_model_api_download_or_training_code() -> None:
    source_text = (
        (PROJECT_ROOT / "src/analysis/plan_e_paper_like_candidate_pools.py").read_text(encoding="utf-8")
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


def _tmp_paths(root: Path) -> PlanECandidatePoolPaths:
    tables = root / "results" / "tables"
    report = root / "report"
    return PlanECandidatePoolPaths(
        dataset_eligibility=tables / "plan_e_paper_like_candidate_pool_dataset_eligibility.csv",
        manifest_summary=tables / "plan_e_paper_like_candidate_pool_manifest_summary.csv",
        validation=tables / "plan_e_paper_like_candidate_pool_validation.csv",
        source_audit=tables / "plan_e_paper_like_candidate_pool_source_audit.csv",
        blocked_datasets=tables / "plan_e_paper_like_candidate_pool_blocked_datasets.csv",
        next_actions=tables / "plan_e_paper_like_candidate_pool_next_actions.csv",
        ml1m_ranking_manifest=tables / "plan_e_ml1m_1000user_ranking_manifest.csv",
        ml1m_reranking_manifest=tables / "plan_e_ml1m_1000user_reranking_manifest.csv",
        beauty_ranking_manifest=tables / "plan_e_beauty_1000user_ranking_manifest.csv",
        beauty_reranking_manifest=tables / "plan_e_beauty_1000user_reranking_manifest.csv",
        sports_ranking_manifest=tables / "plan_e_sports_1000user_ranking_manifest.csv",
        sports_reranking_manifest=tables / "plan_e_sports_1000user_reranking_manifest.csv",
        report=report / "plan_e_paper_like_candidate_pool_construction.md",
    )


def _frame(value: object) -> pd.DataFrame:
    assert isinstance(value, pd.DataFrame)
    return value


def _raw_jsonl_files() -> set[str]:
    if not RAW_OUTPUTS_DIR.exists():
        return set()
    return {path.relative_to(PROJECT_ROOT).as_posix() for path in RAW_OUTPUTS_DIR.glob("*.jsonl")}


def _plan_c_hashes() -> dict[str, str]:
    paths: list[Path] = []
    for root in ("results", "report", "configs", "src", "tests", "data/reconstructed/plan_c"):
        base = PROJECT_ROOT / root
        if not base.exists():
            continue
        paths.extend(path for path in base.rglob("*plan_c*") if path.is_file())
    return {path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path) for path in sorted(paths)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
