from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.plan_e_baseline_alignment import (
    MANIFEST_INVENTORY_COLUMNS,
    MISSING_BASELINE_COLUMNS,
    NEXT_ACTION_COLUMNS,
    NO_TRAINING_METRIC_COLUMNS,
    SCORE_AVAILABILITY_COLUMNS,
    SOURCE_INVENTORY_COLUMNS,
    VALIDATION_COLUMNS,
    PlanEBaselineAlignmentPaths,
    build_all_tables,
    build_markdown_report,
    write_baseline_alignment_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "src/experiments/build_plan_e_baseline_alignment.py"
RAW_OUTPUTS_DIR = PROJECT_ROOT / "results/raw_outputs"

REQUIRED_CHECKS = {
    "all_E2_manifests_present",
    "manifest_schema_valid",
    "candidate_pool_size_20",
    "one_positive_per_user",
    "nineteen_negatives_per_user",
    "no_duplicate_items_per_user",
    "metric_formula_hr_at_5_available",
    "metric_formula_ndcg_at_5_available",
    "candidate_order_reference_metrics_created",
    "random_deterministic_reference_metrics_created",
    "all_paper_baselines_represented",
    "no_neural_training_performed",
    "no_llm_inference_performed",
    "no_ollama_call_performed",
    "no_api_call_performed",
    "no_download_performed",
    "no_lastfm_run_performed",
    "no_raw_outputs_created",
    "no_original_repo_output",
    "no_plan_c_modification",
    "no_exact_reproduction_overclaim",
}


@pytest.fixture(scope="module")
def baseline_tables() -> dict[str, pd.DataFrame]:
    return build_all_tables(PROJECT_ROOT)


def test_all_required_output_files_can_be_generated(tmp_path: Path, baseline_tables: dict[str, pd.DataFrame]) -> None:
    paths = _tmp_paths(tmp_path)
    write_baseline_alignment_outputs(baseline_tables, paths)

    for output_path in paths.required_output_paths():
        assert output_path.exists(), output_path

    assert set(MANIFEST_INVENTORY_COLUMNS).issubset(pd.read_csv(paths.manifest_inventory).columns)
    assert set(SOURCE_INVENTORY_COLUMNS).issubset(pd.read_csv(paths.source_inventory).columns)
    assert set(SCORE_AVAILABILITY_COLUMNS).issubset(pd.read_csv(paths.score_availability).columns)
    assert set(NO_TRAINING_METRIC_COLUMNS).issubset(pd.read_csv(paths.no_training_metrics).columns)
    assert set(MISSING_BASELINE_COLUMNS).issubset(pd.read_csv(paths.missing_baselines).columns)
    assert set(VALIDATION_COLUMNS).issubset(pd.read_csv(paths.validation).columns)
    assert set(NEXT_ACTION_COLUMNS).issubset(pd.read_csv(paths.next_actions).columns)


def test_manifest_inventory_includes_all_six_e2_manifests(baseline_tables: dict[str, pd.DataFrame]) -> None:
    manifest_inventory = baseline_tables["manifest_inventory"]

    assert len(manifest_inventory) == 6
    assert set(manifest_inventory["dataset"]) == {"ML-1M", "Amazon Beauty", "Amazon Sports"}
    assert set(manifest_inventory["task"]) == {"ranking", "reranking"}
    assert (manifest_inventory["exists"] == "true").all()


def test_manifest_inventory_confirms_complete_e2_shape(baseline_tables: dict[str, pd.DataFrame]) -> None:
    manifest_inventory = baseline_tables["manifest_inventory"]

    assert set(manifest_inventory["user_count"].astype(int)) == {1000}
    assert set(manifest_inventory["rows"].astype(int)) == {20000}
    assert set(manifest_inventory["candidate_pool_size"].astype(int)) == {20}
    assert set(manifest_inventory["positive_items_per_user"].astype(int)) == {1}
    assert set(manifest_inventory["negative_items_per_user"].astype(int)) == {19}
    assert (manifest_inventory["exact_1000_user_target_met"] == "true").all()
    assert (manifest_inventory["exact_candidate_pool_20_met"] == "true").all()
    assert (manifest_inventory["exact_1_positive_19_negative_met"] == "true").all()


def test_score_availability_represents_all_paper_baselines(baseline_tables: dict[str, pd.DataFrame]) -> None:
    score_availability = baseline_tables["score_availability"]
    expected = {"BM25", "MostPop", "BPRMF", "GRU4Rec", "SASRec", "LightGCN"}

    assert expected.issubset(set(score_availability["model_or_method"]))
    for method in expected:
        method_rows = score_availability.loc[score_availability["model_or_method"] == method]
        assert len(method_rows) == 6
        assert (method_rows["required_for_paper"] == "true").all()


def test_reference_metrics_and_mostpop_metrics_are_created(baseline_tables: dict[str, pd.DataFrame]) -> None:
    metrics = baseline_tables["no_training_metrics"]

    assert (metrics["model_or_method"] == "candidate_order_reference").sum() == 6
    assert (metrics["model_or_method"] == "random_deterministic_reference").sum() == 6
    assert (metrics["model_or_method"] == "MostPop").sum() == 6
    assert {"hr_at_5", "ndcg_at_5"}.issubset(metrics.columns)


def test_hr_and_ndcg_values_are_valid_probabilities(baseline_tables: dict[str, pd.DataFrame]) -> None:
    metrics = baseline_tables["no_training_metrics"]

    assert ((metrics["hr_at_5"].astype(float) >= 0.0) & (metrics["hr_at_5"].astype(float) <= 1.0)).all()
    assert ((metrics["ndcg_at_5"].astype(float) >= 0.0) & (metrics["ndcg_at_5"].astype(float) <= 1.0)).all()
    assert (metrics["evaluated_users"].astype(int) == 1000).all()
    assert (metrics["missing_users"].astype(int) == 0).all()


def test_missing_baselines_table_is_non_empty_when_aligned_neural_scores_absent(baseline_tables: dict[str, pd.DataFrame]) -> None:
    missing = baseline_tables["missing_baselines"]

    assert not missing.empty
    assert {"BM25", "BPRMF", "GRU4Rec", "SASRec", "LightGCN"}.issubset(set(missing["model_or_method"]))
    assert (missing.loc[missing["model_or_method"].isin(["BPRMF", "GRU4Rec", "SASRec", "LightGCN"]), "classification"] == "needs_training_or_existing_scores").all()


def test_validation_table_contains_required_checks_and_passes(baseline_tables: dict[str, pd.DataFrame]) -> None:
    validation = baseline_tables["validation"]

    assert REQUIRED_CHECKS.issubset(set(validation["check_name"]))
    assert set(validation["passed"]) == {"true"}


def test_report_contains_required_boundaries_and_claim_safety(baseline_tables: dict[str, pd.DataFrame]) -> None:
    report = build_markdown_report(baseline_tables).lower()

    for required in (
        "e3a is not full e3 baseline completion",
        "e3a is a no-training alignment/evaluation audit",
        "e3a performed no inference",
        "e3a made no ollama calls",
        "e3a made no qwen2/mistral calls",
        "e3a made no api calls",
        "e3a performed no downloads",
        "e3a performed no model pulls",
        "e3a performed no training",
        "e3a performed no neural-baseline training",
        "e3a performed no lastfm run",
        "e3a performed no plan c modification",
        "e3a performed no original_repo modification",
        "e3a does not complete full paper baseline reproduction",
        "candidate_order_reference and random_deterministic_reference are sanity checks, not paper baselines",
        "mostpop is a paper baseline here because it is computed from valid available train interactions without model training",
        "neural baselines require aligned scores or future approved scoring/training",
        "lastfm remains blocked/conditional",
        "no full exact reproduction is claimed",
    ):
        assert required in report

    for unsafe in (
        "we achieved exact reproduction",
        "full exact reproduction is complete",
        "neural baseline training was run",
        "llm inference was run",
    ):
        assert unsafe not in report


def test_outputs_are_deterministic_across_repeated_writes(tmp_path: Path, baseline_tables: dict[str, pd.DataFrame]) -> None:
    first = _tmp_paths(tmp_path / "first")
    second = _tmp_paths(tmp_path / "second")

    write_baseline_alignment_outputs(baseline_tables, first)
    write_baseline_alignment_outputs(baseline_tables, second)

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
    paths = PlanEBaselineAlignmentPaths()
    for output_path in paths.required_output_paths():
        relative = output_path.relative_to(PROJECT_ROOT).as_posix()
        assert not relative.startswith("original_repo/")


def test_plan_e_baseline_alignment_sources_do_not_invoke_model_api_download_or_training_code() -> None:
    source_text = (
        (PROJECT_ROOT / "src/analysis/plan_e_baseline_alignment.py").read_text(encoding="utf-8")
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


def _tmp_paths(root: Path) -> PlanEBaselineAlignmentPaths:
    tables = root / "results" / "tables"
    report = root / "report"
    return PlanEBaselineAlignmentPaths(
        manifest_inventory=tables / "plan_e_baseline_alignment_manifest_inventory.csv",
        source_inventory=tables / "plan_e_baseline_alignment_source_inventory.csv",
        score_availability=tables / "plan_e_baseline_alignment_score_availability.csv",
        no_training_metrics=tables / "plan_e_baseline_alignment_no_training_metrics.csv",
        missing_baselines=tables / "plan_e_baseline_alignment_missing_baselines.csv",
        validation=tables / "plan_e_baseline_alignment_validation.csv",
        next_actions=tables / "plan_e_baseline_alignment_next_actions.csv",
        report=report / "plan_e_baseline_alignment.md",
    )


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
    plan_c_data = PROJECT_ROOT / "data/reconstructed/plan_c"
    if plan_c_data.exists():
        paths.extend(path for path in plan_c_data.rglob("*") if path.is_file())
    return {path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path) for path in sorted(paths)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
