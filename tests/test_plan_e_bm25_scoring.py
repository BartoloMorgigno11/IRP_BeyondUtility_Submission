from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.plan_e_bm25_scoring import (
    METRICS_COLUMNS,
    MISSING_COLUMNS,
    NEXT_ACTION_COLUMNS,
    QUERY_CORPUS_AUDIT_COLUMNS,
    SCORES_COLUMNS,
    SOURCE_INVENTORY_COLUMNS,
    VALIDATION_COLUMNS,
    PlanEBM25ScoringPaths,
    build_all_tables,
    build_markdown_report,
    write_bm25_scoring_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "src/experiments/build_plan_e_bm25_scoring.py"
RAW_OUTPUTS_DIR = PROJECT_ROOT / "results/raw_outputs"

REQUIRED_VALIDATION_CHECKS = {
    "all_E2_manifests_present",
    "item_text_sources_discovered_or_blocked",
    "user_query_sources_discovered_or_blocked",
    "no_training_performed",
    "no_neural_training_performed",
    "no_llm_inference_performed",
    "no_ollama_call_performed",
    "no_api_call_performed",
    "no_download_performed",
    "no_lastfm_run_performed",
    "no_raw_outputs_created",
    "no_original_repo_output",
    "no_plan_c_modification",
    "bm25_scores_schema_valid_if_created",
    "bm25_metrics_schema_valid",
    "hr_at_5_between_0_and_1",
    "ndcg_at_5_between_0_and_1",
    "no_exact_reproduction_overclaim",
}


@pytest.fixture(scope="module")
def bm25_tables() -> dict[str, pd.DataFrame]:
    return build_all_tables(PROJECT_ROOT)


def test_all_required_output_files_can_be_generated(tmp_path: Path, bm25_tables: dict[str, pd.DataFrame]) -> None:
    paths = _tmp_paths(tmp_path)
    write_bm25_scoring_outputs(bm25_tables, paths)

    for output_path in paths.required_output_paths():
        assert output_path.exists(), output_path

    assert set(SOURCE_INVENTORY_COLUMNS).issubset(pd.read_csv(paths.source_inventory).columns)
    assert set(QUERY_CORPUS_AUDIT_COLUMNS).issubset(pd.read_csv(paths.query_corpus_audit).columns)
    assert set(SCORES_COLUMNS).issubset(pd.read_csv(paths.scores, nrows=1).columns)
    assert set(METRICS_COLUMNS).issubset(pd.read_csv(paths.metrics).columns)
    assert set(VALIDATION_COLUMNS).issubset(pd.read_csv(paths.validation).columns)
    assert set(MISSING_COLUMNS).issubset(pd.read_csv(paths.missing_or_blocked).columns)
    assert set(NEXT_ACTION_COLUMNS).issubset(pd.read_csv(paths.next_actions).columns)


def test_all_six_e2_manifests_are_represented(bm25_tables: dict[str, pd.DataFrame]) -> None:
    audit = bm25_tables["query_corpus_audit"]

    assert len(audit) == 6
    assert set(audit["dataset"]) == {"ML-1M", "Amazon Beauty", "Amazon Sports"}
    assert set(audit["task"]) == {"ranking", "reranking"}
    assert audit["e2_manifest_path"].str.contains("1000user").all()


def test_source_inventory_and_query_corpus_audit_are_generated(bm25_tables: dict[str, pd.DataFrame]) -> None:
    source_inventory = bm25_tables["source_inventory"]
    audit = bm25_tables["query_corpus_audit"]

    assert not source_inventory.empty
    assert not audit.empty
    assert {"item_metadata", "user_history_interactions"}.issubset(set(source_inventory["artifact_type"]))
    for dataset in ("ML-1M", "Amazon Beauty", "Amazon Sports"):
        dataset_sources = source_inventory.loc[source_inventory["dataset"] == dataset]
        assert (dataset_sources["can_support_item_documents"] == "true").any()
        assert (dataset_sources["can_support_user_queries"] == "true").any()


def test_bm25_metrics_output_exists_and_covers_e2_tasks(bm25_tables: dict[str, pd.DataFrame]) -> None:
    metrics = bm25_tables["metrics"]

    assert not metrics.empty
    assert len(metrics) == 6
    assert set(metrics["dataset"]) == {"ML-1M", "Amazon Beauty", "Amazon Sports"}
    assert set(metrics["task"]) == {"ranking", "reranking"}
    assert set(metrics["candidate_pool_size"].astype(int)) == {20}


def test_bm25_scores_schema_and_ranks_when_scores_are_created(bm25_tables: dict[str, pd.DataFrame]) -> None:
    scores = bm25_tables["scores"]

    if scores.empty:
        pytest.skip("No dataset/task could be BM25-scored from local text/history sources.")
    assert {"user_id", "item_id", "bm25_score", "bm25_rank"}.issubset(scores.columns)
    assert set(scores["candidate_pool_size"].astype(int)) == {20}
    assert scores["bm25_score"].notna().all()
    per_user_ranks = scores.groupby(["dataset", "task", "user_id"])["bm25_rank"].apply(lambda values: sorted(values.astype(int).tolist()))
    assert all(ranks == list(range(1, 21)) for ranks in per_user_ranks)


def test_hr_and_ndcg_are_valid_for_computed_rows(bm25_tables: dict[str, pd.DataFrame]) -> None:
    metrics = bm25_tables["metrics"]
    computed = metrics.loc[metrics["model_or_method"] != "BM25_blocked"]

    if computed.empty:
        pytest.skip("No computed BM25-like metrics are available.")
    assert ((computed["hr_at_5"].astype(float) >= 0.0) & (computed["hr_at_5"].astype(float) <= 1.0)).all()
    assert ((computed["ndcg_at_5"].astype(float) >= 0.0) & (computed["ndcg_at_5"].astype(float) <= 1.0)).all()
    assert (computed["no_training"] == "true").all()
    assert (computed["classification"] == "no_training_text_scoring").all()


def test_blocked_missing_table_is_generated_and_keeps_lastfm_conditional(bm25_tables: dict[str, pd.DataFrame]) -> None:
    audit = bm25_tables["query_corpus_audit"]
    missing = bm25_tables["missing_or_blocked"]

    assert not missing.empty
    assert {"LastFM"}.issubset(set(missing["dataset"]))
    assert (missing.loc[missing["dataset"] == "LastFM", "classification"] == "blocked/conditional").all()
    if (audit["can_score"] != "true").any():
        blocked_valid = set(audit.loc[audit["can_score"] != "true", "dataset"])
        assert blocked_valid.issubset(set(missing["dataset"]))


def test_validation_table_contains_required_checks_and_passes(bm25_tables: dict[str, pd.DataFrame]) -> None:
    validation = bm25_tables["validation"]

    assert REQUIRED_VALIDATION_CHECKS.issubset(set(validation["check_name"]))
    assert set(validation["passed"]) == {"true"}


def test_report_contains_required_boundaries_and_claim_safety(bm25_tables: dict[str, pd.DataFrame]) -> None:
    report = build_markdown_report(bm25_tables).lower()

    for required in (
        "e3b is not full e3 baseline completion",
        "e3b is no-training local text scoring",
        "e3b performed no inference",
        "e3b made no ollama calls",
        "e3b made no qwen2/mistral calls",
        "e3b made no api calls",
        "e3b performed no downloads",
        "e3b performed no model pulls",
        "e3b performed no training",
        "e3b performed no neural-baseline training",
        "e3b performed no lastfm run",
        "e3b performed no plan c modification",
        "e3b performed no original_repo modification",
        "e3b does not run llm inference",
        "e3b does not run neural baseline training",
        "e3b does not complete full paper baseline reproduction",
        "bm25 is exact only if exact paper bm25 conditions are verified",
        "bm25_like_reconstructed",
        "reconstructed_bm25_like",
        "candidate_order_reference, random_deterministic_reference, and mostpop remain from e3a",
        "neural baselines remain for e3c/e3d",
        "lastfm remains blocked/conditional",
        "no full exact reproduction is claimed",
    ):
        assert required in report

    for unsafe in (
        "we achieved exact reproduction",
        "full exact reproduction is complete",
        "exact bm25 reproduction is complete",
        "llm inference was run",
        "neural baseline training was run",
    ):
        assert unsafe not in report


def test_outputs_are_deterministic_across_repeated_writes(tmp_path: Path, bm25_tables: dict[str, pd.DataFrame]) -> None:
    first = _tmp_paths(tmp_path / "first")
    second = _tmp_paths(tmp_path / "second")

    write_bm25_scoring_outputs(bm25_tables, first)
    write_bm25_scoring_outputs(bm25_tables, second)

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
        timeout=600,
    )

    assert "no inference" in completed.stdout.lower()
    assert _raw_jsonl_files() == before_raw_jsonl
    assert _plan_c_hashes() == before_plan_c_hashes


def test_no_output_path_is_under_original_repo() -> None:
    paths = PlanEBM25ScoringPaths()
    for output_path in paths.required_output_paths():
        relative = output_path.relative_to(PROJECT_ROOT).as_posix()
        assert not relative.startswith("original_repo/")


def test_plan_e_bm25_sources_do_not_invoke_model_api_download_or_training_code() -> None:
    source_text = (
        (PROJECT_ROOT / "src/analysis/plan_e_bm25_scoring.py").read_text(encoding="utf-8")
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


def _tmp_paths(root: Path) -> PlanEBM25ScoringPaths:
    tables = root / "results" / "tables"
    report = root / "report"
    return PlanEBM25ScoringPaths(
        source_inventory=tables / "plan_e_bm25_source_inventory.csv",
        query_corpus_audit=tables / "plan_e_bm25_query_corpus_audit.csv",
        scores=tables / "plan_e_bm25_scores.csv",
        metrics=tables / "plan_e_bm25_metrics.csv",
        validation=tables / "plan_e_bm25_validation.csv",
        missing_or_blocked=tables / "plan_e_bm25_missing_or_blocked.csv",
        next_actions=tables / "plan_e_bm25_next_actions.csv",
        report=report / "plan_e_bm25_scoring.md",
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
