from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.analysis.plan_e_paper_condition_alignment import (
    ALIGNMENT_COLUMNS,
    API_COST_COLUMNS,
    EXACTNESS_COLUMNS,
    LOCAL_RUNTIME_COLUMNS,
    MISSING_COLUMNS,
    ROADMAP_COLUMNS,
    SCALEUP_COLUMNS,
    PlanEAuditPaths,
    build_all_tables,
    build_markdown_report,
    write_audit_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "src/experiments/build_plan_e_paper_condition_alignment_audit.py"
RAW_OUTPUTS_DIR = PROJECT_ROOT / "results/raw_outputs"


def test_all_required_output_files_can_be_generated(tmp_path: Path) -> None:
    paths = _tmp_paths(tmp_path)
    tables = build_all_tables(PROJECT_ROOT)
    write_audit_outputs(tables, paths)

    for output_path in paths.required_output_paths():
        assert output_path.exists(), output_path

    assert set(ALIGNMENT_COLUMNS).issubset(pd.read_csv(paths.alignment_matrix).columns)
    assert set(MISSING_COLUMNS).issubset(pd.read_csv(paths.missing_conditions).columns)
    assert set(EXACTNESS_COLUMNS).issubset(pd.read_csv(paths.exactness_feasibility).columns)
    assert set(SCALEUP_COLUMNS).issubset(pd.read_csv(paths.scaleup_call_estimates).columns)
    assert set(LOCAL_RUNTIME_COLUMNS).issubset(pd.read_csv(paths.local_runtime_estimates).columns)
    assert set(API_COST_COLUMNS).issubset(pd.read_csv(paths.api_cost_scenarios).columns)
    assert set(ROADMAP_COLUMNS).issubset(pd.read_csv(paths.recommended_roadmap).columns)


def test_core_paper_conditions_are_represented() -> None:
    tables = build_all_tables(PROJECT_ROOT)
    alignment = tables["alignment_matrix"]
    text = _frame_text(alignment)

    for expected in (
        "Amazon Beauty",
        "Amazon Sports",
        "ML-1M",
        "LastFM",
        "Ranking",
        "Re-ranking",
        "HR@5",
        "NDCG@5",
        "20 candidate items per user",
        "1 positive item plus 19 negative items",
        "1000 test users",
        "BM25",
        "MostPop",
        "BPRMF",
        "GRU4Rec",
        "SASRec",
        "LightGCN",
        "OpenP5-like",
        "LLMRank-like",
        "LLMSRec/ICL",
        "hallucination",
        "candidate position bias",
        "history length sensitivity",
        "generation-involved performance/user profiles",
        "novelty",
        "popularity",
        "fairness",
        "candidate validity",
    ):
        assert expected.lower() in text


def test_current_project_phases_c_d_d5_are_represented() -> None:
    text = _frame_text(build_all_tables(PROJECT_ROOT)["exactness_feasibility"])

    assert "plan c" in text
    assert "reconstructed reproduction" in text
    assert "plan d" in text
    assert "extended_subset" in text
    assert "d5" in text
    assert "diagnostic_subset" in text


def test_candidate_pool_and_1000_user_targets_are_explicit() -> None:
    text = _frame_text(build_all_tables(PROJECT_ROOT)["alignment_matrix"])

    assert "1000 test users" in text
    assert "20 candidate items per user" in text
    assert "1 positive item plus 19 negative items" in text


def test_seven_llm_paper_matrix_and_local_status_are_represented() -> None:
    alignment = build_all_tables(PROJECT_ROOT)["alignment_matrix"]
    llm_rows = alignment.loc[alignment["condition_group"] == "llm"]
    llm_text = _frame_text(llm_rows)

    for llm in (
        "GPT-3.5",
        "GPT-4o-mini",
        "GPT-4o",
        "Claude 3 Haiku",
        "Llama3-70B",
        "Qwen2-7B",
        "Mistral-7B",
    ):
        assert llm.lower() in llm_text
    assert "local qwen2:7b-instruct diagnostics" in llm_text
    assert "local mistral:7b-instruct smoke" in llm_text


def test_closed_source_llms_are_conditional_and_claude_is_blocked_conditional() -> None:
    alignment = build_all_tables(PROJECT_ROOT)["alignment_matrix"]
    closed = alignment.loc[alignment["paper_condition"].isin(["GPT-3.5", "GPT-4o-mini", "GPT-4o"])]

    assert not closed.empty
    assert set(closed["feasibility_class"]) == {"conditional/API-dependent"}
    assert set(closed["exactness_class"]) == {"conditional/API-dependent"}

    claude = alignment.loc[alignment["paper_condition"] == "Claude 3 Haiku"].iloc[0]
    assert claude["feasibility_class"] == "blocked/conditional"
    assert "conditional" in claude["notes"].lower()


def test_llama3_70b_is_hardware_cloud_api_dependent() -> None:
    exactness = build_all_tables(PROJECT_ROOT)["exactness_feasibility"]
    row = exactness.loc[exactness["component_id"] == "LLAMA3_70B"].iloc[0]

    assert row["exactness_class"] == "hardware/cloud/API-dependent"
    assert row["feasibility_class"] == "hardware/cloud/API-dependent"


def test_lastfm_remains_extension_only_or_blocked() -> None:
    exactness = build_all_tables(PROJECT_ROOT)["exactness_feasibility"]
    row = exactness.loc[exactness["component_id"] == "LASTFM"].iloc[0]
    text = " ".join(row.astype(str).tolist()).lower()

    assert "extension_only_no_valid_utility_evaluation" in text
    assert "blocked/conditional" in text
    assert "do not claim lastfm reproduction" in text


def test_report_contains_required_claim_safety_and_no_action_statements() -> None:
    report = build_markdown_report(build_all_tables(PROJECT_ROOT)).lower()

    for required in (
        "e0 performed no inference",
        "e0 made no ollama calls",
        "e0 made no qwen2/mistral calls",
        "e0 made no api calls",
        "e0 performed no downloads",
        "e0 performed no model pulls",
        "e0 performed no training",
        "e0 performed no lastfm run",
        "e0 performed no plan c modification",
        "e0 performed no destructive cleanup",
        "strict candidate-id-only prompting is a project extension, not a paper-exact prompt strategy",
        "local qwen2/mistral 20-user diagnostics are diagnostic_subset results",
        "closed-source models remain conditional/api-dependent",
        "llama3-70b is hardware/cloud/api-dependent",
        "lastfm remains extension_only_no_valid_utility_evaluation or blocked unless valid artifacts are found",
        "no full exact reproduction has been achieved yet",
        "reproduced or reconstructed only the accessible parts and audited the inaccessible parts",
    ):
        assert required in report

    for unsafe_claim in (
        "we achieved exact reproduction",
        "full llm reproduction is complete",
        "we reproduced closed-source models",
        "closed-source models are reproduced",
        "lastfm is reproduced",
    ):
        assert unsafe_claim not in report


def test_outputs_are_deterministic_across_repeated_builds(tmp_path: Path) -> None:
    tables = build_all_tables(PROJECT_ROOT)
    first = _tmp_paths(tmp_path / "first")
    second = _tmp_paths(tmp_path / "second")

    write_audit_outputs(tables, first)
    write_audit_outputs(tables, second)

    for first_path, second_path in zip(first.required_output_paths(), second.required_output_paths()):
        assert first_path.read_bytes() == second_path.read_bytes(), first_path.name


def test_scaleup_estimates_include_required_scenarios() -> None:
    scaleup = build_all_tables(PROJECT_ROOT)["scaleup_call_estimates"]
    scenario_ids = set(scaleup["scenario_id"])

    assert "CALLS_20_USERS" in scenario_ids
    assert "CALLS_100_USERS" in scenario_ids
    assert "CALLS_1000_USERS" in scenario_ids
    assert "CALLS_1000_USERS_3_DATASETS" in scenario_ids
    assert "CALLS_1000_USERS_4_DATASETS_LASTFM_CONDITIONAL" in scenario_ids
    assert "CALLS_RANKING_ONLY" in scenario_ids
    assert "CALLS_RANKING_PLUS_RERANKING" in scenario_ids
    assert "CALLS_PROMPT_STRATEGY_MATRIX" in scenario_ids
    assert "CALLS_LOCAL_2_MODEL_MATRIX" in scenario_ids
    assert "CALLS_PAPER_7_LLM_MATRIX" in scenario_ids
    assert "CALLS_7_LLM_3_PROMPT_3_DATASET_1000_USERS" in scenario_ids
    assert int(scaleup.loc[scaleup["scenario_id"] == "CALLS_7_LLM_3_PROMPT_3_DATASET_1000_USERS", "estimated_total_calls"].iloc[0]) == 126000


def test_cost_scenarios_are_placeholder_not_current_price_claims() -> None:
    costs = build_all_tables(PROJECT_ROOT)["api_cost_scenarios"]
    text = _frame_text(costs)

    for scenario in ("very_low_cost", "low_cost", "medium_cost", "high_cost"):
        assert scenario in text
    assert set(costs["requires_current_price_check"]) == {"true"}
    assert set(costs["source_status"]) == {"manual_placeholder_not_official"}


def test_build_script_does_not_create_raw_jsonl_or_modify_plan_c_files() -> None:
    before_raw_jsonl = _raw_jsonl_files()
    before_plan_c_hashes = _plan_c_hashes()

    completed = subprocess.run(
        [sys.executable, str(RUNNER_PATH)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert "no inference" in completed.stdout.lower()
    assert _raw_jsonl_files() == before_raw_jsonl
    assert _plan_c_hashes() == before_plan_c_hashes


def test_no_output_path_is_under_original_repo() -> None:
    paths = PlanEAuditPaths()
    for output_path in paths.required_output_paths():
        relative = output_path.relative_to(PROJECT_ROOT).as_posix()
        assert not relative.startswith("original_repo/")


def test_plan_e_sources_do_not_invoke_model_api_download_or_training_code() -> None:
    source_text = (
        (PROJECT_ROOT / "src/analysis/plan_e_paper_condition_alignment.py").read_text(encoding="utf-8")
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


def _tmp_paths(root: Path) -> PlanEAuditPaths:
    tables = root / "results" / "tables"
    report = root / "report"
    return PlanEAuditPaths(
        alignment_matrix=tables / "plan_e_paper_condition_alignment_matrix.csv",
        missing_conditions=tables / "plan_e_missing_conditions.csv",
        exactness_feasibility=tables / "plan_e_exactness_feasibility.csv",
        scaleup_call_estimates=tables / "plan_e_scaleup_call_estimates.csv",
        local_runtime_estimates=tables / "plan_e_local_runtime_estimates.csv",
        api_cost_scenarios=tables / "plan_e_api_cost_scenarios.csv",
        recommended_roadmap=tables / "plan_e_recommended_roadmap.csv",
        report=report / "plan_e_paper_condition_alignment_audit.md",
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
