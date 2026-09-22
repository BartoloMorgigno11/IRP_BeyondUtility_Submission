from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "results/tables/plan_d_llm_model_registry.csv"
BACKEND_AUDIT_PATH = PROJECT_ROOT / "results/tables/plan_d_llm_backend_audit.csv"
ROADMAP_PATH = PROJECT_ROOT / "results/tables/plan_d_llm_task_roadmap.csv"
REPORT_PATH = PROJECT_ROOT / "report/plan_d_llm_model_registry_and_backend_audit.md"

PAPER_MODELS = {
    "gpt-3.5-turbo-0125",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06",
    "claude-3-haiku-20240307",
    "llama-3-70b-instruct",
    "qwen-2-7b-instruct",
    "mistral-7b-instruct-v0.2",
}


def test_plan_d_llm_audit_artifacts_exist() -> None:
    assert REGISTRY_PATH.exists()
    assert BACKEND_AUDIT_PATH.exists()
    assert ROADMAP_PATH.exists()
    assert REPORT_PATH.exists()


def test_model_registry_contains_exactly_the_paper_llms_and_expected_statuses() -> None:
    registry = pd.read_csv(REGISTRY_PATH, dtype=str).fillna("")

    assert set(registry["paper_model"]) == PAPER_MODELS
    assert len(registry) == len(PAPER_MODELS)
    assert _model_row(registry, "qwen-2-7b-instruct")["status"] == "recommended_first_local_model"
    assert (
        _model_row(registry, "mistral-7b-instruct-v0.2")["status"]
        == "recommended_second_local_model"
    )
    assert (
        _model_row(registry, "claude-3-haiku-20240307")["status"]
        == "blocked_exact_model_retired"
    )
    assert _model_row(registry, "llama-3-70b-instruct")["status"] == "blocked_local_hardware"
    assert (
        _model_row(registry, "gpt-4o-mini-2024-07-18")["status"]
        == "feasible_with_api_budget"
    )


def test_backend_audit_contains_ollama_and_torch_cuda_rows() -> None:
    audit = pd.read_csv(BACKEND_AUDIT_PATH, dtype=str).fillna("")

    assert not audit.loc[
        (audit["backend"] == "ollama") & (audit["check_name"] == "command_exists")
    ].empty
    cuda_rows = audit.loc[
        (audit["backend"] == "torch")
        & (audit["check_name"].isin(["cuda_available", "cuda_device_name"]))
    ]
    assert len(cuda_rows) >= 2
    assert all(cuda_rows["version_or_result"].str.len() > 0)


def test_task_roadmap_contains_required_d5_phases() -> None:
    roadmap = pd.read_csv(ROADMAP_PATH, dtype=str).fillna("")
    phase_tasks = set(zip(roadmap["phase"], roadmap["task"]))

    assert ("D5C", "ollama_smoke_test") in phase_tasks
    assert ("D5D", "llm_prompt_templates_parser_manifest") in phase_tasks
    assert ("D5E", "llm_ranking_diagnostic") in phase_tasks
    assert ("D5E.1", "qwen2_output_validity_hallucination_analysis") in phase_tasks
    assert ("D5F", "candidate_position_bias") in phase_tasks
    assert ("D5G", "strict_candidate_id_prompt_and_post_filter_diagnostic") in phase_tasks
    assert ("D5H", "mistral_local_availability_smoke_test") in phase_tasks


def test_report_documents_no_inference_no_api_calls_and_scope() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")
    report_lower = report.lower()

    assert "no llm inference was run" in report_lower
    assert "no api calls/downloads were performed" in report_lower
    for expected in ("ollama", "qwen", "mistral", "openai", "claude"):
        assert expected in report_lower
    assert "lastfm is not required here" in report_lower


def test_d5b_tests_are_documentation_only() -> None:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported_modules = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    string_constants = [
        node.value.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]

    assert not {"requests", "openai", "anthropic", "subprocess"} & imported_modules
    assert not any(("ollama " + "run") in value for value in string_constants)
    assert not any(("ollama " + "pull") in value for value in string_constants)


def _model_row(registry: pd.DataFrame, model: str) -> pd.Series:
    rows = registry.loc[registry["paper_model"] == model]
    assert len(rows) == 1
    return rows.iloc[0]
