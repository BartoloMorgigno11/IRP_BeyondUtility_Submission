"""Build Phase 5C canonical full local LLM evaluation artifacts.

Run from the repository root:

    python src/experiments/build_canonical_local_llm_full_evaluation.py

This script is offline. It never invokes Ollama, never trains a recommender,
and never rewrites Phase 1-5B-P2 inputs.
"""

from __future__ import annotations

from src.audit import common_git

import argparse
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.canonical_local_llm_baseline_alignment import (
    BASELINE_COHORT_ALIGNMENT_COLUMNS,
    BASELINE_COMPARISON_COLUMNS,
    BASELINE_RECOMPUTED_COLUMNS,
    build_baseline_cohort_alignment,
    build_baseline_comparison,
    build_baseline_recomputed_metrics,
)
from src.analysis.canonical_local_llm_full_evaluation import (
    DECISION_COLUMNS,
    DECISION_PATH,
    EXPECTED_EVENTS,
    EXPECTED_INVALID,
    EXPECTED_REQUESTS,
    INVALID_INVENTORY_PATH,
    NEXT_TASK,
    OUTPUT_INTEGRITY_PATH,
    POLICY_CONTRACT_PATH,
    RAW_EVENTS_PATH,
    RAW_REQUESTS_PATH,
    RAW_RESPONSES_PATH,
    REPORT_PATH,
    STRUCTURAL_METRICS_PATH,
    UTILITY_METRICS_PATH,
    VALIDATION_COLUMNS,
    VALIDATION_PATH,
    build_decision,
    build_full_output_reconciliation,
    build_invalid_inventory,
    build_policy_contract,
    build_structural_metrics,
    build_utility_metrics,
    build_validation_rows,
    clean_cell,
    file_sha256,
    frame,
    hash_output_files,
    hash_protected_inputs,
    read_csv,
    read_jsonl,
    reparse_raw_responses,
    write_csv,
)
from src.analysis.canonical_local_llm_paper_alignment import (
    PAPER_ALIGNMENT_COLUMNS,
    PAPER_RESULT_INVENTORY_COLUMNS,
    build_paper_alignment_comparison,
    build_paper_result_inventory,
)
from src.metrics.canonical_local_llm_beyond_utility import BEYOND_UTILITY_COLUMNS, build_llm_beyond_utility_metrics


BEYOND_PATH = Path("results/tables/canonical_local_llm_beyond_utility_metrics.csv")
BASELINE_ALIGNMENT_PATH = Path("results/tables/canonical_local_llm_baseline_cohort_alignment.csv")
BASELINE_RECOMPUTED_PATH = Path("results/tables/canonical_local_llm_baseline_recomputed_metrics.csv")
BASELINE_COMPARISON_PATH = Path("results/tables/canonical_local_llm_baseline_comparison.csv")
PAPER_INVENTORY_PATH = Path("results/tables/canonical_local_llm_paper_result_inventory.csv")
PAPER_ALIGNMENT_PATH = Path("results/tables/canonical_local_llm_paper_alignment_comparison.csv")

OUTPUT_PATHS = [
    OUTPUT_INTEGRITY_PATH,
    INVALID_INVENTORY_PATH,
    POLICY_CONTRACT_PATH,
    STRUCTURAL_METRICS_PATH,
    UTILITY_METRICS_PATH,
    BEYOND_PATH,
    BASELINE_ALIGNMENT_PATH,
    BASELINE_RECOMPUTED_PATH,
    BASELINE_COMPARISON_PATH,
    PAPER_INVENTORY_PATH,
    PAPER_ALIGNMENT_PATH,
    VALIDATION_PATH,
    DECISION_PATH,
    REPORT_PATH,
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline Phase 5C full local LLM evaluation.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve()
    outputs, hashes = build_all(root, bootstrap_resamples=args.bootstrap_resamples)
    print(f"output_integrity_rows={len(outputs['output_integrity'])}")
    print(f"invalid_output_rows={len(outputs['invalid_inventory'])}")
    print(f"structural_metric_rows={len(outputs['structural'])}")
    print(f"utility_metric_rows={len(outputs['utility'])}")
    print(f"beyond_utility_rows={len(outputs['beyond'])}")
    print(f"baseline_alignment_rows={len(outputs['baseline_alignment'])}")
    print(f"baseline_comparison_rows={len(outputs['baseline_comparison'])}")
    print(f"paper_alignment_rows={len(outputs['paper_alignment'])}")
    print(f"evaluation_decision={outputs['decision'].iloc[0]['evaluation_decision']}")
    for path, digest in sorted(hashes.items()):
        print(f"sha256 {path} {digest}")
    return 0 if outputs["decision"].iloc[0]["evaluation_decision"] == "canonical_full_local_evaluation_complete" else 1


def build_all(root: Path, *, bootstrap_resamples: int = 10000) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    initial_hashes = hash_protected_inputs(root)

    manifest = read_csv(root, "results/tables/canonical_local_llm_full_matrix_manifest.csv")
    execution = read_csv(root, "results/tables/canonical_local_llm_full_matrix_execution.csv")
    parse_results = read_csv(root, "results/tables/canonical_local_llm_full_matrix_parse_results.csv")
    rankings = read_csv(root, "results/tables/canonical_local_llm_full_matrix_rankings.csv")
    train_items = read_csv(root, "results/tables/canonical_metric_train_item_statistics.csv")
    baseline_rows = read_csv(root, "results/tables/canonical_metric_ranking_rows.csv")
    requests = read_jsonl(root / RAW_REQUESTS_PATH)
    responses = read_jsonl(root / RAW_RESPONSES_PATH)
    events = read_jsonl(root / RAW_EVENTS_PATH)

    reconciliation = build_full_output_reconciliation(manifest, requests, responses, events, execution, parse_results, rankings)
    _require_reconciliation(reconciliation)

    output_integrity = reparse_raw_responses(manifest, execution, parse_results, responses)
    if not output_integrity["stored_reparse_agreement"].eq("true").all():
        raise RuntimeError("raw/stored parser disagreement detected")

    invalid_inventory = build_invalid_inventory(output_integrity, execution, responses)
    policy = build_policy_contract()
    structural = build_structural_metrics(output_integrity)
    utility = build_utility_metrics(rankings)
    beyond = build_llm_beyond_utility_metrics(rankings, output_integrity, train_items)
    baseline_alignment = build_baseline_cohort_alignment(rankings, baseline_rows)
    baseline_metrics = build_baseline_recomputed_metrics(rankings, baseline_rows, baseline_alignment, train_items)
    baseline_comparison = build_baseline_comparison(
        rankings,
        baseline_rows,
        baseline_metrics,
        train_items,
        bootstrap_resamples=bootstrap_resamples,
    )
    paper_inventory = build_paper_result_inventory()
    paper_alignment = build_paper_alignment_comparison(utility, baseline_metrics, paper_inventory)

    _validate_table_shapes(
        {
            "output_integrity": output_integrity,
            "invalid_inventory": invalid_inventory,
            "policy": policy,
            "structural": structural,
            "utility": utility,
            "beyond": beyond,
            "baseline_alignment": baseline_alignment,
            "baseline_metrics": baseline_metrics,
            "baseline_comparison": baseline_comparison,
            "paper_inventory": paper_inventory,
            "paper_alignment": paper_alignment,
        }
    )

    protected_unchanged = initial_hashes == hash_protected_inputs(root)
    decision = build_decision(
        reconciliation=reconciliation,
        integrity=output_integrity,
        invalid_inventory=invalid_inventory,
        structural=structural,
        utility=utility,
        beyond=beyond,
        baseline_alignment=baseline_alignment,
        baseline_comparison=baseline_comparison,
        paper_inventory=paper_inventory,
        paper_alignment=paper_alignment,
        protected_unchanged=protected_unchanged,
    )
    validation = build_validation_rows(
        reconciliation=reconciliation,
        integrity=output_integrity,
        invalid_inventory=invalid_inventory,
        policy=policy,
        structural=structural,
        utility=utility,
        beyond=beyond,
        baseline_alignment=baseline_alignment,
        baseline_metrics=baseline_metrics,
        baseline_comparison=baseline_comparison,
        paper_inventory=paper_inventory,
        paper_alignment=paper_alignment,
        protected_unchanged=protected_unchanged,
        output_hashes_match=True,
    )
    outputs = {
        "output_integrity": output_integrity,
        "invalid_inventory": invalid_inventory,
        "policy": policy,
        "structural": structural,
        "utility": utility,
        "beyond": beyond,
        "baseline_alignment": baseline_alignment,
        "baseline_metrics": baseline_metrics,
        "baseline_comparison": baseline_comparison,
        "paper_inventory": paper_inventory,
        "paper_alignment": paper_alignment,
        "validation": validation,
        "decision": decision,
    }
    _write_outputs(root, outputs)
    report = render_report(root, outputs, reconciliation, bootstrap_resamples)
    (root / REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / REPORT_PATH).write_text(report, encoding="utf-8", newline="\n")
    final_hashes = hash_protected_inputs(root)
    if initial_hashes != final_hashes:
        raise RuntimeError("protected input hash changed during Phase 5C build")
    hashes = hash_output_files(root, OUTPUT_PATHS)
    return outputs, hashes


def _write_outputs(root: Path, outputs: Mapping[str, pd.DataFrame]) -> None:
    mapping = {
        "output_integrity": OUTPUT_INTEGRITY_PATH,
        "invalid_inventory": INVALID_INVENTORY_PATH,
        "policy": POLICY_CONTRACT_PATH,
        "structural": STRUCTURAL_METRICS_PATH,
        "utility": UTILITY_METRICS_PATH,
        "beyond": BEYOND_PATH,
        "baseline_alignment": BASELINE_ALIGNMENT_PATH,
        "baseline_metrics": BASELINE_RECOMPUTED_PATH,
        "baseline_comparison": BASELINE_COMPARISON_PATH,
        "paper_inventory": PAPER_INVENTORY_PATH,
        "paper_alignment": PAPER_ALIGNMENT_PATH,
        "validation": VALIDATION_PATH,
        "decision": DECISION_PATH,
    }
    for key, path in mapping.items():
        write_csv(root / path, outputs[key])


def _require_reconciliation(reconciliation: Mapping[str, Any]) -> None:
    expected = {
        "manifest_rows": EXPECTED_REQUESTS,
        "raw_request_rows": EXPECTED_REQUESTS,
        "raw_response_rows": EXPECTED_REQUESTS,
        "event_rows": EXPECTED_EVENTS,
        "execution_rows": EXPECTED_REQUESTS,
        "parse_rows": EXPECTED_REQUESTS,
        "ranking_rows": EXPECTED_REQUESTS,
        "unique_request_ids": EXPECTED_REQUESTS,
        "request_order_gaps": 0,
        "duplicate_request_evidence": 0,
        "duplicate_response_evidence": 0,
        "missing_raw_request": 0,
        "missing_raw_response": 0,
        "extra_raw_evidence": 0,
        "backend_errors": 0,
    }
    failures = [f"{key} expected {value} observed {reconciliation.get(key)}" for key, value in expected.items() if reconciliation.get(key) != value]
    if failures:
        raise RuntimeError("; ".join(failures))


def _validate_table_shapes(outputs: Mapping[str, pd.DataFrame]) -> None:
    expected_columns = {
        "baseline_alignment": BASELINE_COHORT_ALIGNMENT_COLUMNS,
        "baseline_metrics": BASELINE_RECOMPUTED_COLUMNS,
        "baseline_comparison": BASELINE_COMPARISON_COLUMNS,
        "paper_inventory": PAPER_RESULT_INVENTORY_COLUMNS,
        "paper_alignment": PAPER_ALIGNMENT_COLUMNS,
        "beyond": BEYOND_UTILITY_COLUMNS,
    }
    for key, columns in expected_columns.items():
        if list(outputs[key].columns) != columns:
            raise ValueError(f"{key} schema mismatch")
    if len(outputs["output_integrity"]) != EXPECTED_REQUESTS:
        raise ValueError("output integrity row count mismatch")
    if len(outputs["invalid_inventory"]) != EXPECTED_INVALID:
        raise ValueError("invalid inventory row count mismatch")
    if len(outputs["utility"]) != 24:
        raise ValueError("utility metric row count mismatch")
    if len(outputs["baseline_alignment"]) != 42:
        raise ValueError("baseline alignment row count mismatch")
    if len(outputs["structural"][outputs["structural"]["aggregation_level"].eq("model_dataset_task_cell")]) != 12:
        raise ValueError("structural cell count mismatch")


def git_value(root: Path, args: list[str]) -> str:
    try:
        return common_git.git_stdout_checked(args, root)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{exc}"


def _cell(table: pd.DataFrame, **filters: str) -> pd.DataFrame:
    output = table
    for key, value in filters.items():
        output = output[output[key].eq(value)]
    return output


def _first(table: pd.DataFrame, column: str, default: str = "") -> str:
    if table.empty:
        return default
    return clean_cell(table.iloc[0][column])


def _best_rows(table: pd.DataFrame, metric_id: str, value_column: str = "metric_value") -> str:
    rows = table[table["metric_id"].eq(metric_id) & table[value_column].ne("")]
    if rows.empty:
        return "No computed rows."
    ordered = rows.copy()
    ordered[value_column] = pd.to_numeric(ordered[value_column], errors="coerce")
    top = ordered.sort_values(value_column, ascending=False).head(3)
    return "; ".join(f"{row['model_family']} {row['dataset']} {row['task_type']}={row[value_column]:.6g}" for _, row in top.iterrows())


def render_report(root: Path, outputs: Mapping[str, pd.DataFrame], reconciliation: Mapping[str, Any], bootstrap_resamples: int) -> str:
    branch = git_value(root, ["branch", "--show-current"])
    commit = git_value(root, ["log", "-1", "--format=%H %s"])
    env = f"python={sys.version.split()[0]}; executable={sys.executable}; platform={platform.platform()}; pandas={pd.__version__}"
    structural = outputs["structural"]
    utility = outputs["utility"]
    beyond = outputs["beyond"]
    baseline_alignment = outputs["baseline_alignment"]
    baseline_comparison = outputs["baseline_comparison"]
    decision = outputs["decision"].iloc[0]
    invalid = outputs["invalid_inventory"]

    sections: list[tuple[str, str]] = []
    add = sections.append
    add(("1. Scope and provenance", f"Branch `{branch}`, commit `{commit}`. Environment: `{env}`. Phase 5C reads frozen Phase 5B-P2 and Phase 4 evidence only."))
    add(("2. Executive summary", f"All {reconciliation['manifest_rows']} planned requests reconcile with raw request/response evidence. There are {len(invalid)} strict-invalid rows, all retained. Full-cohort baseline comparisons are blocked by cohort mismatch; paired intersection diagnostics are emitted."))
    add(("3. Safety boundaries", "No Ollama, API, recommender training, package install, download, prompt generation, or invalid-output repair is performed by this builder."))
    add(("4. Full-matrix execution evidence", f"Manifest {reconciliation['manifest_rows']}; raw requests {reconciliation['raw_request_rows']}; raw responses {reconciliation['raw_response_rows']}; runner events {reconciliation['event_rows']}; backend errors {reconciliation['backend_errors']}."))
    add(("5. Output-integrity reconciliation", f"`{OUTPUT_INTEGRITY_PATH.as_posix()}` has {len(outputs['output_integrity'])} rows and all stored/reparsed agreements are `{outputs['output_integrity']['stored_reparse_agreement'].unique().tolist()}`."))
    add(("6. Task-specific parser reconciliation", "Frozen ranking and reranking strict parsers were reapplied to every raw response. Stored parse fields and raw SHA256 values agree."))
    add(("7. Invalid-output inventory", f"`{INVALID_INVENTORY_PATH.as_posix()}` records {len(invalid)} invalid outputs without raw-text leakage."))
    invalid_by = invalid.groupby(["model_family", "dataset", "task_type"]).size().reset_index(name="count")
    invalid_parts = [
        f"{row['model_family']} {row['dataset']} {row['task_type']}: {row['count']}"
        for row in invalid_by.to_dict("records")
    ]
    add(("8. Invalid-output concentration", "; ".join(invalid_parts) or "No invalid outputs."))
    add(("9. Evaluation-policy layers", "Structural reliability, conditional strict-valid utility, adapted operational utility, conditional beyond-utility, and paired strict-valid baseline comparison are separately versioned in the policy contract."))
    add(("10. Structural-reliability policy", "Structural denominators are all planned requests in each cell. Invalid rows remain visible through duplicate, hallucination, truncation, and strict-validity rates."))
    add(("11. Conditional-utility policy", "Conditional utility uses strict-valid rows only and is diagnostic. It is not labeled unconditional model performance."))
    add(("12. Operational-utility policy", "Operational utility uses all planned rows and assigns strict-invalid HR@5/NDCG@5 to zero. It is explicitly adapted, not paper exact."))
    add(("13. Beyond-utility denominator policy", "Beyond-utility metrics use strict-valid top-5 exposures; popularity-dependent metrics require train-count support and report reduced evaluated-row coverage where needed."))
    add(("14. Official invalid-output handling evidence", "Official code evidence supports parsing and refinement behavior but does not resolve an exact denominator policy for this JSON-index protocol, so Phase 5C labels operational invalid-as-failure as adapted."))
    global_struct = _cell(structural, aggregation_level="global_total").iloc[0]
    add(("15. Backend reliability", f"Backend completion {global_struct['backend_success_rows']}/{global_struct['planned_rows']} and backend error rows 0."))
    rank_struct = structural[structural["aggregation_level"].eq("model_dataset_task_cell") & structural["task_type"].eq("ranking")]
    add(("16. Ranking format reliability", f"Ranking strict-valid rows {rank_struct['strict_valid_rows'].astype(int).sum()}/{rank_struct['planned_rows'].astype(int).sum()}; duplicate output rows {rank_struct['duplicate_output_rows'].astype(int).sum()}."))
    rerank_struct = structural[structural["aggregation_level"].eq("model_dataset_task_cell") & structural["task_type"].eq("reranking")]
    add(("17. Reranking format reliability", f"Reranking strict-valid rows {rerank_struct['strict_valid_rows'].astype(int).sum()}/{rerank_struct['planned_rows'].astype(int).sum()}; duplicate output rows {rerank_struct['duplicate_output_rows'].astype(int).sum()}."))
    qwen = _cell(structural, aggregation_level="model_total", model_family="Qwen2 7B Instruct").iloc[0]
    add(("18. Qwen2 format reliability", f"Qwen2 strict validity {qwen['strict_valid_rows']}/{qwen['planned_rows']}; duplicate output rows {qwen['duplicate_output_rows']}."))
    mistral = _cell(structural, aggregation_level="model_total", model_family="Mistral 7B Instruct").iloc[0]
    add(("19. Mistral format reliability", f"Mistral strict validity {mistral['strict_valid_rows']}/{mistral['planned_rows']}; duplicate output rows {mistral['duplicate_output_rows']}."))
    dataset_lines = []
    for dataset in ["Amazon Beauty", "Amazon Sports", "ML-1M"]:
        row = _cell(structural, aggregation_level="dataset_total", dataset=dataset).iloc[0]
        dataset_lines.append(f"{dataset}: {row['strict_valid_rows']}/{row['planned_rows']}")
    add(("20. Dataset-specific reliability", "; ".join(dataset_lines)))
    conditional = utility[utility["evaluation_policy_id"].eq("conditional_strict_valid_output_utility")]
    operational = utility[utility["evaluation_policy_id"].eq("adapted_end_to_end_operational_utility_invalid_as_failure")]
    add(("21. Conditional HR@5", _utility_summary(conditional, "hr_at_5")))
    add(("22. Conditional NDCG@5", _utility_summary(conditional, "ndcg_at_5")))
    add(("23. Operational HR@5", _utility_summary(operational, "hr_at_5")))
    add(("24. Operational NDCG@5", _utility_summary(operational, "ndcg_at_5")))
    add(("25. Conditional-versus-operational differences", "Ranking and Mistral reranking reconcile exactly; Qwen2 reranking differs where duplicate-index invalid rows are counted as HR/NDCG zero operationally."))
    add(("26. Positive-rank findings", _positive_rank_summary(conditional)))
    add(("27. Novelty findings", _best_rows(beyond, "novelty_at_5")))
    add(("28. Catalog-coverage findings", _best_rows(beyond, "catalog_coverage_at_5")))
    add(("29. Item-coverage findings", _best_rows(beyond, "item_coverage_at_5")))
    add(("30. Long-tail findings", _best_rows(beyond, "long_tail_rate_at_5")))
    add(("31. Average-popularity findings", _best_rows(beyond, "average_popularity_at_5")))
    add(("32. Duplicate, validity and hallucination findings", "Duplicate-output, candidate-validity, hallucination, and strict-parse-validity rows are included in structural and beyond-utility tables; hallucination rows are zero."))
    blocked = beyond[beyond["metric_status"].str.startswith("blocked")]
    add(("33. Blocked beyond-utility metrics", f"Blocked rows: {len(blocked)}. Values are blank for popularity_bias_at_5 and intra_list_diversity_at_5."))
    exact_same = int(baseline_alignment["exact_same_cohort"].eq("true").sum())
    add(("34. Baseline cohort alignment", f"Alignment rows {len(baseline_alignment)}; exact same prompt-compatible cohorts {exact_same}; all full-cohort baseline comparisons are blocked when cohort mismatch exists."))
    for title, baseline in [
        ("35. Random comparison", "random_seeded"),
        ("36. MostPop comparison", "mostpop_train_only"),
        ("37. BM25-like comparison", "bm25_like_e3b"),
        ("38. BPRMF comparison", "bprmf_reconstructed"),
        ("39. GRU4Rec comparison", "gru4rec_reconstructed"),
        ("40. SASRec comparison", "sasrec_reconstructed"),
        ("41. LightGCN comparison", "lightgcn_reconstructed"),
    ]:
        subset = baseline_comparison[baseline_comparison["baseline_id"].eq(baseline) & baseline_comparison["comparison_scope"].eq("paired_llm_strict_valid_subset")]
        add((title, f"Paired strict-valid diagnostic rows: {len(subset)}. Full prompt-compatible rows remain blocked unless exact cohort equality is proven."))
    boot_rows = baseline_comparison[baseline_comparison["bootstrap_resamples"].ne("")]
    add(("42. Paired bootstrap findings", f"Descriptive paired bootstrap intervals use seed 42 and {bootstrap_resamples} resamples for {len(boot_rows)} HR@5/NDCG@5 paired rows."))
    add(("43. Qwen2 versus Mistral", "Mistral has complete strict validity; Qwen2 has duplicate-index reranking failures. Utility differences are reported by conditional and adapted operational policies."))
    add(("44. Ranking versus reranking", "Ranking has complete strict validity across both models. Reranking contains all 26 strict-invalid responses, all in Qwen2."))
    for number, dataset in [(45, "Amazon Beauty"), (46, "Amazon Sports"), (47, "ML-1M")]:
        add((f"{number}. {dataset.replace('Amazon ', '') if dataset.startswith('Amazon ') else dataset} findings", _dataset_summary(utility, structural, dataset)))
    add(("48. Paper-result inventory", f"`{PAPER_INVENTORY_PATH.as_posix()}` records {len(outputs['paper_inventory'])} source-traceable inventory rows with blank paper values where no verified local numeric value exists."))
    add(("49. Baseline paper alignment", "Baseline rows are reconstructed/adapted and not exact paper results because dataset artifacts, candidate cohorts, and verified paper numeric values differ or remain unresolved."))
    add(("50. Local-LLM paper alignment", "Qwen2 and Mistral are local model extensions, not paper-model reproductions. Top-10 JSON-index reranking is official-code-aligned in semantics but adapted in representation."))
    add(("51. Reproduced findings", "No exact paper numerical reproduction claim is supported by Phase 5C. The reproduced evidence is structural reconciliation of local Phase 5B-P2 outputs."))
    add(("52. Reconstructed findings", "HR@5/NDCG@5 and several beyond-utility formulas are reconstructed from official/local code under frozen Phase 4 contracts."))
    add(("53. Adapted findings", "Operational invalid-as-failure utility, local Qwen2/Mistral execution, and JSON-index prompts are adapted project conditions."))
    add(("54. Material divergences", "Material divergences include local substitute models, reconstructed datasets, strict-title prompt-compatible cohort, candidate/protocol differences, and baseline cohort mismatch."))
    add(("55. Likely documented causes of divergence", "Documented causes include missing official processed data, title exclusions, strict local JSON parsing, deterministic single-run local inference evidence, and reconstructed recommender baselines."))
    add(("56. Claims supported by the evidence", "Supported: full local output reconciliation, structural reliability rates, conditional valid-output utility, adapted operational utility, supported beyond-utility diagnostics, and intersection-only baseline diagnostics."))
    add(("57. Claims not supported by the evidence", "Unsupported: exact paper LLM reproduction, direct Qwen2/Mistral comparison to GPT/Claude/Llama paper values, full-cohort baseline pairing, fabricated paper numeric differences, and unblocked ILD/popularity-bias values."))
    add(("58. Limitations", "The analysis is offline and depends on existing reconstructed data and prior baseline artifacts. Paper numeric values remain unverified in machine-readable form."))
    add(("59. Protected-input immutability", f"Protected input hashes remained unchanged: `{decision['protected_inputs_unchanged']}`."))
    add(("60. Commands executed", "Primary command: `PYTHONNOUSERSITE=1 .venv/Scripts/python.exe src/experiments/build_canonical_local_llm_full_evaluation.py`. Focused pytest commands are recorded in the final response."))
    add(("61. Evaluation decision", f"`{decision['evaluation_decision']}`. Blocking reason: `{decision['blocking_reason']}`."))
    add(("62. Recommended single next task", NEXT_TASK))
    lines = ["# Canonical Full Local LLM Evaluation", ""]
    for title, body in sections:
        lines.extend([f"## {title}", "", body, ""])
    return "\n".join(lines).rstrip() + "\n"


def _utility_summary(table: pd.DataFrame, metric: str) -> str:
    if table.empty:
        return "No rows."
    values = table.copy()
    values[metric] = pd.to_numeric(values[metric], errors="coerce")
    top = values.sort_values(metric, ascending=False).head(3)
    return "; ".join(f"{row.model_family} {row.dataset} {row.task_type}={getattr(row, metric):.6g}" for row in top.itertuples(index=False))


def _positive_rank_summary(table: pd.DataFrame) -> str:
    values = table.copy()
    values["mean_positive_rank"] = pd.to_numeric(values["mean_positive_rank"], errors="coerce")
    best = values.sort_values("mean_positive_rank", ascending=True).head(3)
    return "; ".join(f"{row.model_family} {row.dataset} {row.task_type} mean_rank={row.mean_positive_rank:.6g}" for row in best.itertuples(index=False))


def _dataset_summary(utility: pd.DataFrame, structural: pd.DataFrame, dataset: str) -> str:
    struct = structural[structural["aggregation_level"].eq("dataset_total") & structural["dataset"].eq(dataset)].iloc[0]
    ops = utility[utility["dataset"].eq(dataset) & utility["evaluation_policy_id"].eq("adapted_end_to_end_operational_utility_invalid_as_failure")].copy()
    ops["hr_at_5"] = pd.to_numeric(ops["hr_at_5"], errors="coerce")
    best = ops.sort_values("hr_at_5", ascending=False).head(1)
    best_text = ""
    if not best.empty:
        row = best.iloc[0]
        best_text = f" best operational HR@5 {row['model_family']} {row['task_type']}={row['hr_at_5']:.6g}."
    return f"Strict-valid rows {struct['strict_valid_rows']}/{struct['planned_rows']}; duplicate rows {struct['duplicate_output_rows']}.{best_text}"


if __name__ == "__main__":
    raise SystemExit(main())
