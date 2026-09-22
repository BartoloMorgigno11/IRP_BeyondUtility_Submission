"""Build Phase 3K SASRec same-cohort consolidation tables and report."""

from __future__ import annotations

from src.audit import common_git

import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_e_sasrec_baseline_consolidation import (
    COMPARISON_COLUMNS,
    FULL_COHORT_ID,
    MATCHED_COHORT_ID,
    SAME_COHORT_METRIC_COLUMNS,
    WARNING_COLUMNS,
    aggregate_warning_records,
    build_paired_comparisons,
    build_same_cohort_metrics,
    normalize_bprmf_rankings,
    normalize_gru4rec_rankings,
    normalize_sasrec_rankings,
    read_csv,
    validate_comparison_frame,
    validate_metric_ranges,
    validate_warning_inventory,
    write_csv,
)


ALL_DATASET_COLUMNS = [
    "model_id",
    "dataset",
    "task_type",
    "candidate_set_id",
    "cohort_id",
    "evaluated_rows",
    "candidate_count",
    "hr_at_5",
    "ndcg_at_5",
    "mean_positive_rank_zero_based",
    "median_positive_rank_zero_based",
    "valid_ranking_rate",
    "duplicate_ranked_item_rate",
    "missing_score_rate",
    "mean_raw_history_length",
    "median_raw_history_length",
    "mean_effective_history_length",
    "median_effective_history_length",
    "history_truncation_rate",
    "source_metrics_path",
    "paper_alignment",
    "notes",
]

OUTPUTS = {
    "all_sasrec": PROJECT_ROOT / "results/tables/plan_e_sasrec_all_datasets_metrics.csv",
    "same_cohort": PROJECT_ROOT / "results/tables/plan_e_sasrec_same_cohort_baselines_metrics.csv",
    "comparison": PROJECT_ROOT / "results/tables/plan_e_sasrec_baseline_comparison.csv",
    "warning_inventory": PROJECT_ROOT / "results/tables/plan_e_sasrec_warning_inventory.csv",
    "report": PROJECT_ROOT / "report/plan_e_sasrec_beauty_sports_and_baseline_comparison.md",
}


def main(argv: list[str] | None = None) -> int:
    del argv
    ensure_outputs_absent()
    validate_training_success()

    all_sasrec = build_all_dataset_sasrec_metrics()
    write_csv(OUTPUTS["all_sasrec"], all_sasrec, ALL_DATASET_COLUMNS)

    traditional_rankings = read_csv(PROJECT_ROOT / "results/tables/traditional_baseline_rankings.csv")
    bprmf_rankings = pd.concat(
        [
            normalize_bprmf_rankings(
                read_csv(PROJECT_ROOT / "results/tables/plan_e_bprmf_beauty_sports_rankings.csv"),
                source_path="results/tables/plan_e_bprmf_beauty_sports_rankings.csv",
            ),
            normalize_bprmf_rankings(
                read_csv(PROJECT_ROOT / "results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv"),
                source_path="results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv",
            ),
        ],
        ignore_index=True,
    )
    gru4rec_rankings = pd.concat(
        [
            normalize_gru4rec_rankings(
                read_csv(PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_rankings.csv"),
                source_path="results/tables/plan_e_gru4rec_beauty_sports_rankings.csv",
            ),
            normalize_gru4rec_rankings(
                read_csv(PROJECT_ROOT / "results/tables/plan_e_gru4rec_ml1m_pilot_rankings.csv"),
                source_path="results/tables/plan_e_gru4rec_ml1m_pilot_rankings.csv",
            ),
        ],
        ignore_index=True,
    )
    sasrec_rankings = pd.concat(
        [
            normalize_sasrec_rankings(
                read_csv(PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_rankings.csv"),
                source_path="results/tables/plan_e_sasrec_beauty_sports_rankings.csv",
            ),
            normalize_sasrec_rankings(
                read_csv(PROJECT_ROOT / "results/tables/plan_e_sasrec_ml1m_pilot_rankings.csv"),
                source_path="results/tables/plan_e_sasrec_ml1m_pilot_rankings.csv",
            ),
        ],
        ignore_index=True,
    )
    cohort_sources = {
        "Amazon Beauty": "results/tables/plan_e_sasrec_beauty_sports_cohort.csv",
        "Amazon Sports": "results/tables/plan_e_sasrec_beauty_sports_cohort.csv",
        "ML-1M": "results/tables/plan_e_sasrec_ml1m_pilot_cohort.csv",
    }
    same_cohort, filtered_frames = build_same_cohort_metrics(
        traditional_rankings=traditional_rankings,
        bprmf_rankings=bprmf_rankings,
        gru4rec_rankings=gru4rec_rankings,
        sasrec_rankings=sasrec_rankings,
        cohort_source_path_by_dataset=cohort_sources,
    )
    validate_metric_ranges(same_cohort)
    if len(same_cohort) != 72:
        raise RuntimeError(f"expected 72 same-cohort metric rows, found {len(same_cohort)}")
    write_csv(OUTPUTS["same_cohort"], same_cohort, SAME_COHORT_METRIC_COLUMNS)

    comparison = build_paired_comparisons(same_cohort, filtered_frames)
    validate_comparison_frame(comparison)
    if len(comparison) != 60:
        raise RuntimeError(f"expected 60 paired comparison rows, found {len(comparison)}")
    write_csv(OUTPUTS["comparison"], comparison, COMPARISON_COLUMNS)

    warnings = build_warning_inventory()
    validate_warning_inventory(warnings)
    write_csv(OUTPUTS["warning_inventory"], warnings, WARNING_COLUMNS)

    write_report(
        OUTPUTS["report"],
        all_sasrec=all_sasrec,
        same_cohort=same_cohort,
        comparison=comparison,
        warnings=warnings,
    )
    validate_written_outputs()
    print("plan_e_sasrec_baseline_consolidation_completed")
    print(f"all_sasrec_rows={len(all_sasrec)}")
    print(f"same_cohort_rows={len(same_cohort)}")
    print(f"comparison_rows={len(comparison)}")
    print(f"warning_rows={len(warnings)}")
    return 0


def ensure_outputs_absent() -> None:
    existing = [str(path) for path in OUTPUTS.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing output(s): " + "; ".join(existing))


def validate_training_success() -> None:
    required = [
        "results/tables/plan_e_sasrec_beauty_sports_item_token_map.csv",
        "results/tables/plan_e_sasrec_beauty_sports_sequence_objective_summary.csv",
        "results/tables/plan_e_sasrec_beauty_sports_cohort.csv",
        "results/tables/plan_e_sasrec_beauty_sports_run_manifest.csv",
        "results/tables/plan_e_sasrec_beauty_sports_training_history.csv",
        "results/tables/plan_e_sasrec_beauty_sports_rankings.csv",
        "results/tables/plan_e_sasrec_beauty_sports_metrics.csv",
        "results/tables/plan_e_sasrec_beauty_sports_determinism.csv",
        "results/tables/plan_e_sasrec_beauty_sports_validation.csv",
        "results/tables/plan_e_sasrec_ml1m_pilot_rankings.csv",
        "results/tables/plan_e_sasrec_ml1m_pilot_metrics.csv",
        "results/tables/plan_e_bprmf_beauty_sports_rankings.csv",
        "results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv",
        "results/tables/plan_e_gru4rec_beauty_sports_rankings.csv",
        "results/tables/plan_e_gru4rec_ml1m_pilot_rankings.csv",
        "results/tables/traditional_baseline_rankings.csv",
    ]
    missing = [path for path in required if not (PROJECT_ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError("missing prerequisite output(s): " + "; ".join(missing))
    determinism = read_csv(PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_determinism.csv")
    validation = read_csv(PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_validation.csv")
    if len(determinism) != 28 or determinism["status"].astype(str).str.lower().ne("passed").any():
        raise RuntimeError("Beauty/Sports SASRec determinism contains a failed or missing check")
    if ((validation["status"].astype(str).str.lower().eq("failed")) & (validation["severity"].astype(str).str.lower().eq("error"))).any():
        raise RuntimeError("Beauty/Sports SASRec validation contains error-level failure")


def build_all_dataset_sasrec_metrics() -> pd.DataFrame:
    sources = [
        ("results/tables/plan_e_sasrec_beauty_sports_metrics.csv", read_csv(PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_metrics.csv")),
        ("results/tables/plan_e_sasrec_ml1m_pilot_metrics.csv", read_csv(PROJECT_ROOT / "results/tables/plan_e_sasrec_ml1m_pilot_metrics.csv")),
    ]
    rows: list[dict[str, Any]] = []
    for source_path, frame in sources:
        for row in frame.to_dict("records"):
            rows.append(
                {
                    "model_id": row["model_id"],
                    "dataset": row["dataset"],
                    "task_type": row["task_type"],
                    "candidate_set_id": row["candidate_set_id"],
                    "cohort_id": row["cohort_id"],
                    "evaluated_rows": row["evaluated_rows"],
                    "candidate_count": row["candidate_count"],
                    "hr_at_5": row["hr_at_5"],
                    "ndcg_at_5": row["ndcg_at_5"],
                    "mean_positive_rank_zero_based": row["mean_positive_rank_zero_based"],
                    "median_positive_rank_zero_based": row["median_positive_rank_zero_based"],
                    "valid_ranking_rate": row["valid_ranking_rate"],
                    "duplicate_ranked_item_rate": row["duplicate_ranked_item_rate"],
                    "missing_score_rate": row["missing_score_rate"],
                    "mean_raw_history_length": row["mean_raw_history_length"],
                    "median_raw_history_length": row["median_raw_history_length"],
                    "mean_effective_history_length": row["mean_effective_history_length"],
                    "median_effective_history_length": row["median_effective_history_length"],
                    "history_truncation_rate": row["history_truncation_rate"],
                    "source_metrics_path": source_path,
                    "paper_alignment": row["paper_alignment"],
                    "notes": "primary-run SASRec metric carried into all-dataset summary",
                }
            )
    frame = pd.DataFrame(rows, columns=ALL_DATASET_COLUMNS).sort_values(
        ["dataset", "task_type", "cohort_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(frame) != 12:
        raise RuntimeError(f"expected 12 all-dataset SASRec metric rows, found {len(frame)}")
    for field in (
        "hr_at_5",
        "ndcg_at_5",
        "valid_ranking_rate",
        "duplicate_ranked_item_rate",
        "missing_score_rate",
        "history_truncation_rate",
    ):
        values = frame[field].astype(float)
        if ((values < 0.0) | (values > 1.0)).any():
            raise RuntimeError(f"{field} outside [0,1] in all-dataset metrics")
    ml1m_source = read_csv(PROJECT_ROOT / "results/tables/plan_e_sasrec_ml1m_pilot_metrics.csv")
    ml1m_join = frame.loc[frame["dataset"].astype(str).eq("ML-1M")].sort_values(["task_type", "cohort_id"]).reset_index(drop=True)
    source_join = ml1m_source.sort_values(["task_type", "cohort_id"]).reset_index(drop=True)
    for field in ("hr_at_5", "ndcg_at_5"):
        if ml1m_join[field].astype(str).tolist() != source_join[field].astype(str).tolist():
            raise RuntimeError(f"ML-1M {field} values do not exactly match Phase 3J")
    return frame


def build_warning_inventory() -> pd.DataFrame:
    records = [
        {
            "dataset": "ALL",
            "execution_role": "focused_tests",
            "stage": "phase_3j_phase_3k_focused_tests",
            "warning_category": "tensor_layout",
            "warning_class": "UserWarning",
            "warning_message": "enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.num_heads is odd",
            "warning_source": "torch/nn/modules/transformer.py",
            "occurrence_count": 7,
            "scientific_correctness_risk": "low",
            "requires_stop": "false",
            "resolution_status": "reviewed_safe",
            "evidence": "focused pytest suite using one-head synthetic SASRec fixtures",
            "notes": "Warning is reproduced by CPU synthetic tests with odd attention heads; Phase 3K project-data runs use two heads.",
        }
    ]
    return aggregate_warning_records(records)


def write_report(
    path: Path,
    *,
    all_sasrec: pd.DataFrame,
    same_cohort: pd.DataFrame,
    comparison: pd.DataFrame,
    warnings: pd.DataFrame,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_run_manifest.csv")
    history = read_csv(PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_training_history.csv")
    cohort = read_csv(PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_cohort.csv")
    sequence = read_csv(PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_sequence_objective_summary.csv")
    token_map = read_csv(PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_item_token_map.csv")
    metrics = read_csv(PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_metrics.csv")
    determinism = read_csv(PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_determinism.csv")
    validation = read_csv(PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_validation.csv")
    rankings = read_csv(PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_rankings.csv")
    beauty = dataset_summary("Amazon Beauty", cohort, sequence, token_map, manifest, history, metrics, rankings, determinism)
    sports = dataset_summary("Amazon Sports", cohort, sequence, token_map, manifest, history, metrics, rankings, determinism)
    report = f"""# Plan E SASRec Beauty and Sports Execution and Same-Cohort Baseline Comparison

## 1. Scope and provenance

- Branch: `{git_output(['branch', '--show-current'])}`.
- Commit: `{git_output(['log', '-1', '--format=%H %s'])}`.
- Scope: controlled Beauty/Sports SASRec execution and same-cohort consolidation with Random, MostPop, BM25-like, BPRMF, GRU4Rec and ML-1M SASRec evidence.

## 2. Executive summary

- Beauty/Sports SASRec ranking rows: `{len(rankings)}`.
- All-dataset SASRec metric rows: `{len(all_sasrec)}`.
- Same-cohort six-baseline metric rows: `{len(same_cohort)}`.
- Paired comparison rows: `{len(comparison)}`.
- Warning inventory rows: `{len(warnings)}`.

## 3. Scientific classification

The runs are `adapted_paper_like_reconstructed`, not exact paper reproduction, because they use reconstructed Plan C data, Plan E2 cohorts and local SASRec code.

## 4. Protected inputs and safety boundaries

The training runner verified protected hashes before and after each dataset. It did not modify Phase 1-3J artifacts, BPRMF or GRU4Rec outputs, candidate manifests, reconstructed interactions, datamaps, title maps, or `original_repo/`.

## 5. CUDA interpreter recovery context

The run uses the recovered repository venv CUDA interpreter after earlier WindowsApps alias failures.

## 6. Absolute interpreter invocation policy

The manifest records `python_executable={single_value(manifest, 'python_executable')}` and `PYTHONNOUSERSITE=1` was required by the runner.

## 7. Controlled configuration

Both datasets use seed `42`, embedding and hidden size `64`, two attention heads, two Transformer blocks, max sequence length `50`, batch size `512`, learning rate `0.001`, five epochs and deterministic CUDA algorithms.

## 8. Execution environment

{environment_line(manifest)}

## 9. Warning inventory

Warning rows: `{len(warnings)}`; blocking warning rows: `{int(warnings['requires_stop'].astype(str).str.lower().eq('true').sum())}`.

## 10. Beauty training data

{beauty['training']}

## 11. Sports training data

{sports['training']}

## 12. Beauty chronology and timestamp ties

{beauty['chronology']}

## 13. Sports chronology and timestamp ties

{sports['chronology']}

## 14. SASRec objective construction

Beauty examples: `{beauty['training_examples']}`. Sports examples: `{sports['training_examples']}`. Both use all-prefix next-item objectives from train-only histories.

## 15. Negative-sampling contract

The retained manifest hashes `negative_sampling_evidence_sha256` for primary and repeat runs; negatives are sampled by the deterministic Plan D user-positive exclusion sampler.

## 16. Item-token mappings

Beauty token rows: `{beauty['token_rows']}`. Sports token rows: `{sports['token_rows']}`. Token `0` is absent from real item rows.

## 17. Padding semantics

Left padding uses index `0`; real item tokens are dense train indices plus one.

## 18. Absolute positional embeddings

Position embedding cardinality is `50`, matching the maximum sequence length.

## 19. Causal and padding masks

Strict cohort rows require `causal_mask_verified=true` and `padding_mask_verified=true`; scoring rows also passed finite-score validation.

## 20. Strict SASRec cohorts

Beauty scored rows: `{beauty['scored']}`. Sports scored rows: `{sports['scored']}`. Combined excluded rows: `{int(beauty['excluded']) + int(sports['excluded'])}`.

## 21. Beauty exclusions

Beauty excluded rows: `{beauty['excluded']}`; exclusion reasons are from `results/tables/plan_e_sasrec_beauty_sports_cohort.csv`.

## 22. Sports exclusions

Sports excluded rows: `{sports['excluded']}`; exclusion reasons are from `results/tables/plan_e_sasrec_beauty_sports_cohort.csv`.

## 23. Beauty primary training

{beauty['primary']}

## 24. Beauty same-seed repeat

{beauty['repeat']}

## 25. Sports primary training

{sports['primary']}

## 26. Sports same-seed repeat

{sports['repeat']}

## 27. Runtime and GPU memory

Combined training-history rows: `{len(history)}`. Peak reserved bytes by dataset are Beauty `{beauty['peak_reserved']}` and Sports `{sports['peak_reserved']}`.

## 28. Checkpoints and metadata

Primary checkpoints retained: `{manifest['checkpoint_path'].astype(str).ne('').sum()}`; repeat checkpoints retained: `{manifest.loc[manifest['execution_role'].eq('same_seed_repeat'), 'checkpoint_path'].astype(str).ne('').sum()}`.

## 29. Beauty ranking results

{metric_line(metrics, 'Amazon Beauty', 'ranking', FULL_COHORT_ID)}

## 30. Beauty reranking-candidate results

{metric_line(metrics, 'Amazon Beauty', 'reranking', FULL_COHORT_ID)}

## 31. Sports ranking results

{metric_line(metrics, 'Amazon Sports', 'ranking', FULL_COHORT_ID)}

## 32. Sports reranking-candidate results

{metric_line(metrics, 'Amazon Sports', 'reranking', FULL_COHORT_ID)}

## 33. Beauty full strict metrics

{metric_pair_line(metrics, 'Amazon Beauty', FULL_COHORT_ID)}

## 34. Beauty matched strict metrics

{metric_pair_line(metrics, 'Amazon Beauty', MATCHED_COHORT_ID)}

## 35. Sports full strict metrics

{metric_pair_line(metrics, 'Amazon Sports', FULL_COHORT_ID)}

## 36. Sports matched strict metrics

{metric_pair_line(metrics, 'Amazon Sports', MATCHED_COHORT_ID)}

## 37. ML-1M Phase 3J integration

ML-1M rows in the all-dataset table: `{len(all_sasrec.loc[all_sasrec['dataset'].eq('ML-1M')])}`. HR@5 and NDCG@5 were checked for exact carry-over from Phase 3J.

## 38. All-dataset SASRec summary

All-dataset rows: `{len(all_sasrec)}` across datasets `{', '.join(sorted(all_sasrec['dataset'].unique()))}`.

## 39. History and truncation comparison

Beauty strict rows with truncation: `{beauty['truncated_rows']}`. Sports strict rows with truncation: `{sports['truncated_rows']}`.

## 40. Same-cohort six-baseline reconstruction

Same-cohort rows: `{len(same_cohort)}` for six baselines, three datasets, two tasks and two SASRec cohorts.

## 41. Random versus SASRec

{comparison_summary(comparison, 'random_seeded')}

## 42. MostPop versus SASRec

{comparison_summary(comparison, 'mostpop_train_only')}

## 43. BM25-like versus SASRec

{comparison_summary(comparison, 'bm25_like_e3b')}

## 44. BPRMF versus SASRec

{comparison_summary(comparison, 'bprmf_reconstructed')}

## 45. GRU4Rec versus SASRec

{comparison_summary(comparison, 'gru4rec_reconstructed')}

## 46. Dataset-specific findings

Beauty same-cohort rows: `{len(same_cohort.loc[same_cohort['dataset'].eq('Amazon Beauty')])}`. Sports same-cohort rows: `{len(same_cohort.loc[same_cohort['dataset'].eq('Amazon Sports')])}`. ML-1M same-cohort rows: `{len(same_cohort.loc[same_cohort['dataset'].eq('ML-1M')])}`.

## 47. Ranking versus reranking interpretation

Ranking rows in comparison: `{len(comparison.loc[comparison['task_type'].eq('ranking')])}`. Reranking rows in comparison: `{len(comparison.loc[comparison['task_type'].eq('reranking')])}`.

## 48. Paired positive-rank findings

Across paired rows, SASRec-better counts sum to `{int(comparison['sasrec_better_positive_rank_rows'].astype(int).sum())}`, comparator-better counts sum to `{int(comparison['comparator_better_positive_rank_rows'].astype(int).sum())}`, and equal counts sum to `{int(comparison['equal_positive_rank_rows'].astype(int).sum())}`.

## 49. Candidate and row identity validation

All comparison rows have `exact_row_key_match=true` and `exact_candidate_list_match=true`.

## 50. Determinism findings

Determinism rows: `{len(determinism)}`; failed rows: `{int(determinism['status'].astype(str).str.lower().eq('failed').sum())}`.

## 51. Warning interpretation

The warning inventory records the reproducible focused-test odd-head Transformer warning and classifies it as low risk for Phase 3K project-data runs, which use two attention heads.

## 52. WindowsApps interpreter limitation

Relative `python` and WindowsApps alias execution remain outside this phase's allowed execution policy.

## 53. Absolute-position reconstruction limitation

Absolute positions are reconstructed from left-padded train histories; this is adapted evidence, not verified original-paper SASRec internals.

## 54. Equal-timestamp reconstruction limitation

Equal timestamp groups are ordered by frozen source row index; this is deterministic but reconstructed.

## 55. Deviations from the original paper

The run uses reconstructed data, local SASRec, strict 20-candidate Plan E2 cohorts and same-cohort adapted comparisons.

## 56. Limitations

No LightGCN, LLM inference, API call, prompt generation, or significance testing was performed.

## 57. Whether Phase 3K succeeded

Phase 3K succeeded if all generated validation rows pass and no blocking warnings are present. Current validation failed rows: `{int(validation['status'].astype(str).str.lower().eq('failed').sum())}`.

## 58. SASRec completion assessment

Beauty, Sports and ML-1M now have SASRec adapted Plan E2 strict-cohort metrics in `results/tables/plan_e_sasrec_all_datasets_metrics.csv`.

## 59. Conditions for beginning LightGCN

The next phase can begin only if these artifacts remain unchanged and the same candidate/cohort identity policy is reused.

## 60. Protected-input immutability

Protected input immutability was checked in the runner and final validation; Phase 2A candidate files were read-only for this phase.

## 61. Commands executed

Commands are recorded in the final response; generated CSVs contain executable, config and source-path provenance.

## 62. Recommended single next task

Phase 3L — LightGCN Plan E2 adapter and graph-construction preflight validation
"""
    path.write_text(report, encoding="utf-8")


def validate_written_outputs() -> None:
    checks = [
        ("all_sasrec", ALL_DATASET_COLUMNS, 12),
        ("same_cohort", SAME_COHORT_METRIC_COLUMNS, 72),
        ("comparison", COMPARISON_COLUMNS, 60),
        ("warning_inventory", WARNING_COLUMNS, None),
    ]
    for key, columns, expected_rows in checks:
        frame = read_csv(OUTPUTS[key])
        if list(frame.columns) != list(columns):
            raise RuntimeError(f"{key} schema mismatch")
        if expected_rows is not None and len(frame) != expected_rows:
            raise RuntimeError(f"{key} expected {expected_rows} rows, found {len(frame)}")
    validate_metric_ranges(read_csv(OUTPUTS["same_cohort"]))
    validate_comparison_frame(read_csv(OUTPUTS["comparison"]))
    validate_warning_inventory(read_csv(OUTPUTS["warning_inventory"]))
    warnings = read_csv(OUTPUTS["warning_inventory"])
    if warnings["requires_stop"].astype(str).str.lower().eq("true").any():
        raise RuntimeError("blocking warning present")
    if not OUTPUTS["report"].exists():
        raise RuntimeError("report was not created")


def dataset_summary(
    dataset: str,
    cohort: pd.DataFrame,
    sequence: pd.DataFrame,
    token_map: pd.DataFrame,
    manifest: pd.DataFrame,
    history: pd.DataFrame,
    metrics: pd.DataFrame,
    rankings: pd.DataFrame,
    determinism: pd.DataFrame,
) -> dict[str, Any]:
    c = cohort.loc[cohort["dataset"].eq(dataset)].copy()
    s = sequence.loc[sequence["dataset"].eq(dataset)].copy()
    t = token_map.loc[token_map["dataset"].eq(dataset)].copy()
    m = manifest.loc[manifest["dataset"].eq(dataset)].copy()
    h = history.loc[history["dataset"].eq(dataset)].copy()
    r = rankings.loc[rankings["dataset"].eq(dataset)].copy()
    d = determinism.loc[determinism["dataset"].eq(dataset)].copy()
    primary = m.loc[m["execution_role"].eq("primary")].iloc[0]
    repeat = m.loc[m["execution_role"].eq("same_seed_repeat")].iloc[0]
    return {
        "training": f"`training_interactions={primary['training_interactions']}`, `training_users={primary['training_users']}`, `training_items={primary['training_items']}`.",
        "chronology": f"Equal timestamp groups from sequence summary total `{int(s['equal_timestamp_group_count'].astype(int).sum())}` with `{int(s['events_in_equal_timestamp_groups'].astype(int).sum())}` affected events.",
        "training_examples": primary["training_examples"],
        "token_rows": len(t),
        "scored": int(c["included_in_sasrec_scoring"].astype(str).str.lower().eq("true").sum()),
        "excluded": int(c["included_in_sasrec_scoring"].astype(str).str.lower().eq("false").sum()),
        "primary": f"Run `{primary['training_run_id']}` completed in `{float(primary['duration_seconds']):.3f}` seconds with checkpoint `{primary['checkpoint_path']}`.",
        "repeat": f"Run `{repeat['training_run_id']}` completed in `{float(repeat['duration_seconds']):.3f}` seconds and retained no checkpoint.",
        "peak_reserved": int(pd.to_numeric(h["cuda_peak_memory_reserved_bytes"], errors="raise").max()),
        "truncated_rows": int(r["history_truncated"].astype(str).str.lower().eq("true").sum()),
        "determinism_passed": int(d["status"].astype(str).str.lower().eq("passed").sum()),
    }


def metric_line(metrics: pd.DataFrame, dataset: str, task: str, cohort_id: str) -> str:
    row = metrics.loc[
        metrics["dataset"].eq(dataset)
        & metrics["task_type"].eq(task)
        & metrics["cohort_id"].eq(cohort_id)
    ].iloc[0]
    return (
        f"`evaluated_rows={row['evaluated_rows']}`, `HR@5={float(row['hr_at_5']):.6f}`, "
        f"`NDCG@5={float(row['ndcg_at_5']):.6f}`, "
        f"`mean_positive_rank_zero_based={float(row['mean_positive_rank_zero_based']):.6f}`."
    )


def metric_pair_line(metrics: pd.DataFrame, dataset: str, cohort_id: str) -> str:
    ranking = metric_line(metrics, dataset, "ranking", cohort_id)
    reranking = metric_line(metrics, dataset, "reranking", cohort_id)
    return f"Ranking: {ranking} Reranking: {reranking}"


def comparison_summary(comparison: pd.DataFrame, comparator: str) -> str:
    rows = comparison.loc[comparison["comparator_baseline_id"].eq(comparator)].copy()
    return (
        f"Rows: `{len(rows)}`; mean HR difference SASRec minus comparator "
        f"`{rows['hr_at_5_difference_sasrec_minus_comparator'].astype(float).mean():.6f}`; "
        f"mean NDCG difference `{rows['ndcg_at_5_difference_sasrec_minus_comparator'].astype(float).mean():.6f}`."
    )


def environment_line(manifest: pd.DataFrame) -> str:
    first = manifest.iloc[0]
    return (
        f"`environment_id={first['environment_id']}`, Python `{first['python_version']}`, "
        f"torch `{first['torch_version']}`, CUDA `{first['torch_cuda_version']}`, device `{first['cuda_device']}`."
    )


def single_value(frame: pd.DataFrame, column: str) -> str:
    values = sorted(set(frame[column].astype(str).tolist()))
    if len(values) != 1:
        raise RuntimeError(f"expected one value in {column}, found {values}")
    return values[0]


def git_output(args: list[str]) -> str:
    try:
        return common_git.git_stdout_checked(args, PROJECT_ROOT, stderr_to_devnull=True)
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
