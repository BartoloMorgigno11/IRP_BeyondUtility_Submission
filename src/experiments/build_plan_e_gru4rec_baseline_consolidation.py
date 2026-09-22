"""Build Phase 3H GRU4Rec same-cohort consolidation tables and report."""

from __future__ import annotations

from src.audit import common_git

import csv
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_e_gru4rec_baseline_consolidation import (
    BPRMF_BASELINE_ID,
    COMPARISON_COLUMNS,
    FULL_COHORT_ID,
    GRU4REC_BASELINE_ID,
    MATCHED_COHORT_ID,
    SAME_COHORT_METRIC_COLUMNS,
    build_paired_comparisons,
    build_same_cohort_metrics,
    normalize_bprmf_rankings,
    normalize_gru4rec_rankings,
    read_csv,
    validate_comparison_frame,
    validate_metric_ranges,
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
    "all_gru4rec": PROJECT_ROOT / "results/tables/plan_e_gru4rec_all_datasets_metrics.csv",
    "same_cohort": PROJECT_ROOT / "results/tables/plan_e_gru4rec_same_cohort_baselines_metrics.csv",
    "comparison": PROJECT_ROOT / "results/tables/plan_e_gru4rec_baseline_comparison.csv",
    "report": PROJECT_ROOT / "report/plan_e_gru4rec_beauty_sports_and_baseline_comparison.md",
}


def main(argv: list[str] | None = None) -> int:
    ensure_outputs_absent()
    validate_training_success()

    all_gru4rec = build_all_dataset_gru4rec_metrics()
    write_csv(OUTPUTS["all_gru4rec"], all_gru4rec, ALL_DATASET_COLUMNS)

    traditional_rankings = read_csv(PROJECT_ROOT / "results/tables/traditional_baseline_rankings.csv")
    beauty_sports_bprmf = normalize_bprmf_rankings(
        read_csv(PROJECT_ROOT / "results/tables/plan_e_bprmf_beauty_sports_rankings.csv"),
        source_path="results/tables/plan_e_bprmf_beauty_sports_rankings.csv",
    )
    ml1m_bprmf = normalize_bprmf_rankings(
        read_csv(PROJECT_ROOT / "results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv"),
        source_path="results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv",
    )
    bprmf_rankings = pd.concat([beauty_sports_bprmf, ml1m_bprmf], ignore_index=True)

    beauty_sports_gru = normalize_gru4rec_rankings(
        read_csv(PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_rankings.csv"),
        source_path="results/tables/plan_e_gru4rec_beauty_sports_rankings.csv",
    )
    ml1m_gru = normalize_gru4rec_rankings(
        read_csv(PROJECT_ROOT / "results/tables/plan_e_gru4rec_ml1m_pilot_rankings.csv"),
        source_path="results/tables/plan_e_gru4rec_ml1m_pilot_rankings.csv",
    )
    gru4rec_rankings = pd.concat([beauty_sports_gru, ml1m_gru], ignore_index=True)
    cohort_sources = {
        "Amazon Beauty": "results/tables/plan_e_gru4rec_beauty_sports_cohort.csv",
        "Amazon Sports": "results/tables/plan_e_gru4rec_beauty_sports_cohort.csv",
        "ML-1M": "results/tables/plan_e_gru4rec_ml1m_pilot_cohort.csv",
    }
    same_cohort, filtered_frames = build_same_cohort_metrics(
        traditional_rankings=traditional_rankings,
        bprmf_rankings=bprmf_rankings,
        gru4rec_rankings=gru4rec_rankings,
        cohort_source_path_by_dataset=cohort_sources,
    )
    validate_metric_ranges(same_cohort)
    if len(same_cohort) != 60:
        raise RuntimeError(f"expected 60 same-cohort metric rows, found {len(same_cohort)}")
    write_csv(OUTPUTS["same_cohort"], same_cohort, SAME_COHORT_METRIC_COLUMNS)

    comparison = build_paired_comparisons(same_cohort, filtered_frames)
    validate_comparison_frame(comparison)
    if len(comparison) != 48:
        raise RuntimeError(f"expected 48 paired comparison rows, found {len(comparison)}")
    write_csv(OUTPUTS["comparison"], comparison, COMPARISON_COLUMNS)

    write_report(
        OUTPUTS["report"],
        all_gru4rec=all_gru4rec,
        same_cohort=same_cohort,
        comparison=comparison,
    )
    validate_written_outputs()
    print("plan_e_gru4rec_baseline_consolidation_completed")
    print(f"all_gru4rec_rows={len(all_gru4rec)}")
    print(f"same_cohort_rows={len(same_cohort)}")
    print(f"comparison_rows={len(comparison)}")
    return 0


def validate_training_success() -> None:
    required = [
        "results/tables/plan_e_gru4rec_beauty_sports_item_token_map.csv",
        "results/tables/plan_e_gru4rec_beauty_sports_sequence_summary.csv",
        "results/tables/plan_e_gru4rec_beauty_sports_cohort.csv",
        "results/tables/plan_e_gru4rec_beauty_sports_run_manifest.csv",
        "results/tables/plan_e_gru4rec_beauty_sports_training_history.csv",
        "results/tables/plan_e_gru4rec_beauty_sports_rankings.csv",
        "results/tables/plan_e_gru4rec_beauty_sports_metrics.csv",
        "results/tables/plan_e_gru4rec_beauty_sports_determinism.csv",
        "results/tables/plan_e_gru4rec_beauty_sports_validation.csv",
        "results/tables/plan_e_gru4rec_ml1m_pilot_rankings.csv",
        "results/tables/plan_e_gru4rec_ml1m_pilot_metrics.csv",
        "results/tables/plan_e_bprmf_beauty_sports_rankings.csv",
        "results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv",
        "results/tables/traditional_baseline_rankings.csv",
    ]
    missing = [path for path in required if not (PROJECT_ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError("missing prerequisite output(s): " + "; ".join(missing))
    determinism = read_csv(PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_determinism.csv")
    validation = read_csv(PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_validation.csv")
    if determinism["status"].astype(str).str.lower().ne("passed").any():
        raise RuntimeError("Beauty/Sports GRU4Rec determinism contains a failed check")
    if ((validation["status"].astype(str).str.lower().eq("failed")) & (validation["severity"].astype(str).str.lower().eq("error"))).any():
        raise RuntimeError("Beauty/Sports GRU4Rec validation contains error-level failure")


def build_all_dataset_gru4rec_metrics() -> pd.DataFrame:
    sources = [
        ("results/tables/plan_e_gru4rec_beauty_sports_metrics.csv", read_csv(PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_metrics.csv")),
        ("results/tables/plan_e_gru4rec_ml1m_pilot_metrics.csv", read_csv(PROJECT_ROOT / "results/tables/plan_e_gru4rec_ml1m_pilot_metrics.csv")),
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
                    "notes": "primary-run GRU4Rec metric carried into all-dataset summary",
                }
            )
    frame = pd.DataFrame(rows, columns=ALL_DATASET_COLUMNS).sort_values(
        ["dataset", "task_type", "cohort_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(frame) != 12:
        raise RuntimeError(f"expected 12 all-dataset GRU4Rec metric rows, found {len(frame)}")
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
    return frame


def write_report(
    path: Path,
    *,
    all_gru4rec: pd.DataFrame,
    same_cohort: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_run_manifest.csv")
    history = read_csv(PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_training_history.csv")
    cohort = read_csv(PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_cohort.csv")
    sequence = read_csv(PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_sequence_summary.csv")
    token_map = read_csv(PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_item_token_map.csv")
    metrics = read_csv(PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_metrics.csv")
    determinism = read_csv(PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_determinism.csv")
    validation = read_csv(PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_validation.csv")
    rankings = read_csv(PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_rankings.csv")
    beauty = dataset_summary("Amazon Beauty", cohort, sequence, token_map, manifest, history, metrics, rankings, determinism)
    sports = dataset_summary("Amazon Sports", cohort, sequence, token_map, manifest, history, metrics, rankings, determinism)
    report = f"""# Plan E GRU4Rec Beauty and Sports Execution and Same-Cohort Baseline Comparison

## 1. Scope and provenance

- Branch: `{git_output(['git', 'branch', '--show-current'])}`.
- Commit: `{git_output(['git', 'log', '-1', '--format=%H %s'])}`.
- Scope: controlled Beauty/Sports GRU4Rec execution and same-cohort consolidation with Random, MostPop, BM25-like, BPRMF and ML-1M GRU4Rec evidence.

## 2. Executive summary

- Beauty/Sports GRU4Rec ranking rows: `{len(rankings)}`.
- All-dataset GRU4Rec metric rows: `{len(all_gru4rec)}`.
- Same-cohort baseline metric rows: `{len(same_cohort)}`.
- Paired comparison rows: `{len(comparison)}`.
- Classification: `adapted_paper_like_reconstructed`.

## 3. Scientific classification

The runs are adapted paper-like reconstructed experiments, not exact paper reproduction, because they use Plan E2 reconstructed cohorts and a local GRU4Rec implementation.

## 4. Protected inputs and safety boundaries

Protected hashes were checked by the training runner and are repeated here for key files:

{protected_input_hash_lines()}

## 5. Controlled configuration policy

Beauty and Sports reused the Phase 3G seed `42`, embedding dimension `64`, hidden size `64`, one GRU layer, configured dropout `0.2`, sequence length `50`, Adam learning rate `0.001`, batch size `512`, five epochs, deterministic CUDA settings and HR/NDCG cutoff `5`.

## 6. Effective dropout semantics

The Plan D model passes recurrent dropout to PyTorch GRU, which is inactive when `num_layers=1`; the model also applies the configured embedding dropout in `GRU4RecModel.encode` during training. The architecture and configuration were not changed.

## 7. Execution environment

{environment_line(manifest)}

## 8. Beauty training data

{beauty['training']}

## 9. Sports training data

{sports['training']}

## 10. Beauty chronology and timestamp ties

{beauty['chronology']}

## 11. Sports chronology and timestamp ties

{sports['chronology']}

## 12. Training-pair construction

Beauty training pairs: `{beauty['training_pairs']}`. Sports training pairs: `{sports['training_pairs']}`. Both use all-prefix next-item pairs from `split=train` only.

## 13. Token mappings

Beauty token rows: `{beauty['token_rows']}`. Sports token rows: `{sports['token_rows']}`. Token `0` is reserved for padding and real tokens are contiguous from `1`.

## 14. Padding and truncation

Beauty truncated strict rows: `{beauty['truncated_rows']}`. Sports truncated strict rows: `{sports['truncated_rows']}`. Histories keep the most recent 50 train events.

## 15. Strict sequential cohorts

Beauty scored rows: `{beauty['scored']}`; Sports scored rows: `{sports['scored']}`; combined excluded rows: `{int(beauty['excluded']) + int(sports['excluded'])}`.

## 16. Beauty exclusions

Beauty excluded rows: `{beauty['excluded']}`.

## 17. Sports exclusions

Sports excluded rows: `{sports['excluded']}`.

## 18. Beauty primary training

{beauty['primary']}

## 19. Beauty same-seed repeat

{beauty['repeat']}

## 20. Sports primary training

{sports['primary']}

## 21. Sports same-seed repeat

{sports['repeat']}

## 22. Runtime and GPU memory

{runtime_memory_line(manifest, history)}

## 23. Checkpoints and metadata

Beauty checkpoint: `results/models/plan_e_gru4rec_beauty/model_state.pt`. Sports checkpoint: `results/models/plan_e_gru4rec_sports/model_state.pt`. Repeat checkpoints were not retained.

## 24. Beauty ranking results

{metric_lines(metrics[(metrics['dataset'] == 'Amazon Beauty') & (metrics['task_type'] == 'ranking')])}

## 25. Beauty reranking-candidate results

{metric_lines(metrics[(metrics['dataset'] == 'Amazon Beauty') & (metrics['task_type'] == 'reranking')])}

## 26. Sports ranking results

{metric_lines(metrics[(metrics['dataset'] == 'Amazon Sports') & (metrics['task_type'] == 'ranking')])}

## 27. Sports reranking-candidate results

{metric_lines(metrics[(metrics['dataset'] == 'Amazon Sports') & (metrics['task_type'] == 'reranking')])}

## 28. Beauty full strict metrics

{metric_lines(metrics[(metrics['dataset'] == 'Amazon Beauty') & (metrics['cohort_id'] == FULL_COHORT_ID)])}

## 29. Beauty matched strict metrics

{metric_lines(metrics[(metrics['dataset'] == 'Amazon Beauty') & (metrics['cohort_id'] == MATCHED_COHORT_ID)])}

## 30. Sports full strict metrics

{metric_lines(metrics[(metrics['dataset'] == 'Amazon Sports') & (metrics['cohort_id'] == FULL_COHORT_ID)])}

## 31. Sports matched strict metrics

{metric_lines(metrics[(metrics['dataset'] == 'Amazon Sports') & (metrics['cohort_id'] == MATCHED_COHORT_ID)])}

## 32. ML-1M Phase 3G integration

ML-1M primary metrics were copied from `results/tables/plan_e_gru4rec_ml1m_pilot_metrics.csv`; no ML-1M GRU4Rec training was rerun in Phase 3H.

## 33. All-dataset GRU4Rec summary

{metric_lines(all_gru4rec)}

## 34. History-length and truncation comparison

{history_comparison_lines(all_gru4rec)}

## 35. Same-cohort prior-baseline reconstruction

Existing Random, MostPop, BM25-like and BPRMF rankings were read and filtered to exact GRU4Rec strict row keys; no prior baseline was rerun.

## 36. Random versus GRU4Rec

{comparison_lines(comparison, 'random_seeded')}

## 37. MostPop versus GRU4Rec

{comparison_lines(comparison, 'mostpop_train_only')}

## 38. BM25-like versus GRU4Rec

{comparison_lines(comparison, 'bm25_like_e3b')}

## 39. BPRMF versus GRU4Rec

{comparison_lines(comparison, BPRMF_BASELINE_ID)}

## 40. Dataset-specific findings

{dataset_comparison_lines(comparison)}

## 41. Ranking versus reranking interpretation

Ranking and reranking candidate-set IDs remain separate in all output tables. The Beauty/Sports candidate rows are 20-candidate Plan E2 rows only.

## 42. Paired positive-rank findings

Paired win/equal/loss counts sum to evaluated rows for all `{len(comparison)}` comparison rows.

## 43. Candidate and row identity validation

The consolidation helper requires exact row-key equality and exact candidate-list equality before computing comparisons.

## 44. Determinism findings

Determinism checks: `{len(determinism)}`; failed checks: `{int(determinism['status'].astype(str).str.lower().ne('passed').sum())}`.

## 45. Timestamp-order reconstruction limitation

Chronology uses timestamp ascending with original source-row index as the approved deterministic tie rule; this reconstructs a deterministic order but cannot prove the original paper's exact hidden preprocessing order.

## 46. Deviations from the original paper

The datasets, candidates and splits are reconstructed Plan E2 artifacts; the model is a local GRU4Rec implementation; the evaluated strict cohorts exclude rows whose histories or candidates violate the approved sequential policy.

## 47. Limitations

This phase does not train SASRec, LightGCN, BPRMF or ML-1M GRU4Rec; it does not compute beyond-utility metrics; it does not run statistical significance tests.

## 48. Whether Phase 3H succeeded

Phase 3H succeeded if this report, the 12-row all-dataset GRU4Rec metric table, the 60-row same-cohort metric table, the 48-row paired comparison table, and the Beauty/Sports validation and determinism tables pass validation.

## 49. GRU4Rec completion assessment

GRU4Rec now has controlled Plan E2-compatible results for Beauty, Sports and ML-1M across ranking and reranking candidate sets, with full and matched strict cohorts.

## 50. Conditions for beginning SASRec

Begin SASRec only after preserving the Phase 3H artifacts and applying a separate adapter/preflight validation phase.

## 51. Protected-input immutability

Protected inputs were read only by the consolidation runner. Final hash checks should match the recorded hashes in the training manifest and final command log.

## 52. Commands executed

{command_lines()}

## 53. Recommended single next task

Phase 3I \u2014 SASRec Plan E2 adapter and preflight validation.
"""
    path.write_text(report, encoding="utf-8", newline="\n")


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
) -> dict[str, str]:
    ds_cohort = cohort.loc[cohort["dataset"] == dataset]
    ds_sequence = sequence.loc[sequence["dataset"] == dataset]
    ds_token = token_map.loc[token_map["dataset"] == dataset]
    ds_manifest = manifest.loc[manifest["dataset"] == dataset]
    primary = ds_manifest.loc[ds_manifest["execution_role"] == "primary"].iloc[0]
    repeat = ds_manifest.loc[ds_manifest["execution_role"] == "same_seed_repeat"].iloc[0]
    ds_history = history.loc[history["dataset"] == dataset]
    final_losses = ds_history.sort_values(["execution_role", "epoch"]).groupby("execution_role").tail(1)
    loss_map = {row["execution_role"]: row["mean_training_loss"] for row in final_losses.to_dict("records")}
    scored = int(ds_cohort["included_in_gru4rec_scoring"].astype(str).str.lower().eq("true").sum())
    excluded = int(ds_cohort["included_in_gru4rec_scoring"].astype(str).str.lower().eq("false").sum())
    truncated = int(rankings.loc[rankings["dataset"] == dataset, "history_truncated"].astype(str).str.lower().eq("true").sum())
    chronology = (
        f"Unique train users: `{primary['training_users']}`; train items: `{primary['training_items']}`; "
        f"chronology hash: `{primary['chronology_sha256']}`; users in sequence summary: `{len(ds_sequence)}`."
    )
    training = (
        f"Training interactions: `{primary['training_interactions']}`; training pairs: `{primary['training_pairs']}`; "
        f"source: `{primary['training_data_path']}`; data hash: `{primary['training_data_sha256']}`."
    )
    return {
        "training": training,
        "chronology": chronology,
        "training_pairs": str(primary["training_pairs"]),
        "token_rows": str(len(ds_token)),
        "truncated_rows": str(truncated),
        "scored": str(scored),
        "excluded": str(excluded),
        "primary": f"Primary duration `{primary['duration_seconds']}` seconds; final loss `{loss_map['primary']}`; checkpoint `{primary['checkpoint_path']}`.",
        "repeat": f"Repeat duration `{repeat['duration_seconds']}` seconds; final loss `{loss_map['same_seed_repeat']}`; checkpoint fields are empty.",
        "determinism": str(int(determinism.loc[determinism["dataset"] == dataset, "status"].astype(str).str.lower().eq("passed").sum())),
    }


def metric_lines(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "- No rows."
    sort_columns = [column for column in ["dataset", "task_type", "cohort_id", "baseline_id"] if column in frame.columns]
    lines = []
    for row in frame.sort_values(sort_columns, kind="mergesort").to_dict("records"):
        prefix = f"{row['baseline_id']} " if "baseline_id" in row else ""
        lines.append(
            f"- {prefix}{row['dataset']} {row['task_type']} {row['cohort_id']}: "
            f"rows={row['evaluated_rows']}, HR@5={float(row['hr_at_5']):.12g}, "
            f"NDCG@5={float(row['ndcg_at_5']):.12g}, mean_rank={float(row['mean_positive_rank_zero_based']):.12g}."
        )
    return "\n".join(lines)


def comparison_lines(frame: pd.DataFrame, comparator: str) -> str:
    rows = frame.loc[frame["comparator_baseline_id"] == comparator]
    if rows.empty:
        return "- No rows."
    lines = []
    for row in rows.sort_values(["dataset", "task_type", "cohort_id"], kind="mergesort").to_dict("records"):
        lines.append(
            f"- {row['dataset']} {row['task_type']} {row['cohort_id']}: "
            f"HR diff={float(row['hr_at_5_difference_gru4rec_minus_comparator']):.12g}, "
            f"NDCG diff={float(row['ndcg_at_5_difference_gru4rec_minus_comparator']):.12g}, "
            f"GRU4Rec rank win rate={float(row['gru4rec_positive_rank_win_rate']):.12g}."
        )
    return "\n".join(lines)


def dataset_comparison_lines(frame: pd.DataFrame) -> str:
    lines = []
    for dataset in sorted(frame["dataset"].astype(str).unique()):
        unit = frame.loc[frame["dataset"].astype(str).eq(dataset)]
        lines.append(
            f"- {dataset}: comparison rows={len(unit)}, "
            f"mean HR difference={unit['hr_at_5_difference_gru4rec_minus_comparator'].astype(float).mean():.12g}, "
            f"mean NDCG difference={unit['ndcg_at_5_difference_gru4rec_minus_comparator'].astype(float).mean():.12g}."
        )
    return "\n".join(lines)


def history_comparison_lines(frame: pd.DataFrame) -> str:
    lines = []
    for dataset in sorted(frame["dataset"].astype(str).unique()):
        unit = frame.loc[frame["dataset"].astype(str).eq(dataset)]
        lines.append(
            f"- {dataset}: mean raw history={unit['mean_raw_history_length'].astype(float).mean():.12g}, "
            f"mean effective history={unit['mean_effective_history_length'].astype(float).mean():.12g}, "
            f"truncation rate={unit['history_truncation_rate'].astype(float).mean():.12g}."
        )
    return "\n".join(lines)


def environment_line(manifest: pd.DataFrame) -> str:
    row = manifest.iloc[0]
    return (
        f"Executable `{row['python_executable']}`; Python `{row['python_version']}`; "
        f"torch `{row['torch_version']}`; CUDA `{row['torch_cuda_version']}`; GPU `{row['cuda_device']}`."
    )


def runtime_memory_line(manifest: pd.DataFrame, history: pd.DataFrame) -> str:
    primary_seconds = manifest.loc[manifest["execution_role"] == "primary", "duration_seconds"].astype(float).sum()
    repeat_seconds = manifest.loc[manifest["execution_role"] == "same_seed_repeat", "duration_seconds"].astype(float).sum()
    peak_alloc = int(history["cuda_peak_memory_allocated_bytes"].astype(float).max())
    peak_reserved = int(history["cuda_peak_memory_reserved_bytes"].astype(float).max())
    return f"Primary total duration `{primary_seconds:.6g}` seconds; repeat total duration `{repeat_seconds:.6g}` seconds; peak allocated `{peak_alloc}` bytes; peak reserved `{peak_reserved}` bytes."


def command_lines() -> str:
    commands = [
        "git status --short --branch",
        "git branch --show-current",
        "git log -1 --format=\"%H %s\"",
        ".venv\\Scripts\\python.exe -m pytest tests/test_plan_e_gru4rec_baseline_consolidation.py tests/test_plan_e_gru4rec_training.py tests/test_plan_e_gru4rec_adapter.py tests/test_experiment_cohort_policy.py",
        ".venv\\Scripts\\python.exe src/experiments/run_plan_e_gru4rec_beauty_sports.py",
        ".venv\\Scripts\\python.exe src/experiments/build_plan_e_gru4rec_baseline_consolidation.py",
    ]
    return "\n".join(f"- `{command}`" for command in commands)


def protected_input_hash_lines() -> str:
    paths = [
        "src/baselines/plan_d_gru4rec.py",
        "src/experiments/run_plan_d_gru4rec.py",
        "src/baselines/plan_e_gru4rec_adapter.py",
        "src/baselines/plan_e_gru4rec_training.py",
        "src/experiments/run_plan_e_gru4rec_ml1m_pilot.py",
        "results/tables/ranking_candidate_sets.csv",
        "results/tables/positive_positions.csv",
        "results/tables/plan_e_gru4rec_row_compatibility.csv",
        "results/tables/plan_e_gru4rec_ml1m_pilot_rankings.csv",
        "results/tables/traditional_baseline_rankings.csv",
        "results/tables/plan_e_bprmf_beauty_sports_rankings.csv",
        "results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv",
        "data/reconstructed/plan_c/amazon_beauty/interactions.csv",
        "data/reconstructed/plan_c/amazon_sports/interactions.csv",
    ]
    return "\n".join(f"- `{path}`: `{sha256(PROJECT_ROOT / path)}`." for path in paths if (PROJECT_ROOT / path).is_file())


def validate_written_outputs() -> None:
    checks = [
        (OUTPUTS["all_gru4rec"], ALL_DATASET_COLUMNS, 12),
        (OUTPUTS["same_cohort"], SAME_COHORT_METRIC_COLUMNS, 60),
        (OUTPUTS["comparison"], COMPARISON_COLUMNS, 48),
    ]
    for path, expected_header, expected_rows in checks:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            row_count = sum(1 for _ in reader)
        if header != expected_header:
            raise RuntimeError(f"bad header for {path}")
        if row_count != expected_rows:
            raise RuntimeError(f"bad row count for {path}: {row_count}")
    if not OUTPUTS["report"].is_file():
        raise RuntimeError("report was not written")


def ensure_outputs_absent() -> None:
    existing = [str(path) for path in OUTPUTS.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing output(s): " + "; ".join(existing))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def git_output(command: Sequence[str]) -> str:
    return common_git.git_stdout_checked(command, PROJECT_ROOT, stderr_to_devnull=True)


if __name__ == "__main__":
    raise SystemExit(main())
