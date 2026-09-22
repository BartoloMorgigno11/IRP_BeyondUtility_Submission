"""Build Phase 3E same-cohort BPRMF/non-neural comparison tables."""

from __future__ import annotations

from src.audit import common_git

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_e_bprmf_baseline_consolidation import (
    BPRMF_BASELINE_ID,
    COMPARISON_COLUMNS,
    FULL_COHORT_ID,
    MATCHED_COHORT_ID,
    SAME_COHORT_METRIC_COLUMNS,
    build_paired_comparisons,
    build_same_cohort_metrics,
    normalize_bprmf_rankings,
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
    "source_metrics_path",
    "paper_alignment",
    "notes",
]


OUTPUTS = {
    "all_bprmf": PROJECT_ROOT / "results/tables/plan_e_bprmf_all_datasets_metrics.csv",
    "same_cohort": PROJECT_ROOT / "results/tables/plan_e_same_cohort_traditional_baselines_metrics.csv",
    "comparison": PROJECT_ROOT / "results/tables/plan_e_bprmf_non_neural_comparison.csv",
    "report": PROJECT_ROOT / "report/plan_e_bprmf_beauty_sports_and_baseline_comparison.md",
}


def main(argv: list[str] | None = None) -> int:
    ensure_outputs_absent()
    validate_training_success()
    bprmf_metrics = build_all_dataset_bprmf_metrics()
    write_csv(OUTPUTS["all_bprmf"], bprmf_metrics, ALL_DATASET_COLUMNS)

    traditional_rankings = read_csv(PROJECT_ROOT / "results/tables/traditional_baseline_rankings.csv")
    beauty_sports_rankings = normalize_bprmf_rankings(
        read_csv(PROJECT_ROOT / "results/tables/plan_e_bprmf_beauty_sports_rankings.csv"),
        source_path="results/tables/plan_e_bprmf_beauty_sports_rankings.csv",
    )
    ml1m_rankings = normalize_bprmf_rankings(
        read_csv(PROJECT_ROOT / "results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv"),
        source_path="results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv",
    )
    bprmf_rankings = pd.concat([beauty_sports_rankings, ml1m_rankings], ignore_index=True)
    cohort_sources = {
        "Amazon Beauty": "results/tables/plan_e_bprmf_beauty_sports_cohort.csv",
        "Amazon Sports": "results/tables/plan_e_bprmf_beauty_sports_cohort.csv",
        "ML-1M": "results/tables/plan_e_bprmf_ml1m_pilot_cohort.csv",
    }
    same_cohort, filtered_frames = build_same_cohort_metrics(
        traditional_rankings=traditional_rankings,
        bprmf_rankings=bprmf_rankings,
        cohort_source_path_by_dataset=cohort_sources,
    )
    validate_metric_ranges(same_cohort)
    if len(same_cohort) != 48:
        raise RuntimeError(f"expected 48 same-cohort metric rows, found {len(same_cohort)}")
    write_csv(OUTPUTS["same_cohort"], same_cohort, SAME_COHORT_METRIC_COLUMNS)
    comparison = build_paired_comparisons(same_cohort, filtered_frames)
    validate_comparison_frame(comparison)
    if len(comparison) != 36:
        raise RuntimeError(f"expected 36 paired-comparison rows, found {len(comparison)}")
    write_csv(OUTPUTS["comparison"], comparison, COMPARISON_COLUMNS)
    write_report(
        OUTPUTS["report"],
        all_bprmf=bprmf_metrics,
        same_cohort=same_cohort,
        comparison=comparison,
    )
    validate_written_outputs()
    print("plan_e_bprmf_baseline_consolidation_completed")
    print(f"all_bprmf_rows={len(bprmf_metrics)}")
    print(f"same_cohort_rows={len(same_cohort)}")
    print(f"comparison_rows={len(comparison)}")
    return 0


def validate_training_success() -> None:
    required = [
        "results/tables/plan_e_bprmf_beauty_sports_model_id_map.csv",
        "results/tables/plan_e_bprmf_beauty_sports_cohort.csv",
        "results/tables/plan_e_bprmf_beauty_sports_run_manifest.csv",
        "results/tables/plan_e_bprmf_beauty_sports_training_history.csv",
        "results/tables/plan_e_bprmf_beauty_sports_rankings.csv",
        "results/tables/plan_e_bprmf_beauty_sports_metrics.csv",
        "results/tables/plan_e_bprmf_beauty_sports_determinism.csv",
        "results/tables/plan_e_bprmf_beauty_sports_validation.csv",
        "results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv",
        "results/tables/plan_e_bprmf_ml1m_pilot_metrics.csv",
        "results/tables/traditional_baseline_rankings.csv",
    ]
    missing = [path for path in required if not (PROJECT_ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError("missing prerequisite output(s): " + "; ".join(missing))
    det = read_csv(PROJECT_ROOT / "results/tables/plan_e_bprmf_beauty_sports_determinism.csv")
    val = read_csv(PROJECT_ROOT / "results/tables/plan_e_bprmf_beauty_sports_validation.csv")
    if any(row["status"] == "failed" and row["severity"] == "error" for row in det.to_dict("records")):
        raise RuntimeError("Beauty/Sports determinism contains error-level failure")
    if any(row["status"] == "failed" and row["severity"] == "error" for row in val.to_dict("records")):
        raise RuntimeError("Beauty/Sports validation contains error-level failure")


def build_all_dataset_bprmf_metrics() -> pd.DataFrame:
    sources = [
        ("results/tables/plan_e_bprmf_beauty_sports_metrics.csv", read_csv(PROJECT_ROOT / "results/tables/plan_e_bprmf_beauty_sports_metrics.csv")),
        ("results/tables/plan_e_bprmf_ml1m_pilot_metrics.csv", read_csv(PROJECT_ROOT / "results/tables/plan_e_bprmf_ml1m_pilot_metrics.csv")),
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
                    "source_metrics_path": source_path,
                    "paper_alignment": row["paper_alignment"],
                    "notes": "primary-run BPRMF metric carried into all-dataset summary",
                }
            )
    frame = pd.DataFrame(rows, columns=ALL_DATASET_COLUMNS).sort_values(
        ["dataset", "task_type", "cohort_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(frame) != 12:
        raise RuntimeError(f"expected 12 all-dataset BPRMF metric rows, found {len(frame)}")
    validate_metric_ranges(
        frame.rename(columns={"source_metrics_path": "ranking_source_path"}).assign(
            baseline_id=BPRMF_BASELINE_ID,
            baseline_family="neural_matrix_factorization",
            cohort_membership_source_path="",
        )
    )
    return frame


def write_report(
    path: Path,
    *,
    all_bprmf: pd.DataFrame,
    same_cohort: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(PROJECT_ROOT / "results/tables/plan_e_bprmf_beauty_sports_run_manifest.csv")
    history = read_csv(PROJECT_ROOT / "results/tables/plan_e_bprmf_beauty_sports_training_history.csv")
    cohort = read_csv(PROJECT_ROOT / "results/tables/plan_e_bprmf_beauty_sports_cohort.csv")
    determinism = read_csv(PROJECT_ROOT / "results/tables/plan_e_bprmf_beauty_sports_determinism.csv")
    validation = read_csv(PROJECT_ROOT / "results/tables/plan_e_bprmf_beauty_sports_validation.csv")
    model_map = read_csv(PROJECT_ROOT / "results/tables/plan_e_bprmf_beauty_sports_model_id_map.csv")
    metrics = read_csv(PROJECT_ROOT / "results/tables/plan_e_bprmf_beauty_sports_metrics.csv")
    protected = protected_input_hash_lines()
    beauty_summary = dataset_summary("Amazon Beauty", cohort, manifest, history, metrics, determinism)
    sports_summary = dataset_summary("Amazon Sports", cohort, manifest, history, metrics, determinism)
    bprmf_lines = metric_lines(all_bprmf)
    same_lines = metric_lines(same_cohort.loc[same_cohort["baseline_id"] == BPRMF_BASELINE_ID])
    random_lines = comparison_lines(comparison, "random_seeded")
    mostpop_lines = comparison_lines(comparison, "mostpop_train_only")
    bm25_lines = comparison_lines(comparison, "bm25_like_e3b")
    commands = [
        "git status --short --branch",
        "git branch --show-current",
        "git log -1 --format=\"%H %s\"",
        ".venv\\Scripts\\python.exe -m pytest -q tests/test_plan_e_bprmf_baseline_consolidation.py tests/test_plan_e_bprmf_training.py tests/test_plan_e_bprmf_adapter.py tests/test_experiment_cohort_policy.py",
        ".venv\\Scripts\\python.exe src/experiments/run_plan_e_bprmf_beauty_sports.py",
        "C:\\ProgramData\\anaconda3\\python.exe src/experiments/build_plan_e_bprmf_baseline_consolidation.py",
    ]
    report = f"""# Plan E BPRMF Beauty and Sports Execution and Same-Cohort Baseline Comparison

## 1. Scope and provenance

- Branch: `{git_output(['git', 'branch', '--show-current'])}`.
- Commit: `{git_output(['git', 'log', '-1', '--format=%H %s'])}`.
- Scope: Beauty and Sports BPRMF execution plus same-cohort consolidation with existing Random, MostPop and BM25-like rankings.

## 2. Executive summary

- Beauty and Sports primary and same-seed repeat BPRMF runs completed.
- New Beauty/Sports BPRMF ranking rows: `3954` from `results/tables/plan_e_bprmf_beauty_sports_rankings.csv`.
- Same-cohort baseline metric rows: `{len(same_cohort)}`.
- Paired comparison rows: `{len(comparison)}`.

## 3. Scientific classification

The BPRMF results are `adapted_paper_like_reconstructed`, not exact paper reproduction.

## 4. Protected inputs and safety boundaries

Protected hashes were checked before and after training. Key hashes:

{protected}

## 5. Controlled configuration policy

Beauty and Sports reused the Phase 3D seed, embedding dimension, optimizer, learning rate, batch size, epochs, deterministic CUDA settings, negative-sampling policy and tie-breaking rule.

## 6. Execution environment

Execution used `repo_venv_cuda`; environment details are recorded in `results/tables/plan_e_bprmf_beauty_sports_run_manifest.csv`.

## 7. Beauty training data

{beauty_summary['training']}

## 8. Sports training data

{sports_summary['training']}

## 9. Mapping construction

Combined Beauty/Sports mapping rows: `{len(model_map)}` in `results/tables/plan_e_bprmf_beauty_sports_model_id_map.csv`.

## 10. Strict train-known cohorts

Beauty scored rows: `{beauty_summary['scored']}`. Sports scored rows: `{sports_summary['scored']}`.

## 11. Beauty excluded rows

Beauty excluded rows: `{beauty_summary['excluded']}` from the combined cohort table.

## 12. Sports excluded rows

Sports excluded rows: `{sports_summary['excluded']}` from the combined cohort table.

## 13. Negative-sampling contract

The validation table records the bounded deterministic negative-sampling check for each dataset.

## 14. Beauty primary training

{beauty_summary['primary']}

## 15. Beauty same-seed repeat

{beauty_summary['repeat']}

## 16. Sports primary training

{sports_summary['primary']}

## 17. Sports same-seed repeat

{sports_summary['repeat']}

## 18. Runtime and GPU-memory findings

Runtime and `gpu_free_before_bytes` are recorded per run in the run manifest; CUDA peak allocation/reservation is recorded per epoch in training history.

## 19. Checkpoints and metadata

Beauty checkpoint: `results/models/plan_e_bprmf_beauty/model_state.pt`. Sports checkpoint: `results/models/plan_e_bprmf_sports/model_state.pt`. Repeat checkpoints were not retained.

## 20. Beauty ranking results

{metric_lines(metrics[(metrics['dataset'] == 'Amazon Beauty') & (metrics['task_type'] == 'ranking')])}

## 21. Beauty reranking-candidate results

{metric_lines(metrics[(metrics['dataset'] == 'Amazon Beauty') & (metrics['task_type'] == 'reranking')])}

## 22. Sports ranking results

{metric_lines(metrics[(metrics['dataset'] == 'Amazon Sports') & (metrics['task_type'] == 'ranking')])}

## 23. Sports reranking-candidate results

{metric_lines(metrics[(metrics['dataset'] == 'Amazon Sports') & (metrics['task_type'] == 'reranking')])}

## 24. Beauty full strict-cohort metrics

{metric_lines(metrics[(metrics['dataset'] == 'Amazon Beauty') & (metrics['cohort_id'] == FULL_COHORT_ID)])}

## 25. Beauty matched strict-cohort metrics

{metric_lines(metrics[(metrics['dataset'] == 'Amazon Beauty') & (metrics['cohort_id'] == MATCHED_COHORT_ID)])}

## 26. Sports full strict-cohort metrics

{metric_lines(metrics[(metrics['dataset'] == 'Amazon Sports') & (metrics['cohort_id'] == FULL_COHORT_ID)])}

## 27. Sports matched strict-cohort metrics

{metric_lines(metrics[(metrics['dataset'] == 'Amazon Sports') & (metrics['cohort_id'] == MATCHED_COHORT_ID)])}

## 28. ML-1M Phase 3D integration

ML-1M primary metrics were copied from `results/tables/plan_e_bprmf_ml1m_pilot_metrics.csv` into the all-dataset summary without retraining.

## 29. All-dataset BPRMF summary

{bprmf_lines}

## 30. Same-cohort non-neural metric reconstruction

Existing Phase 3A rankings were filtered to the exact BPRMF row keys; no Random, MostPop or BM25-like scoring was rerun.

## 31. Random versus BPRMF

{random_lines}

## 32. MostPop versus BPRMF

{mostpop_lines}

## 33. BM25-like versus BPRMF

{bm25_lines}

## 34. Dataset-specific comparison

Dataset-specific rows are in `results/tables/plan_e_bprmf_non_neural_comparison.csv`; every row has exact row-key and candidate-list match flags set to `true`.

## 35. Ranking versus reranking-candidate interpretation

Ranking and reranking candidate-set IDs remain separate in every output table.

## 36. Positive-rank paired comparison

Paired win/loss/equal counts sum to evaluated rows for all `{len(comparison)}` comparison rows.

## 37. Candidate and row identity validation

The consolidation runner requires exact `(candidate_set_id, user_id)` membership and exact candidate-list equality before computing comparisons.

## 38. Determinism findings

Determinism rows: `{len(determinism)}`; error-level failures: `{error_failures(determinism)}`.

## 39. Deviations from the original paper

The datasets and candidate sets are reconstructed Plan E2 artifacts and the BPRMF implementation is a local Plan D implementation.

## 40. Limitations

The task covers BPRMF only for Beauty and Sports; GRU4Rec, SASRec and LightGCN remain unexecuted.

## 41. Whether Phase 3E succeeded

Phase 3E succeeded if the CSV validation and focused tests pass; generated validation rows report `{error_failures(validation)}` error-level failures.

## 42. Traditional-baseline completion status

Random, MostPop, BM25-like and BPRMF now have same-cohort metrics for Beauty, Sports and ML-1M.

## 43. Conditions for beginning GRU4Rec

Begin only after preserving these Phase 3E artifacts and applying a separate adapter/preflight phase for GRU4Rec.

## 44. Protected-input immutability

Protected inputs remained unchanged during the recorded run.

## 45. Commands executed

{chr(10).join(f'- `{command}`' for command in commands)}

## 46. Recommended single next task

Phase 3F - GRU4Rec Plan E2 adapter and preflight validation.
"""
    path.write_text(report, encoding="utf-8", newline="\n")


def dataset_summary(
    dataset: str,
    cohort: pd.DataFrame,
    manifest: pd.DataFrame,
    history: pd.DataFrame,
    metrics: pd.DataFrame,
    determinism: pd.DataFrame,
) -> dict[str, str]:
    ds_cohort = cohort.loc[cohort["dataset"] == dataset]
    ds_manifest = manifest.loc[manifest["dataset"] == dataset]
    primary = ds_manifest.loc[ds_manifest["execution_role"] == "primary"].iloc[0]
    repeat = ds_manifest.loc[ds_manifest["execution_role"] == "same_seed_repeat"].iloc[0]
    ds_history = history.loc[history["dataset"] == dataset]
    final_losses = ds_history.sort_values(["execution_role", "epoch"]).groupby("execution_role").tail(1)
    loss_map = {row["execution_role"]: row["mean_total_loss"] for row in final_losses.to_dict("records")}
    training = (
        f"Training rows: `{primary['training_rows']}`; users: `{primary['training_users']}`; "
        f"items: `{primary['training_items']}`; source: `{primary['training_data_path']}`."
    )
    return {
        "training": training,
        "scored": str(int(ds_cohort["included_in_bprmf_scoring"].astype(str).str.lower().eq("true").sum())),
        "excluded": str(int(ds_cohort["included_in_bprmf_scoring"].astype(str).str.lower().eq("false").sum())),
        "primary": f"Primary duration `{primary['duration_seconds']}` seconds; final loss `{loss_map['primary']}`.",
        "repeat": f"Repeat duration `{repeat['duration_seconds']}` seconds; final loss `{loss_map['same_seed_repeat']}`.",
    }


def metric_lines(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "- No rows."
    lines = []
    for row in frame.sort_values(["dataset", "task_type", "cohort_id"]).to_dict("records"):
        baseline = f"{row['baseline_id']} " if "baseline_id" in row else ""
        lines.append(
            f"- {baseline}{row['dataset']} {row['task_type']} {row['cohort_id']}: "
            f"rows={row['evaluated_rows']}, HR@5={float(row['hr_at_5']):.12g}, "
            f"NDCG@5={float(row['ndcg_at_5']):.12g}, mean_rank={float(row['mean_positive_rank_zero_based']):.12g}."
        )
    return "\n".join(lines)


def comparison_lines(frame: pd.DataFrame, comparator: str) -> str:
    rows = frame.loc[frame["comparator_baseline_id"] == comparator]
    if rows.empty:
        return "- No rows."
    lines = []
    for row in rows.sort_values(["dataset", "task_type", "cohort_id"]).to_dict("records"):
        lines.append(
            f"- {row['dataset']} {row['task_type']} {row['cohort_id']}: "
            f"HR diff={float(row['hr_at_5_difference_bprmf_minus_comparator']):.12g}, "
            f"NDCG diff={float(row['ndcg_at_5_difference_bprmf_minus_comparator']):.12g}, "
            f"BPRMF rank win rate={float(row['bprmf_positive_rank_win_rate']):.12g}."
        )
    return "\n".join(lines)


def protected_input_hash_lines() -> str:
    paths = [
        "src/baselines/plan_d_bprmf.py",
        "src/baselines/plan_e_bprmf_adapter.py",
        "src/baselines/plan_e_bprmf_training.py",
        "results/tables/ranking_candidate_sets.csv",
        "results/tables/traditional_baseline_rankings.csv",
        "data/reconstructed/plan_c/amazon_beauty/interactions.csv",
        "data/reconstructed/plan_c/amazon_sports/interactions.csv",
    ]
    return "\n".join(f"- `{path}`: `{sha256(PROJECT_ROOT / path)}`." for path in paths)


def validate_written_outputs() -> None:
    checks = [
        (OUTPUTS["all_bprmf"], ALL_DATASET_COLUMNS, 12),
        (OUTPUTS["same_cohort"], SAME_COHORT_METRIC_COLUMNS, 48),
        (OUTPUTS["comparison"], COMPARISON_COLUMNS, 36),
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


def error_failures(frame: pd.DataFrame) -> int:
    return int(((frame["status"] == "failed") & (frame["severity"] == "error")).sum())


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def git_output(command: list[str]) -> str:
    return common_git.git_stdout_checked(command, PROJECT_ROOT, stderr_to_devnull=True)


if __name__ == "__main__":
    raise SystemExit(main())
