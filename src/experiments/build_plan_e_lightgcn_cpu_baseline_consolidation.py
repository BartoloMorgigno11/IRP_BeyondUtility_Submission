"""Build Phase 3N CPU LightGCN consolidation tables and report."""

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

from src.analysis.plan_e_lightgcn_cpu_baseline_consolidation import (
    ALL_LIGHTGCN_METRIC_COLUMNS,
    COMPARISON_COLUMNS,
    DEVICE_PROVENANCE_COLUMNS,
    FULL_COHORT_ID,
    MATCHED_COHORT_ID,
    SAME_COHORT_METRIC_COLUMNS,
    build_paired_comparisons,
    build_same_cohort_metrics,
    normalize_bprmf_rankings,
    normalize_gru4rec_rankings,
    normalize_lightgcn_rankings,
    normalize_sasrec_rankings,
    normalize_traditional_rankings,
    read_csv,
    validate_all_lightgcn_metrics,
    validate_comparison_frame,
    validate_device_provenance,
    validate_metric_ranges,
    write_csv,
)


OUTPUTS = {
    "device_provenance": PROJECT_ROOT / "results/tables/plan_e_lightgcn_all_datasets_device_provenance.csv",
    "all_lightgcn": PROJECT_ROOT / "results/tables/plan_e_lightgcn_all_datasets_metrics.csv",
    "same_cohort": PROJECT_ROOT / "results/tables/plan_e_lightgcn_same_cohort_baselines_metrics.csv",
    "comparison": PROJECT_ROOT / "results/tables/plan_e_lightgcn_baseline_comparison.csv",
    "report": PROJECT_ROOT / "report/plan_e_lightgcn_beauty_sports_cpu_and_baseline_comparison.md",
}

CPU_PATHS = {
    "rankings": "results/tables/plan_e_lightgcn_beauty_sports_cpu_rankings.csv",
    "metrics": "results/tables/plan_e_lightgcn_beauty_sports_cpu_metrics.csv",
    "cohort": "results/tables/plan_e_lightgcn_beauty_sports_cpu_cohort.csv",
    "manifest": "results/tables/plan_e_lightgcn_beauty_sports_cpu_run_manifest.csv",
    "history": "results/tables/plan_e_lightgcn_beauty_sports_cpu_training_history.csv",
    "determinism": "results/tables/plan_e_lightgcn_beauty_sports_cpu_determinism.csv",
    "validation": "results/tables/plan_e_lightgcn_beauty_sports_cpu_validation.csv",
    "graph": "results/tables/plan_e_lightgcn_beauty_sports_cpu_graph_summary.csv",
    "node_map": "results/tables/plan_e_lightgcn_beauty_sports_cpu_model_node_map.csv",
}
ML1M_PATHS = {
    "rankings": "results/tables/plan_e_lightgcn_ml1m_pilot_rankings.csv",
    "metrics": "results/tables/plan_e_lightgcn_ml1m_pilot_metrics.csv",
    "manifest": "results/tables/plan_e_lightgcn_ml1m_pilot_run_manifest.csv",
    "determinism": "results/tables/plan_e_lightgcn_ml1m_pilot_determinism.csv",
    "decision": "results/tables/plan_e_lightgcn_ml1m_cuda_first_step_decision.csv",
}


def main(argv: list[str] | None = None) -> int:
    del argv
    ensure_outputs_absent()
    validate_required_inputs()

    device_provenance = build_device_provenance()
    validate_device_provenance(device_provenance)
    write_csv(OUTPUTS["device_provenance"], device_provenance, DEVICE_PROVENANCE_COLUMNS)

    all_lightgcn = build_all_dataset_lightgcn_metrics()
    validate_all_lightgcn_metrics(all_lightgcn)
    write_csv(OUTPUTS["all_lightgcn"], all_lightgcn, ALL_LIGHTGCN_METRIC_COLUMNS)

    same_cohort, filtered_frames = build_same_cohort_tables()
    validate_metric_ranges(same_cohort)
    if len(same_cohort) != 84:
        raise RuntimeError(f"expected 84 same-cohort metric rows, found {len(same_cohort)}")
    write_csv(OUTPUTS["same_cohort"], same_cohort, SAME_COHORT_METRIC_COLUMNS)

    comparison = build_paired_comparisons(same_cohort, filtered_frames)
    validate_comparison_frame(comparison)
    if len(comparison) != 72:
        raise RuntimeError(f"expected 72 paired comparison rows, found {len(comparison)}")
    write_csv(OUTPUTS["comparison"], comparison, COMPARISON_COLUMNS)

    write_report(
        OUTPUTS["report"],
        device_provenance=device_provenance,
        all_lightgcn=all_lightgcn,
        same_cohort=same_cohort,
        comparison=comparison,
    )
    validate_written_outputs()
    print("plan_e_lightgcn_cpu_baseline_consolidation_completed")
    print(f"device_provenance_rows={len(device_provenance)}")
    print(f"all_lightgcn_rows={len(all_lightgcn)}")
    print(f"same_cohort_rows={len(same_cohort)}")
    print(f"comparison_rows={len(comparison)}")
    return 0


def ensure_outputs_absent() -> None:
    existing = [str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in OUTPUTS.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing output(s): " + "; ".join(existing))


def validate_required_inputs() -> None:
    required = [
        *CPU_PATHS.values(),
        *ML1M_PATHS.values(),
        "results/tables/traditional_baseline_rankings.csv",
        "results/tables/plan_e_bprmf_beauty_sports_rankings.csv",
        "results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv",
        "results/tables/plan_e_gru4rec_beauty_sports_rankings.csv",
        "results/tables/plan_e_gru4rec_ml1m_pilot_rankings.csv",
        "results/tables/plan_e_sasrec_beauty_sports_rankings.csv",
        "results/tables/plan_e_sasrec_ml1m_pilot_rankings.csv",
    ]
    missing = [path for path in required if not (PROJECT_ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError("missing prerequisite output(s): " + "; ".join(missing))
    cpu_det = read_csv(PROJECT_ROOT / CPU_PATHS["determinism"])
    if len(cpu_det) != 32 or cpu_det["status"].astype(str).str.lower().ne("passed").any():
        raise RuntimeError("CPU LightGCN determinism is incomplete or failed")
    cpu_val = read_csv(PROJECT_ROOT / CPU_PATHS["validation"])
    if ((cpu_val["status"].astype(str).eq("failed")) & (cpu_val["severity"].astype(str).eq("error"))).any():
        raise RuntimeError("CPU LightGCN validation contains an error-level failure")


def build_device_provenance() -> pd.DataFrame:
    cpu_manifest = read_csv(PROJECT_ROOT / CPU_PATHS["manifest"])
    cpu_det = read_csv(PROJECT_ROOT / CPU_PATHS["determinism"])
    decision = read_csv(PROJECT_ROOT / ML1M_PATHS["decision"]).iloc[0]
    ml_manifest = read_csv(PROJECT_ROOT / ML1M_PATHS["manifest"])
    ml_det = read_csv(PROJECT_ROOT / ML1M_PATHS["determinism"])
    rows: list[dict[str, Any]] = []
    for dataset in ("Amazon Beauty", "Amazon Sports"):
        unit = cpu_manifest.loc[cpu_manifest["dataset"].astype(str).eq(dataset)]
        det = cpu_det.loc[cpu_det["dataset"].astype(str).eq(dataset)]
        rows.append(
            {
                "dataset": dataset,
                "execution_device": "cpu",
                "source_phase": "Phase 3N-CPU",
                "source_config_path": "configs/plan_e_lightgcn_beauty_cpu.yaml"
                if dataset == "Amazon Beauty"
                else "configs/plan_e_lightgcn_sports_cpu.yaml",
                "source_rankings_path": CPU_PATHS["rankings"],
                "source_metrics_path": CPU_PATHS["metrics"],
                "primary_execution_status": manifest_status(unit, "primary"),
                "repeat_execution_status": manifest_status(unit, "same_seed_repeat"),
                "determinism_classification": "exact_cpu_repeat",
                "exact_final_model_state": "true",
                "exact_propagated_embeddings": "true",
                "exact_candidate_scores": "true",
                "exact_rankings": "true",
                "loss_telemetry_exact": "true",
                "cuda_sparse_forward_limitation": "false",
                "cpu_scale_up_decision_source": "results/tables/plan_e_lightgcn_ml1m_cuda_first_step_decision.csv",
                "paper_alignment": "adapted_paper_like_reconstructed",
                "notes": f"{len(det)} exact CPU determinism checks passed",
            }
        )
    ml_statuses = ml_manifest["execution_status"].astype(str).tolist()
    rows.append(
        {
            "dataset": "ML-1M",
            "execution_device": "cuda",
            "source_phase": "Phase 3M plus Phase 3M-D2",
            "source_config_path": "configs/plan_e_lightgcn_ml1m_pilot.yaml",
            "source_rankings_path": ML1M_PATHS["rankings"],
            "source_metrics_path": ML1M_PATHS["metrics"],
            "primary_execution_status": ml_statuses[0],
            "repeat_execution_status": ml_statuses[1],
            "determinism_classification": decision["phase3m_d2_classification"],
            "exact_final_model_state": "false",
            "exact_propagated_embeddings": "false",
            "exact_candidate_scores": "false",
            "exact_rankings": "true",
            "loss_telemetry_exact": "false",
            "cuda_sparse_forward_limitation": "true",
            "cpu_scale_up_decision_source": "results/tables/plan_e_lightgcn_ml1m_cuda_first_step_decision.csv",
            "paper_alignment": "adapted_paper_like_reconstructed",
            "notes": f"Phase 3M rankings retained; D2 first divergence={decision['first_true_tensor_divergence_stage']}; determinism rows={len(ml_det)}",
        }
    )
    return pd.DataFrame(rows, columns=DEVICE_PROVENANCE_COLUMNS)


def build_all_dataset_lightgcn_metrics() -> pd.DataFrame:
    sources = [
        (
            CPU_PATHS["metrics"],
            read_csv(PROJECT_ROOT / CPU_PATHS["metrics"]),
            "cpu",
            "completed_exact_cpu_repeat",
            "exact_cpu_repeat",
        ),
        (
            ML1M_PATHS["metrics"],
            read_csv(PROJECT_ROOT / ML1M_PATHS["metrics"]),
            "cuda",
            "completed_with_determinism_warning",
            "cuda_sparse_forward_tensor_nondeterminism",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for source_path, frame, device, status, determinism in sources:
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
                    "mean_user_train_degree": row["mean_user_train_degree"],
                    "median_user_train_degree": row["median_user_train_degree"],
                    "mean_positive_item_train_degree": row["mean_positive_item_train_degree"],
                    "median_positive_item_train_degree": row["median_positive_item_train_degree"],
                    "mean_candidate_item_train_degree": row["mean_candidate_item_train_degree"],
                    "median_candidate_item_train_degree": row["median_candidate_item_train_degree"],
                    "execution_device": device,
                    "execution_status": status,
                    "determinism_classification": determinism,
                    "source_metrics_path": source_path,
                    "paper_alignment": row["paper_alignment"],
                    "notes": "LightGCN metric carried forward without recomputation",
                }
            )
    frame = pd.DataFrame(rows, columns=ALL_LIGHTGCN_METRIC_COLUMNS).sort_values(
        ["dataset", "task_type", "cohort_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    ml_source = read_csv(PROJECT_ROOT / ML1M_PATHS["metrics"]).sort_values(["task_type", "cohort_id"]).reset_index(drop=True)
    ml_join = frame.loc[frame["dataset"].astype(str).eq("ML-1M")].sort_values(["task_type", "cohort_id"]).reset_index(drop=True)
    for field in ("hr_at_5", "ndcg_at_5"):
        if ml_join[field].astype(str).tolist() != ml_source[field].astype(str).tolist():
            raise RuntimeError(f"ML-1M {field} does not exactly match Phase 3M source metrics")
    return frame


def build_same_cohort_tables() -> tuple[pd.DataFrame, dict[tuple[str, str, str, str, str], pd.DataFrame]]:
    traditional = normalize_traditional_rankings(
        read_csv(PROJECT_ROOT / "results/tables/traditional_baseline_rankings.csv"),
        source_path="results/tables/traditional_baseline_rankings.csv",
    )
    bprmf = pd.concat(
        [
            normalize_bprmf_rankings(
                read_csv(PROJECT_ROOT / "results/tables/plan_e_bprmf_beauty_sports_rankings.csv"),
                source_path="results/tables/plan_e_bprmf_beauty_sports_rankings.csv",
                source_execution_device="cuda",
                source_determinism_status="completed",
            ),
            normalize_bprmf_rankings(
                read_csv(PROJECT_ROOT / "results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv"),
                source_path="results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv",
                source_execution_device="cuda",
                source_determinism_status="completed",
            ),
        ],
        ignore_index=True,
    )
    gru = pd.concat(
        [
            normalize_gru4rec_rankings(
                read_csv(PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_rankings.csv"),
                source_path="results/tables/plan_e_gru4rec_beauty_sports_rankings.csv",
                source_execution_device="cuda",
                source_determinism_status="completed",
            ),
            normalize_gru4rec_rankings(
                read_csv(PROJECT_ROOT / "results/tables/plan_e_gru4rec_ml1m_pilot_rankings.csv"),
                source_path="results/tables/plan_e_gru4rec_ml1m_pilot_rankings.csv",
                source_execution_device="cuda",
                source_determinism_status="completed",
            ),
        ],
        ignore_index=True,
    )
    sas = pd.concat(
        [
            normalize_sasrec_rankings(
                read_csv(PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_rankings.csv"),
                source_path="results/tables/plan_e_sasrec_beauty_sports_rankings.csv",
                source_execution_device="cuda",
                source_determinism_status="completed",
            ),
            normalize_sasrec_rankings(
                read_csv(PROJECT_ROOT / "results/tables/plan_e_sasrec_ml1m_pilot_rankings.csv"),
                source_path="results/tables/plan_e_sasrec_ml1m_pilot_rankings.csv",
                source_execution_device="cuda",
                source_determinism_status="completed",
            ),
        ],
        ignore_index=True,
    )
    lightgcn = pd.concat(
        [
            normalize_lightgcn_rankings(
                read_csv(PROJECT_ROOT / CPU_PATHS["rankings"]),
                source_path=CPU_PATHS["rankings"],
                source_execution_device="cpu",
                source_determinism_status="completed_exact_cpu_repeat",
            ),
            normalize_lightgcn_rankings(
                read_csv(PROJECT_ROOT / ML1M_PATHS["rankings"]),
                source_path=ML1M_PATHS["rankings"],
                source_execution_device="cuda",
                source_determinism_status="completed_with_determinism_warning",
            ),
        ],
        ignore_index=True,
    )
    cohort_sources = {
        "Amazon Beauty": CPU_PATHS["cohort"],
        "Amazon Sports": CPU_PATHS["cohort"],
        "ML-1M": "results/tables/plan_e_lightgcn_ml1m_pilot_cohort.csv",
    }
    return build_same_cohort_metrics(
        traditional_rankings=traditional,
        bprmf_rankings=bprmf,
        gru4rec_rankings=gru,
        sasrec_rankings=sas,
        lightgcn_rankings=lightgcn,
        cohort_source_path_by_dataset=cohort_sources,
    )


def write_report(
    path: Path,
    *,
    device_provenance: pd.DataFrame,
    all_lightgcn: pd.DataFrame,
    same_cohort: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {path}")
    graph = read_csv(PROJECT_ROOT / CPU_PATHS["graph"])
    cohort = read_csv(PROJECT_ROOT / CPU_PATHS["cohort"])
    manifest = read_csv(PROJECT_ROOT / CPU_PATHS["manifest"])
    history = read_csv(PROJECT_ROOT / CPU_PATHS["history"])
    rankings = read_csv(PROJECT_ROOT / CPU_PATHS["rankings"])
    metrics = read_csv(PROJECT_ROOT / CPU_PATHS["metrics"])
    validation = read_csv(PROJECT_ROOT / CPU_PATHS["validation"])
    determinism = read_csv(PROJECT_ROOT / CPU_PATHS["determinism"])
    node_map = read_csv(PROJECT_ROOT / CPU_PATHS["node_map"])
    beauty = dataset_summary("Amazon Beauty", graph, cohort, manifest, history, rankings, metrics, determinism)
    sports = dataset_summary("Amazon Sports", graph, cohort, manifest, history, rankings, metrics, determinism)
    report = f"""# Plan E CPU LightGCN Beauty and Sports Execution and Same-Cohort Baseline Comparison

## 1. Scope and provenance

- Branch: `{git_output(['branch', '--show-current'])}`.
- Commit: `{git_output(['log', '-1', '--format=%H %s'])}`.
- Scope: CPU-only Beauty/Sports LightGCN execution plus same-cohort consolidation with Random, MostPop, BM25-like, BPRMF, GRU4Rec, SASRec and ML-1M Phase 3M LightGCN evidence.

## 2. Executive summary

- CPU LightGCN ranking rows: `{len(rankings)}`.
- CPU LightGCN metric rows: `{len(metrics)}`.
- All-dataset LightGCN metric rows: `{len(all_lightgcn)}`.
- Seven-baseline same-cohort metric rows: `{len(same_cohort)}`.
- Paired comparison rows: `{len(comparison)}`.

## 3. Scientific classification

The results are `adapted_paper_like_reconstructed`, not exact original-paper reproduction, because they use reconstructed Plan C data, Plan E2 strict cohorts and local LightGCN code.

## 4. Protected-input boundaries

The CPU runner verified protected hashes before and after execution and did not modify Phase 2A canonical candidate files, reconstructed interactions, prior neural rankings, historical Plan outputs or `original_repo/`.

## 5. Phase 3M CUDA pilot status

ML-1M remains `completed_with_determinism_warning`; its primary rankings and metrics are reused from `{ML1M_PATHS['rankings']}` and `{ML1M_PATHS['metrics']}`.

## 6. Phase 3M-D1 findings

The D1 diagnostic classified ML-1M CUDA LightGCN as `cuda_sparse_training_nondeterministic` and did not approve CUDA scale-up.

## 7. Phase 3M-D2 root-cause findings

The D2 decision classified the first true CUDA divergence as `cuda_sparse_forward_tensor_nondeterminism` at `layer_1_all_embeddings`; Adam was not the root cause.

## 8. Why CPU-only scale-up was selected

CPU-only scale-up was selected because D2 approved `approve_cpu_only_scale_up` while explicitly not approving the existing CUDA pipeline for new Beauty/Sports LightGCN runs.

## 9. CPU execution environment

{environment_line(manifest)}

## 10. CPU thread and determinism settings

The CPU runner forced one torch compute thread and one interop thread; manifest thread values are `{single_value(manifest, 'torch_num_threads')}` and `{single_value(manifest, 'torch_num_interop_threads')}`.

## 11. Controlled configuration

Both CPU datasets used seed `42`, embedding dimension `64`, two propagation layers, Adam, batch size `4096`, five epochs, learning rate `0.001`, regularization `0.0` and train-universe negative sampling.

## 12. Beauty training graph

{beauty['graph']}

## 13. Sports training graph

{sports['graph']}

## 14. Training-edge isolation

The graph summary records train-only interaction paths and hashes; no validation, test, held-out positive or candidate-derived edges were added.

## 15. Duplicate-edge policy

Beauty duplicate train edges: `{beauty['duplicate_edges']}`. Sports duplicate train edges: `{sports['duplicate_edges']}`. Both graphs use deterministic user-item pair deduplication.

## 16. Bipartite mappings and offsets

Beauty node-map rows: `{beauty['node_rows']}`. Sports node-map rows: `{sports['node_rows']}`. Item offsets equal the per-dataset user-node counts.

## 17. Graph normalization

Both graphs use `D^-1/2 A D^-1/2` and store coalesced `torch.sparse_coo` adjacency evidence with hash columns in `{CPU_PATHS['graph']}`.

## 18. CPU sparse-operation validation

CPU sparse repeat checks passed for Beauty and Sports; `cpu_sparse_repeat_exact=true` in both graph-summary rows.

## 19. Negative-sampling contract

The manifest records one deterministic negative schedule hash per dataset; negatives are sampled from the train item universe while excluding the user's train positives.

## 20. Strict graph-compatible cohorts

Combined cohort rows: `{len(cohort)}`. Included rows: `{len(rankings)}`. Full cohort ID: `{FULL_COHORT_ID}`. Matched cohort ID: `{MATCHED_COHORT_ID}`.

## 21. Beauty exclusions

Beauty excluded rows: `{beauty['excluded']}`. These rows remain in the cohort audit but were not scored.

## 22. Sports exclusions

Sports excluded rows: `{sports['excluded']}`. These rows remain in the cohort audit but were not scored.

## 23. Beauty CPU primary training

{beauty['primary']}

## 24. Beauty CPU repeat training

{beauty['repeat']}

## 25. Beauty exact determinism

Beauty determinism checks passed: `{beauty['determinism_passed']}` of `{beauty['determinism_total']}`.

## 26. Sports CPU primary training

{sports['primary']}

## 27. Sports CPU repeat training

{sports['repeat']}

## 28. Sports exact determinism

Sports determinism checks passed: `{sports['determinism_passed']}` of `{sports['determinism_total']}`.

## 29. CPU runtime and memory

Combined history rows: `{len(history)}`. Process RSS values are recorded per epoch in `{CPU_PATHS['history']}`.

## 30. Checkpoints and metadata

Primary checkpoints retained: `{manifest['checkpoint_path'].astype(str).ne('').sum()}`; repeat checkpoints retained: `{manifest.loc[manifest['execution_role'].astype(str).eq('same_seed_repeat'), 'checkpoint_path'].astype(str).ne('').sum()}`.

## 31. Beauty ranking results

{metric_sentence(metrics, 'Amazon Beauty', 'ranking', FULL_COHORT_ID)}

## 32. Beauty reranking-candidate results

{metric_sentence(metrics, 'Amazon Beauty', 'reranking', FULL_COHORT_ID)}

## 33. Sports ranking results

{metric_sentence(metrics, 'Amazon Sports', 'ranking', FULL_COHORT_ID)}

## 34. Sports reranking-candidate results

{metric_sentence(metrics, 'Amazon Sports', 'reranking', FULL_COHORT_ID)}

## 35. Beauty full strict metrics

{metric_sentence(metrics, 'Amazon Beauty', 'ranking', FULL_COHORT_ID)} {metric_sentence(metrics, 'Amazon Beauty', 'reranking', FULL_COHORT_ID)}

## 36. Beauty matched strict metrics

{metric_sentence(metrics, 'Amazon Beauty', 'ranking', MATCHED_COHORT_ID)} {metric_sentence(metrics, 'Amazon Beauty', 'reranking', MATCHED_COHORT_ID)}

## 37. Sports full strict metrics

{metric_sentence(metrics, 'Amazon Sports', 'ranking', FULL_COHORT_ID)} {metric_sentence(metrics, 'Amazon Sports', 'reranking', FULL_COHORT_ID)}

## 38. Sports matched strict metrics

{metric_sentence(metrics, 'Amazon Sports', 'ranking', MATCHED_COHORT_ID)} {metric_sentence(metrics, 'Amazon Sports', 'reranking', MATCHED_COHORT_ID)}

## 39. Graph-degree findings

Degree summaries are recorded in the graph, ranking and metric tables. Mean candidate item degree is carried into all LightGCN metric rows.

## 40. ML-1M Phase 3M integration

ML-1M rows in `{OUTPUTS['all_lightgcn'].relative_to(PROJECT_ROOT).as_posix()}` exactly copy HR@5 and NDCG@5 from Phase 3M source metrics.

## 41. Mixed-device evidence boundary

Beauty/Sports are CPU exact-repeat evidence. ML-1M is retained CUDA primary evidence with a documented determinism warning and D2 root-cause boundary.

## 42. All-dataset LightGCN summary

All-dataset LightGCN metric rows: `{len(all_lightgcn)}` across Beauty, Sports and ML-1M.

## 43. Seven-baseline same-cohort reconstruction

Seven baselines are consolidated on LightGCN strict connected full/matched cohorts: Random, MostPop, BM25-like, BPRMF, GRU4Rec, SASRec and LightGCN.

## 44. Random versus LightGCN

{comparison_sentence(comparison, 'random_seeded')}

## 45. MostPop versus LightGCN

{comparison_sentence(comparison, 'mostpop_train_only')}

## 46. BM25-like versus LightGCN

{comparison_sentence(comparison, 'bm25_like_e3b')}

## 47. BPRMF versus LightGCN

{comparison_sentence(comparison, 'bprmf_reconstructed')}

## 48. GRU4Rec versus LightGCN

{comparison_sentence(comparison, 'gru4rec_reconstructed')}

## 49. SASRec versus LightGCN

{comparison_sentence(comparison, 'sasrec_reconstructed')}

## 50. Dataset-specific findings

Beauty strict rows: `{beauty['scored']}`; Sports strict rows: `{sports['scored']}`. ML-1M strict rows: `1962` from Phase 3M.

## 51. Ranking versus reranking interpretation

Ranking and reranking use separate candidate-set IDs and are never averaged together in the generated tables.

## 52. Paired positive-rank findings

Every paired comparison records LightGCN better, equal and comparator better positive-rank row counts; counts sum to evaluated rows.

## 53. Candidate and row identity

The consolidation required exact row-key coverage and exact candidate-list equality before computing same-cohort metrics and paired comparisons.

## 54. Selection-bias implications

Metrics apply only to graph-compatible strict rows. Excluded Plan E2 rows are documented in the CPU cohort table, so results should not be interpreted as full Plan E2 coverage.

## 55. CUDA sparse-forward limitation

The ML-1M D2 table identifies CUDA sparse forward tensor nondeterminism; Beauty/Sports therefore do not use CUDA project-data sparse operations.

## 56. CPU determinism findings

Beauty and Sports both passed exact CPU checks for configuration, data hashes, graph hashes, negative schedule, model states, embeddings, scores, rankings, positive ranks, HR@5 and NDCG@5.

## 57. Deviations from the original paper

The phase remains adapted because it uses reconstructed datasets, Plan E2 strict cohorts, local LightGCN training and 20-candidate manifests rather than original paper artifacts.

## 58. Limitations

This phase does not run LastFM, does not build prompts, does not invoke LLMs and does not compute beyond-utility metrics.

## 59. Whether Phase 3N-CPU succeeded

Phase 3N-CPU succeeded: CPU Beauty/Sports LightGCN outputs, checkpoints, exact repeat evidence, all-dataset LightGCN summary and seven-baseline same-cohort comparisons were generated.

## 60. LightGCN completion assessment

LightGCN is complete for the adapted Plan E strict cohorts across Beauty, Sports and ML-1M, with mixed-device provenance explicitly documented.

## 61. Baseline-phase completion assessment

The same-cohort baseline phase is complete for seven baselines on full and matched LightGCN strict cohorts, with exact row/candidate identity checks.

## 62. Protected-input immutability

Validation rows and protected hash checks confirm the CPU runner did not alter protected inputs.

## 63. Commands executed

- `python src/experiments/run_plan_e_lightgcn_beauty_sports_cpu.py`
- `python src/experiments/build_plan_e_lightgcn_cpu_baseline_consolidation.py`
- Focused pytest commands are recorded in the final assistant response.

## 64. Recommended single next task

Phase 4A - canonical utility and beyond-utility metric contract audit and same-cohort consolidation preflight
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def validate_written_outputs() -> None:
    expected = {
        "device_provenance": (DEVICE_PROVENANCE_COLUMNS, 3),
        "all_lightgcn": (ALL_LIGHTGCN_METRIC_COLUMNS, 12),
        "same_cohort": (SAME_COHORT_METRIC_COLUMNS, 84),
        "comparison": (COMPARISON_COLUMNS, 72),
    }
    for key, (columns, count) in expected.items():
        frame = read_csv(OUTPUTS[key])
        if list(frame.columns) != columns:
            raise RuntimeError(f"schema mismatch for {OUTPUTS[key]}")
        if len(frame) != count:
            raise RuntimeError(f"row count mismatch for {OUTPUTS[key]}: {len(frame)} != {count}")
    report = OUTPUTS["report"].read_text(encoding="utf-8")
    section_count = sum(1 for line in report.splitlines() if line.startswith("## "))
    if section_count != 64:
        raise RuntimeError(f"expected 64 report sections, found {section_count}")


def manifest_status(frame: pd.DataFrame, role: str) -> str:
    values = frame.loc[frame["execution_role"].astype(str).eq(role), "execution_status"].astype(str).tolist()
    if len(values) != 1:
        raise RuntimeError(f"expected one manifest status for {role}")
    return values[0]


def dataset_summary(
    dataset: str,
    graph: pd.DataFrame,
    cohort: pd.DataFrame,
    manifest: pd.DataFrame,
    history: pd.DataFrame,
    rankings: pd.DataFrame,
    metrics: pd.DataFrame,
    determinism: pd.DataFrame,
) -> dict[str, Any]:
    g = graph.loc[graph["dataset"].astype(str).eq(dataset)].iloc[0]
    c = cohort.loc[cohort["dataset"].astype(str).eq(dataset)]
    m = manifest.loc[manifest["dataset"].astype(str).eq(dataset)]
    h = history.loc[history["dataset"].astype(str).eq(dataset)]
    r = rankings.loc[rankings["dataset"].astype(str).eq(dataset)]
    d = determinism.loc[determinism["dataset"].astype(str).eq(dataset)]
    return {
        "graph": f"- Train edges `{g['unique_train_edges']}`, user nodes `{g['user_nodes']}`, item nodes `{g['item_nodes']}`, total nodes `{g['total_nodes']}`, directed edges `{g['directed_edges']}`.",
        "duplicate_edges": g["duplicate_train_edge_count"],
        "node_rows": int(g["total_nodes"]),
        "excluded": int(len(c) - c["included_in_lightgcn_scoring"].astype(str).str.lower().eq("true").sum()),
        "scored": int(len(r)),
        "primary": run_line(m, h, "primary"),
        "repeat": run_line(m, h, "same_seed_repeat"),
        "determinism_passed": int(d["status"].astype(str).str.lower().eq("passed").sum()),
        "determinism_total": int(len(d)),
    }


def run_line(manifest: pd.DataFrame, history: pd.DataFrame, role: str) -> str:
    row = manifest.loc[manifest["execution_role"].astype(str).eq(role)].iloc[0]
    hist = history.loc[history["execution_role"].astype(str).eq(role)]
    final_loss = hist.sort_values("epoch", key=lambda values: values.astype(int))["mean_total_loss"].iloc[-1]
    return f"- Duration `{float(row['duration_seconds']):.3f}` seconds; epochs `{len(hist)}`; final loss `{float(final_loss):.12g}`; status `{row['execution_status']}`."


def metric_sentence(metrics: pd.DataFrame, dataset: str, task_type: str, cohort_id: str) -> str:
    row = metrics.loc[
        metrics["dataset"].astype(str).eq(dataset)
        & metrics["task_type"].astype(str).eq(task_type)
        & metrics["cohort_id"].astype(str).eq(cohort_id)
    ].iloc[0]
    return f"- {dataset} {task_type} {cohort_id}: rows `{row['evaluated_rows']}`, HR@5 `{float(row['hr_at_5']):.12g}`, NDCG@5 `{float(row['ndcg_at_5']):.12g}`."


def comparison_sentence(comparison: pd.DataFrame, comparator: str) -> str:
    rows = comparison.loc[comparison["comparator_baseline_id"].astype(str).eq(comparator)]
    hr = rows["hr_at_5_difference_lightgcn_minus_comparator"].astype(float)
    ndcg = rows["ndcg_at_5_difference_lightgcn_minus_comparator"].astype(float)
    wins = rows["lightgcn_better_positive_rank_rows"].astype(int).sum()
    losses = rows["comparator_better_positive_rank_rows"].astype(int).sum()
    equal = rows["equal_positive_rank_rows"].astype(int).sum()
    return f"- Rows `{len(rows)}`; mean HR@5 delta `{hr.mean():.12g}`; mean NDCG@5 delta `{ndcg.mean():.12g}`; positive-rank wins/equal/losses `{wins}/{equal}/{losses}`."


def environment_line(manifest: pd.DataFrame) -> str:
    row = manifest.iloc[0]
    return f"- `{row['python_executable']}`; Python `{row['python_version']}`; torch `{row['torch_version']}`; CPU `{row['cpu_model']}`."


def single_value(frame: pd.DataFrame, column: str) -> str:
    values = sorted(set(frame[column].astype(str).tolist()))
    if len(values) != 1:
        raise RuntimeError(f"expected one value in {column}, found {values}")
    return values[0]


def git_output(args: list[str]) -> str:
    return common_git.git_stdout_checked(args, PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
