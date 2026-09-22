from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

from src.analysis.plan_e_lightgcn_cpu_baseline_consolidation import (
    ALL_BASELINES,
    COMPARATORS,
    DEVICE_PROVENANCE_COLUMNS,
    FULL_COHORT_ID,
    LIGHTGCN_BASELINE_ID,
    MATCHED_COHORT_ID,
    build_paired_comparisons,
    build_same_cohort_metrics,
    cohort_keys_from_lightgcn_rankings,
    exact_determinism_row,
    filter_rankings_to_cohort,
    metrics_from_rankings,
    normalize_bprmf_rankings,
    normalize_gru4rec_rankings,
    normalize_lightgcn_rankings,
    normalize_sasrec_rankings,
    normalize_traditional_rankings,
    paired_positive_rank_summary,
    parse_json_array,
    require_cpu_module,
    require_cpu_tensor,
    require_exact_row_key_equality,
    validate_comparison_frame,
    validate_device_provenance,
    validate_exact_determinism,
    validate_metric_ranges,
    verify_candidate_list_equality,
)


def test_cpu_enforcement_accepts_cpu_and_rejects_cuda_when_available() -> None:
    tensor = torch.ones(2)
    model = torch.nn.Linear(2, 1)

    require_cpu_tensor(tensor, name="synthetic_tensor")
    require_cpu_module(model)

    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="CPU"):
            require_cpu_tensor(torch.ones(1, device="cuda"), name="cuda_tensor")
        with pytest.raises(ValueError, match="CPU"):
            require_cpu_module(torch.nn.Linear(1, 1).to("cuda"))


def test_exact_determinism_requires_zero_tolerance_and_passed_status() -> None:
    row = exact_determinism_row(
        check_id="det_001",
        experiment_id="exp",
        dataset="D1",
        primary_run_id="p",
        repeat_run_id="r",
        check_name="final state",
        primary_value="abc",
        repeat_value="abc",
        evidence="hash",
    )
    frame = pd.DataFrame([row])
    validate_exact_determinism(frame)

    failed = frame.copy()
    failed.loc[0, "repeat_value"] = "def"
    failed.loc[0, "status"] = "failed"
    with pytest.raises(ValueError, match="failed"):
        validate_exact_determinism(failed)

    tolerant = frame.copy()
    tolerant.loc[0, "tolerance"] = "1e-12"
    with pytest.raises(ValueError, match="zero tolerance"):
        validate_exact_determinism(tolerant)


def test_json_parsing_and_empty_cohort_failure() -> None:
    assert parse_json_array('["001", 2]') == ["001", "2"]
    with pytest.raises(ValueError):
        parse_json_array('{"not": "array"}')
    with pytest.raises(ValueError, match="empty cohort"):
        filter_rankings_to_cohort(_rankings("random_seeded"), set())


def test_exact_row_key_filtering_and_source_immutability() -> None:
    frame = _rankings("random_seeded", users=("u1", "u2", "u3"))
    original = frame.copy(deep=True)
    filtered = filter_rankings_to_cohort(frame, {("D1", "ranking", "cs_rank", "u2")})

    assert filtered["user_id"].tolist() == ["u2"]
    pd.testing.assert_frame_equal(frame, original)


def test_lightgcn_bprmf_gru4rec_sasrec_row_equality_and_failure() -> None:
    lightgcn = _lightgcn_rankings()
    for neural in (
        _rankings("bprmf_reconstructed", users=("u1", "u2")),
        _rankings("gru4rec_reconstructed", users=("u1", "u2")),
        _rankings("sasrec_reconstructed", users=("u1", "u2")),
    ):
        require_exact_row_key_equality(lightgcn, neural)

    with pytest.raises(ValueError, match="row-key sets differ"):
        require_exact_row_key_equality(lightgcn, _rankings("bprmf_reconstructed", users=("u1", "u3")))


def test_full_and_matched_lightgcn_cohort_filtering() -> None:
    lightgcn = _lightgcn_rankings()
    full_keys = cohort_keys_from_lightgcn_rankings(lightgcn, FULL_COHORT_ID)
    matched_keys = cohort_keys_from_lightgcn_rankings(lightgcn, MATCHED_COHORT_ID)

    assert len(full_keys) == 2
    assert matched_keys == [("D1", "ranking", "cs_rank", "u1")]


def test_missing_extra_and_candidate_list_mismatch_fail() -> None:
    with pytest.raises(ValueError, match="missing"):
        filter_rankings_to_cohort(
            _rankings("random_seeded", users=("u1",)),
            {("D1", "ranking", "cs_rank", "u1"), ("D1", "ranking", "cs_rank", "u2")},
        )
    with pytest.raises(ValueError, match="extra"):
        filter_rankings_to_cohort(
            _rankings("random_seeded", users=("u1", "u2")),
            {("D1", "ranking", "cs_rank", "u1")},
            require_no_extras=True,
        )
    left = _rankings("a", users=("u1",))
    right = _rankings("b", users=("u1",))
    right.loc[0, "candidate_items"] = json.dumps(["i9", "i2", "i3"])
    with pytest.raises(ValueError, match="candidate-list mismatch"):
        verify_candidate_list_equality(left, right)


def test_metric_recomputation_and_metric_range_validation() -> None:
    rows = _rankings("random_seeded", users=("u1", "u2"))
    metrics = metrics_from_rankings(
        rows,
        baseline_id="random_seeded",
        cohort_id=FULL_COHORT_ID,
        ranking_source_path="rankings.csv",
        cohort_membership_source_path="cohort.csv",
        source_execution_device="not_applicable",
        source_determinism_status="deterministic_saved_rankings",
        paper_alignment="adapted",
        notes="synthetic",
    )

    assert metrics["evaluated_rows"] == 2
    assert metrics["candidate_count"] == 3
    assert metrics["hr_at_5"] == 1.0
    assert metrics["ndcg_at_5"] == pytest.approx((1.0 + 0.5) / 2.0)
    validate_metric_ranges(pd.DataFrame([metrics]))


def test_paired_positive_rank_counts_and_win_rates() -> None:
    lightgcn = _rankings(LIGHTGCN_BASELINE_ID, users=("u1", "u2", "u3"), ranks=(0, 2, 1))
    comp = _rankings("random_seeded", users=("u1", "u2", "u3"), ranks=(1, 2, 0))
    paired = paired_positive_rank_summary(lightgcn, comp)

    assert paired["lightgcn_better_positive_rank_rows"] == 1
    assert paired["equal_positive_rank_rows"] == 1
    assert paired["comparator_better_positive_rank_rows"] == 1
    assert paired["lightgcn_positive_rank_win_rate"] == pytest.approx(1 / 3)


def test_normalizers_preserve_source_fields_and_ml1m_warning_status() -> None:
    base = _rankings("x", users=("u1",))

    lightgcn = normalize_lightgcn_rankings(
        base.assign(eligible_pilot_full="true", eligible_pilot_matched="true"),
        source_path="lg.csv",
        source_execution_device="cuda",
        source_determinism_status="completed_with_determinism_warning",
    )
    assert lightgcn["baseline_id"].iloc[0] == LIGHTGCN_BASELINE_ID
    assert lightgcn["source_determinism_status"].iloc[0] == "completed_with_determinism_warning"
    assert lightgcn["eligible_lightgcn_strict_full"].iloc[0] == "true"

    assert normalize_bprmf_rankings(base.assign(eligible_pilot_full="true"), source_path="b.csv")[
        "baseline_id"
    ].iloc[0] == "bprmf_reconstructed"
    assert normalize_gru4rec_rankings(base.assign(eligible_pilot_full="true"), source_path="g.csv")[
        "baseline_id"
    ].iloc[0] == "gru4rec_reconstructed"
    assert normalize_sasrec_rankings(base.assign(eligible_pilot_full="true"), source_path="s.csv")[
        "baseline_id"
    ].iloc[0] == "sasrec_reconstructed"
    assert normalize_traditional_rankings(base.assign(baseline_id="random_seeded"), source_path="t.csv")[
        "source_execution_device"
    ].iloc[0] == "not_applicable"


def test_build_same_cohort_metrics_separates_dataset_task_and_cohort() -> None:
    lightgcn = pd.concat(
        [
            _lightgcn_rankings(dataset="D1", task="ranking", candidate_set="cs_rank"),
            _lightgcn_rankings(dataset="D1", task="reranking", candidate_set="cs_rerank"),
            _lightgcn_rankings(dataset="D2", task="ranking", candidate_set="cs2_rank"),
        ],
        ignore_index=True,
    )
    bprmf = _concat_baseline("bprmf_reconstructed")
    gru = _concat_baseline("gru4rec_reconstructed")
    sas = _concat_baseline("sasrec_reconstructed")
    trad_frames = []
    for baseline in ("random_seeded", "mostpop_train_only", "bm25_like_e3b"):
        trad_frames.extend(
            [
                _rankings(baseline, dataset="D1", task="ranking", candidate_set="cs_rank"),
                _rankings(baseline, dataset="D1", task="reranking", candidate_set="cs_rerank"),
                _rankings(baseline, dataset="D2", task="ranking", candidate_set="cs2_rank"),
            ]
        )

    metrics, filtered = build_same_cohort_metrics(
        traditional_rankings=pd.concat(trad_frames, ignore_index=True),
        bprmf_rankings=bprmf,
        gru4rec_rankings=gru,
        sasrec_rankings=sas,
        lightgcn_rankings=lightgcn,
        cohort_source_path_by_dataset={"D1": "cohort1.csv", "D2": "cohort2.csv"},
    )

    assert set(metrics["baseline_id"]) == set(ALL_BASELINES)
    assert set(metrics["dataset"]) == {"D1", "D2"}
    assert set(metrics["task_type"]) == {"ranking", "reranking"}
    assert set(metrics["cohort_id"]) == {FULL_COHORT_ID, MATCHED_COHORT_ID}
    assert (LIGHTGCN_BASELINE_ID, "D1", "ranking", "cs_rank", FULL_COHORT_ID) in filtered


def test_build_paired_comparisons_is_deterministic_and_valid() -> None:
    lightgcn = _lightgcn_rankings()
    bprmf = _rankings("bprmf_reconstructed")
    gru = _rankings("gru4rec_reconstructed")
    sas = _rankings("sasrec_reconstructed")
    trad = pd.concat(
        [_rankings(baseline) for baseline in ("random_seeded", "mostpop_train_only", "bm25_like_e3b")],
        ignore_index=True,
    )
    metrics, filtered = build_same_cohort_metrics(
        traditional_rankings=trad,
        bprmf_rankings=bprmf,
        gru4rec_rankings=gru,
        sasrec_rankings=sas,
        lightgcn_rankings=lightgcn,
        cohort_source_path_by_dataset={"D1": "cohort.csv"},
    )
    first = build_paired_comparisons(metrics, filtered)
    second = build_paired_comparisons(metrics.sample(frac=1.0, random_state=1).reset_index(drop=True), filtered)

    pd.testing.assert_frame_equal(first, second)
    validate_comparison_frame(first)
    assert len(first) == len(COMPARATORS) * 2


def test_device_provenance_validation() -> None:
    frame = pd.DataFrame(
        [
            _device_row("Amazon Beauty", "cpu"),
            _device_row("Amazon Sports", "cpu"),
            _device_row("ML-1M", "cuda", status="completed_with_determinism_warning"),
        ],
        columns=DEVICE_PROVENANCE_COLUMNS,
    )

    validate_device_provenance(frame)

    bad = frame.drop(index=0).reset_index(drop=True)
    with pytest.raises(ValueError, match="expected 3"):
        validate_device_provenance(bad)


def _lightgcn_rankings(dataset: str = "D1", task: str = "ranking", candidate_set: str = "cs_rank") -> pd.DataFrame:
    frame = _rankings(LIGHTGCN_BASELINE_ID, dataset=dataset, task=task, candidate_set=candidate_set, users=("u1", "u2"))
    frame["eligible_lightgcn_strict_full"] = ["true", "true"]
    frame["eligible_lightgcn_strict_matched"] = ["true", "false"]
    frame["ranking_source_path"] = "lightgcn.csv"
    frame["source_execution_device"] = "cpu"
    frame["source_determinism_status"] = "completed_exact_cpu_repeat"
    return frame


def _concat_baseline(baseline: str) -> pd.DataFrame:
    return pd.concat(
        [
            _rankings(baseline, dataset="D1", task="ranking", candidate_set="cs_rank"),
            _rankings(baseline, dataset="D1", task="reranking", candidate_set="cs_rerank"),
            _rankings(baseline, dataset="D2", task="ranking", candidate_set="cs2_rank"),
        ],
        ignore_index=True,
    )


def _rankings(
    baseline: str,
    *,
    dataset: str = "D1",
    task: str = "ranking",
    candidate_set: str = "cs_rank",
    users: tuple[str, ...] = ("u1", "u2"),
    ranks: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    ranks = ranks or tuple(range(len(users)))
    rows = []
    for user, rank in zip(users, ranks):
        candidates = ["i1", "i2", "i3"]
        ranked = [candidates[rank], *[item for item in candidates if item != candidates[rank]]]
        rows.append(
            {
                "baseline_id": baseline,
                "dataset": dataset,
                "task_type": task,
                "candidate_set_id": candidate_set,
                "source_path": "source.csv",
                "user_id": user,
                "positive_item_id": candidates[rank],
                "candidate_items": json.dumps(candidates),
                "candidate_scores": json.dumps([0.3, 0.2, 0.1]),
                "ranked_items": json.dumps(ranked),
                "ranked_scores": json.dumps([1.0, 0.5, 0.25]),
                "positive_rank_zero_based": rank,
                "positive_rank_one_based": rank + 1,
                "positive_in_top_5": "true",
                "ndcg_at_5": 1.0 / (rank + 1),
                "ranking_valid": "true",
                "duplicate_ranked_item_count": 0,
                "missing_score_count": 0,
                "ranking_source_path": f"{baseline}.csv",
                "source_execution_device": "cuda" if baseline.endswith("reconstructed") else "not_applicable",
                "source_determinism_status": "completed",
            }
        )
    return pd.DataFrame(rows)


def _device_row(dataset: str, device: str, status: str = "completed") -> dict[str, str]:
    return {
        "dataset": dataset,
        "execution_device": device,
        "source_phase": "synthetic",
        "source_config_path": "config.yaml",
        "source_rankings_path": "rankings.csv",
        "source_metrics_path": "metrics.csv",
        "primary_execution_status": status,
        "repeat_execution_status": status,
        "determinism_classification": status,
        "exact_final_model_state": "true",
        "exact_propagated_embeddings": "true",
        "exact_candidate_scores": "true",
        "exact_rankings": "true",
        "loss_telemetry_exact": "true",
        "cuda_sparse_forward_limitation": "false",
        "cpu_scale_up_decision_source": "synthetic",
        "paper_alignment": "adapted_paper_like_reconstructed",
        "notes": "",
    }
