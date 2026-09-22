"""Run Finalization F6D1 LastFM deterministic baselines."""

from __future__ import annotations

from pathlib import Path

from src.analysis.finalization_f6d1_lastfm_baselines import (
    BASELINE_ORDER,
    CONDITIONAL_RERANKING_PATH,
    EXPECTED_HEAD,
    EXPECTED_BRANCH,
    OUTPUT_TABLES,
    PREDICTIONS_PATH,
    build_all,
    git_output,
    output_hashes,
    protected_hash_status,
)


def main() -> int:
    root = Path.cwd()
    branch = git_output(root, ["branch", "--show-current"])
    head = git_output(root, ["rev-parse", "HEAD"])
    if branch != EXPECTED_BRANCH:
        raise SystemExit(f"Unexpected branch: observed={branch} expected={EXPECTED_BRANCH}")
    if head != EXPECTED_HEAD:
        raise SystemExit(f"Unexpected HEAD: observed={head} expected={EXPECTED_HEAD}")

    before = output_hashes(root)
    result = build_all(root)
    after = output_hashes(root)
    frames = result["frames"]
    utility = frames["utility_metrics"]
    instance = frames["instance_summary"]
    manifest = frames["execution_manifest"]
    reranking = frames["reranking_source_audit"]
    validation = frames["validation"]
    missing = frames["missing_evidence_update"]

    print(f"branch={branch}")
    print(f"head={head}")
    print(f"decision={result['decision']}")
    for model in BASELINE_ORDER:
        model_utility = utility.loc[utility["model"].eq(model)]
        model_instance = instance.loc[instance["model"].eq(model)]
        rows = int(len(frames["predictions"].loc[frames["predictions"]["model"].eq(model)]))
        print(f"{model}_status=completed")
        print(f"{model}_users={model_instance['user_id'].nunique()}")
        print(f"{model}_prediction_rows={rows}")
        print(f"{model}_strict_valid_users={model_instance['strict_valid'].eq('true').sum()}")
        for metric in ["HR@5", "NDCG@5", "mean_positive_rank", "median_positive_rank", "duplicate_rate", "invalid_item_rate"]:
            value = model_utility.loc[model_utility["metric"].eq(metric), "value"].iloc[0]
            print(f"{model}_{metric}={value}")
        runtime = manifest.loc[manifest["model"].eq(model), "runtime_seconds"].iloc[0]
        print(f"{model}_runtime_seconds={runtime}")
    print("bm25_status=completed_bm25_like_adapted")
    print("bm25_classification=bm25_like_adapted_not_exact")
    print(f"reranking_reusable_sources={reranking['can_generate_reranking_candidates'].eq('true').sum()}")
    print(f"reranking_candidates_generated={(root / CONDITIONAL_RERANKING_PATH).exists()}")
    print("reranking_candidate_users=0")
    print("reranking_candidate_rows=0")
    print(f"active_gaps_before={missing['was_active_gap'].eq('true').sum()}")
    print(f"active_gaps_after={missing['remains_active_gap'].eq('true').sum()}")
    print(f"active_gaps_partially_resolved={missing['partially_resolved_by_f6d1'].eq('true').sum()}")
    print(f"validation_pass={validation['status'].eq('pass').sum()}")
    print(f"validation_fail={validation['status'].eq('fail').sum()}")
    print(f"protected_inputs={protected_hash_status(root)}")
    print(f"second_hashes_match={str((not before) or before == after).lower()}")
    print(f"predictions={PREDICTIONS_PATH.as_posix()};rows={len(frames['predictions'])}")
    for key, path in OUTPUT_TABLES.items():
        print(f"table:{key}={path.as_posix()};rows={len(frames[key])}")
    for path, digest in result["hashes"].items():
        print(f"sha256:{path}={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

