"""Run Finalization F6D2 LastFM trainable traditional models."""

from __future__ import annotations

from pathlib import Path

from src.analysis.finalization_f6d2_lastfm_trainable_models import (
    ADAPTED_POOL_PATH,
    CANDIDATE_PREDICTIONS_PATH,
    EXPECTED_BRANCH,
    EXPECTED_HEAD,
    FULL_CATALOG_PREDICTIONS_PATH,
    MODEL_ORDER,
    OUTPUT_TABLES,
    build_all,
    git_output,
    output_hashes,
    phase_input_hash_status,
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
    manifest = frames["execution_manifest"]
    seed_metrics = frames["seed_utility_metrics"]
    aggregate = frames["aggregate_utility_metrics"]
    validation = frames["validation"]
    pool = frames["reranking_pool"]
    source_selection = frames["reranking_source_selection"]

    print(f"branch={branch}")
    print(f"head={head}")
    print(f"decision={result['decision']}")
    print(f"cuda_available={manifest['notes'].astype(str).str.contains('CUDA available=True').any()}")
    print(f"device_values={';'.join(sorted(manifest['device'].astype(str).unique()))}")
    print(f"seed_policy={';'.join(str(seed) for seed in sorted(manifest['seed'].astype(int).unique()))}")
    print(f"candidate_prediction_rows={len(frames['candidate_predictions'])}")
    print(f"full_catalog_prediction_rows={len(frames['full_catalog_predictions'])}")
    print(f"reranking_pool_rows={len(pool)}")
    print(f"reranking_pool_users={pool['user_id'].nunique() if not pool.empty else 0}")
    positive_users = pool.loc[pool["is_positive"].eq("true"), "user_id"].nunique() if not pool.empty else 0
    pool_users = pool["user_id"].nunique() if not pool.empty else 0
    positive_rate = positive_users / pool_users if pool_users else 0.0
    print(f"reranking_pool_positive_inclusion_rate={positive_rate:.12g}")
    print(f"paper_aligned_pool_created=false")
    print(f"adapted_pool_created={(root / ADAPTED_POOL_PATH).exists()}")
    print(f"protected_inputs={protected_hash_status(root)}")
    print(f"f6c_f6d1_inputs={phase_input_hash_status(root)}")
    print(f"second_hashes_match={str((not before) or before == after).lower()}")
    print(f"validation_pass={validation['status'].eq('pass').sum()}")
    print(f"validation_fail={validation['status'].eq('fail').sum()}")
    print(f"candidate_predictions={CANDIDATE_PREDICTIONS_PATH.as_posix()}")
    print(f"full_catalog_predictions={FULL_CATALOG_PREDICTIONS_PATH.as_posix()}")
    print(f"adapted_pool={ADAPTED_POOL_PATH.as_posix()}")

    for model in MODEL_ORDER:
        model_manifest = manifest.loc[manifest["model"].eq(model)]
        selected = source_selection.loc[(source_selection["model"].eq(model)) & (source_selection["selected_for_adapted_pool"].eq("true"))]
        print(f"{model}_completed_seeds={model_manifest['seed'].nunique()}")
        print(f"{model}_selected_seed={selected['seed'].iloc[0] if not selected.empty else ''}")
        for seed in sorted(model_manifest["seed"].astype(int).unique()):
            run = model_manifest.loc[model_manifest["seed"].astype(int).eq(seed)].iloc[0]
            print(f"{model}_seed_{seed}_runtime_seconds={run['runtime_seconds']}")
            print(f"{model}_seed_{seed}_peak_gpu_memory_mb={run['peak_gpu_memory_mb']}")
            print(f"{model}_seed_{seed}_best_epoch={run['best_epoch']}")
        for metric in ["HR@5", "NDCG@5", "mean_positive_rank"]:
            seed_values = seed_metrics.loc[
                (seed_metrics["model"].eq(model))
                & (seed_metrics["evaluation_scope"].eq("candidate_set_20"))
                & (seed_metrics["metric"].eq(metric)),
                ["seed", "value"],
            ]
            print(f"{model}_{metric}_by_seed={';'.join(f'{row.seed}:{row.value}' for row in seed_values.itertuples(index=False))}")
            aggregate_value = aggregate.loc[
                (aggregate["model"].eq(model))
                & (aggregate["evaluation_scope"].eq("candidate_set_20"))
                & (aggregate["metric"].eq(metric)),
                "mean",
            ].iloc[0]
            print(f"{model}_{metric}_aggregate={aggregate_value}")

    for key, path in OUTPUT_TABLES.items():
        print(f"table:{key}={path.as_posix()};rows={len(frames[key])}")
    for path, digest in result["hashes"].items():
        print(f"sha256:{path}={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
