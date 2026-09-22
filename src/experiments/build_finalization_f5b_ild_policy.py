"""Build Finalization F5B adapted ILD policy outputs."""

from __future__ import annotations

from pathlib import Path

from src.analysis.finalization_f5b_ild_policy import (
    EXPECTED_BRANCH,
    EXPECTED_HEAD,
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

    hashes_before = output_hashes(root)
    result = build_all(root)
    frames = result["frames"]
    feature_inventory = frames["item_feature_source_inventory"]
    policy = frames["ild_policy"]
    prediction_inventory = frames["prediction_source_inventory"]
    feature_coverage = frames["item_feature_coverage"]
    ild_results = frames["ild_results"]
    sensitivity = frames["ild_sensitivity"]
    comparisons = frames["valid_ild_comparisons"]
    reconciliation = frames["ild_reconciliation"]
    metric_status = frames["metric_status_update"]
    claims = frames["claim_update"]
    coverage = frames["coverage_impact"]
    anomalies = frames["anomalies"]
    validation = frames["validation"]

    hashes_after = output_hashes(root)

    print(f"branch={branch}")
    print(f"head={head}")
    print(f"decision={result['decision']}")
    print(f"feature_sources={len(feature_inventory)}")
    print(f"primary_policies={policy['is_primary_policy'].eq('true').sum()}")
    print(f"sensitivity_policies={policy['is_sensitivity_policy'].eq('true').sum()}")
    print(f"prediction_sources={len(prediction_inventory)}")
    print(f"computable_prediction_sources={prediction_inventory['can_compute_ild'].eq('true').sum()}")
    print(f"primary_prediction_sources={prediction_inventory['is_primary_source'].eq('true').sum()}")
    print(f"ild_result_rows={len(ild_results)}")
    print(f"primary_ild_results={ild_results['is_primary_result'].eq('true').sum()}")
    print(f"reportable_ild_results={ild_results['is_reportable'].eq('true').sum()}")
    print(f"sensitivity_rows={len(sensitivity)}")
    print(f"valid_comparisons={len(comparisons)}")
    print(f"reconciliation_rows={len(reconciliation)}")
    print(f"metric_status={metric_status.iloc[0]['f5b_status']}")
    print(f"safe_claims={claims['is_safe_for_report'].eq('true').sum()}")
    for _, row in coverage.iterrows():
        print(f"coverage:{row['coverage_metric']}={row['previous_value']}->{row['f5b_value']}:{row['absolute_change']}")
    for _, row in feature_coverage.iterrows():
        print(
            "feature_coverage:"
            f"{row['dataset']}:{row['ild_policy_id']}="
            f"{row['feature_coverage']}:recommendation={row['recommendation_feature_coverage']}:valid_users={row['valid_user_rate']}"
        )
    print(f"anomalies={len(anomalies)}")
    print(f"validation_pass={validation['status'].eq('pass').sum()}")
    print(f"validation_fail={validation['status'].eq('fail').sum()}")
    print(f"protected_inputs={protected_hash_status(root)}")
    print(f"second_hashes_match={str((not hashes_before) or hashes_before == hashes_after).lower()}")
    for path, digest in result["hashes"].items():
        print(f"sha256:{path}={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
