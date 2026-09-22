"""Build Finalization F5C popularity-bias resolution outputs."""

from __future__ import annotations

from pathlib import Path

from src.analysis.finalization_f5c_popularity_bias import (
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
    metric_inventory = frames["popularity_metric_inventory"]
    item_policy = frames["item_popularity_policy"]
    item_stats = frames["item_popularity_statistics"]
    prediction_inventory = frames["prediction_source_inventory"]
    popularity_results = frames["popularity_results"]
    group_exposure = frames["item_group_exposure"]
    shift_summary = frames["user_popularity_shift_summary"]
    comparisons = frames["valid_popularity_comparisons"]
    reconciliation = frames["popularity_reconciliation"]
    bias_resolution = frames["popularity_bias_resolution"]
    metric_status = frames["metric_status_update"]
    claims = frames["claim_update"]
    coverage = frames["coverage_impact"]
    anomalies = frames["anomalies"]
    validation = frames["validation"]
    hashes_after = output_hashes(root)

    def value(metric_id: str) -> str:
        rows = popularity_results.loc[
            popularity_results["metric_definition_id"].eq(metric_id) & popularity_results["is_reportable"].eq("true")
        ]
        return "" if rows.empty else str(rows.iloc[0]["value"])

    print(f"branch={branch}")
    print(f"head={head}")
    print(f"decision={result['decision']}")
    print(f"metric_inventory_rows={len(metric_inventory)}")
    print(f"definitions_recovered={metric_inventory['f5c_resolution'].isin(['resolved_functional','resolved_extension_only']).sum()}")
    print(f"active_popularity_policies={item_policy['policy_status'].eq('active').sum()}")
    print(f"blocked_popularity_policies={item_policy['policy_status'].eq('blocked').sum()}")
    print(f"item_popularity_statistics_rows={len(item_stats)}")
    print(f"prediction_sources={len(prediction_inventory)}")
    print(f"computable_arp_sources={prediction_inventory['can_compute_arp'].eq('true').sum()}")
    print(f"computable_aplt_sources={prediction_inventory['can_compute_aplt'].eq('true').sum()}")
    print(f"computable_shift_sources={prediction_inventory['can_compute_shift'].eq('true').sum()}")
    print(f"computable_popreo_sources={prediction_inventory['can_compute_popreo'].eq('true').sum()}")
    print(f"popularity_result_rows={len(popularity_results)}")
    print(f"reportable_popularity_results={popularity_results['is_reportable'].eq('true').sum()}")
    print(f"arp_at_5_first_reportable={value('arp_at_5')}")
    print(f"average_popularity_at_5_first_reportable={value('average_recommendation_popularity_at_5')}")
    print(f"aplt_at_5_first_reportable={value('aplt_at_5')}")
    print(f"head_exposure_at_5_first_reportable={value('head_exposure_at_5')}")
    print(f"popularity_shift_at_5_first_reportable={value('popularity_shift_at_5')}")
    print(f"popularity_lift_at_5_first_reportable={value('popularity_lift_at_5')}")
    print("popreo_at_5_status=blocked_missing_denominator_policy")
    print(f"group_exposure_rows={len(group_exposure)}")
    print(f"shift_summary_rows={len(shift_summary)}")
    print(f"valid_comparisons={len(comparisons)}")
    print(f"reconciliation_rows={len(reconciliation)}")
    print(f"popularity_bias_status={bias_resolution.iloc[0]['f5c_status']}")
    for _, row in metric_status.iterrows():
        print(f"metric_status:{row['metric_definition_id']}={row['f5c_status']}")
    print(f"safe_claims={claims['is_safe_for_report'].eq('true').sum()}")
    for _, row in coverage.iterrows():
        print(f"coverage:{row['coverage_metric']}={row['previous_value']}->{row['f5c_value']}:{row['absolute_change']}")
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
