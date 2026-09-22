"""Build Finalization F5E calibration outputs."""

from __future__ import annotations

from pathlib import Path

from src.analysis.finalization_f5e_calibration import (
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
    prediction = frames["prediction_source_inventory"]
    results = frames["calibration_results"]
    coverage = frames["calibration_coverage"]
    sensitivity = frames["calibration_sensitivity"]
    comparisons = frames["valid_calibration_comparisons"]
    reconciliation = frames["calibration_reconciliation"]
    tradeoffs = frames["metric_tradeoffs"]
    status = frames["metric_status_update"].iloc[0]
    paper = frames["paper_calibration_status"].iloc[0]
    validation = frames["validation"]
    claims = frames["claim_update"]
    hashes_after = output_hashes(root)

    reportable = results.loc[results["is_reportable"].eq("true")]
    divergence = reportable.loc[reportable["metric"].eq("calibration_divergence_at_5")]
    score = reportable.loc[reportable["metric"].eq("calibration_score_at_5")]
    first_div = divergence.iloc[0].to_dict() if not divergence.empty else {}
    best_score = score.sort_values("value", ascending=False, key=lambda col: col.astype(float), kind="mergesort").iloc[0].to_dict() if not score.empty else {}

    print(f"branch={branch}")
    print(f"head={head}")
    print(f"decision={result['decision']}")
    print(f"paper_contains_calibration={paper['paper_contains_calibration_metric']}")
    print(f"upstream_contains_calibration={paper['upstream_contains_calibration']}")
    print(f"classification={paper['classification']}")
    print(f"prediction_sources={len(prediction)}")
    print(f"computable_prediction_sources={prediction['can_compute_calibration'].eq('true').sum()}")
    print(f"datasets={','.join(sorted(prediction.loc[prediction['can_compute_calibration'].eq('true'), 'dataset'].unique()))}")
    print(f"blocked_datasets={','.join(sorted(prediction.loc[prediction['can_compute_calibration'].eq('false'), 'dataset'].unique()))}")
    print(f"model_count={prediction.loc[prediction['can_compute_calibration'].eq('true'), 'model'].nunique()}")
    print(f"calibration_result_rows={len(results)}")
    print(f"primary_results={results['is_primary_result'].eq('true').sum()}")
    print(f"reportable_results={len(reportable)}")
    print(f"first_reportable_divergence_id={first_div.get('calibration_result_id','')}")
    print(f"first_reportable_divergence_value={first_div.get('value','')}")
    print(f"best_reportable_score_id={best_score.get('calibration_result_id','')}")
    print(f"best_reportable_score_value={best_score.get('value','')}")
    print(f"mean_valid_user_rate={coverage['valid_user_rate'].replace('', '0').astype(float).mean() if not coverage.empty else ''}")
    print(f"mean_history_feature_coverage={coverage['history_feature_coverage'].replace('', '0').astype(float).mean() if not coverage.empty else ''}")
    print(f"mean_recommendation_feature_coverage={coverage['recommendation_feature_coverage'].replace('', '0').astype(float).mean() if not coverage.empty else ''}")
    print(f"sensitivity_rows={len(sensitivity)}")
    print(f"sensitivity_same_order_rows={sensitivity['same_model_ordering'].eq('true').sum() if not sensitivity.empty else 0}")
    print(f"valid_comparisons={len(comparisons)}")
    print(f"reconciliation_rows={len(reconciliation)}")
    print(f"tradeoff_rows={len(tradeoffs)}")
    print(f"metric_status={status['f5e_status']}")
    print(f"safe_claims={claims['is_safe_for_report'].eq('true').sum()}")
    for _, row in frames["coverage_impact"].iterrows():
        print(f"coverage:{row['coverage_metric']}={row['previous_value']}->{row['f5e_value']}:{row['absolute_change']}")
    print(f"anomalies={len(frames['anomalies'])}")
    print(f"validation_pass={validation['status'].eq('pass').sum()}")
    print(f"validation_fail={validation['status'].eq('fail').sum()}")
    print(f"protected_inputs={protected_hash_status(root)}")
    print(f"second_hashes_match={str((not hashes_before) or hashes_before == hashes_after).lower()}")
    for path, digest in result["hashes"].items():
        print(f"sha256:{path}={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

