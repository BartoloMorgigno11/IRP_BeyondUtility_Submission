"""Build Finalization F5D serendipity outputs."""

from __future__ import annotations

from pathlib import Path

from src.analysis.finalization_f5d_serendipity import (
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
    results = frames["serendipity_results"]
    prediction = frames["prediction_source_inventory"]
    summary = frames["serendipity_user_summary"]
    validation = frames["validation"]
    coverage = frames["coverage_impact"]
    status = frames["metric_status_update"].iloc[0]
    claims = frames["claim_update"]
    hashes_after = output_hashes(root)

    reportable = results.loc[results["is_reportable"].eq("true")]
    first = reportable.iloc[0].to_dict() if not reportable.empty else {}
    zero = summary.loc[summary["summary_key"].eq("zero_serendipity_user_rate")]
    ser = summary.loc[summary["summary_key"].eq("at_least_one_serendipitous_item_rate")]

    print(f"branch={branch}")
    print(f"head={head}")
    print(f"decision={result['decision']}")
    print(f"prediction_sources={len(prediction)}")
    print(f"computable_prediction_sources={prediction['can_compute_serendipity'].eq('true').sum()}")
    print(f"serendipity_result_rows={len(results)}")
    print(f"primary_results={results['is_primary_result'].eq('true').sum()}")
    print(f"reportable_results={len(reportable)}")
    print(f"first_reportable_result_id={first.get('serendipity_result_id','')}")
    print(f"first_reportable_value={first.get('value','')}")
    print(f"first_zero_serendipity_rate={zero.iloc[0]['value'] if not zero.empty else ''}")
    print(f"first_serendipitous_item_rate={ser.iloc[0]['value'] if not ser.empty else ''}")
    print(f"metric_status={status['f5d_status']}")
    print(f"safe_claims={claims['is_safe_for_report'].eq('true').sum()}")
    print(f"valid_comparisons={len(frames['valid_serendipity_comparisons'])}")
    print(f"sensitivity_rows={len(frames['serendipity_sensitivity'])}")
    print(f"tradeoff_rows={len(frames['metric_tradeoffs'])}")
    print(f"reconciliation_rows={len(frames['serendipity_reconciliation'])}")
    for _, row in coverage.iterrows():
        print(f"coverage:{row['coverage_metric']}={row['previous_value']}->{row['f5d_value']}:{row['absolute_change']}")
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
