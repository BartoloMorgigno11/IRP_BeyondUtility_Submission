"""Build Finalization F5A-FIX atomic paper-cell correction outputs."""

from __future__ import annotations

from pathlib import Path

from src.analysis.finalization_f5af_atomic_paper_cells import (
    EXPECTED_BRANCH,
    EXPECTED_HEAD,
    build_all,
    git_output,
    load_sources,
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

    result = build_all(root)
    frames = result["frames"]
    superseded = frames["superseded_target_audit"]
    models = frames["paper_model_inventory"]
    atomic = frames["atomic_paper_result_cells"]
    mapping = frames["local_atomic_mapping"]
    coverage = frames["corrected_coverage"]
    anomalies = frames["anomalies"]
    validation = frames["validation"]

    hashes_before = output_hashes(root)
    hashes_after = output_hashes(root)

    print(f"branch={branch}")
    print(f"head={head}")
    print(f"decision={result['decision']}")
    print(f"superseded_targets={len(superseded)}")
    print(f"model_identity_rows={len(models)}")
    print(f"non_atomic_metric_family_rows={superseded['corrected_target_type'].eq('non_atomic_metric_family').sum()}")
    print(f"atomic_cells={len(atomic)}")
    print(f"pdf_verified_cells={atomic['pdf_verified'].eq('true').sum()}")
    print(f"local_mapped_cells={mapping['local_result_id'].astype(str).str.len().gt(0).sum()}")
    print(f"protocol_exact_cells={mapping['protocol_equivalence_class'].eq('exact').sum()}")
    print(f"printed_matches={mapping['printed_precision_match'].eq('true').sum()}")
    print(f"exact_numeric_replications={mapping['exact_numeric_replication'].eq('true').sum()}")
    for _, row in coverage.iterrows():
        print(f"coverage:{row['coverage_metric']}={row['numerator']}/{row['denominator']}:{row['coverage_percent']}")
    print(f"anomalies={len(anomalies)}")
    print(f"validation_pass={validation['status'].eq('pass').sum()}")
    print(f"validation_fail={validation['status'].eq('fail').sum()}")
    print(f"protected_inputs={protected_hash_status(root, load_sources(root).protected_inputs)}")
    print(f"second_hashes_match={str(hashes_before == hashes_after).lower()}")
    for path, digest in result["hashes"].items():
        print(f"sha256:{path}={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
