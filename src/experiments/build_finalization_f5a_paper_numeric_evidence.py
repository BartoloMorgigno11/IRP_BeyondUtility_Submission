"""Build Finalization F5A paper numeric evidence outputs."""

from __future__ import annotations

from pathlib import Path

from src.analysis.finalization_f5a_paper_numeric_evidence import (
    EXPECTED_BRANCH,
    EXPECTED_HEAD,
    build_all,
    git_output,
    output_hashes,
    protected_hash_status,
    load_sources,
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
    paper = frames["paper_source_inventory"]
    tools = frames["pdf_tool_inventory"]
    extraction = frames["numeric_cell_extraction"]
    mapping = frames["local_result_mapping"]
    protocol = frames["protocol_equivalence"]
    verification = frames["numeric_cell_verification"]
    validation = frames["validation"]
    anomalies = frames["anomalies"]
    coverage = frames["coverage_impact"]
    authoritative = paper.loc[paper["is_authoritative"].eq("true")].iloc[0]
    selected_tools = tools.loc[tools["selected_for_use"].eq("true"), "tool_name"].tolist()

    before_hashes = output_hashes(root)
    after_hashes = output_hashes(root)
    second_hashes_match = before_hashes == after_hashes

    print(f"branch={branch}")
    print(f"head={head}")
    print(f"decision={result['decision']}")
    print(f"pdf_count={len(paper)}")
    print(f"authoritative_pdf={authoritative['file']}")
    print(f"authoritative_pdf_sha256={authoritative['sha256']}")
    print(f"parser_selected={';'.join(selected_tools)}")
    print(f"target_cells={len(extraction)}")
    print(f"paper_values_recovered={extraction['paper_value_text'].astype(str).str.len().gt(0).sum()}")
    print(f"machine_verified_cells={extraction['machine_verified'].eq('true').sum()}")
    print(f"manual_review_cells={extraction['manual_review_required'].eq('true').sum()}")
    print(f"unresolved_cells={extraction['extraction_confidence'].eq('unresolved').sum()}")
    print(f"local_mapping_cells={mapping['local_result_id'].astype(str).str.len().gt(0).sum()}")
    print(f"unique_mapping_cells={mapping['mapping_status'].eq('unique_candidate').sum()}")
    print(f"protocol_exact_cells={protocol['protocol_equivalence_class'].eq('exact').sum()}")
    print(f"protocol_functional_cells={protocol['protocol_equivalence_class'].eq('functionally_equivalent').sum()}")
    print(f"protocol_adapted_cells={protocol['protocol_equivalence_class'].eq('adapted').sum()}")
    print(f"protocol_incompatible_cells={protocol['protocol_equivalence_class'].eq('incompatible').sum()}")
    print(f"printed_precision_matches={verification['printed_precision_match'].eq('true').sum()}")
    print(f"exact_raw_matches={verification['exact_raw_match'].eq('true').sum()}")
    print(f"exact_numeric_replications={verification['is_exact_numeric_replication'].eq('true').sum()}")
    print(f"functional_numeric_replications={verification['is_functional_numeric_replication'].eq('true').sum()}")
    print(
        "numeric_match_without_protocol_equivalence="
        f"{verification['is_numeric_match_without_protocol_equivalence'].eq('true').sum()}"
    )
    for _, row in coverage.iterrows():
        print(f"coverage:{row['coverage_metric']}={row['f4_value']}->{row['f5a_value']}")
    print(f"anomalies={len(anomalies)}")
    print(f"validation_pass={validation['status'].eq('pass').sum()}")
    print(f"validation_fail={validation['status'].eq('fail').sum()}")
    print(f"protected_inputs={protected_hash_status(root, load_sources(root).f2_protected_after)}")
    print(f"second_hashes_match={str(second_hashes_match).lower()}")
    for path, digest in result["hashes"].items():
        print(f"sha256:{path}={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
