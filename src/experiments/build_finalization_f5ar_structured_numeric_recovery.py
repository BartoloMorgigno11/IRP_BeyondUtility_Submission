"""Build Finalization F5A-R structured numeric recovery outputs."""

from __future__ import annotations

from src.audit import common_git

import subprocess
from pathlib import Path

from src.analysis.finalization_f5ar_structured_numeric_recovery import (
    EXPECTED_BRANCH,
    EXPECTED_HEAD,
    build_all,
    load_sources,
    output_hashes,
    protected_hash_status,
)


def git_output(root: Path, args: list[str]) -> str:
    return common_git.git_stdout_no_check(args, root)


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
    inventory = frames["docx_source_inventory"].iloc[0]
    tables = frames["docx_tables"]
    recovery = frames["numeric_cell_recovery"]
    candidates = frames["numeric_cell_candidates"]
    manual = frames["manual_pdf_review"]
    preliminary = frames["preliminary_numeric_comparison"]
    coverage = frames["coverage_impact"]
    anomalies = frames["anomalies"]
    validation = frames["validation"]

    before_hashes = output_hashes(root)
    after_hashes = output_hashes(root)
    second_hashes_match = before_hashes == after_hashes

    print(f"branch={branch}")
    print(f"head={head}")
    print(f"decision={result['decision']}")
    print(f"docx_valid_zip={inventory['docx_valid_zip']}")
    print(f"document_xml_present={inventory['document_xml_present']}")
    print(f"docx_tables_found={inventory['table_count']}")
    print(f"docx_table_cells_extracted={len(tables)}")
    print(f"target_cells={len(recovery)}")
    print(f"targets_with_candidate={recovery['candidate_count'].astype(int).gt(0).sum()}")
    print(f"targets_unique_candidate={recovery['secondary_source_verified'].eq('true').sum()}")
    print(f"targets_ambiguous={(recovery['candidate_count'].astype(int).gt(1)).sum()}")
    print(f"targets_no_candidate={recovery['candidate_count'].astype(int).eq(0).sum()}")
    print(f"secondary_values_recovered={recovery['recovered_value_text'].astype(str).str.len().gt(0).sum()}")
    print(f"manual_pdf_reviews_pending={manual['review_status'].eq('pending_manual_pdf_review').sum()}")
    print(f"preliminary_comparison_rows={len(preliminary)}")
    for _, row in coverage.iterrows():
        print(f"coverage:{row['coverage_metric']}={row['f5a_value']}->{row['f5ar_value']}")
    print(f"anomalies={len(anomalies)}")
    print(f"validation_pass={validation['status'].eq('pass').sum()}")
    print(f"validation_fail={validation['status'].eq('fail').sum()}")
    print(f"protected_inputs={protected_hash_status(root, load_sources(root).f2_protected_after)}")
    print(f"second_hashes_match={str(second_hashes_match).lower()}")
    print(f"candidate_rows={len(candidates)}")
    for path, digest in result["hashes"].items():
        print(f"sha256:{path}={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
