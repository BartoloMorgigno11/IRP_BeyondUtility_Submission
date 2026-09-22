"""Build Finalization F6D3 LastFM LLM preflight artifacts."""

from __future__ import annotations

from pathlib import Path

from src.analysis.finalization_f6d3_lastfm_llm_preflight import (
    EXPECTED_BRANCH,
    EXPECTED_HEAD,
    OUTPUT_TABLES,
    build_all,
    git_output,
    output_hashes,
    previous_phase_input_status,
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
    validation = frames["validation"]
    go = frames["llm_go_no_go"]
    adequacy = frames["model_adequacy_summary"]
    positive = frames["positive_inclusion_summary"].loc[frames["positive_inclusion_summary"]["summary_key"].eq("positive_present")].iloc[0]

    print(f"branch={branch}")
    print(f"head={head}")
    print(f"decision={result['decision']}")
    print(f"training_adequacy_decision={';'.join(f'{row.model}:{row.model_adequacy_status}' for row in adequacy.itertuples(index=False))}")
    print("reranking_protocol_decision=paper_policy_unrecovered_adapted_policy_frozen")
    for row in go.sort_values(["model", "task"]).itertuples(index=False):
        print(f"{row.model}_{row.task}_decision={row.decision}")
    print(f"ranking_request_count={len(frames['ranking_manifest'])}")
    print(f"reranking_request_count={len(frames['reranking_manifest'])}")
    print(f"total_request_count={len(frames['ranking_manifest']) + len(frames['reranking_manifest'])}")
    print(f"positive_inclusion={positive.users}/{positive.total_users};rate={positive.rate}")
    print(f"validation_pass={validation['status'].eq('pass').sum()}")
    print(f"validation_fail={validation['status'].eq('fail').sum()}")
    print(f"protected_inputs={protected_hash_status(root)}")
    print(f"previous_phase_inputs={previous_phase_input_status(root)}")
    print(f"second_hashes_match={str((not before) or before == after).lower()}")
    for key, path in OUTPUT_TABLES.items():
        print(f"table:{key}={path.as_posix()};rows={len(frames[key])}")
    for path, digest in result["hashes"].items():
        print(f"sha256:{path}={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
