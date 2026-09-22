"""Build Finalization F6C LastFM readiness outputs."""

from __future__ import annotations

from pathlib import Path

from src.analysis.finalization_f6c_lastfm_readiness import (
    DERIVED_PATHS,
    EXPECTED_BRANCH,
    EXPECTED_HEAD,
    OUTPUT_TABLES,
    build_all,
    file_sha256,
    git_output,
    output_hashes,
    protected_hash_status,
    readiness_statuses,
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
    readiness = readiness_statuses(frames)
    audit = frames["interaction_audit"].set_index("audit_stage")
    metadata = frames["metadata_coverage"].set_index("dataset_stage")
    validation = frames["validation"]
    split_validation = frames["split_validation"]
    candidates = frames["_derived_ranking_candidates"]
    missing = frames["missing_evidence_update"]
    existing = frames["existing_result_inventory"]

    print(f"branch={branch}")
    print(f"head={head}")
    print(f"decision={result['decision']}")
    print(f"lastfm_overall_readiness_status={readiness['overall']}")
    print(f"lastfm_ranking_readiness_status={readiness['ranking']}")
    print(f"lastfm_reranking_readiness_status={readiness['reranking']}")
    print(f"source_inventory_rows={len(frames['source_inventory'])}")
    print(f"selected_source=data/external/plan_c/hetrec2011-lastfm-2k.zip!user_artists.dat")
    print(f"raw_stats=users:{audit.loc['raw','user_count']};items:{audit.loc['raw','item_count']};interactions:{audit.loc['raw','interaction_count']}")
    print(f"reconstructed_stats=users:{audit.loc['final_selected_dataset','user_count']};items:{audit.loc['final_selected_dataset','item_count']};interactions:{audit.loc['final_selected_dataset','interaction_count']}")
    print("paper_stats=unverified_or_mismatch")
    print(f"degree_core_status=user_min:{audit.loc['final_selected_dataset','minimum_user_degree']};item_min:{audit.loc['final_selected_dataset','minimum_item_degree']};class:adapted_5core")
    print(f"timestamp_coverage={audit.loc['final_selected_dataset','timestamp_coverage']}")
    print(f"artist_name_coverage={metadata.loc['final_selected_dataset','name_coverage']}")
    print(f"tag_coverage={metadata.loc['final_selected_dataset','tag_coverage']}")
    print("preprocessing_class=adapted")
    print("split_class=adapted_non_temporal_weight_rank")
    print(f"train_validation_test_users={frames['_derived_train']['user_id'].nunique()}/{frames['_derived_validation']['user_id'].nunique()}/{frames['_derived_test']['user_id'].nunique()}")
    print("candidate_protocol=ranking:20 candidates,1 positive,19 negatives,seed 42,interaction_items universe")
    print(f"ranking_cohort_size={frames['_derived_ranking_cohort']['user_id'].nunique()}")
    print(f"ranking_candidates_generated={candidates['candidate_set_id'].nunique()} sets;{len(candidates)} rows")
    print(f"reranking_cohort_size={frames['_derived_reranking_cohort']['user_id'].nunique()}")
    print(f"existing_lastfm_results={len(existing)} rows;prediction_lists=0")
    print("utility_readiness=ranking_ready_adapted;reranking_blocked")
    print("beyond_utility_readiness=metadata_ready_adapted;prediction_lists_required")
    print("llm_readiness=prompt_inputs_ready_adapted;inference_not_run")
    print(f"missing_evidence_partially_resolved={missing['resolved_by_f6c'].eq('true').sum()}")
    print(f"missing_evidence_residue={missing['resolved_by_f6c'].ne('true').sum()} plus exact gaps on partially resolved rows")
    for _, row in frames["coverage_impact"].iterrows():
        print(f"coverage_change:{row['coverage_family']}={row['previous_value']}->{row['f6c_value']}")
    print(f"anomalies={len(frames['anomalies'])};high_or_critical={frames['anomalies']['severity'].isin(['high','critical']).sum()}")
    print(f"split_validation_pass={split_validation['status'].eq('pass').sum()}")
    print(f"split_validation_fail={split_validation['status'].eq('fail').sum()}")
    print(f"validation_pass={validation['status'].eq('pass').sum()}")
    print(f"validation_fail={validation['status'].eq('fail').sum()}")
    print(f"protected_inputs={protected_hash_status(root)}")
    print(f"second_hashes_match={str((not before) or before == after).lower()}")
    for key, path in OUTPUT_TABLES.items():
        print(f"table:{key}={path.as_posix()};rows={len(frames[key])}")
    derived_rows = {
        "interactions": len(frames["_derived_interactions"]),
        "train": len(frames["_derived_train"]),
        "validation": len(frames["_derived_validation"]),
        "test": len(frames["_derived_test"]),
        "catalog": len(frames["_derived_catalog"]),
        "ranking_cohort": len(frames["_derived_ranking_cohort"]),
        "reranking_cohort": len(frames["_derived_reranking_cohort"]),
        "ranking_candidates": len(frames["_derived_ranking_candidates"]),
    }
    for key, path in DERIVED_PATHS.items():
        print(f"derived:{key}={path.as_posix()};rows={derived_rows[key]}")
    for path, digest in result["hashes"].items():
        print(f"sha256:{path}={digest}")
    print(f"raw_docx_sha256={(file_sha256(root / Path('paper/traduzione_italiana_beyond_utility_llm_recommender.docx')) if (root / Path('paper/traduzione_italiana_beyond_utility_llm_recommender.docx')).exists() else '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

