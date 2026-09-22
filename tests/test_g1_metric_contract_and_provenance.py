"""Regression checks for the final G1 metric contract and provenance authority."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.analysis.g2_history_length_metrics import reciprocal_rank as g2_reciprocal_rank
from src.analysis.g4_f1bzq_metrics import compute_row_contributions
from src.analysis.rq3_completion_scientific_freeze import utility as rq3_utility
from src.extension.f1a_offline_metric_closure import reciprocal_rank as g1_reciprocal_rank
from src.extension.g3_candidate_position_metrics import utility_from_ranked_indices


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "results/manifests/g1_metric_contract_and_provenance.json"
CURRENT_HASH = "2e5c774f552cb47f16622f2a989ce033ee05036ad814558926b40148e8857c5b"
PRE_FIX_HASH = "15ca9025aada73feba7513e06756b6c3ede42542befd82c1a76dbc757b018ae1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranked_with_positive_at(rank_one_based: int | None) -> list[int]:
    ranked = list(range(1, 11))
    if rank_one_based is not None:
        ranked[rank_one_based - 1] = 20
    return ranked


def test_mrr_independent_implementations_are_semantically_equivalent() -> None:
    cases = [(1, True, 1.0), (5, True, 0.2), (10, True, 0.1), (None, True, 0.0), (None, False, 0.0)]
    g4_rows = []
    for index, (rank, strict_valid, _) in enumerate(cases):
        ranked = ranked_with_positive_at(rank) if strict_valid else []
        g4_rows.append(
            {
                "canonical_request_id": f"case-{index}",
                "strict_parse_valid": strict_valid,
                "positive_present_in_pool": True,
                "backend_success": True,
                "parsed_top10_candidate_indices_if_strict_valid": json.dumps(ranked),
                "parsed_top10_item_ids_if_strict_valid": json.dumps([str(item) for item in ranked]),
                "candidate_count": 20,
                "positive_input_position": 20,
            }
        )
    g4, _, defects = compute_row_contributions(pd.DataFrame(g4_rows))
    assert defects == []

    for index, (rank, strict_valid, expected) in enumerate(cases):
        rank_zero = None if rank is None else rank - 1
        ranked = ranked_with_positive_at(rank) if strict_valid else []
        assert g1_reciprocal_rank(rank_zero if strict_valid else None) == expected
        assert g2_reciprocal_rank(rank_zero if strict_valid else None) == expected
        assert utility_from_ranked_indices(
            strict_valid=strict_valid,
            ranked_candidate_indices=json.dumps(ranked),
            positive_candidate_position=20,
        )[2] == expected
        assert float(g4.loc[index, "_mrr_contribution"]) == expected
        rq3_record = {
            "strict_parse_valid": strict_valid,
            "terminal_state": "STRICT_VALID" if strict_valid else "INVALID_LENGTH",
            "ranked_item_ids": json.dumps([str(item) for item in ranked]),
        }
        assert rq3_utility(rq3_record, "20")[4] == expected


def test_manifest_freezes_mrr_and_aplt_alias_contracts() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    mrr = manifest["mrr_contract"]
    assert mrr["canonical_name"] == "mrr"
    assert mrr["inspected_ranking_depth"] == 10
    assert mrr["positive_absent_policy"] == "ZERO"
    assert mrr["terminal_invalid_policy"] == "ZERO"
    assert mrr["semantic_equivalence"]["status"] == "PASS"
    assert mrr["shared_implementation"] is False
    long_tail = manifest["long_tail_contract"]
    assert long_tail["canonical_metric"] == "APLT@5"
    assert long_tail["canonical_metric_id"] == "aplt_at_5"
    assert long_tail["implementation_alias"] == "calACLT"
    assert long_tail["aclt_is_separate_metric"] is False


def test_canonical_artifact_hashes_and_serendipity_successor() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    authority = manifest["canonical_artifact_authority"]
    for path_key, hash_key in (
        ("config_path", "config_sha256"),
        ("canonical_utility_path", "canonical_utility_sha256"),
        ("canonical_beyond_utility_path", "canonical_beyond_utility_sha256"),
        ("canonical_reliability_path", "canonical_reliability_sha256"),
        ("serendipity_correction_manifest", "serendipity_correction_manifest_sha256"),
    ):
        assert sha256(ROOT / authority[path_key]) == authority[hash_key]
    assert authority["canonical_beyond_utility_sha256"] == CURRENT_HASH
    assert authority["historical_pre_fix_sha256"] == PRE_FIX_HASH
    assert authority["raw_inference_unchanged"] is True
    correction = json.loads((ROOT / authority["serendipity_correction_manifest"]).read_text(encoding="utf-8"))
    assert correction["post_fix_artifact_hash"] == CURRENT_HASH
    assert correction["pre_fix_artifact_hash"] == PRE_FIX_HASH
    assert correction["status"] == "PASS"


def test_complete_case_is_utility_only_and_noncanonical() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    contract = manifest["complete_case_artifact"]
    frame = pd.read_csv(ROOT / contract["path"], keep_default_na=False)
    assert sha256(ROOT / contract["path"]) == contract["sha256"]
    assert set(frame["metric"]) == {"hr_at_5", "ndcg_at_5", "mrr"}
    assert "serendipity_at_5" not in set(frame["metric"])
    assert PRE_FIX_HASH not in (ROOT / contract["path"]).read_text(encoding="utf-8")
    assert contract["classification"] == "NON_CANONICAL_FOREGROUND"
    assert contract["corrected_replacement_required"] is False


def test_foreground_registries_do_not_claim_the_pre_fix_hash() -> None:
    foreground = [
        ROOT / "README.md",
        ROOT / "PROJECT_SPEC.md",
        ROOT / "results/manifests/final_submission_figure_inventory.csv",
        ROOT / "results/manifests/final_submission_deferred_inventory.json",
    ]
    foreground.extend((ROOT / "results").glob("**/*scientific_synthesis*.json"))
    foreground.extend((ROOT / "results").glob("**/*figure*manifest*.json"))
    for path in sorted(set(foreground)):
        if path.exists():
            assert PRE_FIX_HASH not in path.read_text(encoding="utf-8", errors="replace"), path
    figure_inventory = (ROOT / "results/manifests/final_submission_figure_inventory.csv").read_text(encoding="utf-8")
    relevant = [line for line in figure_inventory.splitlines() if "g1_primary_ranking_beyond_utility.csv" in line]
    assert relevant
    assert all(CURRENT_HASH in line and PRE_FIX_HASH not in line for line in relevant)
