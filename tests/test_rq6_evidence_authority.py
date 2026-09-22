from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "results/manifests/rq6_evidence_authority.json"
CONTRACT = ROOT / "results/manifests/rq6_scientific_contract.json"
FIGURE_SPEC = ROOT / "results/manifests/rq6_final_figure_spec.json"

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def test_authority_classes_and_hashes() -> None:
    manifest = load_json(AUTHORITY)
    assert manifest["direct_textual_hallucination_reproduced"] is False
    assert manifest["local_canonical_evidence"] == "STRUCTURAL_RELIABILITY"
    assert manifest["direct_comparability_with_paper"] is False
    classes = manifest["authority_classes"]
    canonical = {entry["path"] for entry in classes["CANONICAL"]}
    assert {"results/tables/g1_primary_ranking_reliability.csv","results/tables/g2_history_length_analysis_universe.csv","results/tables/g3_candidate_position_reliability.csv","results/tables/g4_f1bzq_reliability.csv"} <= canonical
    for entries in classes.values():
        for entry in entries:
            path = ROOT / entry["path"]
            assert path.is_file()
            assert sha256(path) == entry["sha256"]
    forensic = {entry["path"]: entry for entry in classes["FORENSIC_ONLY"]}
    for path in ("results/tables/reproduction_r3_hallucination_audit.csv","results/tables/reproduction_r4_hallucination_reconciliation.csv"):
        assert forensic[path]["authoritative_quantitative_use"] is False
    r3 = forensic["results/tables/reproduction_r3_hallucination_audit.csv"]
    assert r3["observed_interpreted_rate_gt_1_count"] == 16
    assert r3["observed_max_interpreted_rate"] == 2189.0
    assert all(not source["present_in_release_candidate"] for source in r3["declared_sources"])
    observed_rates = [float(row["value"]) for row in rows(r3["path"]) if row["metric"].endswith("rate") and row["value"]]
    assert sum(value > 1 for value in observed_rates) == r3["observed_interpreted_rate_gt_1_count"]
    assert max(observed_rates) == r3["observed_max_interpreted_rate"]

def test_r1_supporting_reference_and_portable_provenance() -> None:
    manifest = load_json(AUTHORITY)
    r1 = manifest["authority_classes"]["SUPPORTING_UPSTREAM_REFERENCE"][0]
    assert r1["locally_reproduced"] is False
    assert r1["quantitative_foreground_authority"] is False
    table_rows = rows(r1["path"])
    projection = [{key: row[key] for key in ("dataset","task","model","condition","history_length","profile_generator","candidate_condition","metric","value")} for row in table_rows]
    payload = json.dumps(projection,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    assert len(table_rows) == r1["rows"] == 105
    assert hashlib.sha256(payload).hexdigest() == r1["numeric_projection_sha256"]
    portability = manifest["upstream_reference_portability"]
    assert portability["historical_source_archive_fields"]["operational_dependency"] is False
    for archive in portability["portable_provenance"]["archives"]:
        assert not PureWindowsPath(archive["expected_upstream_repository_relative_path"]).is_absolute()
        assert len(archive["sha256"]) == 64 and archive["size_bytes"] > 0

def test_canonical_reliability_accounting() -> None:
    expected = load_json(CONTRACT)["canonical_strict_valid_accounting"]
    g1 = rows(expected["G1"]["source"])
    observed = {"planned":sum(int(r["planned_N"]) for r in g1),"strict_valid":sum(int(r["strict_valid_N"]) for r in g1),"invalid":sum(int(r["structurally_invalid_N"]) for r in g1),"invalid_length_requests":sum(int(r["length_violation_count"]) for r in g1),"duplicate_index_requests":sum(json.loads(r["parse_status_counts"]).get("duplicate_indices",0) for r in g1),"unknown_index_requests":sum(int(r["hallucination_out_of_candidate_count"]) for r in g1),"backend_failures":sum(int(r["backend_failure_N"]) for r in g1)}
    assert all(expected["G1"][k] == v for k,v in observed.items())
    g2 = rows(expected["G2"]["source"]); status = Counter(r["parse_status"] for r in g2)
    observed = {"planned":len(g2),"strict_valid":sum(r["strict_parse_valid"].lower()=="true" for r in g2),"invalid":sum(r["strict_parse_valid"].lower()!="true" for r in g2),"invalid_length_requests":status["invalid_length"],"duplicate_index_requests":status["duplicate_indices"],"unknown_index_requests":status["unknown_indices"],"unknown_values":sum(int(r["hallucination_count"] or 0) for r in g2),"backend_failures":sum(r["backend_success"].lower()!="true" for r in g2)}
    assert all(expected["G2"][k] == v for k,v in observed.items())
    g3 = next(r for r in rows(expected["G3"]["source"]) if r["scope"]=="overall")
    observed = {"planned":int(g3["planned_requests"]),"strict_valid":int(g3["strict_valid"]),"invalid":int(g3["invalid"]),"invalid_length_requests":int(g3["invalid_length"]),"duplicate_index_requests":int(g3["invalid_duplicate"]),"out_of_candidate_requests":int(g3["hallucination_or_out_of_candidate"]),"backend_failures":int(g3["backend_errors"])}
    assert all(expected["G3"][k] == v for k,v in observed.items())
    g4 = rows(expected["G4"]["source"])
    observed = {"planned":sum(int(r["planned_requests"]) for r in g4),"strict_valid":sum(int(r["strict_valid"]) for r in g4),"invalid":sum(int(r["structural_invalid"]) for r in g4),"invalid_length_requests":sum(int(r["invalid_length"]) for r in g4),"duplicate_index_requests":sum(int(r["invalid_duplicate"]) for r in g4),"unknown_index_requests":sum(int(r["hallucination_or_unknown_index"]) for r in g4),"backend_failures":sum(int(r["backend_errors"]) for r in g4)}
    assert all(expected["G4"][k] == v for k,v in observed.items())

def test_rq5_views_remain_separate() -> None:
    rq5 = load_json(CONTRACT)["rq5_views"]
    assert rq5["overlap_policy"] == "SEPARATE_OVERLAPPING_VIEWS_DO_NOT_SUM"
    for view in rq5["views"]:
        data = rows(view["source"])
        valid = sum(r["strict_valid"].lower() in {"1","true"} for r in data)
        assert (len(data),valid,len(data)-valid) == (view["planned"],view["strict_valid"],view["invalid"])

def test_contract_and_figure_separate_constructs() -> None:
    contract = load_json(CONTRACT)
    assert contract["target_paper_measure"]["canonical_name"] == "free-text textual/catalogue hallucination"
    assert contract["local_canonical_measure"]["canonical_name"] == "constrained-output structural reliability"
    assert contract["direct_comparability"] is False
    assert all(not item["textual_hallucinated_title"] for item in contract["local_canonical_measure"]["failure_types"])
    figure = load_json(FIGURE_SPEC)
    assert figure["status"] == "SPEC_ONLY_NO_RENDER"
    assert figure["rendering"]["generated"] is False
    numeric = set(figure["panels"][1]["numeric_sources"] + figure["panels"][1]["optional_separate_rq5_views"])
    assert numeric.isdisjoint(figure["forbidden_quantitative_sources"])
    assert figure["panels"][0]["quantitative_foreground"] is False

def test_no_foreground_r3_r4_dependency() -> None:
    authority = load_json(AUTHORITY)
    forbidden = {Path(path).name for path in authority["foreground_policy"]["forbidden_quantitative_foreground_sources"]}
    for relative in authority["foreground_policy"]["foreground_registry_paths"]:
        text = (ROOT / relative).read_text(encoding="utf-8-sig")
        assert all(name not in text for name in forbidden)
