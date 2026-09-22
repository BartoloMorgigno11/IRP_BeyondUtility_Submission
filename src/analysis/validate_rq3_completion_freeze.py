"""Read-only verification of the versioned RQ3 completion package; no inference."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
from pathlib import Path
import sys
import tarfile
from typing import Any

import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.analysis import rq3_completion_scientific_freeze as producer


def require(condition: bool, message: str) -> None:
    """Fail closed on an integrity or scientific-contract mismatch."""
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare_frame(actual: pd.DataFrame, path: Path) -> None:
    expected = pd.read_csv(path, keep_default_na=False)
    assert_frame_equal(actual.reset_index(drop=True), expected, check_dtype=False,
                       check_exact=False, rtol=0, atol=1e-12)


def validate(root: Path, config_path: Path) -> dict[str, Any]:
    """Verify bytes, terminal accounting, grid, pairs and all frozen bootstrap CIs."""
    config = producer.configure(root, config_path)
    directory = root / producer.OUTPUT_DIR
    manifest = json.loads((directory / "freeze_manifest.json").read_text())
    original = json.loads((directory / "source_freeze_manifest.json").read_text())
    packaging = json.loads((directory / "evidence_package_manifest.json").read_text())
    require("git_commit" not in manifest, "Ambiguous containing-commit claim")
    require(manifest["base_commit"] == original["git_commit"], "Base checkout mismatch")
    require(sha(directory / "source_freeze_manifest.json") == packaging["source_manifest_sha256"], "Source manifest hash")
    for path, expected in manifest["outputs"].items():
        require(sha(root / path) == expected, f"Output hash: {path}")
    require(sha(root / producer.FINAL_GRID) == manifest["metric_grid_artifact"]["sha256"], "Grid hash")
    for path, check in manifest["old_g2_hash_checks"].items():
        require(sha(root / path) == check["expected"] == check["actual"], f"Old G2 hash: {path}")
    require(all(c["match"] for c in producer.old_g2_hash_checks(root).values()), "Old G2 inherited freeze hashes")
    request = packaging["request_manifest"]
    require(sha(root / request["packaged_path"]) == request["packaged_sha256"], "Request package hash")
    with gzip.open(root / request["packaged_path"], "rb") as stream:
        request_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    require(request_hash == request["original_sha256"] == original["source_terminal_evidence"]["request_manifest_sha256"], "Request original bytes")
    terminal = packaging["terminal_evidence"]
    require(sha(root / terminal["packaged_path"]) == terminal["packaged_sha256"], "Terminal package hash")
    require(sha(root / terminal["member_inventory"]) == terminal["member_inventory_sha256"], "Member inventory hash")
    with (root / terminal["member_inventory"]).open(newline="", encoding="utf-8") as stream:
        inventory = list(csv.DictReader(stream))
    members = {r["member"]: r for r in inventory}
    require(len(members) == len(inventory) == terminal["members"], "Duplicate member inventory")
    requests = pd.read_csv(root / producer.REQUEST_MANIFEST, dtype=str, keep_default_na=False)
    require(not requests["request_key"].duplicated().any(), "Duplicate request keys")
    lookup = requests.set_index("request_key").to_dict("index")
    seen = set()
    with tarfile.open(root / terminal["packaged_path"], "r:gz") as archive:
        for member in archive:
            require(member.isfile() and member.name in members and member.name not in seen, "Unexpected archive member")
            seen.add(member.name)
            data = archive.extractfile(member).read()
            require(hashlib.sha256(data).hexdigest() == members[member.name]["sha256"], "Terminal original byte hash")
            require(len(data) == int(members[member.name]["size_bytes"]), "Terminal byte size")
            record = json.loads(data)
            saved_hash = record.pop("terminal_record_hash")
            payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            require(hashlib.sha256(payload.encode()).hexdigest() == saved_hash, "Terminal record content hash")
            key = member.name.removesuffix(".json")
            require(key in lookup, "Terminal absent from requests")
            row = lookup[key]
            for field in ("dataset", "user_id", "model", "prompt_strategy", "history_length", "candidate_hash", "history_hash"):
                require(str(record[field]) == str(row[field]), f"Terminal/request mismatch: {field}")
            require(isinstance(record["strict_parse_valid"], bool), "Serialized boolean instead of boolean")
    require(seen == set(members) == {key + ".json" for key in lookup}, "Terminal/request coverage")
    logging.info("All packaged bytes and %d terminal records verified", len(seen))
    completion, accounting, digest = producer.load_completion(root)
    require(digest == manifest["source_terminal_evidence"]["terminal_set_sha256"], "Terminal set hash")
    totals = {k: int(accounting[k].sum()) for k in config["experiment"]["expected_totals"]}
    require(totals == config["experiment"]["expected_totals"] == manifest["request_accounting"]["totals"], "Accounting totals")
    compare_frame(accounting, directory / "request_accounting.csv")
    cohort = pd.read_csv(root / producer.COHORT, dtype=str, keep_default_na=False)
    metadata = cohort.set_index(["dataset", "user_id"]).to_dict("index")
    for row in completion.to_dict("records"):
        meta = metadata[(row["dataset"], row["user_id"])]
        for field in ("candidate_set_id", "candidate_hash", "positive_item_id"):
            require(row[field] == meta[field], f"Candidate contract: {field}")
        require(json.loads(row["candidate_items"]) == json.loads(meta["candidate_items"]), "Candidate ordering")
        eligible = "b1_primary_ge10_eligible" if row["history_length"] in (3, 4) else "secondary_full_curve_ge20_eligible"
        require(meta[eligible].lower() == "true", "Cohort membership")
    endpoints = producer.old_endpoint_frames(root, cohort)
    for history in config["experiment"]["expected_by_history"]:
        endpoints[int(history)] = completion.loc[completion["history_length"].eq(int(history))].copy()
    grid_ok, failures = producer.verify_metric_grid(root, endpoints)
    require(grid_ok, str(failures))
    grid = pd.read_csv(root / producer.FINAL_GRID, keep_default_na=False)
    cells = grid.loc[grid["row_type"].eq("metric")].copy()
    cells["history_length"] = cells["history_length"].astype(int)
    keys = ["analysis_membership", "dataset", "model", "prompt_strategy", "history_length", "metric_id"]
    require(not cells.duplicated(keys).any(), "Duplicate metric grid cells")
    old = pd.read_csv(root / producer.CANONICAL_GRID, keep_default_na=False).set_index(keys)
    for row in cells.to_dict("records"):
        history = row["history_length"]
        if history not in config["experiment"]["expected_by_history"]:
            frozen = old.loc[tuple(row[k] for k in keys)]
            require(math.isclose(float(row["value"]), float(frozen["value"]), abs_tol=1e-12, rel_tol=0), "Old grid value changed")
        else:
            subset = endpoints[history]
            subset = subset.loc[subset["dataset"].eq(row["dataset"]) & subset["model"].eq(row["model"]) & subset["prompt_strategy"].eq(row["prompt_strategy"])]
            if row["analysis_membership"] == "SECONDARY_FULL_CURVE_GE20":
                subset = subset.loc[[metadata[(r.dataset, r.user_id)]["secondary_full_curve_ge20_eligible"].lower() == "true" for r in subset.itertuples()]]
            require(len(subset) == int(row["planned_applicable_users"]), "Grid denominator")
            require(int(subset["strict_parse_valid"].sum()) == int(row["strict_valid_users"]), "Grid strict-valid count")
            require(math.isclose(float(subset[row["metric_id"]].mean()), float(row["value"]), abs_tol=1e-12, rel_tol=0), "Completion grid value")
    grids = {name: sorted(cells.loc[cells["cohort"].eq(name), "history_length"].unique().tolist()) for name in ("GE10", "GE20")}
    require(grids == config["experiment"]["history_grids"] == manifest["history_grids"], "History grids")
    pairs, pairability = producer.build_pairs(endpoints)
    require(pairability["status"].eq("PASS").all(), "Pairability failure")
    compare_frame(pairability, directory / "pairability_validation.csv")
    stats = producer.statistical_results(pairs)
    compare_frame(stats, directory / "statistical_results.csv")
    summary = producer.synthesis(root, stats)
    compare_frame(summary, directory / "scientific_synthesis_summary.csv")
    return {"status": "PASS", "base_commit": manifest["base_commit"], "totals": totals,
            "old_g2_unchanged": "PASS", "history_grids": grids,
            "terminal_original_byte_hashes_verified": len(seen), "paired_bootstrap_applied": True,
            "bootstrap_resamples": producer.BOOTSTRAP_RESAMPLES, "statistical_rows_verified": len(stats),
            "all_frozen_CIs_recomputed_and_matched": True, "llm_calls": 0, "result_files_rewritten": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rq3_completion_freeze.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(validate(ROOT, args.config), indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
