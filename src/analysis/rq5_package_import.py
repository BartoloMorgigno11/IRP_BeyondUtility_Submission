"""Selectively archive existing RQ5 evidence; never execute experiment producers."""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess, zipfile
from pathlib import Path

NS = "results/rq5_final_scientific_freeze"
def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def run(root: Path, research: Path, submission: Path) -> None:
    selected = {}
    roots = {"research": research, "submission": submission}
    def add(alias, rel, family, role="source_evidence", authority="SUPPORTING"):
        rel = str(rel).replace("\\", "/")
        path = roots[alias] / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        selected.setdefault((alias, rel), (family, role, authority))
    freeze = json.loads((research / "results/manifests/reproduction_r5d_generation_involved_v2_freeze.json").read_text())
    add("research", "results/manifests/reproduction_r5d_generation_involved_v2_freeze.json", "r5d")
    for section in ["raw_file_hashes", "table_hashes", "source_hashes", "report_hashes", "diagnostic_hashes"]:
        for rel, digest in freeze[section].items():
            data = (research / rel).read_bytes()
            assert sha(data) == digest.lower(), (section, rel)
            add("research", rel, "r5d", section)
    families = {"corrected": "rq5_corrected_full_", "systematic": "rq5_primary_systematic_", "history_v1": "rq5_figure6_history_length_"}
    stems = {"corrected": "rq5_corrected_matched_full_recovery", "systematic": "rq5_primary_systematic_extension", "history_v1": "rq5_figure6_history_length_slice_full"}
    authorities = {"corrected": "FOREGROUND", "systematic": "SUPPORTING", "history_v1": "FORENSIC"}
    for family, prefix in families.items():
        for path in sorted((submission / "results/tables").glob(prefix + "*.csv")):
            add("submission", path.relative_to(submission), family, "table", authorities[family])
        stem = stems[family]
        manifest = "rq5_corrected_matched_recovery_full" if family == "corrected" else stem
        for rel in [f"configs/{stem}.yaml", f"src/analysis/{stem}.py", f"report/{manifest}.md", f"results/manifests/{manifest}.json"]:
            add("submission", rel, family, "historical_producer_or_contract", authorities[family])
        raw_ns = "rq5_corrected_matched_recovery_full" if family == "corrected" else stem
        for path in sorted((submission / "results/llm" / raw_ns).rglob("*.json")):
            add("submission", path.relative_to(submission), family, "raw_evidence", authorities[family])
    for rel in ["configs/rq5_corrected_matched_recovery.yaml", "src/analysis/rq5_corrected_matched_recovery.py", "tests/test_rq5_corrected_matched_recovery.py", "tests/test_rq5_corrected_matched_full_recovery.py", "results/manifests/rq5_corrected_matched_recovery_pilot_v2.json", "report/rq5_corrected_matched_recovery_pilot_v2.md"]:
        add("submission", rel, "pilot_v2", "reuse_dependency", "SUPPORTING")
    for path in sorted((submission / "results/tables").glob("rq5_corrected_pilot_v2_*.csv")):
        add("submission", path.relative_to(submission), "pilot_v2", "reuse_dependency")
    # Include only pilot records actually named by corrected full reuse evidence.
    for path in sorted((submission / "results/llm/rq5_corrected_matched_recovery_full").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("reuse_source_file"):
            add("submission", payload["reuse_source_file"], "pilot_v2", "reused_raw")
    # F6L profiles and rendered ranking inputs are upstream provenance, never F6L statistics as foreground.
    for rel in ["results/manifests/phase_f6l_profile_execution_manifest.json", "src/analysis/phase_f6l_profile_execution.py"]:
        add("research", rel, "f6l", "upstream_contract", "FORENSIC")
    for path in sorted((research / "results/tables").glob("phase_f6l_profile_*.csv")):
        add("research", path.relative_to(research), "f6l", "upstream_input", "FORENSIC")
    for path in sorted((research / "results/llm/phase_f6l_profile_execution_20260801_010000").glob("*.json")):
        add("research", path.relative_to(research), "f6l", "upstream_raw", "FORENSIC")
    forensic = ["rq5_profile_statistical_paired_contributions", "rq5_profile_statistical_results", "rq5_profile_statistical_validation", "rq5_profile_strict_validity_sensitivity", "rq5_aplt_calculability_audit", "rq5_serendipity_calculability_audit", "rq5_paper_figure6_metric_coverage"]
    for name in forensic:
        add("submission", f"results/tables/{name}.csv", "f6l", "historical_statistics", "FORENSIC")
    for rel in ["src/analysis/rq5_profile_scientific_audit.py", "tests/test_rq5_profile_scientific_audit.py", "report/rq5_profile_statistical_analysis.md"]:
        add("submission", rel, "f6l", "historical_analysis", "FORENSIC")
    for name in ["fig_rq5_profile_example_qualitative_values", "fig_rq5_profile_plus_history_extension_values"]:
        add("submission", f"results/tables/{name}.csv", "figure_inputs", "illustration_source")
    for rel in ["data/reconstructed/plan_c/amazon_beauty/interactions.csv", "data/reconstructed/plan_c/amazon_beauty/catalog.csv"]:
        add("research", rel, "popularity", "canonical_data")
    out = root / NS
    out.mkdir(parents=True, exist_ok=True)
    states = {}
    for alias, source in roots.items():
        cmd = ["git", "-c", "safe.directory=" + source.as_posix(), "--no-optional-locks", "-C", str(source)]
        tracked = set(subprocess.check_output(cmd + ["ls-files"], text=True).splitlines())
        untracked = set(subprocess.check_output(cmd + ["ls-files", "--others", "--exclude-standard"], text=True).splitlines())
        states[alias] = (tracked, untracked)
    inventory = []
    for family in sorted({v[0] for v in selected.values()}):
        archive = out / f"evidence_{family}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for (alias, rel), (fam, role, authority) in sorted(selected.items()):
                if fam != family: continue
                data = (roots[alias] / rel).read_bytes()
                member = alias + "/" + rel
                info = zipfile.ZipInfo(member, date_time=(2026, 9, 20, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info, data, compresslevel=9)
                tracked, untracked = states[alias]
                inventory.append({"path": NS + "/" + archive.name + "!" + member, "evidence_family": fam, "role": role, "source_worktree": "master" if alias == "research" else "submission/prof", "tracked_status_before_import": "tracked" if rel in tracked else "untracked" if rel in untracked else "ignored", "sha256": sha(data), "size": len(data), "producer": stems.get(fam, "historical source producer archived with family"), "input_dependencies": "family request manifests and raw dependencies included in these archives", "authority": authority, "required_for_external_reproduction": role != "historical_statistics"})
        print(family, archive.stat().st_size, flush=True)
    prefixes = {"corrected": "rq5_corrected_full", "systematic": "rq5_primary_systematic", "history_v1": "rq5_figure6_history_length"}
    for row in inventory:
        fam = row["evidence_family"]
        if fam in prefixes:
            suffix = {"corrected":"request_universe", "systematic":"ranking_universe", "history_v1":"ranking_manifest"}[fam]
            deps = ["submission/results/tables/" + prefixes[fam] + "_" + suffix + ".csv", "submission/results/tables/" + prefixes[fam] + "_" + ("profile_reuse_audit" if fam=="corrected" else "profile_manifest") + ".csv"]
        elif fam=="r5d": deps=["research/results/manifests/reproduction_r5d_generation_involved_v2_freeze.json"]
        elif fam=="f6l": deps=["research/results/manifests/phase_f6l_profile_execution_manifest.json"]
        elif fam=="pilot_v2": deps=["submission/results/manifests/rq5_corrected_matched_recovery_pilot_v2.json"]
        elif fam=="popularity": deps=["configs/rq5_final_freeze.yaml"]
        else: deps=["research/results/llm/phase_f6l_profile_execution_20260801_010000/F6L-PROFILE-REQ-0415.json", "research/results/tables/reproduction_r5d_generation_involved_v2_metrics_by_condition.csv"]
        row["input_dependencies"]=json.dumps(deps,separators=(",", ":"))
        if row["producer"].startswith("rq5_"): row["producer"]="src/analysis/"+row["producer"]+".py"
    with (out / "package_inventory.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(inventory[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(inventory)
    print("Imported", len(inventory), "selected evidence members")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", required=True, type=Path)
    parser.add_argument("--submission-root", required=True, type=Path)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[2], args.research_root, args.submission_root)
