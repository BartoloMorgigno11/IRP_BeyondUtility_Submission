"""Inventory deferred submission sources without copying or changing their evidence."""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[2]


def git(*args: str, root: Path = ROOT) -> bytes:
    return subprocess.check_output(["git", *args], cwd=root)


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree(reference: str) -> dict[str, str]:
    entries = {}
    for row in git("ls-tree", "-r", "-z", reference).split(b"\0"):
        if row:
            meta, path = row.split(b"\t", 1)
            entries[path.decode()] = meta.decode().split()[2]
    return entries


def rq5_path(path: str) -> bool:
    return any(key in path.lower() for key in ("rq5", "phase_f6l_profile", "reproduction_r5d"))


def family(path: str) -> str:
    if "primary_systematic" in path or "rq5_primary" in path:
        return "UNTRACKED_SYSTEMATIC_EXTENSION"
    if "history_length_slice" in path or "rq5_history_length" in path or "figure6" in path:
        return "HISTORY_LENGTH_SLICE_EXTENSION"
    if "corrected" in path:
        return "CORRECTED_MATCHED_RECOVERY"
    if "phase_f6l_profile" in path or "rq5_profile_statistical" in path or "rq5_profile_strict" in path:
        return "HISTORICAL_F6L_PROFILE_ANALYSIS"
    if "reproduction_r5d" in path:
        return "R5D_SEPARATE_FEASIBILITY_EVIDENCE"
    return "RQ5_SHARED_OR_UNRESOLVED"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Empty inventory: " + str(path))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def inventory(root: Path, config_path: Path) -> dict[str, Any]:
    """Read committed and local RQ5/figure evidence; emit metadata inventories only."""
    cfg = yaml.safe_load((root / config_path).read_text())
    source = (root / cfg["submission_worktree"]).resolve()
    reference = cfg["submission_reference"]
    files = tree(reference)
    output = root / cfg["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    status = git("status", "--porcelain=v1", "-z", "--untracked-files=all", root=source)
    states = {row[3:]: row[:2] for row in status.decode().split("\0") if row}
    selected = {p for p in files if rq5_path(p)}
    for directory in ["configs", "src", "tests", "report", "results/tables", "results/manifests", "results/plots"]:
        for path in (source / directory).rglob("*"):
            if path.is_file():
                relative = path.relative_to(source).as_posix()
                if rq5_path(relative) and "__pycache__" not in relative:
                    selected.add(relative)
    selected.add("tests/test_final_figure_and_table_strategy.py")
    selected.add("results/tables/FINAL_FIGURE_MANIFEST.csv")
    rows = []
    for path in sorted(selected):
        local = source / path
        committed = git("show", reference + ":" + path) if path in files else None
        committed_hash = hashlib.sha256(committed).hexdigest() if committed is not None else ""
        local_hash = file_hash(local) if local.is_file() else ""
        rows.append({"path":path,"scientific_family":family(path),"committed_in_submission":path in files,
                     "present_local":local.is_file(),"git_status":states.get(path,"COMMITTED_UNCHANGED" if path in files else "IGNORED_LOCAL_ONLY"),
                     "committed_sha256":committed_hash,"local_sha256":local_hash,
                     "content_disagrees":bool(path in files and (not local.is_file() or git("hash-object", "--path=" + path, path, root=source).decode().strip() != files[path])),
                     "byte_hash_disagrees":bool(committed_hash and local_hash and committed_hash != local_hash),
                     "integration_decision":cfg["integration_status"]})
    write_csv(output / "final_submission_rq5_inventory.csv",rows)
    dependencies=[]
    for path in sorted(p for p in selected if p.startswith("tests/") and p.endswith(".py")):
        versions={}
        if path in files:
            versions["COMMITTED"] = git("show",reference+":"+path).decode("utf-8-sig")
        if (source/path).is_file():
            versions["LOCAL"] = (source/path).read_text(encoding="utf-8-sig")
        for version,content in versions.items():
            parsed=ast.parse(content)
            for node in parsed.body:
                if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) or not node.name.startswith("test_"):
                    continue
                snippet=ast.get_source_segment(content,node) or ""
                if "rq5" not in (node.name+snippet).lower() and "profile" not in snippet.lower():
                    continue
                constants=sorted({n.value for n in ast.walk(node) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.endswith((".csv",".json",".yaml"))})
                references=constants or ["<indirect dependency: inspect function and module constants>"]
                for dependency in references:
                    candidate=dependency if "/" in dependency else "results/tables/"+dependency
                    dependencies.append({"test_file":path,"version":version,"test_function":node.name,
                        "artifact_reference":dependency,"candidate_path":candidate,"committed_in_submission":candidate in files,
                        "present_local":(source/candidate).is_file(),"dependency_semantics":"MAY_REQUIRE_ABSENCE" if ('not (TABLES / "'+dependency+'")') in snippet else "READ_OR_CONTRACT_REFERENCE",
                        "evidence_set":"CORRECTED_MATCHED_RECOVERY" if "corrected_full" in snippet else family(dependency)})
    write_csv(output / "final_submission_rq5_test_dependencies.csv",dependencies)
    registry_path="results/tables/FINAL_FIGURE_MANIFEST.csv"
    committed_rows=list(csv.DictReader(io.StringIO(git("show",reference+":"+registry_path).decode("utf-8-sig"))))
    local_rows=list(csv.DictReader((source/registry_path).read_text(encoding="utf-8-sig").splitlines()))
    refs={}
    for version,registry in [("committed",committed_rows),("local",local_rows)]:
        for row in registry:
            for role in ["value_csv","manifest","source_files"]:
                for path in re.split(r";\s*",row.get(role,"")):
                    if path.startswith(("results/","configs/","report/")):
                        key=(row["figure_id"],role,path)
                        refs.setdefault(key,{"committed":False,"local":False})[version]=True
    figures=[]
    for (figure_id,role,path),presence in sorted(refs.items()):
        local=source/path
        committed=git("show",reference+":"+path) if path in files else None
        committed_hash=hashlib.sha256(committed).hexdigest() if committed is not None else ""
        local_hash=file_hash(local) if local.is_file() else ""
        manifest_data={}
        if role=="manifest" and local.is_file():
            manifest_data=json.loads(local.read_text(encoding="utf-8-sig"))
        figures.append({"figure_id":figure_id,"role":role,"path":path,"expected_by_committed_registry":presence["committed"],
            "expected_by_local_registry":presence["local"],"already_committed":path in files,"present_local":local.is_file(),
            "present_local_only":local.is_file() and path not in files,"committed_sha256":committed_hash,"local_sha256":local_hash,
            "content_disagrees":bool(path in files and (not local.is_file() or git("hash-object", "--path=" + path, path, root=source).decode().strip() != files[path])),
                     "byte_hash_disagrees":bool(committed_hash and local_hash and committed_hash != local_hash),
            "local_manifest_git_head":manifest_data.get("git_head",manifest_data.get("git_commit","")),
            "verification_status":"REQUIRES_LATER_SCIENTIFIC_AND_RENDERED_ASSET_VERIFICATION","imported":False})
    write_csv(output / "final_submission_figure_inventory.csv",figures)
    compact=[row for row in figures if row["role"] in ("value_csv","manifest") and row["expected_by_committed_registry"]]
    summary={"RQ5_INTEGRATION_STATUS":cfg["integration_status"],"source_reference":reference,
        "source_head":git("rev-parse",reference).decode().strip(),"source_worktree_status_counts":{code:sum(value==code for value in states.values()) for code in sorted(set(states.values()))},
        "RQ5_COMMITTED_FILES":[r["path"] for r in rows if r["committed_in_submission"]],
        "RQ5_UNCOMMITTED_CORRECTED_MATCHED_FILES":[r["path"] for r in rows if r["scientific_family"]=="CORRECTED_MATCHED_RECOVERY" and not r["committed_in_submission"]],
        "RQ5_UNTRACKED_EXTENSION_FILES":[r["path"] for r in rows if "EXTENSION" in r["scientific_family"] and not r["committed_in_submission"]],
        "RQ5_FILES_DISAGREEING":[r["path"] for r in rows if r["content_disagrees"]],
        "FIGURE_SOURCE_FILES_EXPECTED":sorted({r["path"] for r in figures if r["expected_by_committed_registry"]}),
        "FIGURE_SOURCE_FILES_PRESENT_LOCAL_ONLY":sorted({r["path"] for r in figures if r["present_local_only"]}),
        "FIGURE_SOURCE_FILES_ALREADY_COMMITTED":sorted({r["path"] for r in figures if r["already_committed"]}),
        "FIGURE_SOURCE_FILES_REQUIRING_LATER_VERIFICATION":sorted({r["path"] for r in figures}),
        "compact_figure_files_expected":len(compact),"compact_figure_files_local_only":sum(r["present_local_only"] for r in compact),
        "compact_figure_files_already_committed":sum(r["already_committed"] for r in compact),
        "rq5_inventory_rows":len(rows),"rq5_test_dependency_rows":len(dependencies),
        "scope_note":"Inventories contain paths/hashes only. No RQ5 evidence, figure CSVs/manifests or figures were copied. External raw namespaces and indirect test dependencies still require scientific audit.",
        "registry_note":"Committed and dirty local registries are inventoried separately; local presence does not establish correspondence to committed rendered figures."}
    (output/"final_submission_deferred_inventory.json").write_bytes((json.dumps(summary,indent=2,sort_keys=True)+"\n").encode())
    return {k:summary[k] for k in ["RQ5_INTEGRATION_STATUS","rq5_inventory_rows","rq5_test_dependency_rows","compact_figure_files_expected","compact_figure_files_local_only","compact_figure_files_already_committed"]}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/final_submission_inventory.yaml"))
    args=parser.parse_args()
    print(json.dumps(inventory(ROOT,args.config),indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
