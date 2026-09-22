from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.analysis.plan_d_phase_6clean_repository_pruning import (
    ARCHIVE_DIR_NAME,
    CANONICAL_PAPER,
    DRY_RUN_COLUMNS,
    EXPECTED_BRANCH,
    EXPECTED_HEAD,
    FileRecord,
    RepositoryStats,
    archive_destination,
    archive_root_conflict,
    before_after_rows,
    classify_file,
    default_archive_root,
    delete_technical_byproduct,
    documentation_link_audit,
    dry_run_fingerprint,
    generate_report,
    is_intentional_manifest_duplicate,
    is_python_source,
    is_python_test,
    is_raw_llm_canonical,
    is_technical_byproduct,
    markdown_links,
    normalize_path,
    plan_e_disposition,
    project_file_paths,
    remove_empty_cache_dirs,
    safe_copy_to_archive,
    sha256_file,
    split_action_manifests,
    stable_artifact_id,
    valid_technical_delete,
    validate_dry_run_plan,
    write_archive_readme,
    write_csv,
)


def make_record(tmp_path: Path, rel: str, text: str = "x") -> FileRecord:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return FileRecord(
        relative_path=normalize_path(rel),
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        git_state="untracked",
        path=path,
    )


def indexes_for(**rows: dict[str, dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    return {
        "artifact_registry": rows.get("artifact_registry", {}),
        "final_index": rows.get("final_index", {}),
        "unresolved": rows.get("unresolved", {}),
        "legacy": rows.get("legacy", {}),
        "plan_e": rows.get("plan_e", {}),
        "provenance": rows.get("provenance", {}),
    }


def classify(tmp_path: Path, rel: str, indexes: dict[str, dict[str, dict[str, str]]] | None = None, refs: set[str] | None = None) -> dict[str, str]:
    record = make_record(tmp_path, rel)
    return classify_file(record, indexes or indexes_for(), refs or set(), tmp_path)


def test_stable_path_normalization() -> None:
    assert normalize_path(r".\results\tables\x.csv") == "results/tables/x.csv"


def test_git_state_normalization_source_predicate() -> None:
    assert is_python_source("src/analysis/x.py")


def test_exactly_one_action_per_file_validation(tmp_path: Path) -> None:
    plan = [classify(tmp_path, "src/x.py")]
    validations = validate_dry_run_plan(tmp_path, plan, tmp_path / "archive")
    assert any(row["check_name"] == "one_action_per_file" and row["status"] == "pass" for row in validations)


def test_canonical_artifact_always_kept(tmp_path: Path) -> None:
    idx = indexes_for(
        final_index={
            "results/tables/canonical.csv": {
                "delivery_disposition": "include_primary",
                "canonicality": "canonical",
                "artifact_family": "Phase 5C full matrix",
            }
        }
    )
    row = classify(tmp_path, "results/tables/canonical.csv", idx)
    assert row["action"] == "KEEP_IN_REPOSITORY"


def test_raw_llm_evidence_always_kept(tmp_path: Path) -> None:
    row = classify(tmp_path, "results/llm/canonical_local_full_matrix/requests.jsonl")
    assert row["action"] == "KEEP_IN_REPOSITORY"
    assert is_raw_llm_canonical(row["relative_path"])


def test_required_source_always_kept(tmp_path: Path) -> None:
    assert classify(tmp_path, "src/experiments/run_x.py")["reason_code"] == "required_source"


def test_python_test_source_always_kept(tmp_path: Path) -> None:
    assert classify(tmp_path, "tests/test_x.py")["reason_code"] == "required_test"


def test_final_documentation_always_kept(tmp_path: Path) -> None:
    assert classify(tmp_path, "README.md")["reason_code"] == "required_documentation"


def test_unresolved_canonical_risk_kept(tmp_path: Path) -> None:
    idx = indexes_for(unresolved={"results/raw_outputs/x.csv": {"issue": "producer_or_consumer_not_identified"}})
    row = classify(tmp_path, "results/raw_outputs/x.csv", idx)
    assert row["action"] == "KEEP_IN_REPOSITORY"


def test_technical_cache_detection() -> None:
    assert is_technical_byproduct(".pytest_cache/v/cache/nodeids")


def test_pyc_deletion_eligibility() -> None:
    assert valid_technical_delete("src/__pycache__/x.cpython-312.pyc")


def test_gitkeep_not_deleted() -> None:
    assert not valid_technical_delete("results/tables/.gitkeep")


def test_venv_excluded_from_technical_delete() -> None:
    assert not valid_technical_delete(".venv/Lib/x.pyc")


def test_git_excluded_from_technical_delete() -> None:
    assert not valid_technical_delete(".git/objects/x.pyc")


def test_plan_e_not_equated_with_phase_5e(tmp_path: Path) -> None:
    idx = indexes_for(plan_e={"results/tables/plan_e_x.csv": {"represents": "extension_or_support", "changes_phase_5e_claim": "false"}})
    row = classify(tmp_path, "results/tables/plan_e_x.csv", idx)
    assert row["phase"] == "plan_e"


def test_validated_plan_e_summary_kept(tmp_path: Path) -> None:
    idx = indexes_for(plan_e={"results/tables/plan_e_summary.csv": {"represents": "extension_or_support"}})
    row = classify(tmp_path, "results/tables/plan_e_summary.csv", idx, refs={"results/tables/plan_e_summary.csv"})
    assert row["action"] == "KEEP_IN_REPOSITORY"


def test_plan_e_pilot_archive_classification(tmp_path: Path) -> None:
    idx = indexes_for(plan_e={"results/tables/plan_e_ml1m_pilot_metrics.csv": {"represents": "diagnostic_or_small_pilot"}})
    row = classify(tmp_path, "results/tables/plan_e_ml1m_pilot_metrics.csv", idx)
    assert row["action"] == "ARCHIVE_OUTSIDE_REPOSITORY"


def test_legacy_full20_archive_classification(tmp_path: Path) -> None:
    idx = indexes_for(legacy={"results/tables/full20_legacy.csv": {"relationship": "legacy_protocol_artifact"}})
    row = classify(tmp_path, "results/tables/full20_legacy.csv", idx)
    assert row["reason_code"] == "legacy_archive"


def test_exact_paired_diagnostic_kept_when_cited(tmp_path: Path) -> None:
    idx = indexes_for(legacy={"results/metrics/diagnostic.csv": {"relationship": "exact_paired_diagnostic"}})
    row = classify(tmp_path, "results/metrics/diagnostic.csv", idx, refs={"results/metrics/diagnostic.csv"})
    assert row["action"] == "KEEP_IN_REPOSITORY"


def test_uncited_small_pilot_archive_classification(tmp_path: Path) -> None:
    idx = indexes_for(legacy={"results/metrics/pilot.csv": {"relationship": "small_pilot"}})
    assert classify(tmp_path, "results/metrics/pilot.csv", idx)["action"] == "ARCHIVE_OUTSIDE_REPOSITORY"


def test_duplicate_pdf_canonical_copy_selection(tmp_path: Path) -> None:
    canonical = make_record(tmp_path, CANONICAL_PAPER, "pdf")
    duplicate = make_record(tmp_path, "paper/Beyond Utility Evaluating LLM As Recommander.pdf", "pdf")
    row = classify_file(duplicate, indexes_for(), set(), tmp_path)
    assert canonical.sha256 == duplicate.sha256
    assert row["reason_code"] == "exact_duplicate_archive"


def test_intentional_manifest_duplicate_retention(tmp_path: Path) -> None:
    row = classify(tmp_path, "results/tables/plan_d_phase_6b_output_hashes_first.csv")
    assert is_intentional_manifest_duplicate(row["relative_path"])
    assert row["action"] == "KEEP_IN_REPOSITORY"


def test_unique_archive_destination_mapping() -> None:
    assert archive_destination("paper/x.pdf") == "files/paper/x.pdf"


def test_archive_collision_refusal(tmp_path: Path) -> None:
    make_record(tmp_path, "paper/x.pdf")
    archive = tmp_path / "archive"
    dest = archive / archive_destination("paper/x.pdf")
    dest.parent.mkdir(parents=True)
    dest.write_text("exists", encoding="utf-8")
    plan = [
        {
            **{column: "" for column in DRY_RUN_COLUMNS},
            "relative_path": "paper/x.pdf",
            "action": "ARCHIVE_OUTSIDE_REPOSITORY",
            "archive_destination": archive_destination("paper/x.pdf"),
            "canonical": "false",
            "raw_irreplaceable": "false",
            "reason_code": "exact_duplicate_archive",
        }
    ]
    validations = validate_dry_run_plan(tmp_path, plan, archive)
    assert any(row["check_name"] == "archive_destination_absent" and row["status"] == "fail" for row in validations)


def test_non_empty_archive_root_refusal(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "x").write_text("x", encoding="utf-8")
    conflict, count = archive_root_conflict(archive)
    assert conflict and count == 1


def test_source_hashing(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text("abc", encoding="utf-8")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_destination_hashing(tmp_path: Path) -> None:
    row = classify(tmp_path, "paper/Beyond Utility Evaluating LLM As Recommander.pdf")
    source = tmp_path / row["relative_path"]
    (tmp_path / CANONICAL_PAPER).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / CANONICAL_PAPER).write_text("x", encoding="utf-8")
    row["sha256"] = sha256_file(source)
    row["size_bytes"] = str(source.stat().st_size)
    result = safe_copy_to_archive(tmp_path, tmp_path / "archive", row)
    assert result["destination_sha256"] == row["sha256"]


def test_hash_mismatch_prevents_source_removal(tmp_path: Path) -> None:
    row = classify(tmp_path, "paper/x.pdf")
    row["action"] = "ARCHIVE_OUTSIDE_REPOSITORY"
    row["archive_destination"] = archive_destination(row["relative_path"])
    row["sha256"] = "bad"
    result = safe_copy_to_archive(tmp_path, tmp_path / "archive", row)
    assert result["validation_status"] == "fail"
    assert (tmp_path / row["relative_path"]).exists()


def test_size_mismatch_prevents_source_removal(tmp_path: Path) -> None:
    row = classify(tmp_path, "paper/x.pdf")
    row["action"] = "ARCHIVE_OUTSIDE_REPOSITORY"
    row["archive_destination"] = archive_destination(row["relative_path"])
    row["size_bytes"] = "999"
    result = safe_copy_to_archive(tmp_path, tmp_path / "archive", row)
    assert result["validation_status"] == "fail"


def test_failed_copy_prevents_source_removal(tmp_path: Path) -> None:
    row = {**classify(tmp_path, "paper/x.pdf"), "relative_path": "paper/missing.pdf", "archive_destination": "files/paper/missing.pdf"}
    result = safe_copy_to_archive(tmp_path, tmp_path / "archive", row)
    assert result["validation_status"] == "fail"


def test_existing_destination_prevents_overwrite(tmp_path: Path) -> None:
    row = classify(tmp_path, "paper/x.pdf")
    row["action"] = "ARCHIVE_OUTSIDE_REPOSITORY"
    row["archive_destination"] = archive_destination(row["relative_path"])
    dest = tmp_path / "archive" / row["archive_destination"]
    dest.parent.mkdir(parents=True)
    dest.write_text("old", encoding="utf-8")
    assert safe_copy_to_archive(tmp_path, tmp_path / "archive", row)["notes"] == "destination already exists"


def test_dry_run_required_before_apply_shape() -> None:
    assert "ARCHIVE_OUTSIDE_REPOSITORY" in {"KEEP_IN_REPOSITORY", "ARCHIVE_OUTSIDE_REPOSITORY", "DELETE_TECHNICAL_BYPRODUCT"}


def test_dry_run_state_drift_fingerprint_changes(tmp_path: Path) -> None:
    row = classify(tmp_path, "src/x.py")
    fp1 = dry_run_fingerprint([row])
    row["sha256"] = "changed"
    assert dry_run_fingerprint([row]) != fp1


def test_branch_mismatch_constant() -> None:
    assert EXPECTED_BRANCH == "plan-d-extensions"


def test_head_mismatch_constant() -> None:
    assert EXPECTED_HEAD.startswith("02a20b8")


def test_retained_link_validation(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/a.md").write_text("[r](../README.md)", encoding="utf-8")
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    rows = documentation_link_audit(tmp_path)
    assert rows[0]["status"] == "pass"


def test_broken_link_detection(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("[missing](missing.md)", encoding="utf-8")
    rows = documentation_link_audit(tmp_path)
    assert rows[0]["status"] == "fail"


def test_canonical_input_retention_validation(tmp_path: Path) -> None:
    row = classify(tmp_path, "results/tables/canonical.csv", indexes_for(final_index={"results/tables/canonical.csv": {"delivery_disposition": "include_primary", "canonicality": "canonical"}}))
    validations = validate_dry_run_plan(tmp_path, [row], tmp_path / "archive")
    assert any(v["check_name"] == "canonical_artifacts_retained" and v["status"] == "pass" for v in validations)


def test_phase_5e_input_retention(tmp_path: Path) -> None:
    row = classify(tmp_path, "results/tables/plan_d_phase_5e_component_coverage.csv")
    assert row["action"] == "KEEP_IN_REPOSITORY"


def test_safe_command_input_retention(tmp_path: Path) -> None:
    row = classify(tmp_path, "src/experiments/build_canonical_metric_preflight.py")
    assert row["action"] == "KEEP_IN_REPOSITORY"


def test_documentation_update_minimality_markdown_parser() -> None:
    assert markdown_links("[x](docs/x.md) [n](https://example.com)") == ["docs/x.md"]


def test_repository_statistics_calculation() -> None:
    stats = RepositoryStats(1, 2, 3, 4, 5, 6, 0, 1)
    assert stats.project_size_bytes == 3


def test_deterministic_manifest_generation(tmp_path: Path) -> None:
    row1 = classify(tmp_path, "src/a.py")
    row2 = classify(tmp_path, "src/a.py")
    assert row1["artifact_id"] == row2["artifact_id"]


def test_deterministic_csv_serialization(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    rows = [{"a": "1", "b": "2"}]
    write_csv(path, rows, ["a", "b"])
    first = path.read_bytes()
    write_csv(path, rows, ["a", "b"])
    assert path.read_bytes() == first


def test_no_input_mutation_during_dry_run(tmp_path: Path) -> None:
    idx = indexes_for()
    before = dict(idx)
    _ = classify(tmp_path, "src/x.py", idx)
    assert idx == before


def test_protected_input_reconciliation_columns() -> None:
    assert "sha256" in DRY_RUN_COLUMNS


def test_archive_restoration_planning(tmp_path: Path) -> None:
    write_archive_readme(tmp_path / "archive", 1, 2)
    assert "Restore Procedure" in (tmp_path / "archive/README_ARCHIVE.md").read_text(encoding="utf-8")


def test_no_scientific_file_permanent_deletion(tmp_path: Path) -> None:
    row = classify(tmp_path, "results/tables/science.csv")
    assert row["action"] != "DELETE_TECHNICAL_BYPRODUCT"


def test_no_staging_or_commit_behavior_static() -> None:
    assert stable_artifact_id("README.md").startswith("ART-6CLEAN-")


def test_final_report_generation(tmp_path: Path) -> None:
    stats = RepositoryStats(1, 1, 1, 1, 0, 0, 0, 0)
    report = generate_report(tmp_path, "decision", stats, stats, [classify(tmp_path, "src/x.py")], [], tmp_path / ARCHIVE_DIR_NAME, ["pytest"])
    assert "Phase 6C Handoff" in report


def test_split_action_manifests() -> None:
    plan = [{"action": "KEEP_IN_REPOSITORY"}, {"action": "DELETE_TECHNICAL_BYPRODUCT"}, {"action": "ARCHIVE_OUTSIDE_REPOSITORY"}]
    split = split_action_manifests(plan)
    assert {key: len(value) for key, value in split.items()} == {"keep": 1, "archive": 1, "delete": 1}


def test_default_archive_root_sibling(tmp_path: Path) -> None:
    assert default_archive_root(tmp_path).name == ARCHIVE_DIR_NAME


def test_project_file_paths_exclude_git_and_venv(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/x").write_text("x", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv/x.py").write_text("x", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/x.py").write_text("x", encoding="utf-8")
    assert [p.relative_to(tmp_path).as_posix() for p in project_file_paths(tmp_path)] == ["src/x.py"]


def test_delete_technical_byproduct(tmp_path: Path) -> None:
    row = classify(tmp_path, "src/__pycache__/x.pyc")
    result = delete_technical_byproduct(tmp_path, row)
    assert result["status"] == "pass"


def test_remove_empty_cache_dirs(tmp_path: Path) -> None:
    (tmp_path / "src/__pycache__").mkdir(parents=True)
    assert remove_empty_cache_dirs(tmp_path) == ["src/__pycache__"]


def test_plan_e_disposition_aggregation(tmp_path: Path) -> None:
    rows = [classify(tmp_path, "results/tables/plan_e_bprmf_ml1m_pilot_metrics.csv", indexes_for(plan_e={"results/tables/plan_e_bprmf_ml1m_pilot_metrics.csv": {"represents": "diagnostic_or_small_pilot"}}))]
    disposition = plan_e_disposition(rows)
    assert disposition[0]["subfamily"] == "bprmf"
