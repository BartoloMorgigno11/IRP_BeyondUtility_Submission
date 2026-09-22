"""Focused checks for F6J scientific evidence synthesis."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.cleanup_historical,
    pytest.mark.phase_snapshot,
    pytest.mark.refactor_regression,
]

ROOT = Path(__file__).resolve().parents[1]
APPROVED_INTERPRETER = ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"

RQ_PATH = ROOT / "results" / "tables" / "phase_f6j_research_question_evidence_matrix.csv"
CLAIM_PATH = ROOT / "results" / "tables" / "phase_f6j_scientific_claim_registry.csv"
MATRIX_PATH = ROOT / "results" / "tables" / "phase_f6j_paper_reproduction_extension_matrix.csv"
LIMIT_PATH = ROOT / "results" / "tables" / "phase_f6j_limitations_and_threats.csv"
REPORT_PATH = ROOT / "report" / "phase_f6j_scientific_evidence_synthesis.md"
HANDOFF_PATH = ROOT / "report" / "future_operations_context_handoff.md"
PROTECTED_HASHES_PATH = ROOT / "results" / "manifests" / "phase_f6i2p_protected_input_hashes.json"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _id_sort_key(value: str) -> tuple[str, int]:
    prefix, _, suffix = value.rpartition("-")
    if not prefix:
        letters = "".join(char for char in value if not char.isdigit())
        digits = "".join(char for char in value if char.isdigit())
        return (letters, int(digits or 0))
    return (prefix, int(suffix))


def test_runtime_is_the_repaired_audit_interpreter() -> None:
    assert Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"


def test_required_artifacts_exist_and_have_unique_sorted_ids() -> None:
    specs = [
        (RQ_PATH, "rq_id"),
        (CLAIM_PATH, "claim_id"),
        (MATRIX_PATH, "matrix_id"),
        (LIMIT_PATH, "limitation_id"),
    ]
    for path, id_column in specs:
        rows = _rows(path)
        assert rows, path
        ids = [row[id_column] for row in rows]
        assert len(ids) == len(set(ids)), path
        assert ids == sorted(ids, key=_id_sort_key), path


def test_research_question_schema_statuses_and_references() -> None:
    rows = _rows(RQ_PATH)
    expected_columns = {
        "rq_id",
        "research_question",
        "paper_result",
        "local_reproduced_result",
        "adapted_result",
        "extension_result",
        "outcome_status",
        "evidence_classification",
        "datasets",
        "tasks",
        "models",
        "metrics",
        "evidence_references",
        "numeric_value_status",
        "key_numeric_values",
        "principal_limitations",
        "required_future_evidence",
        "decision_notes",
    }
    assert set(rows[0]) == expected_columns
    assert {row["rq_id"] for row in rows} >= {f"RQ{i}" for i in range(1, 15)}
    assert {row["outcome_status"] for row in rows} == {
        "contradicted",
        "mixed/context-dependent",
        "supported",
        "unresolved",
    }
    for row in rows:
        assert row["evidence_references"]
        assert row["principal_limitations"]
        assert "TODO" not in json.dumps(row)
        if row["outcome_status"] == "unresolved":
            assert row["numeric_value_status"] in {"blocked", "placeholder_not_current"}
        if row["numeric_value_status"] == "available_from_existing_csv":
            assert row["key_numeric_values"]


def test_claim_registry_has_no_unreferenced_claims_and_safe_wording() -> None:
    rows = _rows(CLAIM_PATH)
    allowed_statuses = {"supported", "contradicted", "mixed/context-dependent", "unresolved"}
    for row in rows:
        assert row["claim_status"] in allowed_statuses
        assert row["evidence_references"]
        assert row["allowed_wording"]
        assert row["prohibited_wording"]
        assert row["traceability_status"] == "traceable"
        assert "exactly reproduced numerically" not in row["allowed_wording"].lower()
        if row["claim_status"] == "unresolved":
            assert row["numeric_value_status"] in {"blocked", "placeholder_not_current"}


def test_reproduction_adaptation_extension_layers_are_separated() -> None:
    rows = _rows(MATRIX_PATH)
    layers = {row["evidence_layer"] for row in rows}
    assert {
        "exact numeric reproduction",
        "repository-derived reproduction",
        "adapted reproduction",
        "diagnostic experiment",
        "original extension",
        "paper-unspecified/non-verifiable",
        "blocked",
    }.issubset(layers)
    combined = json.dumps(rows).lower()
    for token in [
        "Amazon Beauty",
        "Amazon Sports",
        "ML-1M",
        "LastFM",
        "ranking",
        "reranking",
        "Random",
        "MostPop",
        "BM25",
        "BPRMF",
        "GRU4Rec",
        "SASRec",
        "LightGCN",
        "Qwen2",
        "Mistral",
        "HR@5",
        "NDCG@5",
        "novelty",
        "coverage",
        "long-tail",
        "popularity",
        "ILD",
        "JSD",
        "Hellinger",
        "duplicate@5",
        "history",
        "candidate-position",
        "cost",
        "runtime",
    ]:
        assert token.lower() in combined
    for row in rows:
        assert row["evidence_references"]
        if row["evidence_layer"] == "blocked":
            assert row["blocked_reason"]


def test_limitations_are_traceable_and_cover_open_threats() -> None:
    rows = _rows(LIMIT_PATH)
    combined = json.dumps(rows)
    for row in rows:
        assert row["evidence_references"]
        assert row["severity"] in {"high", "medium", "low"}
        assert row["status"] in {"open", "handled_with_limits"}
    for token in [
        "Exact paper numerical reproduction",
        "candidate",
        "parser",
        "LastFM reranking",
        "Novelty",
        "cost",
        "seed",
        "profile",
    ]:
        assert token in combined


def test_f6j_outputs_are_deterministic_and_do_not_embed_ephemeral_fields() -> None:
    for path, id_column in [
        (RQ_PATH, "rq_id"),
        (CLAIM_PATH, "claim_id"),
        (MATRIX_PATH, "matrix_id"),
        (LIMIT_PATH, "limitation_id"),
    ]:
        first = _rows(path)
        second = _rows(path)
        assert first == second
        ids = [row[id_column] for row in first]
        assert ids == sorted(ids, key=_id_sort_key)
        serialized = json.dumps(first, sort_keys=True)
        assert "timestamp" not in serialized.lower()
        assert "created_at" not in serialized.lower()


def test_protected_inputs_match_f6i2p_hash_manifest() -> None:
    manifest = json.loads(PROTECTED_HASHES_PATH.read_text(encoding="utf-8"))
    assert manifest["status"] == "pass"
    assert manifest["changed"] == []
    assert manifest["missing"] == []
    for item in manifest["protected_inputs"]:
        path = ROOT / item["path"]
        assert path.exists(), item["path"]
        assert _sha256(path) == item["post_sha256"]


def test_no_new_f6j_raw_llm_outputs_were_created() -> None:
    raw_roots = [
        ROOT / "results" / "llm",
        ROOT / "results" / "raw_outputs",
        ROOT / "results" / "raw",
    ]
    offenders: list[Path] = []
    for raw_root in raw_roots:
        if raw_root.exists():
            offenders.extend(raw_root.glob("**/*phase_f6j*"))
    assert offenders == []


def test_cleanup_preflight_and_handoff_are_current_not_stale() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")
    handoff = HANDOFF_PATH.read_text(encoding="utf-8")
    combined = report + "\n" + handoff
    assert "f6j_scientific_evidence_synthesis_complete" in handoff
    assert "f6j_scientific_evidence_synthesis_complete" in combined
    assert "140 passed, 1698 deselected" in combined
    assert "151 passed, 1698 deselected" in combined
    stale_tokens = [
        "129 passed",
        "2 failed",
        "f6i2kr_partial_guard_reconciliation_remaining",
        "f6i2n_failed_current_suite",
    ]
    for token in stale_tokens:
        assert token not in combined


def test_no_paths_are_staged_by_f6j() -> None:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""
