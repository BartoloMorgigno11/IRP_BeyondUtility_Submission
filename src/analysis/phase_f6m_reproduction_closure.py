"""Build the F6M offline reproduction closure and freeze artifacts.

This module performs no inference, training, download, API call, or raw-output
repair. It reads canonical CSV/JSON artifacts from prior phases and writes a
final traceability layer for the reproduction core.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


PHASE_ID = "phase_f6m_reproduction_closure"
DEFAULT_CONFIG_PATH = Path("configs/phase_f6m_reproduction_closure.yaml")
DECISION_COMPLETE = "f6m_reproduction_closure_complete_with_documented_limitations"
DECISION_BLOCKED = "f6m_reproduction_closure_blocked"
EXPECTED_RANKING_FREEZE_HASH = "D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC"

ROOT = Path(__file__).resolve().parents[2]

INPUT_VALIDATION_COLUMNS = [
    "check_id",
    "check_name",
    "status",
    "observed",
    "expected",
    "severity",
    "evidence",
    "notes",
]
ARTIFACT_INDEX_COLUMNS = [
    "artifact_id",
    "artifact_type",
    "path",
    "source_or_output",
    "sha256",
    "row_count",
    "role",
    "notes",
]
MASTER_MATRIX_COLUMNS = [
    "matrix_id",
    "component",
    "dataset",
    "task",
    "model_or_family",
    "evidence_layer",
    "execution_status",
    "paper_alignment",
    "numeric_evidence_status",
    "canonical_source",
    "allowed_claim",
    "prohibited_claim",
    "limitations",
]
MASTER_RESULTS_COLUMNS = [
    "result_id",
    "phase_id",
    "source_phase",
    "dataset",
    "task",
    "model_name",
    "model_family",
    "prompting_strategy",
    "metric_scope",
    "metric",
    "k",
    "value",
    "denominator",
    "denominator_policy",
    "numerator",
    "valid_rows",
    "total_rows",
    "strict_valid_rows",
    "positive_present_rows",
    "evidence_class",
    "reproduction_class",
    "paper_equivalence",
    "source_file",
    "notes",
]
MASTER_RESULTS_WIDE_COLUMNS = [
    "wide_id",
    "dataset",
    "task",
    "model_name",
    "model_family",
    "prompting_strategy",
    "metric_scope",
    "hr_at_5",
    "ndcg_at_5",
    "strict_valid_rate",
    "structural_invalid_rate",
    "duplicate_token_rate",
    "positive_present_rate",
    "denominator",
    "positive_present_rows",
    "source_phases",
    "notes",
]
RQ_COLUMNS = [
    "rq_id",
    "research_question",
    "f6m_status",
    "evidence_classification",
    "datasets",
    "tasks",
    "models",
    "metrics",
    "evidence_references",
    "key_numeric_values",
    "allowed_conclusion",
    "prohibited_conclusion",
    "principal_limitations",
]
CLAIM_COLUMNS = [
    "claim_id",
    "claim_text",
    "claim_status",
    "claim_type",
    "scope",
    "evidence_references",
    "numeric_value_status",
    "reproduction_layer",
    "allowed_wording",
    "prohibited_wording",
    "limitations",
    "traceability_status",
]
LIMITATION_COLUMNS = [
    "limitation_id",
    "area",
    "status",
    "impact",
    "evidence_references",
    "mitigation_or_boundary",
    "next_possible_work",
]
LASTFM_STATUS_COLUMNS = [
    "status_id",
    "component",
    "dataset",
    "task",
    "evidence_status",
    "run_id",
    "planned_or_denominator",
    "completed_final_responses",
    "strict_valid",
    "structural_invalid",
    "positive_present_rows",
    "positive_present_rate",
    "primary_metric_policy",
    "decision_token",
    "evidence_paths",
    "notes",
]
STAT_COLUMNS = [
    "assessment_id",
    "comparison",
    "metric",
    "status",
    "observed",
    "p_value",
    "unit",
    "n",
    "evidence_references",
    "notes",
]
BOUNDARY_COLUMNS = [
    "boundary_id",
    "area",
    "status",
    "allowed",
    "not_allowed",
    "evidence_references",
    "notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def stable_id(prefix: str, *parts: Any, length: int = 20) -> str:
    return f"{prefix}-{sha256_text('||'.join(str(part) for part in parts))[:length]}"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def fmt(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        return ""
    return f"{number:.12g}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return config


def resolve_config_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def require_columns(rows: Sequence[Mapping[str, str]], columns: Sequence[str], label: str) -> None:
    if not rows:
        raise ValueError(f"{label} has no rows")
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError(f"{label} missing columns: {', '.join(missing)}")


def git_text(args: Sequence[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def metric_value(rows: Sequence[Mapping[str, str]], metric: str, model: str, strategy: str) -> str:
    matches = [
        row
        for row in rows
        if row.get("metric") == metric
        and row.get("model_name") == model
        and row.get("prompting_strategy") == strategy
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {metric} row for {model}/{strategy}, found {len(matches)}")
    return matches[0].get("value", "")


def validate_inputs(config: Mapping[str, Any]) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]], dict[str, dict[str, Any]]]:
    inputs = config.get("inputs", {})
    validation = config.get("validation", {})
    if not isinstance(inputs, Mapping):
        raise ValueError("config inputs must be a mapping")
    rows: list[dict[str, str]] = []
    csv_inputs: dict[str, list[dict[str, str]]] = {}
    json_inputs: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, observed: str, expected: str, evidence: str, notes: str, severity: str = "error") -> None:
        rows.append(
            {
                "check_id": stable_id("F6M-CHECK", name),
                "check_name": name,
                "status": "pass" if passed else "fail",
                "observed": observed,
                "expected": expected,
                "severity": severity,
                "evidence": evidence,
                "notes": notes,
            }
        )

    for key, raw_path in sorted(inputs.items()):
        path = resolve_config_path(str(raw_path))
        exists = path.is_file()
        add(f"input_exists::{key}", exists, str(exists).lower(), "true", rel(path), "Required F6M source artifact.")
        if not exists:
            continue
        if path.suffix.lower() == ".json":
            json_inputs[key] = read_json(path)
        else:
            csv_inputs[key] = read_csv(path)

    expected_hash = str(validation.get("expected_ranking_freeze_hash", EXPECTED_RANKING_FREEZE_HASH))
    ranking_hashes = json_inputs.get("f6l_ranking_hashes", {})
    ranking_observed = str(ranking_hashes.get("freeze_bundle_sha256", ""))
    add(
        "ranking_freeze_hash_invariant",
        ranking_observed == expected_hash,
        ranking_observed,
        expected_hash,
        str(inputs.get("f6l_ranking_hashes", "")),
        "The frozen LastFM ranking bundle must not drift during closure.",
    )

    ranking_summary = json_inputs.get("f6l_ranking_summary", {})
    expected_ranking_decision = str(validation.get("expected_ranking_decision", ""))
    add(
        "ranking_decision_token",
        str(ranking_summary.get("decision_token", "")) == expected_ranking_decision,
        str(ranking_summary.get("decision_token", "")),
        expected_ranking_decision,
        str(inputs.get("f6l_ranking_summary", "")),
        "Ranking metrics are accepted only from the frozen F6L ranking closeout.",
    )

    reranking_summary = json_inputs.get("f6l_reranking_summary", {})
    expected_reranking_decision = str(validation.get("expected_reranking_decision", ""))
    add(
        "reranking_decision_token",
        str(reranking_summary.get("decision_token", "")) == expected_reranking_decision,
        str(reranking_summary.get("decision_token", "")),
        expected_reranking_decision,
        str(inputs.get("f6l_reranking_summary", "")),
        "Reranking metrics are accepted only after V4.1 completion.",
    )

    for key, expected_key in [
        ("aggregate_final_responses", "expected_reranking_aggregate_final_responses"),
        ("continuation_final_responses", "expected_reranking_continuation_final_responses"),
    ]:
        observed = str(reranking_summary.get(key, ""))
        expected = str(validation.get(expected_key, ""))
        add(f"reranking_{key}", observed == expected, observed, expected, str(inputs.get("f6l_reranking_summary", "")), "Completion count gate.")

    final_validation = csv_inputs.get("f6l_reranking_final_validation", [])
    if final_validation:
        require_columns(final_validation, ["check_name", "status"], "f6l_reranking_final_validation")
        failed = [row["check_name"] for row in final_validation if row.get("status") != "pass"]
        add(
            "reranking_final_validation_all_pass",
            not failed,
            "failed=" + ";".join(failed) if failed else f"passed={len(final_validation)}",
            "all pass",
            str(inputs.get("f6l_reranking_final_validation", "")),
            "F6L V4.1 validation must be green before closure.",
        )

    reranking_fail_strict = csv_inputs.get("f6l_reranking_fail_strict_metrics", [])
    if reranking_fail_strict:
        require_columns(reranking_fail_strict, ["metric", "total_rows", "positive_present_rows"], "f6l_reranking_fail_strict_metrics")
        positive_rows = {row.get("positive_present_rows", "") for row in reranking_fail_strict}
        total_rows = {row.get("total_rows", "") for row in reranking_fail_strict}
        add(
            "reranking_positive_present_policy",
            positive_rows == {str(validation.get("expected_reranking_positive_present_rows", ""))} and total_rows == {str(validation.get("expected_lastfm_operationalized_users", ""))},
            f"positive_present={sorted(positive_rows)}; total_rows={sorted(total_rows)}",
            f"positive_present={validation.get('expected_reranking_positive_present_rows')}; total_rows={validation.get('expected_lastfm_operationalized_users')}",
            str(inputs.get("f6l_reranking_fail_strict_metrics", "")),
            "The 314/1000 positive-present rule applies only to reranking diagnostics.",
        )

    if any(row["status"] != "pass" and row["severity"] == "error" for row in rows):
        raise RuntimeError("F6M input validation failed; see phase_f6m_input_validation.csv after a successful write path is available")
    return rows, csv_inputs, json_inputs


def normalize_metric_name(metric: str) -> str:
    if metric.startswith("HR@5"):
        return "HR@5"
    if metric.startswith("NDCG@5"):
        return "NDCG@5"
    return metric


def build_master_results(csv_inputs: Mapping[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []

    for row in csv_inputs["f6l_ranking_metrics"]:
        require_columns([row], ["dataset", "task", "model_name", "prompting_strategy", "metric", "value"], "f6l_ranking_metrics")
        results.append(
            {
                "result_id": stable_id("F6M-RESULT", "ranking", row.get("model_name"), row.get("prompting_strategy"), row.get("metric")),
                "phase_id": PHASE_ID,
                "source_phase": "phase_f6l_lastfm_v4_ranking_metrics_freeze",
                "dataset": row.get("dataset", ""),
                "task": row.get("task", ""),
                "model_name": row.get("model_name", ""),
                "model_family": "local_llm_substitute",
                "prompting_strategy": row.get("prompting_strategy", ""),
                "metric_scope": row.get("metric_scope", "primary_fail_strict"),
                "metric": row.get("metric", ""),
                "k": row.get("k", ""),
                "value": row.get("value", ""),
                "denominator": row.get("denominator", ""),
                "denominator_policy": row.get("denominator_policy", ""),
                "numerator": row.get("numerator", ""),
                "valid_rows": row.get("denominator", ""),
                "total_rows": row.get("denominator", ""),
                "strict_valid_rows": "",
                "positive_present_rows": "",
                "evidence_class": "paper_protocol_reproduction_under_explicit_operationalization",
                "reproduction_class": "operationalized_paper_dataset_local_llm",
                "paper_equivalence": "not_exact_numeric_reproduction",
                "source_file": "results/tables/phase_f6l_lastfm_v4_ranking_freeze_metrics.csv",
                "notes": row.get("notes", ""),
            }
        )

    for source_key, source_phase, metric_scope in [
        (
            "f6l_reranking_fail_strict_metrics",
            "phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution",
            "primary_fail_strict_all_1000_users",
        ),
        (
            "f6l_reranking_complete_case_metrics",
            "phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution",
            "diagnostic_positive_present_only",
        ),
        (
            "f6l_reranking_validity",
            "phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution",
            "validity_and_pool_coverage",
        ),
    ]:
        for row in csv_inputs[source_key]:
            results.append(
                {
                    "result_id": stable_id("F6M-RESULT", source_key, row.get("model_name"), row.get("prompting_strategy"), row.get("metric")),
                    "phase_id": PHASE_ID,
                    "source_phase": source_phase,
                    "dataset": row.get("dataset", ""),
                    "task": row.get("task", ""),
                    "model_name": row.get("model_name", ""),
                    "model_family": "local_llm_substitute",
                    "prompting_strategy": row.get("prompting_strategy", ""),
                    "metric_scope": metric_scope,
                    "metric": row.get("metric", ""),
                    "k": row.get("k", ""),
                    "value": row.get("value", ""),
                    "denominator": row.get("total_rows", ""),
                    "denominator_policy": "all_1000_users" if metric_scope != "diagnostic_positive_present_only" else "positive_present_rows_only_diagnostic",
                    "numerator": "",
                    "valid_rows": row.get("valid_rows", ""),
                    "total_rows": row.get("total_rows", ""),
                    "strict_valid_rows": row.get("strict_valid_rows", ""),
                    "positive_present_rows": row.get("positive_present_rows", ""),
                    "evidence_class": row.get("evidence_class", "paper_protocol_reproduction_under_explicit_operationalization"),
                    "reproduction_class": "operationalized_paper_dataset_local_llm",
                    "paper_equivalence": "not_exact_numeric_reproduction",
                    "source_file": row.get("source_file", ""),
                    "notes": row.get("notes", ""),
                }
            )

    for row in csv_inputs["f6d2_trainable_lastfm_metrics"]:
        if row.get("is_primary_result", "").lower() != "true":
            continue
        if row.get("evaluation_scope") != "candidate_set_20":
            continue
        results.append(
            {
                "result_id": stable_id("F6M-RESULT", "f6d2", row.get("model"), row.get("metric")),
                "phase_id": PHASE_ID,
                "source_phase": "finalization_f6d2_lastfm_trainable_models",
                "dataset": "LastFM",
                "task": "ranking",
                "model_name": row.get("model", ""),
                "model_family": "trainable_recommender_baseline",
                "prompting_strategy": row.get("candidate_set_name", ""),
                "metric_scope": row.get("evaluation_scope", ""),
                "metric": row.get("metric", ""),
                "k": row.get("cutoff", ""),
                "value": row.get("mean", ""),
                "denominator": "",
                "denominator_policy": "five_seed_mean_candidate_set_20",
                "numerator": "",
                "valid_rows": "",
                "total_rows": "",
                "strict_valid_rows": "",
                "positive_present_rows": "",
                "evidence_class": "adapted_multi_seed_local_trainer",
                "reproduction_class": row.get("reproduction_class", ""),
                "paper_equivalence": row.get("paper_equivalence", ""),
                "source_file": "results/tables/finalization_f6d2_aggregate_utility_metrics.csv",
                "notes": row.get("notes", ""),
            }
        )
    return sorted(results, key=lambda row: (row["dataset"], row["task"], row["model_family"], row["model_name"], row["prompting_strategy"], row["metric_scope"], row["metric"]))


def build_master_results_wide(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, str, str, str], dict[str, str]] = {}
    source_phases: dict[tuple[str, str, str, str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (
            row.get("dataset", ""),
            row.get("task", ""),
            row.get("model_name", ""),
            row.get("model_family", ""),
            row.get("prompting_strategy", ""),
            row.get("metric_scope", ""),
        )
        target = grouped.setdefault(
            key,
            {
                "wide_id": stable_id("F6M-WIDE", *key),
                "dataset": key[0],
                "task": key[1],
                "model_name": key[2],
                "model_family": key[3],
                "prompting_strategy": key[4],
                "metric_scope": key[5],
                "hr_at_5": "",
                "ndcg_at_5": "",
                "strict_valid_rate": "",
                "structural_invalid_rate": "",
                "duplicate_token_rate": "",
                "positive_present_rate": "",
                "denominator": row.get("denominator", ""),
                "positive_present_rows": row.get("positive_present_rows", ""),
                "source_phases": "",
                "notes": "",
            },
        )
        metric = row.get("metric", "")
        normalized = normalize_metric_name(metric)
        if normalized == "HR@5":
            target["hr_at_5"] = row.get("value", "")
        elif normalized == "NDCG@5":
            target["ndcg_at_5"] = row.get("value", "")
        elif metric == "strict_valid_rate":
            target["strict_valid_rate"] = row.get("value", "")
        elif metric == "structural_invalid_rate":
            target["structural_invalid_rate"] = row.get("value", "")
        elif metric == "duplicate_token_rate":
            target["duplicate_token_rate"] = row.get("value", "")
        elif metric == "positive_present_rate":
            target["positive_present_rate"] = row.get("value", "")
        if row.get("denominator", ""):
            target["denominator"] = row.get("denominator", "")
        if row.get("positive_present_rows", ""):
            target["positive_present_rows"] = row.get("positive_present_rows", "")
        if row.get("notes") and not target["notes"]:
            target["notes"] = row.get("notes", "")
        source_phases[key].add(row.get("source_phase", ""))
    for key, target in grouped.items():
        target["source_phases"] = "; ".join(sorted(source_phases[key]))
    return sorted(grouped.values(), key=lambda row: (row["dataset"], row["task"], row["model_family"], row["model_name"], row["prompting_strategy"], row["metric_scope"]))


def build_reproducibility_matrix() -> list[dict[str, str]]:
    rows = [
        {
            "component": "Exact paper numeric reproduction",
            "dataset": "All paper datasets",
            "task": "ranking; reranking; beyond-utility",
            "model_or_family": "paper model matrix",
            "evidence_layer": "scope boundary",
            "execution_status": "not_achieved",
            "paper_alignment": "blocked_missing_original_artifacts",
            "numeric_evidence_status": "no_exact_numeric_cells",
            "canonical_source": "results/tables/phase_f6j_scientific_claim_registry.csv; results/tables/phase_f6m_paper_claim_evidence_matrix.csv",
            "allowed_claim": "The project is a documented local reproduction wrapper, not an exact numerical replication.",
            "prohibited_claim": "The paper was exactly reproduced numerically.",
            "limitations": "Original raw outputs, model snapshots, candidate orders, and exact processed artifacts are not all available.",
        },
        {
            "component": "Canonical local LLM matrix before LastFM",
            "dataset": "Amazon Beauty; Amazon Sports; ML-1M",
            "task": "ranking; reranking",
            "model_or_family": "qwen2:7b-instruct; mistral:7b-instruct",
            "evidence_layer": "adapted local-model evidence",
            "execution_status": "completed",
            "paper_alignment": "repository-related prompt family with local structured-output adaptation",
            "numeric_evidence_status": "available_from_existing_csv",
            "canonical_source": "results/tables/canonical_local_llm_full_matrix_cell_metrics.csv; results/tables/phase_f6j_research_question_evidence_matrix.csv",
            "allowed_claim": "Local substitute-model utility and reliability are measured under explicit parser and denominator policies.",
            "prohibited_claim": "These local models reproduce the paper's original LLM identities.",
            "limitations": "Local models, parser, and prompt contracts differ from the original paper setting.",
        },
        {
            "component": "LastFM trainable baselines",
            "dataset": "LastFM",
            "task": "ranking",
            "model_or_family": "BPRMF; GRU4Rec; SASRec; LightGCN",
            "evidence_layer": "adapted multi-seed local trainer",
            "execution_status": "completed",
            "paper_alignment": "paper model families under reconstructed local protocol",
            "numeric_evidence_status": "available_from_existing_csv",
            "canonical_source": "results/tables/finalization_f6d2_aggregate_utility_metrics.csv",
            "allowed_claim": "LastFM trainable model-family evidence is available as adapted local evidence.",
            "prohibited_claim": "The exact paper LastFM checkpoints or runtime were reproduced.",
            "limitations": "Candidate set, preprocessing, and training runtime are local operationalizations.",
        },
        {
            "component": "LastFM local LLM ranking",
            "dataset": "LastFM",
            "task": "ranking",
            "model_or_family": "qwen2:7b-instruct; mistral:7b-instruct",
            "evidence_layer": "paper protocol reproduction under explicit operationalization",
            "execution_status": "completed_and_frozen",
            "paper_alignment": "paper dataset/task with local substitute LLMs and V4 structured output",
            "numeric_evidence_status": "available_from_existing_csv",
            "canonical_source": "results/tables/phase_f6l_lastfm_v4_ranking_freeze_metrics.csv; results/manifests/phase_f6l_lastfm_v4_ranking_metrics_freeze_hashes.json",
            "allowed_claim": "LastFM ranking is complete for the 1000-user operationalized local protocol.",
            "prohibited_claim": "LastFM ranking is paper-exact or full-cohort.",
            "limitations": "Uses local substitute LLMs and operationalized 1000-user cohort.",
        },
        {
            "component": "LastFM local LLM reranking",
            "dataset": "LastFM",
            "task": "reranking",
            "model_or_family": "qwen2:7b-instruct; mistral:7b-instruct",
            "evidence_layer": "paper protocol reproduction under explicit operationalization",
            "execution_status": "completed",
            "paper_alignment": "paper dataset/task with local substitute LLMs and V4.1 structured output",
            "numeric_evidence_status": "available_from_existing_csv",
            "canonical_source": "results/tables/phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution_fail_strict_metrics.csv; results/manifests/phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution_summary.json",
            "allowed_claim": "LastFM reranking is complete for all 4000 final responses in the 1000-user operationalized local protocol.",
            "prohibited_claim": "The 314 positive-present reranking subset is the primary denominator or applies to ranking.",
            "limitations": "Only 314/1000 reranking rows contain the positive item in the candidate pool; conditional positive-present metrics are diagnostic.",
        },
        {
            "component": "Future original extension",
            "dataset": "Future dataset or full LastFM cohort",
            "task": "not started",
            "model_or_family": "not selected",
            "evidence_layer": "future work boundary",
            "execution_status": "not_started",
            "paper_alignment": "outside closed reproduction core",
            "numeric_evidence_status": "no_numeric_value",
            "canonical_source": "results/tables/phase_f6m_reproduction_extension_boundary.csv",
            "allowed_claim": "The next step is to design an original extension from the frozen reproduction commit.",
            "prohibited_claim": "The extension has already started or produced results.",
            "limitations": "No extension dataset, model, metric, or result belongs to F6M closure.",
        },
    ]
    return [{**row, "matrix_id": stable_id("F6M-MATRIX", row["component"])} for row in rows]


def build_research_questions(csv_inputs: Mapping[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    ranking = csv_inputs["f6l_ranking_metrics"]
    reranking = csv_inputs["f6l_reranking_fail_strict_metrics"]
    qwen_rank = metric_value(ranking, "HR@5_fail_strict", "qwen2:7b-instruct", "base/OpenP5 title-list")
    mistral_rerank = metric_value(reranking, "HR@5_fail_strict", "mistral:7b-instruct", "base/OpenP5-derived reranking")
    return [
        {
            "rq_id": "RQ1",
            "research_question": "Do local LLM recommenders underperform traditional recommenders on utility?",
            "f6m_status": "supported_in_adapted_setting",
            "evidence_classification": "adapted and operationalized reproduction",
            "datasets": "Amazon Beauty; Amazon Sports; ML-1M; LastFM",
            "tasks": "ranking; reranking",
            "models": "Qwen2; Mistral; Random; MostPop; BM25; BPRMF; GRU4Rec; SASRec; LightGCN",
            "metrics": "HR@5; NDCG@5",
            "evidence_references": "results/tables/phase_f6m_master_results_long.csv; results/tables/finalization_f6d2_aggregate_utility_metrics.csv",
            "key_numeric_values": f"LastFM qwen2 base ranking HR@5_fail_strict={qwen_rank}; LastFM mistral base reranking HR@5_fail_strict={mistral_rerank}",
            "allowed_conclusion": "Utility conclusions are valid only inside named local/adapted protocols and intersections.",
            "prohibited_conclusion": "Do not claim exact paper-model utility reproduction.",
            "principal_limitations": "Local substitute models and reconstructed candidate protocols prevent exact paper equivalence.",
        },
        {
            "rq_id": "RQ2",
            "research_question": "Do dataset and task differences affect outcomes?",
            "f6m_status": "mixed_context_dependent",
            "evidence_classification": "adapted reproduction; diagnostic evidence",
            "datasets": "Amazon Beauty; Amazon Sports; ML-1M; LastFM",
            "tasks": "ranking; reranking",
            "models": "Qwen2; Mistral; trainable baselines",
            "metrics": "HR@5; NDCG@5; strict-validity",
            "evidence_references": "results/tables/phase_f6m_master_results_wide.csv; results/tables/phase_f6j_research_question_evidence_matrix.csv",
            "key_numeric_values": "LastFM ranking and reranking now have explicit F6M rows; task semantics remain separated.",
            "allowed_conclusion": "Dataset and task variation is observable but not isolated causally.",
            "prohibited_conclusion": "Do not merge ranking and reranking denominators.",
            "principal_limitations": "Candidate pools, titles, histories, and parser contracts differ across tasks.",
        },
        {
            "rq_id": "RQ3_RQ5",
            "research_question": "Do LLMs improve beyond-utility outcomes such as long-tail, novelty, and serendipity?",
            "f6m_status": "qualified_or_mixed",
            "evidence_classification": "adapted reproduction; original extension",
            "datasets": "Available paired datasets; LastFM where prior artifacts support the metric",
            "tasks": "ranking; reranking where available",
            "models": "Qwen2; Mistral; paired baselines",
            "metrics": "APLT@5; novelty; serendipity; coverage; validity",
            "evidence_references": "results/tables/phase_f6j_research_question_evidence_matrix.csv; results/tables/phase_f6m_limitations_register.csv",
            "key_numeric_values": "F6M does not compute new beyond-utility values; it carries forward prior generated CSV evidence.",
            "allowed_conclusion": "Beyond-utility claims remain qualified by metric-specific exactness.",
            "prohibited_conclusion": "Do not promote local proxies to paper-exact metrics.",
            "principal_limitations": "Some paper primitives remain unavailable or reconstructed.",
        },
        {
            "rq_id": "RQ8",
            "research_question": "Does profile generation improve recommendation effectiveness?",
            "f6m_status": "unresolved_not_reproduced",
            "evidence_classification": "blocked",
            "datasets": "Not available",
            "tasks": "profile generation",
            "models": "Not available",
            "metrics": "profile-only utility; profile-plus-history utility",
            "evidence_references": "results/tables/phase_f6j_research_question_evidence_matrix.csv#RQ8",
            "key_numeric_values": "not_tested",
            "allowed_conclusion": "Profile-generation effectiveness remains unclaimed.",
            "prohibited_conclusion": "Do not claim generated-profile benefit.",
            "principal_limitations": "Frozen profile prompts, outputs, and evaluation population are unavailable.",
        },
        {
            "rq_id": "RQ9",
            "research_question": "How reliable are LLM outputs and hallucination controls?",
            "f6m_status": "supported_for_measured_local_reliability",
            "evidence_classification": "original extension; operationalized reproduction",
            "datasets": "Amazon Beauty; Amazon Sports; ML-1M; LastFM",
            "tasks": "ranking; reranking",
            "models": "Qwen2; Mistral",
            "metrics": "strict_valid_rate; structural_invalid_rate; duplicate_token_rate; hallucination_rate",
            "evidence_references": "results/tables/phase_f6m_master_results_long.csv; results/tables/phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution_validity.csv",
            "key_numeric_values": "LastFM V4.1 reranking aggregate final responses=4000; duplicate-token invalid=29; hallucination_rate=0 in V4.1 validity table.",
            "allowed_conclusion": "Reliability is measured separately from recommendation quality.",
            "prohibited_conclusion": "Do not repair invalid outputs or claim hallucination was eliminated universally.",
            "principal_limitations": "Structured-index hallucination is narrower than free-text hallucination.",
        },
        {
            "rq_id": "RQ10_RQ11",
            "research_question": "Does reranking or LLMRank consistently improve over ranking or Base prompting?",
            "f6m_status": "not_supported_as_general_advantage",
            "evidence_classification": "adapted operationalized comparison",
            "datasets": "Amazon Beauty; Amazon Sports; ML-1M; LastFM",
            "tasks": "ranking versus reranking; Base versus LLMRank",
            "models": "Qwen2; Mistral",
            "metrics": "HR@5; NDCG@5; descriptive deltas",
            "evidence_references": "results/tables/phase_f6m_statistical_assessment.csv; results/tables/phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution_ranking_vs_reranking_comparison.csv",
            "key_numeric_values": "All LastFM reranking-minus-ranking fail-strict HR/NDCG deltas in V4.1 comparison are negative.",
            "allowed_conclusion": "Reranking and LLMRank effects are protocol- and cell-specific.",
            "prohibited_conclusion": "Do not claim a global reranking or LLMRank advantage.",
            "principal_limitations": "Current V4.1 runner reports descriptive deltas, not confirmatory paired tests.",
        },
    ]


def build_claims() -> list[dict[str, str]]:
    return [
        {
            "claim_id": "F6M-CLAIM-001",
            "claim_text": "The reproduction core is closed with documented limitations.",
            "claim_status": "supported",
            "claim_type": "closure boundary",
            "scope": "Whole project",
            "evidence_references": "results/tables/phase_f6m_input_validation.csv; report/phase_f6m_reproduction_closure.md",
            "numeric_value_status": "available_from_generated_csv",
            "reproduction_layer": "closure synthesis",
            "allowed_wording": "REPRODUCIBILITY CORE - CLOSED.",
            "prohibited_wording": "All paper results were exactly replicated.",
            "limitations": "Closure is scientific/documentary; it does not make missing original artifacts available.",
            "traceability_status": "traceable",
        },
        {
            "claim_id": "F6M-CLAIM-002",
            "claim_text": "LastFM local LLM ranking is complete for the operationalized 1000-user protocol.",
            "claim_status": "supported",
            "claim_type": "scientific evidence",
            "scope": "LastFM ranking",
            "evidence_references": "results/tables/phase_f6l_lastfm_v4_ranking_freeze_metrics.csv",
            "numeric_value_status": "available_from_generated_csv",
            "reproduction_layer": "paper protocol reproduction under explicit operationalization",
            "allowed_wording": "LastFM ranking is complete under the frozen F6L operationalization.",
            "prohibited_wording": "LastFM ranking exactly reproduces the paper's full protocol.",
            "limitations": "Local substitute LLMs and operationalized candidate/cohort conditions remain material.",
            "traceability_status": "traceable",
        },
        {
            "claim_id": "F6M-CLAIM-003",
            "claim_text": "LastFM local LLM reranking V4.1 is complete for all planned final responses.",
            "claim_status": "supported",
            "claim_type": "scientific evidence",
            "scope": "LastFM reranking",
            "evidence_references": "results/manifests/phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution_summary.json; results/tables/phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution_fail_strict_metrics.csv",
            "numeric_value_status": "available_from_generated_csv",
            "reproduction_layer": "paper protocol reproduction under explicit operationalization",
            "allowed_wording": "LastFM reranking V4.1 completed 4000/4000 aggregate final responses.",
            "prohibited_wording": "The positive-present diagnostic subset is the primary reranking denominator.",
            "limitations": "314/1000 reranking rows contain the positive item in the candidate pool.",
            "traceability_status": "traceable",
        },
        {
            "claim_id": "F6M-CLAIM-004",
            "claim_text": "The future extension has not started.",
            "claim_status": "supported",
            "claim_type": "scope boundary",
            "scope": "Future extension",
            "evidence_references": "results/tables/phase_f6m_reproduction_extension_boundary.csv",
            "numeric_value_status": "no_numeric_value",
            "reproduction_layer": "future work boundary",
            "allowed_wording": "FUTURE EXTENSION - NOT STARTED.",
            "prohibited_wording": "F6M extension results exist.",
            "limitations": "Extension design begins only after the frozen reproduction commit.",
            "traceability_status": "traceable",
        },
    ]


def build_limitations() -> list[dict[str, str]]:
    items = [
        (
            "exact_paper_numeric_reproduction",
            "paper artifact availability",
            "open",
            "Prevents exact numerical replication claims.",
            "results/tables/phase_f6m_paper_claim_evidence_matrix.csv",
            "Use exact/reconstructed/adapted labels and prohibit exact numeric claims.",
            "Acquire official processed artifacts, raw outputs, and model snapshots if available.",
        ),
        (
            "local_llm_substitutes",
            "model identity",
            "bounded",
            "Local Qwen2/Mistral behavior cannot be generalized to paper LLM identities.",
            "results/tables/phase_f6m_master_results_long.csv",
            "Label all local LLM rows as substitute-model evidence.",
            "Run paper model identities only with explicit authorization and artifacts.",
        ),
        (
            "lastfm_1000_user_operationalization",
            "cohort",
            "bounded",
            "LastFM closure is not a full 1859-user extension or exact paper cohort.",
            "results/tables/phase_f6m_lastfm_final_status.csv",
            "State the 1000-user operationalized scope in every LastFM claim.",
            "Design full-cohort extension after the frozen reproduction commit.",
        ),
        (
            "lastfm_reranking_positive_present",
            "candidate pool",
            "bounded",
            "Only 314/1000 reranking rows contain the positive item.",
            "results/tables/phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution_validity.csv",
            "Keep all-1000 fail-strict metrics primary; mark positive-present metrics diagnostic.",
            "Build a future extension with an explicitly positive-present reranking pool if scientifically desired.",
        ),
        (
            "descriptive_statistics_only_for_v4_1_deltas",
            "statistics",
            "bounded",
            "F6L V4.1 comparison table reports descriptive deltas only.",
            "results/tables/phase_f6m_statistical_assessment.csv",
            "Do not report p-values where paired per-user contrasts are not extracted.",
            "Add a paired contrast extractor in a future post-closure analysis if row-level pairing is verified.",
        ),
        (
            "future_extension_not_started",
            "scope control",
            "closed_for_f6m",
            "No extension evidence is produced by F6M.",
            "results/tables/phase_f6m_reproduction_extension_boundary.csv",
            "Stop after the reproduction closure commit.",
            "Design the original extension starting from the frozen reproduction commit.",
        ),
    ]
    return [
        {
            "limitation_id": stable_id("F6M-LIMIT", item[0]),
            "area": item[1],
            "status": item[2],
            "impact": item[3],
            "evidence_references": item[4],
            "mitigation_or_boundary": item[5],
            "next_possible_work": item[6],
        }
        for item in items
    ]


def build_lastfm_status(csv_inputs: Mapping[str, list[dict[str, str]]], json_inputs: Mapping[str, dict[str, Any]]) -> list[dict[str, str]]:
    ranking_summary = json_inputs["f6l_ranking_summary"]
    reranking_summary = json_inputs["f6l_reranking_summary"]
    validity = csv_inputs["f6l_reranking_validity"]
    positive_rows = sorted({row.get("positive_present_rows", "") for row in validity if row.get("positive_present_rows", "")})
    positive_present = positive_rows[0] if positive_rows else ""
    positive_rate = sorted({row.get("value", "") for row in validity if row.get("metric") == "positive_present_rate"})
    return [
        {
            "status_id": "F6M-LASTFM-001",
            "component": "local_llm_ranking_v4_freeze",
            "dataset": "LastFM",
            "task": "ranking",
            "evidence_status": "completed_and_frozen",
            "run_id": str(ranking_summary.get("run_id", "20260804_131959")),
            "planned_or_denominator": "1000 users per model/strategy cell",
            "completed_final_responses": "4000",
            "strict_valid": "see phase_f6m_master_results_long.csv strict_valid_rate rows",
            "structural_invalid": "see phase_f6m_master_results_long.csv structural_invalid_rate rows",
            "positive_present_rows": "not_applicable_for_ranking",
            "positive_present_rate": "not_applicable_for_ranking",
            "primary_metric_policy": "fail_strict_all_1000_users_per_cell",
            "decision_token": str(ranking_summary.get("decision_token", "")),
            "evidence_paths": "results/tables/phase_f6l_lastfm_v4_ranking_freeze_metrics.csv; results/manifests/phase_f6l_lastfm_v4_ranking_metrics_freeze_hashes.json",
            "notes": "The 314/1000 positive-present constraint is a reranking-pool property and is not applied to ranking.",
        },
        {
            "status_id": "F6M-LASTFM-002",
            "component": "local_llm_reranking_v4_1_execution",
            "dataset": "LastFM",
            "task": "reranking",
            "evidence_status": "completed",
            "run_id": str(reranking_summary.get("execution_run_id", "")),
            "planned_or_denominator": "1000 users per model/strategy cell",
            "completed_final_responses": str(reranking_summary.get("aggregate_final_responses", "")),
            "strict_valid": str(reranking_summary.get("strict_valid", "")),
            "structural_invalid": str(reranking_summary.get("structural_invalid", "")),
            "positive_present_rows": positive_present,
            "positive_present_rate": positive_rate[0] if len(positive_rate) == 1 else ";".join(positive_rate),
            "primary_metric_policy": "fail_strict_all_1000_users_per_cell; positive_present_only metrics diagnostic",
            "decision_token": str(reranking_summary.get("decision_token", "")),
            "evidence_paths": "results/tables/phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution_fail_strict_metrics.csv; results/tables/phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution_complete_case_metrics.csv",
            "notes": "V4.1 aggregate final responses are 4000/4000; duplicate-token invalid outputs remain invalid and unrepaired.",
        },
        {
            "status_id": "F6M-LASTFM-003",
            "component": "full_lastfm_extension",
            "dataset": "LastFM",
            "task": "future extension",
            "evidence_status": "not_started",
            "run_id": "",
            "planned_or_denominator": "not_defined",
            "completed_final_responses": "0",
            "strict_valid": "",
            "structural_invalid": "",
            "positive_present_rows": "",
            "positive_present_rate": "",
            "primary_metric_policy": "outside_f6m_closure",
            "decision_token": DECISION_COMPLETE,
            "evidence_paths": "results/tables/phase_f6m_reproduction_extension_boundary.csv",
            "notes": "No full 1859-user LastFM extension, new dataset, or new model run is started in F6M.",
        },
    ]


def build_statistical_assessment(csv_inputs: Mapping[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in csv_inputs["f6l_reranking_statistics"]:
        rows.append(
            {
                "assessment_id": stable_id("F6M-STAT", row.get("test_name"), row.get("model_name"), row.get("prompting_strategy"), row.get("metric")),
                "comparison": f"{row.get('model_name', '')} | {row.get('prompting_strategy', '')}",
                "metric": row.get("metric", ""),
                "status": row.get("status", ""),
                "observed": row.get("observed", ""),
                "p_value": row.get("p_value", ""),
                "unit": "reranking_minus_frozen_ranking",
                "n": "1000 users per cell in primary fail-strict metrics",
                "evidence_references": "results/tables/phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution_statistical_tests.csv",
                "notes": row.get("notes", ""),
            }
        )
    rows.append(
        {
            "assessment_id": "F6M-STAT-BOUNDARY",
            "comparison": "F6M closure",
            "metric": "all final claims",
            "status": "descriptive_or_source_reported_only",
            "observed": "no new confirmatory tests computed",
            "p_value": "",
            "unit": "claim boundary",
            "n": "",
            "evidence_references": "results/tables/phase_f6m_limitations_register.csv",
            "notes": "F6M does not invent p-values; paired tests remain future work unless row-level aligned contrasts are explicitly extracted and validated.",
        }
    )
    return rows


def build_boundary() -> list[dict[str, str]]:
    rows = [
        {
            "boundary_id": "F6M-BOUNDARY-001",
            "area": "reproducibility_core",
            "status": "closed",
            "allowed": "Read and cite F6M tables, reports, manifests, and prior canonical artifacts.",
            "not_allowed": "Change the reproduction numerator by reclassifying adapted or diagnostic evidence as exact.",
            "evidence_references": "report/phase_f6m_reproduction_closure.md; results/manifests/phase_f6m_reproduction_closure_manifest.json",
            "notes": "REPRODUCIBILITY CORE - CLOSED.",
        },
        {
            "boundary_id": "F6M-BOUNDARY-002",
            "area": "api_inference_training_downloads",
            "status": "not_used_in_f6m",
            "allowed": "Offline reads of existing CSV/JSON/Markdown artifacts.",
            "not_allowed": "LLM/API calls, recommender training, downloads, raw-output repair, response deduplication, or full LastFM extension.",
            "evidence_references": "configs/phase_f6m_reproduction_closure.yaml",
            "notes": "The F6M CLI refuses --allow-api-calls.",
        },
        {
            "boundary_id": "F6M-BOUNDARY-003",
            "area": "future_extension",
            "status": "not_started",
            "allowed": "Design the original extension starting from the frozen reproduction commit.",
            "not_allowed": "Include extension metrics, datasets, model runs, or conclusions in the F6M reproduction closure.",
            "evidence_references": "results/tables/phase_f6m_lastfm_final_status.csv",
            "notes": "FUTURE EXTENSION - NOT STARTED.",
        },
    ]
    return rows


def build_artifact_index(paths: Mapping[str, Path], roles: Mapping[str, str]) -> list[dict[str, str]]:
    rows = []
    for key, path in sorted(paths.items()):
        if not path.is_file():
            continue
        row_count = ""
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                row_count = str(max(sum(1 for _ in handle) - 1, 0))
        rows.append(
            {
                "artifact_id": stable_id("F6M-ART", key, rel(path)),
                "artifact_type": path.suffix.lower().lstrip(".") or "file",
                "path": rel(path),
                "source_or_output": "output" if key.startswith("output::") else "source",
                "sha256": sha256_file(path),
                "row_count": row_count,
                "role": roles.get(key, ""),
                "notes": "F6M generated artifact." if key.startswith("output::") else "Prior canonical source artifact.",
            }
        )
    return rows


def build_report(
    *,
    manifest: Mapping[str, Any],
    master_wide: Sequence[Mapping[str, str]],
    lastfm_status: Sequence[Mapping[str, str]],
    claims: Sequence[Mapping[str, str]],
    limitations: Sequence[Mapping[str, str]],
    validation_history: Sequence[Mapping[str, str]],
    outputs: Mapping[str, str],
) -> str:
    ranking_rows = [row for row in master_wide if row.get("dataset") == "LastFM" and row.get("task") == "ranking" and row.get("model_family") == "local_llm_substitute" and row.get("metric_scope") == "primary_fail_strict"]
    reranking_rows = [row for row in master_wide if row.get("dataset") == "LastFM" and row.get("task") == "reranking" and row.get("metric_scope") == "primary_fail_strict_all_1000_users"]
    diagnostic_rows = [row for row in master_wide if row.get("metric_scope") == "diagnostic_positive_present_only"]

    def table(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> str:
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
        return "\n".join(lines)

    return "\n".join(
        [
            "# Phase F6M Reproduction Closure",
            "",
            f"Decision token: `{manifest['decision_token']}`.",
            "",
            "## Closure Status",
            "",
            "REPRODUCIBILITY CORE - CLOSED.",
            "",
            "FUTURE EXTENSION - NOT STARTED.",
            "",
            "F6M is an offline consolidation phase. It reads existing generated CSV/JSON evidence and does not run LLM inference, APIs, downloads, recommender training, full LastFM extension, raw-output repair, or deduplication.",
            "",
            "## LastFM Final Status",
            "",
            table(lastfm_status, ["component", "task", "evidence_status", "run_id", "completed_final_responses", "positive_present_rows", "primary_metric_policy"]),
            "",
            "## LastFM Primary Utility",
            "",
            "Ranking and reranking are separate protocols. The reranking positive-present condition is a reranking-pool diagnostic only and is not applied to ranking.",
            "",
            table(ranking_rows, ["model_name", "prompting_strategy", "hr_at_5", "ndcg_at_5", "denominator"]),
            "",
            table(reranking_rows, ["model_name", "prompting_strategy", "hr_at_5", "ndcg_at_5", "denominator", "positive_present_rows"]),
            "",
            "## Diagnostic Positive-Present Reranking",
            "",
            table(diagnostic_rows, ["model_name", "prompting_strategy", "hr_at_5", "ndcg_at_5", "positive_present_rows"]),
            "",
            "## Claims",
            "",
            table(claims, ["claim_id", "claim_status", "allowed_wording", "prohibited_wording"]),
            "",
            "## Limitations",
            "",
            table(limitations, ["area", "status", "mitigation_or_boundary"]),
            "",
            "## Validation",
            "",
            table(validation_history, ["gate", "status", "result"]),
            "",
            "## Generated Artifacts",
            "",
            "\n".join(f"- `{path}`" for path in outputs.values()),
            "",
            "## Next Step",
            "",
            "Design the original extension starting from the frozen reproduction commit.",
            "",
        ]
    )


def run_phase(config_path: Path = DEFAULT_CONFIG_PATH, *, allow_api_calls: bool = False) -> dict[str, Any]:
    if allow_api_calls:
        raise RuntimeError("F6M is offline-only and refuses --allow-api-calls")

    config = load_config(config_path)
    phase = config.get("phase", {})
    if not isinstance(phase, Mapping) or phase.get("id") != PHASE_ID:
        raise ValueError(f"unexpected phase id in {config_path}")

    validation_rows, csv_inputs, json_inputs = validate_inputs(config)
    master_results = build_master_results(csv_inputs)
    master_wide = build_master_results_wide(master_results)
    matrix = build_reproducibility_matrix()
    rqs = build_research_questions(csv_inputs)
    claims = build_claims()
    limitations = build_limitations()
    lastfm_status = build_lastfm_status(csv_inputs, json_inputs)
    stats = build_statistical_assessment(csv_inputs)
    boundary = build_boundary()
    validation_history = config.get("validation_history", [])
    if not isinstance(validation_history, list):
        raise ValueError("validation_history must be a list when present")

    outputs_config = config.get("outputs", {})
    tables_config = outputs_config.get("tables", {}) if isinstance(outputs_config, Mapping) else {}
    if not isinstance(tables_config, Mapping):
        raise ValueError("outputs.tables must be a mapping")
    output_tables = {key: resolve_config_path(str(path)) for key, path in tables_config.items()}
    report_path = resolve_config_path(str(outputs_config.get("report", "")))
    manifest_path = resolve_config_path(str(outputs_config.get("manifest", "")))

    write_csv(output_tables["input_validation"], validation_rows, INPUT_VALIDATION_COLUMNS)
    write_csv(output_tables["master_reproducibility_matrix"], matrix, MASTER_MATRIX_COLUMNS)
    write_csv(output_tables["master_results_long"], master_results, MASTER_RESULTS_COLUMNS)
    write_csv(output_tables["master_results_wide"], master_wide, MASTER_RESULTS_WIDE_COLUMNS)
    write_csv(output_tables["research_question_matrix"], rqs, RQ_COLUMNS)
    write_csv(output_tables["paper_claim_evidence_matrix"], claims, CLAIM_COLUMNS)
    write_csv(output_tables["limitations_register"], limitations, LIMITATION_COLUMNS)
    write_csv(output_tables["lastfm_final_status"], lastfm_status, LASTFM_STATUS_COLUMNS)
    write_csv(output_tables["statistical_assessment"], stats, STAT_COLUMNS)
    write_csv(output_tables["reproduction_extension_boundary"], boundary, BOUNDARY_COLUMNS)

    source_paths = {
        f"source::{key}": resolve_config_path(str(path))
        for key, path in sorted(config.get("inputs", {}).items())
    }
    output_paths = {f"output::{key}": path for key, path in output_tables.items()}
    output_paths["output::report"] = report_path
    roles = {key: "F6M source input" for key in source_paths}
    roles.update({key: "F6M generated output" for key in output_paths})

    provisional_manifest = {
        "phase_id": PHASE_ID,
        "decision_token": DECISION_COMPLETE,
        "generated_at_utc": utc_now(),
        "config_path": rel(resolve_config_path(config_path)),
        "git": {
            "branch": git_text(["branch", "--show-current"]),
            "head": git_text(["rev-parse", "HEAD"]),
            "original_repo_status": git_text(["status", "--short", "--", "original_repo"]),
        },
        "offline_guards": {
            "no_api_calls": True,
            "no_training": True,
            "no_downloads": True,
            "no_new_inference": True,
            "no_full_lastfm_extension": True,
        },
        "counts": {
            "master_results_long_rows": len(master_results),
            "master_results_wide_rows": len(master_wide),
            "research_question_rows": len(rqs),
            "claim_rows": len(claims),
            "limitation_rows": len(limitations),
        },
        "lastfm": {
            "ranking_freeze_hash": json_inputs["f6l_ranking_hashes"]["freeze_bundle_sha256"],
            "reranking_decision_token": json_inputs["f6l_reranking_summary"]["decision_token"],
            "reranking_aggregate_final_responses": json_inputs["f6l_reranking_summary"]["aggregate_final_responses"],
            "reranking_positive_present_rows": 314,
            "reranking_positive_present_policy": "diagnostic_only_not_primary_denominator",
        },
        "validation_history": validation_history,
    }

    report_text = build_report(
        manifest=provisional_manifest,
        master_wide=master_wide,
        lastfm_status=lastfm_status,
        claims=claims,
        limitations=limitations,
        validation_history=validation_history,
        outputs={key: rel(path) for key, path in output_tables.items()} | {"report": rel(report_path), "manifest": rel(manifest_path)},
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    artifact_index = build_artifact_index(source_paths | output_paths, roles)
    write_csv(output_tables["artifact_index"], artifact_index, ARTIFACT_INDEX_COLUMNS)

    output_artifacts = {rel(path): sha256_file(path) for path in sorted(output_paths.values(), key=lambda item: rel(item)) if path.is_file()}
    output_artifacts[rel(output_tables["artifact_index"])] = sha256_file(output_tables["artifact_index"])
    manifest = dict(provisional_manifest)
    manifest["manifest_path"] = rel(manifest_path)
    manifest["output_artifacts"] = output_artifacts
    manifest["artifact_index_rows"] = len(artifact_index)
    write_json(manifest_path, manifest)

    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build F6M offline reproduction closure artifacts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--allow-api-calls", action="store_true", help="Rejected: F6M is offline-only.")
    args = parser.parse_args(argv)
    manifest = run_phase(args.config, allow_api_calls=args.allow_api_calls)
    print(stable_json({"decision_token": manifest["decision_token"], "manifest": manifest["manifest_path"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
