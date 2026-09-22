"""Compute G2 history-length metrics from the frozen analytic universe.

This script is offline-only: it reads retained G1/G2 evidence and frozen
metric contracts, computes cell-level descriptive metrics, and writes the
G2 metric freeze artifacts. It does not call Ollama, start G3, or modify any
frozen protocol/execution/evidence artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.canonical_metric_contract import (  # noqa: E402
    build_long_tail_set,
    build_metric_registry,
    candidate_validity_rate_at_k,
    catalog_coverage_ratio,
    duplicate_rate_at_k,
    first_positive_rank_zero_based,
    hr_at_k_from_rank,
    ndcg_at_k_from_rank,
    novelty_self_information,
    validate_metric_registry,
    validate_one_positive,
)


EXPECTED_HEAD = "70d808982645a4b607cef1b3d2444d1febe0d298"
PHASE_ID = "g2_history_length_metrics_freeze"
DECISION_FROZEN = "G2_METRICS_FROZEN"
DECISION_BLOCKED = "G2_METRICS_BLOCKED"
TOP_K = 5

PRIMARY_VIEW = "PRIMARY_B1_GE10"
LONG_VIEW = "LONG_HISTORY_B2_GE20"
SECONDARY_VIEW = "SECONDARY_FULL_CURVE_GE20"
TRUE_ZERO_VIEW = "FOODCOM_TRUE_ZERO_HISTORY_DIAGNOSTIC"
REPORTING_VIEWS = [PRIMARY_VIEW, LONG_VIEW, SECONDARY_VIEW, TRUE_ZERO_VIEW]
HISTORY_BY_VIEW = {
    PRIMARY_VIEW: [0, 1, 2, 5, 10],
    LONG_VIEW: [10, 20],
    SECONDARY_VIEW: [0, 1, 2, 5, 10, 20],
    TRUE_ZERO_VIEW: [0],
}
DATASETS = ["beauty", "sports", "ml1m", "lastfm", "foodcom"]
MODELS = ["qwen2:7b-instruct", "mistral:7b-instruct", "gemma2:9b"]
PROMPTS = ["BASE", "LLMRANK"]

G1_EVIDENCE = "G1_EXACT_REUSE_TERMINAL_EVIDENCE"
G2_EVIDENCE = "G2_TERMINAL_EVIDENCE"

UTILITY_METRICS = ("hr_at_5", "ndcg_at_5", "mrr")
BEYOND_UTILITY_METRICS = (
    "novelty_at_5",
    "catalog_coverage_at_5",
    "item_coverage_at_5",
    "long_tail_rate_at_5",
    "average_popularity_at_5",
)
STRUCTURAL_METRICS = ("duplicate_rate_at_5", "candidate_validity_rate")
RELIABILITY_METRICS = (
    "backend_success_rate",
    "strict_parse_valid_rate",
    "invalid_length_rate",
    "duplicate_index_rate",
    "invalid_candidate_reference_rate",
)
BLOCKED_BEYOND_UTILITY_METRICS = ("popularity_bias_at_5", "intra_list_diversity_at_5")
COMPUTED_METRICS = UTILITY_METRICS + BEYOND_UTILITY_METRICS + STRUCTURAL_METRICS + RELIABILITY_METRICS

INVALID_CANDIDATE_CLASSES = {
    "HALLUCINATION_OR_OUT_OF_CANDIDATE",
    "INVALID_CANDIDATE",
    "INVALID_CANDIDATE_REFERENCE",
    "HALLUCINATION",
}

PATHS = {
    "protocol_config": Path("configs/g2_history_length_sensitivity.yaml"),
    "execution_config": Path("configs/g2_history_length_execution.yaml"),
    "runner": Path("src/extension/g2_history_length_execution.py"),
    "protocol_manifest": Path("results/manifests/g2_history_length_protocol_freeze.json"),
    "execution_manifest": Path("results/manifests/g2_history_length_execution_freeze.json"),
    "retention_manifest": Path("results/manifests/g2_history_length_execution_retention_manifest.json"),
    "analysis_manifest": Path("results/manifests/g2_history_length_analysis_universe_freeze.json"),
    "request_universe": Path("results/tables/g2_history_length_request_universe.csv"),
    "analysis_universe": Path("results/tables/g2_history_length_analysis_universe.csv"),
    "membership_universe": Path("results/tables/g2_history_length_analysis_membership_universe.csv"),
    "cell_counts": Path("results/tables/g2_history_length_analysis_cell_counts.csv"),
    "denominator_contract": Path("results/tables/g2_history_length_denominator_contract.csv"),
    "g1_reuse": Path("results/tables/g2_history_length_g1_execution_reuse.csv"),
    "g1_strict_parsing": Path("results/tables/g1_primary_ranking_strict_parsing.csv"),
    "g2_responses": Path("results/llm/extension_canonical/g2_history_length_sensitivity/responses.jsonl"),
    "train_item_statistics": Path("results/tables/canonical_metric_train_item_statistics.csv"),
    "lastfm_train": Path("data/processed/finalization_f6c_lastfm_interactions.csv"),
    "foodcom_train": Path("data/raw/foodcom/d0a_extracted/e7a43e623c61/interactions_train.csv"),
    "metric_contract_source": Path("src/metrics/canonical_metric_contract.py"),
    "metric_computation_source": Path("src/metrics/canonical_metric_computation.py"),
    "notes": Path("report/notes.md"),
}
OUTPUTS = {
    "metric_results": Path("results/tables/g2_history_length_metric_results.csv"),
    "metric_denominators": Path("results/tables/g2_history_length_metric_denominators.csv"),
    "metric_validation": Path("results/tables/g2_history_length_metric_validation.csv"),
    "metric_undefined": Path("results/tables/g2_history_length_metric_undefined.csv"),
    "manifest": Path("results/manifests/g2_history_length_metrics_freeze.json"),
    "report": Path("report/g2_history_length_metrics_validation.md"),
}

EXPECTED_DIRTY_PATHS = {
    "src/analysis/g2_history_length_metrics.py",
    "report/notes.md",
    *[path.as_posix() for path in OUTPUTS.values()],
}


@dataclass(frozen=True)
class PopularityData:
    dataset: str
    train_counts: Mapping[str, int]
    total_train_interactions: int
    train_seen_catalog: frozenset[str]
    long_tail_items: frozenset[str]
    long_tail_threshold: float
    source_path: str
    source_sha256: str
    source_policy: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute frozen G2 history-length metrics.")
    parser.add_argument("--output-root", default=".", help="Project root; defaults to current working directory.")
    parser.add_argument("--no-write", action="store_true", help="Validate and compute in memory without writing outputs.")
    parser.add_argument(
        "--allow-generator-dirty",
        action="store_true",
        help="Allow this generator and expected metric outputs to be dirty while frozen inputs remain unchanged.",
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, *parts: Any) -> str:
    payload = stable_json([str(part) for part in parts])
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32].upper()}"


def read_csv(root: Path, path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(root / path, dtype=str, keep_default_na=False, low_memory=False, **kwargs)


def read_json(root: Path, path: Path) -> dict[str, Any]:
    return json.loads((root / path).read_text(encoding="utf-8"))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def bool_text(value: Any) -> str:
    return "true" if str(value).strip().lower() == "true" else "false"


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        raise ValueError("non-finite float cannot be serialized")
    return f"{float(value):.17g}"


def parse_json_array(value: Any) -> list[str]:
    if value is None or str(value).strip() == "":
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("expected a JSON array")
    return [str(item) for item in parsed]


def reciprocal_rank(rank_zero_based: int | None) -> float:
    return 0.0 if rank_zero_based is None else 1.0 / float(int(rank_zero_based) + 1)


def git_output(root: Path, args: list[str], *, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def git_dirty_paths(status: str) -> list[str]:
    paths: list[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        if line.startswith("?? "):
            text = line[3:]
        elif len(line) >= 3 and line[2] == " ":
            text = line[3:]
        elif len(line) >= 2 and line[1] == " ":
            text = line[2:]
        else:
            text = line[3:] if len(line) > 3 else line
        if " -> " in text:
            text = text.split(" -> ", 1)[1]
        paths.append(text.replace("\\", "/"))
    return sorted(paths)


def status_is_clean_or_expected(status: str, allow_generator_dirty: bool) -> bool:
    paths = git_dirty_paths(status)
    if not paths:
        return True
    if not allow_generator_dirty:
        return False
    return set(paths).issubset(EXPECTED_DIRTY_PATHS)


def canonical_dataset(value: str) -> str:
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    mapping = {
        "amazon beauty": "beauty",
        "beauty": "beauty",
        "amazon sports": "sports",
        "sports": "sports",
        "ml 1m": "ml1m",
        "ml1m": "ml1m",
        "lastfm": "lastfm",
        "last fm": "lastfm",
        "foodcom": "foodcom",
        "food.com": "foodcom",
        "food com": "foodcom",
    }
    if text not in mapping:
        raise ValueError(f"unknown dataset label: {value}")
    return mapping[text]


def load_train_source(
    root: Path,
    path: Path,
    *,
    user_col: str,
    item_col: str,
    split_col: str | None,
    split_value: str = "train",
) -> tuple[Counter[str], int]:
    frame = read_csv(root, path)
    if split_col is not None:
        frame = frame.loc[frame[split_col].astype(str).str.lower().eq(split_value)].copy()
    if user_col not in frame.columns or item_col not in frame.columns:
        raise ValueError(f"{path} missing {user_col}/{item_col}")
    items = frame[item_col].astype(str)
    counts = Counter(items.tolist())
    return counts, int(sum(counts.values()))


def load_popularity_data(root: Path) -> tuple[dict[str, PopularityData], list[dict[str, Any]]]:
    stats = read_csv(root, PATHS["train_item_statistics"])
    popularity: dict[str, PopularityData] = {}
    source_rows: list[dict[str, Any]] = []

    for dataset_label, group in stats.groupby(stats["dataset"].map(canonical_dataset), sort=False):
        dataset = str(dataset_label)
        counts = {str(row["item_id"]): int(float(row["train_interaction_count"])) for _, row in group.iterrows()}
        total = int(float(group.iloc[0]["total_train_interactions"]))
        threshold = float(group.iloc[0]["long_tail_threshold"])
        long_tail, resolved_threshold = build_long_tail_set(counts, threshold=threshold)
        source_path = str(group.iloc[0].get("source_path", PATHS["train_item_statistics"].as_posix()))
        source_sha = str(group.iloc[0].get("source_sha256", ""))
        popularity[dataset] = PopularityData(
            dataset=dataset,
            train_counts=counts,
            total_train_interactions=total,
            train_seen_catalog=frozenset(counts),
            long_tail_items=frozenset(long_tail),
            long_tail_threshold=resolved_threshold,
            source_path=source_path,
            source_sha256=source_sha,
            source_policy="canonical_metric_train_item_statistics",
        )
        source_rows.append(
            {
                "dataset": dataset,
                "source_path": source_path,
                "source_sha256": source_sha,
                "source_policy": "canonical_metric_train_item_statistics",
                "train_seen_catalog_items": len(counts),
                "total_train_interactions": total,
                "long_tail_threshold": format_float(resolved_threshold),
            }
        )

    for dataset, path, item_col in [
        ("lastfm", PATHS["lastfm_train"], "item_id"),
        ("foodcom", PATHS["foodcom_train"], "recipe_id"),
    ]:
        if dataset in popularity:
            continue
        counts, total = load_train_source(
            root,
            path,
            user_col="user_id",
            item_col=item_col,
            split_col="split" if dataset == "lastfm" else None,
        )
        long_tail, threshold = build_long_tail_set(counts)
        source_path = path.as_posix()
        source_sha = sha256_file(root / path)
        popularity[dataset] = PopularityData(
            dataset=dataset,
            train_counts=dict(counts),
            total_train_interactions=total,
            train_seen_catalog=frozenset(counts),
            long_tail_items=frozenset(long_tail),
            long_tail_threshold=threshold,
            source_path=source_path,
            source_sha256=source_sha,
            source_policy="offline_train_only_counts_for_g2_metric_freeze",
        )
        source_rows.append(
            {
                "dataset": dataset,
                "source_path": source_path,
                "source_sha256": source_sha,
                "source_policy": "offline_train_only_counts_for_g2_metric_freeze",
                "train_seen_catalog_items": len(counts),
                "total_train_interactions": total,
                "long_tail_threshold": format_float(threshold),
            }
        )

    missing = sorted(set(DATASETS).difference(popularity))
    if missing:
        raise ValueError(f"missing popularity data for datasets: {missing}")
    return popularity, sorted(source_rows, key=lambda row: row["dataset"])


def load_g2_responses(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    with (root / PATHS["g2_responses"]).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            request_id = str(record.get("canonical_request_id") or record.get("request_id") or "")
            counts[request_id] += 1
            rows.append(
                {
                    "source_canonical_request_id": request_id,
                    "ranked_item_ids": str(record.get("ranked_item_ids", "")),
                    "terminal_state_evidence": str(record.get("terminal_state", "")),
                    "strict_parse_valid_evidence": bool_text(record.get("strict_parse_valid", False)),
                    "backend_success_evidence": bool_text(record.get("backend_success", False)),
                    "raw_response_hash_evidence": str(record.get("raw_response_hash", "")),
                    "jsonl_line": line_no,
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs["duplicate_ids"] = sum(1 for value in counts.values() if value > 1)
    return frame


def load_g1_parsing(root: Path) -> pd.DataFrame:
    frame = read_csv(root, PATHS["g1_strict_parsing"])
    frame = frame[
        [
            "canonical_request_id",
            "ranked_item_ids",
            "terminal_structural_status",
            "strict_parse_valid",
            "raw_response_hash",
        ]
    ].copy()
    frame = frame.rename(
        columns={
            "canonical_request_id": "source_canonical_request_id",
            "terminal_structural_status": "terminal_state_evidence",
            "strict_parse_valid": "strict_parse_valid_evidence",
            "raw_response_hash": "raw_response_hash_evidence",
        }
    )
    frame["strict_parse_valid_evidence"] = frame["strict_parse_valid_evidence"].map(bool_text)
    frame["backend_success_evidence"] = "true"
    return frame


def evidence_lookup(root: Path, failures: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    g2 = load_g2_responses(root)
    g1 = load_g1_parsing(root)
    if int(g2.attrs.get("duplicate_ids", 0)) != 0:
        failures.append("G2_RESPONSE_LEDGER_DUPLICATES")
    if g2["source_canonical_request_id"].nunique() != len(g2):
        failures.append("G2_RESPONSE_ID_UNIQUENESS_FAILED")
    if g1["source_canonical_request_id"].nunique() != len(g1):
        failures.append("G1_STRICT_PARSE_ID_UNIQUENESS_FAILED")
    combined = pd.concat(
        [
            g1.assign(evidence_source=G1_EVIDENCE),
            g2.assign(evidence_source=G2_EVIDENCE),
        ],
        ignore_index=True,
        sort=False,
    )
    if combined.duplicated(["evidence_source", "source_canonical_request_id"]).any():
        failures.append("COMBINED_EVIDENCE_DUPLICATES")
    return combined, {
        "g1_strict_parse_rows": int(len(g1)),
        "g2_response_rows": int(len(g2)),
        "g2_response_duplicate_ids": int(g2.attrs.get("duplicate_ids", 0)),
    }


def build_row_metrics(root: Path, failures: list[str]) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    universe = read_csv(root, PATHS["analysis_universe"])
    requests = read_csv(root, PATHS["request_universe"])[
        ["canonical_request_id", "candidate_items", "candidate_count", "matched_g1_request_id"]
    ].copy()
    g1_reuse = read_csv(root, PATHS["g1_reuse"])
    evidence, evidence_counts = evidence_lookup(root, failures)

    if len(universe) != 88050 or universe["canonical_request_id"].nunique() != 88050:
        failures.append("ANALYSIS_UNIVERSE_ROW_COUNT_OR_UNIQUENESS_FAILED")
    if len(requests) != 88050 or requests["canonical_request_id"].nunique() != 88050:
        failures.append("REQUEST_UNIVERSE_ROW_COUNT_OR_UNIQUENESS_FAILED")
    if len(g1_reuse) != 18660 or g1_reuse["g2_canonical_request_id"].nunique() != 18660:
        failures.append("G1_REUSE_COUNT_OR_UNIQUENESS_FAILED")
    if g1_reuse["reuse_status"].ne("G1_EXACT_REUSE_TERMINAL_ZERO_CALL").any():
        failures.append("G1_REUSE_STATUS_MISMATCH")
    if g1_reuse["ollama_call_required"].map(bool_value).any():
        failures.append("G1_REUSE_HAS_OLLAMA_REQUIRED_TRUE")

    merged = universe.merge(requests, on="canonical_request_id", how="left", validate="one_to_one")
    merged = merged.merge(
        evidence,
        on=["evidence_source", "source_canonical_request_id"],
        how="left",
        validate="many_to_one",
    )
    if merged["ranked_item_ids"].isna().any():
        failures.append("MISSING_RANKED_ITEM_EVIDENCE")
    if not merged["raw_response_hash"].fillna("").eq(merged["raw_response_hash_evidence"].fillna("")).all():
        failures.append("RAW_RESPONSE_HASH_MISMATCH")
    if not merged["strict_parse_valid"].map(bool_text).eq(merged["strict_parse_valid_evidence"].map(bool_text)).all():
        failures.append("STRICT_PARSE_VALID_MISMATCH")
    if not merged["terminal_state"].fillna("").eq(merged["terminal_state_evidence"].fillna("")).all():
        failures.append("TERMINAL_STATE_MISMATCH")

    popularity, popularity_sources = load_popularity_data(root)
    records: list[dict[str, Any]] = []
    candidate_validation_failures = 0
    strict_short_rankings = 0
    strict_top5_missing_popularity_rows = 0
    strict_positive_absent_rows = 0

    for row in merged.to_dict("records"):
        request_id = str(row["canonical_request_id"])
        dataset = canonical_dataset(str(row["dataset"]))
        candidates = parse_json_array(row["candidate_items"])
        if str(row.get("candidate_count", "")) and len(candidates) != int(float(row["candidate_count"])):
            candidate_validation_failures += 1
        try:
            validate_one_positive(candidates, row["positive_item_id"])
        except ValueError:
            candidate_validation_failures += 1

        strict_valid = bool_value(row["strict_parse_valid"])
        ranked_items: list[str] = []
        top5: list[str] = []
        rank_zero: int | None = None
        hit_at_5 = 0.0
        ndcg_at_5 = 0.0
        mrr = 0.0
        duplicate_rate = math.nan
        candidate_validity = math.nan
        long_tail_values: list[float] = []
        popularity_values: list[float] = []
        novelty_values: list[float] = []
        top5_missing_popularity_count = 0

        if strict_valid:
            ranked_items = parse_json_array(row["ranked_item_ids"])
            if len(ranked_items) < TOP_K:
                strict_short_rankings += 1
            top5 = ranked_items[:TOP_K]
            rank_zero = first_positive_rank_zero_based(ranked_items, row["positive_item_id"])
            if rank_zero is None:
                strict_positive_absent_rows += 1
            hit_at_5 = hr_at_k_from_rank(rank_zero, TOP_K)
            ndcg_at_5 = ndcg_at_k_from_rank(rank_zero, TOP_K)
            mrr = reciprocal_rank(rank_zero)
            duplicate_rate = duplicate_rate_at_k(top5, TOP_K)
            candidate_validity = candidate_validity_rate_at_k(top5, candidates, k=TOP_K)
            pdata = popularity[dataset]
            counts = [int(pdata.train_counts.get(str(item), 0)) for item in top5]
            top5_missing_popularity_count = sum(1 for count in counts if count <= 0)
            if top5_missing_popularity_count:
                strict_top5_missing_popularity_rows += 1
            long_tail_values = [1.0 if str(item) in pdata.long_tail_items else 0.0 for item in top5]
            popularity_values = [float(count) for count in counts]
            if top5_missing_popularity_count == 0:
                novelty_values = [
                    novelty_self_information(count, pdata.total_train_interactions) for count in counts
                ]

        invalid_class = str(row["invalid_class"])
        records.append(
            {
                "canonical_request_id": request_id,
                "dataset": dataset,
                "evidence_source": str(row["evidence_source"]),
                "strict_parse_valid": bool_text(strict_valid),
                "backend_success": bool_text(row["backend_success"]),
                "terminal_state": str(row["terminal_state"]),
                "invalid_class": invalid_class,
                "rank_zero_based": "" if rank_zero is None else str(rank_zero),
                "positive_absent_from_ranked_output": bool_text(strict_valid and rank_zero is None),
                "hr_at_5": hit_at_5,
                "ndcg_at_5": ndcg_at_5,
                "mrr": mrr,
                "top5_items": top5,
                "candidate_items": candidates,
                "duplicate_rate_at_5": duplicate_rate,
                "candidate_validity_rate": candidate_validity,
                "long_tail_values": long_tail_values,
                "popularity_values": popularity_values,
                "novelty_values": novelty_values,
                "top5_missing_popularity_count": top5_missing_popularity_count,
                "invalid_length": bool_text(invalid_class == "INVALID_LENGTH"),
                "invalid_duplicate": bool_text(invalid_class == "INVALID_DUPLICATE"),
                "invalid_candidate_reference": bool_text(invalid_class in INVALID_CANDIDATE_CLASSES),
            }
        )

    if candidate_validation_failures:
        failures.append(f"CANDIDATE_VALIDATION_FAILURES:{candidate_validation_failures}")
    if strict_short_rankings:
        failures.append(f"STRICT_VALID_WITH_SHORT_TOP5:{strict_short_rankings}")

    row_metrics = pd.DataFrame(records)
    return row_metrics, {
        **evidence_counts,
        "strict_positive_absent_rows": strict_positive_absent_rows,
        "strict_top5_missing_popularity_rows": strict_top5_missing_popularity_rows,
    }, popularity_sources


def metric_contract_map() -> dict[str, dict[str, str]]:
    registry = build_metric_registry()
    validate_metric_registry(registry)
    return {
        contract.metric_id: {
            "metric_family": contract.metric_family,
            "cutoff": contract.cutoff,
            "contract_status": contract.contract_status,
            "higher_is_better": contract.higher_is_better,
            "denominator_definition": contract.denominator_definition,
            "blocking_reason": contract.blocking_reason,
        }
        for contract in registry
    }


def metric_metadata(metric_id: str, contracts: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    if metric_id in contracts:
        row = dict(contracts[metric_id])
        row["direction_source"] = "canonical_metric_contract"
        return row
    if metric_id == "mrr":
        return {
            "metric_family": "utility",
            "cutoff": "all",
            "contract_status": "ready_reconstructed_g1_semantics",
            "higher_is_better": "true",
            "denominator_definition": "planned/applicable users; reciprocal rank over ranked output; absent positive -> 0",
            "blocking_reason": "",
            "direction_source": "frozen_g1_runner_and_official_calMRR",
        }
    if metric_id in RELIABILITY_METRICS:
        return {
            "metric_family": "reliability",
            "cutoff": "not_applicable",
            "contract_status": "g2_protocol_reliability_metric",
            "higher_is_better": "false" if metric_id.startswith("invalid") or metric_id == "duplicate_index_rate" else "true",
            "denominator_definition": "planned/applicable users",
            "blocking_reason": "",
            "direction_source": "g2_history_length_sensitivity_yaml",
        }
    raise ValueError(f"unknown metric metadata: {metric_id}")


def flatten(items: Iterable[Sequence[str]]) -> list[str]:
    return [str(item) for seq in items for item in list(seq)]


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot aggregate empty values")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def metric_row(
    *,
    membership: str,
    dataset: str,
    model: str,
    prompt: str,
    history_length: int,
    metric_id: str,
    value: float | None,
    status: str,
    planned: int,
    strict_valid: int,
    invalid: int,
    defined_users: int,
    undefined_users: int,
    denominator_positions: int,
    item_numerator: int | str,
    item_denominator: int | str,
    g1_reuse: int,
    g2_terminal: int,
    contracts: Mapping[str, Mapping[str, str]],
    notes: str,
) -> dict[str, str]:
    meta = metric_metadata(metric_id, contracts)
    return {
        "result_id": stable_id("G2MET", membership, dataset, model, prompt, history_length, metric_id),
        "phase_id": PHASE_ID,
        "analysis_membership": membership,
        "dataset": dataset,
        "model": model,
        "prompt_strategy": prompt,
        "history_length": str(history_length),
        "metric_id": metric_id,
        "metric_family": meta["metric_family"],
        "k": "" if meta["cutoff"] in {"all", "not_applicable"} else meta["cutoff"],
        "value": "" if value is None else format_float(value),
        "status": status,
        "planned_applicable_users": str(planned),
        "strict_valid_users": str(strict_valid),
        "invalid_users": str(invalid),
        "defined_user_count": str(defined_users),
        "undefined_user_count": str(undefined_users),
        "denominator_positions": str(denominator_positions),
        "item_numerator_count": str(item_numerator),
        "item_denominator_count": str(item_denominator),
        "g1_exact_reuse_rows": str(g1_reuse),
        "g2_terminal_rows": str(g2_terminal),
        "contract_status": meta["contract_status"],
        "higher_is_better": meta["higher_is_better"],
        "direction_source": meta["direction_source"],
        "denominator_definition": meta["denominator_definition"],
        "notes": notes,
    }


def denominator_policy(metric_id: str) -> str:
    if metric_id in UTILITY_METRICS:
        return "UTILITY_PLANNED_APPLICABLE_INVALID_AS_ZERO"
    if metric_id == "novelty_at_5":
        return "BEYOND_STRICT_VALID_NO_SMOOTHING_POSITIVE_TRAIN_SUPPORT_SUBSET"
    if metric_id in {"catalog_coverage_at_5", "item_coverage_at_5"}:
        return "BEYOND_STRICT_VALID_COHORT_LEVEL_COVERAGE"
    if metric_id in {"long_tail_rate_at_5", "average_popularity_at_5"}:
        return "BEYOND_STRICT_VALID_TOP5_POSITION_MICRO_MEAN"
    if metric_id in STRUCTURAL_METRICS:
        return "STRUCTURAL_STRICT_VALID_USERS"
    if metric_id in RELIABILITY_METRICS:
        return "RELIABILITY_PLANNED_APPLICABLE_USERS"
    return "NOT_APPLICABLE"


def denominator_notes(metric_id: str) -> str:
    if metric_id in UTILITY_METRICS:
        return "planned_applicable_users is the analytic denominator; invalid terminal evidence contributes zero utility"
    if metric_id == "novelty_at_5":
        return "strict-valid rows only; defined_user_count excludes rows with any zero train-count top-5 item"
    if metric_id == "catalog_coverage_at_5":
        return "strict-valid rows define exposures; item_denominator_count is the dataset train-seen catalog"
    if metric_id == "item_coverage_at_5":
        return "strict-valid rows define exposures; item_denominator_count is the strict-valid cell candidate union"
    if metric_id in {"long_tail_rate_at_5", "average_popularity_at_5"}:
        return "strict-valid top-5 positions define the denominator"
    if metric_id in STRUCTURAL_METRICS:
        return "strict-valid rows define the user denominator and top-5 positions define denominator_positions"
    if metric_id in RELIABILITY_METRICS:
        return "planned_applicable_users is the analytic denominator"
    return ""


def compute_metrics(root: Path, failures: list[str]) -> dict[str, Any]:
    head_before = git_output(root, ["rev-parse", "HEAD"])
    status_before = git_output(root, ["status", "--short", "--untracked-files=all"], check=False)
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=root, text=True, capture_output=True)

    manifests = {
        "protocol": read_json(root, PATHS["protocol_manifest"]),
        "execution": read_json(root, PATHS["execution_manifest"]),
        "retention": read_json(root, PATHS["retention_manifest"]),
        "analysis": read_json(root, PATHS["analysis_manifest"]),
    }
    universe = read_csv(root, PATHS["analysis_universe"])
    membership = read_csv(root, PATHS["membership_universe"])
    cell_counts = read_csv(root, PATHS["cell_counts"]).rename(columns={"reporting_view": "analysis_membership"})
    for frame in [membership, cell_counts]:
        frame["history_length"] = pd.to_numeric(frame["history_length"], errors="raise").astype(int)
    cell_counts["planned_applicable_users"] = pd.to_numeric(
        cell_counts["planned_applicable_users"], errors="raise"
    ).astype(int)
    applicable_cells = cell_counts.loc[cell_counts["planned_applicable_users"].gt(0)].copy()

    row_metrics, evidence_counts, popularity_sources = build_row_metrics(root, failures)
    work = membership.merge(
        row_metrics,
        on="canonical_request_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_row"),
    )
    if work["hr_at_5"].isna().any():
        failures.append("MEMBERSHIP_ROW_METRIC_JOIN_MISSING")

    group_cols = ["analysis_membership", "dataset", "model", "prompt_strategy", "history_length"]
    expected_keys = {
        tuple(row[col] for col in group_cols)
        for row in applicable_cells[group_cols + ["planned_applicable_users"]].to_dict("records")
    }
    actual_keys = {tuple(key) for key, _ in work.groupby(group_cols, sort=False)}
    missing_keys = sorted(expected_keys.difference(actual_keys))
    extra_keys = sorted(actual_keys.difference(expected_keys))
    if missing_keys:
        failures.append(f"MISSING_METRIC_CELL_KEYS:{len(missing_keys)}")
    if extra_keys:
        failures.append(f"EXTRA_METRIC_CELL_KEYS:{len(extra_keys)}")

    contracts = metric_contract_map()
    expected_lookup = {
        tuple(row[col] for col in group_cols): row
        for row in applicable_cells[group_cols + ["planned_applicable_users", "strict_valid_rows", "invalid_rows"]].to_dict(
            "records"
        )
    }
    popularity, _ = load_popularity_data(root)
    result_rows: list[dict[str, str]] = []
    denominator_rows: list[dict[str, str]] = []
    undefined_rows: list[dict[str, str]] = []

    for key, group in work.groupby(group_cols, sort=True, dropna=False):
        membership_name, dataset, model, prompt, history_length = key
        expected = expected_lookup.get(key)
        if expected is None:
            continue
        planned = int(expected["planned_applicable_users"])
        strict_valid = int(group["strict_parse_valid"].map(bool_value).sum())
        invalid = int(planned - strict_valid)
        if len(group) != planned:
            failures.append(f"PLANNED_USER_DENOMINATOR_MISMATCH:{key}:{len(group)}:{planned}")
        if strict_valid != int(expected["strict_valid_rows"]):
            failures.append(f"STRICT_VALID_DENOMINATOR_MISMATCH:{key}:{strict_valid}:{expected['strict_valid_rows']}")
        if invalid != int(expected["invalid_rows"]):
            failures.append(f"INVALID_DENOMINATOR_MISMATCH:{key}:{invalid}:{expected['invalid_rows']}")

        g1_reuse = int(group["evidence_source"].eq(G1_EVIDENCE).sum())
        g2_terminal = int(group["evidence_source"].eq(G2_EVIDENCE).sum())
        valid_group = group.loc[group["strict_parse_valid"].map(bool_value)].copy()
        pdata = popularity[str(dataset)]

        for metric_id in UTILITY_METRICS:
            value = mean([float(value) for value in group[metric_id].tolist()])
            row = metric_row(
                membership=str(membership_name),
                dataset=str(dataset),
                model=str(model),
                prompt=str(prompt),
                history_length=int(history_length),
                metric_id=metric_id,
                value=value,
                status="COMPUTED",
                planned=planned,
                strict_valid=strict_valid,
                invalid=invalid,
                defined_users=planned,
                undefined_users=0,
                denominator_positions=planned,
                item_numerator="",
                item_denominator="",
                g1_reuse=g1_reuse,
                g2_terminal=g2_terminal,
                contracts=contracts,
                notes="utility denominator is planned/applicable users; non-strict terminal responses are zero utility",
            )
            result_rows.append(row)
            denominator_rows.append(row)

        if strict_valid > 0:
            recommended_items = flatten(valid_group["top5_items"].tolist())
            candidate_union = set(flatten(valid_group["candidate_items"].tolist()))

            novelty_defined = valid_group.loc[
                pd.to_numeric(valid_group["top5_missing_popularity_count"], errors="raise").eq(0)
            ]
            novelty_values = flatten(novelty_defined["novelty_values"].tolist())
            novelty_value = mean([float(value) for value in novelty_values]) if novelty_values else None
            novelty_row = metric_row(
                membership=str(membership_name),
                dataset=str(dataset),
                model=str(model),
                prompt=str(prompt),
                history_length=int(history_length),
                metric_id="novelty_at_5",
                value=novelty_value,
                status="COMPUTED" if novelty_value is not None else "UNDEFINED_NO_POSITIVE_TRAIN_SUPPORT",
                planned=planned,
                strict_valid=strict_valid,
                invalid=invalid,
                defined_users=int(len(novelty_defined)),
                undefined_users=int(strict_valid - len(novelty_defined)),
                denominator_positions=int(len(novelty_values)),
                item_numerator="",
                item_denominator="",
                g1_reuse=g1_reuse,
                g2_terminal=g2_terminal,
                contracts=contracts,
                notes="strict-valid rows only; rows with any zero train-count top-5 item are excluded for no-smoothing novelty",
            )
            result_rows.append(novelty_row)
            denominator_rows.append(novelty_row)
            if int(strict_valid - len(novelty_defined)) > 0:
                novelty_undefined_row = dict(novelty_row)
                novelty_undefined_row["status"] = "DEFINED_SUBSET_USED"
                novelty_undefined_row["notes"] = "computed over metric-specific defined subset; see result row for value"
                novelty_undefined_row["undefined_reason"] = (
                    "zero_train_popularity_support_for_at_least_one_top5_item"
                )
                undefined_rows.append(
                    novelty_undefined_row
                )

            catalog_value = catalog_coverage_ratio(recommended_items, pdata.train_seen_catalog)
            catalog_numerator = len(set(recommended_items).intersection(pdata.train_seen_catalog))
            row = metric_row(
                membership=str(membership_name),
                dataset=str(dataset),
                model=str(model),
                prompt=str(prompt),
                history_length=int(history_length),
                metric_id="catalog_coverage_at_5",
                value=catalog_value,
                status="COMPUTED",
                planned=planned,
                strict_valid=strict_valid,
                invalid=invalid,
                defined_users=strict_valid,
                undefined_users=0,
                denominator_positions=strict_valid * TOP_K,
                item_numerator=catalog_numerator,
                item_denominator=len(pdata.train_seen_catalog),
                g1_reuse=g1_reuse,
                g2_terminal=g2_terminal,
                contracts=contracts,
                notes="cohort-level unique top-5 items over train-seen catalog; strict-valid rows only",
            )
            result_rows.append(row)
            denominator_rows.append(row)

            item_value = len(set(recommended_items).intersection(candidate_union)) / len(candidate_union)
            row = metric_row(
                membership=str(membership_name),
                dataset=str(dataset),
                model=str(model),
                prompt=str(prompt),
                history_length=int(history_length),
                metric_id="item_coverage_at_5",
                value=item_value,
                status="COMPUTED",
                planned=planned,
                strict_valid=strict_valid,
                invalid=invalid,
                defined_users=strict_valid,
                undefined_users=0,
                denominator_positions=strict_valid * TOP_K,
                item_numerator=len(set(recommended_items).intersection(candidate_union)),
                item_denominator=len(candidate_union),
                g1_reuse=g1_reuse,
                g2_terminal=g2_terminal,
                contracts=contracts,
                notes="cohort-level unique top-5 items over strict-valid cell candidate union",
            )
            result_rows.append(row)
            denominator_rows.append(row)

            long_tail_values = flatten(valid_group["long_tail_values"].tolist())
            row = metric_row(
                membership=str(membership_name),
                dataset=str(dataset),
                model=str(model),
                prompt=str(prompt),
                history_length=int(history_length),
                metric_id="long_tail_rate_at_5",
                value=mean([float(value) for value in long_tail_values]),
                status="COMPUTED",
                planned=planned,
                strict_valid=strict_valid,
                invalid=invalid,
                defined_users=strict_valid,
                undefined_users=0,
                denominator_positions=len(long_tail_values),
                item_numerator=int(sum(float(value) for value in long_tail_values)),
                item_denominator=len(long_tail_values),
                g1_reuse=g1_reuse,
                g2_terminal=g2_terminal,
                contracts=contracts,
                notes=f"strict-valid top-5 positions; long-tail threshold={format_float(pdata.long_tail_threshold)}",
            )
            result_rows.append(row)
            denominator_rows.append(row)

            popularity_values = flatten(valid_group["popularity_values"].tolist())
            row = metric_row(
                membership=str(membership_name),
                dataset=str(dataset),
                model=str(model),
                prompt=str(prompt),
                history_length=int(history_length),
                metric_id="average_popularity_at_5",
                value=mean([float(value) for value in popularity_values]),
                status="COMPUTED",
                planned=planned,
                strict_valid=strict_valid,
                invalid=invalid,
                defined_users=strict_valid,
                undefined_users=0,
                denominator_positions=len(popularity_values),
                item_numerator="",
                item_denominator="",
                g1_reuse=g1_reuse,
                g2_terminal=g2_terminal,
                contracts=contracts,
                notes="strict-valid top-5 positions; zero train-count items contribute zero, matching frozen ARP semantics",
            )
            result_rows.append(row)
            denominator_rows.append(row)

            for metric_id in STRUCTURAL_METRICS:
                value = mean([float(value) for value in valid_group[metric_id].tolist()])
                row = metric_row(
                    membership=str(membership_name),
                    dataset=str(dataset),
                    model=str(model),
                    prompt=str(prompt),
                    history_length=int(history_length),
                    metric_id=metric_id,
                    value=value,
                    status="COMPUTED",
                    planned=planned,
                    strict_valid=strict_valid,
                    invalid=invalid,
                    defined_users=strict_valid,
                    undefined_users=0,
                    denominator_positions=strict_valid * TOP_K,
                    item_numerator="",
                    item_denominator="",
                    g1_reuse=g1_reuse,
                    g2_terminal=g2_terminal,
                    contracts=contracts,
                    notes="strict-valid rows only; structural quality metric from frozen canonical contract",
                )
                result_rows.append(row)
                denominator_rows.append(row)
        else:
            for metric_id in BEYOND_UTILITY_METRICS + STRUCTURAL_METRICS:
                row = metric_row(
                    membership=str(membership_name),
                    dataset=str(dataset),
                    model=str(model),
                    prompt=str(prompt),
                    history_length=int(history_length),
                    metric_id=metric_id,
                    value=None,
                    status="UNDEFINED_NO_STRICT_VALID_ROWS",
                    planned=planned,
                    strict_valid=strict_valid,
                    invalid=invalid,
                    defined_users=0,
                    undefined_users=0,
                    denominator_positions=0,
                    item_numerator="",
                    item_denominator="",
                    g1_reuse=g1_reuse,
                    g2_terminal=g2_terminal,
                    contracts=contracts,
                    notes="beyond-utility and structural denominators require strict-valid rows",
                )
                result_rows.append(row)
                denominator_rows.append(row)
                undefined_rows.append({**row, "undefined_reason": "no_strict_valid_rows"})

        reliability_values = {
            "backend_success_rate": [1.0 if bool_value(value) else 0.0 for value in group["backend_success"].tolist()],
            "strict_parse_valid_rate": [1.0 if bool_value(value) else 0.0 for value in group["strict_parse_valid"].tolist()],
            "invalid_length_rate": [1.0 if bool_value(value) else 0.0 for value in group["invalid_length"].tolist()],
            "duplicate_index_rate": [1.0 if bool_value(value) else 0.0 for value in group["invalid_duplicate"].tolist()],
            "invalid_candidate_reference_rate": [
                1.0 if bool_value(value) else 0.0 for value in group["invalid_candidate_reference"].tolist()
            ],
        }
        for metric_id, values in reliability_values.items():
            row = metric_row(
                membership=str(membership_name),
                dataset=str(dataset),
                model=str(model),
                prompt=str(prompt),
                history_length=int(history_length),
                metric_id=metric_id,
                value=mean(values),
                status="COMPUTED",
                planned=planned,
                strict_valid=strict_valid,
                invalid=invalid,
                defined_users=planned,
                undefined_users=0,
                denominator_positions=planned,
                item_numerator="",
                item_denominator="",
                g1_reuse=g1_reuse,
                g2_terminal=g2_terminal,
                contracts=contracts,
                notes="reliability denominator is planned/applicable users",
            )
            result_rows.append(row)
            denominator_rows.append(row)

        for metric_id in BLOCKED_BEYOND_UTILITY_METRICS:
            meta = metric_metadata(metric_id, contracts)
            undefined_rows.append(
                {
                    "result_id": stable_id("G2MET-BLOCKED", membership_name, dataset, model, prompt, history_length, metric_id),
                    "phase_id": PHASE_ID,
                    "analysis_membership": str(membership_name),
                    "dataset": str(dataset),
                    "model": str(model),
                    "prompt_strategy": str(prompt),
                    "history_length": str(history_length),
                    "metric_id": metric_id,
                    "metric_family": meta["metric_family"],
                    "k": meta["cutoff"],
                    "value": "",
                    "status": "BLOCKED_BY_FROZEN_CONTRACT",
                    "planned_applicable_users": str(planned),
                    "strict_valid_users": str(strict_valid),
                    "invalid_users": str(invalid),
                    "defined_user_count": "0",
                    "undefined_user_count": str(strict_valid),
                    "denominator_positions": "0",
                    "item_numerator_count": "",
                    "item_denominator_count": "",
                    "g1_exact_reuse_rows": str(g1_reuse),
                    "g2_terminal_rows": str(g2_terminal),
                    "contract_status": meta["contract_status"],
                    "higher_is_better": meta["higher_is_better"],
                    "direction_source": meta["direction_source"],
                    "denominator_definition": meta["denominator_definition"],
                    "notes": "not computed; no substitute formula assigned",
                    "undefined_reason": meta["blocking_reason"],
                }
            )

    results = pd.DataFrame(result_rows).sort_values(
        ["analysis_membership", "dataset", "model", "prompt_strategy", "history_length", "metric_id"],
        kind="mergesort",
    )
    denominators = pd.DataFrame(denominator_rows).sort_values(
        ["analysis_membership", "dataset", "model", "prompt_strategy", "history_length", "metric_id"],
        kind="mergesort",
    )
    denominators = denominators[
        [
            "result_id",
            "phase_id",
            "analysis_membership",
            "dataset",
            "model",
            "prompt_strategy",
            "history_length",
            "metric_id",
            "metric_family",
            "k",
            "status",
            "planned_applicable_users",
            "strict_valid_users",
            "invalid_users",
            "defined_user_count",
            "undefined_user_count",
            "denominator_positions",
            "item_numerator_count",
            "item_denominator_count",
            "g1_exact_reuse_rows",
            "g2_terminal_rows",
            "denominator_definition",
            "notes",
        ]
    ].copy()
    denominators["denominator_policy_id"] = denominators["metric_id"].map(denominator_policy)
    denominators["denominator_semantics"] = denominators["metric_id"].map(denominator_notes)
    undefined = pd.DataFrame(undefined_rows).sort_values(
        ["analysis_membership", "dataset", "model", "prompt_strategy", "history_length", "metric_id"],
        kind="mergesort",
    )

    metric_cell_counts = results.groupby("metric_id", sort=True)["result_id"].count().astype(int).to_dict()
    for metric_id in COMPUTED_METRICS:
        if metric_cell_counts.get(metric_id, 0) != len(applicable_cells):
            failures.append(f"COMPUTED_METRIC_CELL_COUNT_MISMATCH:{metric_id}:{metric_cell_counts.get(metric_id, 0)}")

    utility_denominator_bad = results.loc[
        results["metric_id"].isin(UTILITY_METRICS)
        & results["defined_user_count"].ne(results["planned_applicable_users"])
    ]
    if not utility_denominator_bad.empty:
        failures.append(f"UTILITY_DENOMINATOR_POLICY_MISMATCH:{len(utility_denominator_bad)}")
    beyond_rows = results.loc[results["metric_id"].isin(BEYOND_UTILITY_METRICS + STRUCTURAL_METRICS)]
    beyond_bad = beyond_rows.loc[
        beyond_rows["status"].eq("COMPUTED")
        & (
            pd.to_numeric(beyond_rows["defined_user_count"], errors="raise")
            > pd.to_numeric(beyond_rows["strict_valid_users"], errors="raise")
        )
    ]
    if not beyond_bad.empty:
        failures.append(f"BEYOND_DENOMINATOR_POLICY_MISMATCH:{len(beyond_bad)}")

    observed_levels = {
        view: sorted(
            int(value)
            for value in work.loc[work["analysis_membership"].eq(view), "history_length"].drop_duplicates().tolist()
        )
        for view in REPORTING_VIEWS
    }
    for view, expected_levels in HISTORY_BY_VIEW.items():
        if observed_levels.get(view, []) != expected_levels:
            failures.append(f"HISTORY_LEVEL_MISMATCH:{view}:{observed_levels.get(view, [])}")

    universe_counts = {
        "logical_requests": int(len(universe)),
        "g1_exact_reuse": int(universe["evidence_source"].eq(G1_EVIDENCE).sum()),
        "g2_terminal_evidence": int(universe["evidence_source"].eq(G2_EVIDENCE).sum()),
        "strict_valid": int(universe["strict_parse_valid"].map(bool_value).sum()),
        "invalid": int((~universe["strict_parse_valid"].map(bool_value)).sum()),
        "backend_errors": int(universe["terminal_state"].eq("BACKEND_FAILURE").sum()),
        "missing": int(universe["missing_evidence"].map(bool_value).sum()),
        "duplicate_terminal": int(universe["duplicate_terminal_evidence"].map(bool_value).sum()),
        "ambiguous": int(universe["ambiguous_evidence"].map(bool_value).sum()),
    }
    for name, expected_value in [
        ("logical_requests", 88050),
        ("g1_exact_reuse", 18660),
        ("g2_terminal_evidence", 69390),
        ("strict_valid", 86768),
        ("invalid", 1282),
        ("backend_errors", 0),
        ("missing", 0),
        ("duplicate_terminal", 0),
        ("ambiguous", 0),
    ]:
        if universe_counts[name] != expected_value:
            failures.append(f"UNIVERSE_COUNT_MISMATCH:{name}:{universe_counts[name]}:{expected_value}")

    validation_checks = {
        "head_before_matches_precondition": head_before == EXPECTED_HEAD,
        "git_clean_or_expected_metric_paths_before_generation": status_is_clean_or_expected(
            status_before, allow_generator_dirty=True
        ),
        "git_diff_check_clean_before_generation": diff_check.returncode == 0,
        "protocol_manifest_frozen": manifests["protocol"].get("decision_token")
        == "G2_NESTED_HISTORY_PROTOCOL_FROZEN",
        "execution_manifest_frozen": manifests["execution"].get("decision_token") == "G2_EXECUTION_RUNNER_FROZEN",
        "retention_manifest_frozen": manifests["retention"].get("decision_token") == "G2_EXECUTION_RETENTION_FROZEN",
        "analysis_manifest_frozen": manifests["analysis"].get("decision_token") == "G2_ANALYTIC_UNIVERSE_FROZEN",
        "analysis_universe_rows_match": universe_counts["logical_requests"] == 88050,
        "g1_reuse_rows_match": universe_counts["g1_exact_reuse"] == 18660,
        "g2_terminal_rows_match": universe_counts["g2_terminal_evidence"] == 69390,
        "strict_valid_total_match": universe_counts["strict_valid"] == 86768,
        "invalid_total_match": universe_counts["invalid"] == 1282,
        "backend_errors_zero": universe_counts["backend_errors"] == 0,
        "missing_duplicate_ambiguous_zero": universe_counts["missing"] == 0
        and universe_counts["duplicate_terminal"] == 0
        and universe_counts["ambiguous"] == 0,
        "expected_metric_cells_present": not missing_keys and not extra_keys,
        "computed_metric_rows_complete": all(
            metric_cell_counts.get(metric_id, 0) == len(applicable_cells) for metric_id in COMPUTED_METRICS
        ),
        "blocked_beyond_utility_metrics_not_substituted": set(BLOCKED_BEYOND_UTILITY_METRICS).issubset(
            set(undefined["metric_id"].drop_duplicates().tolist())
        ),
        "utility_denominator_policy_valid": utility_denominator_bad.empty,
        "beyond_utility_denominator_policy_valid": beyond_bad.empty,
        "no_ollama_calls_executed": True,
        "g3_not_started": True,
    }
    for check, passed in validation_checks.items():
        if not passed:
            failures.append(f"VALIDATION_CHECK_FALSE:{check}")

    validation = pd.DataFrame(
        [
            {
                "check_id": stable_id("G2MET-CHK", check),
                "check_name": check,
                "status": "PASS" if passed else "FAIL",
                "expected": "true",
                "actual": bool_text(passed),
                "notes": "",
            }
            for check, passed in validation_checks.items()
        ]
    )

    membership_summary = (
        work.groupby("analysis_membership", sort=True)
        .agg(
            logical_membership_rows=("canonical_request_id", "count"),
            cells=("canonical_request_id", lambda series: int(work.loc[series.index, group_cols].drop_duplicates().shape[0])),
            g1_exact_reuse_rows=("evidence_source", lambda series: int(series.eq(G1_EVIDENCE).sum())),
            g2_terminal_rows=("evidence_source", lambda series: int(series.eq(G2_EVIDENCE).sum())),
            strict_valid_rows=("strict_parse_valid", lambda series: int(series.map(bool_value).sum())),
        )
        .reset_index()
    )
    membership_summary["invalid_rows"] = (
        membership_summary["logical_membership_rows"].astype(int) - membership_summary["strict_valid_rows"].astype(int)
    )

    result_summary = {
        "applicable_cells": int(len(applicable_cells)),
        "metric_result_rows": int(len(results)),
        "metric_denominator_rows": int(len(denominators)),
        "undefined_rows": int(len(undefined)),
        "computed_metric_ids": list(COMPUTED_METRICS),
        "blocked_metric_ids": list(BLOCKED_BEYOND_UTILITY_METRICS),
        "metric_cell_counts": {str(k): int(v) for k, v in metric_cell_counts.items()},
        "history_levels_by_membership": observed_levels,
        "strict_positive_absent_rows": int(evidence_counts["strict_positive_absent_rows"]),
        "strict_top5_missing_popularity_rows": int(evidence_counts["strict_top5_missing_popularity_rows"]),
        "undefined_status_counts": undefined["status"].value_counts().sort_index().astype(int).to_dict(),
    }
    input_hashes = {
        path.as_posix(): sha256_file(root / path)
        for path in [
            PATHS["protocol_config"],
            PATHS["execution_config"],
            PATHS["runner"],
            PATHS["protocol_manifest"],
            PATHS["execution_manifest"],
            PATHS["retention_manifest"],
            PATHS["analysis_manifest"],
            PATHS["request_universe"],
            PATHS["analysis_universe"],
            PATHS["membership_universe"],
            PATHS["cell_counts"],
            PATHS["denominator_contract"],
            PATHS["g1_reuse"],
            PATHS["g1_strict_parsing"],
            PATHS["g2_responses"],
            PATHS["train_item_statistics"],
            PATHS["lastfm_train"],
            PATHS["foodcom_train"],
            PATHS["metric_contract_source"],
            PATHS["metric_computation_source"],
        ]
    }

    return {
        "head_before": head_before,
        "status_before": status_before,
        "diff_check_stdout": diff_check.stdout.strip(),
        "diff_check_stderr": diff_check.stderr.strip(),
        "decision_token": DECISION_FROZEN if not failures else DECISION_BLOCKED,
        "failures": failures,
        "tables": {
            "metric_results": results,
            "metric_denominators": denominators,
            "metric_validation": validation,
            "metric_undefined": undefined,
        },
        "manifest_parts": {
            "universe_counts": universe_counts,
            "evidence_counts": evidence_counts,
            "membership_summary": membership_summary.to_dict("records"),
            "result_summary": result_summary,
            "validation_checks": validation_checks,
            "popularity_sources": popularity_sources,
            "input_hashes": input_hashes,
            "missing_metric_cell_keys": [list(key) for key in missing_keys[:20]],
            "extra_metric_cell_keys": [list(key) for key in extra_keys[:20]],
        },
    }


def build_report(parts: Mapping[str, Any], output_hashes: Mapping[str, str]) -> str:
    manifest_parts = parts["manifest_parts"]
    result_summary = manifest_parts["result_summary"]
    universe_counts = manifest_parts["universe_counts"]
    membership_summary = manifest_parts["membership_summary"]
    validation = parts["tables"]["metric_validation"]
    undefined = parts["tables"]["metric_undefined"]

    lines = [
        "# G2 History-Length Metrics Validation",
        "",
        f"- Phase: `{PHASE_ID}`",
        f"- Decision: `{parts['decision_token']}`",
        f"- HEAD before generation: `{parts['head_before']}`",
        "- Scope: frozen G2 analytic universe and retained G1/G2 evidence only; zero Ollama calls; no G3.",
        "",
        "## Accounting",
        "",
        f"- Logical requests: `{universe_counts['logical_requests']}`",
        f"- G1 exact reuse: `{universe_counts['g1_exact_reuse']}`",
        f"- G2 terminal evidence: `{universe_counts['g2_terminal_evidence']}`",
        f"- Strict-valid: `{universe_counts['strict_valid']}`",
        f"- Invalid terminal evidence: `{universe_counts['invalid']}`",
        f"- Backend errors: `{universe_counts['backend_errors']}`",
        f"- Missing/duplicate/ambiguous evidence: `{universe_counts['missing']}` / `{universe_counts['duplicate_terminal']}` / `{universe_counts['ambiguous']}`",
        "",
        "## Cells And Metrics",
        "",
        f"- Applicable cells: `{result_summary['applicable_cells']}`",
        f"- Computed metric rows: `{result_summary['metric_result_rows']}`",
        f"- Denominator rows: `{result_summary['metric_denominator_rows']}`",
        f"- Undefined/blocked rows: `{result_summary['undefined_rows']}`",
        f"- Computed metrics: `{';'.join(result_summary['computed_metric_ids'])}`",
        f"- Blocked by frozen contract, not substituted: `{';'.join(result_summary['blocked_metric_ids'])}`",
        "",
        "## Denominator Contract",
        "",
        "- Utility: planned/applicable users; non-strict terminal responses count as zero utility.",
        "- Beyond-utility: strict-valid users; novelty additionally uses a no-smoothing positive-train-support subset.",
        "- Coverage: strict-valid users define the exposure set; catalog coverage denominator is train-seen catalog; item coverage denominator is the strict-valid cell candidate union.",
        "- Structural quality: strict-valid users.",
        "- Reliability: planned/applicable users.",
        "",
        "## Membership Summary",
        "",
        "| membership | rows | cells | G1 reuse | G2 terminal | strict-valid | invalid |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in membership_summary:
        lines.append(
            "| {analysis_membership} | {logical_membership_rows} | {cells} | {g1_exact_reuse_rows} | {g2_terminal_rows} | {strict_valid_rows} | {invalid_rows} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Passing checks: `{int(validation['status'].eq('PASS').sum())}` / `{len(validation)}`",
            f"- Strict-valid rows with absent positive in ranked output: `{result_summary['strict_positive_absent_rows']}`",
            f"- Strict-valid rows with at least one zero train-count top-5 item for novelty: `{result_summary['strict_top5_missing_popularity_rows']}`",
            f"- Undefined status counts: `{stable_json(result_summary['undefined_status_counts'])}`",
            "",
            "## Artifacts",
            "",
        ]
    )
    for rel_path, digest in sorted(output_hashes.items()):
        lines.append(f"- `{rel_path}` SHA256 `{digest}`")
    if parts["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- `{failure}`" for failure in parts["failures"])
    lines.append("")
    return "\n".join(lines)


def append_notes(root: Path, parts: Mapping[str, Any]) -> None:
    result_summary = parts["manifest_parts"]["result_summary"]
    universe_counts = parts["manifest_parts"]["universe_counts"]
    text = (
        f"\n## {utc_now()} - G2 metric freeze\n\n"
        f"- Decision: `{parts['decision_token']}`.\n"
        "- Scope: computed G2 metrics from frozen analytic universe only; no Ollama calls, no G3, no protocol changes, no evidence edits.\n"
        f"- Final accounting: logical `{universe_counts['logical_requests']}`; G1 reuse `{universe_counts['g1_exact_reuse']}`; "
        f"G2 terminal evidence `{universe_counts['g2_terminal_evidence']}`; strict-valid `{universe_counts['strict_valid']}`; invalid `{universe_counts['invalid']}`.\n"
        f"- Cells computed: `{result_summary['applicable_cells']}`; metric rows `{result_summary['metric_result_rows']}`; "
        f"undefined/blocked rows `{result_summary['undefined_rows']}`.\n"
        "- Denominators: utility planned/applicable with invalid-as-zero; beyond-utility strict-valid with metric-specific defined subsets.\n"
        f"- Outputs: `{OUTPUTS['metric_results'].as_posix()}`, `{OUTPUTS['metric_denominators'].as_posix()}`, "
        f"`{OUTPUTS['metric_validation'].as_posix()}`, `{OUTPUTS['metric_undefined'].as_posix()}`, `{OUTPUTS['manifest'].as_posix()}`.\n"
        "- Next safe step: descriptive G2 readout/statistical analysis from frozen metric tables; do not start G3 yet.\n"
    )
    with (root / PATHS["notes"]).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_outputs(root: Path, parts: dict[str, Any]) -> dict[str, str]:
    for name, frame in parts["tables"].items():
        write_csv(root / OUTPUTS[name], frame)

    table_hashes = {
        OUTPUTS[name].as_posix(): sha256_file(root / OUTPUTS[name])
        for name in ["metric_results", "metric_denominators", "metric_validation", "metric_undefined"]
    }
    report = build_report(parts, table_hashes)
    (root / OUTPUTS["report"]).parent.mkdir(parents=True, exist_ok=True)
    (root / OUTPUTS["report"]).write_text(report, encoding="utf-8", newline="\n")
    report_hash = sha256_file(root / OUTPUTS["report"])
    manifest_output_hashes = {
        **table_hashes,
        OUTPUTS["report"].as_posix(): report_hash,
    }
    manifest = {
        "phase_id": PHASE_ID,
        "decision_token": parts["decision_token"],
        "created_at_utc": utc_now(),
        "precondition_head": EXPECTED_HEAD,
        "head_before_generation": parts["head_before"],
        "zero_inference": True,
        "ollama_calls_executed": 0,
        "g3_started": False,
        "protocol_changes": False,
        "frozen_evidence_modified": False,
        "failures": parts["failures"],
        "input_hashes": parts["manifest_parts"]["input_hashes"],
        "output_hashes": manifest_output_hashes,
        "report_path": OUTPUTS["report"].as_posix(),
        "universe_counts": parts["manifest_parts"]["universe_counts"],
        "evidence_counts": parts["manifest_parts"]["evidence_counts"],
        "result_summary": parts["manifest_parts"]["result_summary"],
        "membership_summary": parts["manifest_parts"]["membership_summary"],
        "validation_checks": parts["manifest_parts"]["validation_checks"],
        "popularity_sources": parts["manifest_parts"]["popularity_sources"],
    }
    (root / OUTPUTS["manifest"]).parent.mkdir(parents=True, exist_ok=True)
    (root / OUTPUTS["manifest"]).write_text(stable_json(manifest) + "\n", encoding="utf-8")
    output_hashes = {
        **manifest_output_hashes,
        OUTPUTS["manifest"].as_posix(): sha256_file(root / OUTPUTS["manifest"]),
    }

    append_notes(root, parts)
    return output_hashes


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve()
    failures: list[str] = []
    parts = compute_metrics(root, failures)
    if args.no_write:
        print(stable_json({"decision_token": parts["decision_token"], "failures": parts["failures"]}))
        return 0 if not parts["failures"] else 1
    output_hashes = write_outputs(root, parts)
    print(
        stable_json(
            {
                "decision_token": parts["decision_token"],
                "failures": parts["failures"],
                "output_hashes": output_hashes,
                "result_summary": parts["manifest_parts"]["result_summary"],
            }
        )
    )
    return 0 if not parts["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
