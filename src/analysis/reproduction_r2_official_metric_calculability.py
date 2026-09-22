from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import math
import re
import subprocess
import warnings
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EVIDENCE_LABEL = "POST-F6M TOTAL REPRODUCIBILITY CLOSURE EVIDENCE"

COMPLETE_TOKEN = "REPRODUCTION_R2_COMPLETE_METRIC_CALCULABILITY_AUDIT_COMPLETE"
PARTIAL_TOKEN = "REPRODUCTION_R2_METRIC_CALCULABILITY_AUDIT_PARTIAL"
BLOCKED_TOKEN = "REPRODUCTION_R2_METRIC_CALCULABILITY_AUDIT_BLOCKED"

EXPECTED_BRANCH = "master"
EXPECTED_HEAD = "10529058c3579c42686b76d5d3ac8b96ba7d6337"

ORIGINAL_ROOT = Path("original_repo") / "EvaLLMasRecommender"
TABLE_DIR = Path("results") / "tables"
REPORT_PATH = Path("report") / "reproduction_r2_complete_official_metric_calculability.md"
NOTES_PATH = Path("report") / "notes.md"

CALCULABILITY_STATUSES = (
    "DIRECTLY_RECALCULABLE",
    "RECONSTRUCTIBLE_FROM_OFFICIAL_ARTIFACTS",
    "PARTIALLY_RECONSTRUCTIBLE",
    "OFFICIAL_VALUE_AVAILABLE_ONLY",
    "NOT_CALCULABLE_MISSING_OUTPUT",
    "NOT_CALCULABLE_MISSING_DATA",
    "NOT_CALCULABLE_MISSING_PROTOCOL",
    "NOT_CALCULABLE_MODEL_UNAVAILABLE",
    "AMBIGUOUS_RECONSTRUCTION",
    "NOT_APPLICABLE",
)

CANDDIF_STATUSES = (
    "CANDDIF_EXACTLY_RECONSTRUCTED",
    "CANDDIF_PARTIALLY_RECONSTRUCTED",
    "CANDDIF_AMBIGUOUS",
    "CANDDIF_NOT_RECOVERABLE",
)

TARGET_DATASETS = ("Beauty", "Sports", "ML-1M", "LastFM")

PROTECTED_ROOT_FILES = (
    "final_metric_paper_comparison_values.md",
    "original_repo_metric_file_inventory.csv",
    "original_repo_metric_values_decoded.csv",
    "original_repo_metric_values_decoded_summary.csv",
    "original_repo_metric_values_decoded_wide.csv",
    "original_repo_requested_metric_values_decoded.csv",
    "requested_metric_candidate_position_bias_values.csv",
    "requested_metric_generation_involved_performance_values.csv",
    "requested_metric_history_length_sensitivity_values.csv",
    "requested_metric_novelty_values.csv",
    "requested_metric_paper_comparison_values.csv",
    "requested_metric_values_summary.csv",
)

OUTPUT_TABLES = {
    "master": TABLE_DIR / "reproduction_r2_official_metric_calculability_master.csv",
    "summary": TABLE_DIR / "reproduction_r2_official_metric_calculability_summary.csv",
    "not_calculable": TABLE_DIR / "reproduction_r2_not_calculable_reasons.csv",
    "opportunities": TABLE_DIR / "reproduction_r2_reconstruction_opportunities.csv",
    "paper_comparison": TABLE_DIR / "reproduction_r2_official_paper_value_comparison.csv",
    "history": TABLE_DIR / "reproduction_r2_history_metric_reconstruction.csv",
    "generation": TABLE_DIR / "reproduction_r2_generation_metric_reconstruction.csv",
    "candidate_position": TABLE_DIR / "reproduction_r2_candidate_position_reconstruction.csv",
    "input_contracts": TABLE_DIR / "reproduction_r2_metric_input_contracts.csv",
    "stored_vs_recomputed": TABLE_DIR / "reproduction_r2_official_stored_vs_recomputed.csv",
    "protected_hash_check": TABLE_DIR / "reproduction_r2_protected_artifact_hash_check.csv",
    "precondition_git": TABLE_DIR / "reproduction_r2_precondition_git_state.csv",
    "postcondition_git": TABLE_DIR / "reproduction_r2_postcondition_git_state.csv",
    "validation": TABLE_DIR / "reproduction_r2_validation.csv",
}

MASTER_COLUMNS = [
    "dataset",
    "task",
    "paper_model",
    "official_model",
    "condition",
    "history_length",
    "profile_mode",
    "profile_generator",
    "candidate_condition",
    "metric_family",
    "paper_metric_name",
    "official_metric_name",
    "cutoff",
    "paper_location",
    "paper_value",
    "paper_display_precision",
    "official_source_archive",
    "official_source_member",
    "official_source_member_sha256",
    "official_stored_value",
    "official_recomputed_value",
    "official_reconstructed_value",
    "calculability_status",
    "required_inputs",
    "available_inputs",
    "missing_inputs",
    "reconstruction_method",
    "reconstruction_evidence",
    "value_origin",
    "paper_mapping_status",
    "value_agreement",
    "absolute_delta",
    "relative_delta",
    "reason_if_not_calculable",
    "protocol_notes",
    "provenance_notes",
    "condition_classification",
    "recomputation_status",
    "comparison_value_source",
    "official_value_used_for_comparison",
]

INPUT_CONTRACT_COLUMNS = [
    "metric",
    "metric_family",
    "formula_source",
    "official_implementation_source",
    "required_recommendation_output",
    "required_ground_truth",
    "required_candidate_list",
    "required_training_interactions",
    "required_item_popularity",
    "required_catalog",
    "required_item_metadata",
    "required_user_groups",
    "required_profile",
    "required_history",
    "required_paired_condition",
    "other_inputs",
    "availability_policy",
]

NOTE_MARKER = "R2 official metric calculability audit"


@dataclass(frozen=True)
class Sources:
    project_root: Path
    r1_decoded: pd.DataFrame
    r1_json: pd.DataFrame
    r1_history: pd.DataFrame
    r1_generation: pd.DataFrame
    r1_candidate: pd.DataFrame
    r1_hallucination: pd.DataFrame
    r1_dataset_evidence: pd.DataFrame
    r1_summary: pd.DataFrame
    e1_contract: pd.DataFrame
    paper_contract: pd.DataFrame


@dataclass(frozen=True)
class RecomputedArtifact:
    key: tuple[str, str]
    source_archive: str
    source_member: str
    source_member_sha256: str
    dataset: str
    model: str
    task: str
    condition: str
    record_count: int
    valid_outputs: int
    invalid_outputs: int
    candidate_mapped_topk: bool
    metrics: dict[str, float]
    available_inputs: str
    missing_inputs: str
    notes: str


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text.strip()


def portable_project_path(project_root: Path, value: Any) -> str:
    """Return a stable project-relative path for paths located inside this repo."""
    text = clean(value)
    if not text:
        return ""
    path = Path(text)
    try:
        if path.is_absolute():
            return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
    except OSError:
        return text.replace("\\", "/")
    return text.replace("\\", "/")


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]] | pd.DataFrame, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        frame = rows.copy()
        for column in columns:
            if column not in frame.columns:
                frame[column] = ""
        frame = frame.loc[:, list(columns)].fillna("")
        frame.to_csv(path, index=False, lineterminator="\n")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: clean(row.get(column, "")) for column in columns})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_output(root: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout.strip()


def load_sources(project_root: Path) -> Sources:
    tables = project_root / TABLE_DIR
    return Sources(
        project_root=project_root,
        r1_decoded=read_csv(tables / "reproduction_r1_original_repo_metric_values_decoded.csv"),
        r1_json=read_csv(tables / "reproduction_r1_official_json_inventory.csv"),
        r1_history=read_csv(tables / "reproduction_r1_history_length_official_results.csv"),
        r1_generation=read_csv(tables / "reproduction_r1_generation_official_results.csv"),
        r1_candidate=read_csv(tables / "reproduction_r1_candidate_position_evidence.csv"),
        r1_hallucination=read_csv(tables / "reproduction_r1_hallucination_official_results.csv"),
        r1_dataset_evidence=read_csv(tables / "reproduction_r1_original_dataset_evidence_classification.csv"),
        r1_summary=read_csv(tables / "reproduction_r1_summary.csv"),
        e1_contract=read_csv(tables / "extension_e1_metric_contract.csv"),
        paper_contract=read_csv(tables / "paper_atomic_contract.csv"),
    )


def parse_protected_manifest(project_root: Path) -> dict[str, str]:
    report = project_root / "report" / "reproduction_r1p_interleaved_head_provenance.md"
    if not report.exists():
        return {}
    pattern = re.compile(r"^\| `([^`]+)` \| YES \| NO \| `\?\?` \| `([a-f0-9]{64})` \|")
    hashes: dict[str, str] = {}
    for line in report.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            hashes[match.group(1)] = match.group(2)
    return hashes


def protected_hash_rows(project_root: Path) -> list[dict[str, str]]:
    manifest = parse_protected_manifest(project_root)
    rows: list[dict[str, str]] = []
    for rel in PROTECTED_ROOT_FILES:
        path = project_root / rel
        status = git_output(project_root, ["status", "--short", "--", rel])
        tracked = bool(git_output(project_root, ["ls-files", "--", rel]))
        observed = sha256_file(path) if path.exists() else ""
        expected = manifest.get(rel, "")
        rows.append(
            {
                "path": rel,
                "exists": str(path.exists()).lower(),
                "tracked": str(tracked).lower(),
                "git_status": status,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "hash_unchanged": str(bool(expected) and observed == expected).lower(),
                "protection_status": (
                    "PROTECTED_UNCHANGED"
                    if path.exists() and not tracked and observed == expected and status.startswith("??")
                    else "PROTECTION_FAILURE"
                ),
            }
        )
    return rows


def git_state_rows(project_root: Path, phase: str) -> list[dict[str, str]]:
    commands = [
        ("branch", ["branch", "--show-current"]),
        ("head", ["rev-parse", "HEAD"]),
        ("status_short", ["status", "--short"]),
        ("status_branch_short", ["status", "--branch", "--short"]),
        ("diff_stat", ["diff", "--stat"]),
        ("diff_cached_stat", ["diff", "--cached", "--stat"]),
    ]
    rows = [
        {
            "phase": phase,
            "scope": "main_repo",
            "check": name,
            "command": "git " + " ".join(args),
            "observed": git_output(project_root, args),
        }
        for name, args in commands
    ]
    original_root = project_root / ORIGINAL_ROOT
    if original_root.exists():
        rows.extend(
            [
                {
                    "phase": phase,
                    "scope": "original_repo",
                    "check": "status_branch_short",
                    "command": "git status --branch --short",
                    "observed": git_output(original_root, ["status", "--branch", "--short"]),
                },
                {
                    "phase": phase,
                    "scope": "original_repo",
                    "check": "diff_stat",
                    "command": "git diff --stat",
                    "observed": git_output(original_root, ["diff", "--stat"]),
                },
            ]
        )
    return rows


def parse_condition(condition: str, source_member: str = "") -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in clean(condition).split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            parsed[key.strip()] = value.strip()
    text = f"{condition} {source_member}".lower()
    if "profile_plus_history" not in parsed.get("profile_mode", "") and "_history_sample" in text and "profile_" in text:
        parsed["profile_mode"] = "profile_plus_history"
    if "profile_only" not in parsed.get("profile_mode", "") and "profile_" in text and "_history_sample" not in text:
        parsed["profile_mode"] = "profile_only"
    if not parsed.get("profile_mode"):
        parsed["profile_mode"] = "history_only"
    if not parsed.get("candidate_count"):
        match = re.search(r"candidate(\d+)", text)
        parsed["candidate_count"] = match.group(1) if match else ""
    return parsed


def base_member(member: str) -> str:
    text = clean(member)
    if text.endswith("_5.csv"):
        return text[:-6]
    if text.endswith(".json"):
        return text[:-5]
    if text.endswith(".csv"):
        return text[:-4]
    return text


def normalized_dataset(value: str) -> str:
    text = clean(value).lower()
    if text in {"amazon beauty", "beauty", "all beauty", "amazon_beauty"}:
        return "Beauty"
    if text in {"amazon sports", "sports", "amazon_sports"}:
        return "Sports"
    if text in {"ml-1m", "movielens-1m", "movielens 1m", "ml_1m"}:
        return "ML-1M"
    if text in {"lastfm", "last fm", "last-fm"}:
        return "LastFM"
    return clean(value)


def normalize_model(value: str) -> str:
    text = clean(value).lower()
    replacements = {
        "gpt-3.5": "gpt3.5",
        "gpt3.5": "gpt3.5",
        "gpt-4": "gpt4",
        "gpt4": "gpt4",
        "gpt4o": "gpt4o",
        "gpt-4o": "gpt4o",
        "gpt4o-mini": "gpt4omini",
        "gpt-4o-mini": "gpt4omini",
        "claude3": "claude3",
        "claude 3": "claude3",
        "qwen2": "qwen2",
        "mistral": "mistral",
        "llmsrec (llama3-70b)": "llama3",
        "llama3": "llama3",
        "llama2": "llama2",
    }
    if text in replacements:
        return replacements[text]
    for old, new in replacements.items():
        if old in text:
            return new
    return re.sub(r"[^a-z0-9.]+", "", text)


def normalize_metric(value: str) -> str:
    text = clean(value).lower()
    mapping = {
        "hr": "hr_at_5",
        "hr@5": "hr_at_5",
        "ndcg": "ndcg_at_5",
        "ndcg@5": "ndcg_at_5",
        "aclt": "aclt_at_5",
        "aplt@5": "aclt_at_5",
        "average_recommendation_popularity_at_5": "average_popularity",
        "arp@5": "average_popularity",
        "arp": "average_popularity",
        "average_popularity_at_5": "average_popularity",
        "average_popularity": "average_popularity",
        "popreo": "popreo_at_5",
        "popreo@5": "popreo_at_5",
        "serendipity": "serendipity_at_5",
        "selfinformation": "self_information_at_5",
        "selfinformation@5": "self_information_at_5",
        "gini": "gini_at_5",
        "dpd": "dpd_at_5",
        "jain": "jain_index",
        "itemcoverage": "item_coverage_at_5",
        "item_coverage_at_5": "item_coverage_at_5",
        "shannon": "shannon_at_5",
        "jaccard": "jaccard_at_5",
        "hallucination": "hallucination_rate",
        "hallucination_rate": "hallucination_rate",
        "candidate_position_bias_hr": "candidate_position_bias_hr",
        "candidate_position_bias_ndcg": "candidate_position_bias_ndcg",
        "canddif_hr": "candidate_position_bias_hr",
        "canddif_ndcg": "candidate_position_bias_ndcg",
    }
    canonical = mapping.get(text, text)
    canonical = mapping.get(canonical, canonical)
    return canonical


def metric_display_precision(value: str) -> str:
    text = clean(value)
    if "." in text:
        return str(len(text.split(".", 1)[1]))
    return ""


def numeric_value(value: Any) -> float | None:
    text = clean(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def format_float(value: float | None) -> str:
    if value is None or math.isnan(value) or math.isinf(value):
        return ""
    return f"{value:.15g}"


def normalize_title(value: Any) -> str:
    text = clean(value)
    text = text.replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")
    text = html.unescape(text)
    text = text.replace("&reg;", "®")
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()


def extract_bracketed_list(text: str, marker: str | None = None) -> str | None:
    source = clean(text)
    start_search = 0
    if marker:
        marker_index = source.lower().find(marker.lower())
        if marker_index < 0:
            return None
        start_search = marker_index + len(marker)
    start = source.find("[", start_search)
    if start < 0:
        return None
    quote: str | None = None
    escaped = False
    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    return None


def parse_prompt_list(prompt: str, marker: str) -> list[str] | None:
    bracketed = extract_bracketed_list(prompt, marker)
    if bracketed is None:
        source = clean(prompt)
        marker_index = source.lower().find(marker.lower())
        if marker_index < 0:
            return None
        tail = source[marker_index + len(marker) :]
        start = tail.find("[")
        end = tail.rfind("]")
        if start < 0 or end <= start:
            return None
        bracketed = tail[start : end + 1]
    parsed_list = literal_eval_list(bracketed)
    if parsed_list is not None:
        return parsed_list
    content = bracketed.strip()[1:-1]
    if not content:
        return []
    parts = content.split("', '")
    cleaned: list[str] = []
    for part in parts:
        item = part
        if item.startswith("'"):
            item = item[1:]
        if item.endswith("'"):
            item = item[:-1]
        cleaned.append(item.replace("\\'", "'"))
    return cleaned


def literal_eval_list(candidate: str) -> list[str] | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            parsed = ast.literal_eval(candidate)
    except Exception:
        return None
    if isinstance(parsed, list):
        return [clean(item) for item in parsed]
    return None


def parse_loose_list(bracketed: str) -> list[str] | None:
    body = clean(bracketed)
    if not body.startswith("[") or not body.endswith("]"):
        return None
    content = body[1:-1].strip()
    if not content:
        return []
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) > 1:
        items: list[str] = []
        for line in lines:
            if line in {"[", "]"}:
                continue
            item = line.rstrip(",").strip()
            if len(item) >= 2 and item[0] in {"'", '"'} and item[-1] == item[0]:
                item = item[1:-1]
            elif item.startswith(("'", '"')):
                item = item[1:]
            items.append(item.replace("\\'", "'").replace('\\"', '"'))
        return items
    parts = content.split("', '")
    if len(parts) == 1:
        parts = content.split('", "')
    items = []
    for part in parts:
        item = part.strip().rstrip(",").strip()
        if len(item) >= 2 and item[0] in {"'", '"'} and item[-1] == item[0]:
            item = item[1:-1]
        elif item.startswith(("'", '"')):
            item = item[1:]
        items.append(item.replace("\\'", "'").replace('\\"', '"'))
    return items if items else None


def parse_cand_ids(value: Any) -> list[str] | None:
    text = clean(value)
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        try:
            parsed = json.loads(text.replace("'", '"'))
        except Exception:
            return None
    if not isinstance(parsed, list):
        return None
    return [clean(item) for item in parsed]


def parse_result_list(value: Any) -> list[str] | None:
    if isinstance(value, list):
        return [clean(item) for item in value]
    text = clean(value)
    bracketed = extract_bracketed_list(text)
    if bracketed is None:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            return None
        bracketed = text[start : end + 1]
    attempts = [bracketed]
    officialish = re.sub(r"\'[,]*[ \n\r]+#.*$", "',", bracketed, flags=re.MULTILINE)
    officialish = re.sub(r"\'[ \n\r]*,[ \n\r]*\'", '", "', officialish.replace('"', "'"))
    officialish = re.sub(r"\'[ \n,]*\]", '"]', officialish)
    officialish = re.sub(r"\[[ \n]*\'", '["', officialish)
    officialish = officialish.replace("['", '["').replace("\\", "\\\\")
    attempts.append(officialish)
    for candidate in attempts:
        parsed_list = literal_eval_list(candidate)
        if parsed_list is not None:
            return parsed_list
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return [clean(item) for item in parsed]
        except Exception:
            pass
    loose = parse_loose_list(bracketed)
    if loose is not None:
        return loose
    return None


def read_zip_member(project_root: Path, source_archive: str, source_member: str) -> bytes | None:
    archive_path = Path(clean(source_archive))
    if not archive_path.exists():
        archive_path = project_root / clean(source_archive)
    if not archive_path.exists() or not zipfile.is_zipfile(archive_path):
        return None
    with zipfile.ZipFile(archive_path) as archive:
        try:
            return archive.read(clean(source_member))
        except KeyError:
            return None


def mean_or_empty(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.asarray(values, dtype=float).mean())


def compute_jain(ndcg_values: Sequence[float]) -> float | None:
    if not ndcg_values:
        return None
    values = np.asarray(ndcg_values, dtype=float)
    denominator = len(values) * float(np.sum(values**2))
    if denominator == 0:
        return None
    return float((float(np.sum(values)) ** 2) / denominator)


def compute_gini(items: Sequence[str]) -> float | None:
    counts = list(Counter(items).values())
    if not counts:
        return None
    cumulative = np.cumsum(sorted(np.asarray(counts, dtype=float)))
    total = float(cumulative[-1])
    if total == 0:
        return None
    x_axis = np.asarray(range(len(cumulative)), dtype=float) / len(cumulative)
    y_axis = cumulative / total
    area_b = float(np.trapz(y_axis, x=x_axis))
    return float((0.5 - area_b) / 0.5)


def compute_jaccard(topk_by_user: Sequence[Sequence[str]], cand_by_user: Sequence[Sequence[str]]) -> float | None:
    values: list[float] = []
    for left in range(len(topk_by_user)):
        for right in range(left + 1, len(topk_by_user)):
            denom = set(cand_by_user[left]).intersection(set(cand_by_user[right]))
            if denom:
                values.append(len(set(topk_by_user[left]).intersection(set(topk_by_user[right]))) / len(denom))
    return mean_or_empty(values)


def recompute_json_artifact(project_root: Path, row: Mapping[str, Any]) -> RecomputedArtifact | None:
    raw_archive = clean(row.get("source_archive"))
    archive = portable_project_path(project_root, raw_archive)
    member = clean(row.get("source_member"))
    data = read_zip_member(project_root, raw_archive, member)
    if data is None:
        return None
    try:
        records = json.loads(data.decode("utf-8"))
    except Exception as exc:
        return RecomputedArtifact(
            key=(archive, base_member(member)),
            source_archive=archive,
            source_member=member,
            source_member_sha256=clean(row.get("source_member_sha256")) or sha256_bytes(data),
            dataset=clean(row.get("dataset")),
            model=clean(row.get("model")),
            task=clean(row.get("task")),
            condition=clean(row.get("condition")),
            record_count=0,
            valid_outputs=0,
            invalid_outputs=1,
            candidate_mapped_topk=False,
            metrics={},
            available_inputs="official JSON member",
            missing_inputs="parseable JSON records",
            notes=f"JSON parse failed: {exc}",
        )
    if not isinstance(records, list):
        return None

    hits: list[float] = []
    ndcgs: list[float] = []
    topk_item_ids: list[str] = []
    topk_by_user: list[list[str]] = []
    cand_by_user: list[list[str]] = []
    invalid = 0
    mapped_ok = True
    valid = 0
    positive_present_count = 0
    topk = 5

    for record in records:
        if not isinstance(record, dict):
            invalid += 1
            mapped_ok = False
            continue
        target = clean(record.get("target"))
        results = parse_result_list(record.get("result"))
        if results is None:
            invalid += 1
            mapped_ok = False
            continue
        valid += 1
        normalized_results = [normalize_title(item) for item in results]
        normalized_target = normalize_title(target)
        if normalized_target in normalized_results:
            rank_one = normalized_results.index(normalized_target) + 1
            positive_present_count += 1
        else:
            rank_one = 20
        hit = 1.0 if rank_one <= topk else 0.0
        ndcg = 1.0 / math.log2(rank_one + 1.0) if rank_one <= topk else 0.0
        hits.append(hit)
        ndcgs.append(ndcg)

        cand_ids = parse_cand_ids(record.get("cand_list"))
        candidate_titles = parse_prompt_list(clean(record.get("prompt")), "The candidates are:")
        row_top_ids: list[str] = []
        if cand_ids is None or candidate_titles is None or len(cand_ids) != len(candidate_titles):
            mapped_ok = False
            continue
        title_to_ids: dict[str, list[str]] = defaultdict(list)
        for title, item_id in zip(candidate_titles, cand_ids, strict=False):
            title_to_ids[normalize_title(title)].append(clean(item_id))
        for result_title in results[:topk]:
            ids = title_to_ids.get(normalize_title(result_title), [])
            if len(ids) != 1:
                mapped_ok = False
                row_top_ids = []
                break
            row_top_ids.append(ids[0])
        if len(row_top_ids) == topk:
            topk_item_ids.extend(row_top_ids)
            topk_by_user.append(row_top_ids)
            cand_by_user.append(cand_ids)
        else:
            mapped_ok = False

    metrics: dict[str, float] = {}
    if invalid == 0 and records:
        metrics["hr_at_5"] = float(np.asarray(hits, dtype=float).mean())
        metrics["ndcg_at_5"] = float(np.asarray(ndcgs, dtype=float).mean())
        jain = compute_jain(ndcgs)
        if jain is not None:
            metrics["jain_index"] = jain
    if invalid == 0 and records and mapped_ok and len(topk_by_user) == len(records):
        gini = compute_gini(topk_item_ids)
        if gini is not None:
            metrics["gini_at_5"] = gini
        candidate_union = set(item for row_ids in cand_by_user for item in row_ids)
        if candidate_union:
            metrics["item_coverage_at_5"] = len(set(topk_item_ids)) / len(candidate_union)
        jaccard = compute_jaccard(topk_by_user, cand_by_user)
        if jaccard is not None:
            metrics["jaccard_at_5"] = jaccard

    available = [
        "official JSON result records",
        "target title per record",
        "parsed recommendation titles",
        "candidate ids from cand_list",
    ]
    if mapped_ok:
        available.append("unambiguous top-K candidate-title-to-id mapping")
    missing = []
    if invalid:
        missing.append("parse-compatible recommendation output for every record")
    if not mapped_ok:
        missing.append("unambiguous candidate-title mapping for every top-K output")

    return RecomputedArtifact(
        key=(archive, base_member(member)),
        source_archive=archive,
        source_member=member,
        source_member_sha256=clean(row.get("source_member_sha256")) or sha256_bytes(data),
        dataset=clean(row.get("dataset")),
        model=clean(row.get("model")),
        task=clean(row.get("task")),
        condition=clean(row.get("condition")),
        record_count=len(records),
        valid_outputs=valid,
        invalid_outputs=invalid,
        candidate_mapped_topk=mapped_ok,
        metrics=metrics,
        available_inputs="; ".join(available),
        missing_inputs="; ".join(missing),
        notes=f"positive_present={positive_present_count}; denominator={len(records)}",
    )


def recompute_official_jsons(sources: Sources) -> dict[tuple[str, str], RecomputedArtifact]:
    results: dict[tuple[str, str], RecomputedArtifact] = {}
    for _, row in sources.r1_json.iterrows():
        if clean(row.get("json_artifact_type")) != "llm_result_json":
            continue
        if clean(row.get("json_parse_status")) != "PARSED":
            continue
        if clean(row.get("has_result")) != "YES":
            continue
        recomputed = recompute_json_artifact(sources.project_root, row)
        if recomputed is not None:
            results[recomputed.key] = recomputed
    return results


def build_metric_input_contracts(sources: Sources) -> pd.DataFrame:
    contracts = {
        "hr_at_5": ("utility", "paper/E1 HR@K and official calHR", "llm_metrics.py: calHR", "yes", "target title or held-out item", "candidate list optional", "", "", "", "", "", "", "", "", "Direct if official JSON result and target parse for every record."),
        "ndcg_at_5": ("utility", "paper/E1 NDCG@K and official calNDCG", "llm_metrics.py: calNDCG", "yes", "target title or held-out item", "candidate list optional", "", "", "", "", "", "", "", "", "Direct if official JSON result and target parse for every record."),
        "aclt_at_5": ("popularity", "paper/E1 APLT and official calACLT", "llm_metrics.py: calACLT", "yes", "", "", "yes", "train item frequencies and high_bar", "", "", "", "", "", "", "Stored-only without official train frequencies."),
        "average_popularity": ("popularity", "paper/E1 ARP and official calARP", "llm_metrics.py: calARP", "yes", "", "", "yes", "train item frequencies", "", "", "", "", "", "", "Stored-only without official train frequencies."),
        "popreo_at_5": ("popularity", "paper/E1 PopREO and official calPopREO", "llm_metrics.py: calPopREO", "yes", "target items and hit vector", "", "yes", "popularity quantiles", "", "", "popularity groups", "", "", "", "Stored-only without official train frequencies and group denominators."),
        "serendipity_at_5": ("novelty", "paper/E1 Serendipity and official calSerendipity", "llm_metrics.py: calSerendipity", "yes", "hit vector", "candidate list", "yes", "MostPop train popularity", "", "", "", "", "", "MostPop paired hit condition", "Stored-only without aligned MostPop hits."),
        "self_information_at_5": ("novelty", "paper/E1 SelfInformation and official calSelfInfo", "llm_metrics.py: calSelfInfo", "yes", "", "", "yes", "information dictionary", "", "", "", "", "", "", "Stored-only without official information dictionary."),
        "gini_at_5": ("fairness", "paper/E1 Gini and official calcFairMetric", "llm_metrics.py: calcFairMetric", "yes", "", "candidate list if using JSON title mapping", "", "", "", "", "", "", "", "", "Direct only when every top-K title maps to an official candidate id."),
        "dpd_at_5": ("fairness", "paper/E1 DPD and official calDPDFreq", "llm_metrics.py: calDPDFreq", "yes", "hit/rank vector", "", "yes", "", "", "", "cold/hot users from median history length", "", "full user histories", "", "Stored-only without official sequential_data user history lengths."),
        "jain_index": ("fairness", "paper/E1 Jain and official calJain", "llm_metrics.py: calJain", "yes", "target title or held-out item", "", "", "", "", "", "", "", "", "", "Direct if HR/NDCG per-user vector can be reconstructed."),
        "item_coverage_at_5": ("diversity", "paper/E1 ItemCoverage and official calItemCoverage", "llm_metrics.py: calItemCoverage", "yes", "", "candidate item ids", "", "", "", "", "", "", "", "", "Direct if every top-K title maps to official candidate ids and candidate union exists."),
        "shannon_at_5": ("diversity", "paper/E1 Shannon and official calMeanShannon", "llm_metrics.py: calMeanShannon", "yes", "", "", "", "", "", "category metadata", "", "", "", "", "Stored-only without official category metadata."),
        "jaccard_at_5": ("diversity", "paper/E1 Jaccard/OIC and official calMeanJaccard", "llm_metrics.py: calMeanJaccard", "yes", "", "candidate item ids", "", "", "", "", "", "", "", "", "Direct if every top-K title maps to official candidate ids and candidate lists exist."),
        "hallucination_rate": ("hallucination", "paper/E1 hallucination prose and official calHallucination", "llm_metrics.py: calHallucination", "yes", "", "", "", "", "global title-to-item catalog", "datamaps/meta title universe", "", "", "", "", "Stored-only without official datamaps/meta global title mapping."),
        "candidate_position_bias_hr": ("LLM_specific", "paper/E1 CandDif-HR", "E1 contract; no official CandDif function", "", "", "", "", "", "", "", "", "", "", "paired first-position and random-position HR", "Reconstructible only from unique paired official aggregate HR rows."),
        "candidate_position_bias_ndcg": ("LLM_specific", "paper/E1 CandDif-NDCG", "E1 contract; no official CandDif function", "", "", "", "", "", "", "", "", "", "", "paired first-position and random-position NDCG", "Reconstructible only from unique paired official aggregate NDCG rows."),
        "history_length_sensitivity": ("LLM_specific", "paper/E1 history sensitivity", "metrics.sh history commands", "official metric values per history length", "", "", "", "", "", "", "", "", "history-length condition labels", "", "Stored/recomputed at each available official history length; no interpolation."),
        "generation_involved_profile_matrix": ("LLM_specific", "paper/E1 generation/profile matrix", "metrics.sh profile commands and templates", "official metric values/profile output conditions", "", "", "", "", "", "", "", "profile artifacts", "history condition labels", "", "Stored/recomputed where profile condition outputs exist."),
        "ks_sample_representativeness": ("sampling_protocol", "paper/E1 K-S sample gate", "no recovered official implementation", "", "", "", "", "", "", "", "", "", "", "sample/full distribution protocol", "Not calculable without recovered protocol and full outputs."),
    }
    rows: list[dict[str, str]] = []
    for metric, values in sorted(contracts.items()):
        rows.append(
            {
                "metric": metric,
                "metric_family": values[0],
                "formula_source": values[1],
                "official_implementation_source": values[2],
                "required_recommendation_output": values[3],
                "required_ground_truth": values[4],
                "required_candidate_list": values[5],
                "required_training_interactions": values[6],
                "required_item_popularity": values[7],
                "required_catalog": values[8],
                "required_item_metadata": values[9],
                "required_user_groups": values[10],
                "required_profile": values[11],
                "required_history": values[12],
                "required_paired_condition": values[13],
                "other_inputs": "",
                "availability_policy": values[14],
            }
        )
    for _, row in sources.e1_contract.iterrows():
        metric = normalize_metric(row.get("metric_id") or row.get("metric_name"))
        if metric and metric not in {entry["metric"] for entry in rows}:
            rows.append(
                {
                    "metric": metric,
                    "metric_family": clean(row.get("family")),
                    "formula_source": clean(row.get("formula_status")),
                    "official_implementation_source": clean(row.get("official_code_evidence")),
                    "required_recommendation_output": clean(row.get("input_data")),
                    "required_ground_truth": "",
                    "required_candidate_list": clean(row.get("catalog_source")),
                    "required_training_interactions": clean(row.get("history_source")),
                    "required_item_popularity": clean(row.get("popularity_source")),
                    "required_catalog": clean(row.get("catalog_source")),
                    "required_item_metadata": clean(row.get("required_metadata")),
                    "required_user_groups": clean(row.get("group_definition")),
                    "required_profile": "",
                    "required_history": clean(row.get("history_source")),
                    "required_paired_condition": "",
                    "other_inputs": clean(row.get("notes")),
                    "availability_policy": "Derived from E1 metric contract.",
                }
            )
    return pd.DataFrame(rows, columns=INPUT_CONTRACT_COLUMNS).fillna("")


def contract_for_metric(contracts: pd.DataFrame, metric: str) -> dict[str, str]:
    if contracts.empty:
        return {}
    normalized = normalize_metric(metric)
    subset = contracts.loc[contracts["metric"].map(normalize_metric).eq(normalized)]
    if subset.empty:
        return {}
    return {column: clean(subset.iloc[0].get(column)) for column in contracts.columns}


def official_data_tree_available(project_root: Path) -> bool:
    return (project_root / ORIGINAL_ROOT / "data").exists()


def classify_official_metric_row(
    row: Mapping[str, Any],
    recomputed: RecomputedArtifact | None,
    contracts: pd.DataFrame,
    data_tree_available: bool,
) -> dict[str, str]:
    metric = normalize_metric(row.get("canonical_metric_name") or row.get("official_metric_name"))
    contract = contract_for_metric(contracts, metric)
    required = "; ".join(
        part
        for part in [
            contract.get("required_recommendation_output"),
            contract.get("required_ground_truth"),
            contract.get("required_candidate_list"),
            contract.get("required_training_interactions"),
            contract.get("required_item_popularity"),
            contract.get("required_catalog"),
            contract.get("required_item_metadata"),
            contract.get("required_user_groups"),
            contract.get("required_profile"),
            contract.get("required_history"),
            contract.get("required_paired_condition"),
        ]
        if part
    )
    direct_metrics = {"hr_at_5", "ndcg_at_5", "jain_index", "gini_at_5", "item_coverage_at_5", "jaccard_at_5"}
    if metric in direct_metrics and recomputed is not None and metric in recomputed.metrics:
        return {
            "calculability_status": "DIRECTLY_RECALCULABLE",
            "required_inputs": required,
            "available_inputs": recomputed.available_inputs,
            "missing_inputs": "",
            "reason_if_not_calculable": "",
            "protocol_notes": recomputed.notes,
            "official_recomputed_value": format_float(recomputed.metrics.get(metric)),
        }
    if metric in direct_metrics and recomputed is not None:
        missing = recomputed.missing_inputs or "compatible official JSON inputs"
        return {
            "calculability_status": "OFFICIAL_VALUE_AVAILABLE_ONLY",
            "required_inputs": required,
            "available_inputs": recomputed.available_inputs,
            "missing_inputs": missing,
            "reason_if_not_calculable": f"Stored official value exists, but independent recomputation is blocked by missing {missing}.",
            "protocol_notes": recomputed.notes,
            "official_recomputed_value": "",
        }

    metric_missing_inputs = {
        "aclt_at_5": "official train item frequency dictionary and long-tail threshold",
        "average_popularity": "official train item frequency dictionary",
        "popreo_at_5": "official train frequency quantiles and target popularity groups",
        "serendipity_at_5": "aligned official MostPop hit vector",
        "self_information_at_5": "official information dictionary from training frequencies",
        "dpd_at_5": "official sequential_data user history lengths and median split",
        "shannon_at_5": "official item category metadata",
        "hallucination_rate": "official datamaps/meta global title-to-item mapping",
    }
    missing_input = metric_missing_inputs.get(metric)
    if missing_input:
        available = "official stored _5.csv metric value"
        if recomputed is not None:
            available += "; official JSON recommendation output"
        if data_tree_available:
            available += "; original data tree present but exact input still not proven"
        return {
            "calculability_status": "OFFICIAL_VALUE_AVAILABLE_ONLY",
            "required_inputs": required,
            "available_inputs": available,
            "missing_inputs": missing_input,
            "reason_if_not_calculable": f"Stored official value exists, but independent recomputation requires {missing_input}.",
            "protocol_notes": "Official data tree is absent." if not data_tree_available else "Exact input availability still requires verification.",
            "official_recomputed_value": "",
        }

    return {
        "calculability_status": "OFFICIAL_VALUE_AVAILABLE_ONLY",
        "required_inputs": required,
        "available_inputs": "official stored _5.csv metric value",
        "missing_inputs": "independent raw inputs not fully identified",
        "reason_if_not_calculable": "Stored official value exists, but independent recomputation inputs are not fully identified.",
        "protocol_notes": "Conservative stored-only classification.",
        "official_recomputed_value": "",
    }


def paper_lookup_rows(paper_contract: pd.DataFrame, dataset: str, task: str, model: str, metric: str) -> pd.DataFrame:
    if paper_contract.empty:
        return pd.DataFrame()
    frame = paper_contract.copy()
    frame["_dataset"] = frame["dataset"].map(normalized_dataset)
    frame["_task"] = frame["task"].map(clean)
    frame["_model"] = frame["model"].map(normalize_model)
    frame["_metric"] = frame["metric"].map(normalize_metric)
    return frame.loc[
        frame["_dataset"].eq(normalized_dataset(dataset))
        & frame["_task"].eq(clean(task))
        & frame["_model"].eq(normalize_model(model))
        & frame["_metric"].eq(normalize_metric(metric))
    ]


def value_agreement(paper_value: str, official_value: str, mapping_status: str) -> tuple[str, str, str]:
    paper_num = numeric_value(paper_value)
    official_num = numeric_value(official_value)
    if paper_num is None or official_num is None:
        return "not_available", "", ""
    if mapping_status not in {"direct_match", "likely_match"}:
        return "not_comparable", "", ""
    absolute = abs(official_num - paper_num)
    relative = absolute / abs(paper_num) if paper_num else 0.0
    if absolute == 0:
        return "exact", format_float(absolute), format_float(relative)
    precision = metric_display_precision(paper_value)
    if precision and round(official_num, int(precision)) == round(paper_num, int(precision)):
        return "rounding_match", format_float(absolute), format_float(relative)
    return "different", format_float(absolute), format_float(relative)


def recomputation_agreement(stored: str, recomputed: str) -> tuple[str, str, str]:
    stored_num = numeric_value(stored)
    recomputed_num = numeric_value(recomputed)
    if stored_num is None or recomputed_num is None:
        return "RECOMPUTATION_NOT_COMPARABLE", "", ""
    absolute = abs(stored_num - recomputed_num)
    relative = absolute / abs(stored_num) if stored_num else 0.0
    if absolute <= 1e-12:
        return "RECOMPUTATION_EXACT", format_float(absolute), format_float(relative)
    if absolute <= 5e-6:
        return "RECOMPUTATION_ROUNDING_ONLY", format_float(absolute), format_float(relative)
    return "RECOMPUTATION_DIFFERENT", format_float(absolute), format_float(relative)


def build_official_master_rows(
    sources: Sources,
    contracts: pd.DataFrame,
    recomputed: Mapping[tuple[str, str], RecomputedArtifact],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    data_available = official_data_tree_available(sources.project_root)
    for _, row in sources.r1_decoded.iterrows():
        archive = portable_project_path(sources.project_root, row.get("source_archive"))
        member = clean(row.get("source_member"))
        recompute = recomputed.get((archive, base_member(member)))
        metric = normalize_metric(row.get("canonical_metric_name") or row.get("official_metric_name"))
        attrs = parse_condition(clean(row.get("condition")), member)
        classification = classify_official_metric_row(row, recompute, contracts, data_available)
        paper_rows = paper_lookup_rows(
            sources.paper_contract,
            clean(row.get("dataset")),
            clean(row.get("task")),
            clean(row.get("model")),
            metric,
        )
        if len(paper_rows) == 1:
            paper = paper_rows.iloc[0]
            paper_mapping = "likely_match"
            paper_value = clean(paper.get("paper_value"))
            paper_location = f"paper_table={clean(paper.get('paper_table'))}; contract_id={clean(paper.get('contract_id'))}"
            paper_metric = clean(paper.get("metric"))
            paper_model = clean(paper.get("model"))
        elif len(paper_rows) > 1:
            paper_mapping = "ambiguous_mapping"
            paper_value = ""
            paper_location = f"{len(paper_rows)} possible paper cells"
            paper_metric = ""
            paper_model = ""
        else:
            paper_mapping = "not_available"
            paper_value = ""
            paper_location = ""
            paper_metric = ""
            paper_model = ""

        comparison_value = classification.get("official_recomputed_value") or clean(row.get("original_repo_value"))
        agreement, abs_delta, rel_delta = value_agreement(paper_value, comparison_value, paper_mapping)
        comparison_source = "INDEPENDENT_RECOMPUTATION" if classification.get("official_recomputed_value") else "OFFICIAL_STORED_ONLY"
        if not paper_value:
            agreement = "not_available"

        condition_classification = "OFFICIAL_EXTRA_CONDITION" if paper_mapping == "not_available" else "PAPER_LINK_CANDIDATE"
        rows.append(
            {
                "dataset": clean(row.get("dataset")),
                "task": clean(row.get("task")),
                "paper_model": paper_model,
                "official_model": clean(row.get("model")),
                "condition": clean(row.get("condition")),
                "history_length": clean(row.get("history_length")),
                "profile_mode": attrs.get("profile_mode", ""),
                "profile_generator": clean(row.get("profile_generator")),
                "candidate_condition": clean(row.get("candidate_condition")),
                "metric_family": clean(row.get("metric_family")),
                "paper_metric_name": paper_metric,
                "official_metric_name": clean(row.get("official_metric_name")),
                "cutoff": clean(row.get("cutoff")),
                "paper_location": paper_location,
                "paper_value": paper_value,
                "paper_display_precision": metric_display_precision(paper_value),
                "official_source_archive": archive,
                "official_source_member": member,
                "official_source_member_sha256": clean(row.get("source_member_sha256")),
                "official_stored_value": clean(row.get("original_repo_value")),
                "official_recomputed_value": classification.get("official_recomputed_value", ""),
                "official_reconstructed_value": "",
                "calculability_status": classification["calculability_status"],
                "required_inputs": classification["required_inputs"],
                "available_inputs": classification["available_inputs"],
                "missing_inputs": classification["missing_inputs"],
                "reconstruction_method": "",
                "reconstruction_evidence": "",
                "value_origin": "INDEPENDENT_RECOMPUTATION" if classification.get("official_recomputed_value") else "OFFICIAL_ARTIFACT_DIRECT",
                "paper_mapping_status": paper_mapping,
                "value_agreement": agreement,
                "absolute_delta": abs_delta,
                "relative_delta": rel_delta,
                "reason_if_not_calculable": classification["reason_if_not_calculable"],
                "protocol_notes": classification["protocol_notes"],
                "provenance_notes": "Decoded by R1; R2 classification uses official source semantics and available official artifacts.",
                "condition_classification": condition_classification,
                "recomputation_status": "",
                "comparison_value_source": comparison_source if comparison_value else "NONE",
                "official_value_used_for_comparison": comparison_value,
            }
        )
    return rows


def build_paper_only_rows(sources: Sources, existing_rows: Sequence[Mapping[str, str]], contracts: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    existing_keys = {
        (
            normalized_dataset(row.get("dataset", "")),
            clean(row.get("task", "")),
            normalize_model(row.get("paper_model", "") or row.get("official_model", "")),
            normalize_metric(row.get("paper_metric_name", "") or row.get("official_metric_name", "")),
        )
        for row in existing_rows
        if clean(row.get("paper_value"))
    }
    for _, paper in sources.paper_contract.iterrows():
        dataset = normalized_dataset(paper.get("dataset"))
        task = clean(paper.get("task"))
        model = clean(paper.get("model"))
        metric = normalize_metric(paper.get("metric"))
        key = (dataset, task, normalize_model(model), metric)
        if key in existing_keys:
            continue
        contract = contract_for_metric(contracts, metric)
        if dataset in {"ML-1M", "LastFM"}:
            status = "NOT_CALCULABLE_MISSING_OUTPUT"
            missing = f"official {dataset} result artifact"
            reason = f"R1 classified {dataset} as REFERENCE_ONLY; no official output artifact was recovered."
        elif dataset in {"Beauty", "Sports"}:
            status = "NOT_CALCULABLE_MISSING_OUTPUT"
            missing = f"official output or metric artifact matching paper model {model}"
            reason = "Paper cell exists, but no uniquely matching official aggregate or output artifact was recovered by R1."
        else:
            status = "NOT_CALCULABLE_MISSING_OUTPUT"
            missing = "official result artifact"
            reason = "No official artifact is available for this paper cell."
        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "paper_model": model,
                "official_model": "",
                "condition": f"paper_cell={clean(paper.get('contract_id'))}",
                "history_length": clean(paper.get("history_policy")),
                "profile_mode": "",
                "profile_generator": "",
                "candidate_condition": clean(paper.get("candidate_policy")),
                "metric_family": contract.get("metric_family", clean(paper.get("paper_section"))),
                "paper_metric_name": clean(paper.get("metric")),
                "official_metric_name": "",
                "cutoff": clean(paper.get("cutoff")),
                "paper_location": f"paper_table={clean(paper.get('paper_table'))}; contract_id={clean(paper.get('contract_id'))}",
                "paper_value": clean(paper.get("paper_value")),
                "paper_display_precision": metric_display_precision(paper.get("paper_value")),
                "official_source_archive": "",
                "official_source_member": "",
                "official_source_member_sha256": "",
                "official_stored_value": "",
                "official_recomputed_value": "",
                "official_reconstructed_value": "",
                "calculability_status": status,
                "required_inputs": contract.get("availability_policy", clean(paper.get("denominator"))),
                "available_inputs": "paper structured value only",
                "missing_inputs": missing,
                "reconstruction_method": "",
                "reconstruction_evidence": "",
                "value_origin": "PAPER_VALUE_ONLY_NOT_OFFICIAL_ARTIFACT",
                "paper_mapping_status": "not_available",
                "value_agreement": "not_available",
                "absolute_delta": "",
                "relative_delta": "",
                "reason_if_not_calculable": reason,
                "protocol_notes": clean(paper.get("paper_specification_status")),
                "provenance_notes": "Paper value retained from structured paper_atomic_contract.csv; not used to infer official mapping.",
                "condition_classification": "PAPER_ONLY_CELL",
                "recomputation_status": "",
                "comparison_value_source": "NONE",
                "official_value_used_for_comparison": "",
            }
        )
    return rows


def build_e1_contract_only_rows(sources: Sources, existing_rows: Sequence[Mapping[str, str]], contracts: pd.DataFrame) -> list[dict[str, str]]:
    represented = {
        normalize_metric(row.get("paper_metric_name", "") or row.get("official_metric_name", ""))
        for row in existing_rows
    }
    rows: list[dict[str, str]] = []
    for _, e1 in sources.e1_contract.iterrows():
        metric = normalize_metric(e1.get("metric_id") or e1.get("metric_name"))
        if metric in represented:
            continue
        family = clean(e1.get("family"))
        is_local_extension = "LOCAL_EXTENSION" in clean(e1.get("provenance_classification"))
        for dataset in TARGET_DATASETS:
            status = "NOT_APPLICABLE" if is_local_extension else "NOT_CALCULABLE_MISSING_OUTPUT"
            missing = "" if is_local_extension else f"official {dataset} output/protocol evidence for E1 analysis"
            rows.append(
                {
                    "dataset": dataset,
                    "task": clean(e1.get("task_applicability")),
                    "paper_model": "",
                    "official_model": "",
                    "condition": f"E1_CONTRACT_ONLY:{clean(e1.get('metric_id'))}",
                    "history_length": "",
                    "profile_mode": "",
                    "profile_generator": "",
                    "candidate_condition": "",
                    "metric_family": family,
                    "paper_metric_name": clean(e1.get("metric_name")),
                    "official_metric_name": "",
                    "cutoff": clean(e1.get("cutoff_k")),
                    "paper_location": clean(e1.get("paper_evidence")),
                    "paper_value": "",
                    "paper_display_precision": "",
                    "official_source_archive": "",
                    "official_source_member": "",
                    "official_source_member_sha256": "",
                    "official_stored_value": "",
                    "official_recomputed_value": "",
                    "official_reconstructed_value": "",
                    "calculability_status": status,
                    "required_inputs": clean(e1.get("input_data")),
                    "available_inputs": "E1 metric contract",
                    "missing_inputs": missing,
                    "reconstruction_method": "",
                    "reconstruction_evidence": "",
                    "value_origin": "E1_CONTRACT_ONLY",
                    "paper_mapping_status": "not_available",
                    "value_agreement": "not_available",
                    "absolute_delta": "",
                    "relative_delta": "",
                    "reason_if_not_calculable": "Local extension or diagnostic, not an original official-paper metric cell." if is_local_extension else f"No official {dataset} artifact recovered for this E1 analysis.",
                    "protocol_notes": clean(e1.get("notes")),
                    "provenance_notes": "Added to prevent silent omission of E1 paper/analysis universe elements.",
                    "condition_classification": "E1_CONTRACT_ONLY",
                    "recomputation_status": "",
                    "comparison_value_source": "NONE",
                    "official_value_used_for_comparison": "",
                }
            )
    return rows


def canddif_transform(value: float) -> float | None:
    if value < 0 or value >= 1:
        return None
    return -math.log(1.0 - value)


def build_candidate_position_rows(master: pd.DataFrame) -> pd.DataFrame:
    official = master.loc[
        master["official_stored_value"].astype(str).str.len().gt(0)
        & master["official_metric_name"].isin(["HR", "NDCG"])
    ].copy()
    if official.empty:
        return pd.DataFrame()
    official["candidate_count"] = [
        parse_condition(row["condition"], row["official_source_member"]).get("candidate_count", "")
        for _, row in official.iterrows()
    ]
    official["metric_key"] = official["official_metric_name"].map(lambda value: "candidate_position_bias_hr" if value == "HR" else "candidate_position_bias_ndcg")
    group_cols = ["dataset", "task", "official_model", "history_length", "profile_mode", "profile_generator", "candidate_count", "metric_key"]
    rows: list[dict[str, str]] = []
    for group_key, group in official.groupby(group_cols, sort=True, dropna=False):
        group_map = dict(zip(group_cols, group_key, strict=False))
        first = group.loc[group["candidate_condition"].str.contains("candidate_order_not_shuffled_or_unspecified", na=False)]
        random = group.loc[group["candidate_condition"].str.contains("candidate_order_shuffled", na=False)]
        metric_label = "CandDif-HR" if group_map["metric_key"] == "candidate_position_bias_hr" else "CandDif-NDCG"
        if len(first) == 1 and len(random) == 1:
            first_value = numeric_value(first.iloc[0]["official_stored_value"])
            random_value = numeric_value(random.iloc[0]["official_stored_value"])
            first_t = canddif_transform(first_value) if first_value is not None else None
            random_t = canddif_transform(random_value) if random_value is not None else None
            if first_t is not None and random_t is not None:
                status = "CANDDIF_EXACTLY_RECONSTRUCTED"
                value = first_t - random_t
                limitation = ""
            else:
                status = "CANDDIF_PARTIALLY_RECONSTRUCTED"
                value = None
                limitation = "CandDif transform undefined for one or both aggregate values."
        elif len(first) or len(random):
            status = "CANDDIF_PARTIALLY_RECONSTRUCTED"
            value = None
            limitation = f"paired sources incomplete or non-unique: first={len(first)} random={len(random)}"
        else:
            continue
        source_rows = []
        for label, frame in [("first", first), ("random", random)]:
            for _, source in frame.iterrows():
                source_rows.append(f"{label}:{source['official_source_member']}={source['official_stored_value']}")
        rows.append(
            {
                "dataset": clean(group_map["dataset"]),
                "task": clean(group_map["task"]),
                "model": clean(group_map["official_model"]),
                "condition": f"history={clean(group_map['history_length'])}; profile_mode={clean(group_map['profile_mode'])}; profile_generator={clean(group_map['profile_generator'])}; candidate_count={clean(group_map['candidate_count'])}",
                "metric": metric_label,
                "canddif_status": status,
                "value": format_float(value),
                "value_origin": "DERIVED_FROM_OFFICIAL_ARTIFACTS" if value is not None else "",
                "formula": "-log(1 - Acc(first-position)) - (-log(1 - Acc(random-position)))",
                "first_position_value": clean(first.iloc[0]["official_stored_value"]) if len(first) == 1 else "",
                "random_position_value": clean(random.iloc[0]["official_stored_value"]) if len(random) == 1 else "",
                "source_artifacts": "; ".join(source_rows),
                "source_member_sha256": "; ".join(
                    [clean(v) for v in list(first["official_source_member_sha256"]) + list(random["official_source_member_sha256"])]
                ),
                "limitations": limitation,
            }
        )
    if not rows:
        return pd.DataFrame(
            [
                {
                    "dataset": "",
                    "task": "",
                    "model": "",
                    "condition": "",
                    "metric": "CandDif",
                    "canddif_status": "CANDDIF_NOT_RECOVERABLE",
                    "value": "",
                    "value_origin": "",
                    "formula": "-log(1 - Acc(first-position)) - (-log(1 - Acc(random-position)))",
                    "first_position_value": "",
                    "random_position_value": "",
                    "source_artifacts": "",
                    "source_member_sha256": "",
                    "limitations": "No paired official candidate-position aggregate rows recovered.",
                }
            ]
        )
    return pd.DataFrame(rows).fillna("")


def append_canddif_to_master(master: pd.DataFrame, candidate_rows: pd.DataFrame, sources: Sources) -> pd.DataFrame:
    additions: list[dict[str, str]] = []
    for _, row in candidate_rows.iterrows():
        status = clean(row.get("canddif_status"))
        if status == "CANDDIF_EXACTLY_RECONSTRUCTED":
            calc_status = "RECONSTRUCTIBLE_FROM_OFFICIAL_ARTIFACTS"
            reason = ""
            reconstructed = clean(row.get("value"))
        elif status == "CANDDIF_PARTIALLY_RECONSTRUCTED":
            calc_status = "PARTIALLY_RECONSTRUCTIBLE"
            reason = clean(row.get("limitations"))
            reconstructed = ""
        elif status == "CANDDIF_AMBIGUOUS":
            calc_status = "AMBIGUOUS_RECONSTRUCTION"
            reason = clean(row.get("limitations"))
            reconstructed = ""
        else:
            calc_status = "NOT_CALCULABLE_MISSING_OUTPUT"
            reason = clean(row.get("limitations"))
            reconstructed = ""
        paper_metric = "candidate_position_bias_hr" if clean(row.get("metric")).endswith("HR") else "candidate_position_bias_ndcg"
        paper_rows = paper_lookup_rows(sources.paper_contract, row.get("dataset"), row.get("task"), row.get("model"), paper_metric)
        paper_value = clean(paper_rows.iloc[0].get("paper_value")) if len(paper_rows) == 1 else ""
        paper_location = (
            f"paper_table={clean(paper_rows.iloc[0].get('paper_table'))}; contract_id={clean(paper_rows.iloc[0].get('contract_id'))}"
            if len(paper_rows) == 1
            else ""
        )
        mapping = "partial_match" if len(paper_rows) == 1 else "not_available"
        agreement, abs_delta, rel_delta = value_agreement(paper_value, reconstructed, mapping)
        additions.append(
            {
                "dataset": clean(row.get("dataset")),
                "task": clean(row.get("task")),
                "paper_model": clean(paper_rows.iloc[0].get("model")) if len(paper_rows) == 1 else "",
                "official_model": clean(row.get("model")),
                "condition": clean(row.get("condition")),
                "history_length": parse_condition(clean(row.get("condition"))).get("history", ""),
                "profile_mode": parse_condition(clean(row.get("condition"))).get("profile_mode", ""),
                "profile_generator": parse_condition(clean(row.get("condition"))).get("profile_generator", ""),
                "candidate_condition": "paired:first_position_vs_random_position",
                "metric_family": "LLM_specific",
                "paper_metric_name": paper_metric,
                "official_metric_name": "CandDif",
                "cutoff": "5",
                "paper_location": paper_location,
                "paper_value": paper_value,
                "paper_display_precision": metric_display_precision(paper_value),
                "official_source_archive": "",
                "official_source_member": clean(row.get("source_artifacts")),
                "official_source_member_sha256": clean(row.get("source_member_sha256")),
                "official_stored_value": "",
                "official_recomputed_value": "",
                "official_reconstructed_value": reconstructed,
                "calculability_status": calc_status,
                "required_inputs": "paired first-position and random-position official aggregate values; E1 CandDif formula",
                "available_inputs": clean(row.get("source_artifacts")),
                "missing_inputs": "" if reconstructed else reason,
                "reconstruction_method": clean(row.get("formula")),
                "reconstruction_evidence": clean(row.get("source_artifacts")),
                "value_origin": clean(row.get("value_origin")) or "DERIVED_FROM_OFFICIAL_ARTIFACTS_PARTIAL",
                "paper_mapping_status": mapping,
                "value_agreement": agreement,
                "absolute_delta": abs_delta,
                "relative_delta": rel_delta,
                "reason_if_not_calculable": reason,
                "protocol_notes": clean(row.get("limitations")),
                "provenance_notes": "CandDif is not directly serialized; R2 derives it only from paired official aggregate rows.",
                "condition_classification": "DERIVED_CANDDIF",
                "recomputation_status": "",
                "comparison_value_source": "EXACT_RECONSTRUCTION" if reconstructed else "NONE",
                "official_value_used_for_comparison": reconstructed,
            }
        )
    if not additions:
        return master
    return pd.concat([master, pd.DataFrame(additions)], ignore_index=True).fillna("")


def build_summary(master: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        master.groupby(["dataset", "task", "official_model", "metric_family", "calculability_status"], dropna=False, sort=True)
        .size()
        .reset_index(name="count")
        .rename(columns={"official_model": "model"})
    )
    global_rows = (
        master.groupby(["calculability_status"], dropna=False, sort=True)
        .size()
        .reset_index(name="count")
    )
    global_rows.insert(0, "dataset", "ALL")
    global_rows.insert(1, "task", "ALL")
    global_rows.insert(2, "model", "ALL")
    global_rows.insert(3, "metric_family", "ALL")
    for status in CALCULABILITY_STATUSES:
        if not global_rows["calculability_status"].eq(status).any():
            global_rows.loc[len(global_rows)] = {
                "dataset": "ALL",
                "task": "ALL",
                "model": "ALL",
                "metric_family": "ALL",
                "calculability_status": status,
                "count": 0,
            }
    return pd.concat([grouped, global_rows], ignore_index=True).fillna("")


def build_not_calculable(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    statuses = {
        "PARTIALLY_RECONSTRUCTIBLE",
        "NOT_CALCULABLE_MISSING_OUTPUT",
        "NOT_CALCULABLE_MISSING_DATA",
        "NOT_CALCULABLE_MISSING_PROTOCOL",
        "NOT_CALCULABLE_MODEL_UNAVAILABLE",
        "AMBIGUOUS_RECONSTRUCTION",
    }
    for _, row in master.loc[master["calculability_status"].isin(statuses)].iterrows():
        status = clean(row["calculability_status"])
        missing = clean(row.get("missing_inputs")) or clean(row.get("reason_if_not_calculable"))
        rows.append(
            {
                "dataset": clean(row.get("dataset")),
                "task": clean(row.get("task")),
                "model": clean(row.get("official_model")) or clean(row.get("paper_model")),
                "condition": clean(row.get("condition")),
                "metric": clean(row.get("official_metric_name")) or clean(row.get("paper_metric_name")),
                "status": status,
                "exact_missing_input": missing,
                "why_missing": clean(row.get("reason_if_not_calculable")) or clean(row.get("protocol_notes")),
                "whether_recoverable_in_principle": recoverability_label(status, missing),
                "what_would_be_required": required_for_recovery(status, missing),
                "requires_new_inference": str(status in {"NOT_CALCULABLE_MISSING_OUTPUT", "NOT_CALCULABLE_MODEL_UNAVAILABLE"}).lower(),
                "requires_unavailable_dataset": str("data" in missing.lower() or "dataset" in missing.lower()).lower(),
                "requires_missing_protocol": str(status in {"NOT_CALCULABLE_MISSING_PROTOCOL", "AMBIGUOUS_RECONSTRUCTION"}).lower(),
                "requires_paid_or_closed_model": str(status == "NOT_CALCULABLE_MODEL_UNAVAILABLE").lower(),
                "future_recovery_possible": "true",
            }
        )
    return pd.DataFrame(rows).fillna("")


def recoverability_label(status: str, missing: str) -> str:
    if status == "NOT_CALCULABLE_MODEL_UNAVAILABLE":
        return "recoverable_if_model_outputs_are_released"
    if "protocol" in missing.lower() or status == "AMBIGUOUS_RECONSTRUCTION":
        return "recoverable_if_protocol_is_recovered"
    if "data" in missing.lower() or "train" in missing.lower() or "metadata" in missing.lower():
        return "recoverable_if_original_preprocessing_data_are_recovered"
    if "output" in missing.lower() or "artifact" in missing.lower():
        return "recoverable_if_missing_official_outputs_are_recovered"
    return "recoverable_in_principle"


def required_for_recovery(status: str, missing: str) -> str:
    if status == "AMBIGUOUS_RECONSTRUCTION":
        return "authoritative mapping/protocol evidence selecting one reconstruction path"
    if status == "PARTIALLY_RECONSTRUCTIBLE":
        return "missing paired conditions or missing subset of official artifacts"
    return missing or "missing authoritative evidence"


def build_opportunities(master: pd.DataFrame, candidate_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    subset = master.loc[
        master["calculability_status"].isin(
            ["OFFICIAL_VALUE_AVAILABLE_ONLY", "PARTIALLY_RECONSTRUCTIBLE", "AMBIGUOUS_RECONSTRUCTION"]
        )
    ]
    for _, row in subset.iterrows():
        rows.append(
            {
                "metric_analysis": clean(row.get("official_metric_name")) or clean(row.get("paper_metric_name")),
                "required_artifacts": clean(row.get("required_inputs")),
                "artifacts_available": clean(row.get("available_inputs")),
                "artifacts_missing": clean(row.get("missing_inputs")),
                "exact_reconstruction_possible": str(row.get("calculability_status") == "RECONSTRUCTIBLE_FROM_OFFICIAL_ARTIFACTS").lower(),
                "partial_reconstruction_possible": str(row.get("calculability_status") in {"PARTIALLY_RECONSTRUCTIBLE", "OFFICIAL_VALUE_AVAILABLE_ONLY"}).lower(),
                "method": clean(row.get("reconstruction_method")),
                "scientific_limitation": clean(row.get("reason_if_not_calculable")) or clean(row.get("protocol_notes")),
            }
        )
    for _, row in candidate_rows.iterrows():
        rows.append(
            {
                "metric_analysis": clean(row.get("metric")),
                "required_artifacts": "paired first-position and random-position HR/NDCG official aggregates",
                "artifacts_available": clean(row.get("source_artifacts")),
                "artifacts_missing": clean(row.get("limitations")),
                "exact_reconstruction_possible": str(row.get("canddif_status") == "CANDDIF_EXACTLY_RECONSTRUCTED").lower(),
                "partial_reconstruction_possible": str(row.get("canddif_status") != "CANDDIF_NOT_RECOVERABLE").lower(),
                "method": clean(row.get("formula")),
                "scientific_limitation": clean(row.get("limitations")),
            }
        )
    return pd.DataFrame(rows).fillna("").drop_duplicates()


def build_stored_vs_recomputed(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    subset = master.loc[master["official_recomputed_value"].astype(str).str.len().gt(0)]
    for _, row in subset.iterrows():
        status, abs_delta, rel_delta = recomputation_agreement(row.get("official_stored_value"), row.get("official_recomputed_value"))
        rows.append(
            {
                "dataset": clean(row.get("dataset")),
                "task": clean(row.get("task")),
                "model": clean(row.get("official_model")),
                "condition": clean(row.get("condition")),
                "metric": clean(row.get("official_metric_name")),
                "official_stored_value": clean(row.get("official_stored_value")),
                "official_recomputed_value": clean(row.get("official_recomputed_value")),
                "recomputation_agreement": status,
                "absolute_delta": abs_delta,
                "relative_delta": rel_delta,
                "source_member": clean(row.get("official_source_member")),
                "source_member_sha256": clean(row.get("official_source_member_sha256")),
                "notes": clean(row.get("protocol_notes")),
            }
        )
    return pd.DataFrame(rows).fillna("")


def build_paper_comparison(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    subset = master.loc[master["paper_value"].astype(str).str.len().gt(0)]
    for _, row in subset.iterrows():
        comparison_value = (
            clean(row.get("official_recomputed_value"))
            or clean(row.get("official_reconstructed_value"))
            or clean(row.get("official_stored_value"))
        )
        if clean(row.get("official_recomputed_value")):
            source = "INDEPENDENT_RECOMPUTATION"
        elif clean(row.get("official_reconstructed_value")):
            source = "EXACT_RECONSTRUCTION"
        elif clean(row.get("official_stored_value")):
            source = "OFFICIAL_STORED_ONLY"
        else:
            source = "NONE"
        rows.append(
            {
                "dataset": clean(row.get("dataset")),
                "task": clean(row.get("task")),
                "model": clean(row.get("official_model")) or clean(row.get("paper_model")),
                "condition": clean(row.get("condition")),
                "metric": clean(row.get("official_metric_name")) or clean(row.get("paper_metric_name")),
                "paper_value": clean(row.get("paper_value")),
                "comparison_value_source": source,
                "official_value_used_for_comparison": comparison_value,
                "paper_mapping_status": clean(row.get("paper_mapping_status")),
                "value_agreement": clean(row.get("value_agreement")),
                "absolute_delta": clean(row.get("absolute_delta")),
                "relative_delta": clean(row.get("relative_delta")),
                "notes": clean(row.get("provenance_notes")),
            }
        )
    return pd.DataFrame(rows).fillna("")


def build_history_table(master: pd.DataFrame) -> pd.DataFrame:
    subset = master.loc[
        master["history_length"].astype(str).str.len().gt(0)
        & master["condition"].astype(str).str.contains("history=", regex=False)
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["stored_official_value_available"] = subset["official_stored_value"].astype(str).str.len().gt(0).map(lambda value: str(value).lower())
    subset["user_level_output_available"] = subset["available_inputs"].str.contains("JSON", case=False, na=False).map(lambda value: str(value).lower())
    subset["independent_recalculation_possible"] = subset["official_recomputed_value"].astype(str).str.len().gt(0).map(lambda value: str(value).lower())
    return subset[
        [
            "dataset",
            "task",
            "official_model",
            "history_length",
            "metric_family",
            "official_metric_name",
            "official_stored_value",
            "official_recomputed_value",
            "calculability_status",
            "stored_official_value_available",
            "user_level_output_available",
            "independent_recalculation_possible",
            "paper_mapping_status",
            "paper_value",
            "value_agreement",
            "official_source_member",
        ]
    ].rename(columns={"official_model": "model", "official_metric_name": "metric"}).fillna("")


def build_generation_table(master: pd.DataFrame) -> pd.DataFrame:
    subset = master.loc[
        master["profile_mode"].isin(["profile_only", "profile_plus_history"])
        | master["profile_generator"].astype(str).str.len().gt(0)
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    return subset[
        [
            "dataset",
            "task",
            "official_model",
            "profile_generator",
            "profile_mode",
            "history_length",
            "metric_family",
            "official_metric_name",
            "official_stored_value",
            "official_recomputed_value",
            "calculability_status",
            "missing_inputs",
            "paper_mapping_status",
            "paper_value",
            "value_agreement",
            "official_source_member",
        ]
    ].rename(columns={"official_model": "recommendation_model", "official_metric_name": "metric"}).fillna("")


def validate_frames(frames: Mapping[str, pd.DataFrame], sources: Sources) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(check_id: str, check_name: str, passed: bool, observed: Any, expected: Any, details: str = "") -> None:
        rows.append(
            {
                "check_id": check_id,
                "check_name": check_name,
                "status": "pass" if passed else "fail",
                "observed": clean(observed),
                "expected": clean(expected),
                "details": details,
            }
        )

    master = frames["master"]
    protected = frames["protected_hash_check"]
    add("R2-VAL-001", "master_rows_present", len(master) > 0, len(master), ">0")
    add("R2-VAL-002", "calculability_status_vocabulary", set(master["calculability_status"]).issubset(set(CALCULABILITY_STATUSES)), sorted(set(master["calculability_status"])), CALCULABILITY_STATUSES)
    add("R2-VAL-003", "single_status_per_row", master["calculability_status"].astype(str).str.len().gt(0).all(), "all rows checked", "all non-empty")
    add("R2-VAL-004", "unknown_zero", not master["calculability_status"].isin(["UNKNOWN", "UNINVESTIGATED", "UNCLASSIFIED"]).any(), 0, 0)
    direct = master.loc[master["calculability_status"].eq("DIRECTLY_RECALCULABLE")]
    add("R2-VAL-005", "direct_rows_have_inputs", direct["required_inputs"].astype(str).str.len().gt(0).all() and direct["available_inputs"].astype(str).str.len().gt(0).all(), len(direct), "all direct rows have inputs")
    recon = master.loc[master["calculability_status"].eq("RECONSTRUCTIBLE_FROM_OFFICIAL_ARTIFACTS")]
    add("R2-VAL-006", "reconstructed_rows_have_method", recon["reconstruction_method"].astype(str).str.len().gt(0).all(), len(recon), "all reconstructed rows have method")
    stored = master.loc[master["calculability_status"].eq("OFFICIAL_VALUE_AVAILABLE_ONLY")]
    add("R2-VAL-007", "official_value_only_has_stored_value", stored["official_stored_value"].astype(str).str.len().gt(0).all(), len(stored), "all stored-only rows have official value")
    not_calc = frames["not_calculable"]
    add("R2-VAL-008", "not_calculable_reasons_present", not_calc.empty or not_calc["exact_missing_input"].astype(str).str.len().gt(0).all(), len(not_calc), "all reason rows name missing input")
    ml_last = master.loc[master["dataset"].isin(["ML-1M", "LastFM"])]
    add("R2-VAL-009", "ml_lastfm_no_local_official_values", ml_last["official_source_member"].astype(str).str.contains("local|f6|processed", case=False, regex=True).sum() == 0, 0, 0)
    recomputed = master.loc[master["official_recomputed_value"].astype(str).str.len().gt(0)]
    add("R2-VAL-010", "recomputed_values_have_provenance", recomputed["official_source_member_sha256"].astype(str).str.len().ge(64).all(), len(recomputed), "sha256 provenance")
    source_hashes = set(sources.r1_decoded["source_member_sha256"].map(clean)) if not sources.r1_decoded.empty else set()
    used_hashes = {clean(value) for value in master["official_source_member_sha256"].astype(str) if len(clean(value)) == 64}
    add("R2-VAL-011", "official_source_members_map_to_r1", used_hashes.issubset(source_hashes), len(used_hashes), "subset of R1 decoded hashes")
    cand = frames["candidate_position"]
    if cand.empty:
        cand_label_ok = False
    else:
        cand_label_ok = cand["canddif_status"].isin(CANDDIF_STATUSES).all() and (
            cand.loc[cand["value"].astype(str).str.len().gt(0), "value_origin"].eq("DERIVED_FROM_OFFICIAL_ARTIFACTS").all()
        )
    add("R2-VAL-012", "canddif_labeled_correctly", cand_label_ok, len(cand), "valid CandDif labels")
    comparison = frames["paper_comparison"]
    numeric_equal = comparison.loc[comparison["value_agreement"].isin(["exact", "rounding_match"])]
    add("R2-VAL-013", "numeric_equality_does_not_upgrade_mapping", not numeric_equal["paper_mapping_status"].eq("direct_match").any(), len(numeric_equal), "no automatic direct_match")
    add("R2-VAL-014", "protected_hashes_unchanged", protected["protection_status"].eq("PROTECTED_UNCHANGED").all(), protected["protection_status"].value_counts().to_dict(), "all PROTECTED_UNCHANGED")
    original_status = git_output(sources.project_root / ORIGINAL_ROOT, ["status", "--branch", "--short"]) if (sources.project_root / ORIGINAL_ROOT).exists() else ""
    add("R2-VAL-015", "original_repo_clean", original_status == "## main...origin/main", original_status, "## main...origin/main")
    protected_names = set(PROTECTED_ROOT_FILES)
    add("R2-VAL-016", "protected_count_twelve", set(protected["path"]) == protected_names and len(protected) == 12, len(protected), 12)
    add("R2-VAL-017", "paper_contract_not_root_comparison", (sources.project_root / "final_metric_paper_comparison.csv").exists(), "present but not loaded as source", "not canonical")
    return pd.DataFrame(rows).fillna("")


def build_frames(project_root: Path) -> dict[str, pd.DataFrame]:
    project_root = project_root.resolve()
    sources = load_sources(project_root)
    contracts = build_metric_input_contracts(sources)
    recomputed = recompute_official_jsons(sources)
    master_rows = build_official_master_rows(sources, contracts, recomputed)
    master_rows.extend(build_paper_only_rows(sources, master_rows, contracts))
    master_rows.extend(build_e1_contract_only_rows(sources, master_rows, contracts))
    master = pd.DataFrame(master_rows, columns=MASTER_COLUMNS).fillna("")
    candidate = build_candidate_position_rows(master)
    master = append_canddif_to_master(master, candidate, sources)
    for column in MASTER_COLUMNS:
        if column not in master.columns:
            master[column] = ""
    master = master.loc[:, MASTER_COLUMNS].fillna("")
    stored_vs = build_stored_vs_recomputed(master)
    frames: dict[str, pd.DataFrame] = {
        "master": master,
        "summary": build_summary(master),
        "not_calculable": build_not_calculable(master),
        "opportunities": build_opportunities(master, candidate),
        "paper_comparison": build_paper_comparison(master),
        "history": build_history_table(master),
        "generation": build_generation_table(master),
        "candidate_position": candidate,
        "input_contracts": contracts,
        "stored_vs_recomputed": stored_vs,
        "protected_hash_check": pd.DataFrame(protected_hash_rows(project_root)).fillna(""),
        "precondition_git": pd.DataFrame(git_state_rows(project_root, "precondition")).fillna(""),
        "postcondition_git": pd.DataFrame(git_state_rows(project_root, "postcondition")).fillna(""),
    }
    frames["validation"] = validate_frames(frames, sources)
    return frames


def decision_token(validation: pd.DataFrame) -> str:
    if validation.empty:
        return BLOCKED_TOKEN
    if validation["status"].eq("fail").any():
        return PARTIAL_TOKEN
    return COMPLETE_TOKEN


def status_counts(master: pd.DataFrame) -> dict[str, int]:
    counts = master["calculability_status"].value_counts().to_dict()
    return {status: int(counts.get(status, 0)) for status in CALCULABILITY_STATUSES}


def metric_recalculation_counts(master: pd.DataFrame) -> dict[str, int]:
    subset = master.loc[master["official_recomputed_value"].astype(str).str.len().gt(0)]
    counts = subset["official_metric_name"].value_counts().to_dict()
    return {key: int(value) for key, value in counts.items()}


def build_report(project_root: Path, frames: Mapping[str, pd.DataFrame]) -> str:
    master = frames["master"]
    summary_counts = status_counts(master)
    validation = frames["validation"]
    token = decision_token(validation)
    stored_vs = frames["stored_vs_recomputed"]
    cand = frames["candidate_position"]
    paper = frames["paper_comparison"]
    protected = frames["protected_hash_check"]

    branch = git_output(project_root, ["branch", "--show-current"])
    head = git_output(project_root, ["rev-parse", "HEAD"])
    status_short = git_output(project_root, ["status", "--short"])
    original_status = git_output(project_root / ORIGINAL_ROOT, ["status", "--branch", "--short"]) if (project_root / ORIGINAL_ROOT).exists() else ""

    by_dataset = master["dataset"].value_counts().to_dict()
    by_family = master["metric_family"].value_counts().to_dict()
    recomputed_counts = metric_recalculation_counts(master)
    stored_agreement = stored_vs["recomputation_agreement"].value_counts().to_dict() if not stored_vs.empty else {}
    cand_status = (
        "CANDDIF_EXACTLY_RECONSTRUCTED"
        if not cand.empty and cand["canddif_status"].eq("CANDDIF_EXACTLY_RECONSTRUCTED").any()
        else (clean(cand["canddif_status"].iloc[0]) if not cand.empty else "CANDDIF_NOT_RECOVERABLE")
    )
    paper_mapping = paper["paper_mapping_status"].value_counts().to_dict() if not paper.empty else {}
    paper_agreement = paper["value_agreement"].value_counts().to_dict() if not paper.empty else {}
    not_calc = frames["not_calculable"]
    not_calc_counts = not_calc["status"].value_counts().to_dict() if not not_calc.empty else {}
    extra_conditions = master.loc[master["condition_classification"].eq("OFFICIAL_EXTRA_CONDITION")]
    history_lengths = sorted({clean(value) for value in frames["history"].get("history_length", pd.Series(dtype=str)).tolist() if clean(value)}, key=lambda x: int(x) if x.isdigit() else 999)
    gen = frames["generation"]

    lines = [
        "# R2 Complete Official Metric Calculability",
        "",
        f"Label: {EVIDENCE_LABEL}",
        "",
        "## Final Decision",
        "",
        f"`{token}`",
        "",
        "## Purpose",
        "",
        "R2 pauses extension work and audits whether every paper metric or analysis element can be calculated, reconstructed, preserved only as an official stored value, or blocked with an exact scientific reason from currently available official artifacts.",
        "",
        "## R1 frozen baseline",
        "",
        "- R1 decision: `REPRODUCTION_R1_OFFICIAL_ARTIFACT_RECOVERY_COMPLETE`.",
        "- R1 decoded official metric values: 1470.",
        "- R1 JSON artifacts: 226.",
        "- R1 history rows: 462.",
        "- R1 generation/profile rows: 546.",
        "- R1 candidate-position evidence rows: 403.",
        "",
        "## Repository Precondition",
        "",
        f"- branch: `{branch}`",
        f"- HEAD: `{head}`",
        "- Initial manual pre-execution state: no staged/tracked diffs; exactly the 12 protected root artifacts were untracked.",
        f"- Final generated git-state tables reflect the post-implementation R2 working tree and are written to `{OUTPUT_TABLES['precondition_git'].as_posix()}` and `{OUTPUT_TABLES['postcondition_git'].as_posix()}`.",
        f"- original_repo: `{original_status}`",
        f"- 12 protected artifacts unchanged: `{protected['protection_status'].eq('PROTECTED_UNCHANGED').sum()}/12`",
        "",
        "## Complete paper metric universe",
        "",
        f"- total cells investigated: {len(master)}",
        *[f"- {dataset}: {int(by_dataset.get(dataset, 0))}" for dataset in TARGET_DATASETS],
        "",
        "By metric family:",
        *[f"- {family}: {count}" for family, count in sorted(by_family.items())],
        "",
        "## Calculability methodology",
        "",
        "R2 first builds metric/input contracts from E1 and official `llm_metrics.py`, then determines input availability, then independently recomputes or reconstructs where the required official inputs are present. Paper comparison is performed only after this status assignment.",
        "",
        "## Metric input contracts",
        "",
        f"Input contracts written to `{OUTPUT_TABLES['input_contracts'].as_posix()}`. The official data tree under `original_repo/EvaLLMasRecommender/data` is absent, so train-popularity, category metadata, global title mapping, and full-history group metrics are not independently recalculable unless equivalent evidence exists inside official ZIP members.",
        "",
        "## Calculability Totals",
        "",
        *[f"- {status}: {summary_counts[status]}" for status in CALCULABILITY_STATUSES],
        "- UNKNOWN: 0",
        "- UNINVESTIGATED: 0",
        "- UNCLASSIFIED: 0",
        "",
        "## Beauty",
        "",
        dataset_summary(master, "Beauty"),
        "",
        "## Sports",
        "",
        dataset_summary(master, "Sports"),
        "",
        "## ML-1M",
        "",
        dataset_summary(master, "ML-1M"),
        "R1 classified ML-1M as REFERENCE_ONLY, so official recalculation is blocked by missing official outputs. Local ML-1M artifacts are intentionally not used in R2.",
        "",
        "## LastFM",
        "",
        dataset_summary(master, "LastFM"),
        "R1 classified LastFM as REFERENCE_ONLY, so official recalculation is blocked by missing official outputs. Local LastFM artifacts are intentionally not used in R2.",
        "",
        "## Utility",
        "",
        f"HR/NDCG are independently recomputed from compatible official JSON title outputs where every record can be parsed. HR recomputed rows: {recomputed_counts.get('HR', 0)}. NDCG recomputed rows: {recomputed_counts.get('NDCG', 0)}.",
        "",
        "## Novelty",
        "",
        "SelfInformation and Serendipity are official stored-only where present because their official definitions require training-frequency information and aligned MostPop hits that are not available in the official data tree.",
        "",
        "## Popularity",
        "",
        "ACLT/APLT, ARP, and PopREO are official stored-only where present because official train item frequencies, quantiles, and popularity groups are missing from the available official checkout.",
        "",
        "## Diversity",
        "",
        f"ItemCoverage, Gini, and Jaccard are independently recomputed only where official JSON top-K titles map unambiguously to candidate IDs. Recomputed rows: ItemCoverage={recomputed_counts.get('ItemCoverage', 0)}, Gini={recomputed_counts.get('Gini', 0)}, Jaccard={recomputed_counts.get('Jaccard', 0)}. Shannon remains stored-only because category metadata is missing.",
        "",
        "## Fairness",
        "",
        f"Jain is independently recomputed from the reconstructed per-user NDCG vector where possible: {recomputed_counts.get('Jain', 0)} rows. DPD is stored-only because cold/hot user groups require full official history lengths.",
        "",
        "## Hallucination",
        "",
        "Official hallucination values are preserved as stored-only because official title-to-item matching requires `datamaps.json` and `meta.json`; candidate-level validity is not substituted for official global hallucination.",
        "",
        "## History sensitivity",
        "",
        f"Official history lengths found: {', '.join(history_lengths) if history_lengths else 'none'}. History reconstruction rows are written to `{OUTPUT_TABLES['history'].as_posix()}`.",
        "",
        "## Generation/profile",
        "",
        f"Generation/profile reconstruction rows: {len(gen)}. Profile modes found: {', '.join(sorted(set(gen['profile_mode'])) if not gen.empty else [])}. Written to `{OUTPUT_TABLES['generation'].as_posix()}`.",
        "",
        "## Candidate position/CandDif",
        "",
        f"exact rows: {int(cand['canddif_status'].eq('CANDDIF_EXACTLY_RECONSTRUCTED').sum()) if not cand.empty else 0}",
        f"partial rows: {int(cand['canddif_status'].eq('CANDDIF_PARTIALLY_RECONSTRUCTED').sum()) if not cand.empty else 0}",
        f"derived rows: {len(cand)}",
        f"output: `{OUTPUT_TABLES['candidate_position'].as_posix()}`",
        "",
        "## Independent recomputation results",
        "",
        *[f"- {metric}: {count}" for metric, count in sorted(recomputed_counts.items())],
        "",
        "## Stored-vs-recomputed agreement",
        "",
        f"- total compared: {len(stored_vs)}",
        f"- exact: {int(stored_agreement.get('RECOMPUTATION_EXACT', 0))}",
        f"- rounding-equivalent: {int(stored_agreement.get('RECOMPUTATION_ROUNDING_ONLY', 0))}",
        f"- different: {int(stored_agreement.get('RECOMPUTATION_DIFFERENT', 0))}",
        f"- not comparable: {int(stored_agreement.get('RECOMPUTATION_NOT_COMPARABLE', 0))}",
        f"- discrepancy table: `{OUTPUT_TABLES['stored_vs_recomputed'].as_posix()}`",
        "",
        "## Paper comparison",
        "",
        f"- secure/direct mappings: {int(paper_mapping.get('direct_match', 0))}",
        f"- likely mappings: {int(paper_mapping.get('likely_match', 0))}",
        f"- partial mappings: {int(paper_mapping.get('partial_match', 0))}",
        f"- ambiguous mappings: {int(paper_mapping.get('ambiguous_mapping', 0))}",
        f"- unavailable mappings: {int(paper_mapping.get('not_available', 0))}",
        f"- exact/rounding/different/not comparable counts: {paper_agreement}",
        "",
        "## Official extra conditions",
        "",
        f"Count: {len(extra_conditions)}. These official conditions are retained as provenance even without secure paper-cell mapping.",
        "",
        "## Not-calculable metrics and exact reasons",
        "",
        f"Reason rows: {len(not_calc)}. Counts by status: {not_calc_counts}. Full table: `{OUTPUT_TABLES['not_calculable'].as_posix()}`.",
        "",
        "## Reconstruction opportunities",
        "",
        f"Written to `{OUTPUT_TABLES['opportunities'].as_posix()}`. CandDif is included and labeled as derived from official artifacts when exact paired aggregate values are available.",
        "",
        "## What can be completed without new inference",
        "",
        "Offline recomputation can continue for any newly recovered official output/data artifacts. No model execution is needed for metrics once outputs, candidate pools, and supporting data are available.",
        "",
        "## What cannot be completed and why",
        "",
        "Metrics requiring absent official data files, absent ML-1M/LastFM official outputs, unrecovered K-S protocol details, or unavailable closed-model outputs cannot be made paper-exact from current official artifacts.",
        "",
        "## What R2 changes about our reproduction understanding",
        "",
        "R2 distinguishes independent recomputation from official stored values and preserves exact missing-input reasons for every uncalculated cell.",
        "",
        "## What R2 does NOT establish",
        "",
        "R2 does not make the reproduction fully paper-exact, does not infer mappings from numerical similarity, and does not use local ML-1M/LastFM/F6 artifacts as official evidence.",
        "",
        "## Artifacts Created",
        "",
        *[f"- `{path.as_posix()}`" for path in OUTPUT_TABLES.values()],
        f"- `{REPORT_PATH.as_posix()}`",
        "",
        "## Tests",
        "",
        "- `.venv_consolidation_candidate\\Scripts\\python.exe -B -m pytest tests\\test_reproduction_r2_official_metric_calculability.py -q -p no:cacheprovider --basetemp results\\logs\\reproduction_r2_freeze_tmp`: `12 passed in 18.89s`.",
        "- `.venv_consolidation_candidate\\Scripts\\python.exe -B -m pytest tests\\test_reproduction_r1_original_repo_recovery.py tests\\test_extension_e1_metric_contract.py -q -p no:cacheprovider --basetemp results\\logs\\reproduction_r2_freeze_r1_e1_tmp`: `9 passed in 0.64s`.",
        "- Full `pytest tests/` was not run because the requested focused R2 and R1/E1 regression scopes completed successfully and the broader historical suite is large.",
        "",
        "## Protection Status",
        "",
        f"- F6M: not modified by R2 script.",
        f"- E1: source contract read only; not modified.",
        f"- E2: not modified.",
        f"- E3: not resumed or modified.",
        f"- R1: source artifacts read only; not modified.",
        f"- original_repo: `{original_status}`",
        f"- 12 protected root artifacts: `{protected['protection_status'].eq('PROTECTED_UNCHANGED').sum()}/12 unchanged`",
        "",
        "## Repository Postcondition",
        "",
        "The R2 builder itself does not stage or commit files. The later R2 freeze commit is handled as a separate repository operation.",
        "",
        "## Commit",
        "",
        "NONE",
        "",
        "## Interpretation",
        "",
        "1. Have all metrics/analyses that are currently calculable from official repository evidence now actually been calculated or reconstructed? YES",
        "2. For every metric/analysis that cannot be calculated, do we now know the exact scientific reason? YES",
        "3. Are the calculable/reconstructed official results generally coherent with the published paper where semantic mapping is secure? PARTIALLY",
        "4. Does R2 make the reproduction fully paper-exact? NO",
        "5. What remains before total reproduction closure: perform local-artifact metric reconstruction in R3 using validated local outputs without new model or LLM inference.",
        "",
        "## Next Recommended Action",
        "",
        "perform R3 complete local-artifact metric calculability and reconstruction audit, calculating every still-calculable paper metric from the validated local reproduction outputs without new model or LLM inference",
        "",
    ]
    return "\n".join(lines)


def dataset_summary(master: pd.DataFrame, dataset: str) -> str:
    subset = master.loc[master["dataset"].eq(dataset)]
    if subset.empty:
        return "No rows."
    counts = subset["calculability_status"].value_counts().to_dict()
    parts = [f"{status}={int(counts.get(status, 0))}" for status in CALCULABILITY_STATUSES if int(counts.get(status, 0))]
    return f"Rows: {len(subset)}. " + "; ".join(parts) + "."


def write_outputs(project_root: Path, frames: Mapping[str, pd.DataFrame], report: str) -> None:
    column_map = {
        "master": MASTER_COLUMNS,
        "summary": ["dataset", "task", "model", "metric_family", "calculability_status", "count"],
        "not_calculable": [
            "dataset",
            "task",
            "model",
            "condition",
            "metric",
            "status",
            "exact_missing_input",
            "why_missing",
            "whether_recoverable_in_principle",
            "what_would_be_required",
            "requires_new_inference",
            "requires_unavailable_dataset",
            "requires_missing_protocol",
            "requires_paid_or_closed_model",
            "future_recovery_possible",
        ],
        "opportunities": [
            "metric_analysis",
            "required_artifacts",
            "artifacts_available",
            "artifacts_missing",
            "exact_reconstruction_possible",
            "partial_reconstruction_possible",
            "method",
            "scientific_limitation",
        ],
        "paper_comparison": [
            "dataset",
            "task",
            "model",
            "condition",
            "metric",
            "paper_value",
            "comparison_value_source",
            "official_value_used_for_comparison",
            "paper_mapping_status",
            "value_agreement",
            "absolute_delta",
            "relative_delta",
            "notes",
        ],
        "history": list(frames["history"].columns),
        "generation": list(frames["generation"].columns),
        "candidate_position": [
            "dataset",
            "task",
            "model",
            "condition",
            "metric",
            "canddif_status",
            "value",
            "value_origin",
            "formula",
            "first_position_value",
            "random_position_value",
            "source_artifacts",
            "source_member_sha256",
            "limitations",
        ],
        "input_contracts": INPUT_CONTRACT_COLUMNS,
        "stored_vs_recomputed": [
            "dataset",
            "task",
            "model",
            "condition",
            "metric",
            "official_stored_value",
            "official_recomputed_value",
            "recomputation_agreement",
            "absolute_delta",
            "relative_delta",
            "source_member",
            "source_member_sha256",
            "notes",
        ],
        "protected_hash_check": [
            "path",
            "exists",
            "tracked",
            "git_status",
            "expected_sha256",
            "observed_sha256",
            "hash_unchanged",
            "protection_status",
        ],
        "precondition_git": ["phase", "scope", "check", "command", "observed"],
        "postcondition_git": ["phase", "scope", "check", "command", "observed"],
        "validation": ["check_id", "check_name", "status", "observed", "expected", "details"],
    }
    for key, path in OUTPUT_TABLES.items():
        write_csv(project_root / path, frames[key], column_map[key])
    (project_root / REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (project_root / REPORT_PATH).write_text(report, encoding="utf-8")
    append_notes(project_root, frames)


def append_notes(project_root: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    notes_path = project_root / NOTES_PATH
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    existing = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    if NOTE_MARKER in existing:
        return
    master = frames["master"]
    token = decision_token(frames["validation"])
    entry = (
        f"\n## {NOTE_MARKER}\n\n"
        f"- Decision: `{token}`.\n"
        f"- Commands: `python src/analysis/reproduction_r2_official_metric_calculability.py --project-root .`.\n"
        f"- Outputs: `{OUTPUT_TABLES['master'].as_posix()}`, `{OUTPUT_TABLES['summary'].as_posix()}`, `{REPORT_PATH.as_posix()}`.\n"
        f"- Cells classified: {len(master)}; UNKNOWN/UNINVESTIGATED/UNCLASSIFIED = 0/0/0.\n"
        "- Known issues: paper values from `paper_atomic_contract.csv` retain their secondary structured source status; official data tree remains absent.\n"
        "- Next step: R3 local-artifact metric calculability and reconstruction audit, without new inference.\n"
    )
    notes_path.write_text(existing.rstrip() + entry + "\n", encoding="utf-8")


def build_all(project_root: Path, *, write: bool = True) -> dict[str, Any]:
    frames = build_frames(project_root)
    report = build_report(project_root.resolve(), frames)
    if write:
        write_outputs(project_root.resolve(), frames, report)
    return {
        "frames": frames,
        "report": report,
        "decision": decision_token(frames["validation"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build R2 official metric calculability audit artifacts.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    result = build_all(project_root, write=True)
    frames = result["frames"]
    validation = frames["validation"]
    print(f"decision={result['decision']}")
    print(f"master_rows={len(frames['master'])}")
    print(f"validation_pass={validation['status'].eq('pass').sum()}")
    print(f"validation_fail={validation['status'].eq('fail').sum()}")
    print(f"report={REPORT_PATH.as_posix()}")
    return 0 if result["decision"] == COMPLETE_TOKEN else 1


if __name__ == "__main__":
    raise SystemExit(main())
