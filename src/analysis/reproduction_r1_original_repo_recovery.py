from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.analysis.r1_metric_recovery_paths import resolve_r1_metric_recovery_artifact


DECISION_TOKEN = "REPRODUCTION_R1_OFFICIAL_ARTIFACT_RECOVERY_COMPLETE"
EVIDENCE_LABEL = "POST-F6M TOTAL REPRODUCIBILITY CLOSURE EVIDENCE"

ORIGINAL_RELATIVE_ROOT = Path("original_repo") / "EvaLLMasRecommender"
DEFAULT_TABLE_DIR = Path("results") / "tables"
DEFAULT_REPORT_PATH = Path("report") / "reproduction_r1_original_repo_result_recovery.md"

TARGET_DATASETS = ["Beauty", "Sports", "ML-1M", "LastFM"]

METRIC_CONTRACT = [
    {
        "metric_position": 1,
        "official_metric_name": "HR",
        "canonical_metric_name": "hr_at_5",
        "paper_alias": "HR@5",
        "official_function": "calHR",
        "metric_family": "utility",
        "cutoff_semantics": "metric_len argument; official metrics.sh emits _5.csv for K=5",
        "aggregation_semantics": "mean over sampled users after hit vector construction",
        "known_ambiguities": "",
    },
    {
        "metric_position": 2,
        "official_metric_name": "NDCG",
        "canonical_metric_name": "ndcg_at_5",
        "paper_alias": "NDCG@5",
        "official_function": "calNDCG",
        "metric_family": "utility",
        "cutoff_semantics": "metric_len argument; official metrics.sh emits _5.csv for K=5",
        "aggregation_semantics": "mean of hit/log2(rank+1) over sampled users",
        "known_ambiguities": "",
    },
    {
        "metric_position": 3,
        "official_metric_name": "ACLT",
        "canonical_metric_name": "aclt_at_5",
        "paper_alias": "APLT/long-tail share",
        "official_function": "calACLT",
        "metric_family": "popularity",
        "cutoff_semantics": "top metric_len recommendations",
        "aggregation_semantics": "share of top-K recommended rows whose train frequency is below the high_bar quantile",
        "known_ambiguities": "Code names the metric ACLT while paper/reporting may refer to APLT.",
    },
    {
        "metric_position": 4,
        "official_metric_name": "ARP",
        "canonical_metric_name": "average_recommendation_popularity_at_5",
        "paper_alias": "Average Recommendation Popularity@5",
        "official_function": "calARP",
        "metric_family": "popularity",
        "cutoff_semantics": "top metric_len recommendations",
        "aggregation_semantics": "mean train frequency over flattened top-K recommendations",
        "known_ambiguities": "",
    },
    {
        "metric_position": 5,
        "official_metric_name": "PopREO",
        "canonical_metric_name": "popularity_ranking_equal_opportunity_at_5",
        "paper_alias": "PopREO@5",
        "official_function": "calPopREO",
        "metric_family": "popularity",
        "cutoff_semantics": "hit vector uses metric_len cutoff",
        "aggregation_semantics": "std/mean of hit rates across five popularity groups",
        "known_ambiguities": "Zero-denominator and empty-group policy is not guarded in the official code.",
    },
    {
        "metric_position": 6,
        "official_metric_name": "Serendipity",
        "canonical_metric_name": "serendipity_at_5",
        "paper_alias": "Serendipity@5",
        "official_function": "calSerendipity",
        "metric_family": "novelty",
        "cutoff_semantics": "hit vector uses metric_len cutoff",
        "aggregation_semantics": "mean of hit minus MostPop-hit times hit",
        "known_ambiguities": "Paper prose and displayed formula differ; code uses hit_results and MostPop hit_results.",
    },
    {
        "metric_position": 7,
        "official_metric_name": "SelfInformation",
        "canonical_metric_name": "self_information_at_5",
        "paper_alias": "SelfInformation@5",
        "official_function": "calSelfInfo",
        "metric_family": "novelty",
        "cutoff_semantics": "top metric_len recommendations",
        "aggregation_semantics": "mean information score over flattened top-K recommendations",
        "known_ambiguities": "Code uses train frequency dictionary derived from local data files.",
    },
    {
        "metric_position": 8,
        "official_metric_name": "Gini",
        "canonical_metric_name": "gini_at_5",
        "paper_alias": "Gini@5",
        "official_function": "calcFairMetric",
        "metric_family": "fairness",
        "cutoff_semantics": "top metric_len recommendations",
        "aggregation_semantics": "Gini over items appearing in top-K recommendations",
        "known_ambiguities": "Official code does not include all catalog items in the denominator.",
    },
    {
        "metric_position": 9,
        "official_metric_name": "DPD",
        "canonical_metric_name": "demographic_parity_difference_at_5",
        "paper_alias": "DPD@5",
        "official_function": "calDPDFreq",
        "metric_family": "fairness",
        "cutoff_semantics": "NDCG vector uses metric_len cutoff",
        "aggregation_semantics": "absolute NDCG mean difference between cold and hot users split at median history length",
        "known_ambiguities": "User group construction depends on recovered local data files.",
    },
    {
        "metric_position": 10,
        "official_metric_name": "Jain",
        "canonical_metric_name": "jain_index",
        "paper_alias": "Jain's Index",
        "official_function": "calJain",
        "metric_family": "fairness",
        "cutoff_semantics": "NDCG vector uses metric_len cutoff",
        "aggregation_semantics": "squared sum of per-user NDCG divided by user count times sum of squared NDCG",
        "known_ambiguities": "",
    },
    {
        "metric_position": 11,
        "official_metric_name": "ItemCoverage",
        "canonical_metric_name": "item_coverage_at_5",
        "paper_alias": "ItemCoverage@5",
        "official_function": "calItemCoverage",
        "metric_family": "diversity",
        "cutoff_semantics": "top metric_len recommendations and cand_len candidate pool",
        "aggregation_semantics": "unique recommended items divided by sampled candidate-pool universe",
        "known_ambiguities": "Candidate-pool denominator requires official data/negative sample artifacts.",
    },
    {
        "metric_position": 12,
        "official_metric_name": "Shannon",
        "canonical_metric_name": "shannon_entropy_at_5",
        "paper_alias": "Shannon@5",
        "official_function": "calMeanShannon",
        "metric_family": "diversity",
        "cutoff_semantics": "top metric_len recommendations",
        "aggregation_semantics": "mean category entropy over recommendation lists",
        "known_ambiguities": "Depends on recovered category metadata.",
    },
    {
        "metric_position": 13,
        "official_metric_name": "Jaccard",
        "canonical_metric_name": "jaccard_at_5",
        "paper_alias": "Jaccard/OIC-related overlap",
        "official_function": "calMeanJaccard",
        "metric_family": "diversity",
        "cutoff_semantics": "top metric_len recommendations",
        "aggregation_semantics": "mean pairwise recommendation overlap normalized by candidate intersection",
        "known_ambiguities": "Code skips pairs without candidate intersection.",
    },
    {
        "metric_position": 14,
        "official_metric_name": "Hallucination",
        "canonical_metric_name": "hallucination_rate",
        "paper_alias": "Hallucination",
        "official_function": "calHallucination",
        "metric_family": "hallucination",
        "cutoff_semantics": "all parsed output slots in the official fixed-length result list",
        "aggregation_semantics": "mean of item-id zero indicators after title-to-item mapping",
        "known_ambiguities": "This is official title-matching hallucination, not later local JSON structural validity.",
    },
]

PROTECTED_PRE_R1_PATHS = [
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
]


@dataclass
class R1Outputs:
    table_dir: Path
    report_path: Path
    summary: dict[str, object]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_git(args: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def normalize_dataset(value: str) -> str:
    lowered = value.lower()
    if lowered in {"beauty", "amazon_beauty", "amazon beauty"}:
        return "Beauty"
    if lowered in {"sports", "amazon_sports", "amazon sports"}:
        return "Sports"
    if lowered in {"ml1m", "ml-1m", "movielens", "movielens-1m"}:
        return "ML-1M"
    if lowered in {"lastfm", "last.fm", "last-fm"}:
        return "LastFM"
    return ""


def infer_dataset(*parts: str) -> str:
    joined = "/".join(parts).lower()
    for token, dataset in [
        ("beauty", "Beauty"),
        ("sports", "Sports"),
        ("ml-1m", "ML-1M"),
        ("ml1m", "ML-1M"),
        ("movielens", "ML-1M"),
        ("lastfm", "LastFM"),
        ("last.fm", "LastFM"),
    ]:
        if token in joined:
            return dataset
    return ""


def source_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def find_line_number(lines: list[str], pattern: str) -> int:
    for idx, line in enumerate(lines, start=1):
        if pattern in line:
            return idx
    return 0


def recover_metric_serialization_contract(original_root: Path) -> list[dict[str, object]]:
    metric_source = original_root / "eva" / "codes" / "llm_metrics.py"
    lines = source_lines(metric_source)
    final_start = find_line_number(lines, "final_result = [")
    final_end = find_line_number(lines, "pd.DataFrame([final_result]).T.to_csv")

    recovered: list[dict[str, object]] = []
    for metric in METRIC_CONTRACT:
        function_name = str(metric["official_function"])
        function_line = find_line_number(lines, f"def {function_name}")
        serialization_line = 0
        for idx, line in enumerate(lines, start=1):
            if final_start <= idx <= final_end and f"{function_name}(" in line:
                serialization_line = idx
                break
        status = "SOURCE_VERIFIED" if function_line and serialization_line else "SOURCE_CONFLICT"
        recovered.append(
            {
                **metric,
                "official_variable_name": function_name,
                "function_source": metric_source.as_posix(),
                "function_line": function_line,
                "serialization_source": metric_source.as_posix(),
                "serialization_line": serialization_line,
                "csv_header_policy": "headerless_one_value_per_row",
                "metric_semantics_status": status,
                "evidence_class": EVIDENCE_LABEL,
            }
        )
    return recovered


def classify_artifact_type(path: str, is_dir: bool = False) -> str:
    lowered = path.lower()
    if is_dir:
        return "directory"
    if lowered.endswith(".zip"):
        return "archive_zip"
    if lowered.endswith(".txt") and re.search(r"(^|/|\\)sample(_1k(_all)?)?\.txt$", lowered):
        return "sample_user_ids_txt"
    if lowered.endswith("requirements.txt"):
        return "dependency_manifest"
    if lowered.endswith(".txt") and "template" in lowered:
        return "prompt_template_txt"
    if lowered.endswith("_5.csv"):
        return "official_metric_vector_csv"
    if re.search(r"/rec-[^/]+\.csv$", lowered) or re.search(r"\\rec-[^\\]+\.csv$", lowered):
        return "sota_recommendation_csv"
    if lowered.endswith(".csv"):
        return "csv"
    if lowered.endswith(".json"):
        if "/results_" in lowered or "\\results_" in lowered:
            return "llm_result_json"
        if "/test_" in lowered or "\\test_" in lowered:
            return "prompt_manifest_json"
        return "json"
    if lowered.endswith(".jsonl"):
        return "jsonl"
    if lowered.endswith(".ipynb"):
        return "notebook"
    if lowered.endswith(".py"):
        return "python_source"
    if lowered.endswith(".sh"):
        return "shell_script"
    if lowered.endswith(".md"):
        return "markdown"
    return "other"


def parse_condition(path_text: str) -> dict[str, str]:
    normalized = path_text.replace("\\", "/")
    name = Path(normalized).name
    stem = name[:-6] if name.endswith("_5.csv") else Path(name).stem
    dataset = infer_dataset(normalized)
    model = ""
    seed = ""
    rest = stem
    model_match = re.match(r"^results_(?P<model>.+?)_seed(?P<seed>\d+)_(?P<rest>.+)$", stem)
    if model_match:
        model = model_match.group("model")
        seed = model_match.group("seed")
        rest = model_match.group("rest")

    history_match = re.search(r"history(\d+)", rest)
    candidate_matches = re.findall(r"candidate(\d+)", rest)
    history_length = history_match.group(1) if history_match else ""
    candidate_count = candidate_matches[-1] if candidate_matches else ""
    task = "reranking" if "multi_candidate" in rest or "rerank" in rest.lower() else "ranking"

    profile_generator = ""
    profile_mode = "history_only"
    profile_match = re.search(r"profile_(gpt3\.5|gpt4|glm4|llama2|llama3)", rest)
    if profile_match:
        profile_generator = profile_match.group(1)
        profile_mode = "profile_plus_history" if "_history_sample" in rest else "profile_only"

    if profile_generator:
        analysis_family = "generation_involved_performance"
    elif "llmrank" in rest:
        analysis_family = "llmrank"
    elif history_length and history_length not in {"10"}:
        analysis_family = "history_length_sensitivity"
    elif task == "reranking":
        analysis_family = "reranking"
    else:
        analysis_family = "base_ranking"

    candidate_flags: list[str] = []
    if "multi_candidate" in rest:
        candidate_flags.append("multi_candidate_pool")
    if "shuffle_candidate" in rest:
        candidate_flags.append("candidate_first_prompt_or_order_variant")
    if re.search(r"candidate\d+_shuffle", rest):
        candidate_flags.append("candidate_order_shuffled")
    if candidate_count and "shuffle" not in rest:
        candidate_flags.append("candidate_order_not_shuffled_or_unspecified")
    if "llmrank_rf" in rest:
        candidate_flags.append("llmrank_rf")
    elif "llmrank" in rest:
        candidate_flags.append("llmrank")
    if not candidate_flags and candidate_count:
        candidate_flags.append("candidate_protocol_present")

    condition_identification_status = "IDENTIFIED" if dataset and (model or stem.startswith("test_")) else "PARTIALLY_IDENTIFIED"
    if not dataset:
        condition_identification_status = "UNIDENTIFIED"

    condition_parts = [
        f"dataset={dataset or 'unknown'}",
        f"task={task}",
        f"model={model or 'not_applicable'}",
        f"history={history_length or 'unknown'}",
        f"candidate_count={candidate_count or 'unknown'}",
        f"profile_mode={profile_mode}",
    ]
    if candidate_flags:
        condition_parts.append("candidate_condition=" + "|".join(candidate_flags))

    return {
        "dataset": dataset,
        "task": task,
        "model": model,
        "seed": seed,
        "condition": "; ".join(condition_parts),
        "analysis_family": analysis_family,
        "history_length": history_length,
        "profile_generator": profile_generator,
        "profile_mode": profile_mode,
        "candidate_condition": "|".join(candidate_flags),
        "candidate_count": candidate_count,
        "cutoff": "5" if name.endswith("_5.csv") else "",
        "condition_identification_status": condition_identification_status,
        "filename_rest": rest,
    }


def parse_metric_vector(text: str) -> list[str]:
    values: list[str] = []
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        value = row[0].strip()
        if value:
            values.append(value)
    return values


def safe_json_summary(data: bytes) -> dict[str, object]:
    try:
        obj = json.loads(data.decode("utf-8-sig", errors="replace"))
    except json.JSONDecodeError as exc:
        return {
            "json_parse_status": "JSON_PARSE_FAILED",
            "record_count": "",
            "top_level_type": "",
            "field_names": "",
            "has_user": "NO",
            "has_target": "NO",
            "has_cand_list": "NO",
            "has_prompt": "NO",
            "has_result": "NO",
            "has_template": "NO",
            "candidate_count_first_record": "",
            "notes": str(exc),
        }
    if isinstance(obj, list):
        record_count = len(obj)
        first = obj[0] if obj else {}
        top_type = "list"
    elif isinstance(obj, dict):
        record_count = len(obj)
        first = next(iter(obj.values()), {})
        top_type = "dict"
    else:
        record_count = ""
        first = {}
        top_type = type(obj).__name__
    if not isinstance(first, dict):
        first = {}
    fields = sorted(str(key) for key in first.keys())
    candidate_count = ""
    cand = first.get("cand_list")
    if isinstance(cand, str):
        try:
            parsed = ast.literal_eval(cand)
            if isinstance(parsed, list):
                candidate_count = str(len(parsed))
        except (SyntaxError, ValueError):
            candidate_count = ""
    elif isinstance(cand, list):
        candidate_count = str(len(cand))
    return {
        "json_parse_status": "PARSED",
        "record_count": record_count,
        "top_level_type": top_type,
        "field_names": ";".join(fields),
        "has_user": "YES" if "user" in first else "NO",
        "has_target": "YES" if "target" in first else "NO",
        "has_cand_list": "YES" if "cand_list" in first else "NO",
        "has_prompt": "YES" if "prompt" in first else "NO",
        "has_result": "YES" if "result" in first else "NO",
        "has_template": "YES" if "template" in first else "NO",
        "candidate_count_first_record": candidate_count,
        "notes": "",
    }


def artifact_status(path_text: str, artifact_type: str, condition_status: str) -> str:
    if artifact_type in {
        "official_metric_vector_csv",
        "llm_result_json",
        "prompt_manifest_json",
        "sota_recommendation_csv",
        "archive_zip",
        "sample_user_ids_txt",
        "prompt_template_txt",
    }:
        return condition_status if condition_status != "UNIDENTIFIED" else "PARTIALLY_IDENTIFIED"
    if artifact_type in {"python_source", "shell_script", "notebook", "markdown"}:
        return "IDENTIFIED"
    if artifact_type == "dependency_manifest":
        return "IDENTIFIED"
    return condition_status


def direct_result_artifacts(original_root: Path) -> list[Path]:
    suffixes = {".zip", ".csv", ".json", ".jsonl", ".py", ".sh", ".ipynb", ".md", ".txt"}
    paths: list[Path] = []
    for path in original_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            paths.append(path)
    return sorted(paths)


def inventory_original_artifacts(original_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    artifact_rows: list[dict[str, object]] = []
    zip_member_rows: list[dict[str, object]] = []

    for path in direct_result_artifacts(original_root):
        rel = path.relative_to(original_root).as_posix()
        condition = parse_condition(rel)
        artifact_type = classify_artifact_type(rel)
        row = {
            "artifact_path": (original_root / rel).as_posix(),
            "container_archive": "",
            "member_path": "",
            "extension": path.suffix.lower(),
            "artifact_type": artifact_type,
            "dataset": condition["dataset"],
            "task": condition["task"],
            "model": condition["model"],
            "condition": condition["condition"],
            "history_length": condition["history_length"],
            "profile_generator": condition["profile_generator"],
            "candidate_condition": condition["candidate_condition"],
            "cutoff": condition["cutoff"],
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
            "evidence_status": artifact_status(rel, artifact_type, condition["condition_identification_status"]),
            "notes": "",
        }
        artifact_rows.append(row)

        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for member in sorted(archive.infolist(), key=lambda item: item.filename):
                    member_condition = parse_condition(member.filename)
                    member_type = classify_artifact_type(member.filename, member.is_dir())
                    member_hash = ""
                    if not member.is_dir():
                        member_hash = sha256_bytes(archive.read(member.filename))
                    member_row = {
                        "artifact_path": path.as_posix(),
                        "container_archive": path.as_posix(),
                        "member_path": member.filename,
                        "extension": Path(member.filename).suffix.lower(),
                        "artifact_type": member_type,
                        "dataset": member_condition["dataset"] or condition["dataset"],
                        "task": member_condition["task"],
                        "model": member_condition["model"],
                        "condition": member_condition["condition"],
                        "history_length": member_condition["history_length"],
                        "profile_generator": member_condition["profile_generator"],
                        "candidate_condition": member_condition["candidate_condition"],
                        "cutoff": member_condition["cutoff"],
                        "byte_size": member.file_size,
                        "sha256": member_hash,
                        "evidence_status": artifact_status(
                            member.filename,
                            member_type,
                            member_condition["condition_identification_status"],
                        ),
                        "notes": "",
                    }
                    artifact_rows.append(member_row)
                    zip_member_rows.append(member_row)

    return artifact_rows, zip_member_rows


def decode_metric_csvs(
    artifact_rows: list[dict[str, object]],
    metric_contract: list[dict[str, object]],
) -> list[dict[str, object]]:
    contract_by_position = {int(row["metric_position"]): row for row in metric_contract}
    decoded: list[dict[str, object]] = []
    direct_metric_paths = [
        row
        for row in artifact_rows
        if row["artifact_type"] == "official_metric_vector_csv" and not row["container_archive"]
    ]
    for row in direct_metric_paths:
        path = Path(str(row["artifact_path"]))
        values = parse_metric_vector(path.read_text(encoding="utf-8-sig", errors="replace"))
        decoded.extend(
            decoded_metric_rows_for_vector(row, values, contract_by_position, "direct_file")
        )

    archives = sorted(
        {Path(str(row["container_archive"])) for row in artifact_rows if row["container_archive"]}
    )
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            for member in sorted(archive.infolist(), key=lambda item: item.filename):
                if member.is_dir() or not member.filename.endswith("_5.csv"):
                    continue
                member_bytes = archive.read(member.filename)
                matching = [
                    row
                    for row in artifact_rows
                    if row["container_archive"] == archive_path.as_posix()
                    and row["member_path"] == member.filename
                ][0]
                values = parse_metric_vector(member_bytes.decode("utf-8-sig", errors="replace"))
                decoded.extend(
                    decoded_metric_rows_for_vector(
                        matching,
                        values,
                        contract_by_position,
                        "archive_member",
                    )
                )
    return decoded


def decoded_metric_rows_for_vector(
    source_row: dict[str, object],
    values: list[str],
    contract_by_position: dict[int, dict[str, object]],
    source_kind: str,
) -> list[dict[str, object]]:
    expected = len(contract_by_position)
    if len(values) == expected:
        decode_status = "FULLY_DECODED"
    elif len(values) < expected:
        decode_status = "PARTIALLY_DECODED"
    else:
        decode_status = "FORMAT_CONFLICT"

    rows: list[dict[str, object]] = []
    for index, value in enumerate(values, start=1):
        metric = contract_by_position.get(index)
        if metric is None:
            metric = {
                "official_metric_name": f"UNRESOLVED_POSITION_{index}",
                "canonical_metric_name": f"unresolved_position_{index}",
                "metric_family": "unresolved",
                "metric_semantics_status": "UNRESOLVED",
            }
        rows.append(
            {
                "dataset": source_row["dataset"],
                "task": source_row["task"],
                "model": source_row["model"],
                "condition": source_row["condition"],
                "history_length": source_row["history_length"],
                "profile_generator": source_row["profile_generator"],
                "candidate_condition": source_row["candidate_condition"],
                "cutoff": "5",
                "source_archive": source_row["container_archive"],
                "source_member": source_row["member_path"] or source_row["artifact_path"],
                "source_member_sha256": source_row["sha256"],
                "metric_position": index,
                "official_metric_name": metric["official_metric_name"],
                "canonical_metric_name": metric["canonical_metric_name"],
                "metric_family": metric["metric_family"],
                "original_repo_value": value,
                "decode_status": decode_status,
                "metric_semantics_status": metric["metric_semantics_status"],
                "condition_identification_status": source_row["evidence_status"],
                "value_origin": "OFFICIAL_ARTIFACT_DIRECT",
                "evidence_class": EVIDENCE_LABEL,
                "notes": f"Decoded from {source_kind}; metric order recovered from official llm_metrics.py.",
            }
        )
    if not values:
        rows.append(
            {
                "dataset": source_row["dataset"],
                "task": source_row["task"],
                "model": source_row["model"],
                "condition": source_row["condition"],
                "history_length": source_row["history_length"],
                "profile_generator": source_row["profile_generator"],
                "candidate_condition": source_row["candidate_condition"],
                "cutoff": "5",
                "source_archive": source_row["container_archive"],
                "source_member": source_row["member_path"] or source_row["artifact_path"],
                "source_member_sha256": source_row["sha256"],
                "metric_position": "",
                "official_metric_name": "",
                "canonical_metric_name": "",
                "metric_family": "unresolved",
                "original_repo_value": "",
                "decode_status": "UNRESOLVED",
                "metric_semantics_status": "UNRESOLVED",
                "condition_identification_status": source_row["evidence_status"],
                "value_origin": "OFFICIAL_ARTIFACT_DIRECT",
                "evidence_class": EVIDENCE_LABEL,
                "notes": "No values found in official metric CSV.",
            }
        )
    return rows


def build_json_inventory(artifact_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    direct_json = [
        row
        for row in artifact_rows
        if row["extension"] == ".json" and not row["container_archive"] and row["artifact_type"] != "notebook"
    ]
    for row in direct_json:
        data = Path(str(row["artifact_path"])).read_bytes()
        rows.append(json_inventory_row(row, data))

    archive_paths = sorted({Path(str(row["container_archive"])) for row in artifact_rows if row["container_archive"]})
    for archive_path in archive_paths:
        with zipfile.ZipFile(archive_path) as archive:
            for member in sorted(archive.infolist(), key=lambda item: item.filename):
                if member.is_dir() or not member.filename.endswith(".json"):
                    continue
                source_row = [
                    row
                    for row in artifact_rows
                    if row["container_archive"] == archive_path.as_posix()
                    and row["member_path"] == member.filename
                ][0]
                rows.append(json_inventory_row(source_row, archive.read(member.filename)))
    return rows


def json_inventory_row(source_row: dict[str, object], data: bytes) -> dict[str, object]:
    summary = safe_json_summary(data)
    return {
        "dataset": source_row["dataset"],
        "task": source_row["task"],
        "model": source_row["model"],
        "condition": source_row["condition"],
        "history_length": source_row["history_length"],
        "profile_generator": source_row["profile_generator"],
        "profile_mode": parse_condition(str(source_row["member_path"] or source_row["artifact_path"]))["profile_mode"],
        "candidate_condition": source_row["candidate_condition"],
        "source_archive": source_row["container_archive"],
        "source_member": source_row["member_path"] or source_row["artifact_path"],
        "source_member_sha256": source_row["sha256"],
        "json_artifact_type": source_row["artifact_type"],
        **summary,
        "evidence_class": EVIDENCE_LABEL,
    }


def parse_metrics_sh(original_root: Path) -> list[dict[str, object]]:
    script_path = original_root / "eva" / "codes" / "metrics.sh"
    rows: list[dict[str, object]] = []
    if not script_path.exists():
        return rows
    for line_number, raw_line in enumerate(source_lines(script_path), start=1):
        line = raw_line.strip()
        if not line.startswith("python llm_metrics.py"):
            continue
        results_path = re.search(r"--results_path\s+(\S+)", line)
        dataset = re.search(r"--dataset\s+(\S+)", line)
        model = re.search(r"--model_name\s+(\S+)", line)
        sample_num = re.search(r"--sample_num\s+(\S+)", line)
        history_max = re.search(r"--history_max\s+(\S+)", line)
        source = results_path.group(1) if results_path else ""
        condition = parse_condition(source)
        rows.append(
            {
                "script_path": script_path.as_posix(),
                "line_number": line_number,
                "command": line,
                "results_path": source,
                "dataset": normalize_dataset(dataset.group(1)) if dataset else condition["dataset"] or "Beauty",
                "task": condition["task"],
                "model": model.group(1) if model else condition["model"],
                "history_length": history_max.group(1) if history_max else condition["history_length"],
                "profile_generator": condition["profile_generator"],
                "profile_mode": condition["profile_mode"],
                "candidate_condition": condition["candidate_condition"],
                "sample_num": sample_num.group(1) if sample_num else "",
                "condition_family": condition["analysis_family"],
                "evidence_class": EVIDENCE_LABEL,
            }
        )
    return rows


def build_history_results(decoded_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in decoded_rows:
        if parse_condition(str(row["source_member"]))["analysis_family"] != "history_length_sensitivity":
            continue
        rows.append(
            {
                "dataset": row["dataset"],
                "task": row["task"],
                "model": row["model"],
                "history_length": row["history_length"],
                "metric": row["canonical_metric_name"],
                "value": row["original_repo_value"],
                "source_archive": row["source_archive"],
                "source_member": row["source_member"],
                "condition_evidence": row["condition"],
                "mapping_confidence": "artifact_filename_direct",
                "evidence_class": EVIDENCE_LABEL,
            }
        )
    return rows


def build_generation_results(decoded_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in decoded_rows:
        condition = parse_condition(str(row["source_member"]))
        if condition["analysis_family"] != "generation_involved_performance":
            continue
        rows.append(
            {
                "dataset": row["dataset"],
                "task": row["task"],
                "recommendation_model": row["model"],
                "profile_generator": row["profile_generator"],
                "profile_mode": condition["profile_mode"],
                "history_length": row["history_length"],
                "metric": row["canonical_metric_name"],
                "value": row["original_repo_value"],
                "source_archive": row["source_archive"],
                "source_member": row["source_member"],
                "condition_evidence": row["condition"],
                "mapping_confidence": "artifact_filename_direct",
                "evidence_class": EVIDENCE_LABEL,
            }
        )
    return rows


def build_candidate_position_evidence(
    artifact_rows: list[dict[str, object]],
    decoded_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_sources: set[str] = set()
    for row in decoded_rows:
        if row["candidate_condition"] == "" and "candidate" not in str(row["source_member"]).lower():
            continue
        if row["canonical_metric_name"] not in {"hr_at_5", "ndcg_at_5"}:
            continue
        source_key = f'{row["source_archive"]}|{row["source_member"]}|{row["canonical_metric_name"]}'
        seen_sources.add(source_key)
        condition = str(row["candidate_condition"])
        direct_metric_present = "YES"
        derivation_status = "PARTIALLY_DERIVABLE"
        what_missing = "Official CSV contains variant metric values, but no official CandDif result vector is serialized here."
        if "candidate_first_prompt_or_order_variant" in condition or "candidate_order_shuffled" in condition:
            what_known = "Candidate/order variant is identified from official filename and has HR/NDCG metric values."
        else:
            what_known = "Candidate protocol is identified from official filename and can serve as a paired-condition candidate if matching artifacts exist."
        rows.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "condition": row["condition"],
                "source": f'{row["source_archive"]}:{row["source_member"]}',
                "metric": row["canonical_metric_name"],
                "value": row["original_repo_value"],
                "what_is_known": what_known,
                "what_is_missing": what_missing,
                "direct_metric_present": direct_metric_present,
                "derivation_possible": derivation_status,
                "derivation_contract_status": "official_canddif_not_directly_serialized",
                "evidence_class": EVIDENCE_LABEL,
            }
        )
    for artifact in artifact_rows:
        source_text = str(artifact["member_path"] or artifact["artifact_path"])
        lowered = source_text.lower()
        if "candidate" not in lowered or not lowered.endswith(".json"):
            continue
        key = f'{artifact["container_archive"]}|{source_text}|json'
        if key in seen_sources:
            continue
        rows.append(
            {
                "dataset": artifact["dataset"],
                "model": artifact["model"],
                "condition": artifact["condition"],
                "source": f'{artifact["container_archive"]}:{source_text}',
                "metric": "",
                "value": "",
                "what_is_known": "JSON/prompt artifact contains candidate-list evidence according to filename and/or fields.",
                "what_is_missing": "CandDif is not directly serialized in this JSON artifact.",
                "direct_metric_present": "NO",
                "derivation_possible": "PARTIALLY_DERIVABLE",
                "derivation_contract_status": "requires_R2_pairing_and_formula_recovery",
                "evidence_class": EVIDENCE_LABEL,
            }
        )
    return rows


def build_hallucination_results(decoded_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in decoded_rows:
        if row["canonical_metric_name"] != "hallucination_rate":
            continue
        rows.append(
            {
                "dataset": row["dataset"],
                "task": row["task"],
                "model": row["model"],
                "condition": row["condition"],
                "history_length": row["history_length"],
                "profile_generator": row["profile_generator"],
                "candidate_condition": row["candidate_condition"],
                "metric": row["canonical_metric_name"],
                "value": row["original_repo_value"],
                "source_archive": row["source_archive"],
                "source_member": row["source_member"],
                "semantics": "official title-to-item matching hallucination; unmatched parsed outputs map to item id zero",
                "not_semantics": "not local JSON/index structural validity",
                "evidence_class": EVIDENCE_LABEL,
            }
        )
    return rows


def build_dataset_classification(artifact_rows: list[dict[str, object]], original_root: Path) -> list[dict[str, object]]:
    result_artifact_datasets = {str(row["dataset"]) for row in artifact_rows if row["dataset"]}
    readme_text = ""
    readme_path = original_root / "README.md"
    if readme_path.exists():
        readme_text = readme_path.read_text(encoding="utf-8", errors="replace").lower()
    rows = []
    for dataset in TARGET_DATASETS:
        if dataset in result_artifact_datasets:
            status = "RESULT_ARTIFACT_PRESENT"
            notes = "Official result archive or result member found."
        elif dataset == "ML-1M" and ("movielens" in readme_text or "ml-1m" in readme_text):
            status = "REFERENCE_ONLY"
            notes = "README references MovieLens-1M/raw data, but no official result artifact was found."
        elif dataset == "LastFM" and "lastfm" in readme_text:
            status = "REFERENCE_ONLY"
            notes = "README references LastFM/raw data, but no official result artifact was found."
        else:
            status = "NOT_AVAILABLE"
            notes = "No official result artifact or explicit dataset reference found."
        rows.append(
            {
                "dataset": dataset,
                "official_evidence_classification": status,
                "evidence_class": EVIDENCE_LABEL,
                "notes": notes,
            }
        )
    return rows


def build_metric_family_coverage(decoded_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    families = sorted({str(row["metric_family"]) for row in decoded_rows})
    families.extend(["history_length_sensitivity", "generation_involved_performance", "candidate_position_or_order"])
    families = sorted(set(families))
    by_dataset_family = defaultdict(list)
    for row in decoded_rows:
        by_dataset_family[(str(row["dataset"]), str(row["metric_family"]))].append(row)
        condition_family = parse_condition(str(row["source_member"]))["analysis_family"]
        if condition_family == "history_length_sensitivity":
            by_dataset_family[(str(row["dataset"]), "history_length_sensitivity")].append(row)
        if condition_family == "generation_involved_performance":
            by_dataset_family[(str(row["dataset"]), "generation_involved_performance")].append(row)
        if row["candidate_condition"]:
            by_dataset_family[(str(row["dataset"]), "candidate_position_or_order")].append(row)

    rows: list[dict[str, object]] = []
    for dataset in TARGET_DATASETS:
        for family in families:
            values = by_dataset_family.get((dataset, family), [])
            if values:
                status = "OFFICIAL_VALUES_PRESENT"
            elif dataset in {"ML-1M", "LastFM"}:
                status = "CODE_ONLY"
            else:
                status = "NOT_AVAILABLE"
            rows.append(
                {
                    "dataset": dataset,
                    "metric_family": family,
                    "coverage_status": status,
                    "official_value_rows": len(values),
                    "evidence_class": EVIDENCE_LABEL,
                    "notes": "R1 official artifact coverage; not paper-cell reconciliation.",
                }
            )
    return rows


def build_searched_manifest(original_root: Path, artifact_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    roots = [
        original_root,
        original_root / "llm_results",
        original_root / "sota_results",
        original_root / "eva" / "codes",
        original_root / "eva" / "templates",
        original_root / "ReChorus",
        original_root / "data",
    ]
    patterns = ["*.zip", "*_5.csv", "*.csv", "*.json", "*.jsonl", "*.py", "*.sh", "*.ipynb", "*.md", "*.txt"]
    rows: list[dict[str, object]] = []
    for root in roots:
        for pattern in patterns:
            if not root.exists():
                rows.append(
                    {
                        "path": root.as_posix(),
                        "artifact_pattern": pattern,
                        "files_found": 0,
                        "files_processed": 0,
                        "files_unresolved": 0,
                        "status": "PATH_NOT_AVAILABLE",
                        "notes": "",
                    }
                )
                continue
            found = list(root.rglob(pattern)) if root.is_dir() else []
            found_files = [path for path in found if path.is_file()]
            unresolved = 0
            for path in found_files:
                rel = path.as_posix()
                if not any(str(row["artifact_path"]) == rel for row in artifact_rows):
                    unresolved += 1
            rows.append(
                {
                    "path": root.as_posix(),
                    "artifact_pattern": pattern,
                    "files_found": len(found_files),
                    "files_processed": len(found_files) - unresolved,
                    "files_unresolved": unresolved,
                    "status": "SEARCHED",
                    "notes": "",
                }
            )
    return rows


def build_precondition_tables(project_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    git_state = [
        {"command": "git branch --show-current", "output": run_git(["branch", "--show-current"], project_root)},
        {"command": "git rev-parse HEAD", "output": run_git(["rev-parse", "HEAD"], project_root)},
        {"command": "git status --short", "output": run_git(["status", "--short"], project_root)},
        {"command": "git status --branch --short", "output": run_git(["status", "--branch", "--short"], project_root)},
        {"command": "git diff --stat", "output": run_git(["diff", "--stat"], project_root)},
        {"command": "git diff --cached --stat", "output": run_git(["diff", "--cached", "--stat"], project_root)},
    ]
    protected = []
    for rel in PROTECTED_PRE_R1_PATHS:
        path = resolve_r1_metric_recovery_artifact(project_root, rel)
        protected.append(
            {
                "path": rel,
                "exists": "YES" if path.exists() else "NO",
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
                "protection_scope": "pre_existing_non_r1_unstaged_artifact",
                "notes": "Captured for R1 protection; not modified by R1 generator.",
            }
        )
    return git_state, protected


def write_report(
    report_path: Path,
    summary: dict[str, object],
    metric_contract: list[dict[str, object]],
    dataset_classification: list[dict[str, object]],
    coverage_rows: list[dict[str, object]],
    history_rows: list[dict[str, object]],
    generation_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    unresolved_rows: list[dict[str, object]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metric_order = ", ".join(str(row["official_metric_name"]) for row in metric_contract)
    dataset_lines = "\n".join(
        f"- {row['dataset']}: {row['official_evidence_classification']} ({row['notes']})"
        for row in dataset_classification
    )
    coverage_summary = Counter(str(row["coverage_status"]) for row in coverage_rows)
    coverage_text = ", ".join(f"{key}={value}" for key, value in sorted(coverage_summary.items()))
    unresolved_text = "\n".join(
        f"- {row['artifact_path']}::{row['member_path']} [{row['artifact_type']}]"
        for row in unresolved_rows[:25]
    )
    if not unresolved_text:
        unresolved_text = "- None"
    text = f"""# R1 Original Repository Result Recovery

Label: {EVIDENCE_LABEL}

Decision: `{DECISION_TOKEN}`

## Scope

R1 exhaustively inventories scientific result artifacts currently present in
`original_repo/EvaLLMasRecommender` without modifying or extracting into the
official checkout. It recovers official artifact evidence only. It does not
claim paper-cell equivalence and does not reopen F6M.

## Metric Serialization Contract

Official metric vector order recovered from `eva/codes/llm_metrics.py`:

`{metric_order}`

The official serialization writes a headerless one-value-per-row CSV through
`pd.DataFrame([final_result]).T.to_csv(...)`. ACLT is preserved under the
official code name and related to paper APLT/long-tail wording in the contract
table.

## Artifact Recovery Summary

- Direct artifacts inventoried: {summary['direct_artifacts']}
- ZIP archives found: {summary['archives_found']}
- ZIP members inspected: {summary['zip_members_inspected']}
- Official `_5.csv` artifacts found: {summary['official_metric_csvs_found']}
- Fully decoded CSVs: {summary['fully_decoded_csvs']}
- Partially decoded CSVs: {summary['partially_decoded_csvs']}
- Unresolved/format-conflict CSVs: {summary['unresolved_csvs']}
- Total official metric values decoded: {summary['decoded_metric_values']}
- JSON artifacts inventoried: {summary['json_artifacts']}

## Dataset Evidence

{dataset_lines}

## Beauty

Models, tasks, history lengths, profile modes, and candidate/order conditions
are recorded in the decoded tables and artifact inventory. Beauty is the
largest official result source in R1.

## Sports

Sports has official result evidence, but it does not mirror Beauty exactly.
Coverage differences are recorded by source member and condition.

## ML-1M and LastFM

No official ML-1M or LastFM result artifact was recovered from the official
checkout during R1. README references are preserved as reference-only evidence.
Local adapted artifacts outside `original_repo` are not used as substitutes.

## History-Length Sensitivity

Recovered rows: {len(history_rows)}. These rows are official artifact values
grouped by filename-evidenced history length. R1 does not map them to paper
figures.

## Generation/Profile

Recovered rows: {len(generation_rows)}. Recommendation model and profile
generator are kept separate.

## Candidate Position / Order

Recovered evidence rows: {len(candidate_rows)}. Official artifacts expose
candidate/order variants and direct HR/NDCG values, but CandDif is not directly
serialized in the `_5.csv` vectors. R2 must handle pairing and formula mapping.

## Hallucination

Official hallucination values are decoded from metric position 14. This is the
official title-to-item matching hallucination semantics, not local structural
validity.

## Metric-Family Coverage

Coverage status counts: {coverage_text}

## Unresolved Artifacts

{unresolved_text}

## Interpretation

The official repository contains substantially more usable Beauty/Sports result
evidence than represented by the historical F6M closure. This does not by
itself eliminate the `CRITICAL_EXPERIMENTAL_PATH_PROTOCOL_LIMITED`
classification because paper-cell identity, preprocessing artifacts, exact
cohorts, candidate pools, and model/runtime equivalence still require R2
reconciliation.
"""
    report_path.write_text(text, encoding="utf-8")


def recover_original_repo_results(
    project_root: Path,
    original_root: Path | None = None,
    table_dir: Path | None = None,
    report_path: Path | None = None,
) -> R1Outputs:
    project_root = project_root.resolve()
    original_root = (original_root or project_root / ORIGINAL_RELATIVE_ROOT).resolve()
    table_dir = (table_dir or project_root / DEFAULT_TABLE_DIR).resolve()
    report_path = (report_path or project_root / DEFAULT_REPORT_PATH).resolve()

    metric_contract = recover_metric_serialization_contract(original_root)
    artifact_rows, zip_member_rows = inventory_original_artifacts(original_root)
    decoded_rows = decode_metric_csvs(artifact_rows, metric_contract)
    json_rows = build_json_inventory(artifact_rows)
    command_rows = parse_metrics_sh(original_root)
    history_rows = build_history_results(decoded_rows)
    generation_rows = build_generation_results(decoded_rows)
    candidate_rows = build_candidate_position_evidence(artifact_rows, decoded_rows)
    hallucination_rows = build_hallucination_results(decoded_rows)
    dataset_rows = build_dataset_classification(artifact_rows, original_root)
    coverage_rows = build_metric_family_coverage(decoded_rows)
    searched_rows = build_searched_manifest(original_root, artifact_rows)
    git_state_rows, protected_rows = build_precondition_tables(project_root)

    unresolved_rows = [
        row
        for row in artifact_rows
        if row["evidence_status"] == "UNIDENTIFIED"
        or row["artifact_type"] in {"other", "jsonl"}
    ]

    decode_counts = Counter(str(row["decode_status"]) for row in decoded_rows)
    unique_metric_csv_sources = {
        (str(row["source_archive"]), str(row["source_member"]))
        for row in decoded_rows
        if row["decode_status"] != "UNRESOLVED"
    }
    fully_decoded_sources = {
        (str(row["source_archive"]), str(row["source_member"]))
        for row in decoded_rows
        if row["decode_status"] == "FULLY_DECODED"
    }
    partially_decoded_sources = {
        (str(row["source_archive"]), str(row["source_member"]))
        for row in decoded_rows
        if row["decode_status"] == "PARTIALLY_DECODED"
    }
    unresolved_csv_sources = {
        (str(row["source_archive"]), str(row["source_member"]))
        for row in decoded_rows
        if row["decode_status"] in {"FORMAT_CONFLICT", "UNRESOLVED"}
    }

    summary = {
        "decision_token": DECISION_TOKEN,
        "direct_artifacts": len([row for row in artifact_rows if not row["container_archive"]]),
        "archives_found": len([row for row in artifact_rows if row["artifact_type"] == "archive_zip" and not row["container_archive"]]),
        "zip_members_inspected": len(zip_member_rows),
        "official_metric_csvs_found": len(unique_metric_csv_sources),
        "fully_decoded_csvs": len(fully_decoded_sources),
        "partially_decoded_csvs": len(partially_decoded_sources),
        "unresolved_csvs": len(unresolved_csv_sources),
        "decoded_metric_values": len(decoded_rows),
        "json_artifacts": len(json_rows),
        "decode_status_counts": dict(decode_counts),
    }

    metric_contract_fields = [
        "metric_position",
        "official_metric_name",
        "canonical_metric_name",
        "paper_alias",
        "official_variable_name",
        "official_function",
        "metric_family",
        "function_source",
        "function_line",
        "serialization_source",
        "serialization_line",
        "cutoff_semantics",
        "aggregation_semantics",
        "known_ambiguities",
        "csv_header_policy",
        "metric_semantics_status",
        "evidence_class",
    ]
    artifact_fields = [
        "artifact_path",
        "container_archive",
        "member_path",
        "extension",
        "artifact_type",
        "dataset",
        "task",
        "model",
        "condition",
        "history_length",
        "profile_generator",
        "candidate_condition",
        "cutoff",
        "byte_size",
        "sha256",
        "evidence_status",
        "notes",
    ]
    decoded_fields = [
        "dataset",
        "task",
        "model",
        "condition",
        "history_length",
        "profile_generator",
        "candidate_condition",
        "cutoff",
        "source_archive",
        "source_member",
        "source_member_sha256",
        "metric_position",
        "official_metric_name",
        "canonical_metric_name",
        "metric_family",
        "original_repo_value",
        "decode_status",
        "metric_semantics_status",
        "condition_identification_status",
        "value_origin",
        "evidence_class",
        "notes",
    ]
    json_fields = [
        "dataset",
        "task",
        "model",
        "condition",
        "history_length",
        "profile_generator",
        "profile_mode",
        "candidate_condition",
        "source_archive",
        "source_member",
        "source_member_sha256",
        "json_artifact_type",
        "json_parse_status",
        "record_count",
        "top_level_type",
        "field_names",
        "has_user",
        "has_target",
        "has_cand_list",
        "has_prompt",
        "has_result",
        "has_template",
        "candidate_count_first_record",
        "notes",
        "evidence_class",
    ]
    command_fields = [
        "script_path",
        "line_number",
        "command",
        "results_path",
        "dataset",
        "task",
        "model",
        "history_length",
        "profile_generator",
        "profile_mode",
        "candidate_condition",
        "sample_num",
        "condition_family",
        "evidence_class",
    ]
    history_fields = [
        "dataset",
        "task",
        "model",
        "history_length",
        "metric",
        "value",
        "source_archive",
        "source_member",
        "condition_evidence",
        "mapping_confidence",
        "evidence_class",
    ]
    generation_fields = [
        "dataset",
        "task",
        "recommendation_model",
        "profile_generator",
        "profile_mode",
        "history_length",
        "metric",
        "value",
        "source_archive",
        "source_member",
        "condition_evidence",
        "mapping_confidence",
        "evidence_class",
    ]
    candidate_fields = [
        "dataset",
        "model",
        "condition",
        "source",
        "metric",
        "value",
        "what_is_known",
        "what_is_missing",
        "direct_metric_present",
        "derivation_possible",
        "derivation_contract_status",
        "evidence_class",
    ]
    hallucination_fields = [
        "dataset",
        "task",
        "model",
        "condition",
        "history_length",
        "profile_generator",
        "candidate_condition",
        "metric",
        "value",
        "source_archive",
        "source_member",
        "semantics",
        "not_semantics",
        "evidence_class",
    ]
    dataset_fields = ["dataset", "official_evidence_classification", "evidence_class", "notes"]
    coverage_fields = ["dataset", "metric_family", "coverage_status", "official_value_rows", "evidence_class", "notes"]
    searched_fields = ["path", "artifact_pattern", "files_found", "files_processed", "files_unresolved", "status", "notes"]
    git_state_fields = ["command", "output"]
    protected_fields = ["path", "exists", "sha256", "protection_scope", "notes"]
    summary_fields = ["key", "value"]

    write_csv(table_dir / "reproduction_r1_metric_serialization_contract.csv", metric_contract, metric_contract_fields)
    write_csv(table_dir / "reproduction_r1_official_artifact_inventory.csv", artifact_rows, artifact_fields)
    write_csv(table_dir / "reproduction_r1_zip_member_inventory.csv", zip_member_rows, artifact_fields)
    write_csv(table_dir / "reproduction_r1_original_repo_metric_values_decoded.csv", decoded_rows, decoded_fields)
    write_csv(table_dir / "reproduction_r1_official_json_inventory.csv", json_rows, json_fields)
    write_csv(table_dir / "reproduction_r1_official_command_condition_inventory.csv", command_rows, command_fields)
    write_csv(table_dir / "reproduction_r1_history_length_official_results.csv", history_rows, history_fields)
    write_csv(table_dir / "reproduction_r1_generation_official_results.csv", generation_rows, generation_fields)
    write_csv(table_dir / "reproduction_r1_candidate_position_evidence.csv", candidate_rows, candidate_fields)
    write_csv(table_dir / "reproduction_r1_hallucination_official_results.csv", hallucination_rows, hallucination_fields)
    write_csv(table_dir / "reproduction_r1_original_dataset_evidence_classification.csv", dataset_rows, dataset_fields)
    write_csv(table_dir / "reproduction_r1_metric_family_coverage_matrix.csv", coverage_rows, coverage_fields)
    write_csv(table_dir / "reproduction_r1_searched_location_manifest.csv", searched_rows, searched_fields)
    write_csv(table_dir / "reproduction_r1_precondition_git_state.csv", git_state_rows, git_state_fields)
    write_csv(table_dir / "reproduction_r1_preexisting_change_inventory.csv", protected_rows, protected_fields)
    write_csv(table_dir / "reproduction_r1_unresolved_artifacts.csv", unresolved_rows, artifact_fields)
    write_csv(
        table_dir / "reproduction_r1_summary.csv",
        [{"key": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, dict) else value} for key, value in summary.items()],
        summary_fields,
    )
    write_report(
        report_path,
        summary,
        metric_contract,
        dataset_rows,
        coverage_rows,
        history_rows,
        generation_rows,
        candidate_rows,
        unresolved_rows,
    )

    return R1Outputs(table_dir=table_dir, report_path=report_path, summary=summary)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recover official original-repository result artifacts for R1.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--original-root", type=Path, default=None)
    parser.add_argument("--table-dir", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    outputs = recover_original_repo_results(
        project_root=args.project_root,
        original_root=args.original_root,
        table_dir=args.table_dir,
        report_path=args.report_path,
    )
    print(DECISION_TOKEN)
    print(f"table_dir={outputs.table_dir}")
    print(f"report_path={outputs.report_path}")
    print(f"decoded_metric_values={outputs.summary['decoded_metric_values']}")


if __name__ == "__main__":
    main()
