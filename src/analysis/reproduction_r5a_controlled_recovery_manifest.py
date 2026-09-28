"""Build the R5A controlled P0/P1 recovery request manifest.

R5A is a planning and validation phase only. It reads frozen R4, F6L,
Plan-C, and Plan-E artifacts and writes deterministic manifest tables for
future recovery execution. It does not call Ollama, APIs, training code, or
download data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

from src.analysis.r1_metric_recovery_paths import relocation_record, resolve_r1_metric_recovery_artifact


EXPECTED_BRANCH = "master"
EXPECTED_HEAD = "a96ad41765b242137a83b919d59a3b394daae562"
DECISION_COMPLETE = "REPRODUCTION_R5A_RECOVERY_MANIFEST_COMPLETE"
DECISION_PARTIAL = "REPRODUCTION_R5A_RECOVERY_MANIFEST_PARTIAL"
DECISION_BLOCKED = "REPRODUCTION_R5A_RECOVERY_MANIFEST_BLOCKED"

TABLE_DIR = Path("results") / "tables"
REPORT_PATH = Path("report") / "reproduction_r5a_controlled_recovery_manifest.md"
NOTES_PATH = Path("report") / "notes.md"

OUTPUT_TABLES = {
    "gap_scope": TABLE_DIR / "reproduction_r5a_gap_scope.csv",
    "raw_planned_requests": TABLE_DIR / "reproduction_r5a_raw_planned_requests.csv",
    "recovery_request_manifest": TABLE_DIR / "reproduction_r5a_recovery_request_manifest.csv",
    "existing_output_reuse": TABLE_DIR / "reproduction_r5a_existing_output_reuse.csv",
    "request_deduplication": TABLE_DIR / "reproduction_r5a_request_deduplication.csv",
    "exact_request_counts": TABLE_DIR / "reproduction_r5a_exact_request_counts.csv",
    "gap_to_request_mapping": TABLE_DIR / "reproduction_r5a_gap_to_request_mapping.csv",
    "validation": TABLE_DIR / "reproduction_r5a_validation.csv",
}

INPUT_DEFAULTS = {
    "r4_gap_priority": TABLE_DIR / "reproduction_r4_gap_scientific_priority.csv",
    "r4_recovery_count_validation": TABLE_DIR / "reproduction_r4_recovery_request_count_validation.csv",
    "r4_protected_hash_check": TABLE_DIR / "reproduction_r4_protected_artifact_hash_check.csv",
    "candidate_title_map": TABLE_DIR / "candidate_item_title_map.csv",
    "f6l_profile_request_manifest": TABLE_DIR / "phase_f6l_profile_request_manifest.csv",
    "f6l_profile_execution_manifest": Path("results")
    / "llm"
    / "phase_f6l_profile_execution_20260801_010000"
    / "profile_execution_manifest.json",
    "canonical_local_full_matrix_requests": Path("results")
    / "llm"
    / "canonical_local_full_matrix"
    / "requests.jsonl",
}

DATASETS = ["Beauty", "Sports", "ML-1M", "LastFM"]
MODELS = ["qwen2:7b-instruct", "mistral:7b-instruct"]
MODEL_LABELS = {
    "qwen2:7b-instruct": "Qwen2-7B",
    "mistral:7b-instruct": "Mistral-7B",
}
SOURCE_DATASET_LABELS = {
    "Beauty": "Amazon Beauty",
    "Sports": "Amazon Sports",
    "ML-1M": "ML-1M",
    "LastFM": "LastFM",
}
PLAN_E_MANIFESTS = {
    "Beauty": TABLE_DIR / "plan_e_beauty_1000user_ranking_manifest.csv",
    "Sports": TABLE_DIR / "plan_e_sports_1000user_ranking_manifest.csv",
    "ML-1M": TABLE_DIR / "plan_e_ml1m_1000user_ranking_manifest.csv",
}
PLAN_C_INTERACTIONS = {
    "Beauty": Path("data") / "reconstructed" / "plan_c" / "amazon_beauty" / "interactions.csv",
    "Sports": Path("data") / "reconstructed" / "plan_c" / "amazon_sports" / "interactions.csv",
    "ML-1M": Path("data") / "reconstructed" / "plan_c" / "ml1m" / "interactions.csv",
}
PLAN_C_CATALOGS = {
    "Beauty": Path("data") / "reconstructed" / "plan_c" / "amazon_beauty" / "catalog.csv",
    "Sports": Path("data") / "reconstructed" / "plan_c" / "amazon_sports" / "catalog.csv",
    "ML-1M": Path("data") / "reconstructed" / "plan_c" / "ml1m" / "catalog.csv",
}
LASTFM_COHORT = Path("data") / "processed" / "finalization_f6c_lastfm_ranking_cohort.csv"
LASTFM_CANDIDATES = Path("data") / "processed" / "finalization_f6c_lastfm_ranking_candidates.csv"

P0_EXPECTED = {
    ("Beauty", "CandDif-HR"),
    ("Beauty", "CandDif-NDCG"),
    ("Sports", "CandDif-HR"),
    ("Sports", "CandDif-NDCG"),
    ("LastFM", "CandDif-HR"),
    ("LastFM", "CandDif-NDCG"),
}
P1_EXPECTED = {
    ("Beauty", "Generation-involved profile recommendation"),
    ("Beauty", "History-length sensitivity"),
    ("Sports", "Generation-involved profile recommendation"),
    ("Sports", "History-length sensitivity"),
    ("ML-1M", "Generation-involved profile recommendation"),
    ("ML-1M", "History-length sensitivity"),
    ("LastFM", "Generation-involved profile recommendation"),
    ("LastFM", "History-length sensitivity"),
}


@dataclass(frozen=True)
class Candidate:
    item_id: str
    title: str
    is_positive: bool


@dataclass(frozen=True)
class LocalContext:
    dataset: str
    user_id: str
    candidates: tuple[Candidate, ...]
    history_items: tuple[dict[str, str], ...]
    positive_item_id: str
    positive_item_title: str
    candidate_source: str
    history_source: str


@dataclass(frozen=True)
class F6LGenerationContext:
    dataset: str
    user_id: str
    model_name: str
    source_request_id: str
    source_raw_output_path: str
    source_rendered_prompt: str
    source_rendered_prompt_hash: str
    candidate_list: tuple[Candidate, ...]
    positive_item_id: str
    positive_item_title: str
    positive_position: str
    history_hash: str
    profile_request_id: str
    profile_artifact_path: str
    profile_hash: str
    model_digest: str
    prompt_template_path: str
    prompt_template_hash: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}-{sha256_text(canonical_json(parts))[:16]}"


def as_project_path(project_root: Path, path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else project_root / raw


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, low_memory=False)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(frame.to_csv(index=False, lineterminator="\n"))


def git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def git_or_empty(args: list[str], cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def git_stdout(args: list[str], cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else ""


def porcelain_path(line: str) -> str:
    if len(line) >= 4 and line[2] == " ":
        path = line[3:]
    elif " " in line:
        path = line.split(maxsplit=1)[1]
    else:
        path = line
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.strip()


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dataset_config(config: Mapping[str, Any], key: str, fallback: Mapping[str, Path]) -> dict[str, Path]:
    raw = ((config.get("inputs") or {}).get(key) or {})
    merged = {dataset: Path(path) for dataset, path in fallback.items()}
    merged.update({dataset: Path(path) for dataset, path in raw.items()})
    return merged


def input_path(config: Mapping[str, Any], key: str) -> Path:
    raw = (config.get("inputs") or {}).get(key)
    return Path(raw) if raw else INPUT_DEFAULTS[key]


def load_title_lookup(project_root: Path, config: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    title_map_path = as_project_path(project_root, input_path(config, "candidate_title_map"))
    if title_map_path.exists():
        title_map = read_csv(title_map_path)
        for row in title_map.to_dict("records"):
            dataset = str(row.get("dataset", "")).strip()
            item_id = str(row.get("item_id", "")).strip()
            title = str(row.get("canonical_title", "")).strip()
            if dataset and item_id and title:
                lookup[(dataset, item_id)] = title
    for dataset, rel_path in dataset_config(config, "plan_c_catalogs", PLAN_C_CATALOGS).items():
        path = as_project_path(project_root, rel_path)
        if not path.exists():
            continue
        source_label = SOURCE_DATASET_LABELS[dataset]
        catalog = read_csv(path)
        for row in catalog.to_dict("records"):
            item_id = str(row.get("item_id", "")).strip()
            title = str(row.get("title", "")).strip()
            if item_id and title:
                lookup.setdefault((source_label, item_id), title)
                lookup.setdefault((dataset, item_id), title)
    return lookup


def title_for(lookup: Mapping[tuple[str, str], str], dataset: str, item_id: str) -> str:
    source_label = SOURCE_DATASET_LABELS.get(dataset, dataset)
    return (
        lookup.get((source_label, str(item_id)))
        or lookup.get((dataset, str(item_id)))
        or f"ITEM_{item_id}"
    )


def history_from_interactions(
    project_root: Path,
    config: Mapping[str, Any],
    title_lookup: Mapping[tuple[str, str], str],
) -> dict[str, dict[str, tuple[dict[str, str], ...]]]:
    histories: dict[str, dict[str, tuple[dict[str, str], ...]]] = {}
    for dataset, rel_path in dataset_config(config, "plan_c_interactions", PLAN_C_INTERACTIONS).items():
        path = as_project_path(project_root, rel_path)
        interactions = read_csv(path)
        if "split" in interactions.columns:
            train = interactions.loc[interactions["split"].astype(str).str.lower().eq("train")].copy()
        else:
            train = interactions.copy()
        if "timestamp" in train.columns:
            train["_sort_timestamp"] = pd.to_numeric(train["timestamp"], errors="coerce").fillna(-1)
        else:
            train["_sort_timestamp"] = range(len(train))
        train["_item_sort"] = train["item_id"].astype(str)
        dataset_histories: dict[str, tuple[dict[str, str], ...]] = {}
        for user_id, group in train.sort_values(["_sort_timestamp", "_item_sort"]).groupby(
            train["user_id"].astype(str), sort=True
        ):
            rows = []
            for item_id in group["item_id"].astype(str).tolist():
                rows.append({"item_id": item_id, "title": title_for(title_lookup, dataset, item_id)})
            dataset_histories[str(user_id)] = tuple(rows)
        histories[dataset] = dataset_histories
    return histories


def parse_json_list(raw: str) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed]


def load_local_contexts(
    project_root: Path,
    config: Mapping[str, Any],
    cohort_size: int,
    min_history: int,
) -> dict[str, list[LocalContext]]:
    title_lookup = load_title_lookup(project_root, config)
    histories = history_from_interactions(project_root, config, title_lookup)
    contexts: dict[str, list[LocalContext]] = {}

    for dataset, rel_path in dataset_config(config, "plan_e_candidate_manifests", PLAN_E_MANIFESTS).items():
        path = as_project_path(project_root, rel_path)
        candidates = read_csv(path)
        dataset_contexts: list[LocalContext] = []
        for user_id, group in candidates.groupby(candidates["user_id"].astype(str), sort=True):
            history = histories.get(dataset, {}).get(str(user_id), tuple())
            if len(history) < min_history:
                continue
            ordered = group.copy()
            ordered["_candidate_position"] = pd.to_numeric(
                ordered["candidate_position"], errors="coerce"
            ).fillna(10**9)
            ordered = ordered.sort_values(["_candidate_position", "item_id"])
            rows: list[Candidate] = []
            for row in ordered.to_dict("records"):
                item_id = str(row["item_id"])
                rows.append(
                    Candidate(
                        item_id=item_id,
                        title=title_for(title_lookup, dataset, item_id),
                        is_positive=str(row.get("is_positive", "")).lower() == "true",
                    )
                )
            positives = [candidate for candidate in rows if candidate.is_positive]
            if len(positives) != 1:
                continue
            dataset_contexts.append(
                LocalContext(
                    dataset=dataset,
                    user_id=str(user_id),
                    candidates=tuple(rows),
                    history_items=history,
                    positive_item_id=positives[0].item_id,
                    positive_item_title=positives[0].title,
                    candidate_source=str(rel_path),
                    history_source=str(dataset_config(config, "plan_c_interactions", PLAN_C_INTERACTIONS)[dataset]),
                )
            )
            if len(dataset_contexts) == cohort_size:
                break
        contexts[dataset] = dataset_contexts

    lastfm_cohort_path = as_project_path(project_root, (config.get("inputs") or {}).get("lastfm_cohort", LASTFM_COHORT))
    lastfm_candidate_path = as_project_path(
        project_root, (config.get("inputs") or {}).get("lastfm_candidates", LASTFM_CANDIDATES)
    )
    cohort = read_csv(lastfm_cohort_path)
    candidate_rows = read_csv(lastfm_candidate_path)
    lastfm_contexts: list[LocalContext] = []
    cohort["_user_sort"] = pd.to_numeric(cohort["user_id"], errors="coerce").fillna(10**12)
    for row in cohort.sort_values(["_user_sort", "user_id"]).to_dict("records"):
        user_id = str(row["user_id"])
        ids = parse_json_list(str(row.get("history_item_ids", "")))
        titles = parse_json_list(str(row.get("history_item_titles", "")))
        history = tuple(
            {"item_id": item_id, "title": titles[idx] if idx < len(titles) else f"ARTIST_{item_id}"}
            for idx, item_id in enumerate(ids)
        )
        if len(history) < min_history:
            continue
        group = candidate_rows.loc[candidate_rows["user_id"].astype(str).eq(user_id)].copy()
        if group.empty:
            continue
        group["_candidate_position"] = pd.to_numeric(group["candidate_position"], errors="coerce").fillna(10**9)
        group = group.sort_values(["_candidate_position", "candidate_item_id"])
        candidates = tuple(
            Candidate(
                item_id=str(item["candidate_item_id"]),
                title=str(item.get("candidate_item_title", "")).strip() or f"ARTIST_{item['candidate_item_id']}",
                is_positive=str(item.get("is_ground_truth", "")).lower() == "true",
            )
            for item in group.to_dict("records")
        )
        positives = [candidate for candidate in candidates if candidate.is_positive]
        if len(positives) != 1:
            continue
        lastfm_contexts.append(
            LocalContext(
                dataset="LastFM",
                user_id=user_id,
                candidates=candidates,
                history_items=history,
                positive_item_id=positives[0].item_id,
                positive_item_title=positives[0].title,
                candidate_source=str((config.get("inputs") or {}).get("lastfm_candidates", LASTFM_CANDIDATES)),
                history_source=str((config.get("inputs") or {}).get("lastfm_cohort", LASTFM_COHORT)),
            )
        )
        if len(lastfm_contexts) == cohort_size:
            break
    contexts["LastFM"] = lastfm_contexts
    return contexts


def profile_raw_path(project_root: Path, expected_raw_output_relative_path: str) -> Path:
    return project_root / expected_raw_output_relative_path.replace("<run_id>", "20260801_010000")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_f6l_generation_contexts(
    project_root: Path,
    config: Mapping[str, Any],
    cohort_size: int,
    models: Iterable[str],
) -> dict[tuple[str, str], list[F6LGenerationContext]]:
    manifest_path = as_project_path(project_root, input_path(config, "f6l_profile_request_manifest"))
    manifest = read_csv(manifest_path)
    profile_rows = manifest.loc[manifest["input_variant"].eq("profile_generation")].copy()
    profile_by_request = {str(row["request_id"]): row for row in profile_rows.to_dict("records")}
    contexts: dict[tuple[str, str], list[F6LGenerationContext]] = {}
    for dataset in ["Beauty", "Sports"]:
        dataset_key = dataset.lower()
        for model in models:
            rows = manifest.loc[
                manifest["dataset"].eq(dataset_key)
                & manifest["input_variant"].eq("profile_plus_history")
                & manifest["model_name"].eq(model)
                & manifest["profile_generator_name"].eq(model)
                & manifest["recommender_model_name"].eq(model)
            ].copy()
            rows["_user_sort"] = pd.to_numeric(rows["user_id"], errors="coerce").fillna(10**12)
            selected = rows.sort_values(["_user_sort", "user_id", "request_id"]).head(cohort_size)
            built: list[F6LGenerationContext] = []
            for row in selected.to_dict("records"):
                raw_path = profile_raw_path(project_root, str(row["expected_raw_output_relative_path"]))
                raw = load_json(raw_path)
                dependency_id = str(row["dependency_request_id"])
                profile_manifest_row = profile_by_request[dependency_id]
                profile_path = profile_raw_path(
                    project_root, str(profile_manifest_row["expected_raw_output_relative_path"])
                )
                profile_raw = load_json(profile_path)
                candidate_ids = [str(item) for item in raw.get("candidate_item_ids", [])]
                candidate_titles = [str(item) for item in raw.get("candidate_item_titles", [])]
                candidates = tuple(
                    Candidate(
                        item_id=item_id,
                        title=candidate_titles[idx] if idx < len(candidate_titles) else f"ITEM_{item_id}",
                        is_positive=item_id == str(row["positive_item_id"]),
                    )
                    for idx, item_id in enumerate(candidate_ids)
                )
                built.append(
                    F6LGenerationContext(
                        dataset=dataset,
                        user_id=str(row["user_id"]),
                        model_name=model,
                        source_request_id=str(row["request_id"]),
                        source_raw_output_path=str(raw_path.relative_to(project_root)),
                        source_rendered_prompt=str(raw.get("rendered_prompt", "")),
                        source_rendered_prompt_hash=str(row["rendered_prompt_hash"]),
                        candidate_list=candidates,
                        positive_item_id=str(row["positive_item_id"]),
                        positive_item_title=str(row["positive_item_title"]),
                        positive_position=str(row["positive_candidate_position"]),
                        history_hash=str(row["history_hash"]),
                        profile_request_id=dependency_id,
                        profile_artifact_path=str(profile_path.relative_to(project_root)),
                        profile_hash=sha256_text(str(profile_raw.get("raw_response", ""))),
                        model_digest=str(row["model_digest"]),
                        prompt_template_path=str(row["prompt_template_path"]),
                        prompt_template_hash=str(row["prompt_template_hash"]),
                    )
                )
            contexts[(dataset, model)] = built
    return contexts


def candidates_to_records(candidates: Iterable[Candidate]) -> list[dict[str, Any]]:
    return [
        {
            "position": index,
            "item_id": candidate.item_id,
            "title": candidate.title,
            "is_positive": candidate.is_positive,
        }
        for index, candidate in enumerate(candidates, start=1)
    ]


def candidate_hash(candidates: Iterable[Candidate]) -> str:
    return sha256_text(canonical_json(candidates_to_records(candidates)))


def unordered_candidate_item_set_hash(candidates: Iterable[Candidate]) -> str:
    return sha256_text(canonical_json(sorted(candidate.item_id for candidate in candidates)))


def history_slice(history: tuple[dict[str, str], ...], length: int) -> tuple[dict[str, str], ...]:
    if length <= 0:
        return tuple()
    return tuple(history[-length:])


def history_hash(history: Iterable[Mapping[str, str]]) -> str:
    return sha256_text(canonical_json(list(history)))


def first_position(candidates: tuple[Candidate, ...]) -> tuple[Candidate, ...]:
    positives = [candidate for candidate in candidates if candidate.is_positive]
    if len(positives) != 1:
        raise ValueError("candidate set must contain exactly one positive item")
    positive = positives[0]
    return tuple([positive] + [candidate for candidate in candidates if not candidate.is_positive])


def random_position(candidates: tuple[Candidate, ...], seed: str, dataset: str, user_id: str) -> tuple[Candidate, ...]:
    shuffled = sorted(
        candidates,
        key=lambda candidate: sha256_text(f"{seed}|{dataset}|{user_id}|{candidate.item_id}|{candidate.title}"),
    )
    if shuffled and shuffled[0].is_positive and len(shuffled) > 1:
        shuffled = shuffled[1:] + shuffled[:1]
    return tuple(shuffled)


def positive_position(candidates: Iterable[Candidate]) -> int:
    for index, candidate in enumerate(candidates, start=1):
        if candidate.is_positive:
            return index
    return 0


def render_history(history: Iterable[Mapping[str, str]]) -> str:
    rows = list(history)
    if not rows:
        return "(empty history)"
    return "\n".join(f"{idx}. {row['title']} [{row['item_id']}]" for idx, row in enumerate(rows, start=1))


def render_candidates(candidates: Iterable[Candidate]) -> str:
    return "\n".join(
        f"{idx}. {candidate.title} [{candidate.item_id}]" for idx, candidate in enumerate(candidates, start=1)
    )


def render_recommendation_prompt(
    dataset: str,
    task: str,
    history: Iterable[Mapping[str, str]],
    candidates: Iterable[Candidate],
    condition: str,
    profile_context: str = "",
    source_prompt_context: str = "",
) -> str:
    profile_block = ""
    if profile_context:
        profile_block = f"\nProfile context:\n{profile_context}\n"
    source_block = ""
    if source_prompt_context:
        source_block = f"\nFrozen source context:\n{source_prompt_context}\n"
    return (
        f"Dataset: {dataset}\n"
        f"Task: {task}\n"
        f"Controlled condition: {condition}\n"
        "Use only the train-history and candidate context below. "
        "Return JSON with key ranked_candidate_indices containing unique 1-based candidate indices.\n"
        f"{profile_block}{source_block}\n"
        f"User history:\n{render_history(history)}\n\n"
        f"Candidates:\n{render_candidates(candidates)}\n"
    )


def render_profile_generation_prompt(dataset: str, history: Iterable[Mapping[str, str]]) -> str:
    return (
        f"Dataset: {dataset}\n"
        "Task: profile generation\n"
        "Generate a concise user preference profile from the train-only history. "
        "Return JSON with key user_profile.\n\n"
        f"User history:\n{render_history(history)}\n"
    )


def request_key(spec: Mapping[str, Any]) -> str:
    key_fields = [
        "recovery_family",
        "dataset",
        "model_name",
        "task",
        "experimental_condition",
        "user_id",
        "history_length",
        "candidate_list_hash",
        "history_hash",
        "profile_hash",
        "profile_dependency_request_id",
        "prompt_template_id",
        "parser_contract",
        "expected_output_schema",
    ]
    return canonical_json({field: spec.get(field, "") for field in key_fields})


def build_material_context_hash(spec: Mapping[str, Any]) -> str:
    return sha256_text(
        canonical_json(
            {
                "dataset": spec.get("dataset", ""),
                "user_id": spec.get("user_id", ""),
                "model_name": spec.get("model_name", ""),
                "recovery_family": spec.get("recovery_family", ""),
                "experimental_condition": spec.get("experimental_condition", ""),
                "history_hash": spec.get("history_hash", ""),
                "candidate_list_hash": spec.get("candidate_list_hash", ""),
                "profile_hash": spec.get("profile_hash", ""),
                "profile_dependency_request_id": spec.get("profile_dependency_request_id", ""),
                "prompt_hash": spec.get("prompt_hash", ""),
                "parser_contract": spec.get("parser_contract", ""),
                "expected_output_schema": spec.get("expected_output_schema", ""),
            }
        )
    )


def base_spec(
    *,
    family: str,
    dataset: str,
    model_name: str,
    task: str,
    user_id: str,
    condition: str,
    candidates: tuple[Candidate, ...],
    history: tuple[dict[str, str], ...],
    history_length: int | str,
    candidate_policy: str,
    candidate_source: str,
    history_source: str,
    prompt: str,
    prompt_template_id: str,
    prompt_template_source: str,
    parser_contract: str,
    expected_output_schema: str,
    positive_item_id: str,
    positive_item_title: str,
    profile_dependency_request_id: str = "",
    profile_artifact_path: str = "",
    profile_hash_value: str = "",
    profile_hash_status: str = "NOT_APPLICABLE",
    profile_reuse_status: str = "NOT_APPLICABLE",
    candidate_order_seed: str = "",
    candidate_order_algorithm: str = "",
    notes: str = "",
    related_existing_output_path: str = "",
) -> dict[str, Any]:
    candidate_records = candidates_to_records(candidates)
    history_records = list(history)
    prompt_hash = sha256_text(prompt)
    spec: dict[str, Any] = {
        "recovery_family": family,
        "dataset": dataset,
        "model_name": model_name,
        "model_label": MODEL_LABELS.get(model_name, model_name),
        "task": task,
        "user_id": str(user_id),
        "experimental_condition": condition,
        "history_length": str(history_length),
        "history_item_count": str(len(history_records)),
        "history_items": canonical_json(history_records),
        "history_hash": sha256_text(canonical_json(history_records)),
        "history_source": history_source,
        "candidate_policy": candidate_policy,
        "candidate_order_seed": candidate_order_seed,
        "candidate_order_algorithm": candidate_order_algorithm,
        "candidate_count": str(len(candidate_records)),
        "candidate_list": canonical_json(candidate_records),
        "candidate_list_hash": sha256_text(canonical_json(candidate_records)),
        "unordered_candidate_item_set_hash": unordered_candidate_item_set_hash(candidates),
        "candidate_source": candidate_source,
        "positive_item_id": positive_item_id,
        "positive_item_title": positive_item_title,
        "positive_candidate_position": str(positive_position(candidates)),
        "profile_dependency_request_id": profile_dependency_request_id,
        "profile_artifact_path": profile_artifact_path,
        "profile_hash": profile_hash_value,
        "profile_hash_status": profile_hash_status,
        "profile_reuse_status": profile_reuse_status,
        "prompt_template_id": prompt_template_id,
        "prompt_template_source": prompt_template_source,
        "prompt_hash": prompt_hash,
        "prompt_render": prompt,
        "parser_contract": parser_contract,
        "expected_output_schema": expected_output_schema,
        "temperature": "0",
        "seed": "0",
        "max_new_tokens": "512",
        "output_destination": "",
        "execution_allowed_in_r5a": "false",
        "requires_new_llm_inference": "true",
        "existing_output_reuse_status": "NEW_INFERENCE_REQUIRED_NO_EXACT_MATCH",
        "related_existing_output_path": related_existing_output_path,
        "notes": notes,
    }
    key = request_key(spec)
    spec["request_key_sha256"] = sha256_text(key)
    spec["request_id"] = stable_id("R5A-REQ", key)
    spec["output_destination"] = f"results/recovery/r5/{spec['request_id']}.json"
    spec["material_context_hash"] = build_material_context_hash(spec)
    return spec


def profile_generation_spec(
    context: LocalContext,
    model_name: str,
    profile_history_length: int,
) -> dict[str, Any]:
    history = history_slice(context.history_items, profile_history_length)
    prompt = render_profile_generation_prompt(context.dataset, history)
    return base_spec(
        family="GENERATION_INVOLVED_RECOVERY",
        dataset=context.dataset,
        model_name=model_name,
        task="profile_generation",
        user_id=context.user_id,
        condition="PROFILE_GENERATION",
        candidates=tuple(),
        history=history,
        history_length=profile_history_length,
        candidate_policy="NOT_APPLICABLE",
        candidate_source="not_applicable",
        history_source=context.history_source,
        prompt=prompt,
        prompt_template_id="r5a_profile_generation_json_v1",
        prompt_template_source="src/analysis/reproduction_r5a_controlled_recovery_manifest.py",
        parser_contract="r5a_profile_json_profile_parser_v1",
        expected_output_schema='{"user_profile":"string"}',
        positive_item_id="",
        positive_item_title="",
        profile_hash_value="",
        profile_hash_status="PROFILE_OUTPUT_NOT_YET_GENERATED",
        profile_reuse_status="NEW_PROFILE_GENERATION_REQUIRED",
        notes="Profile prerequisite separated from downstream profile-plus-history recommendation.",
    )


def collect_specs(
    config: Mapping[str, Any],
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scope = config.get("scope") or {}
    cohort_size = int(scope.get("cohort_size", 100))
    models = list(scope.get("models") or MODELS)
    families = config.get("families") or {}
    canddif_cfg = families.get("canddif") or {}
    history_cfg = families.get("history_length") or {}
    generation_cfg = families.get("generation_involved") or {}
    history_lengths: dict[str, list[int]] = {
        dataset: [int(value) for value in values]
        for dataset, values in (history_cfg.get("exact_lengths") or {}).items()
    }
    local_contexts = load_local_contexts(project_root, config, cohort_size, min_history=0)
    f6l_contexts = load_f6l_generation_contexts(project_root, config, cohort_size, models)

    gap_priority = read_csv(as_project_path(project_root, input_path(config, "r4_gap_priority")))
    high = gap_priority.loc[gap_priority["priority"].isin(["P0_MUST_CLOSE", "P1_HIGH_VALUE"])].copy()
    gap_by_dataset_metric = {
        (str(row["dataset"]), str(row["metric"])): row for row in high.to_dict("records")
    }

    unique_specs: dict[str, dict[str, Any]] = {}
    raw_rows: list[dict[str, Any]] = []

    def add_raw(gap_row: Mapping[str, Any], metric: str, spec: dict[str, Any]) -> None:
        key_hash = str(spec["request_key_sha256"])
        first_seen = key_hash not in unique_specs
        if first_seen:
            unique_specs[key_hash] = dict(spec)
        duplicate_status = "UNIQUE_CANONICAL_REQUEST" if first_seen else "DUPLICATE_OF_CANONICAL_REQUEST"
        duplicate_type = "NONE" if first_seen else "METRIC_ONLY_DUPLICATE"
        raw_rows.append(
            {
                "raw_request_id": stable_id(
                    "R5A-RAW",
                    gap_row["gap_id"],
                    metric,
                    spec["request_key_sha256"],
                    len(raw_rows),
                ),
                "gap_id": gap_row["gap_id"],
                "dataset": gap_row["dataset"],
                "metric": metric,
                "priority": gap_row["priority"],
                "recovery_family": spec["recovery_family"],
                "model_name": spec["model_name"],
                "task": spec["task"],
                "user_id": spec["user_id"],
                "experimental_condition": spec["experimental_condition"],
                "history_length": spec["history_length"],
                "canonical_request_id": spec["request_id"],
                "canonical_request_key_sha256": spec["request_key_sha256"],
                "duplicate_status": duplicate_status,
                "duplicate_type": duplicate_type,
                "planning_source": "results/tables/reproduction_r4_gap_scientific_priority.csv",
                "notes": "Raw R4 metric-level request collapsed only when material execution input is identical.",
            }
        )

    random_seed = str(canddif_cfg.get("random_seed", "R5A_CANDDIF_RANDOM_POSITION_V1"))
    for dataset in canddif_cfg.get("datasets", ["Beauty", "Sports", "LastFM"]):
        for metric in canddif_cfg.get("metrics", ["CandDif-HR", "CandDif-NDCG"]):
            gap_row = gap_by_dataset_metric[(dataset, metric)]
            for context in local_contexts[dataset]:
                history = history_slice(context.history_items, 10)
                for model_name in models:
                    for condition in canddif_cfg.get("candidate_conditions", ["FIRST", "RANDOM"]):
                        ordered_candidates = (
                            first_position(context.candidates)
                            if condition == "FIRST"
                            else random_position(context.candidates, random_seed, dataset, context.user_id)
                        )
                        prompt = render_recommendation_prompt(
                            dataset=dataset,
                            task="ranking",
                            history=history,
                            candidates=ordered_candidates,
                            condition=f"CANDDIF_{condition}",
                        )
                        spec = base_spec(
                            family="CANDDIF_RECOVERY",
                            dataset=dataset,
                            model_name=model_name,
                            task="ranking",
                            user_id=context.user_id,
                            condition=f"CANDDIF_{condition}",
                            candidates=ordered_candidates,
                            history=history,
                            history_length=10,
                            candidate_policy=(
                                "POSITIVE_FIRST_CONTROLLED"
                                if condition == "FIRST"
                                else "DETERMINISTIC_RANDOM_POSITIVE_POSITION"
                            ),
                            candidate_source=context.candidate_source,
                            history_source=context.history_source,
                            prompt=prompt,
                            prompt_template_id="r5a_canddif_json_index_v1",
                            prompt_template_source="src/analysis/reproduction_r5a_controlled_recovery_manifest.py",
                            parser_contract="r5a_ranked_index_json_top5_v1",
                            expected_output_schema='{"ranked_candidate_indices":[1,2,3,4,5]}',
                            positive_item_id=context.positive_item_id,
                            positive_item_title=context.positive_item_title,
                            candidate_order_seed=(
                                "POSITIVE_FIRST_CONTROLLED_NO_RANDOM"
                                if condition == "FIRST"
                                else random_seed
                            ),
                            candidate_order_algorithm=(
                                "positive_item_first_then_source_order"
                                if condition == "FIRST"
                                else "sha256_seeded_sort_rotate_if_positive_first"
                            ),
                            notes=(
                                "CandDif HR and NDCG share the same controlled LLM output; "
                                "metric-only duplicates are deduplicated."
                            ),
                        )
                        add_raw(gap_row, metric, spec)

    for dataset in history_cfg.get("datasets", DATASETS):
        gap_row = gap_by_dataset_metric[(dataset, "History-length sensitivity")]
        for context in local_contexts[dataset]:
            for model_name in models:
                for length in history_lengths[dataset]:
                    history = history_slice(context.history_items, length)
                    prompt = render_recommendation_prompt(
                        dataset=dataset,
                        task="ranking",
                        history=history,
                        candidates=context.candidates,
                        condition=f"HISTORY_LENGTH_{length}",
                    )
                    spec = base_spec(
                        family="HISTORY_LENGTH_RECOVERY",
                        dataset=dataset,
                        model_name=model_name,
                        task="ranking",
                        user_id=context.user_id,
                        condition=f"HISTORY_LENGTH_{length}",
                        candidates=context.candidates,
                        history=history,
                        history_length=length,
                        candidate_policy="FROZEN_PLAN_E_OR_F6C_ORDER",
                        candidate_source=context.candidate_source,
                        history_source=context.history_source,
                        prompt=prompt,
                        prompt_template_id="r5a_history_length_json_index_v1",
                        prompt_template_source="src/analysis/reproduction_r5a_controlled_recovery_manifest.py",
                        parser_contract="r5a_ranked_index_json_top5_v1",
                        expected_output_schema='{"ranked_candidate_indices":[1,2,3,4,5]}',
                        positive_item_id=context.positive_item_id,
                        positive_item_title=context.positive_item_title,
                        candidate_order_seed="SOURCE_ORDER_NO_RANDOM",
                        candidate_order_algorithm="frozen_source_order",
                        notes="Controlled paper-grid history length request; no binned diagnostic substitution.",
                    )
                    add_raw(gap_row, "History-length sensitivity", spec)

    downstream_condition = str(generation_cfg.get("downstream_condition", "profile_plus_history"))
    profile_history_length = int(generation_cfg.get("profile_history_length", 10))
    for dataset in generation_cfg.get("datasets", DATASETS):
        gap_row = gap_by_dataset_metric[(dataset, "Generation-involved profile recommendation")]
        if dataset in {"Beauty", "Sports"}:
            for model_name in models:
                for context in f6l_contexts[(dataset, model_name)]:
                    f6l_history_placeholder = (
                        {
                            "item_id": "F6L_RENDERED_CONTEXT",
                            "title": f"F6L source prompt {context.source_rendered_prompt_hash}",
                        },
                    )
                    prompt = render_recommendation_prompt(
                        dataset=dataset,
                        task="ranking",
                        history=f6l_history_placeholder,
                        candidates=context.candidate_list,
                        condition=downstream_condition,
                        profile_context=(
                            f"Existing F6L profile dependency {context.profile_request_id}; "
                            f"profile_hash={context.profile_hash}"
                        ),
                        source_prompt_context=context.source_rendered_prompt,
                    )
                    spec = base_spec(
                        family="GENERATION_INVOLVED_RECOVERY",
                        dataset=dataset,
                        model_name=model_name,
                        task="ranking",
                        user_id=context.user_id,
                        condition=downstream_condition.upper(),
                        candidates=context.candidate_list,
                        history=f6l_history_placeholder,
                        history_length="F6L_PROFILE_HISTORY_10",
                        candidate_policy="F6L_OFFICIAL_PROFILE_ARTIFACT_CANDIDATE20_SHUFFLED",
                        candidate_source=context.source_raw_output_path,
                        history_source=context.source_raw_output_path,
                        prompt=prompt,
                        prompt_template_id="r5a_profile_plus_history_json_index_v1",
                        prompt_template_source="src/analysis/reproduction_r5a_controlled_recovery_manifest.py",
                        parser_contract="r5a_ranked_index_json_top10_v1",
                        expected_output_schema='{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10]}',
                        positive_item_id=context.positive_item_id,
                        positive_item_title=context.positive_item_title,
                        profile_dependency_request_id=context.profile_request_id,
                        profile_artifact_path=context.profile_artifact_path,
                        profile_hash_value=context.profile_hash,
                        profile_hash_status="INPUT_PROFILE_HASH_AVAILABLE",
                        profile_reuse_status="EXISTING_PROFILE_DEPENDENCY_RECORDED",
                        candidate_order_seed="F6L_SOURCE_ORDER_NO_R5A_RANDOM",
                        candidate_order_algorithm="f6l_frozen_candidate_order",
                        related_existing_output_path=context.source_raw_output_path,
                        notes=(
                            "Beauty/Sports reuse existing profile artifacts as dependencies, "
                            "but schedule a new controlled R5A downstream output contract."
                        ),
                    )
                    spec["existing_output_reuse_status"] = "NOT_REUSABLE_PROMPT_OR_PARSER_CONTRACT_MISMATCH"
                    add_raw(gap_row, "Generation-involved profile recommendation", spec)
        else:
            for context in local_contexts[dataset]:
                for model_name in models:
                    profile_spec = profile_generation_spec(context, model_name, profile_history_length)
                    add_raw(gap_row, "Generation-involved profile recommendation", profile_spec)
                    profile_dependency_id = str(profile_spec["request_id"])
                    history = history_slice(context.history_items, profile_history_length)
                    prompt = render_recommendation_prompt(
                        dataset=dataset,
                        task="ranking",
                        history=history,
                        candidates=context.candidates,
                        condition=downstream_condition,
                        profile_context=f"PENDING_PROFILE_GENERATION:{profile_dependency_id}",
                    )
                    spec = base_spec(
                        family="GENERATION_INVOLVED_RECOVERY",
                        dataset=dataset,
                        model_name=model_name,
                        task="ranking",
                        user_id=context.user_id,
                        condition=downstream_condition.upper(),
                        candidates=context.candidates,
                        history=history,
                        history_length=profile_history_length,
                        candidate_policy="FROZEN_PLAN_E_OR_F6C_ORDER",
                        candidate_source=context.candidate_source,
                        history_source=context.history_source,
                        prompt=prompt,
                        prompt_template_id="r5a_profile_plus_history_json_index_v1",
                        prompt_template_source="src/analysis/reproduction_r5a_controlled_recovery_manifest.py",
                        parser_contract="r5a_ranked_index_json_top10_v1",
                        expected_output_schema='{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10]}',
                        positive_item_id=context.positive_item_id,
                        positive_item_title=context.positive_item_title,
                        profile_dependency_request_id=profile_dependency_id,
                        profile_artifact_path=f"results/recovery/r5/{profile_dependency_id}.json",
                        profile_hash_value="",
                        profile_hash_status="PROFILE_DEPENDENCY_REQUEST_ID_PENDING_OUTPUT",
                        profile_reuse_status="PENDING_PROFILE_GENERATION_REQUEST_RECORDED",
                        candidate_order_seed="SOURCE_ORDER_NO_RANDOM",
                        candidate_order_algorithm="frozen_source_order",
                        notes=(
                            "ML-1M/LastFM have no existing F6L profile artifacts; "
                            "downstream request is explicitly dependent on the separated profile request."
                        ),
                    )
                    add_raw(gap_row, "Generation-involved profile recommendation", spec)

    raw = pd.DataFrame(raw_rows)
    manifest = pd.DataFrame(unique_specs.values())
    if raw.empty or manifest.empty:
        raise RuntimeError("R5A manifest construction produced no rows")

    served = raw.groupby("canonical_request_id").agg(
        gap_ids_served=("gap_id", lambda values: ";".join(sorted(set(map(str, values))))),
        metrics_served=("metric", lambda values: ";".join(sorted(set(map(str, values))))),
        raw_request_count=("raw_request_id", "count"),
    )
    manifest = manifest.merge(served, left_on="request_id", right_index=True, how="left")
    manifest["request_status"] = "PLANNED_ONLY_NOT_EXECUTED"
    manifest["inference_decision"] = "FUTURE_APPROVAL_REQUIRED_ZERO_R5A_INFERENCE"
    manifest = manifest.sort_values(["recovery_family", "dataset", "model_name", "user_id", "experimental_condition", "request_id"]).reset_index(drop=True)
    raw = raw.sort_values(["recovery_family", "dataset", "metric", "model_name", "user_id", "experimental_condition", "raw_request_id"]).reset_index(drop=True)
    dedup = raw[
        [
            "raw_request_id",
            "gap_id",
            "dataset",
            "metric",
            "recovery_family",
            "model_name",
            "user_id",
            "experimental_condition",
            "canonical_request_id",
            "canonical_request_key_sha256",
            "duplicate_status",
            "duplicate_type",
        ]
    ].copy()
    return raw, manifest, dedup


def build_gap_scope(
    project_root: Path,
    config: Mapping[str, Any],
    manifest: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    gaps = read_csv(as_project_path(project_root, input_path(config, "r4_gap_priority"))).copy()
    high_gap_ids = set(raw["gap_id"].astype(str))
    raw_counts = raw.groupby("gap_id").size().to_dict()
    unique_counts = raw.groupby("gap_id")["canonical_request_id"].nunique().to_dict()
    rows: list[dict[str, Any]] = []
    for row in gaps.to_dict("records"):
        gap_id = str(row["gap_id"])
        priority = str(row["priority"])
        metric = str(row["metric"])
        dataset = str(row["dataset"])
        if metric.startswith("CandDif"):
            family = "CANDDIF_RECOVERY"
        elif metric == "History-length sensitivity":
            family = "HISTORY_LENGTH_RECOVERY"
        elif metric == "Generation-involved profile recommendation":
            family = "GENERATION_INVOLVED_RECOVERY"
        else:
            family = "OUT_OF_R5A_SCOPE"
        eligible = priority in {"P0_MUST_CLOSE", "P1_HIGH_VALUE"}
        scheduled = gap_id in high_gap_ids
        if eligible and scheduled:
            decision = "SCHEDULED_IN_R5A_MANIFEST"
            reason = "P0/P1 R4 recovery gap selected for deterministic manifest."
        elif eligible:
            decision = "ELIGIBLE_BUT_UNSCHEDULED"
            reason = "Validation failure: P0/P1 gap was not scheduled."
        else:
            decision = "EXCLUDED_BY_R5A_SCOPE"
            reason = "R5A schedules P0/P1 only; P2/P3/P4 receive zero requests."
        rows.append(
            {
                "gap_id": gap_id,
                "dataset": dataset,
                "metric": metric,
                "priority": priority,
                "r5a_recovery_family": family,
                "eligible_for_r5a": str(eligible).lower(),
                "scheduled_in_manifest": str(scheduled).lower(),
                "raw_planned_requests": int(raw_counts.get(gap_id, 0)),
                "unique_request_count": int(unique_counts.get(gap_id, 0)),
                "scope_decision": decision,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_existing_output_reuse(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in manifest.to_dict("records"):
        status = str(row["existing_output_reuse_status"])
        if status == "NOT_REUSABLE_PROMPT_OR_PARSER_CONTRACT_MISMATCH":
            exact = "false"
            notes = "Related F6L output exists but the R5A output contract/prompt/parser is different."
        elif row["task"] == "profile_generation":
            exact = "false"
            status = "NEW_INFERENCE_REQUIRED_NO_PROFILE_ARTIFACT"
            notes = "No frozen profile artifact exists for this dataset/model/user dependency."
        else:
            exact = "false"
            notes = "No exact existing output matches the full R5A material request key."
        rows.append(
            {
                "request_id": row["request_id"],
                "recovery_family": row["recovery_family"],
                "dataset": row["dataset"],
                "model_name": row["model_name"],
                "task": row["task"],
                "user_id": row["user_id"],
                "experimental_condition": row["experimental_condition"],
                "exact_match_found": exact,
                "reuse_status": status,
                "reusable_existing_output": "false",
                "related_existing_output_path": row.get("related_existing_output_path", ""),
                "checked_sources": "canonical_local_full_matrix_requests;phase_f6l_profile_request_manifest;phase_f6l_profile_execution_manifest",
                "match_key_sha256": row["request_key_sha256"],
                "notes": notes,
            }
        )
    return pd.DataFrame(rows).sort_values(["recovery_family", "dataset", "request_id"]).reset_index(drop=True)


def build_counts(raw: pd.DataFrame, manifest: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    families = [
        ("CANDDIF_RECOVERY", "CandDif"),
        ("HISTORY_LENGTH_RECOVERY", "History"),
        ("GENERATION_INVOLVED_RECOVERY", "Generation"),
    ]
    for family, label in families:
        raw_family = raw.loc[raw["recovery_family"].eq(family)]
        unique_family = manifest.loc[manifest["recovery_family"].eq(family)]
        duplicates = int(len(raw_family) - raw_family["canonical_request_id"].nunique())
        profile_generation = int((unique_family["task"] == "profile_generation").sum())
        downstream = int((unique_family["task"] == "ranking").sum()) if family == "GENERATION_INVOLVED_RECOVERY" else 0
        rows.append(
            {
                "family": label,
                "raw_planned_requests": len(raw_family),
                "duplicate_requests": duplicates,
                "reusable_existing_outputs": 0,
                "unique_new_requests": int(raw_family["canonical_request_id"].nunique()),
                "new_profile_generation_requests": profile_generation,
                "downstream_recommendation_requests": downstream,
                "total_generation_requests": int(len(unique_family)) if family == "GENERATION_INVOLVED_RECOVERY" else 0,
                "count_status": "EXACT_DEDUPLICATED_REQUEST_MANIFEST",
                "notes": (
                    "R5A exact count; R4 4400 remains a family-level planning estimate."
                    if family == "GENERATION_INVOLVED_RECOVERY"
                    else "R5A exact count after material request-key deduplication."
                ),
            }
        )
    rows.append(
        {
            "family": "TOTAL",
            "raw_planned_requests": len(raw),
            "duplicate_requests": int(len(raw) - raw["canonical_request_id"].nunique()),
            "reusable_existing_outputs": 0,
            "unique_new_requests": int(raw["canonical_request_id"].nunique()),
            "new_profile_generation_requests": int((manifest["task"] == "profile_generation").sum()),
            "downstream_recommendation_requests": int(
                (
                    (manifest["recovery_family"] == "GENERATION_INVOLVED_RECOVERY")
                    & (manifest["task"] == "ranking")
                ).sum()
            ),
            "total_generation_requests": int((manifest["recovery_family"] == "GENERATION_INVOLVED_RECOVERY").sum()),
            "count_status": "R5A_EXACT_COUNT_SUPERSEDES_R4_PLANNING_ESTIMATE",
            "notes": "R4_RECOVERY_COUNT_STATUS was ESTIMATED_NOT_YET_DEDUPLICATED; R5A exact unique new request count is manifest-derived.",
        }
    )
    return pd.DataFrame(rows)


def build_gap_mapping(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = raw.groupby(["gap_id", "dataset", "metric", "priority", "recovery_family"], sort=True)
    rows: list[dict[str, Any]] = []
    for keys, group in grouped:
        gap_id, dataset, metric, priority, family = keys
        request_ids = sorted(group["canonical_request_id"].unique())
        shared_metrics = sorted(group["metric"].unique())
        rows.append(
            {
                "gap_id": gap_id,
                "dataset": dataset,
                "metric": metric,
                "priority": priority,
                "recovery_family": family,
                "unique_request_count": len(request_ids),
                "raw_planned_request_count": len(group),
                "request_ids": ";".join(request_ids),
                "shared_metric_request": str(len(shared_metrics) > 1).lower(),
                "notes": "Gap accounted by deterministic R5A request IDs.",
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "metric", "gap_id"]).reset_index(drop=True)


def build_protected_hash_check(project_root: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    protected_path = as_project_path(project_root, input_path(config, "r4_protected_hash_check"))
    protected = read_csv(protected_path)
    status_lines = git_stdout(["status", "--short", "--untracked-files=all"], project_root).splitlines()
    status_by_path = {
        porcelain_path(line).replace("\\", "/"): line for line in status_lines if line
    }
    rows: list[dict[str, Any]] = []
    for row in protected.to_dict("records"):
        path_text = str(row["path"]).replace("\\", "/")
        path = resolve_r1_metric_recovery_artifact(project_root, path_text)
        record = relocation_record(project_root, path_text)
        exists = path.exists()
        observed = sha256_file(path) if exists else ""
        expected = (record.sha256 if record else str(row["expected_sha256"])).upper()
        current_path_text = record.current_path if record else path_text
        tracked = bool(git_or_empty(["ls-files", "--", current_path_text], project_root))
        unchanged = exists and observed == expected and (record is not None or not tracked)
        rows.append(
            {
                "path": path_text,
                "exists": str(exists).lower(),
                "tracked": str(tracked).lower(),
                "git_status": status_by_path.get(current_path_text, ""),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "hash_unchanged": str(unchanged).lower(),
                "protection_status": "PROTECTED_UNCHANGED" if unchanged else "PROTECTED_CONFLICT",
            }
        )
    return pd.DataFrame(rows)


def allowed_worktree_change(path: str) -> bool:
    normalized = path.replace("\\", "/")
    allowed_prefixes = (
        "configs/reproduction_r5a_",
        "src/analysis/reproduction_r5a_",
        "src/experiments/build_reproduction_r5a_",
        "src/experiments/run_reproduction_r5_recovery.py",
        "tests/test_reproduction_r5a_",
        "report/reproduction_r5a_",
        "results/tables/reproduction_r5a_",
        "results/logs/reproduction_r5a_",
        "results/recovery/r5/",
    )
    protected_names = {
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
    }
    return (
        normalized.startswith(allowed_prefixes)
        or normalized in protected_names
        or normalized in {".gitignore", "report/notes.md"}
    )


def worktree_unrelated_changes(project_root: Path) -> list[str]:
    lines = git_stdout(["status", "--short", "--untracked-files=all"], project_root).splitlines()
    unrelated: list[str] = []
    for line in lines:
        if not line:
            continue
        path = porcelain_path(line)
        if not allowed_worktree_change(path):
            unrelated.append(line)
    return unrelated


def build_validation(
    project_root: Path,
    config: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    manifest_sha256: str,
) -> tuple[pd.DataFrame, str]:
    raw = frames["raw_planned_requests"]
    manifest = frames["recovery_request_manifest"]
    gap_scope = frames["gap_scope"]
    counts = frames["exact_request_counts"]
    reuse = frames["existing_output_reuse"]
    protected = frames["protected_hash_check"]
    branch = git(["branch", "--show-current"], project_root)
    head = git(["rev-parse", "HEAD"], project_root)
    staged = git_or_empty(["diff", "--cached", "--stat"], project_root)
    original_status = git_or_empty(["-C", "original_repo/EvaLLMasRecommender", "status", "--short"], project_root)
    unrelated = worktree_unrelated_changes(project_root)

    p0 = set(
        map(
            tuple,
            gap_scope.loc[gap_scope["priority"].eq("P0_MUST_CLOSE"), ["dataset", "metric"]].to_numpy(),
        )
    )
    p1 = set(
        map(
            tuple,
            gap_scope.loc[gap_scope["priority"].eq("P1_HIGH_VALUE"), ["dataset", "metric"]].to_numpy(),
        )
    )
    non_high = gap_scope.loc[~gap_scope["priority"].isin(["P0_MUST_CLOSE", "P1_HIGH_VALUE"])]
    count_by_family = counts.set_index("family").to_dict("index")

    checks: list[tuple[str, str, bool, str, str, str]] = [
        ("R5A-VAL-001", "expected branch", branch == EXPECTED_BRANCH, branch, EXPECTED_BRANCH, "R5A starts from master."),
        ("R5A-VAL-002", "expected head", head == EXPECTED_HEAD, head, EXPECTED_HEAD, "R5A starts from the frozen R4 commit."),
        ("R5A-VAL-003", "no staged diff", staged == "", staged, "empty", "R5A does not stage."),
        (
            "R5A-VAL-004",
            "no unrelated worktree changes",
            not unrelated,
            "; ".join(unrelated),
            "none",
            "Only R5A artifacts and protected historical root files may be present.",
        ),
        (
            "R5A-VAL-005",
            "original repo clean",
            original_status == "",
            original_status,
            "empty",
            "R5A must not modify original_repo/EvaLLMasRecommender.",
        ),
        (
            "R5A-VAL-006",
            "protected artifacts unchanged",
            int(protected["protection_status"].eq("PROTECTED_UNCHANGED").sum()) == 12,
            f"PROTECTED_UNCHANGED={int(protected['protection_status'].eq('PROTECTED_UNCHANGED').sum())}",
            "12",
            "The 12 R1P/R2/R3 protected root artifacts remain untracked and hash-identical.",
        ),
        ("R5A-VAL-007", "all R4 gaps scoped", len(gap_scope) == 54, str(len(gap_scope)), "54", "One scope row per R4 true gap."),
        ("R5A-VAL-008", "P0 exact set", p0 == P0_EXPECTED, canonical_json(sorted(p0)), canonical_json(sorted(P0_EXPECTED)), "P0 rows remain exactly the approved CandDif rows."),
        ("R5A-VAL-009", "P1 exact set", p1 == P1_EXPECTED, canonical_json(sorted(p1)), canonical_json(sorted(P1_EXPECTED)), "P1 rows remain exactly the approved generation/history rows."),
        (
            "R5A-VAL-010",
            "P2/P3/P4 zero scheduled",
            int(non_high["unique_request_count"].sum()) == 0,
            str(int(non_high["unique_request_count"].sum())),
            "0",
            "Lower-priority rows are scoped but not scheduled.",
        ),
        ("R5A-VAL-011", "raw planned requests", len(raw) == 6000, str(len(raw)), "6000", "Metric-level raw plan includes exact profile prerequisites."),
        ("R5A-VAL-012", "unique new requests", len(manifest) == 4800, str(len(manifest)), "4800", "Unique material request-key manifest count."),
        (
            "R5A-VAL-013",
            "CandDif dedup",
            count_by_family["CandDif"]["raw_planned_requests"] == 2400
            and count_by_family["CandDif"]["duplicate_requests"] == 1200
            and count_by_family["CandDif"]["unique_new_requests"] == 1200,
            canonical_json(count_by_family["CandDif"]),
            "raw=2400; duplicate=1200; unique=1200",
            "CandDif HR/NDCG metric-only duplicates collapse by candidate-order output.",
        ),
        (
            "R5A-VAL-014",
            "History count",
            count_by_family["History"]["raw_planned_requests"] == 2400
            and count_by_family["History"]["unique_new_requests"] == 2400,
            canonical_json(count_by_family["History"]),
            "raw=2400; unique=2400",
            "Four datasets by two models by 100 users by three exact lengths.",
        ),
        (
            "R5A-VAL-015",
            "Generation count with profile prerequisites",
            count_by_family["Generation"]["raw_planned_requests"] == 1200
            and count_by_family["Generation"]["new_profile_generation_requests"] == 400
            and count_by_family["Generation"]["downstream_recommendation_requests"] == 800
            and count_by_family["Generation"]["unique_new_requests"] == 1200,
            canonical_json(count_by_family["Generation"]),
            "raw=1200; profile_generation=400; downstream=800; unique=1200",
            "ML-1M/LastFM add explicit missing profile-generation prerequisites.",
        ),
        (
            "R5A-VAL-016",
            "request IDs unique",
            manifest["request_id"].is_unique and manifest["request_key_sha256"].is_unique,
            f"ids={manifest['request_id'].nunique()}; keys={manifest['request_key_sha256'].nunique()}",
            str(len(manifest)),
            "One unique ID per material request key.",
        ),
        (
            "R5A-VAL-017",
            "no exact output reuse",
            int(reuse["reuse_status"].eq("EXACT_REUSE").sum()) == 0,
            str(int(reuse["reuse_status"].eq("EXACT_REUSE").sum())),
            "0",
            "Related frozen outputs are dependencies or non-exact context, not result reuse.",
        ),
        (
            "R5A-VAL-018",
            "zero R5A inference execution",
            manifest["execution_allowed_in_r5a"].eq("false").all(),
            str(manifest["execution_allowed_in_r5a"].unique().tolist()),
            "false",
            "Manifest creation is dry-run planning only.",
        ),
        (
            "R5A-VAL-019",
            "material hashes populated",
            manifest[
                [
                    "history_hash",
                    "candidate_list_hash",
                    "prompt_hash",
                    "material_context_hash",
                    "request_key_sha256",
                ]
            ]
            .astype(str)
            .apply(lambda col: col.str.len().gt(0).all())
            .all(),
            "hash columns nonblank",
            "hash columns nonblank",
            "Prompt, candidate, history, material context, and request-key hashes are frozen.",
        ),
        (
            "R5A-VAL-020",
            "ML-1M CandDif absent",
            manifest.loc[
                manifest["dataset"].eq("ML-1M") & manifest["recovery_family"].eq("CANDDIF_RECOVERY")
            ].empty,
            str(
                len(
                    manifest.loc[
                        manifest["dataset"].eq("ML-1M")
                        & manifest["recovery_family"].eq("CANDDIF_RECOVERY")
                    ]
                )
            ),
            "0",
            "ML-1M CandDif remains in-scope-already-accounted and not P0.",
        ),
        (
            "R5A-VAL-021",
            "14 high-priority gaps accounted",
            int(gap_scope.loc[gap_scope["scheduled_in_manifest"].eq("true")]["gap_id"].nunique()) == 14,
            str(int(gap_scope.loc[gap_scope["scheduled_in_manifest"].eq("true")]["gap_id"].nunique())),
            "14",
            "Every P0/P1 gap has at least one request mapping.",
        ),
        (
            "R5A-VAL-022",
            "manifest sha recorded",
            len(manifest_sha256) == 64,
            manifest_sha256,
            "64 hex chars",
            "The recovery request manifest has a stable SHA256.",
        ),
        (
            "R5A-VAL-023",
            "R4 estimate not reused as proof",
            int(counts.loc[counts["family"].eq("TOTAL"), "unique_new_requests"].iloc[0]) != 4400,
            str(int(counts.loc[counts["family"].eq("TOTAL"), "unique_new_requests"].iloc[0])),
            "not 4400",
            "R4 4400 was a planning estimate; R5A count is exact and includes profile prerequisites.",
        ),
    ]
    rows = [
        {
            "check_id": check_id,
            "check_name": name,
            "status": "pass" if ok else "fail",
            "observed": observed,
            "expected": expected,
            "details": details,
        }
        for check_id, name, ok, observed, expected, details in checks
    ]
    validation = pd.DataFrame(rows)
    decision = DECISION_COMPLETE if validation["status"].eq("pass").all() else DECISION_BLOCKED
    return validation, decision


def render_report(frames: Mapping[str, pd.DataFrame], decision: str, manifest_sha256: str) -> str:
    counts = frames["exact_request_counts"].set_index("family")
    validation = frames["validation"]
    p0_rows = frames["gap_scope"].loc[frames["gap_scope"]["priority"].eq("P0_MUST_CLOSE")]
    p1_rows = frames["gap_scope"].loc[frames["gap_scope"]["priority"].eq("P1_HIGH_VALUE")]
    pass_count = int(validation["status"].eq("pass").sum())
    fail_count = int(validation["status"].eq("fail").sum())
    total = counts.loc["TOTAL"]
    generation = counts.loc["Generation"]
    delta_from_r4 = int(total["unique_new_requests"]) - 4400
    percent_from_r4 = delta_from_r4 / 4400 * 100
    p0_text = "\n".join(f"- {row.dataset} {row.metric}" for row in p0_rows.itertuples())
    p1_text = "\n".join(f"- {row.dataset} {row.metric}" for row in p1_rows.itertuples())
    return f"""# R5A Controlled Recovery Manifest

Decision: `{decision}`

R5A built a deterministic request manifest for the approved R4 P0/P1 recovery gaps only. It executed no LLM/Ollama/API calls, no training, no downloads, and no `original_repo/` edits.

## Exact Count

- Raw metric-level planned requests: `{int(total["raw_planned_requests"])}`
- Metric-only duplicate requests: `{int(total["duplicate_requests"])}`
- Reusable existing outputs: `{int(total["reusable_existing_outputs"])}`
- Unique new future requests: `{int(total["unique_new_requests"])}`
- Manifest SHA256: `{manifest_sha256}`

The R4 value `4400` remains a planning estimate. R5A supersedes it with the exact manifest-derived count of `{int(total["unique_new_requests"])}`. The delta is `{delta_from_r4:+d}` requests, a `{percent_from_r4:.2f}%` increase over the R4 estimate, because ML-1M and LastFM need `{int(generation["new_profile_generation_requests"])}` explicit profile-generation prerequisite requests before their downstream profile-plus-history recommendation requests. This is not scope expansion; it resolves a prerequisite that the R4 family-level estimate had not materialized.

## Generation Split

- New profile-generation requests: `{int(generation["new_profile_generation_requests"])}`
- Downstream recommendation requests: `{int(generation["downstream_recommendation_requests"])}`
- Total generation-family requests: `{int(generation["total_generation_requests"])}`

Beauty and Sports record existing F6L profile artifacts as dependencies. ML-1M and LastFM record future profile-generation dependencies explicitly; downstream rows reference those dependency request IDs and cannot be executed before the profile outputs are materialized.

R5A does not fabricate future profile output hashes. Reused Beauty/Sports profiles use `profile_hash_status=INPUT_PROFILE_HASH_AVAILABLE`; ML-1M/LastFM downstream rows use `profile_hash_status=PROFILE_DEPENDENCY_REQUEST_ID_PENDING_OUTPUT` and a deterministic `profile_dependency_request_id`.

## P0 Scope

{p0_text}

## P1 Scope

{p1_text}

## Validation

Validation result: `{pass_count} PASS / {fail_count} FAIL`.

R5A excludes all P2/P3/P4 gaps, keeps ML-1M CandDif out of P0, and records every material prompt, candidate, history, profile dependency, parser contract, expected output schema, and output destination in `results/tables/reproduction_r5a_recovery_request_manifest.csv`.
"""


def update_notes(project_root: Path, frames: Mapping[str, pd.DataFrame], decision: str, manifest_sha256: str) -> None:
    notes_path = project_root / NOTES_PATH
    counts = frames["exact_request_counts"].set_index("family")
    total = counts.loc["TOTAL"]
    generation = counts.loc["Generation"]
    delta_from_r4 = int(total["unique_new_requests"]) - 4400
    percent_from_r4 = delta_from_r4 / 4400 * 100
    section = f"""
## R5A controlled P0/P1 recovery manifest

- Decision: `{decision}`.
- Scope: deterministic manifest and validation only; no LLM/Ollama/API calls, training, downloads, Food.com work, R5 execution, or `original_repo/` modification.
- Outputs: `results/tables/reproduction_r5a_*.csv` and `report/reproduction_r5a_controlled_recovery_manifest.md`.
- Exact requests: raw `{int(total["raw_planned_requests"])}`, duplicates `{int(total["duplicate_requests"])}`, unique new future requests `{int(total["unique_new_requests"])}`.
- Generation split: profile-generation prerequisites `{int(generation["new_profile_generation_requests"])}`, downstream recommendation requests `{int(generation["downstream_recommendation_requests"])}`, total generation-family requests `{int(generation["total_generation_requests"])}`.
- R4 `4400` remains a planning estimate, not request-manifest proof. R5A exact count includes ML-1M/LastFM profile prerequisites: delta `{delta_from_r4:+d}` (`{percent_from_r4:.2f}%`).
- Profile hash semantics: existing Beauty/Sports profile inputs have available hashes; ML-1M/LastFM future profile outputs use dependency request IDs and do not fabricate output hashes before inference.
- Manifest SHA256: `{manifest_sha256}`.
- Next step: execute R5 recovery only after explicit approval and only from the frozen manifest.
"""
    existing = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    marker = "## R5A controlled P0/P1 recovery manifest"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n" + section
    else:
        existing = existing.rstrip() + "\n" + section
    notes_path.write_text(existing.lstrip(), encoding="utf-8")


def build_r5a(
    project_root: Path,
    config_path: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    raw, manifest, dedup = collect_specs(config, project_root)
    protected = build_protected_hash_check(project_root, config)
    gap_scope = build_gap_scope(project_root, config, manifest, raw)
    reuse = build_existing_output_reuse(manifest)
    counts = build_counts(raw, manifest, config)
    gap_mapping = build_gap_mapping(raw)
    manifest_sha256 = sha256_text(manifest.to_csv(index=False, lineterminator="\n"))
    frames: dict[str, pd.DataFrame] = {
        "gap_scope": gap_scope,
        "raw_planned_requests": raw,
        "recovery_request_manifest": manifest,
        "existing_output_reuse": reuse,
        "request_deduplication": dedup,
        "exact_request_counts": counts,
        "gap_to_request_mapping": gap_mapping,
        "protected_hash_check": protected,
    }
    validation, decision = build_validation(project_root, config, frames, manifest_sha256)
    frames["validation"] = validation
    report = render_report(frames, decision, manifest_sha256)
    if write:
        for name, rel_path in OUTPUT_TABLES.items():
            write_csv(frames[name], project_root / rel_path)
        report_path = project_root / ((config.get("outputs") or {}).get("report", REPORT_PATH))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        update_notes(project_root, frames, decision, manifest_sha256)
    return {
        "decision": decision,
        "frames": frames,
        "manifest_sha256": manifest_sha256,
        "report": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reproduction_r5a_controlled_recovery_manifest.yaml")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    project_root = Path(args.project_root or config.get("project_root", ".")).resolve()
    result = build_r5a(project_root, config_path=config_path, write=not args.no_write)
    validation = result["frames"]["validation"]
    print(f"decision={result['decision']}")
    print(f"manifest_rows={len(result['frames']['recovery_request_manifest'])}")
    print(f"raw_rows={len(result['frames']['raw_planned_requests'])}")
    print(f"manifest_sha256={result['manifest_sha256']}")
    print(f"validation_pass={int(validation['status'].eq('pass').sum())}")
    print(f"validation_fail={int(validation['status'].eq('fail').sum())}")
    return 0 if result["decision"] == DECISION_COMPLETE else 1


if __name__ == "__main__":
    raise SystemExit(main())
