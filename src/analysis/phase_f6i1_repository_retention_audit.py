"""Phase F6I1 repository retention audit and safe cleanup batch 1.

This module reconstructs a conservative file inventory, dependency edge table,
retention classification, duplicate map, and delete-candidate manifest. Cleanup
is intentionally limited to Python bytecode cache files outside protected
runtime/data/paper/original-repository areas, and only when a retained source
file provides a regeneration path.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import logging
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - project requirements include PyYAML
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
PHASE_ID = "phase_f6i1"
PHASE_NAME = "phase_f6i1_repository_retention_audit_and_safe_cleanup_batch1"
RUNTIME_CLASSIFICATION = "defensible_repaired_clone_exact_package_versions"
RECOVERED_INTERPRETER = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = PROJECT_ROOT / "results" / "manifests"
REPORT_PATH = PROJECT_ROOT / "report" / f"{PHASE_NAME}.md"
INITIAL_CAPTURE_PATH = MANIFESTS_DIR / f"{PHASE_ID}_initial_capture.json"
CLEANUP_EXECUTION_PATH = TABLES_DIR / f"{PHASE_ID}_cleanup_execution.csv"
VALIDATION_PATH = TABLES_DIR / f"{PHASE_ID}_validation.csv"

OUTPUT_PATHS = {
    "inventory": TABLES_DIR / f"{PHASE_ID}_relevant_file_inventory.csv",
    "edges": TABLES_DIR / f"{PHASE_ID}_dependency_edges.csv",
    "classification": TABLES_DIR / f"{PHASE_ID}_retention_classification.csv",
    "delete_candidates": TABLES_DIR / f"{PHASE_ID}_delete_now_candidates.csv",
    "manual_review": TABLES_DIR / f"{PHASE_ID}_manual_review.csv",
    "future_dependency": TABLES_DIR / f"{PHASE_ID}_future_dependency_map.csv",
    "duplicates": TABLES_DIR / f"{PHASE_ID}_duplicate_groups.csv",
}

RETENTION_CLASSES = {
    "keep_protected",
    "keep_canonical_runtime",
    "keep_canonical_builder",
    "keep_canonical_test",
    "keep_current_documentation",
    "keep_future_dependency",
    "keep_provenance_only",
    "merge_later",
    "archive_later",
    "delete_now_exact_duplicate",
    "delete_now_regenerable",
    "delete_now_obsolete_safe",
    "manual_review",
}

INVENTORY_COLUMNS = [
    "path",
    "size_bytes",
    "sha256",
    "tracked_state",
    "phase",
    "purpose",
    "producer",
    "inputs",
    "outputs",
    "importers",
    "callers",
    "tests",
    "reports_and_manifests_referencing",
    "unique_scientific_information",
    "canonical_replacement",
    "future_roadmap_dependencies",
    "category",
    "extension",
    "inspection_status",
]

EDGE_COLUMNS = [
    "edge_id",
    "source_path",
    "target_path",
    "edge_type",
    "dag_layer",
    "evidence",
    "source_phase",
    "target_phase",
]

CLASSIFICATION_COLUMNS = [
    "path",
    "class",
    "reason",
    "deletion_gate_status",
    "canonical_replacement",
    "dependency_evidence",
    "unique_information_check",
    "future_roadmap_check",
    "validation_command",
    "rollback_source",
]

DELETE_COLUMNS = [
    "path",
    "sha256",
    "size_bytes",
    "class",
    "reason",
    "canonical_replacement",
    "dependency_evidence",
    "unique_information_check",
    "future_roadmap_check",
    "validation_command",
    "rollback_source",
]

MANUAL_REVIEW_COLUMNS = [
    "path",
    "reason",
    "risk",
    "blocking_gate",
    "recommended_next_action",
    "dependency_evidence",
]

FUTURE_COLUMNS = [
    "path",
    "roadmap_dependency",
    "evidence",
    "retention_effect",
]

DUPLICATE_COLUMNS = [
    "duplicate_group_id",
    "sha256",
    "size_bytes",
    "member_path",
    "canonical_path",
    "canonical_selection_reason",
    "delete_eligible",
    "gate_status",
]

CLEANUP_COLUMNS = [
    "path",
    "pre_sha256",
    "pre_size_bytes",
    "action",
    "status",
    "post_exists",
    "validation_evidence",
]

VALIDATION_COLUMNS = [
    "validation_id",
    "check_name",
    "expected",
    "observed",
    "status",
    "severity",
    "evidence",
]

TEXT_EXTENSIONS = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
REFERENCE_EXTENSIONS = {
    ".csv",
    ".docx",
    ".gz",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".npy",
    ".npz",
    ".pdf",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
    ".zip",
}

ENV_UNIT_DIRS = {
    ".venv",
    ".venv_audit_recovery",
    ".venv_f6g2_py312",
    ".venv_repro",
    ".venv_repro_system_site",
    "results/environment/cpython-3.12.10-amd64",
    "results/environment/phase_f6f1r_isolated_f6f1",
    "results/logs/phase_f6g2r_pytest_tmp",
    "results/logs/phase_f6g2r_manual_pytest_tmp",
    ".pytest_cache",
}

SKIP_DIR_NAMES = {".git", ".mypy_cache", ".ruff_cache"}
PROTECTED_PREFIXES = (
    "original_repo/",
    "paper/",
    "data/raw/",
    "results/llm/",
    "results/predictions/",
    "results/checkpoints/",
    "results/models/",
)
PROTECTED_ROOT_FILES = {
    "AGENTS.md",
    "README.md",
    "PROJECT_SPEC.md",
    "CODEX_PROMPTS.md",
    "requirements.txt",
    "environment.yml",
    ".gitignore",
    "pytest.ini",
}
CURRENT_CONTEXT_PATHS = {
    "report/future_operations_context_handoff.md",
    "results/manifests/current_project_context_snapshot.json",
    "results/tables/workspace_relevant_file_inventory.csv",
    "results/tables/workspace_phase_and_supersession_registry.csv",
}

FUTURE_ROADMAP_TERMS = {
    "four original datasets": ["beauty", "sports", "ml1m", "movie", "lastfm", "amazon"],
    "ranking and reranking": ["ranking", "reranking", "re-ranking"],
    "traditional baselines": ["random", "mostpop", "most_pop", "bm25", "bprmf", "gru4rec", "sasrec", "lightgcn"],
    "local open LLMs": ["qwen2", "mistral", "llama", "70b"],
    "possible GPT and Claude evaluation": ["gpt", "claude", "api", "openai", "anthropic"],
    "base/OpenP5 prompt": ["openp5", "base prompt", "base/openp5"],
    "LLMRank": ["llmrank"],
    "LLMSRec": ["llmsrec", "llmsrec-style"],
    "top-20 candidate generation": ["top-20", "20 candidates", "candidate_pool_20", "candidate_position"],
    "repository-derived top-10 output": ["top-10", "top10", "output length", "repository_derived"],
    "history length and zero-history": ["history_length", "zero-history", "zero_history", "history length"],
    "reduced traditional training data D'": ["d'", "reduced traditional", "trainable", "training data"],
    "candidate-position bias": ["candidate position", "position_bias", "candidate_position"],
    "profile generation": ["profile generation", "profile"],
    "hallucination": ["hallucination"],
    "utility": ["hr@", "ndcg", "utility"],
    "novelty": ["novelty", "long_tail", "aplt"],
    "serendipity": ["serendipity"],
    "fairness": ["fairness"],
    "diversity": ["diversity", "ild"],
    "nondeterminism": ["nondeterminism", "determinism", "seed"],
    "fifth dataset": ["fifth dataset", "additional dataset"],
    "prompt-format and quantization extensions": ["prompt-format", "prompt format", "quantization"],
    "final report and reproducibility package": ["final report", "reproducibility package", "artifact catalog"],
}

UNIQUE_INFO_TERMS = {
    "prompt": ["prompt", "template", "openp5", "llmrank", "llmsrec"],
    "parser": ["parser", "parse", "json parser"],
    "metric": ["metric", "hr@", "ndcg", "coverage", "hallucination", "novelty", "serendipity", "fairness", "diversity"],
    "schema": ["schema", "columns", "dataframe", "contract"],
    "seed": ["seed"],
    "hyperparameter": ["epochs", "learning_rate", "batch_size", "hidden", "embedding", "dropout"],
    "cohort": ["cohort"],
    "candidate policy": ["candidate", "top-20", "top20", "candidate_position"],
    "model version": ["qwen2", "mistral", "llama", "gpt", "claude", "model_name"],
    "scientific decision": ["decision", "policy", "claim", "blocked", "evidence", "exactness"],
}

PATH_TOKEN_RE = re.compile(
    r"(?P<path>(?:\.{0,2}[\\/])?(?:[A-Za-z0-9_. -]+[\\/])+[A-Za-z0-9_. -]+"
    r"\.(?:csv|docx|gz|ini|json|jsonl|md|npy|npz|pdf|pkl|png|pt|pth|py|txt|ya?ml|zip))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FileRecord:
    path: str
    absolute_path: Path
    size_bytes: int
    sha256: str
    tracked_state: str
    category: str
    extension: str
    is_directory_unit: bool
    inspection_status: str
    text_sample: str = ""


@dataclass(frozen=True)
class PhaseTables:
    inventory: list[dict[str, str]]
    edges: list[dict[str, str]]
    classification: list[dict[str, str]]
    delete_candidates: list[dict[str, str]]
    manual_review: list[dict[str, str]]
    future_dependency: list[dict[str, str]]
    duplicates: list[dict[str, str]]
    initial_capture: dict[str, Any]


def normalize_path(path: str | Path) -> str:
    """Return a stable repository-relative path with forward slashes."""
    value = str(path).replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", value):
        try:
            value = Path(value).resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return value
    while value.startswith("./"):
        value = value[2:]
    return value.rstrip("/") if value != "." else "."


def is_under(path: str, prefix: str) -> bool:
    normalized = normalize_path(path)
    prefix = normalize_path(prefix)
    return normalized == prefix or normalized.startswith(prefix + "/")


def run_command(args: Sequence[str], timeout: int = 60) -> tuple[int, str, str]:
    completed = subprocess.run(
        list(args),
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def git_lines(args: Sequence[str]) -> list[str]:
    code, stdout, stderr = run_command(["git", *args])
    if code != 0:
        logging.warning("git %s failed: %s", " ".join(args), stderr)
        return []
    return [line for line in stdout.splitlines() if line]


def git_state_maps() -> dict[str, set[str]]:
    tracked = {normalize_path(path) for path in git_lines(["ls-files"])}
    modified = {normalize_path(path) for path in git_lines(["diff", "--name-only"])}
    staged = {normalize_path(path) for path in git_lines(["diff", "--cached", "--name-only"])}
    code, stdout, stderr = run_command(["git", "status", "--porcelain=v1", "--untracked-files=normal"], timeout=60)
    if code != 0:
        logging.warning("git untracked status failed: %s", stderr)
        untracked = set()
        untracked_dirs = set()
    else:
        untracked = set()
        untracked_dirs = set()
        for line in stdout.splitlines():
            if not line.startswith("?? "):
                continue
            path = normalize_path(line[3:].strip().strip('"'))
            if path.endswith("/"):
                untracked_dirs.add(path.rstrip("/"))
            else:
                untracked.add(path)
    return {
        "tracked": tracked,
        "modified": modified,
        "staged": staged,
        "untracked": untracked,
        "untracked_dirs": untracked_dirs,
        "ignored": set(),
    }


def git_ignored_count() -> int:
    code, stdout, stderr = run_command(["git", "status", "--porcelain=v1", "--ignored", "--untracked-files=normal"], timeout=60)
    if code != 0:
        logging.warning("git ignored status failed: %s", stderr)
        return 0
    return sum(1 for line in stdout.splitlines() if line.startswith("!!"))


def tracked_state(path: str, states: Mapping[str, set[str]]) -> str:
    normalized = normalize_path(path)
    if normalized in states.get("staged", set()):
        return "staged"
    if normalized in states.get("modified", set()):
        return "tracked_modified"
    if normalized in states.get("tracked", set()):
        return "tracked_clean"
    if normalized in states.get("untracked", set()):
        return "untracked"
    if any(is_under(normalized, directory) for directory in states.get("untracked_dirs", set())):
        return "untracked"
    if normalized in states.get("ignored", set()):
        return "ignored"
    return "untracked_or_nested"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def unit_digest(path: str, marker: str) -> str:
    return hashlib.sha256(f"{PHASE_ID}:{path}:{marker}".encode("utf-8")).hexdigest().upper()


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = sorted(
        [dict(row) for row in rows],
        key=lambda row: tuple(str(row.get(column, "")) for column in columns),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in materialized:
            writer.writerow({column: row.get(column, "") for column in columns})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def split_semicolon(value: str) -> list[str]:
    return [normalize_path(part.strip()) for part in value.split(";") if part.strip()]


def category_for(path: str) -> str:
    normalized = normalize_path(path)
    if normalized.endswith("/"):
        return "directory_unit"
    if normalized.startswith("original_repo/"):
        return "official_repository"
    if normalized.startswith("paper/"):
        return "paper"
    if normalized.startswith("data/raw/"):
        return "raw_data"
    if normalized.startswith("data/"):
        return "local_data"
    if normalized.startswith("docs/"):
        return "docs"
    if normalized.startswith("report/"):
        return "report"
    if normalized.startswith("results/llm/"):
        return "raw_llm_output"
    if normalized.startswith("results/checkpoints/"):
        return "model_checkpoint"
    if normalized.startswith("results/predictions/"):
        return "prediction"
    if normalized.startswith("results/models/"):
        return "model_artifact"
    if normalized.startswith("results/manifests/"):
        return "manifest"
    if normalized.startswith("results/tables/"):
        return "result_table"
    if normalized.startswith("results/"):
        return "result_other"
    if normalized.startswith("src/"):
        return "source_code"
    if normalized.startswith("tests/"):
        return "test"
    if normalized.startswith("configs/"):
        return "config"
    return "root_or_other"


def phase_for(path: str, text_sample: str = "") -> str:
    haystack = f"{normalize_path(path).lower()} {text_sample[:2000].lower()}"
    patterns = [
        "phase_f6i1",
        "phase_f6h2",
        "phase_f6h1",
        "phase_f6g3",
        "phase_f6g2r",
        "phase_f6g2",
        "phase_f6g1",
        "phase_f6f2",
        "phase_f6f1r",
        "phase_f6f1",
        "phase_f6e4",
        "phase_f6e3",
        "phase_f6e2",
        "phase_f6e1",
        "finalization_f6d4a",
        "finalization_f6d3",
        "finalization_f6d2",
        "finalization_f6d1",
        "finalization_f6c",
        "finalization_f6b",
        "finalization_f6a",
        "finalization_f5ar",
        "finalization_f5af",
        "finalization_f5a",
        "finalization_f5b",
        "finalization_f5c",
        "finalization_f5d",
        "finalization_f5e",
        "finalization_f4",
        "finalization_f3",
        "finalization_f2",
        "plan_e",
        "plan_d_phase_6clean",
        "plan_d_phase_6c",
        "plan_d_phase_6b",
        "plan_d_phase_6a",
        "plan_d_phase_5e",
        "plan_d_phase_5d",
        "plan_d",
        "plan_c",
        "plan_b",
    ]
    for pattern in patterns:
        if pattern in haystack:
            return pattern
    if path.startswith("original_repo/"):
        return "original_repository"
    if path.startswith("paper/"):
        return "paper"
    if path.startswith("src/"):
        return "implementation"
    if path.startswith("tests/"):
        return "test_suite"
    return "unassigned"


def purpose_for(path: str, category: str, text_sample: str) -> str:
    normalized = normalize_path(path)
    lower = f"{normalized.lower()} {text_sample[:2000].lower()}"
    if is_cache_path(normalized):
        return "python bytecode cache generated by local test or builder execution"
    if category == "directory_unit":
        return "environment or inaccessible temporary directory retained as a complete unit"
    if category == "source_code":
        if "/experiments/" in normalized or "/analysis/" in normalized:
            return "offline builder, runner, audit, or analysis implementation"
        if "/evaluation/" in normalized or "/metrics/" in normalized:
            return "metric, parser, or evaluation contract implementation"
        if "/llm/" in normalized or "/prompts/" in normalized:
            return "LLM prompt, parser, runner, or model interaction implementation"
        if "/baselines/" in normalized:
            return "traditional recommender baseline implementation"
        return "project source implementation"
    if category == "test":
        return "automated regression test for retained implementation"
    if category == "config":
        return "dataset or experiment configuration"
    if category in {"raw_data", "paper", "official_repository", "raw_llm_output", "prediction", "model_checkpoint", "model_artifact"}:
        return "protected scientific input, output, model, or upstream reference"
    if category == "result_table":
        if "validation" in lower:
            return "validation table"
        if "manifest" in lower:
            return "manifest or provenance table"
        if "metric" in lower:
            return "metric result table"
        return "generated result or audit table"
    if category == "report":
        return "phase report, final report input, or provenance narrative"
    if category == "manifest":
        return "machine-readable provenance or context manifest"
    if category == "docs":
        return "current reproducibility or artifact documentation"
    return "repository support file"


def is_text_candidate(path: str, size: int) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in TEXT_EXTENSIONS and size <= 5_000_000


def text_sample(path: Path, relative: str, size: int) -> tuple[str, str]:
    if not is_text_candidate(relative, size):
        if Path(relative).suffix.lower() in TEXT_EXTENSIONS:
            try:
                with path.open("rb") as handle:
                    data = handle.read(250_000)
                return data.decode("utf-8", errors="replace"), "hash_and_text_prefix_large_file"
            except OSError as exc:
                return "", f"hash_only_text_read_failed:{exc.__class__.__name__}"
        return "", "hash_only_binary_or_large_non_text"
    try:
        return path.read_text(encoding="utf-8", errors="replace"), "hash_and_full_text"
    except OSError as exc:
        return "", f"hash_only_text_read_failed:{exc.__class__.__name__}"


def should_skip_unit(path: str) -> bool:
    normalized = normalize_path(path)
    return any(normalized == unit or normalized.startswith(unit + "/") for unit in ENV_UNIT_DIRS)


def discover_records(root: Path) -> list[FileRecord]:
    states = git_state_maps()
    records: list[FileRecord] = []
    seen_units: set[str] = set()

    def add_unit(relative: str, absolute: Path, marker: str) -> None:
        normalized = normalize_path(relative).rstrip("/") + "/"
        if normalized in seen_units:
            return
        seen_units.add(normalized)
        try:
            size = absolute.stat().st_size
        except OSError:
            size = 0
        records.append(
            FileRecord(
                path=normalized,
                absolute_path=absolute,
                size_bytes=size,
                sha256=unit_digest(normalized, marker),
                tracked_state=tracked_state(normalized.rstrip("/"), states),
                category="directory_unit",
                extension="directory",
                is_directory_unit=True,
                inspection_status=marker,
            )
        )

    for unit in sorted(ENV_UNIT_DIRS):
        absolute = root / unit
        if absolute.exists() and absolute.is_dir():
            add_unit(unit, absolute, "directory_unit_not_expanded")

    for current, dirs, files in os.walk(root, topdown=True, onerror=None):
        current_path = Path(current)
        try:
            relative_current = normalize_path(current_path.relative_to(root))
        except ValueError:
            continue
        if relative_current == ".":
            relative_current = ""
        kept_dirs: list[str] = []
        for dirname in sorted(dirs):
            child = current_path / dirname
            child_rel = normalize_path(child.relative_to(root))
            if dirname in SKIP_DIR_NAMES:
                continue
            if should_skip_unit(child_rel):
                continue
            kept_dirs.append(dirname)
        dirs[:] = kept_dirs

        for filename in sorted(files):
            path = current_path / filename
            relative = normalize_path(path.relative_to(root))
            if should_skip_unit(relative):
                continue
            try:
                size = path.stat().st_size
                digest = sha256_file(path)
            except OSError as exc:
                records.append(
                    FileRecord(
                        path=relative,
                        absolute_path=path,
                        size_bytes=0,
                        sha256="UNREADABLE",
                        tracked_state=tracked_state(relative, states),
                        category=category_for(relative),
                        extension=Path(relative).suffix.lower().lstrip(".") or "no_extension",
                        is_directory_unit=False,
                        inspection_status=f"unreadable:{exc.__class__.__name__}",
                    )
                )
                continue
            sample, inspection = text_sample(path, relative, size)
            records.append(
                FileRecord(
                    path=relative,
                    absolute_path=path,
                    size_bytes=size,
                    sha256=digest,
                    tracked_state=tracked_state(relative, states),
                    category=category_for(relative),
                    extension=Path(relative).suffix.lower().lstrip(".") or "no_extension",
                    is_directory_unit=False,
                    inspection_status=inspection,
                    text_sample=sample,
                )
            )
    return sorted(records, key=lambda record: record.path)


def build_module_map(records: Sequence[FileRecord]) -> dict[str, str]:
    modules: dict[str, str] = {}
    for record in records:
        if not record.path.endswith(".py"):
            continue
        path = Path(record.path)
        parts = list(path.with_suffix("").parts)
        if parts[-1] == "__init__":
            module = ".".join(parts[:-1])
        else:
            module = ".".join(parts)
        modules[module] = record.path
        if module.startswith("src."):
            modules[module[4:]] = record.path
    return modules


def resolve_module(module: str, module_map: Mapping[str, str]) -> str:
    if module in module_map:
        return module_map[module]
    parts = module.split(".")
    while len(parts) > 1:
        parts.pop()
        shortened = ".".join(parts)
        if shortened in module_map:
            return module_map[shortened]
    return ""


def parse_python_imports(record: FileRecord, module_map: Mapping[str, str]) -> list[tuple[str, str]]:
    if not record.path.endswith(".py") or not record.text_sample:
        return []
    try:
        tree = ast.parse(record.text_sample)
    except SyntaxError:
        return []
    imports: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = resolve_module(alias.name, module_map)
                if target and target != record.path:
                    imports.append((target, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            candidates = [module]
            candidates.extend(f"{module}.{alias.name}" for alias in node.names if module)
            for candidate in candidates:
                target = resolve_module(candidate, module_map)
                if target and target != record.path:
                    imports.append((target, f"from {module} import ..."))
                    break
    return sorted(set(imports))


def resolve_reference_token(token: str, source_path: str, known_paths: Mapping[str, str], basename_map: Mapping[str, list[str]]) -> str:
    cleaned = token.strip().strip("\"'`.,;:)(").replace("\\", "/")
    if not cleaned:
        return ""
    if re.match(r"^[A-Za-z]:/", cleaned):
        try:
            resolved = normalize_path(Path(cleaned).resolve())
        except OSError:
            return ""
        return resolved if resolved in known_paths else ""
    candidates = [normalize_path(cleaned)]
    if cleaned.startswith("../") or cleaned.startswith("./"):
        source_dir = Path(source_path).parent
        candidates.append(normalize_path(source_dir / cleaned))
    for candidate in candidates:
        if candidate in known_paths:
            return candidate
    basename = Path(cleaned).name
    matches = basename_map.get(basename, [])
    if len(matches) == 1:
        return matches[0]
    return ""


def parse_path_references(record: FileRecord, known_paths: Mapping[str, str], basename_map: Mapping[str, list[str]]) -> list[tuple[str, str]]:
    if not record.text_sample or not should_scan_path_references(record):
        return []
    references = []
    for match in PATH_TOKEN_RE.finditer(record.text_sample):
        target = resolve_reference_token(match.group("path"), record.path, known_paths, basename_map)
        if target and target != record.path:
            references.append((target, match.group("path")))
    return sorted(set(references))


def should_scan_path_references(record: FileRecord) -> bool:
    """Limit free-text path scanning to dependency-bearing file families."""
    if record.category in {"source_code", "test", "config", "report", "docs", "manifest", "root_or_other"}:
        return True
    if record.category == "result_table":
        lower = record.path.lower()
        dependency_table_terms = [
            "manifest",
            "registry",
            "lineage",
            "provenance",
            "source_inventory",
            "artifact",
            "context_snapshot",
            "validation",
            "input_hash",
            "output_hash",
        ]
        return any(term in lower for term in dependency_table_terms) and len(record.text_sample) <= 500_000
    return False


def add_edge(
    edges: list[dict[str, str]],
    source: str,
    target: str,
    edge_type: str,
    evidence: str,
    records_by_path: Mapping[str, FileRecord],
) -> None:
    if not source or not target or source == target:
        return
    edge_key = f"{source}|{target}|{edge_type}|{evidence[:120]}"
    edge_id = "EDGE-F6I1-" + hashlib.sha1(edge_key.encode("utf-8")).hexdigest()[:12].upper()
    edges.append(
        {
            "edge_id": edge_id,
            "source_path": source,
            "target_path": target,
            "edge_type": edge_type,
            "dag_layer": dag_layer(source, target, edge_type),
            "evidence": evidence[:500],
            "source_phase": phase_for(source, records_by_path.get(source, FileRecord(source, PROJECT_ROOT / source, 0, "", "", "", "", False, "")).text_sample),
            "target_phase": phase_for(target, records_by_path.get(target, FileRecord(target, PROJECT_ROOT / target, 0, "", "", "", "", False, "")).text_sample),
        }
    )


def dag_layer(source: str, target: str, edge_type: str) -> str:
    if edge_type == "python_import":
        return "implementation_module -> importer_or_runner"
    if edge_type.startswith("phase_registry_source"):
        return "builder/runner -> registered_output_or_report"
    if edge_type.startswith("phase_registry_config"):
        return "config -> builder/runner"
    if edge_type.startswith("lineage_input"):
        return "source/config/input -> builder/runner/raw output/parser/metric/table/report"
    if source.startswith("results/tables/") and target.startswith("report/"):
        return "table -> report"
    if source.startswith("src/") and target.startswith("results/"):
        return "builder/runner -> raw output/table/manifest"
    if source.startswith("configs/") and target.startswith("src/"):
        return "config -> builder/runner"
    return "referencer -> referenced"


def build_edges(records: Sequence[FileRecord]) -> list[dict[str, str]]:
    records_by_path = {record.path: record for record in records}
    known_paths = {record.path: record.path for record in records}
    basename_map: dict[str, list[str]] = defaultdict(list)
    for record in records:
        basename_map[Path(record.path.rstrip("/")).name].append(record.path)
    module_map = build_module_map(records)
    edges: list[dict[str, str]] = []

    for record in records:
        for target, evidence in parse_python_imports(record, module_map):
            add_edge(edges, target, record.path, "python_import", evidence, records_by_path)
        for target, evidence in parse_path_references(record, known_paths, basename_map):
            edge_type = "text_path_reference"
            if record.path.endswith(".md"):
                edge_type = "report_or_doc_reference"
            elif record.path.endswith((".csv", ".json", ".jsonl")):
                edge_type = "manifest_reference"
            elif record.path.endswith((".yaml", ".yml")):
                edge_type = "config_path_reference"
            add_edge(edges, record.path, target, edge_type, evidence, records_by_path)

        if record.path.endswith(".py") and "subprocess." in record.text_sample:
            add_edge(edges, record.path, "external_command:subprocess", "subprocess_call", "subprocess.*", records_by_path)

    add_phase_registry_edges(edges, records_by_path)
    add_lineage_edges(edges, records_by_path)
    unique = {f"{row['source_path']}|{row['target_path']}|{row['edge_type']}|{row['evidence']}": row for row in edges}
    return sorted(unique.values(), key=lambda row: (row["source_path"], row["target_path"], row["edge_type"], row["evidence"]))


def add_phase_registry_edges(edges: list[dict[str, str]], records_by_path: Mapping[str, FileRecord]) -> None:
    registry = read_csv_rows(PROJECT_ROOT / "results" / "tables" / "workspace_phase_and_supersession_registry.csv")
    for row in registry:
        phase_id = row.get("phase_id", "")
        sources = split_semicolon(row.get("source_files", ""))
        configs = split_semicolon(row.get("config_files", ""))
        tests = split_semicolon(row.get("test_files", ""))
        outputs = split_semicolon(row.get("output_files", ""))
        reports = split_semicolon(row.get("report_files", ""))
        for config in configs:
            for source in sources:
                add_edge(edges, config, source, "phase_registry_config_to_source", f"phase={phase_id}", records_by_path)
        for source in sources:
            for output in outputs:
                add_edge(edges, source, output, "phase_registry_source_to_output", f"phase={phase_id}", records_by_path)
            for report in reports:
                add_edge(edges, source, report, "phase_registry_source_to_report", f"phase={phase_id}", records_by_path)
        for output in outputs:
            for report in reports:
                add_edge(edges, output, report, "phase_registry_output_to_report", f"phase={phase_id}", records_by_path)
            for test in tests:
                add_edge(edges, output, test, "phase_registry_output_to_test", f"phase={phase_id}", records_by_path)


def add_lineage_edges(edges: list[dict[str, str]], records_by_path: Mapping[str, FileRecord]) -> None:
    lineage_sources = [
        PROJECT_ROOT / "results" / "tables" / "artifact_lineage_and_supersession.csv",
        PROJECT_ROOT / "results" / "tables" / "plan_d_phase_6a_provenance_dependencies.csv",
    ]
    for path in lineage_sources:
        for row in read_csv_rows(path):
            artifact = normalize_path(row.get("artifact_path", ""))
            if not artifact:
                continue
            for source in split_semicolon(row.get("source_inputs", "")):
                add_edge(edges, source, artifact, "lineage_input_to_artifact", f"lineage={path.name}", records_by_path)
            producer = row.get("producer", "") or row.get("producer_or_consumer", "")
            if producer and producer.startswith("src/"):
                add_edge(edges, producer, artifact, "lineage_producer_to_artifact", f"lineage={path.name}", records_by_path)


def edge_indexes(edges: Sequence[Mapping[str, str]]) -> dict[str, dict[str, list[str]]]:
    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    importers: dict[str, list[str]] = defaultdict(list)
    tests: dict[str, list[str]] = defaultdict(list)
    reports: dict[str, list[str]] = defaultdict(list)
    callers: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        source = edge["source_path"]
        target = edge["target_path"]
        incoming[target].append(source)
        outgoing[source].append(target)
        if edge["edge_type"] == "python_import":
            importers[source].append(target)
            if target.startswith("tests/"):
                tests[source].append(target)
            else:
                callers[source].append(target)
        if target.startswith("report/") or target.startswith("results/manifests/"):
            reports[source].append(target)
        if source.startswith("report/") or source.startswith("results/manifests/"):
            reports[target].append(source)
    return {
        "incoming": incoming,
        "outgoing": outgoing,
        "importers": importers,
        "tests": tests,
        "reports": reports,
        "callers": callers,
    }


def join(values: Iterable[str], limit: int = 25) -> str:
    unique = sorted({value for value in values if value})
    if len(unique) > limit:
        return ";".join(unique[:limit]) + f";...(+{len(unique) - limit})"
    return ";".join(unique)


def future_dependencies(record: FileRecord) -> list[tuple[str, str]]:
    if is_cache_path(record.path):
        return []
    lower = f"{record.path.lower()} {record.text_sample[:50_000].lower()}"
    dependencies = []
    for dependency, terms in FUTURE_ROADMAP_TERMS.items():
        matched = [term for term in terms if term in lower]
        if matched:
            dependencies.append((dependency, ",".join(matched[:5])))
    if record.category in {"config", "source_code", "test", "local_data", "raw_data", "raw_llm_output", "prediction", "model_checkpoint", "model_artifact"}:
        dependencies.append(("future roadmap execution substrate", record.category))
    return sorted(set(dependencies))


def unique_scientific_information(record: FileRecord) -> str:
    if is_cache_path(record.path):
        return "no: Python bytecode cache only"
    if record.is_directory_unit:
        return "yes: retained as complete runtime or inaccessible state unit"
    if record.category in {
        "config",
        "local_data",
        "raw_data",
        "paper",
        "official_repository",
        "raw_llm_output",
        "prediction",
        "model_checkpoint",
        "model_artifact",
        "manifest",
        "result_table",
        "report",
    }:
        return f"yes: {record.category} may carry scientific/provenance evidence"
    lower = f"{record.path.lower()} {record.text_sample[:50_000].lower()}"
    matches = []
    for label, terms in UNIQUE_INFO_TERMS.items():
        if any(term in lower for term in terms):
            matches.append(label)
    if matches:
        return "yes: " + ",".join(matches)
    if record.category in {"source_code", "test"}:
        return f"yes: {record.category} behavior or regression evidence"
    return "no: no scientific terms detected in inspected sample"


def is_protected_path(path: str) -> bool:
    normalized = normalize_path(path).rstrip("/")
    if is_cache_path(normalized):
        return any(normalized.startswith(prefix) for prefix in PROTECTED_PREFIXES) or any(
            normalized == unit or normalized.startswith(unit + "/") for unit in ENV_UNIT_DIRS
        )
    if normalized in PROTECTED_ROOT_FILES or normalized in CURRENT_CONTEXT_PATHS:
        return True
    if any(normalized.startswith(prefix) for prefix in PROTECTED_PREFIXES):
        return True
    if normalized.startswith("configs/"):
        return True
    if normalized.startswith("src/evaluation/") or normalized.startswith("src/metrics/") or normalized.startswith("src/prompts/"):
        return True
    if normalized.startswith("src/llm/") and ("parser" in normalized or "prompt" in normalized):
        return True
    if any(normalized == unit or normalized.startswith(unit + "/") for unit in ENV_UNIT_DIRS):
        return True
    return False


def is_cache_path(path: str) -> bool:
    normalized = normalize_path(path)
    parts = normalized.split("/")
    return "__pycache__" in parts or normalized.endswith((".pyc", ".pyo"))


def source_for_pyc(path: str) -> str:
    normalized = normalize_path(path)
    if not normalized.endswith((".pyc", ".pyo")) or "/__pycache__/" not in normalized:
        return ""
    pycache_parent, filename = normalized.rsplit("/__pycache__/", 1)
    module = filename.split(".", 1)[0]
    source = f"{pycache_parent}/{module}.py"
    return source if (PROJECT_ROOT / source).exists() else ""


def cache_deletion_gate(record: FileRecord, edge_index: Mapping[str, Mapping[str, list[str]]]) -> tuple[bool, str, str]:
    if not is_cache_path(record.path):
        return False, "not_cache", ""
    if is_protected_path(record.path):
        return False, "protected_runtime_or_scientific_area", ""
    source = source_for_pyc(record.path)
    if not source:
        return False, "no_retained_source_for_regeneration", ""
    if edge_index["incoming"].get(record.path) or edge_index["outgoing"].get(record.path):
        return False, "referenced_by_dependency_graph", source
    return True, "all_delete_gates_pass_for_python_bytecode_cache", source


def duplicate_groups(records: Sequence[FileRecord], classification_by_path: Mapping[str, str]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, int], list[FileRecord]] = defaultdict(list)
    for record in records:
        if record.is_directory_unit or record.sha256 in {"", "UNREADABLE"}:
            continue
        grouped[(record.sha256, record.size_bytes)].append(record)
    rows: list[dict[str, str]] = []
    group_number = 0
    for (digest, size), members in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        if len(members) < 2:
            continue
        group_number += 1
        canonical = choose_duplicate_canonical(members, classification_by_path)
        for member in sorted(members, key=lambda item: item.path):
            delete_eligible = (
                classification_by_path.get(member.path) == "delete_now_exact_duplicate"
                and member.path != canonical.path
            )
            rows.append(
                {
                    "duplicate_group_id": f"DUP-F6I1-{group_number:04d}",
                    "sha256": digest,
                    "size_bytes": str(size),
                    "member_path": member.path,
                    "canonical_path": canonical.path,
                    "canonical_selection_reason": duplicate_canonical_reason(canonical),
                    "delete_eligible": str(delete_eligible).lower(),
                    "gate_status": "pass" if delete_eligible else "not_delete_batch1_or_path_semantic",
                }
            )
    return rows


def duplicate_score(record: FileRecord, classification_by_path: Mapping[str, str]) -> tuple[int, str]:
    classification = classification_by_path.get(record.path, "manual_review")
    score = 100
    if classification in {"keep_protected", "keep_canonical_runtime"}:
        score = 0
    elif record.tracked_state.startswith("tracked"):
        score = 10
    elif classification.startswith("keep_"):
        score = 20
    elif classification == "keep_provenance_only":
        score = 30
    elif classification == "manual_review":
        score = 50
    elif classification.startswith("delete_now"):
        score = 90
    return (score, record.path)


def choose_duplicate_canonical(records: Sequence[FileRecord], classification_by_path: Mapping[str, str]) -> FileRecord:
    return sorted(records, key=lambda record: duplicate_score(record, classification_by_path))[0]


def duplicate_canonical_reason(record: FileRecord) -> str:
    if record.path.startswith("original_repo/") or record.path.startswith("paper/") or record.path.startswith("data/raw/"):
        return "protected_authoritative_or_raw_path"
    if record.tracked_state.startswith("tracked"):
        return "tracked_repository_path"
    if record.category in {"source_code", "test", "config"}:
        return "executable_or_test_path"
    return "lowest_risk_sorted_path"


def classify_record(record: FileRecord, edge_index: Mapping[str, Mapping[str, list[str]]]) -> tuple[str, str, str]:
    path = record.path.rstrip("/")
    category = record.category
    if is_cache_path(path):
        passed, reason, source = cache_deletion_gate(record, edge_index)
        if passed:
            return "delete_now_regenerable", reason, source
        return "manual_review", reason, source
    if record.is_directory_unit:
        if path.startswith(".venv") or path.startswith("results/environment/"):
            return "keep_canonical_runtime", "runtime directory retained as an atomic environment unit", ""
        return "manual_review", "inaccessible or temporary directory retained because contents were not expanded", ""
    if is_protected_path(path):
        return "keep_protected", "explicitly protected by phase exclusions or repository rules", ""
    if category == "source_code":
        if path.startswith("src/analysis/") or path.startswith("src/experiments/") or path.startswith("src/baselines/") or path.startswith("src/data/"):
            return "keep_canonical_builder", "retained executable source or builder for roadmap tasks", ""
        return "keep_future_dependency", "retained source may support parser, prompt, metric, or future extension work", ""
    if category == "test":
        return "keep_canonical_test", "retained regression coverage for current canonical implementation", ""
    if category in {"docs"}:
        return "keep_current_documentation", "current documentation retained; Markdown consolidation excluded from batch 1", ""
    if category == "report":
        if path in CURRENT_CONTEXT_PATHS or "final" in path or "canonical" in path or path == "report/notes.md":
            return "keep_current_documentation", "current or canonical report retained", ""
        return "keep_provenance_only", "phase report may contain unique decision evidence; Markdown consolidation excluded", ""
    if category in {"manifest", "result_table"}:
        future = future_dependencies(record)
        if future:
            return "keep_future_dependency", "generated artifact maps to at least one future-roadmap dependency", ""
        return "keep_provenance_only", "generated artifact retained for provenance; non-use not sufficient for deletion", ""
    if category in {"local_data", "raw_data", "raw_llm_output", "prediction", "model_checkpoint", "model_artifact"}:
        return "keep_future_dependency", "data/output/model artifact retained for current or future roadmap capability", ""
    if category == "root_or_other":
        return "keep_current_documentation", "root support file retained by repository policy", ""
    return "manual_review", "unresolved currentness or dependency role; fail closed", ""


def validation_command() -> str:
    return (
        "PYTHONNOUSERSITE=1 .\\.venv_audit_recovery\\Scripts\\python.exe -B "
        "-m pytest tests/test_phase_f6i1_repository_retention_audit.py -q"
    )


def build_inventory_rows(records: Sequence[FileRecord], edges: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    indexes = edge_indexes(edges)
    rows = []
    for record in records:
        futures = future_dependencies(record)
        rows.append(
            {
                "path": record.path,
                "size_bytes": str(record.size_bytes),
                "sha256": record.sha256,
                "tracked_state": record.tracked_state,
                "phase": phase_for(record.path, record.text_sample),
                "purpose": purpose_for(record.path, record.category, record.text_sample),
                "producer": infer_producer(record.path, indexes),
                "inputs": join(indexes["incoming"].get(record.path, [])),
                "outputs": join(indexes["outgoing"].get(record.path, [])),
                "importers": join(indexes["importers"].get(record.path, [])),
                "callers": join(indexes["callers"].get(record.path, [])),
                "tests": join(indexes["tests"].get(record.path, [])),
                "reports_and_manifests_referencing": join(indexes["reports"].get(record.path, [])),
                "unique_scientific_information": unique_scientific_information(record),
                "canonical_replacement": "",
                "future_roadmap_dependencies": join([dependency for dependency, _ in futures]),
                "category": record.category,
                "extension": record.extension,
                "inspection_status": record.inspection_status,
            }
        )
    return rows


def infer_producer(path: str, indexes: Mapping[str, Mapping[str, list[str]]]) -> str:
    incoming = indexes["incoming"].get(path, [])
    source_candidates = [source for source in incoming if source.startswith("src/")]
    if source_candidates:
        return join(source_candidates, limit=10)
    if path.startswith("src/") or path.startswith("tests/") or path in PROTECTED_ROOT_FILES:
        return "repository_file"
    if path.startswith("original_repo/"):
        return "official_repository_snapshot"
    if path.startswith("paper/"):
        return "downloaded_paper_or_user_supplied_paper"
    if path.startswith("data/raw/"):
        return "raw_dataset_intake"
    return "unknown_or_prior_phase"


def build_classification_rows(
    records: Sequence[FileRecord],
    edges: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    indexes = edge_indexes(edges)
    rows: list[dict[str, str]] = []
    delete_rows: list[dict[str, str]] = []
    manual_rows: list[dict[str, str]] = []
    classification_by_path: dict[str, str] = {}
    for record in records:
        classification, reason, replacement = classify_record(record, indexes)
        classification_by_path[record.path] = classification
        dependency_evidence = dependency_evidence_for(record.path, indexes)
        unique_check = unique_scientific_information(record)
        future_check = "pass:no future roadmap dependency detected" if not future_dependencies(record) else (
            "blocked:" + join([dependency for dependency, _ in future_dependencies(record)], limit=8)
        )
        if classification.startswith("delete_now"):
            gate_status = "pass"
            canonical_replacement = replacement or "regenerate from retained source/builder"
            rollback = canonical_replacement
        elif classification == "manual_review":
            gate_status = "blocked:manual_review_required"
            canonical_replacement = replacement
            rollback = ""
        else:
            gate_status = "not_applicable:retained"
            canonical_replacement = replacement
            rollback = ""
        row = {
            "path": record.path,
            "class": classification,
            "reason": reason,
            "deletion_gate_status": gate_status,
            "canonical_replacement": canonical_replacement,
            "dependency_evidence": dependency_evidence,
            "unique_information_check": unique_check,
            "future_roadmap_check": future_check,
            "validation_command": validation_command() if classification.startswith("delete_now") else "",
            "rollback_source": rollback,
        }
        rows.append(row)
        if classification.startswith("delete_now"):
            delete_rows.append(
                {
                    "path": record.path,
                    "sha256": record.sha256,
                    "size_bytes": str(record.size_bytes),
                    "class": classification,
                    "reason": reason,
                    "canonical_replacement": canonical_replacement,
                    "dependency_evidence": dependency_evidence,
                    "unique_information_check": unique_check,
                    "future_roadmap_check": future_check,
                    "validation_command": validation_command(),
                    "rollback_source": rollback,
                }
            )
        if classification == "manual_review":
            manual_rows.append(
                {
                    "path": record.path,
                    "reason": reason,
                    "risk": "medium" if not is_cache_path(record.path) else "low",
                    "blocking_gate": blocking_gate_for(reason),
                    "recommended_next_action": "inspect manually before any later archive/delete batch",
                    "dependency_evidence": dependency_evidence,
                }
            )
    return rows, delete_rows, manual_rows, classification_by_path


def dependency_evidence_for(path: str, indexes: Mapping[str, Mapping[str, list[str]]]) -> str:
    incoming = indexes["incoming"].get(path, [])
    outgoing = indexes["outgoing"].get(path, [])
    parts = []
    if incoming:
        parts.append("incoming=" + join(incoming, limit=8))
    if outgoing:
        parts.append("outgoing=" + join(outgoing, limit=8))
    return "|".join(parts) if parts else "no direct graph edge found; retained/deleted only through explicit gates"


def blocking_gate_for(reason: str) -> str:
    if reason == "no_retained_source_for_regeneration":
        return "gate_9_canonical_replacement_or_regeneration_path"
    if reason == "referenced_by_dependency_graph":
        return "gate_3_active_reference"
    if reason == "protected_runtime_or_scientific_area":
        return "gate_1_or_phase_exclusion"
    if "inaccessible" in reason:
        return "content_inspection_incomplete"
    return "uncertainty_fail_closed"


def build_future_rows(records: Sequence[FileRecord]) -> list[dict[str, str]]:
    rows = []
    for record in records:
        for dependency, evidence in future_dependencies(record):
            rows.append(
                {
                    "path": record.path,
                    "roadmap_dependency": dependency,
                    "evidence": evidence,
                    "retention_effect": "prevents_delete_now",
                }
            )
    return rows


def build_initial_capture(records: Sequence[FileRecord], phase_registry_occupied: bool) -> dict[str, Any]:
    branch = git_lines(["branch", "--show-current"])
    head = git_lines(["rev-parse", "HEAD"])
    status_code, status_stdout, status_stderr = run_command(["git", "status", "--porcelain=v1", "--branch"], timeout=60)
    states = git_state_maps()
    extension_counts = Counter(record.extension for record in records)
    directory_sizes = Counter()
    for record in records:
        top = record.path.split("/", 1)[0]
        directory_sizes[top] += record.size_bytes
    status_lines = [line for line in status_stdout.splitlines() if line and not line.startswith("##")]
    tracked_modified = len(states["modified"])
    untracked = len(states["untracked"]) + len(states.get("untracked_dirs", set()))
    ignored = git_ignored_count()
    tracked = len(states["tracked"])
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "runtime_caveat": "not exact original runtime reproduction",
        "approved_runtime_interpreter": str(RECOVERED_INTERPRETER),
        "branch": branch[0] if branch else "",
        "expected_branch": EXPECTED_BRANCH,
        "head": head[0] if head else "",
        "expected_head": EXPECTED_HEAD,
        "branch_matches_expected": bool(branch and branch[0] == EXPECTED_BRANCH),
        "head_matches_expected": bool(head and head[0] == EXPECTED_HEAD),
        "git_status_returncode": status_code,
        "git_status_stderr": status_stderr,
        "git_status_full": status_stdout,
        "git_status_row_count": len(status_lines),
        "tracked_count": tracked,
        "modified_count": tracked_modified,
        "untracked_count": untracked,
        "ignored_count": ignored,
        "repository_relevant_size_bytes": sum(record.size_bytes for record in records),
        "directory_sizes_bytes": dict(sorted(directory_sizes.items())),
        "file_counts_by_extension": dict(sorted(extension_counts.items())),
        "python_files": sum(1 for record in records if record.path.endswith(".py")),
        "yaml_files": sum(1 for record in records if record.path.endswith((".yaml", ".yml"))),
        "tests": sum(1 for record in records if record.path.startswith("tests/") and record.path.endswith(".py")),
        "markdown_reports": sum(1 for record in records if record.path.startswith("report/") and record.path.endswith(".md")),
        "manifests": sum(1 for record in records if record.category == "manifest" or "manifest" in record.path.lower()),
        "result_tables": sum(1 for record in records if record.category == "result_table"),
        "prompt_and_codex_files": sum(1 for record in records if "prompt" in record.path.lower() or "codex" in record.path.lower()),
        "phase_registry_inspected": "results/tables/workspace_phase_and_supersession_registry.csv",
        "phase_namespace_occupied": phase_registry_occupied,
    }


def phase_namespace_occupied() -> bool:
    registry_path = PROJECT_ROOT / "results" / "tables" / "workspace_phase_and_supersession_registry.csv"
    if not registry_path.exists():
        return False
    text = registry_path.read_text(encoding="utf-8", errors="replace").lower()
    return PHASE_ID in text


def build_phase_tables() -> PhaseTables:
    occupied = phase_namespace_occupied()
    records = discover_records(PROJECT_ROOT)
    edges = build_edges(records)
    inventory = build_inventory_rows(records, edges)
    classification, delete_candidates, manual_review, classification_by_path = build_classification_rows(records, edges)
    futures = build_future_rows(records)
    duplicates = duplicate_groups(records, classification_by_path)
    initial_capture = build_initial_capture(records, occupied)
    return PhaseTables(
        inventory=inventory,
        edges=edges,
        classification=classification,
        delete_candidates=delete_candidates,
        manual_review=manual_review,
        future_dependency=futures,
        duplicates=duplicates,
        initial_capture=initial_capture,
    )


def write_phase_outputs(tables: PhaseTables) -> None:
    write_csv(OUTPUT_PATHS["inventory"], tables.inventory, INVENTORY_COLUMNS)
    write_csv(OUTPUT_PATHS["edges"], tables.edges, EDGE_COLUMNS)
    write_csv(OUTPUT_PATHS["classification"], tables.classification, CLASSIFICATION_COLUMNS)
    write_csv(OUTPUT_PATHS["delete_candidates"], tables.delete_candidates, DELETE_COLUMNS)
    write_csv(OUTPUT_PATHS["manual_review"], tables.manual_review, MANUAL_REVIEW_COLUMNS)
    write_csv(OUTPUT_PATHS["future_dependency"], tables.future_dependency, FUTURE_COLUMNS)
    write_csv(OUTPUT_PATHS["duplicates"], tables.duplicates, DUPLICATE_COLUMNS)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    INITIAL_CAPTURE_PATH.write_text(json.dumps(tables.initial_capture, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def semantic_digest(tables: PhaseTables) -> str:
    payload = {
        "inventory_paths": [(row["path"], row["sha256"], row["size_bytes"], row["tracked_state"]) for row in tables.inventory],
        "edges": tables.edges,
        "classification": tables.classification,
        "delete_candidates": tables.delete_candidates,
        "manual_review": tables.manual_review,
        "future_dependency": tables.future_dependency,
        "duplicates": tables.duplicates,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest().upper()


def validate_tables(tables: PhaseTables, determinism_observed: str = "") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(check: str, expected: str, observed: object, passed: bool, severity: str = "blocking", evidence: str = "") -> None:
        rows.append(
            {
                "validation_id": f"VAL-F6I1-{len(rows) + 1:03d}",
                "check_name": check,
                "expected": expected,
                "observed": str(observed),
                "status": "pass" if passed else "fail",
                "severity": severity,
                "evidence": evidence,
            }
        )

    paths = [row["path"] for row in tables.inventory]
    add("branch_matches_expected", EXPECTED_BRANCH, tables.initial_capture.get("branch"), bool(tables.initial_capture.get("branch_matches_expected")))
    add("head_matches_expected", EXPECTED_HEAD, tables.initial_capture.get("head"), bool(tables.initial_capture.get("head_matches_expected")))
    add("phase_namespace_unoccupied_before_generation", "false", tables.initial_capture.get("phase_namespace_occupied"), not tables.initial_capture.get("phase_namespace_occupied"))
    add("one_inventory_row_per_path", "unique paths", len(set(paths)) == len(paths), len(set(paths)) == len(paths))
    bad_classes = [row["class"] for row in tables.classification if row["class"] not in RETENTION_CLASSES]
    add("valid_retention_classes", "0 invalid classes", len(bad_classes), not bad_classes)
    required_outputs = [normalize_path(path) for path in OUTPUT_PATHS.values()]
    missing_output_rows = [path for path in required_outputs if path not in {row["path"] for row in tables.inventory} and (PROJECT_ROOT / path).exists()]
    add("phase_outputs_inventoried_when_present", "0 missing output rows", len(missing_output_rows), not missing_output_rows, evidence=";".join(missing_output_rows))
    protected_delete = [row["path"] for row in tables.delete_candidates if is_protected_path(row["path"])]
    add("no_protected_delete_candidates", "0 protected delete candidates", len(protected_delete), not protected_delete, evidence=";".join(protected_delete[:10]))
    missing_replacement = [row["path"] for row in tables.delete_candidates if not row["canonical_replacement"] and not row["rollback_source"]]
    add("delete_candidates_have_replacement_or_regeneration", "0 missing replacements", len(missing_replacement), not missing_replacement, evidence=";".join(missing_replacement[:10]))
    bad_unique = [row["path"] for row in tables.delete_candidates if row["unique_information_check"].startswith("yes:")]
    add("delete_candidates_have_no_unique_scientific_information", "0 unique-info candidates", len(bad_unique), not bad_unique, evidence=";".join(bad_unique[:10]))
    bad_future = [row["path"] for row in tables.delete_candidates if row["future_roadmap_check"].startswith("blocked:")]
    add("delete_candidates_not_future_dependencies", "0 future-dependent candidates", len(bad_future), not bad_future, evidence=";".join(bad_future[:10]))
    if determinism_observed:
        add("two_run_manifest_determinism", "same semantic digest", determinism_observed, determinism_observed == "pass")
    return rows


def merge_delete_candidate_history(current_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Preserve candidate details across reruns after files have been deleted."""
    merged: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv_rows(OUTPUT_PATHS["delete_candidates"]):
        if row.get("path") and row.get("sha256"):
            merged[(row["path"], row["sha256"])] = {column: row.get(column, "") for column in DELETE_COLUMNS}
    for row in current_rows:
        merged[(row["path"], row["sha256"])] = {column: row.get(column, "") for column in DELETE_COLUMNS}
    for cleanup in read_csv_rows(CLEANUP_EXECUTION_PATH):
        if cleanup.get("status") != "pass":
            continue
        path = cleanup.get("path", "")
        digest = cleanup.get("pre_sha256", "")
        if not path or not digest or (path, digest) in merged:
            continue
        source = source_for_pyc(path)
        replacement = source or "regenerate from retained source/builder"
        merged[(path, digest)] = {
            "path": path,
            "sha256": digest,
            "size_bytes": cleanup.get("pre_size_bytes", ""),
            "class": "delete_now_regenerable",
            "reason": "all_delete_gates_pass_for_python_bytecode_cache",
            "canonical_replacement": replacement,
            "dependency_evidence": "historical cleanup execution row; no direct graph edge found at deletion time",
            "unique_information_check": "no: Python bytecode cache only",
            "future_roadmap_check": "pass:no future roadmap dependency detected",
            "validation_command": validation_command(),
            "rollback_source": replacement,
        }
    return sorted(merged.values(), key=lambda row: (row["path"], row["sha256"]))


def execute_cleanup(delete_candidates: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for candidate in sorted(delete_candidates, key=lambda row: row["path"]):
        relative = normalize_path(candidate["path"])
        absolute = (PROJECT_ROOT / relative).resolve()
        try:
            absolute.relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            rows.append(cleanup_row(candidate, "skip", "fail", "unknown", "path outside project root"))
            continue
        if is_protected_path(relative) or not is_cache_path(relative):
            rows.append(cleanup_row(candidate, "skip", "fail", str(absolute.exists()).lower(), "gate recheck failed"))
            continue
        source = source_for_pyc(relative)
        if not source:
            rows.append(cleanup_row(candidate, "skip", "fail", str(absolute.exists()).lower(), "source regeneration path missing"))
            continue
        if not absolute.exists():
            rows.append(cleanup_row(candidate, "delete_now", "already_absent", "false", "candidate already absent"))
            continue
        try:
            absolute.unlink()
            rows.append(cleanup_row(candidate, "delete_now", "pass", str(absolute.exists()).lower(), f"retained_source={source}"))
        except OSError as exc:
            rows.append(cleanup_row(candidate, "delete_now", "fail", str(absolute.exists()).lower(), exc.__class__.__name__))
    remove_empty_cache_directories()
    return rows


def cleanup_row(candidate: Mapping[str, str], action: str, status: str, post_exists: str, evidence: str) -> dict[str, str]:
    return {
        "path": candidate["path"],
        "pre_sha256": candidate["sha256"],
        "pre_size_bytes": candidate["size_bytes"],
        "action": action,
        "status": status,
        "post_exists": post_exists,
        "validation_evidence": evidence,
    }


def remove_empty_cache_directories() -> None:
    cache_dirs: list[Path] = []
    for current, dirs, _files in os.walk(PROJECT_ROOT, topdown=True, onerror=None):
        current_path = Path(current)
        kept_dirs = []
        for dirname in sorted(dirs):
            child = current_path / dirname
            relative = normalize_path(child.relative_to(PROJECT_ROOT))
            if should_skip_unit(relative) or dirname in SKIP_DIR_NAMES:
                continue
            kept_dirs.append(dirname)
            if dirname == "__pycache__":
                cache_dirs.append(child)
        dirs[:] = kept_dirs
    for path in sorted(cache_dirs, key=lambda item: len(item.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            continue


def generate_report(
    tables: PhaseTables,
    validations: Sequence[Mapping[str, str]],
    cleanup_rows: Sequence[Mapping[str, str]],
) -> None:
    classification_counts = Counter(row["class"] for row in tables.classification)
    category_counts = Counter(row["category"] for row in tables.inventory)
    delete_status_counts = Counter(row["status"] for row in cleanup_rows)
    failed_validations = [row for row in validations if row["status"] != "pass" and row["severity"] == "blocking"]
    capture = tables.initial_capture
    deleted_count = sum(1 for row in cleanup_rows if row["status"] == "pass")
    report = f"""# Phase F6I1 Repository Retention Audit And Safe Cleanup Batch 1

Decision: `f6i1_repository_retention_audit_completed_with_safe_batch1_cleanup`.

Runtime classification: `{RUNTIME_CLASSIFICATION}`. This is not exact original runtime reproduction.

## Scope

- Phase namespace selected: `{PHASE_ID}`.
- Phase registry inspected: `results/tables/workspace_phase_and_supersession_registry.csv`.
- Namespace occupied before generation: `{capture.get("phase_namespace_occupied")}`.
- Branch: `{capture.get("branch")}`.
- HEAD: `{capture.get("head")}`.
- Cleanup policy: delete only Python bytecode cache files outside protected paths when the retained source file exists.

## Initial Capture

- Git status rows: `{capture.get("git_status_row_count")}`.
- Tracked files: `{capture.get("tracked_count")}`.
- Modified tracked files: `{capture.get("modified_count")}`.
- Untracked files: `{capture.get("untracked_count")}`.
- Ignored files: `{capture.get("ignored_count")}`.
- Relevant inventory rows: `{len(tables.inventory)}`.
- Dependency edges: `{len(tables.edges)}`.
- Duplicate group member rows: `{len(tables.duplicates)}`.
- Repository relevant size bytes, excluding expanded environment package trees: `{capture.get("repository_relevant_size_bytes")}`.

## Classification Summary

{markdown_counter(classification_counts)}

## Category Summary

{markdown_counter(category_counts)}

## Pre-Deletion Manifests

- `results/tables/phase_f6i1_relevant_file_inventory.csv`
- `results/tables/phase_f6i1_dependency_edges.csv`
- `results/tables/phase_f6i1_retention_classification.csv`
- `results/tables/phase_f6i1_delete_now_candidates.csv`
- `results/tables/phase_f6i1_manual_review.csv`
- `results/tables/phase_f6i1_future_dependency_map.csv`
- `results/tables/phase_f6i1_duplicate_groups.csv`

The phase wrote these manifests before each cleanup pass. The final inventory reflects the retained post-cleanup file set; the delete-now candidate and cleanup execution tables preserve cumulative deletion evidence. The delete manifest fails closed when a row lacks a retained source/regeneration path.

## Cleanup Result

- Delete-now candidates: `{len(tables.delete_candidates)}`.
- Files deleted: `{deleted_count}`.
- Cleanup status counts: `{dict(sorted(delete_status_counts.items()))}`.
- Non-cache exact duplicate scientific artifacts deleted: `0`.
- Original repository, paper files, raw data, raw LLM outputs, predictions, checkpoints, candidates, histories, prompt templates, parsers, metrics, active configs, reports, and runtime units were retained.

## Validation

- Blocking validation failures: `{len(failed_validations)}`.
- Validation table: `results/tables/phase_f6i1_validation.csv`.
- Cleanup execution table: `results/tables/phase_f6i1_cleanup_execution.csv`.

## Limitations

- `.venv`, `.venv_audit_recovery`, sibling recovery environments, packaged CPython runtime trees, and inaccessible pytest/log directories are represented as directory units, not package-file inventories.
- Manual-review rows are intentionally retained. No import/reference absence was used as standalone deletion evidence.
- Markdown consolidation, Python module merging, YAML architecture rewriting, and non-regenerable script/config/test deletion were not performed in batch 1.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def markdown_counter(counter: Counter[str]) -> str:
    if not counter:
        return "- none"
    return "\n".join(f"- `{key}`: `{value}`" for key, value in sorted(counter.items()))


def append_notes(tables: PhaseTables, cleanup_rows: Sequence[Mapping[str, str]], validations: Sequence[Mapping[str, str]]) -> None:
    notes_path = PROJECT_ROOT / "report" / "notes.md"
    deleted = sum(1 for row in cleanup_rows if row["status"] == "pass")
    failures = sum(1 for row in validations if row["status"] != "pass" and row["severity"] == "blocking")
    entry = f"""

## 2026-07-29 - Phase F6I1 Repository Retention Audit And Safe Cleanup Batch 1

### Changes

- Reconstructed a conservative repository dependency graph and file inventory for `{PHASE_ID}`.
- Classified every inventoried project-relevant path into one retention class.
- Generated the requested pre-deletion manifests before cleanup.
- Executed batch-1 cleanup only for Python bytecode cache files with retained source-file regeneration paths.

### Commands Run

- `PYTHONNOUSERSITE=1 .\\.venv_audit_recovery\\Scripts\\python.exe -B src/experiments/build_phase_f6i1_repository_retention_audit.py --execute-cleanup --verify-determinism`
- `PYTHONNOUSERSITE=1 .\\.venv_audit_recovery\\Scripts\\python.exe -B -m pytest tests/test_phase_f6i1_repository_retention_audit.py -q`

No inference, API call, download, model pull, recommender training, checkpoint mutation, raw data mutation, original repository mutation, Git staging, commit, reset, clean, or checkout was run.

### Outputs Generated

- `src/analysis/phase_f6i1_repository_retention_audit.py`
- `src/experiments/build_phase_f6i1_repository_retention_audit.py`
- `tests/test_phase_f6i1_repository_retention_audit.py`
- `results/tables/phase_f6i1_*.csv`
- `results/manifests/phase_f6i1_initial_capture.json`
- `report/phase_f6i1_repository_retention_audit_and_safe_cleanup_batch1.md`

### Result

- Inventory rows: `{len(tables.inventory)}`.
- Dependency edges: `{len(tables.edges)}`.
- Delete-now candidates: `{len(tables.delete_candidates)}`.
- Files deleted: `{deleted}`.
- Blocking validation failures: `{failures}`.

### Notes And Limitations

- Runtime classification remains `{RUNTIME_CLASSIFICATION}` and is not exact original runtime reproduction.
- Environment directories were retained as complete units and package files were not classified individually.
- Manual-review artifacts remain in place for later review.
- No scientific result, prompt, parser, metric implementation, raw response, prediction, checkpoint, candidate set, history, config, report, or original repository file was deleted.
"""
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    existing = notes_path.read_text(encoding="utf-8", errors="replace") if notes_path.exists() else ""
    if "Phase F6I1 Repository Retention Audit And Safe Cleanup Batch 1" not in existing:
        notes_path.write_text(existing.rstrip() + entry + "\n", encoding="utf-8")


def assert_delete_manifest_closed(delete_candidates: Sequence[Mapping[str, str]]) -> None:
    bad = [
        row["path"]
        for row in delete_candidates
        if not row.get("canonical_replacement") and not row.get("rollback_source")
    ]
    if bad:
        raise RuntimeError("Delete manifest has rows without canonical replacement/regeneration path: " + ";".join(bad[:20]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-cleanup", action="store_true", help="Delete only gate-passing batch-1 files after manifests are written.")
    parser.add_argument("--verify-determinism", action="store_true", help="Build tables twice in memory and compare semantic digests.")
    parser.add_argument("--skip-notes", action="store_true", help="Do not append report/notes.md.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    tables = build_phase_tables()
    current_delete_candidates = list(tables.delete_candidates)
    tables.delete_candidates[:] = merge_delete_candidate_history(tables.delete_candidates)
    determinism = ""
    if args.verify_determinism:
        second = build_phase_tables()
        second.delete_candidates[:] = merge_delete_candidate_history(second.delete_candidates)
        determinism = "pass" if semantic_digest(tables) == semantic_digest(second) else "fail"
    validations = validate_tables(tables, determinism)
    assert_delete_manifest_closed(tables.delete_candidates)
    write_phase_outputs(tables)

    cleanup_rows: list[dict[str, str]] = []
    if args.execute_cleanup:
        prior_cleanup_rows = read_csv_rows(CLEANUP_EXECUTION_PATH)
        cleanup_rows = [dict(row) for row in prior_cleanup_rows]
        seen_cleanup_keys = {
            (row.get("path", ""), row.get("pre_sha256", ""), row.get("status", ""))
            for row in cleanup_rows
        }
        for row in execute_cleanup(current_delete_candidates):
            key = (row.get("path", ""), row.get("pre_sha256", ""), row.get("status", ""))
            if key not in seen_cleanup_keys:
                cleanup_rows.append(row)
                seen_cleanup_keys.add(key)
    write_csv(CLEANUP_EXECUTION_PATH, cleanup_rows, CLEANUP_COLUMNS)
    write_csv(VALIDATION_PATH, validations, VALIDATION_COLUMNS)
    generate_report(tables, validations, cleanup_rows)
    if not args.skip_notes:
        append_notes(tables, cleanup_rows, validations)

    blocking_failures = [row for row in validations if row["status"] != "pass" and row["severity"] == "blocking"]
    if blocking_failures:
        logging.error("Blocking validations failed: %s", "; ".join(row["check_name"] for row in blocking_failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
