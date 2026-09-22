"""Configuration loading and validation helpers.

These helpers are intentionally narrow: they validate project configuration
shape and resolve relative paths, but they do not load datasets, metrics, model
outputs, or API clients.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


PROJECT_MARKERS = ("README.md", "requirements.txt")
REQUIRED_TOP_LEVEL_SECTIONS = (
    "dataset",
    "paths",
    "original_repo",
    "experiment",
    "api",
    "models",
    "metrics",
    "output",
)
REQUIRED_PATH_KEYS = (
    "raw_dir",
    "processed_dir",
    "sample_dir",
    "interactions_path",
    "catalog_path",
    "candidates_path",
    "ground_truth_path",
    "llm_outputs_path",
    "baseline_outputs_path",
)
PATH_KEY_SUFFIXES = ("_path", "_dir")
PATH_KEY_NAMES = {"root", "source_file"}


class ConfigError(ValueError):
    """Raised when a project configuration is missing or invalid."""


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the project root by walking upward from ``start`` or the cwd."""

    current = Path.cwd() if start is None else Path(start)
    current = current.resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if all((candidate / marker).is_file() for marker in PROJECT_MARKERS):
            return candidate

    marker_list = ", ".join(PROJECT_MARKERS)
    raise ConfigError(f"Could not find project root containing: {marker_list}")


def load_yaml_config(
    config_path: str | Path,
    project_root: str | Path | None = None,
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """Load a YAML config from a path relative to the project root."""

    root = Path(project_root).resolve() if project_root is not None else find_project_root()
    path = _resolve_input_config_path(config_path, root)
    if not path.is_file():
        raise ConfigError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ConfigError(f"Config must be a YAML mapping: {path}")

    if validate:
        validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the project experiment config schema used by this repository."""

    _require_mapping(config, "config")
    _validate_top_level_sections(config)
    _validate_dataset(config["dataset"])
    _validate_paths_section(config["paths"], "paths")
    _validate_original_repo(config["original_repo"])
    _validate_experiment(config["experiment"])
    _validate_api(config["api"])
    _validate_models(config["models"])
    _validate_metrics(config["metrics"])
    _validate_output(config["output"])

    if "seed" in config:
        _require_integer(config["seed"], "seed")
    if "baselines" in config:
        _validate_baselines(config["baselines"], "baselines")
    if "llm_outputs" in config:
        _validate_llm_outputs(config["llm_outputs"], "llm_outputs")


def resolve_project_path(path_value: str | Path | None, project_root: str | Path) -> Path | None:
    """Resolve one configured path relative to ``project_root``.

    ``None`` is preserved. Absolute configured paths are rejected to keep configs
    portable across machines.
    """

    if path_value is None:
        return None
    if not isinstance(path_value, (str, Path)):
        raise ConfigError(f"Expected path string or null, got {type(path_value).__name__}")

    raw_path = Path(path_value)
    if raw_path.is_absolute():
        raise ConfigError(f"Configured paths must be relative to the project root: {path_value}")

    root = Path(project_root).resolve()
    resolved = (root / raw_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConfigError(f"Configured path escapes the project root: {path_value}") from exc
    return resolved


def resolve_configured_paths(
    config: Mapping[str, Any],
    project_root: str | Path | None = None,
) -> dict[str, Path | None]:
    """Resolve every path-like value in a config to an absolute ``Path``."""

    root = Path(project_root).resolve() if project_root is not None else find_project_root()
    resolved: dict[str, Path | None] = {}
    for dotted_key, value in _iter_path_values(config):
        resolved[dotted_key] = resolve_project_path(value, root)
    return resolved


def _resolve_input_config_path(config_path: str | Path, project_root: Path) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        try:
            path.resolve().relative_to(project_root)
        except ValueError as exc:
            raise ConfigError(f"Config path must be inside the project root: {path}") from exc
        return path.resolve()
    return (project_root / path).resolve()


def _validate_top_level_sections(config: Mapping[str, Any]) -> None:
    missing = [section for section in REQUIRED_TOP_LEVEL_SECTIONS if section not in config]
    if missing:
        raise ConfigError(f"Missing required top-level section(s): {', '.join(missing)}")
    for section in REQUIRED_TOP_LEVEL_SECTIONS:
        _require_mapping(config[section], section)


def _validate_dataset(dataset: Any) -> None:
    _require_mapping(dataset, "dataset")
    _require_keys(dataset, ("name",), "dataset")
    _require_non_empty_string(dataset["name"], "dataset.name")

    # Backward-compatible validation for the initial Prompt 03 shape, where
    # dataset path fields were also present under the dataset section.
    present_path_keys = [key for key in REQUIRED_PATH_KEYS if key in dataset]
    if present_path_keys:
        _require_keys(dataset, REQUIRED_PATH_KEYS, "dataset")
        _validate_path_values(dataset, "dataset", REQUIRED_PATH_KEYS)


def _validate_paths_section(paths: Any, section_name: str) -> None:
    _require_mapping(paths, section_name)
    _require_keys(paths, REQUIRED_PATH_KEYS, section_name)
    _validate_path_values(paths, section_name, REQUIRED_PATH_KEYS)


def _validate_original_repo(original_repo: Any) -> None:
    _require_mapping(original_repo, "original_repo")
    _require_keys(original_repo, ("root",), "original_repo")
    _validate_path_values(original_repo, "original_repo", tuple(original_repo.keys()))


def _validate_experiment(experiment: Any) -> None:
    _require_mapping(experiment, "experiment")
    _require_keys(
        experiment,
        ("task", "k_values", "seed", "sample_size", "allow_api_calls"),
        "experiment",
    )
    _require_non_empty_string(experiment["task"], "experiment.task")
    _require_integer(experiment["seed"], "experiment.seed")
    _require_boolean(experiment["allow_api_calls"], "experiment.allow_api_calls")
    _require_sequence(experiment["k_values"], "experiment.k_values")
    if not all(isinstance(k, int) and k > 0 for k in experiment["k_values"]):
        raise ConfigError("experiment.k_values must contain positive integers")
    if experiment["allow_api_calls"]:
        raise ConfigError("experiment.allow_api_calls must remain false by default")


def _validate_api(api: Any) -> None:
    _require_mapping(api, "api")
    _require_keys(api, ("allow_api_calls", "provider", "model", "api_key_env_var"), "api")
    _require_boolean(api["allow_api_calls"], "api.allow_api_calls")
    if api["allow_api_calls"]:
        raise ConfigError("api.allow_api_calls must remain false by default")
    for key in ("provider", "model", "api_key_env_var"):
        if api[key] is not None and not isinstance(api[key], str):
            raise ConfigError(f"api.{key} must be a string or null")


def _validate_models(models: Any) -> None:
    _require_mapping(models, "models")
    _require_keys(models, ("baselines", "llm_outputs"), "models")
    _validate_baselines(models["baselines"], "models.baselines")
    _validate_llm_outputs(models["llm_outputs"], "models.llm_outputs")
    if "baseline_outputs" in models:
        _validate_baseline_outputs(models["baseline_outputs"], "models.baseline_outputs")


def _validate_baselines(baselines: Any, field_name: str) -> None:
    _require_sequence(baselines, field_name)
    if not all(isinstance(item, str) and item for item in baselines):
        raise ConfigError(f"{field_name} must contain non-empty strings")


def _validate_llm_outputs(llm_outputs: Any, field_name: str) -> None:
    _require_sequence(llm_outputs, field_name)
    required = ("model_name", "source_file", "archive_member", "run_id")
    for index, entry in enumerate(llm_outputs):
        entry_name = f"{field_name}[{index}]"
        _require_mapping(entry, entry_name)
        _require_keys(entry, required, entry_name)
        _require_non_empty_string(entry["model_name"], f"{entry_name}.model_name")
        _require_non_empty_string(entry["archive_member"], f"{entry_name}.archive_member")
        _require_non_empty_string(entry["run_id"], f"{entry_name}.run_id")
        _validate_path_values(entry, entry_name, ("source_file",))


def _validate_baseline_outputs(baseline_outputs: Any, field_name: str) -> None:
    _require_sequence(baseline_outputs, field_name)
    required = ("model_name", "source_file", "archive_member")
    for index, entry in enumerate(baseline_outputs):
        entry_name = f"{field_name}[{index}]"
        _require_mapping(entry, entry_name)
        _require_keys(entry, required, entry_name)
        _require_non_empty_string(entry["model_name"], f"{entry_name}.model_name")
        _require_non_empty_string(entry["archive_member"], f"{entry_name}.archive_member")
        _validate_path_values(entry, entry_name, ("source_file",))


def _validate_metrics(metrics: Any) -> None:
    _require_mapping(metrics, "metrics")
    _require_keys(metrics, ("utility", "beyond_utility"), "metrics")
    for key in ("utility", "beyond_utility"):
        _require_sequence(metrics[key], f"metrics.{key}")
        if not all(isinstance(item, str) and item for item in metrics[key]):
            raise ConfigError(f"metrics.{key} must contain non-empty strings")
    if "matching_protocols" in metrics:
        _require_sequence(metrics["matching_protocols"], "metrics.matching_protocols")
        if not all(isinstance(item, str) and item for item in metrics["matching_protocols"]):
            raise ConfigError("metrics.matching_protocols must contain non-empty strings")


def _validate_output(output: Any) -> None:
    _require_mapping(output, "output")
    required = (
        "metrics_dir",
        "tables_dir",
        "plots_dir",
        "raw_outputs_dir",
        "dataset_audit_path",
        "metrics_path",
        "summary_path",
        "analysis_path",
    )
    _require_keys(output, required, "output")
    _validate_path_values(output, "output", required)


def _validate_path_values(section: Mapping[str, Any], section_name: str, keys: Sequence[str]) -> None:
    for key in keys:
        value = section[key]
        if value is None:
            continue
        if not isinstance(value, str):
            raise ConfigError(f"{section_name}.{key} must be a relative path string or null")
        if Path(value).is_absolute():
            raise ConfigError(f"{section_name}.{key} must not be an absolute path")


def _iter_path_values(value: Any, prefix: str = "") -> list[tuple[str, str | Path | None]]:
    results: list[tuple[str, str | Path | None]] = []
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            dotted_key = f"{prefix}.{key}" if prefix else str(key)
            if _is_path_key(str(key)):
                results.append((dotted_key, nested_value))
            else:
                results.extend(_iter_path_values(nested_value, dotted_key))
    elif isinstance(value, list):
        for index, nested_value in enumerate(value):
            results.extend(_iter_path_values(nested_value, f"{prefix}[{index}]"))
    return results


def _is_path_key(key: str) -> bool:
    return key in PATH_KEY_NAMES or key.endswith(PATH_KEY_SUFFIXES)


def _require_mapping(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")


def _require_sequence(value: Any, name: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(f"{name} must be a list")


def _require_keys(mapping: Mapping[str, Any], keys: Sequence[str], section_name: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"Missing required key(s) in {section_name}: {', '.join(missing)}")


def _require_non_empty_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name} must be a non-empty string")


def _require_integer(value: Any, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer")


def _require_boolean(value: Any, name: str) -> None:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean")
