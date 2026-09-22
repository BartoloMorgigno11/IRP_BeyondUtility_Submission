"""Project scaffold smoke test.

Run from the repository root:

    python src/experiments/exp00_smoke_test.py
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path


LOGGER = logging.getLogger("exp00_smoke_test")


@dataclass(frozen=True)
class CheckResult:
    """Result for one smoke-test check."""

    name: str
    passed: bool
    critical: bool
    message: str

    @property
    def status(self) -> str:
        if self.passed:
            return "PASS"
        if self.critical:
            return "FAIL"
        return "WARN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run project scaffold smoke checks.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config path to check if provided.",
    )
    return parser.parse_args()


def check_path(repo_root: Path, relative_path: str, description: str, critical: bool = True) -> CheckResult:
    path = repo_root / relative_path
    exists = path.exists()
    return CheckResult(
        name=description,
        passed=exists,
        critical=critical,
        message=f"{relative_path} {'exists' if exists else 'is missing'}",
    )


def check_directory(
    repo_root: Path,
    relative_path: str,
    description: str,
    critical: bool = True,
) -> CheckResult:
    path = repo_root / relative_path
    passed = path.is_dir()
    return CheckResult(
        name=description,
        passed=passed,
        critical=critical,
        message=f"{relative_path} {'is a directory' if passed else 'is missing or not a directory'}",
    )


def check_import(module_name: str, critical: bool = True) -> CheckResult:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - message is what matters for CLI smoke checks
        return CheckResult(
            name=f"Import {module_name}",
            passed=False,
            critical=critical,
            message=f"could not import {module_name}: {exc}",
        )
    return CheckResult(
        name=f"Import {module_name}",
        passed=True,
        critical=critical,
        message=f"imported {module_name}",
    )


def build_checks(repo_root: Path, config_path: Path | None) -> list[CheckResult]:
    checks: list[CheckResult] = []
    current_dir = Path.cwd().resolve()

    checks.append(
        CheckResult(
            name="Run location",
            passed=current_dir == repo_root,
            critical=True,
            message=f"cwd={current_dir}; expected={repo_root}",
        )
    )

    expected_dirs = [
        ("src", "Source root"),
        ("src/data", "Data package"),
        ("src/baselines", "Baselines package"),
        ("src/prompts", "Prompts package"),
        ("src/evaluation", "Evaluation package"),
        ("src/experiments", "Experiments package"),
        ("src/utils", "Utils package"),
        ("configs", "Configs folder"),
        ("data", "Data folder"),
        ("data/raw", "Raw data folder"),
        ("data/processed", "Processed data folder"),
        ("data/samples", "Samples folder"),
        ("results", "Results folder"),
        ("results/raw_outputs", "Raw outputs folder"),
        ("results/metrics", "Metrics folder"),
        ("results/tables", "Tables folder"),
        ("results/plots", "Plots folder"),
        ("tests", "Tests folder"),
        ("report", "Report folder"),
        ("original_repo", "Original repo parent"),
        ("original_repo/EvaLLMasRecommender", "Original repository checkout"),
        ("paper", "Paper folder"),
    ]
    checks.extend(check_directory(repo_root, path, label) for path, label in expected_dirs)

    expected_files = [
        ("README.md", "Repository README"),
        ("requirements.txt", "Project requirements"),
        ("report/codebase_map.md", "Original repo codebase map"),
    ]
    checks.extend(check_path(repo_root, path, label) for path, label in expected_files)

    checks.append(
        check_path(
            repo_root,
            "paper/2411.00331v1.pdf",
            "Paper PDF",
            critical=False,
        )
    )

    if config_path is not None:
        absolute_config = (repo_root / config_path).resolve() if not config_path.is_absolute() else config_path
        checks.append(
            CheckResult(
                name="Optional config file",
                passed=absolute_config.is_file(),
                critical=False,
                message=f"{config_path} {'exists' if absolute_config.is_file() else 'is not available yet'}",
            )
        )

    src_modules = [
        "src",
        "src.data",
        "src.baselines",
        "src.prompts",
        "src.evaluation",
        "src.experiments",
        "src.utils",
    ]
    checks.extend(check_import(module_name) for module_name in src_modules)

    return checks


def log_results(checks: list[CheckResult]) -> None:
    for result in checks:
        LOGGER.info("%-4s | %-30s | %s", result.status, result.name, result.message)

    passed = sum(result.passed for result in checks)
    failed = sum((not result.passed) and result.critical for result in checks)
    warnings = sum((not result.passed) and (not result.critical) for result in checks)

    LOGGER.info("Summary: %s passed, %s failed, %s warnings", passed, failed, warnings)
    if failed:
        LOGGER.error("Smoke test FAILED")
    else:
        LOGGER.info("Smoke test PASSED")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    checks = build_checks(repo_root=repo_root, config_path=args.config)
    log_results(checks)

    critical_failures = [result for result in checks if not result.passed and result.critical]
    return 1 if critical_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
