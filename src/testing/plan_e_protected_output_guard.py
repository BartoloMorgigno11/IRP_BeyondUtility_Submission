from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ORIGINAL_RUN = subprocess.run
_PATCHED = False
LAST_REDIRECTS: list[Path] = []


def pytest_configure(config: Any) -> None:
    global _PATCHED
    if _PATCHED:
        return
    subprocess.run = _guarded_run  # type: ignore[assignment]
    _PATCHED = True


def pytest_unconfigure(config: Any) -> None:
    global _PATCHED
    if not _PATCHED:
        return
    subprocess.run = _ORIGINAL_RUN  # type: ignore[assignment]
    _PATCHED = False


def _guarded_run(args: Any, *popenargs: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    runner = _plan_e_runner_from_args(args)
    if runner is None:
        return _ORIGINAL_RUN(args, *popenargs, **kwargs)

    if "PYTEST_CURRENT_TEST" not in os.environ:
        return _ORIGINAL_RUN(args, *popenargs, **kwargs)

    output_root = Path(tempfile.mkdtemp(prefix="irp_plan_e_pytest_outputs_"))
    LAST_REDIRECTS.append(output_root)
    try:
        stdout_text = _run_plan_e_builder_to_temp(runner, output_root)
        return _completed(args, kwargs, stdout_text, "")
    except Exception as exc:  # pragma: no cover - exercised only on guarded builder failure.
        completed = _completed(args, kwargs, "", f"{type(exc).__name__}: {exc}")
        completed.returncode = 1
        if kwargs.get("check"):
            raise subprocess.CalledProcessError(
                completed.returncode,
                args,
                output=completed.stdout,
                stderr=completed.stderr,
            ) from exc
        return completed


def _plan_e_runner_from_args(args: Any) -> str | None:
    if isinstance(args, (str, bytes)):
        return None
    try:
        command = list(args)
    except TypeError:
        return None
    if len(command) < 2:
        return None

    runner = Path(str(command[1])).resolve()
    known = {
        PROJECT_ROOT / "src/experiments/build_plan_e_baseline_alignment.py",
        PROJECT_ROOT / "src/experiments/build_plan_e_bm25_scoring.py",
        PROJECT_ROOT / "src/experiments/build_plan_e_full_data_inventory.py",
    }
    if runner in known:
        return runner.name
    return None


def _run_plan_e_builder_to_temp(runner_name: str, output_root: Path) -> str:
    if runner_name == "build_plan_e_baseline_alignment.py":
        from src.analysis.plan_e_baseline_alignment import (
            PHASE_ID,
            PHASE_TITLE,
            PlanEBaselineAlignmentPaths,
            build_all_tables,
            write_baseline_alignment_outputs,
        )

        tables = output_root / "results" / "tables"
        report = output_root / "report"
        paths = PlanEBaselineAlignmentPaths(
            manifest_inventory=tables / "plan_e_baseline_alignment_manifest_inventory.csv",
            source_inventory=tables / "plan_e_baseline_alignment_source_inventory.csv",
            score_availability=tables / "plan_e_baseline_alignment_score_availability.csv",
            no_training_metrics=tables / "plan_e_baseline_alignment_no_training_metrics.csv",
            missing_baselines=tables / "plan_e_baseline_alignment_missing_baselines.csv",
            validation=tables / "plan_e_baseline_alignment_validation.csv",
            next_actions=tables / "plan_e_baseline_alignment_next_actions.csv",
            report=report / "plan_e_baseline_alignment.md",
        )
        write_baseline_alignment_outputs(build_all_tables(PROJECT_ROOT), paths)
        return _stdout(PHASE_ID, PHASE_TITLE, paths.report)

    if runner_name == "build_plan_e_bm25_scoring.py":
        from src.analysis.plan_e_bm25_scoring import (
            PHASE_ID,
            PHASE_TITLE,
            PlanEBM25ScoringPaths,
            build_all_tables,
            write_bm25_scoring_outputs,
        )

        tables = output_root / "results" / "tables"
        report = output_root / "report"
        paths = PlanEBM25ScoringPaths(
            source_inventory=tables / "plan_e_bm25_source_inventory.csv",
            query_corpus_audit=tables / "plan_e_bm25_query_corpus_audit.csv",
            scores=tables / "plan_e_bm25_scores.csv",
            metrics=tables / "plan_e_bm25_metrics.csv",
            validation=tables / "plan_e_bm25_validation.csv",
            missing_or_blocked=tables / "plan_e_bm25_missing_or_blocked.csv",
            next_actions=tables / "plan_e_bm25_next_actions.csv",
            report=report / "plan_e_bm25_scoring.md",
        )
        write_bm25_scoring_outputs(build_all_tables(PROJECT_ROOT), paths)
        return _stdout(PHASE_ID, PHASE_TITLE, paths.report)

    if runner_name == "build_plan_e_full_data_inventory.py":
        from src.analysis.plan_e_full_data_inventory import (
            PHASE_ID,
            PHASE_TITLE,
            PlanEInventoryPaths,
            build_all_tables,
            write_inventory_outputs,
        )

        tables = output_root / "results" / "tables"
        report = output_root / "report"
        paths = PlanEInventoryPaths(
            dataset_summary=tables / "plan_e_full_data_inventory_dataset_summary.csv",
            artifact_summary=tables / "plan_e_full_data_inventory_artifact_summary.csv",
            manifest_summary=tables / "plan_e_full_data_inventory_manifest_summary.csv",
            metric_summary=tables / "plan_e_full_data_inventory_metric_summary.csv",
            raw_output_hygiene=tables / "plan_e_full_data_inventory_raw_output_hygiene.csv",
            scaleup_readiness=tables / "plan_e_full_data_inventory_scaleup_readiness.csv",
            call_estimates=tables / "plan_e_full_data_inventory_call_estimates.csv",
            runtime_estimates=tables / "plan_e_full_data_inventory_runtime_estimates.csv",
            next_actions=tables / "plan_e_full_data_inventory_next_actions.csv",
            report=report / "plan_e_full_data_inventory.md",
        )
        write_inventory_outputs(build_all_tables(PROJECT_ROOT), paths)
        return _stdout(PHASE_ID, PHASE_TITLE, paths.report)

    raise ValueError(f"Unsupported guarded Plan E runner: {runner_name}")


def _stdout(phase_id: str, phase_title: str, report: Path) -> str:
    return "\n".join(
        (
            f"{phase_id} {phase_title}",
            "Guarded pytest execution: redirected to temporary output root.",
            "No inference, no Ollama calls, no API calls, no downloads, no model pulls, no training.",
            f"report: {report.as_posix()}",
            "",
        )
    )


def _completed(args: Any, kwargs: dict[str, Any], stdout_text: str, stderr_text: str) -> subprocess.CompletedProcess[Any]:
    wants_text = bool(kwargs.get("text") or kwargs.get("universal_newlines") or kwargs.get("encoding"))
    stdout: str | bytes = stdout_text if wants_text else stdout_text.encode()
    stderr: str | bytes = stderr_text if wants_text else stderr_text.encode()
    return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr=stderr)
