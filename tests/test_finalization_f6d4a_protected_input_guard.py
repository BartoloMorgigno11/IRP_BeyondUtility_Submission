from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from src.testing import plan_e_protected_output_guard as guard


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_F2_OUTPUTS = (
    PROJECT_ROOT / "report/plan_e_bm25_scoring.md",
    PROJECT_ROOT / "report/plan_e_full_data_inventory.md",
    PROJECT_ROOT / "results/tables/plan_e_baseline_alignment_source_inventory.csv",
    PROJECT_ROOT / "results/tables/plan_e_bm25_source_inventory.csv",
    PROJECT_ROOT / "results/tables/plan_e_full_data_inventory_artifact_summary.csv",
    PROJECT_ROOT / "results/tables/plan_e_full_data_inventory_dataset_summary.csv",
    PROJECT_ROOT / "results/tables/plan_e_full_data_inventory_manifest_summary.csv",
    PROJECT_ROOT / "results/tables/plan_e_full_data_inventory_metric_summary.csv",
    PROJECT_ROOT / "results/tables/plan_e_full_data_inventory_raw_output_hygiene.csv",
    PROJECT_ROOT / "results/tables/plan_e_full_data_inventory_scaleup_readiness.csv",
)


def test_plan_e_builder_subprocess_guard_redirects_outputs_and_preserves_f2_inputs() -> None:
    before = _hashes(PROTECTED_F2_OUTPUTS)

    for runner in (
        PROJECT_ROOT / "src/experiments/build_plan_e_baseline_alignment.py",
        PROJECT_ROOT / "src/experiments/build_plan_e_bm25_scoring.py",
        PROJECT_ROOT / "src/experiments/build_plan_e_full_data_inventory.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(runner)],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert "no inference" in completed.stdout.lower()
        assert "temporary output root" in completed.stdout.lower()

    assert _hashes(PROTECTED_F2_OUTPUTS) == before
    assert guard.LAST_REDIRECTS
    for output_root in guard.LAST_REDIRECTS[-3:]:
        assert output_root.exists()
        assert not output_root.is_relative_to(PROJECT_ROOT)


def _hashes(paths: tuple[Path, ...]) -> dict[str, str]:
    return {path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path) for path in paths}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
