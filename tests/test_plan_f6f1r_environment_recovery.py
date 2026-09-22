from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.analysis import plan_f6f1r_environment_recovery as f6f1r


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_minimal_config(root: Path) -> None:
    (root / "configs").mkdir()
    (root / "results" / "tables").mkdir(parents=True)
    (root / "results" / "manifests").mkdir(parents=True)
    (root / "report").mkdir()
    (root / "configs" / "f6f1.yaml").write_text(
        "\n".join(
            [
                "outputs:",
                "  report: report/phase_f6f1_beyond_utility_metrics.md",
                "  freeze_manifest: results/manifests/phase_f6f1_beyond_utility_freeze.json",
                "  tables:",
                "    metric_contract: results/tables/phase_f6f1_metric_contract.csv",
            ]
        ),
        encoding="utf-8",
    )
    (root / "results" / "manifests" / "phase_f6f1_beyond_utility_freeze.json").write_text("{}", encoding="utf-8")


def test_import_without_side_effects() -> None:
    assert f6f1r.PHASE_ID == "phase_f6f1r"
    assert f6f1r.MANIFEST_PATH.name == "phase_f6f1r_environment_freeze.json"


def test_no_network_imports() -> None:
    tree = ast.parse(Path(f6f1r.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not {"requests", "httpx", "socket", "urllib"}.intersection(imported)


def test_no_ollama_request_import() -> None:
    tree = ast.parse(Path(f6f1r.__file__).read_text(encoding="utf-8"))
    imported = {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert "ollama" not in imported


def test_no_inference_training_or_cuda_imports() -> None:
    tree = ast.parse(Path(f6f1r.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
    assert "torch" not in imported
    assert "tensorflow" not in imported


def test_broken_venv_audit_does_not_modify_launchers(tmp_path: Path) -> None:
    scripts = tmp_path / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    (tmp_path / ".venv" / "pyvenv.cfg").write_text("home = C:\\Users\\x\\AppData\\Local\\Microsoft\\WindowsApps\n", encoding="utf-8")
    launcher = scripts / "python.exe"
    launcher.write_bytes(b"launcher")
    (scripts / "pythonw.exe").write_bytes(b"launcherw")
    before = _sha(launcher)
    f6f1r.audit_broken_venv(tmp_path)
    assert _sha(launcher) == before


def test_official_f6f1_outputs_not_modified_by_path_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_minimal_config(tmp_path)
    monkeypatch.setattr(f6f1r, "F6F1_CONFIG", Path("configs/f6f1.yaml"))
    monkeypatch.setattr(f6f1r, "OFFICIAL_F6F1_MANIFEST", Path("results/manifests/phase_f6f1_beyond_utility_freeze.json"))
    before = _sha(tmp_path / "results" / "manifests" / "phase_f6f1_beyond_utility_freeze.json")
    with pytest.raises(ValueError):
        f6f1r.reject_official_output_root(tmp_path, Path("results/tables"))
    assert _sha(tmp_path / "results" / "manifests" / "phase_f6f1_beyond_utility_freeze.json") == before


def test_official_f6e4c_outputs_are_read_only_by_integrity_rows() -> None:
    before = {"f": {"artifact": "f6e4c", "phase": "F6E4C", "path": "f", "exists": True, "sha256": "a"}}
    rows = f6f1r.build_input_integrity_rows(before, before)
    assert rows[0]["changed"] == "false"
    assert rows[0]["missing"] == "false"


def test_python_inventory_schema_for_missing_interpreter(tmp_path: Path) -> None:
    row, _ = f6f1r.inspect_interpreter(tmp_path / "missing-python.exe", tmp_path, 1)
    assert set(f6f1r.PYTHON_INVENTORY_COLUMNS).issubset(row)
    assert row["alias_or_stub_classification"] == "missing"


def test_interpreter_stub_classification() -> None:
    path = Path(r"C:\Example\AppData\Local\Microsoft\WindowsApps\python.exe")
    assert f6f1r.classify_interpreter_path(path, True, 999, "", 0) == "app_execution_alias_stub"


def test_dependency_contract_schema_without_python(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("numpy>=1.23\npandas>=1.5\n", encoding="utf-8")
    rows, _ = f6f1r.build_dependency_contract(tmp_path, None)
    assert rows
    assert set(f6f1r.DEPENDENCY_COLUMNS).issubset(rows[0])


def test_redacts_credential_like_environment_variables() -> None:
    redacted = f6f1r.redact_environment({"OPENAI_API_KEY": "secret", "PATH": "plain"})
    assert redacted["OPENAI_API_KEY"] == "<redacted>"
    assert redacted["PATH"] == "plain"


def test_output_root_guard_rejects_official_report_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_minimal_config(tmp_path)
    monkeypatch.setattr(f6f1r, "F6F1_CONFIG", Path("configs/f6f1.yaml"))
    monkeypatch.setattr(f6f1r, "OFFICIAL_F6F1_MANIFEST", Path("results/manifests/phase_f6f1_beyond_utility_freeze.json"))
    with pytest.raises(ValueError):
        f6f1r.reject_official_output_root(tmp_path, Path("report"))


def test_frozen_output_path_rejection_for_manifest_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_minimal_config(tmp_path)
    monkeypatch.setattr(f6f1r, "F6F1_CONFIG", Path("configs/f6f1.yaml"))
    monkeypatch.setattr(f6f1r, "OFFICIAL_F6F1_MANIFEST", Path("results/manifests/phase_f6f1_beyond_utility_freeze.json"))
    with pytest.raises(ValueError):
        f6f1r.reject_official_output_root(tmp_path, Path("results/manifests"))


def test_isolated_output_config_writing_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_minimal_config(tmp_path)
    monkeypatch.setattr(f6f1r, "F6F1_CONFIG", Path("configs/f6f1.yaml"))
    monkeypatch.setattr(f6f1r, "OFFICIAL_F6F1_MANIFEST", Path("results/manifests/phase_f6f1_beyond_utility_freeze.json"))
    config = f6f1r.isolated_f6f1_config(tmp_path, Path("results/environment/isolated"))
    assert config["outputs"]["report"].startswith("results/environment/isolated/")
    assert config["outputs"]["tables"]["metric_contract"].startswith("results/environment/isolated/")


def test_scientific_artifact_comparison_byte_identical(tmp_path: Path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("a,b\n1,2.5\n", encoding="utf-8")
    right.write_text("a,b\n1,2.5\n", encoding="utf-8")
    result = f6f1r.compare_csv_files(left, right)
    assert result["semantic"] is True
    assert result["numeric_equal"] is True
    assert result["max_diff"] == 0.0


def test_runtime_metadata_separation_classification() -> None:
    row = f6f1r.comparison_row(
        "freeze_manifest",
        "json",
        Path("official.json"),
        Path("isolated.json"),
        "a",
        "b",
        False,
        True,
        {"row_count": "", "column_count": "", "schema_equal": "", "numeric_equal": "", "max_diff": ""},
        "scientifically_identical_runtime_metadata_differs",
        "runtime only",
    )
    assert row["mismatch_classification"] == "scientifically_identical_runtime_metadata_differs"


def test_write_csv_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    rows = [{"a": "2", "b": "z"}, {"a": "1", "b": "a"}]
    f6f1r.write_csv(path, rows, ["a", "b"])
    first = _sha(path)
    f6f1r.write_csv(path, list(reversed(rows)), ["a", "b"])
    assert _sha(path) == first


def test_protected_input_invariance_detects_change() -> None:
    before = {"x": {"artifact": "protected", "phase": "F2", "path": "x", "exists": True, "sha256": "a"}}
    after = {"x": {"artifact": "protected", "phase": "F2", "path": "x", "exists": True, "sha256": "b"}}
    rows = f6f1r.build_input_integrity_rows(before, after)
    assert rows[0]["changed"] == "true"
    assert rows[0]["classification"] == "fail"


def test_missing_or_blocked_records_system_site_fallback() -> None:
    rows = f6f1r.build_missing_or_blocked("reconstructed_venv_with_system_site_packages", [], [], [])
    assert any(row["classification"] == "not_fully_isolated" for row in rows)


def test_final_decision_allows_runtime_metadata_difference() -> None:
    decision = f6f1r.final_decision(
        "isolated_local_venv",
        [{"status": "pass"}],
        [{"status": "pass"}],
        [{"mismatch_classification": "scientifically_identical_runtime_metadata_differs"}],
        [{"classification": "pass"}],
        {"scientific_content_hash": f6f1r.FROZEN_F6F1_HASH},
    )
    assert decision == "f6f1r_environment_recovery_complete_with_runtime_metadata_differences"


def test_recommended_next_action_is_exact_value() -> None:
    assert f6f1r.recommended_next_action("f6f1r_blocked_missing_local_dependencies", "blocked") == "reconstruct_local_dependency_bundle"


def test_csv_comparison_detects_scientific_difference(tmp_path: Path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("a,b\n1,2\n", encoding="utf-8")
    right.write_text("a,b\n1,3\n", encoding="utf-8")
    result = f6f1r.compare_csv_files(left, right)
    assert result["semantic"] is False
    assert result["numeric_equal"] is False
