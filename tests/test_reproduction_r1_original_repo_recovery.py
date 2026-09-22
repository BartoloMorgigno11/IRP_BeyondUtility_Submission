from __future__ import annotations

import csv
import zipfile
from pathlib import Path

from src.analysis import reproduction_r1_original_repo_recovery as r1


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_minimal_official_source(root: Path) -> None:
    codes = root / "eva" / "codes"
    codes.mkdir(parents=True)
    metric_calls = [
        "calHR(hit)",
        "calNDCG(hit, rank)",
        "calACLT(id_results, train_id_freq_dict, high_bar, metric_len)",
        "calARP(train_id_freq_dict, id_results, metric_len)",
        "calPopREO(hit, quantiles, train_id_freq_dict, id_targets)",
        "calSerendipity(hit, hit_pop)",
        "calSelfInfo(id_results, metric_len, information)",
        "calcFairMetric(id_results, metric_len)",
        "calDPDFreq(hit, rank, cold_user, hot_user)",
        "calJain(hit, rank)",
        "calItemCoverage(id_results, metric_len, negative_samples, test_samples, cand_len=cand_len)",
        "calMeanShannon(id_results, id2categories, metric_len)",
        "calMeanJaccard(id_results, metric_len, cand_list)",
        "calHallucination(id_results)",
    ]
    definitions = "\n".join(
        f"def {row['official_function']}(*args, **kwargs):\n    return 0\n"
        for row in r1.METRIC_CONTRACT
    )
    final_result = "final_result = [" + ",\n".join(metric_calls) + "]\n"
    (codes / "llm_metrics.py").write_text(
        definitions
        + "\nif __name__ == '__main__':\n    "
        + final_result
        + "    pd.DataFrame([final_result]).T.to_csv('x_5.csv', index=False, header=None)\n",
        encoding="utf-8",
    )
    (codes / "metrics.sh").write_text(
        "python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history1_candidate20_shuffle_sample\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("MovieLens-1M LastFM Beauty Sports\n", encoding="utf-8")


def _write_fake_zip(root: Path) -> None:
    (root / "llm_results").mkdir(parents=True)
    values = "\n".join(str(idx / 100) for idx in range(1, 15)) + "\n"
    json_payload = (
        '[{"user":"1","target":"Item A","cand_list":"[\'1\', \'2\']","prompt":"p",'
        '"result":"[\'Item A\']","template":0}]'
    )
    with zipfile.ZipFile(root / "llm_results" / "beauty.zip", "w") as archive:
        archive.writestr(
            "beauty/results_gpt3.5_seed0_test_history1_candidate20_shuffle_sample_5.csv",
            values,
        )
        archive.writestr(
            "beauty/results_gpt3.5_seed0_test_history1_candidate20_shuffle_sample.json",
            json_payload,
        )


def test_recover_metric_contract_traces_official_source(tmp_path: Path) -> None:
    original_root = tmp_path / "original_repo" / "EvaLLMasRecommender"
    _write_minimal_official_source(original_root)

    contract = r1.recover_metric_serialization_contract(original_root)

    assert len(contract) == 14
    assert [row["official_metric_name"] for row in contract][:2] == ["HR", "NDCG"]
    assert all(row["metric_semantics_status"] == "SOURCE_VERIFIED" for row in contract)
    assert contract[2]["official_metric_name"] == "ACLT"
    assert "APLT" in contract[2]["paper_alias"]


def test_recover_original_repo_results_on_synthetic_repo(tmp_path: Path) -> None:
    project_root = tmp_path
    original_root = project_root / "original_repo" / "EvaLLMasRecommender"
    _write_minimal_official_source(original_root)
    _write_fake_zip(original_root)

    outputs = r1.recover_original_repo_results(project_root=project_root)
    decoded = _read_csv(outputs.table_dir / "reproduction_r1_original_repo_metric_values_decoded.csv")
    json_rows = _read_csv(outputs.table_dir / "reproduction_r1_official_json_inventory.csv")
    coverage = _read_csv(outputs.table_dir / "reproduction_r1_metric_family_coverage_matrix.csv")
    dataset_classification = _read_csv(
        outputs.table_dir / "reproduction_r1_original_dataset_evidence_classification.csv"
    )

    assert outputs.summary["official_metric_csvs_found"] == 1
    assert outputs.summary["decoded_metric_values"] == 14
    assert {row["decode_status"] for row in decoded} == {"FULLY_DECODED"}
    assert {row["source_member_sha256"] for row in decoded}
    assert all(len(row["source_member_sha256"]) == 64 for row in decoded)
    assert json_rows[0]["has_cand_list"] == "YES"
    assert {row["dataset"] for row in coverage} == {"Beauty", "Sports", "ML-1M", "LastFM"}
    assert {
        row["official_evidence_classification"]
        for row in dataset_classification
        if row["dataset"] in {"ML-1M", "LastFM"}
    } == {"REFERENCE_ONLY"}


def test_generated_r1_artifacts_are_provenance_complete() -> None:
    decoded_path = RESULTS_DIR / "reproduction_r1_original_repo_metric_values_decoded.csv"
    contract_path = RESULTS_DIR / "reproduction_r1_metric_serialization_contract.csv"
    inventory_path = RESULTS_DIR / "reproduction_r1_official_artifact_inventory.csv"
    coverage_path = RESULTS_DIR / "reproduction_r1_metric_family_coverage_matrix.csv"

    assert decoded_path.exists()
    assert contract_path.exists()
    assert inventory_path.exists()
    assert coverage_path.exists()

    contract = _read_csv(contract_path)
    decoded = _read_csv(decoded_path)
    inventory = _read_csv(inventory_path)
    coverage = _read_csv(coverage_path)

    assert len(contract) == 14
    assert all(row["metric_semantics_status"] == "SOURCE_VERIFIED" for row in contract)
    assert decoded
    assert all(row["source_member"] for row in decoded)
    assert all(len(row["source_member_sha256"]) == 64 for row in decoded)
    assert {row["decode_status"] for row in decoded} == {"FULLY_DECODED"}
    assert any(row["artifact_type"] == "llm_result_json" for row in inventory)
    assert {row["dataset"] for row in coverage} == {"Beauty", "Sports", "ML-1M", "LastFM"}


def test_generated_r1_decoded_source_members_exist() -> None:
    decoded = _read_csv(RESULTS_DIR / "reproduction_r1_original_repo_metric_values_decoded.csv")
    checked_sources: set[tuple[str, str]] = set()

    for row in decoded:
        key = (row["source_archive"], row["source_member"])
        if key in checked_sources:
            continue
        checked_sources.add(key)
        if row["source_archive"]:
            archive_path = PROJECT_ROOT / row["source_archive"]
            assert archive_path.exists()
            with zipfile.ZipFile(archive_path) as archive:
                assert row["source_member"] in archive.namelist()
        else:
            assert Path(row["source_member"]).exists()
