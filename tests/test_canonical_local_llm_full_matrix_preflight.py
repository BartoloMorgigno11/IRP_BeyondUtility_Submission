from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.canonical_local_llm_full_matrix_preflight import (
    CHUNK_SIZE,
    EXPECTED_CHUNKS,
    EXPECTED_REQUESTS,
    MANIFEST_COLUMNS,
    build_chunk_manifest,
    chunk_id_for_order,
    frame,
    hash_protected_inputs,
    validate_chunks,
    validate_manifest,
)


def _synthetic_manifest() -> pd.DataFrame:
    rows = []
    order = 1
    model_specs = [
        ("Qwen2 7B Instruct", "qwen2:7b-instruct", "dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755"),
        ("Mistral 7B Instruct", "mistral:7b-instruct", "6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf"),
    ]
    counts = {"Amazon Beauty": 990, "Amazon Sports": 892, "ML-1M": 1000}
    for model, tag, digest in model_specs:
        for task in ["ranking", "reranking"]:
            for dataset, count in counts.items():
                for idx in range(count):
                    rows.append(
                        {
                            "request_id": f"CLFULL-REQ-{order:06d}",
                            "request_order": str(order),
                            "chunk_id": chunk_id_for_order(order),
                            "model_family": model,
                            "model_tag": tag,
                            "model_digest": digest,
                            "dataset": dataset,
                            "task_type": task,
                            "candidate_set_id": f"{dataset}-{task}",
                            "user_id": f"user-{idx:04d}",
                            "source_path": "synthetic.csv",
                            "prompt_contract_id": "llm_rank_index_json_top5_v1" if task == "ranking" else "llm_rerank_index_json_top10_officialcode_v1",
                            "prompt_contract_sha256": "prompt-hash",
                            "parser_contract_id": "llm_rank_strict_primary_parser_v1" if task == "ranking" else "llm_rerank_top10_strict_primary_parser_v1",
                            "parser_contract_sha256": "parser-hash",
                            "execution_contract_id": f"exec-{model}-{task}",
                            "execution_contract_sha256": "exec-hash",
                            "candidate_order_sha256": "candidate-hash",
                            "history_render_sha256": "history-hash",
                            "candidate_count": "20",
                            "requested_output_count": "5" if task == "ranking" else "10",
                            "temperature": "0",
                            "seed": "",
                            "max_output_tokens": "256" if task == "ranking" else "512",
                            "response_format": "json_object",
                            "streaming": "false",
                            "concurrency": "1",
                            "scientific_retry_count": "0",
                            "infrastructure_retry_count": "0",
                            "primary_prompt_compatible": "true",
                            "manifest_status": "planned_not_executed",
                            "blocking_reason": "",
                            "notes": "synthetic",
                        }
                    )
                    order += 1
    return frame(rows, MANIFEST_COLUMNS)


def test_manifest_has_expected_counts_ordering_contracts_and_chunks():
    manifest = _synthetic_manifest()
    validate_manifest(manifest)
    assert len(manifest) == EXPECTED_REQUESTS
    assert manifest.iloc[0]["request_id"] == "CLFULL-REQ-000001"
    assert manifest.iloc[-1]["request_id"] == "CLFULL-REQ-011528"
    assert len(manifest[manifest["task_type"].eq("ranking")]) == 5764
    assert len(manifest[manifest["task_type"].eq("reranking")]) == 5764
    assert manifest.iloc[0]["model_family"] == "Qwen2 7B Instruct"
    assert manifest.iloc[0]["task_type"] == "ranking"
    assert manifest.iloc[2882]["task_type"] == "reranking"
    chunks = build_chunk_manifest(manifest)
    validate_chunks(chunks)
    assert len(chunks) == EXPECTED_CHUNKS
    assert chunks.iloc[0]["planned_request_count"] == str(CHUNK_SIZE)
    assert chunks.iloc[-1]["planned_request_count"] == "8"


def test_manifest_rejects_wrong_task_specific_contract_and_count():
    manifest = _synthetic_manifest()
    bad = manifest.copy()
    bad.loc[bad["task_type"].eq("ranking").idxmax(), "requested_output_count"] = "10"
    with pytest.raises(ValueError, match="requested output"):
        validate_manifest(bad)
    bad = manifest.copy()
    bad.loc[bad["task_type"].eq("reranking").idxmax(), "parser_contract_id"] = "llm_rerank_strict_primary_parser_v1"
    with pytest.raises(ValueError, match="reranking contract"):
        validate_manifest(bad)


def test_protected_hashes_detect_local_file_mutation(tmp_path):
    (tmp_path / "README.md").write_text("before\n", encoding="utf-8")
    (tmp_path / "original_repo").mkdir()
    (tmp_path / "original_repo" / "evidence.txt").write_text("x\n", encoding="utf-8")
    before = hash_protected_inputs(tmp_path)
    assert "README.md" in before
    assert "original_repo/evidence.txt" in before
    (tmp_path / "README.md").write_text("after\n", encoding="utf-8")
    after = hash_protected_inputs(tmp_path)
    assert before != after
