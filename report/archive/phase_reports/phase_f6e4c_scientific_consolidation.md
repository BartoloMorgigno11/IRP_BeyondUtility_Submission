# Phase F6E4C Scientific Consolidation

## 1. Initial repository state

{"branch":"plan-d-extensions","directory":"C:\\Users\\Bartolo\\Desktop\\PROGETTI\\IRP_BeyondUtility_Project","head":"02a20b8b509b8cf834a8766dd1cf1f751ae204bf","last_commit":"02a20b8 Add Plan E3B BM25-like scoring on E2 manifests","python_executable":"C:\\Users\\Bartolo\\Desktop\\PROGETTI\\IRP_BeyondUtility_Project\\.venv\\Scripts\\python.exe","python_version":"3.12.10","staged_files":"","timestamp_policy":"not_recorded_in_deterministic_freeze","worktree_policy":"full git status recorded in final operator response"}

## 2. Frozen-input verification

Input rows: 21; protected={"changed":0,"checked":2312,"missing":0}.

## 3. Raw F6E4B hashes and counts

{"results/llm/phase_f6e4b_mistral_adapted_full_run_raw.jsonl":"ad5477e832ae745cfcc881efc7d0a9e11c6bc4c97d5df33d6b83be3e1d9e86f1","results/llm/phase_f6e4b_qwen2_adapted_full_run_raw.jsonl":"090deed07de41786f707ddcb9f6ceaf656992a756b32cfea80ab4eef7ff24e96"}

## 4. Request-level reconstruction

{"decision":"f6e4c_scientific_consolidation_complete_with_corrected_derived_metrics","invalid_outputs":1437,"requests":30492,"strict_valid":29055}

## 5. Metric-contract audit

Rows: 161; all checks pass.

## 6. Operational/conditional reconciliation

Operational metrics use all logical requests with invalid outputs set to zero; conditional metrics use strict-valid rows only.

## 7. Superseded F6E4B metrics

Superseded rows: 8.

## 8. Corrected authoritative metrics

F6E4C request-level and summary tables are authoritative for the adapted full run.

## 9. Invalid-output audit

Rows: 1437.

## 10. Mistral LLMRank LastFM diagnosis

{"audit_id":"F6E4C-MISTRAL-LLMRANK-LASTFM","candidate_position_distribution":"{\"positive_11_20\":927,\"positive_6_10\":435,\"positive_top5\":497}","classified_cause":"core_adapter_instruction_conflict","conditional_hr_at5":"0.532058492688414","conditional_ndcg_at5":"0.397937468140233","generated_tokens_mean":"66.611081226465842","history_length_distribution":"{\"history_6_10\":1846,\"history_le_5\":13}","invalid":"970","invalid_rate":"0.521785906401291","invalid_reason_distribution":"{\"duplicate_indices\":4,\"extra_text_or_code_fence\":76,\"invalid_length\":890}","latency_mean":"2.376025302420755","logical_requests":"1859","notes":"No prompt is modified in F6E4C; this is a diagnostic classification only.","operational_hr_at5":"0.254437869822485","operational_ndcg_at5":"0.190299305635647","output_length_distribution":"{\"0\":76,\"10\":4,\"11\":10,\"19\":25,\"20\":813,\"21\":39,\"23\":1,\"24\":1,\"34\":1}","percent_output_length_10":"0.480365788058096","percent_output_length_20":"0.43733189887036","prompt_adapter_conflict_evidence":"Many invalid outputs have length 20 while the adapted JSON contract requires top-10 indices only.","strict_valid":"889"}

## 11. Cell summaries

Rows: 8.

## 12. Dataset/task summaries

Rows: 28.

## 13. Base-vs-LLMRank paired comparison

Rows: 64; bootstrap=10000; seed=42.

## 14. Qwen2-vs-Mistral paired comparison

Rows: 64; bootstrap=10000; seed=42.

## 15. History-length analysis

Rows: 48; classification observational_history_length_analysis.

## 16. Candidate-position analysis

Rows: 84; classification adapted_candidate_position_analysis.

## 17. Exact-vs-adapted evidence

Rows: 4.

## 18. Research-question evidence map

Rows: 10.

## 19. Claim registry

Rows: 6.

## 20. Implementation files

src/analysis/plan_f6e4c_scientific_consolidation.py and CLI builder.

## 21. Tests added

tests/test_plan_f6e4c_scientific_consolidation.py.

## 22. Targeted regression results

Recorded in final operator response after execution.

## 23. Excluded unsafe tests

Inference, training CUDA, download, URL/API tests are excluded.

## 24. Determinism

Builder output is byte-deterministic when run twice on the same frozen inputs.

## 25. Protected-input verification

{"changed":0,"checked":2312,"missing":0}

## 26. Scientific freeze manifest

results/manifests/phase_f6e4c_scientific_freeze.json

## 27. Missing or blocked evidence

Rows: 3.

## 28. Final decision

f6e4c_scientific_consolidation_complete_with_corrected_derived_metrics

## 29. Recommended next action

keep_current_matrix_frozen

## 30. Final git state

Recorded in final operator response; no staging or commit is performed.

## 31. Files created or modified

Only F6E4C source, CLI, tests, report, tables and manifest.

## 32. Remaining limitations

Adapted local run; not paper-exact; LastFM reranking remains excluded.
