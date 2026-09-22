# Canonical Local LLM Top-10 Reranking Confirmation Pilot

## 1. Scope and provenance

Phase 5B-P1G executed the frozen 60-request top-10 reranking confirmation manifest using C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe.

## 2. Executive summary

Completed 60/60 requests. Cells passing readiness: 6/6. Decision: approve_full_local_execution_with_ranking_v1_reranking_top10.

## 3. Safety boundaries

Only local loopback Ollama was used. No downloads, prompt revisions, ranking requests, full matrix execution, or prior-artifact rewrites were performed.

## 4. Phase 5B-P1F protocol decision

P1F approved official-code-aligned top-10 confirmation after rejecting full-20 as paper-exact.

## 5. Phase 5B-P1G preflight

The preflight approved execution with 30 disjoint rows and 60 planned requests.

## 6. Corrected prompt-compatible counts

Counts were verified at 5,764 prompt-compatible rows, with 2,882 ranking and 2,882 reranking rows.

## 7. Corrected full request count

The corrected future two-model full request count is 11,528.

## 8. Official-code top-10 evidence

Official-code evidence supports 20 candidates, 10 returned items, and cutoff 5.

## 9. Paper-evidence limitation

Paper top-10 wording remains not directly text-verified; no paper-exact claim is made.

## 10. Top-10 alignment classification

official_code_aligned_top10_semantics_adapted_json_indices

## 11. Frozen common prompt

Prompt contract llm_rerank_index_json_top10_officialcode_v1 was used for both models.

## 12. Frozen strict parser

Strict parser llm_rerank_top10_strict_primary_parser_v1 was used; no salvage parser entered readiness.

## 13. Difference from full-20 adaptation

The pilot does not require a complete 20-item permutation and does not execute full-20 prompts.

## 14. Confirmation population

Population is the frozen prompt-compatible reranking layer only.

## 15. Previous-pilot exclusions

Previous-pilot overlap count: 0.

## 16. Deterministic confirmation sample

The sample was not regenerated; stored sample_order and hashes were preserved.

## 17. Confirmation sample composition

10 Beauty, 10 Sports, and 10 ML-1M rows.

## 18. Frozen request manifest

60 requests, 30 per model, 10 per model/dataset cell.

## 19. Local Ollama verification

Ollama server version: 0.31.1.

## 20. Qwen2 digest verification

qwen2:7b-instruct digest dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755.

## 21. Mistral digest verification

mistral:7b-instruct digest 6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf.

## 22. Resume-safe execution

Completed requests are skipped on rerun; ambiguous or duplicate evidence aborts execution.

## 23. Raw-request retention

Raw request records retained: 60.

## 24. Raw-response retention

Raw response/error records retained: 60.

## 25. Backend completion

Backend completion rate: 1; backend error records: 0.

## 26. Latency and token findings

Mean duration seconds by cell and generated-token medians are in results/tables/canonical_local_llm_top10_confirmation_metrics.csv.

## 27. Strict-parser findings

Strict-valid rows: 59/60; minimum cell validity rate: 0.9.

## 28. JSON and schema findings

JSON-valid rows: 60; schema-valid rows: 60.

## 29. Output-count findings

Output-count-valid rows: 60.

## 30. Index-range findings

Index-range-valid rows: 60.

## 31. Uniqueness findings

Uniqueness-valid rows: 59.

## 32. Candidate-validity findings

Candidate-valid rows: 59.

## 33. Duplicate and hallucination findings

Duplicate total: 1; hallucination total: 0.

## 34. Truncation assessment

Output truncation detected: false.

## 35. Qwen2 Beauty readiness

10/10 strict-valid, rate 1, readiness true.

## 36. Qwen2 Sports readiness

10/10 strict-valid, rate 1, readiness true.

## 37. Qwen2 ML-1M readiness

9/10 strict-valid, rate 0.9, readiness true.

## 38. Mistral Beauty readiness

10/10 strict-valid, rate 1, readiness true.

## 39. Mistral Sports readiness

10/10 strict-valid, rate 1, readiness true.

## 40. Mistral ML-1M readiness

10/10 strict-valid, rate 1, readiness true.

## 41. Utility diagnostics

Utility-evaluated strict-valid rows: 59; values are diagnostic only.

## 42. Why utility does not determine readiness

Readiness uses structural validity, candidate validity, duplicate-free and hallucination-free checks only.

## 43. Readiness threshold

Frozen threshold: 0.80; at least 8/10 strict-valid per cell.

## 44. Top-10 protocol generalization assessment

Generalization to full execution is approve_full_local_execution_with_ranking_v1_reranking_top10 based on six frozen cells.

## 45. Full-execution decision

approve_full_local_execution_with_ranking_v1_reranking_top10

## 46. Approved prompt combination

Ranking: llm_rank_index_json_top5_v1; reranking: llm_rerank_index_json_top10_officialcode_v1.

## 47. Full-execution request count

11528

## 48. Required corrective action

none

## 49. Limitations

This is a structural confirmation pilot; utility diagnostics are not final performance and beyond-utility metrics were not computed.

## 50. Protected-input immutability

Protected hashes were checked before and after execution; runner fails on mutation.

## 51. Commands executed

- `git status --short --branch`
- `git branch --show-current`
- `git log -1 --format="%H %s"`
- `$env:PYTHONNOUSERSITE='1'; & 'C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe' -m pytest tests\test_canonical_local_llm_top10_confirmation_runner.py ...`
- `$env:PYTHONNOUSERSITE='1'; & 'C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe' src\experiments\run_canonical_local_llm_top10_confirmation_pilot.py`

## 52. Recommended single next task

Phase 5B-P2 - execute the full canonical local Qwen2 and Mistral matrix using ranking prompt v1 and official-code-aligned top-10 reranking on 5,764 frozen prompt-compatible rows
