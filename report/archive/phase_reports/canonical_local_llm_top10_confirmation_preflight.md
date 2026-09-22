# Canonical Local LLM Top-10 Reranking Confirmation Preflight

## 1. Scope and provenance

Offline Phase 5B-P1G-PREFLIGHT contract freeze using branch plan-d-extensions at commit 02a20b8b509b8cf834a8766dd1cf1f751ae204bf. The approved interpreter is .venv/Scripts/python.exe with PYTHONNOUSERSITE=1. Outputs are generated from existing repository evidence only.

## 2. Executive summary

The authoritative global prompt-compatible count is 5764 rows, not 5814. Ranking and reranking each contribute 2882 prompt-compatible rows. The frozen top-10 sample has 30 rows and the planned manifest has 60 requests.

## 3. Safety boundaries

No Ollama endpoint, LLM request, public web access, download, environment mutation, prompt v3, raw request JSONL, or raw response JSONL is created by this preflight.

## 4. Phase 5B-P1F decision

P1F selected option_b_strict_top10_confirmation after finding that the executable official-code path supports 20 candidates, 10 returned items, cutoff 5, and no complete 20-item permutation requirement.

## 5. Full candidate-layer counts

The full candidate layer contains 6000 rows across six dataset/task strata. Each dataset/task stratum contributes 1000 rows.

## 6. Candidate strict-title counts

Candidate strict-title rows total 5814. Candidate-title exclusions total 186 rows.

## 7. History-title exclusions

History-title exclusions after candidate strict-title filtering total 50 rows. This is the missing subtraction behind the P1F count discrepancy.

## 8. Prompt-compatible counts

Primary prompt-compatible rows total 5764. All source agreement checks in the reconciliation table pass.

## 9. Ranking-row count

Ranking prompt-compatible rows are 2882, yielding 5764 planned requests for two models.

## 10. Reranking-row count

Reranking prompt-compatible rows are 2882, yielding 5764 planned requests for two models.

## 11. Two-model request-count reconciliation

Complete ranking plus reranking request count is 11528 = 5764 x 2.

## 12. P1F count discrepancy

The discrepancy is classified as candidate_strict_title_count_used_before_history_exclusions: P1F used candidate strict-title counts before rendered-history title exclusions.

## 13. Official candidate-count evidence

Official evidence for 20 candidates comes from candidate20 result paths and llm_metrics.py cand_len default 20, with source rows in canonical_local_llm_reranking_top10_source_contract_audit.csv.

## 14. Official output-count evidence

Official templates and llms.py request top 10/title lists; output_refine.py and llm_metrics.py handle target length 10.

## 15. Evaluation-cutoff evidence

llm_metrics.py defines metric_len default 5 and metrics.sh delegates to that default unless overridden.

## 16. Paper-evidence limitation

The paper PDF text is not independently extracted in this phase. The strongest label used is official_code_aligned_top10_paper_pdf_text_unverified; top-10 is not called paper exact.

## 17. Top-10 task semantics

The frozen task reranks the supplied 20 candidates and requests exactly 10 ordered recommendations for later cutoff-5 evaluation.

## 18. Difference from full-20 adaptation

Prior full-20 v1/v2 contracts required a complete permutation and are retained as adapted_full_permutation artifacts, not reused here.

## 19. Top-10 prompt contract

The approved prompt contract is llm_rerank_index_json_top10_officialcode_v1 with hash d169be63f64a52e05b1da88d6c920bed823d2e6de3e8f6fd0df24eab35ecee9e and paper alignment official_code_aligned_top10_semantics_adapted_json_indices.

## 20. Common prompt across models

The manifest uses one prompt contract for Qwen2 and Mistral; model_specific_wording is false.

## 21. Candidate representation

Candidates are represented by the existing rendered candidate titles with one-based labels 1..20 and preserved candidate_order_sha256.

## 22. History representation

Histories reuse the existing Phase 5A rendered train-only chronological history, preserving history_render_sha256.

## 23. Positive-label concealment

The prompt contract sets positive_label_hidden=true; positive item IDs are carried only for audit linkage, not prompt exposure.

## 24. Top-10 output grammar

Required output is exactly one JSON object with key ranked_candidate_indices and 10 unique integer indices in 1..20.

## 25. Strict parser contract

The primary parser contract is llm_rerank_top10_strict_primary_parser_v1 with schema hash b7837a80010ab9291dd7c694255a49e69dd6f7823c5e8cc67cd687a8d6b86555; it rejects prose, code fences, duplicate indices, partial output, full-20 output, and out-of-range indices.

## 26. Diagnostic parser contract

No diagnostic salvage parser row is frozen for this phase because existing salvage policy is not reused for readiness or utility.

## 27. Execution parameters

The manifest reuses existing local execution contracts: temperature 0, non-streaming, concurrency 1, zero scientific retries, zero infrastructure retries, response_format json_object, and max_output_tokens 512 where present in the source contract.

## 28. Previous-pilot row inventory

The previous reranking union contains 60 unique row identities after deduplicating v1/v2 repeats. By dataset: {"Amazon Beauty":20,"Amazon Sports":20,"ML-1M":20}.

## 29. Previous-pilot exclusions

Every previous v1 pilot, v2 corrective pilot, and hybrid holdout row identity is excluded before deterministic selection.

## 30. Deterministic disjoint selection

Selection key is SHA256(dataset + '|' + task_type + '|' + candidate_set_id + '|' + user_id), sorted ascending within dataset, with tie-breaks by candidate_set_id and user_id. Utility, validity, positive position, history length, and popularity are not inputs.

## 31. Confirmation sample composition

Sample rows by dataset are {"Amazon Beauty":10,"Amazon Sports":10,"ML-1M":10}; total rows = 30.

## 32. Request-manifest composition

Manifest cell sizes are {"('Mistral 7B Instruct', 'Amazon Beauty')":10,"('Mistral 7B Instruct', 'Amazon Sports')":10,"('Mistral 7B Instruct', 'ML-1M')":10,"('Qwen2 7B Instruct', 'Amazon Beauty')":10,"('Qwen2 7B Instruct', 'Amazon Sports')":10,"('Qwen2 7B Instruct', 'ML-1M')":10}; total requests = 60.

## 33. Model assignments

Qwen2 uses qwen2:7b-instruct and Mistral uses mistral:7b-instruct, with digests copied from canonical_local_ollama_service_recovery.csv.

## 34. Readiness threshold

The future gate is frozen at at least 8/10 strict-valid outputs per model/dataset cell and strict-validity >= 0.80.

## 35. Utility-independence

Utility is not used for row selection, contract construction, readiness, or preflight approval.

## 36. Paper-alignment classification

Prompt JSON/index representation is official_code_aligned_top10_semantics_adapted_json_indices; source evidence alignment remains official_code_aligned_top10_paper_pdf_text_unverified.

## 37. Validation findings

22/22 validation checks pass in canonical_local_llm_top10_confirmation_preflight_validation.csv.

## 38. Protected-input immutability

59 protected input files were hashed before and after generation; the runner fails if any hash changes.

## 39. Commands executed

- `git status --short --branch`
- `git branch --show-current`
- `git log -1 --format="%H %s"`
- `$env:PYTHONNOUSERSITE='1'; .\.venv\Scripts\python.exe -c "import sys, platform, pandas, numpy; ..."`
- `$env:PYTHONNOUSERSITE='1'; .\.venv\Scripts\python.exe src\experiments\build_canonical_local_llm_top10_confirmation_preflight.py`
- `$env:PYTHONNOUSERSITE='1'; .\.venv\Scripts\python.exe -m pytest tests\test_canonical_local_llm_top10_confirmation_preflight.py ...`

## 40. Preflight decision

Decision: approve_top10_confirmation_pilot_execution. Blocking reason: none.

## 41. Recommended single next task

Phase 5B-P1G - execute the frozen 60-request official-code-aligned top-10 reranking confirmation pilot on the new disjoint sample
