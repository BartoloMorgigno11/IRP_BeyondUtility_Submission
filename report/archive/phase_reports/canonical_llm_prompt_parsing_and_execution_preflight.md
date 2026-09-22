# Canonical LLM Prompt, Parsing and Execution Contract Preflight

## 1. Scope and provenance
- Branch: `plan-d-extensions`.
- Commit: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests`.
- Environment: `executable=C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe;python=3.12.10;pandas=3.0.3;numpy=2.5.0;tokenizer_libs={"sentencepiece":false,"tiktoken":false,"tokenizers":false,"transformers":false};cwd=C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project;cpu_only=true`.
- Protected input files hashed: `110`.

## 2. Executive summary
- Full Plan E2 LLM candidate layer: `6000` rows.
- Candidate strict-title rows: `5814` with dataset counts `{'Amazon Beauty': 1988, 'Amazon Sports': 1826, 'ML-1M': 2000}`.
- Prompt-render-valid rows after rendered-history title checks: `5764`.
- No LLM execution, API call, Ollama invocation, or new recommendation result was produced.

## 3. Safety boundaries
- Phase 5A freezes contracts only. It writes contract/audit tables and does not create request or response files.

## 4. Completed baseline and metric layers
- Phase 4B metric matrices are treated as protected inputs and are not regenerated here.

## 5. Paper LLM experiment inventory
- Paper families represented: GPT-3.5, GPT-4o-mini, GPT-4o, Claude 3 Haiku, and Llama3-70B-Instruct.

## 6. Official prompt inventory
- Official templates under `original_repo/EvaLLMasRecommender/eva/templates/` request title-based recommendation lists, often top 10.

## 7. Local prompt inventory
- Plan D local prompts use strict JSON and candidate IDs for bounded diagnostics.

## 8. Prompt-definition conflicts
- Official title-list prompts conflict with local strict JSON ID prompts; Phase 5A freezes an adapted index-JSON contract.

## 9. Ranking task semantics
- Ranking uses train-only chronological history plus the persisted 20 candidates and requests top-5 ordered candidate indices.

## 10. Reranking task semantics
- Reranking treats the supplied candidate order as an initial order and requests a full 20-index permutation.

## 11. User-history contract
- History source is train split only, ordered by timestamp then source row, with held-out positive excluded from rendered history.

## 12. Candidate-rendering contract
- Candidate rendering preserves the exact 20-item order and displays one-based index labels with titles.

## 13. Candidate-order preservation
- Candidate order hashes are recorded for every strict-title prompt audit row.

## 14. Positive-label concealment
- Positive-label exposure checks passed for `5764` valid prompt renderings.

## 15. Ranking prompt contract
- `llm_rank_index_json_top5_v1` is ready as an adapted contract.

## 16. Reranking prompt contract
- `llm_rerank_index_json_full20_v1` is ready as an adapted contract.

## 17. Generated-profile prompt contract
- Profile generation remains blocked/ambiguous and is audit-only.

## 18. Output-format contract
- Required output is a strict JSON object with `ranked_candidate_indices`.

## 19. Strict parser contract
- Strict parser rows: `2`.

## 20. Diagnostic salvage-parser contract
- Salvage parser rows are diagnostic only and cannot overwrite strict primary results.

## 21. Invalid-output taxonomy
- Parse validity, candidate validity, output length, duplicate indices, and unknown indices are separate conditions.

## 22. Candidate validity
- Every output index must map to one of the supplied 20 candidates.

## 23. Hallucination definition
- Primary hallucination means an output index outside the supplied candidate set or an item not recoverable from candidate membership.

## 24. Duplicate-output handling
- Duplicates invalidate strict output; no duplicate removal is permitted for primary results.

## 25. Retry policy
- Scientific retry count is frozen to 0; infrastructure retries are also 0 in the frozen contract.

## 26. Model inventory
- Model inventory rows: `7`.

## 27. Paper models
- Closed/API and large paper models are represented but not executable in this offline phase.

## 28. Local extension models
- Locally tested extension models allowed for future execution: `['Qwen2 7B Instruct', 'Mistral 7B Instruct']`.

## 29. Closed-API conditional models
- GPT and Claude rows are conditional or blocked because no provider query is allowed in Phase 5A.

## 30. Retired or unavailable models
- Claude 3 Haiku is recorded as blocked/conditional from local Plan D registry evidence.

## 31. Hardware-blocked models
- Llama3-70B-Instruct is blocked for local execution on the project target hardware.

## 32. Execution-parameter contract
- Execution parameter rows: `14`.

## 33. Deterministic adapted settings
- Local adapted execution uses temperature 0, JSON response format, no streaming, concurrency 1, and raw-response retention.

## 34. Canonical LLM cohort
- Full rows: `6000`; strict candidate-title rows: `5814`; prompt-compatible rows: `5764`.

## 35. Title-coverage exclusions
- Candidate title exclusions reconcile to 186 rows; rendered-history title exclusions are recorded separately.

## 36. Prompt rendering validation
- Prompt rendering audit rows: `5814`; valid renderings: `5764`.

## 37. Existing-response parser audit
- Existing parser audit rows: `180`; existing responses were not regenerated.

## 38. Canonical execution matrix
- Execution matrix rows: `50`.

## 39. Request-count inventory
- Planned request count across all represented cells: `40708`. This is a preflight count only.

## 40. Token-envelope estimates
- Estimated token envelope total across represented cells: `12717295` to `25414356` using character/whitespace estimates.

## 41. Nondeterminism extension
- Exactly one result-independent bounded-repeat policy is recommended.

## 42. Candidate-position-bias extension
- Position bias is a separate diagnostic layer with fixed permutations, not part of canonical single-order utility.

## 43. History-length extension
- History-length diagnostics are separate from canonical execution and use fixed variants.

## 44. Generated-profile extension
- Profile generation remains blocked until prompt, model, reuse, parser, and leakage policies are frozen.

## 45. Paper-aligned execution cells
- Paper-model cells are included as blocked or conditional; no paper-exact LLM execution is ready offline.

## 46. Adapted local execution cells
- Qwen2 and Mistral local cells are contract-ready but not executed.

## 47. Blocked execution cells
- Blocked/conditional issues recorded: `8`; affected model families include `['GPT-3.5', 'GPT-4o-mini', 'GPT-4o', 'Claude 3 Haiku', 'Llama3-70B-Instruct']`.

## 48. Required raw-response retention
- Future execution must retain raw responses, parsed outputs, attempt metadata, and prompt hashes.

## 49. Reproducibility manifest requirements
- Future manifests must include model identifier, prompt/parser/execution contract IDs, row IDs, hashes, and attempt number.

## 50. Cost-analysis boundary
- No monetary cost table or current price estimate is produced in Phase 5A.

## 51. Exact-reproduction limitations
- Exact paper prompts, provider availability, retry semantics, and title-list parsing remain insufficiently aligned for exact reproduction.

## 52. Policy decisions
- Recommended policy IDs: `[{'ambiguity_category': 'prompt_source', 'policy_id': 'prompt_adapted_strict_json'}, {'ambiguity_category': 'candidate_representation', 'policy_id': 'candidate_one_based_index_title'}, {'ambiguity_category': 'output_format', 'policy_id': 'output_json_indices'}, {'ambiguity_category': 'parser', 'policy_id': 'parser_strict_plus_diagnostic_salvage'}, {'ambiguity_category': 'retry', 'policy_id': 'retry_no_scientific_retry'}, {'ambiguity_category': 'generation', 'policy_id': 'generation_deterministic_adapted'}, {'ambiguity_category': 'cohort', 'policy_id': 'cohort_prompt_compatible_strict_title'}, {'ambiguity_category': 'nondeterminism', 'policy_id': 'nondet_fixed_bounded_repeats'}, {'ambiguity_category': 'position_bias', 'policy_id': 'position_deterministic_bounded'}]`.

## 53. Execution stop conditions
- Stop on candidate mutation, positive-label exposure, parser ambiguity, provider retry hiding, or protected-input mutation.

## 54. Tests
- Synthetic tests cover prompt registry, history rendering, candidate rendering, parsers, request counts, token estimates, and policy rules.

## 55. Determinism validation
- The builder is deterministic when output hashes match across consecutive runs.

## 56. Protected-input immutability
- Protected hashes are compared before and after writing Phase 5A outputs.

## 57. Commands executed
- Exact commands are recorded in the final response.

## 58. Whether Phase 5A succeeded
- Phase 5A succeeds when schemas validate, counts reconcile, tests pass, output hashes are deterministic, and protected inputs remain unchanged.

## 59. LLM execution readiness assessment
- Local Qwen2 and Mistral execution is contract-ready for prompt-compatible rows, with candidate/history title exclusions documented.

## 60. Recommended single next task
Phase 5B — canonical local Qwen2 and Mistral ranking/reranking execution on frozen strict-title cohorts
