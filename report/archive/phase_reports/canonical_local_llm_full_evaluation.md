# Canonical Full Local LLM Evaluation

## 1. Scope and provenance

Branch `plan-d-extensions`, commit `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests`. Environment: `python=3.12.10; executable=C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv\Scripts\python.exe; platform=Windows-11-10.0.26200-SP0; pandas=3.0.3`. Phase 5C reads frozen Phase 5B-P2 and Phase 4 evidence only.

## 2. Executive summary

All 11528 planned requests reconcile with raw request/response evidence. There are 26 strict-invalid rows, all retained. Full-cohort baseline comparisons are blocked by cohort mismatch; paired intersection diagnostics are emitted.

## 3. Safety boundaries

No Ollama, API, recommender training, package install, download, prompt generation, or invalid-output repair is performed by this builder.

## 4. Full-matrix execution evidence

Manifest 11528; raw requests 11528; raw responses 11528; runner events 23056; backend errors 0.

## 5. Output-integrity reconciliation

`results/tables/canonical_local_llm_full_output_integrity.csv` has 11528 rows and all stored/reparsed agreements are `['true']`.

## 6. Task-specific parser reconciliation

Frozen ranking and reranking strict parsers were reapplied to every raw response. Stored parse fields and raw SHA256 values agree.

## 7. Invalid-output inventory

`results/tables/canonical_local_llm_full_invalid_output_inventory.csv` records 26 invalid outputs without raw-text leakage.

## 8. Invalid-output concentration

Qwen2 7B Instruct Amazon Beauty reranking: 1; Qwen2 7B Instruct Amazon Sports reranking: 3; Qwen2 7B Instruct ML-1M reranking: 22

## 9. Evaluation-policy layers

Structural reliability, conditional strict-valid utility, adapted operational utility, conditional beyond-utility, and paired strict-valid baseline comparison are separately versioned in the policy contract.

## 10. Structural-reliability policy

Structural denominators are all planned requests in each cell. Invalid rows remain visible through duplicate, hallucination, truncation, and strict-validity rates.

## 11. Conditional-utility policy

Conditional utility uses strict-valid rows only and is diagnostic. It is not labeled unconditional model performance.

## 12. Operational-utility policy

Operational utility uses all planned rows and assigns strict-invalid HR@5/NDCG@5 to zero. It is explicitly adapted, not paper exact.

## 13. Beyond-utility denominator policy

Beyond-utility metrics use strict-valid top-5 exposures; popularity-dependent metrics require train-count support and report reduced evaluated-row coverage where needed.

## 14. Official invalid-output handling evidence

Official code evidence supports parsing and refinement behavior but does not resolve an exact denominator policy for this JSON-index protocol, so Phase 5C labels operational invalid-as-failure as adapted.

## 15. Backend reliability

Backend completion 11528/11528 and backend error rows 0.

## 16. Ranking format reliability

Ranking strict-valid rows 5764/5764; duplicate output rows 0.

## 17. Reranking format reliability

Reranking strict-valid rows 5738/5764; duplicate output rows 26.

## 18. Qwen2 format reliability

Qwen2 strict validity 5738/5764; duplicate output rows 26.

## 19. Mistral format reliability

Mistral strict validity 5764/5764; duplicate output rows 0.

## 20. Dataset-specific reliability

Amazon Beauty: 3959/3960; Amazon Sports: 3565/3568; ML-1M: 3978/4000

## 21. Conditional HR@5

Mistral 7B Instruct ML-1M ranking=0.421; Mistral 7B Instruct Amazon Beauty ranking=0.309091; Qwen2 7B Instruct ML-1M ranking=0.296

## 22. Conditional NDCG@5

Mistral 7B Instruct ML-1M ranking=0.251237; Mistral 7B Instruct Amazon Beauty ranking=0.193251; Qwen2 7B Instruct Amazon Beauty ranking=0.182245

## 23. Operational HR@5

Mistral 7B Instruct ML-1M ranking=0.421; Mistral 7B Instruct Amazon Beauty ranking=0.309091; Qwen2 7B Instruct ML-1M ranking=0.296

## 24. Operational NDCG@5

Mistral 7B Instruct ML-1M ranking=0.251237; Mistral 7B Instruct Amazon Beauty ranking=0.193251; Qwen2 7B Instruct Amazon Beauty ranking=0.182245

## 25. Conditional-versus-operational differences

Ranking and Mistral reranking reconcile exactly; Qwen2 reranking differs where duplicate-index invalid rows are counted as HR/NDCG zero operationally.

## 26. Positive-rank findings

Mistral 7B Instruct Amazon Beauty ranking mean_rank=1.80065; Qwen2 7B Instruct Amazon Beauty ranking mean_rank=1.9; Mistral 7B Instruct ML-1M ranking mean_rank=1.92637

## 27. Novelty findings

Qwen2 7B Instruct Amazon Sports reranking=14.7925; Mistral 7B Instruct Amazon Sports reranking=14.7886; Qwen2 7B Instruct Amazon Sports ranking=14.7565

## 28. Catalog-coverage findings

Mistral 7B Instruct ML-1M reranking=0.732649; Qwen2 7B Instruct ML-1M reranking=0.731839; Qwen2 7B Instruct ML-1M ranking=0.728868

## 29. Item-coverage findings

Mistral 7B Instruct ML-1M reranking=0.735445; Qwen2 7B Instruct ML-1M reranking=0.734633; Qwen2 7B Instruct ML-1M ranking=0.731654

## 30. Long-tail findings

Qwen2 7B Instruct ML-1M reranking=0.807385; Qwen2 7B Instruct Amazon Beauty reranking=0.793902; Mistral 7B Instruct Amazon Beauty reranking=0.791075

## 31. Average-popularity findings

Mistral 7B Instruct ML-1M ranking=379.7; Mistral 7B Instruct ML-1M reranking=338.528; Qwen2 7B Instruct ML-1M ranking=295.323

## 32. Duplicate, validity and hallucination findings

Duplicate-output, candidate-validity, hallucination, and strict-parse-validity rows are included in structural and beyond-utility tables; hallucination rows are zero.

## 33. Blocked beyond-utility metrics

Blocked rows: 24. Values are blank for popularity_bias_at_5 and intra_list_diversity_at_5.

## 34. Baseline cohort alignment

Alignment rows 42; exact same prompt-compatible cohorts 0; all full-cohort baseline comparisons are blocked when cohort mismatch exists.

## 35. Random comparison

Paired strict-valid diagnostic rows: 108. Full prompt-compatible rows remain blocked unless exact cohort equality is proven.

## 36. MostPop comparison

Paired strict-valid diagnostic rows: 108. Full prompt-compatible rows remain blocked unless exact cohort equality is proven.

## 37. BM25-like comparison

Paired strict-valid diagnostic rows: 108. Full prompt-compatible rows remain blocked unless exact cohort equality is proven.

## 38. BPRMF comparison

Paired strict-valid diagnostic rows: 108. Full prompt-compatible rows remain blocked unless exact cohort equality is proven.

## 39. GRU4Rec comparison

Paired strict-valid diagnostic rows: 108. Full prompt-compatible rows remain blocked unless exact cohort equality is proven.

## 40. SASRec comparison

Paired strict-valid diagnostic rows: 108. Full prompt-compatible rows remain blocked unless exact cohort equality is proven.

## 41. LightGCN comparison

Paired strict-valid diagnostic rows: 108. Full prompt-compatible rows remain blocked unless exact cohort equality is proven.

## 42. Paired bootstrap findings

Descriptive paired bootstrap intervals use seed 42 and 10000 resamples for 168 HR@5/NDCG@5 paired rows.

## 43. Qwen2 versus Mistral

Mistral has complete strict validity; Qwen2 has duplicate-index reranking failures. Utility differences are reported by conditional and adapted operational policies.

## 44. Ranking versus reranking

Ranking has complete strict validity across both models. Reranking contains all 26 strict-invalid responses, all in Qwen2.

## 45. Beauty findings

Strict-valid rows 3959/3960; duplicate rows 1. best operational HR@5 Mistral 7B Instruct ranking=0.309091.

## 46. Sports findings

Strict-valid rows 3565/3568; duplicate rows 3. best operational HR@5 Qwen2 7B Instruct ranking=0.293722.

## 47. ML-1M findings

Strict-valid rows 3978/4000; duplicate rows 22. best operational HR@5 Mistral 7B Instruct ranking=0.421.

## 48. Paper-result inventory

`results/tables/canonical_local_llm_paper_result_inventory.csv` records 11 source-traceable inventory rows with blank paper values where no verified local numeric value exists.

## 49. Baseline paper alignment

Baseline rows are reconstructed/adapted and not exact paper results because dataset artifacts, candidate cohorts, and verified paper numeric values differ or remain unresolved.

## 50. Local-LLM paper alignment

Qwen2 and Mistral are local model extensions, not paper-model reproductions. Top-10 JSON-index reranking is official-code-aligned in semantics but adapted in representation.

## 51. Reproduced findings

No exact paper numerical reproduction claim is supported by Phase 5C. The reproduced evidence is structural reconciliation of local Phase 5B-P2 outputs.

## 52. Reconstructed findings

HR@5/NDCG@5 and several beyond-utility formulas are reconstructed from official/local code under frozen Phase 4 contracts.

## 53. Adapted findings

Operational invalid-as-failure utility, local Qwen2/Mistral execution, and JSON-index prompts are adapted project conditions.

## 54. Material divergences

Material divergences include local substitute models, reconstructed datasets, strict-title prompt-compatible cohort, candidate/protocol differences, and baseline cohort mismatch.

## 55. Likely documented causes of divergence

Documented causes include missing official processed data, title exclusions, strict local JSON parsing, deterministic single-run local inference evidence, and reconstructed recommender baselines.

## 56. Claims supported by the evidence

Supported: full local output reconciliation, structural reliability rates, conditional valid-output utility, adapted operational utility, supported beyond-utility diagnostics, and intersection-only baseline diagnostics.

## 57. Claims not supported by the evidence

Unsupported: exact paper LLM reproduction, direct Qwen2/Mistral comparison to GPT/Claude/Llama paper values, full-cohort baseline pairing, fabricated paper numeric differences, and unblocked ILD/popularity-bias values.

## 58. Limitations

The analysis is offline and depends on existing reconstructed data and prior baseline artifacts. Paper numeric values remain unverified in machine-readable form.

## 59. Protected-input immutability

Protected input hashes remained unchanged: `true`.

## 60. Commands executed

Primary command: `PYTHONNOUSERSITE=1 .venv/Scripts/python.exe src/experiments/build_canonical_local_llm_full_evaluation.py`. Focused pytest commands are recorded in the final response.

## 61. Evaluation decision

`canonical_full_local_evaluation_complete`. Blocking reason: ``.

## 62. Recommended single next task

Phase 5D — consolidate LLM robustness extensions, nondeterminism, candidate-position bias, history-length sensitivity, profile experiments, and reliability findings
