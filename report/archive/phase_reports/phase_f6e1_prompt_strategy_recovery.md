# F6E1 - Prompt Strategy Recovery, Formalization and Evidence Freeze

## 1. Initial repository state
Repository state is recorded by the final operator command log; F6E1 artifacts are generated without staging or commit.

## 2. Local source discovery
Inventoried local prompt/parser sources: 23.
Discovery covers upstream templates, upstream prompt generation/evaluation code, local JSON/index contracts, F6A/F6B/F6D3 audit tables, local paper artifacts, and frozen local request evidence.

## 3. Prompt source inventory
Authoritative table: results/tables/phase_f6e1_prompt_source_inventory.csv.

## 4. Base prompt recovery
Strategy paper_base_openp5_parametric_v1 is classified as exact_parametric: official template variants are present in original_repo/EvaLLMasRecommender/eva/templates/template.txt.
The output contract remains title-list based in upstream code; exact OpenP5 lineage is not asserted.

## 5. LLMRank recovery
Strategy paper_llmrank_recency_parametric_v1 is classified as exact_parametric: llmrank.txt and llmrank_rf.txt are present locally.
The recency-focused source includes last_item and is frozen literally, including its wording inconsistency.

## 6. LLMSRec recovery
Strategy paper_llmsrec_reconstructed_v1 is classified as reconstructed.
Literal components exist for in-context prompting and profile generation/injection, but full canonical orchestration is not proven as one exact local prompt.

## 7. JSON/index local ablation audit
Strategy local_json_index_ablation_v1 is classified as local_ablation.
It uses strict JSON one-based candidate indices and is not promoted to an exact paper prompt.

## 8. Placeholder contract
Placeholder rows frozen: 68.

## 9. Parser contract
Parser rows frozen: 6. Strict validity is separate from hallucination/candidate validity.

## 10. Model-strategy matrix
Matrix rows frozen: 11: paper-aligned core, cross-model extension, and local ablation cells.

## 11. Scientific exactness classification
OpenP5-derived base prompt=exact_parametric, LLMRank recency-focused=exact_parametric, LLMSRec in-context/profile strategy=reconstructed, Local JSON/index prompt=local_ablation

## 12. Implementation files
- src/llm/plan_f6e1_prompt_strategies.py
- src/experiments/build_phase_f6e1_prompt_strategy_recovery.py
- tests/test_plan_f6e1_prompt_strategies.py

## 13. Tests added
F6E1 tests check import purity, network/client absence, deterministic rendering, placeholder/parser contracts, blocked strategy behavior, raw hash invariance, and deterministic builder output.

## 14. Targeted test results
Recorded in final operator response after pytest execution.

## 15. Full-suite result
Recorded in final operator response after pytest execution.

## 16. Determinism
Manifest scientific content hash: a6c5298d2bb68f47f36d29315e1bc849e2e33ed576fa02ba4329abe8ecbe09ea. Timestamp is excluded from this hash.

## 17. Protected-input verification
Protected-input and raw hash checks are read-only and recorded in final operator response.

## 18. Prompt freeze manifest
Authoritative manifest: results/manifests/phase_f6e1_prompt_freeze.json.

## 19. Missing or blocked evidence
Blocked/missing evidence rows frozen: 9.

## 20. Final decision
f6e1_prompt_strategy_freeze_complete_with_documented_blocks.

## 21. Final git state
Recorded in final operator response; no staging or commit performed by F6E1.

## 22. Files created or modified
Only F6E1 implementation, test, report, CSV tables and manifest are part of this phase.

## 23. Remaining limitations
No inference was run; paper-exact model versions, exact LLMSRec orchestration, exact OpenP5 lineage, and Llama3 70B runtime feasibility remain blocked or conditional.

## 24. Safe next action
Use the frozen matrix to design a separate runtime preflight before any F6E2/F6E execution.
