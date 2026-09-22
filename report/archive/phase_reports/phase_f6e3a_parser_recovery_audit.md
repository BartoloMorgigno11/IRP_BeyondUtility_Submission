# Phase F6E3A - Exact-Prompt Output Contract And Parser Recovery Audit

Decision: `f6e3a_parser_recovery_complete_gate_failed_model_outputs`.

## 1. Initial repository state

Captured in the operator response to keep builder output deterministic; F6E3A builder itself does not call git.

## 2. Frozen-input verification

F6E3 raw hashes matched frozen values. Protected summary: {"f2_changed":0,"f2_checked":1144,"f6d4a_groups_checked":4,"f6d4a_total_changed":0}.

## 3. Raw-output inventory

Frozen F6E3 responses analyzed: 32; no new requests sent.

## 4. Observed format classification

{"numbered_candidate_titles|bullet_candidate_titles|mixed_titles_and_reasoning|partial_ranking":1,"numbered_candidate_titles|bullet_candidate_titles|quoted_candidate_titles|mixed_titles_and_reasoning":1,"numbered_candidate_titles|bullet_candidate_titles|quoted_candidate_titles|mixed_titles_and_reasoning|partial_ranking":3,"numbered_candidate_titles|bullet_candidate_titles|quoted_candidate_titles|mixed_titles_and_reasoning|partial_ranking|candidate_metadata_serialization_delimiter_collision":1,"numbered_candidate_titles|quoted_candidate_titles|mixed_titles_and_reasoning":5,"numbered_candidate_titles|quoted_candidate_titles|mixed_titles_and_reasoning|partial_ranking":18,"numbered_candidate_titles|quoted_candidate_titles|mixed_titles_and_reasoning|partial_ranking|candidate_metadata_serialization_delimiter_collision":3}

## 5. Prompt-output contract audit

Rendered prompts were compared against upstream system-output requirements and observed response formats.

## 6. Upstream parser discovery

Upstream evidence is original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py; replay classification is upstream_adapter_functional.

## 7. Upstream parser replay

Upstream-compatible strict valid: 0/32.

## 8. Parser ladder

Current strict 0/32; normalized strict 0/32; permissive diagnostic recovered 0/32.

## 9. Candidate-title ambiguity

{"unique":32}

## 10. Hallucination audit

{"ambiguous_candidate_mapping":2,"genuine_candidate_hallucination":6,"not_hallucination_format_or_incomplete":24}

## 11. Qwen2 Base analysis

logical=8; current=0; upstream=0; normalized=0; permissive=0; causes={"genuine_candidate_hallucination":1,"incomplete_model_output":7}

## 12. Mistral Base analysis

logical=8; current=0; upstream=0; normalized=0; permissive=0; causes={"genuine_candidate_hallucination":2,"incomplete_model_output":6}

## 13. Qwen2 LLMRank analysis

logical=8; current=0; upstream=0; normalized=0; permissive=0; causes={"candidate_title_ambiguity":1,"genuine_candidate_hallucination":3,"incomplete_model_output":4}

## 14. Mistral LLMRank analysis

logical=8; current=0; upstream=0; normalized=0; permissive=0; causes={"candidate_title_ambiguity":1,"incomplete_model_output":7}

## 15. Root-cause attribution

{"candidate_title_ambiguity":2,"genuine_candidate_hallucination":6,"incomplete_model_output":24}

## 16. Parser correction

No F6E3 parser correction is applied by F6E3A unless strict-compatible recovery is demonstrated; permissive recovery remains diagnostic.

## 17. Tests added

tests/test_plan_f6e3a_parser_recovery_audit.py covers offline purity, parser ladder, raw invariance and deterministic writes.

## 18. Analysis-only regeneration

F6E3A writes only phase_f6e3a_* report, tables and manifest.

## 19. Gate reassessment

{"current_local_original":"fail","normalized_strict_compatible":"fail","permissive_diagnostic_not_gate_eligible":"not_gate_eligible","upstream_compatible_strict":"fail"}

## 20. Determinism

Builder uses frozen timestamps and sorted/hash-stable table construction.

## 21. Targeted tests

Recorded in operator response after execution.

## 22. Full-suite result

Recorded in operator response after execution.

## 23. Protected-input verification

{"f2_changed":0,"f2_checked":1144,"f6d4a_groups_checked":4,"f6d4a_total_changed":0}

## 24. Parser-recovery manifest

Scientific content hash: a060787682876c73a036b760fefa744ef81d5004d1ae67e63b54b0bac0f40c58.

## 25. Missing or blocked evidence

{"documented_candidate_metadata_problem":1,"f6e3a_parser_recovery_complete_gate_failed_model_outputs":1,"not_executed_by_policy":1}

## 26. Final decision

`f6e3a_parser_recovery_complete_gate_failed_model_outputs`.

## 27. Recommended next action

`run_adapted_output_contract_pilot`.

## 28. Final git state

Captured in operator response; no staging or commit is performed by the builder.

## 29. Files created or modified

F6E3A creates only report/phase_f6e3a_parser_recovery_audit.md, results/tables/phase_f6e3a_*.csv, and results/manifests/phase_f6e3a_parser_recovery_freeze.json.

## 30. Remaining limitations

No upstream batch execution with original data paths is performed; the adapter is functional and candidate-scoped for frozen F6E3 rows.
