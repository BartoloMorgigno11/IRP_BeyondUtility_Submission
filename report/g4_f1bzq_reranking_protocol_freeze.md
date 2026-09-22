# G4 F1BZQ Reranking Protocol Freeze

Decision token: `G4_F1BZQ_RERANKING_PROTOCOL_FROZEN`

## Scientific Status

- G4 is an adapted extension reranking experiment, not paper-exact.
- F1BZQ is the sole authoritative candidate-pool source.
- Superseded F1BR/F1BZC pools and old F6M reranking responses are excluded.

## Universe

- Logical requests: `29292`.
- Cells: `30`.
- Response reuse: `0`.
- Future Ollama calls: `29292`.
- Candidate hash rows: `4882`.
- Dataset user counts: `{"beauty":990,"foodcom":1000,"lastfm":1000,"ml1m":1000,"sports":892}`.

## Contract

- Task: `reranking`.
- History: `visible_last_10_or_available`.
- Candidate pool: `BALANCED_RANK_ROUND_ROBIN_UNIQUE_20`, 20 items, no positive forcing.
- Output: exactly `10` unique 1-based candidate indices.
- Parser: `llm_rank_strict_index_top10_v1`.

## Candidate Recall

metric_id,dataset,pool_policy,denominator,evaluation_N,positive_present_N,positive_absent_N,candidate_recall_at_20,force_positive,target_identity_used_for_ranking,candidate_recall_tautological,source,status
candidate_recall_at_20,beauty,ADAPTED_BALANCED_MULTI_MODEL_RERANKING_POOL,all_users,990,49,941,0.049494949494949494,false,false,false,results/tables/f1bzq_candidate_recall.csv,FROZEN_FROM_F1BZQ
candidate_recall_at_20,sports,ADAPTED_BALANCED_MULTI_MODEL_RERANKING_POOL,all_users,892,28,864,0.031390134529147982,false,false,false,results/tables/f1bzq_candidate_recall.csv,FROZEN_FROM_F1BZQ
candidate_recall_at_20,ml1m,ADAPTED_BALANCED_MULTI_MODEL_RERANKING_POOL,all_users,1000,193,807,0.193,false,false,false,results/tables/f1bzq_candidate_recall.csv,FROZEN_FROM_F1BZQ
candidate_recall_at_20,lastfm,ADAPTED_BALANCED_MULTI_MODEL_RERANKING_POOL,all_users,1000,469,531,0.46899999999999997,false,false,false,results/tables/f1bzq_candidate_recall.csv,FROZEN_FROM_F1BZQ
candidate_recall_at_20,foodcom,ADAPTED_BALANCED_MULTI_MODEL_RERANKING_POOL,all_users,1000,4,996,0.0040000000000000001,false,false,false,results/tables/f1bzq_candidate_recall.csv,FROZEN_FROM_F1BZQ

## Metrics

- Included metric rows: `33`.
- Excluded metric rows: `5`.
- CandDif, PopREO, DPD, monetary marginal cost, and unimplemented metrics are excluded.

## Statistics

- Statistical comparison rows: `75`.
- Internal comparisons use paired-user bootstrap with 10000 resamples, seed 20260901, and two-sided 95% percentile CI.
- P-values are not frozen and Holm correction is not applied.
- G1 vs G4 is descriptive pipeline-level comparison only.

## Zero-Call Dry Run

design,logical_requests,exact_response_reuse,future_ollama_calls,cells,existing_terminal_g4_evidence_before,existing_terminal_g4_evidence_after,request_jsonl_rows_before,request_jsonl_rows_after,llm_inference_calls_executed_in_this_invocation,dry_run,status
G4_F1BZQ_ADAPTED_RERANKING,29292,0,29292,30,24727,29292,24727,29292,4565,false,EXECUTION_ATTEMPT_RECORDED

## Validation

- Validation checks: `31`.
- Failures: `0`.

G4_F1BZQ_RERANKING_PROTOCOL_FROZEN
