# F1BZQ Authoritative Reranking Pool Materialization

Decision token: `F1BZQ_AUTHORITATIVE_RERANKING_POOLS_VALIDATED`

## Source Scoring

- Source cell statuses: `{"COMPUTED":20}`.
- User/model source cases: `19528`.
- Source universe: raw catalog intersected with frozen checkpoint item vocabulary, minus legitimate pre-test seen items.
- Source seed policy: all five F1BZ seeds are mean-scored; no final-test seed selection is used.

## Corrected Pools

- Pool statuses: `{"AUTHORITATIVE_POOL_COMPUTED":4882}`.
- Pool users: `4882`.
- Pool candidate rows: `97640`.
- Contribution totals: `{"BPRMF_contribution":31436,"GRU4Rec_contribution":27837,"LightGCN_contribution":18396,"SASRec_contribution":19971}`.
- First-model 20,0,0,0 pools: `0`.
- Pool construction imports the frozen F1BZP balanced rank round-robin helper.

## Candidate Recall

dataset,pool_policy,evaluation_N,positive_present_N,positive_absent_N,candidate_recall_at_20,positive_rank_mean,positive_rank_median,candidate_recall_tautological,force_positive,target_identity_used_for_ranking,status,reason
beauty,ADAPTED_BALANCED_MULTI_MODEL_RERANKING_POOL,990,49,941,0.049494949494949494,7.9387755102040813,6,false,false,false,COMPUTED,Candidate Recall@20 computed after corrected pool construction; targets were not used for ranking.
sports,ADAPTED_BALANCED_MULTI_MODEL_RERANKING_POOL,892,28,864,0.031390134529147982,7.9285714285714288,8,false,false,false,COMPUTED,Candidate Recall@20 computed after corrected pool construction; targets were not used for ranking.
ml1m,ADAPTED_BALANCED_MULTI_MODEL_RERANKING_POOL,1000,193,807,0.193,8.4559585492227978,8,false,false,false,COMPUTED,Candidate Recall@20 computed after corrected pool construction; targets were not used for ranking.
lastfm,ADAPTED_BALANCED_MULTI_MODEL_RERANKING_POOL,1000,469,531,0.46899999999999997,5.6503198294243067,3,false,false,false,COMPUTED,Candidate Recall@20 computed after corrected pool construction; targets were not used for ranking.
foodcom,ADAPTED_BALANCED_MULTI_MODEL_RERANKING_POOL,1000,4,996,0.0040000000000000001,9.25,8.5,false,false,false,COMPUTED,Candidate Recall@20 computed after corrected pool construction; targets were not used for ranking.

## Food.com Warm/Cold

dataset,scope,evaluation_N,positive_present_N,positive_absent_N,candidate_recall_at_20,candidate_recall_tautological,force_positive,status,reason
foodcom,ALL_USERS,1000,4,996,0.0040000000000000001,false,false,COMPUTED,Food.com warm/cold split preserved from the frozen final-training history contract.
foodcom,WARM_USERS,273,4,269,0.014652014652014652,false,false,COMPUTED,Food.com warm/cold split preserved from the frozen final-training history contract.
foodcom,TRUE_ZERO_HISTORY_USERS,727,0,727,0,false,false,COMPUTED,Food.com warm/cold split preserved from the frozen final-training history contract.

## Reranking Readiness

- Readiness statuses: `{"READY_TO_INFER":30}`.
- Ready rows: `30` / `30`.
- Ready LLM requests: `29292` / `29292`.
- LLM inference executed: `0`.

## Supersession

- Supersession rows: `5`.
- Old tautological and single-model diagnostic pool artifacts remain preserved but non-authoritative.

## Validation

- Pool validation checks: `18`; failures `0`.
- Final validation checks: `31`; failures `0`.
