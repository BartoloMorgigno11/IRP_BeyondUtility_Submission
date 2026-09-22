# Final Paper/Official/Local Scientific Reconciliation

## 1. Executive verdict

The final local experimental path is complete and internally consistent:
`4800/4800` requests are accounted, `4795/4800` are valid, `5/4800` are
terminal invalid, and no final row has a non-zero retry count.

No result is classified as `PAPER_EXACT`. The local reproduction is scientifically
useful as a paper-aligned adapted reproduction, but it does not share the paper's
full model identity, protocol identity, aggregation, and evaluation population.

## 2. Experimental accounting

| phase | expected | accounted | valid | invalid | missing | unexpected | raw_json | unauthorized_retries | duplicate_request_ids | duplicate_output_destinations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R5B_CORRECTED | 1200 | 1200 | 1200 | 0 | 0 | 0 | 1200 | 0 | 0 | 0 |
| R5C_CORRECTED | 2400 | 2400 | 2400 | 0 | 0 | 0 | 2400 | 0 | 0 | 0 |
| R5D_FROZEN | 1200 | 1200 | 1195 | 5 | 0 | 0 | 1200 | 0 | 0 | 0 |
| TOTAL_FINAL | 4800 | 4800 | 4795 | 5 | 0 | 0 | 4800 | 0 | 0 | 0 |

The superseded R5B original, R5C original, and interrupted R5D run are retained
only as forensic history and are not counted as final results.

## 3. Frozen protocols/commits

| phase | decision | freeze_commit | protocol_hash | raw_namespace | paper_exact |
| --- | --- | --- | --- | --- | --- |
| R5B_CORRECTED | R5B_CORRECTED_RECOVERY_COMPLETE | 83b46454866ea61f3e05eec81d555be53e4c41e8 | 9596F36E1B1DBC15B68679DDF32114E6AE09914B418FB31DC112072384B708D5 | results/recovery/r5b_corrected | false |
| R5C_CORRECTED | R5C_CORRECTED_HISTORY_LENGTH_COMPLETE | 98a93d9892d8e9b21c9cdfa981e0ebea0a987500 | 3AAD4945AF443A7F50846BFD044C54FEFD5B77EDBD3725B1D1F1C62D829ECDEE | results/recovery/r5c_history_length_corrected | false |
| R5D_FROZEN | R5D_FROZEN_PROTOCOL_COMPLETION_VALID | ba1dbe8367216f7ea1044b2ff8db69318f8678ee | AA76CD3E936A969A57CDA2E9E54E22B73D6C1D3DB65C0795F86E181CEDFE3B1A | results/recovery/r5d_generation_involved_v2 | false |

Frozen commit provenance is fixed at:

- R5B corrected freeze: `83b46454866ea61f3e05eec81d555be53e4c41e8`.
- R5C corrected freeze: `98a93d9892d8e9b21c9cdfa981e0ebea0a987500`.
- R5D frozen completion: `ba1dbe8367216f7ea1044b2ff8db69318f8678ee`.

R5D uses `FAIL_STRICT_ALL_MANIFEST_REQUESTS_INVALID_ZERO`: terminal invalid outputs remain in the manifest
denominator and receive zero HR/NDCG credit.

## 4. R5B CandDif findings

CandDif is not the raw FIRST-RANDOM accuracy delta. The exact formula used in
the final CSV is:

```text
CandDif = log((1 - Acc_random) / (1 - Acc_first))
```

Positive CandDif means the FIRST condition outperformed RANDOM under that metric.

- CandDif-HR positive: `5/6`.
- CandDif-NDCG positive: `6/6`.
- Maximum CandDif-HR: `LastFM / qwen2:7b-instruct = 1.0986122886681102`.
- Maximum CandDif-NDCG: `LastFM / qwen2:7b-instruct = 0.694729398051804`.
- Beauty/Mistral CandDif-HR: `-0.1144103511777443`.

| dataset | model_name | HR_first | HR_random | CandDif_HR | NDCG_first | NDCG_random | CandDif_NDCG |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Beauty | mistral:7b-instruct | 0.26 | 0.34 | -0.1144103511777443 | 0.2242241792884654 | 0.1854617137594331 | 0.0487578451470065 |
| Beauty | qwen2:7b-instruct | 0.49 | 0.26 | 0.3722394604798439 | 0.4154328157675572 | 0.1446265352644804 | 0.3806664570368846 |
| LastFM | mistral:7b-instruct | 0.59 | 0.5 | 0.1984509387238382 | 0.5277551241576647 | 0.3268845933716282 | 0.3544191400142057 |
| LastFM | qwen2:7b-instruct | 0.81 | 0.43 | 1.0986122886681102 | 0.6490598039415529 | 0.297008201438006 | 0.694729398051804 |
| Sports | mistral:7b-instruct | 0.42 | 0.35 | 0.1139442593492176 | 0.3823994729415447 | 0.2127786071600896 | 0.2426676691245337 |
| Sports | qwen2:7b-instruct | 0.49 | 0.33 | 0.2728669866666401 | 0.4449587034781755 | 0.2130209442960639 | 0.3491591161090149 |

Interpretation: CandDif is robust directionally but not universal. The only
negative final CandDif cell is Beauty/Mistral HR; all NDCG CandDif cells are
positive.

## 5. R5C History-Length findings

The final R5C corrected results support history-length sensitivity, not a
universal monotonic-benefit claim.

- Long-short delta cells: `16`.
- Positive long-short deltas: `12/16`.
- Negative long-short deltas: `4/16`.
- Monotonic non-decreasing metric series: `8/16`.

| dataset | model_name | metric | history_length | value |
| --- | --- | --- | --- | --- |
| Beauty | mistral:7b-instruct | HR@5_delta_long_minus_short | 0->20 | 0.11 |
| Beauty | mistral:7b-instruct | NDCG@5_delta_long_minus_short | 0->20 | 0.0754617205108133 |
| Beauty | qwen2:7b-instruct | HR@5_delta_long_minus_short | 0->20 | 0.07 |
| Beauty | qwen2:7b-instruct | NDCG@5_delta_long_minus_short | 0->20 | 0.0454328225189375 |
| LastFM | mistral:7b-instruct | HR@5_delta_long_minus_short | 5->20 | 0.02 |
| LastFM | mistral:7b-instruct | NDCG@5_delta_long_minus_short | 5->20 | 0.0040768202233279 |
| LastFM | qwen2:7b-instruct | HR@5_delta_long_minus_short | 5->20 | -0.06 |
| LastFM | qwen2:7b-instruct | NDCG@5_delta_long_minus_short | 5->20 | -0.0344378864288988 |
| ML-1M | mistral:7b-instruct | HR@5_delta_long_minus_short | 5->20 | 0.1 |
| ML-1M | mistral:7b-instruct | NDCG@5_delta_long_minus_short | 5->20 | 0.0795204727211674 |
| ML-1M | qwen2:7b-instruct | HR@5_delta_long_minus_short | 5->20 | -0.07 |
| ML-1M | qwen2:7b-instruct | NDCG@5_delta_long_minus_short | 5->20 | -0.033850721304326 |
| Sports | mistral:7b-instruct | HR@5_delta_long_minus_short | 5->20 | 0.0199999999999999 |
| Sports | mistral:7b-instruct | NDCG@5_delta_long_minus_short | 5->20 | 0.0077964206794891 |
| Sports | qwen2:7b-instruct | HR@5_delta_long_minus_short | 5->20 | 0.08 |
| Sports | qwen2:7b-instruct | NDCG@5_delta_long_minus_short | 5->20 | 0.0402962866813216 |

The effect is both model-dependent and dataset-dependent. Qwen2 is negative on
LastFM and ML-1M for both HR and NDCG long-short deltas, while Beauty is positive
for both models and both metrics.

## 6. R5D Generation-Involved findings

R5D is fully accounted under the frozen protocol, but not all-valid. The invalids
are terminal and counted with zero credit.

- R5D valid: `795/800` ranking metric rows by dataset/model cells.
- R5D invalid: `5/800`.
- Highest HR@5: `LastFM / mistral:7b-instruct = 0.57`.
- Highest NDCG@5: `LastFM / mistral:7b-instruct = 0.3580338653159218`.

| dataset | model_name | metric | value | n_manifest | n_valid | n_invalid |
| --- | --- | --- | --- | --- | --- | --- |
| Beauty | mistral:7b-instruct | HR@5 | 0.33 | 100 | 97 | 3 |
| Beauty | mistral:7b-instruct | NDCG@5 | 0.2130390190604181 | 100 | 97 | 3 |
| Beauty | qwen2:7b-instruct | HR@5 | 0.29 | 100 | 100 | 0 |
| Beauty | qwen2:7b-instruct | NDCG@5 | 0.2105569598266048 | 100 | 100 | 0 |
| LastFM | mistral:7b-instruct | HR@5 | 0.57 | 100 | 100 | 0 |
| LastFM | mistral:7b-instruct | NDCG@5 | 0.3580338653159218 | 100 | 100 | 0 |
| LastFM | qwen2:7b-instruct | HR@5 | 0.25 | 100 | 100 | 0 |
| LastFM | qwen2:7b-instruct | NDCG@5 | 0.1588148397719527 | 100 | 100 | 0 |
| ML-1M | mistral:7b-instruct | HR@5 | 0.43 | 100 | 100 | 0 |
| ML-1M | mistral:7b-instruct | NDCG@5 | 0.2729313812146781 | 100 | 100 | 0 |
| ML-1M | qwen2:7b-instruct | HR@5 | 0.4 | 100 | 100 | 0 |
| ML-1M | qwen2:7b-instruct | NDCG@5 | 0.2836137924064225 | 100 | 100 | 0 |
| Sports | mistral:7b-instruct | HR@5 | 0.24 | 100 | 99 | 1 |
| Sports | mistral:7b-instruct | NDCG@5 | 0.1514927993781802 | 100 | 99 | 1 |
| Sports | qwen2:7b-instruct | HR@5 | 0.25 | 100 | 99 | 1 |
| Sports | qwen2:7b-instruct | NDCG@5 | 0.1451654270781114 | 100 | 99 | 1 |

Terminal invalids:

| request_id | model_name | dataset | validity_class | parse_status | output_length | expected_output_count |
| --- | --- | --- | --- | --- | --- | --- |
| R5D-RANK-59C06DDCFD78BFF4 | mistral:7b-instruct | Beauty | DUPLICATE | duplicate_indices | 10 | 10 |
| R5D-RANK-E491A3315449174D | mistral:7b-instruct | Beauty | OUT_OF_RANGE | unknown_indices | 10 | 10 |
| R5D-RANK-167AC5F3AB797DBA | mistral:7b-instruct | Beauty | OUT_OF_RANGE | unknown_indices | 10 | 10 |
| R5D-RANK-FA218507694B581F | mistral:7b-instruct | Sports | WRONG_LENGTH | invalid_length | 11 | 10 |
| R5D-RANK-94B4099CCA078776 | qwen2:7b-instruct | Sports | DUPLICATE | duplicate_indices | 10 | 10 |

## 7. Paper vs Official vs Local matrix

| Metric/Gap | Paper | Official | Local | Numeric comparable? | Direction comparable? | Classification | Limitation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R5B Beauty mistral:7b-instruct CandDif-HR | CandDif uses log((1-Acc_random)/(1-Acc_first)); positive means FIRST outperforms RANDOM. | official_exact=6; official_partial=112; no same local model identity. | -0.114410351177744 | NO | YES | DIVERGENT | Local open-model corrected protocol; same formula, but not same model identity/evaluation population as paper. |
| R5B Beauty mistral:7b-instruct CandDif-NDCG | CandDif uses log((1-Acc_random)/(1-Acc_first)); positive means FIRST outperforms RANDOM. | official_exact=6; official_partial=112; no same local model identity. | 0.048757845147006 | NO | YES | PAPER_ALIGNED_ADAPTED | Local open-model corrected protocol; same formula, but not same model identity/evaluation population as paper. |
| R5B Beauty qwen2:7b-instruct CandDif-HR | CandDif uses log((1-Acc_random)/(1-Acc_first)); positive means FIRST outperforms RANDOM. | official_exact=6; official_partial=112; no same local model identity. | 0.372239460479844 | NO | YES | PAPER_ALIGNED_ADAPTED | Local open-model corrected protocol; same formula, but not same model identity/evaluation population as paper. |
| R5B Beauty qwen2:7b-instruct CandDif-NDCG | CandDif uses log((1-Acc_random)/(1-Acc_first)); positive means FIRST outperforms RANDOM. | official_exact=6; official_partial=112; no same local model identity. | 0.380666457036885 | NO | YES | PAPER_ALIGNED_ADAPTED | Local open-model corrected protocol; same formula, but not same model identity/evaluation population as paper. |
| R5B LastFM mistral:7b-instruct CandDif-HR | CandDif uses log((1-Acc_random)/(1-Acc_first)); positive means FIRST outperforms RANDOM. | official_exact=0; official_partial=0; no same local model identity. | 0.198450938723838 | NO | YES | PAPER_ALIGNED_ADAPTED | Local open-model corrected protocol; same formula, but not same model identity/evaluation population as paper. |
| R5B LastFM mistral:7b-instruct CandDif-NDCG | CandDif uses log((1-Acc_random)/(1-Acc_first)); positive means FIRST outperforms RANDOM. | official_exact=0; official_partial=0; no same local model identity. | 0.354419140014206 | NO | YES | PAPER_ALIGNED_ADAPTED | Local open-model corrected protocol; same formula, but not same model identity/evaluation population as paper. |
| R5B LastFM qwen2:7b-instruct CandDif-HR | CandDif uses log((1-Acc_random)/(1-Acc_first)); positive means FIRST outperforms RANDOM. | official_exact=0; official_partial=0; no same local model identity. | 1.09861228866811 | NO | YES | PAPER_ALIGNED_ADAPTED | Local open-model corrected protocol; same formula, but not same model identity/evaluation population as paper. |
| R5B LastFM qwen2:7b-instruct CandDif-NDCG | CandDif uses log((1-Acc_random)/(1-Acc_first)); positive means FIRST outperforms RANDOM. | official_exact=0; official_partial=0; no same local model identity. | 0.694729398051804 | NO | YES | PAPER_ALIGNED_ADAPTED | Local open-model corrected protocol; same formula, but not same model identity/evaluation population as paper. |
| R5B Sports mistral:7b-instruct CandDif-HR | CandDif uses log((1-Acc_random)/(1-Acc_first)); positive means FIRST outperforms RANDOM. | official_exact=0; official_partial=48; no same local model identity. | 0.113944259349218 | NO | YES | PAPER_ALIGNED_ADAPTED | Local open-model corrected protocol; same formula, but not same model identity/evaluation population as paper. |
| R5B Sports mistral:7b-instruct CandDif-NDCG | CandDif uses log((1-Acc_random)/(1-Acc_first)); positive means FIRST outperforms RANDOM. | official_exact=0; official_partial=48; no same local model identity. | 0.242667669124534 | NO | YES | PAPER_ALIGNED_ADAPTED | Local open-model corrected protocol; same formula, but not same model identity/evaluation population as paper. |
| R5B Sports qwen2:7b-instruct CandDif-HR | CandDif uses log((1-Acc_random)/(1-Acc_first)); positive means FIRST outperforms RANDOM. | official_exact=0; official_partial=48; no same local model identity. | 0.27286698666664 | NO | YES | PAPER_ALIGNED_ADAPTED | Local open-model corrected protocol; same formula, but not same model identity/evaluation population as paper. |
| R5B Sports qwen2:7b-instruct CandDif-NDCG | CandDif uses log((1-Acc_random)/(1-Acc_first)); positive means FIRST outperforms RANDOM. | official_exact=0; official_partial=48; no same local model identity. | 0.349159116109015 | NO | YES | PAPER_ALIGNED_ADAPTED | Local open-model corrected protocol; same formula, but not same model identity/evaluation population as paper. |
| R5C Beauty mistral:7b-instruct HR@5_delta_long_minus_short 0->20 | History-length sensitivity is a paper LLM-specific analysis; artifacts do not establish a universal monotonic-benefit claim. | official_grid=0; 1; 10; 15; 2; 20; 3; 4; 5; grid_classification=BINNED_LOCAL_GRID; trend_agreement=PARTIAL_DIRECTION | 0.11 | NO | YES | DIRECTIONAL_COMPARISON_ONLY | Corrected local open-model grid; exact paper bins/model identities are not matched. |
| R5C Beauty mistral:7b-instruct NDCG@5_delta_long_minus_short 0->20 | History-length sensitivity is a paper LLM-specific analysis; artifacts do not establish a universal monotonic-benefit claim. | official_grid=0; 1; 10; 15; 2; 20; 3; 4; 5; grid_classification=BINNED_LOCAL_GRID; trend_agreement=PARTIAL_DIRECTION | 0.075461720510813 | NO | YES | DIRECTIONAL_COMPARISON_ONLY | Corrected local open-model grid; exact paper bins/model identities are not matched. |
| R5C Beauty qwen2:7b-instruct HR@5_delta_long_minus_short 0->20 | History-length sensitivity is a paper LLM-specific analysis; artifacts do not establish a universal monotonic-benefit claim. | official_grid=0; 1; 10; 15; 2; 20; 3; 4; 5; grid_classification=BINNED_LOCAL_GRID; trend_agreement=PARTIAL_DIRECTION | 0.07 | NO | YES | DIRECTIONAL_COMPARISON_ONLY | Corrected local open-model grid; exact paper bins/model identities are not matched. |
| R5C Beauty qwen2:7b-instruct NDCG@5_delta_long_minus_short 0->20 | History-length sensitivity is a paper LLM-specific analysis; artifacts do not establish a universal monotonic-benefit claim. | official_grid=0; 1; 10; 15; 2; 20; 3; 4; 5; grid_classification=BINNED_LOCAL_GRID; trend_agreement=PARTIAL_DIRECTION | 0.045432822518938 | NO | YES | DIRECTIONAL_COMPARISON_ONLY | Corrected local open-model grid; exact paper bins/model identities are not matched. |
| R5C LastFM mistral:7b-instruct HR@5_delta_long_minus_short 5->20 | History-length sensitivity is a paper LLM-specific analysis; artifacts do not establish a universal monotonic-benefit claim. | official_grid=no official grid; grid_classification=BINNED_LOCAL_GRID; trend_agreement=PARTIAL_DIRECTION | 0.02 | NO | YES | DIRECTIONAL_COMPARISON_ONLY | Corrected local open-model grid; exact paper bins/model identities are not matched. |
| R5C LastFM mistral:7b-instruct NDCG@5_delta_long_minus_short 5->20 | History-length sensitivity is a paper LLM-specific analysis; artifacts do not establish a universal monotonic-benefit claim. | official_grid=no official grid; grid_classification=BINNED_LOCAL_GRID; trend_agreement=PARTIAL_DIRECTION | 0.004076820223328 | NO | YES | DIRECTIONAL_COMPARISON_ONLY | Corrected local open-model grid; exact paper bins/model identities are not matched. |
| R5C LastFM qwen2:7b-instruct HR@5_delta_long_minus_short 5->20 | History-length sensitivity is a paper LLM-specific analysis; artifacts do not establish a universal monotonic-benefit claim. | official_grid=no official grid; grid_classification=BINNED_LOCAL_GRID; trend_agreement=PARTIAL_DIRECTION | -0.06 | NO | YES | DIRECTIONAL_COMPARISON_ONLY | Corrected local open-model grid; exact paper bins/model identities are not matched. |
| R5C LastFM qwen2:7b-instruct NDCG@5_delta_long_minus_short 5->20 | History-length sensitivity is a paper LLM-specific analysis; artifacts do not establish a universal monotonic-benefit claim. | official_grid=no official grid; grid_classification=BINNED_LOCAL_GRID; trend_agreement=PARTIAL_DIRECTION | -0.034437886428899 | NO | YES | DIRECTIONAL_COMPARISON_ONLY | Corrected local open-model grid; exact paper bins/model identities are not matched. |
| R5C ML-1M mistral:7b-instruct HR@5_delta_long_minus_short 5->20 | History-length sensitivity is a paper LLM-specific analysis; artifacts do not establish a universal monotonic-benefit claim. | official_grid=no official grid; grid_classification=BINNED_LOCAL_GRID; trend_agreement=PARTIAL_DIRECTION | 0.1 | NO | YES | DIRECTIONAL_COMPARISON_ONLY | Corrected local open-model grid; exact paper bins/model identities are not matched. |
| R5C ML-1M mistral:7b-instruct NDCG@5_delta_long_minus_short 5->20 | History-length sensitivity is a paper LLM-specific analysis; artifacts do not establish a universal monotonic-benefit claim. | official_grid=no official grid; grid_classification=BINNED_LOCAL_GRID; trend_agreement=PARTIAL_DIRECTION | 0.079520472721167 | NO | YES | DIRECTIONAL_COMPARISON_ONLY | Corrected local open-model grid; exact paper bins/model identities are not matched. |
| R5C ML-1M qwen2:7b-instruct HR@5_delta_long_minus_short 5->20 | History-length sensitivity is a paper LLM-specific analysis; artifacts do not establish a universal monotonic-benefit claim. | official_grid=no official grid; grid_classification=BINNED_LOCAL_GRID; trend_agreement=PARTIAL_DIRECTION | -0.07 | NO | YES | DIRECTIONAL_COMPARISON_ONLY | Corrected local open-model grid; exact paper bins/model identities are not matched. |
| R5C ML-1M qwen2:7b-instruct NDCG@5_delta_long_minus_short 5->20 | History-length sensitivity is a paper LLM-specific analysis; artifacts do not establish a universal monotonic-benefit claim. | official_grid=no official grid; grid_classification=BINNED_LOCAL_GRID; trend_agreement=PARTIAL_DIRECTION | -0.033850721304326 | NO | YES | DIRECTIONAL_COMPARISON_ONLY | Corrected local open-model grid; exact paper bins/model identities are not matched. |
| R5C Sports mistral:7b-instruct HR@5_delta_long_minus_short 5->20 | History-length sensitivity is a paper LLM-specific analysis; artifacts do not establish a universal monotonic-benefit claim. | official_grid=10; 15; 20; 5; grid_classification=BINNED_LOCAL_GRID; trend_agreement=PARTIAL_DIRECTION | 0.02 | NO | YES | DIRECTIONAL_COMPARISON_ONLY | Corrected local open-model grid; exact paper bins/model identities are not matched. |
| R5C Sports mistral:7b-instruct NDCG@5_delta_long_minus_short 5->20 | History-length sensitivity is a paper LLM-specific analysis; artifacts do not establish a universal monotonic-benefit claim. | official_grid=10; 15; 20; 5; grid_classification=BINNED_LOCAL_GRID; trend_agreement=PARTIAL_DIRECTION | 0.007796420679489 | NO | YES | DIRECTIONAL_COMPARISON_ONLY | Corrected local open-model grid; exact paper bins/model identities are not matched. |
| R5C Sports qwen2:7b-instruct HR@5_delta_long_minus_short 5->20 | History-length sensitivity is a paper LLM-specific analysis; artifacts do not establish a universal monotonic-benefit claim. | official_grid=10; 15; 20; 5; grid_classification=BINNED_LOCAL_GRID; trend_agreement=PARTIAL_DIRECTION | 0.08 | NO | YES | DIRECTIONAL_COMPARISON_ONLY | Corrected local open-model grid; exact paper bins/model identities are not matched. |
| R5C Sports qwen2:7b-instruct NDCG@5_delta_long_minus_short 5->20 | History-length sensitivity is a paper LLM-specific analysis; artifacts do not establish a universal monotonic-benefit claim. | official_grid=10; 15; 20; 5; grid_classification=BINNED_LOCAL_GRID; trend_agreement=PARTIAL_DIRECTION | 0.040296286681322 | NO | YES | DIRECTIONAL_COMPARISON_ONLY | Corrected local open-model grid; exact paper bins/model identities are not matched. |
| R5D Beauty mistral:7b-instruct PROFILE_PLUS_HISTORY HR@5 | Generation-involved profile recommendation is a paper LLM-specific analysis. | official_conditions=profile_only=888; profile_plus_history=224; remaining_generation_gaps=1 | 0.33; n_valid=97/100; n_invalid=3 | NO | NO_DIRECT_NUMERIC_BASELINE | NO_DIRECT_COMPARISON | No same-model/same-protocol official numeric cell; R5D is frozen local adapted evidence with invalid-zero denominator policy. |
| R5D Beauty mistral:7b-instruct PROFILE_PLUS_HISTORY NDCG@5 | Generation-involved profile recommendation is a paper LLM-specific analysis. | official_conditions=profile_only=888; profile_plus_history=224; remaining_generation_gaps=1 | 0.213039019060418; n_valid=97/100; n_invalid=3 | NO | NO_DIRECT_NUMERIC_BASELINE | NO_DIRECT_COMPARISON | No same-model/same-protocol official numeric cell; R5D is frozen local adapted evidence with invalid-zero denominator policy. |
| R5D Beauty qwen2:7b-instruct PROFILE_PLUS_HISTORY HR@5 | Generation-involved profile recommendation is a paper LLM-specific analysis. | official_conditions=profile_only=888; profile_plus_history=224; remaining_generation_gaps=1 | 0.29; n_valid=100/100; n_invalid=0 | NO | NO_DIRECT_NUMERIC_BASELINE | NO_DIRECT_COMPARISON | No same-model/same-protocol official numeric cell; R5D is frozen local adapted evidence with invalid-zero denominator policy. |
| R5D Beauty qwen2:7b-instruct PROFILE_PLUS_HISTORY NDCG@5 | Generation-involved profile recommendation is a paper LLM-specific analysis. | official_conditions=profile_only=888; profile_plus_history=224; remaining_generation_gaps=1 | 0.210556959826605; n_valid=100/100; n_invalid=0 | NO | NO_DIRECT_NUMERIC_BASELINE | NO_DIRECT_COMPARISON | No same-model/same-protocol official numeric cell; R5D is frozen local adapted evidence with invalid-zero denominator policy. |
| R5D LastFM mistral:7b-instruct PROFILE_PLUS_HISTORY HR@5 | Generation-involved profile recommendation is a paper LLM-specific analysis. | official_conditions=no official generation row; remaining_generation_gaps=1 | 0.57; n_valid=100/100; n_invalid=0 | NO | NO_DIRECT_NUMERIC_BASELINE | NO_DIRECT_COMPARISON | No same-model/same-protocol official numeric cell; R5D is frozen local adapted evidence with invalid-zero denominator policy. |
| R5D LastFM mistral:7b-instruct PROFILE_PLUS_HISTORY NDCG@5 | Generation-involved profile recommendation is a paper LLM-specific analysis. | official_conditions=no official generation row; remaining_generation_gaps=1 | 0.358033865315922; n_valid=100/100; n_invalid=0 | NO | NO_DIRECT_NUMERIC_BASELINE | NO_DIRECT_COMPARISON | No same-model/same-protocol official numeric cell; R5D is frozen local adapted evidence with invalid-zero denominator policy. |
| R5D LastFM qwen2:7b-instruct PROFILE_PLUS_HISTORY HR@5 | Generation-involved profile recommendation is a paper LLM-specific analysis. | official_conditions=no official generation row; remaining_generation_gaps=1 | 0.25; n_valid=100/100; n_invalid=0 | NO | NO_DIRECT_NUMERIC_BASELINE | NO_DIRECT_COMPARISON | No same-model/same-protocol official numeric cell; R5D is frozen local adapted evidence with invalid-zero denominator policy. |
| R5D LastFM qwen2:7b-instruct PROFILE_PLUS_HISTORY NDCG@5 | Generation-involved profile recommendation is a paper LLM-specific analysis. | official_conditions=no official generation row; remaining_generation_gaps=1 | 0.158814839771953; n_valid=100/100; n_invalid=0 | NO | NO_DIRECT_NUMERIC_BASELINE | NO_DIRECT_COMPARISON | No same-model/same-protocol official numeric cell; R5D is frozen local adapted evidence with invalid-zero denominator policy. |
| R5D ML-1M mistral:7b-instruct PROFILE_PLUS_HISTORY HR@5 | Generation-involved profile recommendation is a paper LLM-specific analysis. | official_conditions=no official generation row; remaining_generation_gaps=1 | 0.43; n_valid=100/100; n_invalid=0 | NO | NO_DIRECT_NUMERIC_BASELINE | NO_DIRECT_COMPARISON | No same-model/same-protocol official numeric cell; R5D is frozen local adapted evidence with invalid-zero denominator policy. |
| R5D ML-1M mistral:7b-instruct PROFILE_PLUS_HISTORY NDCG@5 | Generation-involved profile recommendation is a paper LLM-specific analysis. | official_conditions=no official generation row; remaining_generation_gaps=1 | 0.272931381214678; n_valid=100/100; n_invalid=0 | NO | NO_DIRECT_NUMERIC_BASELINE | NO_DIRECT_COMPARISON | No same-model/same-protocol official numeric cell; R5D is frozen local adapted evidence with invalid-zero denominator policy. |
| R5D ML-1M qwen2:7b-instruct PROFILE_PLUS_HISTORY HR@5 | Generation-involved profile recommendation is a paper LLM-specific analysis. | official_conditions=no official generation row; remaining_generation_gaps=1 | 0.4; n_valid=100/100; n_invalid=0 | NO | NO_DIRECT_NUMERIC_BASELINE | NO_DIRECT_COMPARISON | No same-model/same-protocol official numeric cell; R5D is frozen local adapted evidence with invalid-zero denominator policy. |
| R5D ML-1M qwen2:7b-instruct PROFILE_PLUS_HISTORY NDCG@5 | Generation-involved profile recommendation is a paper LLM-specific analysis. | official_conditions=no official generation row; remaining_generation_gaps=1 | 0.283613792406422; n_valid=100/100; n_invalid=0 | NO | NO_DIRECT_NUMERIC_BASELINE | NO_DIRECT_COMPARISON | No same-model/same-protocol official numeric cell; R5D is frozen local adapted evidence with invalid-zero denominator policy. |
| R5D Sports mistral:7b-instruct PROFILE_PLUS_HISTORY HR@5 | Generation-involved profile recommendation is a paper LLM-specific analysis. | official_conditions=profile_only=454; profile_plus_history=70; remaining_generation_gaps=1 | 0.24; n_valid=99/100; n_invalid=1 | NO | NO_DIRECT_NUMERIC_BASELINE | NO_DIRECT_COMPARISON | No same-model/same-protocol official numeric cell; R5D is frozen local adapted evidence with invalid-zero denominator policy. |
| R5D Sports mistral:7b-instruct PROFILE_PLUS_HISTORY NDCG@5 | Generation-involved profile recommendation is a paper LLM-specific analysis. | official_conditions=profile_only=454; profile_plus_history=70; remaining_generation_gaps=1 | 0.15149279937818; n_valid=99/100; n_invalid=1 | NO | NO_DIRECT_NUMERIC_BASELINE | NO_DIRECT_COMPARISON | No same-model/same-protocol official numeric cell; R5D is frozen local adapted evidence with invalid-zero denominator policy. |
| R5D Sports qwen2:7b-instruct PROFILE_PLUS_HISTORY HR@5 | Generation-involved profile recommendation is a paper LLM-specific analysis. | official_conditions=profile_only=454; profile_plus_history=70; remaining_generation_gaps=1 | 0.25; n_valid=99/100; n_invalid=1 | NO | NO_DIRECT_NUMERIC_BASELINE | NO_DIRECT_COMPARISON | No same-model/same-protocol official numeric cell; R5D is frozen local adapted evidence with invalid-zero denominator policy. |
| R5D Sports qwen2:7b-instruct PROFILE_PLUS_HISTORY NDCG@5 | Generation-involved profile recommendation is a paper LLM-specific analysis. | official_conditions=profile_only=454; profile_plus_history=70; remaining_generation_gaps=1 | 0.145165427078111; n_valid=99/100; n_invalid=1 | NO | NO_DIRECT_NUMERIC_BASELINE | NO_DIRECT_COMPARISON | No same-model/same-protocol official numeric cell; R5D is frozen local adapted evidence with invalid-zero denominator policy. |

## 8. Agreements

| family | cells | sign_agreement_yes | sign_agreement_no | numeric_comparable_yes | direction_comparable_yes | magnitude_agreement | ranking_agreement | qualitative_agreement_cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R5B | 12 | 11 | 1 | 0 | 12 | not evaluated unless same-model numeric baseline exists | not evaluated against paper; local rank observations reported in final report | 11 |
| R5C | 16 | 16 | 0 | 0 | 16 | not evaluated unless same-model numeric baseline exists | not evaluated against paper; local rank observations reported in final report | 16 |
| R5D | 16 | 0 | 0 | 0 | 0 | not evaluated unless same-model numeric baseline exists | not evaluated against paper; local rank observations reported in final report | 16 |
| TOTAL | 44 | 27 | 1 | 0 | 28 | no paper-exact magnitude comparison available | no paper-exact rank-order comparison available | 43 |

CandDif shows the strongest qualitative agreement: `11/12` final CandDif metric
cells have the expected positive sign. History-length shows sensitivity, but the
sign and monotonicity vary by model and dataset. Generation-involved performance
is valid as local adapted evidence, but lacks same-protocol paper/official
numeric cells for direct comparison.

## 9. Divergences

| result | divergence | classification | evidence | limitation |
| --- | --- | --- | --- | --- |
| R5B Beauty mistral:7b-instruct CandDif-HR | Local CandDif sign is negative where paper-style candidate-position effect expects positive sensitivity. | LOCAL_ADAPTED_DIVERGENCE | -0.114410351177744 | Local open-model corrected protocol; same formula, but not same model identity/evaluation population as paper. |
| R5C LastFM mistral:7b-instruct HR@5 | History-length curve is not monotonic non-decreasing. | LOCAL_ADAPTED_DIVERGENCE | h=5 value=0.58; h=10 value=0.67; h=20 value=0.6 | This is a divergence from a simple 'more history always helps' narrative, not necessarily from a paper claim. |
| R5C LastFM mistral:7b-instruct NDCG@5 | History-length curve is not monotonic non-decreasing. | LOCAL_ADAPTED_DIVERGENCE | h=5 value=0.379591; h=10 value=0.440101; h=20 value=0.383668 | This is a divergence from a simple 'more history always helps' narrative, not necessarily from a paper claim. |
| R5C LastFM qwen2:7b-instruct HR@5 | History-length curve is not monotonic non-decreasing. | MODEL_IDENTITY_DIFFERENCE | h=5 value=0.38; h=10 value=0.33; h=20 value=0.32 | This is a divergence from a simple 'more history always helps' narrative, not necessarily from a paper claim. |
| R5C LastFM qwen2:7b-instruct NDCG@5 | History-length curve is not monotonic non-decreasing. | MODEL_IDENTITY_DIFFERENCE | h=5 value=0.24109; h=10 value=0.199828; h=20 value=0.206653 | This is a divergence from a simple 'more history always helps' narrative, not necessarily from a paper claim. |
| R5C ML-1M qwen2:7b-instruct HR@5 | History-length curve is not monotonic non-decreasing. | MODEL_IDENTITY_DIFFERENCE | h=5 value=0.37; h=10 value=0.34; h=20 value=0.3 | This is a divergence from a simple 'more history always helps' narrative, not necessarily from a paper claim. |
| R5C ML-1M qwen2:7b-instruct NDCG@5 | History-length curve is not monotonic non-decreasing. | MODEL_IDENTITY_DIFFERENCE | h=5 value=0.24141; h=10 value=0.228567; h=20 value=0.207559 | This is a divergence from a simple 'more history always helps' narrative, not necessarily from a paper claim. |
| R5C Sports mistral:7b-instruct HR@5 | History-length curve is not monotonic non-decreasing. | LOCAL_ADAPTED_DIVERGENCE | h=5 value=0.4; h=10 value=0.39; h=20 value=0.42 | This is a divergence from a simple 'more history always helps' narrative, not necessarily from a paper claim. |
| R5C Sports mistral:7b-instruct NDCG@5 | History-length curve is not monotonic non-decreasing. | LOCAL_ADAPTED_DIVERGENCE | h=5 value=0.232748; h=10 value=0.228774; h=20 value=0.240545 | This is a divergence from a simple 'more history always helps' narrative, not necessarily from a paper claim. |
| R5D Beauty HR@5 model contrast | Large local model-dependent gap in generation-involved performance. | MODEL_IDENTITY_DIFFERENCE | Mistral-Qwen2=0.04 | No same-model paper baseline; describes local adapted behavior only. |
| R5D LastFM HR@5 model contrast | Large local model-dependent gap in generation-involved performance. | MODEL_IDENTITY_DIFFERENCE | Mistral-Qwen2=0.32 | No same-model paper baseline; describes local adapted behavior only. |
| R5D LastFM NDCG@5 model contrast | Large local model-dependent gap in generation-involved performance. | MODEL_IDENTITY_DIFFERENCE | Mistral-Qwen2=0.199219 | No same-model paper baseline; describes local adapted behavior only. |

The divergences are not post-hoc repaired. They are reported as local adapted
behavior and are mostly attributable to model identity, dataset, and protocol
differences.

## 10. Model dependence

Model identity is a dominant limitation and an observed effect. In R5B, Qwen2 has
the largest CandDif on LastFM, while Beauty/Mistral HR is the one negative
CandDif cell. In R5C, Qwen2 is negative for LastFM and ML-1M long-short deltas.
In R5D, Mistral is much stronger on LastFM, while Qwen2 is close or slightly
ahead on parts of ML-1M/Sports depending on HR vs NDCG.

## 11. Dataset dependence

Dataset dependence is substantial. Beauty shows positive history-length deltas
for both models. LastFM and ML-1M show negative long-short effects for Qwen2.
R5D is strongest on LastFM for Mistral and on ML-1M for Qwen2.

## 12. Protocol limitations

The main blocker for paper-exact reproduction is identity mismatch: local open
models, corrected JSON/index contracts, local frozen candidate/request cohorts,
and explicit invalid-output accounting differ from the paper/official execution
environment. Therefore direction and qualitative behavior can be evaluated, but
numeric paper reproduction cannot be claimed.

## 13. Forensic history

Superseded artifacts remain preserved:

- R5B original failed because the output-length contract was materially
  insufficient and produced widespread wrong-length outputs.
- R5C original had 15 Qwen2/ML-1M item-ID substitutions; post-hoc repair was not
  scientifically allowed, so R5C_CORRECTED was created.
- R5D initially stopped at 600/1200 after 3 genuine invalid outputs; frozen
  resume completed only never-executed rows and retained terminal invalids.

## 14. Final scientific conclusions

1. Main results reproduced: final local adapted CandDif, History-Length, and
   Generation-Involved recovery gaps are all accounted.
2. Directional only: history sensitivity and much of CandDif, because exact
   paper model/protocol identity is absent.
3. Not directly comparable: generation-involved numeric values have no
   same-model/same-protocol official baseline.
4. Divergent: Beauty/Mistral CandDif-HR is negative; several history curves are
   non-monotonic or negative long-short.
5. Model identity matters materially and prevents `PAPER_EXACT` claims.
6. Effects are model-dependent.
7. Effects are dataset-dependent.
8. Longer history does not universally help; it helps in `12/16` long-short
   metric cells and is monotonic in `8/16` three-point metric series.
9. CandDif is robust qualitatively but not universal: `11/12` final cells are
   positive.
10. Generation-Involved results are interpretable as local adapted evidence, with
    strongest Mistral performance on LastFM and 5 terminal invalid outputs
    handled by the strict denominator policy.
11. The local results support the paper's qualitative themes more than they
    reproduce exact numbers.
12. The principal limitation is lack of identical model+dataset+protocol+metric
    identity.

## 15. Reproducibility status

| key | value |
| --- | --- |
| final_scientific_status | LOCAL_ADAPTED_REPRODUCTION_COMPLETE_NOT_PAPER_EXACT |
| final_requests_accounted | 4800/4800 |
| valid_rate | 4795/4800 |
| terminal_invalid_rate | 5/4800 |
| unauthorized_retries | 0 |
| r5b_freeze_sha | 83b46454866ea61f3e05eec81d555be53e4c41e8 |
| r5c_freeze_sha | 98a93d9892d8e9b21c9cdfa981e0ebea0a987500 |
| r5d_freeze_sha | ba1dbe8367216f7ea1044b2ff8db69318f8678ee |
| paper_exact_cells | 0 |
| paper_aligned_adapted_cells | 11 |
| directional_only_cells | 16 |
| divergent_cells | 1 |
| no_direct_comparison_cells | 16 |
| not_comparable_cells | 0 |

Validation:

| check_id | check_name | status | observed | expected |
| --- | --- | --- | --- | --- |
| FINAL-001 | expected manifest rows | pass | 4800 | 4800 |
| FINAL-002 | accounted rows | pass | 4800 | 4800 |
| FINAL-003 | missing rows | pass | 0 | 0 |
| FINAL-004 | unexpected raw outputs | pass | 0 | 0 |
| FINAL-005 | unauthorized retries | pass | 0 | 0 |
| FINAL-006 | duplicate final request IDs | pass | 0 | 0 |
| FINAL-007 | valid count | pass | 4795 | 4795 |
| FINAL-008 | terminal invalid count | pass | 5 | 5 |
| FINAL-009 | reconciliation matrix rows | pass | 44 | 44 |
| FINAL-010 | paper exact cells zero | pass | 0 | 0 |

Global scientific status: `LOCAL_ADAPTED_REPRODUCTION_COMPLETE_NOT_PAPER_EXACT`.
More inference required: `NO` for the current local adapted experimental path;
`YES` only if the target changes to paper-exact model/protocol reproduction.
