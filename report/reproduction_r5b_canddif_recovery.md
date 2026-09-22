# R5B Controlled CandDif Recovery

Decision: `REPRODUCTION_R5B_CANDDIF_RECOVERY_COMPLETE`

## Purpose

Execute and audit only the frozen R5A CandDif recovery subset, preserving one atomic local Ollama JSON result per request and computing offline HR@5, NDCG@5, CandDif-HR, and CandDif-NDCG.

## Frozen R5A Execution Contract

R5B uses `results/tables/reproduction_r5a_recovery_request_manifest.csv` as the only request universe. It does not regenerate prompts, candidates, histories, models, users, request IDs, or output destinations.

## Manifest SHA

Manifest SHA256: `290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD`

## Execution Scope

Selected rows: `1200` `CANDDIF_RECOVERY` requests. History selected: `0`. Generation selected: `0`. ML-1M CandDif selected: `0`.

## Runtime

| model_name | model_available | model_digest | model_quantization_level | request_count | succeeded | backend_failed | not_executed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mistral:7b-instruct | true | 6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf | Q4_K_M | 600 | 600 | 0 | 0 |
| qwen2:7b-instruct | true | dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755 | Q4_0 | 600 | 600 | 0 | 0 |

## Models

R5B used only the frozen model identifiers in the manifest: `mistral:7b-instruct` and `qwen2:7b-instruct`.

## Datasets/Cohorts

Datasets: `Beauty`, `Sports`, and `LastFM`. Each dataset/model condition has the frozen 100-user R5A CandDif cohort.

## FIRST/RANDOM Protocol

Each logical pair has one `CANDDIF_FIRST` row and one `CANDDIF_RANDOM` row. The unordered candidate item set and history are shared; candidate order is the intended controlled difference.

## Execution Completeness

- Frozen CandDif requests: `1200`
- Terminal outputs: `1200`
- Succeeded backend calls: `1200`
- Backend failures: `0`
- Not executed: `0`

## Output Validity

- Valid strict-parser outputs: `52`
- Invalid strict-parser outputs preserved: `1148`
- Denominator policy: `FAIL_STRICT_ALL_MANIFEST_REQUESTS_INVALID_ZERO`

## HR/NDCG By Condition

| dataset | model_name | experimental_condition | n_manifest | n_valid | n_invalid | HR@5 | NDCG@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Beauty | mistral:7b-instruct | CANDDIF_FIRST | 100 | 4 | 96 | 0.01 | 0.01 |
| Beauty | mistral:7b-instruct | CANDDIF_RANDOM | 100 | 7 | 93 | 0.03 | 0.018175293653079346 |
| Beauty | qwen2:7b-instruct | CANDDIF_FIRST | 100 | 1 | 99 | 0.0 | 0.0 |
| Beauty | qwen2:7b-instruct | CANDDIF_RANDOM | 100 | 0 | 100 | 0.0 | 0.0 |
| LastFM | mistral:7b-instruct | CANDDIF_FIRST | 100 | 14 | 86 | 0.08 | 0.08 |
| LastFM | mistral:7b-instruct | CANDDIF_RANDOM | 100 | 12 | 88 | 0.06 | 0.0452371901428583 |
| LastFM | qwen2:7b-instruct | CANDDIF_FIRST | 100 | 0 | 100 | 0.0 | 0.0 |
| LastFM | qwen2:7b-instruct | CANDDIF_RANDOM | 100 | 0 | 100 | 0.0 | 0.0 |
| Sports | mistral:7b-instruct | CANDDIF_FIRST | 100 | 9 | 91 | 0.07 | 0.07 |
| Sports | mistral:7b-instruct | CANDDIF_RANDOM | 100 | 4 | 96 | 0.01 | 0.003868528072345416 |
| Sports | qwen2:7b-instruct | CANDDIF_FIRST | 100 | 1 | 99 | 0.0 | 0.0 |
| Sports | qwen2:7b-instruct | CANDDIF_RANDOM | 100 | 0 | 100 | 0.0 | 0.0 |

## CandDif-HR Results

| dataset | model_name | n_users | HR_first | HR_random | CandDif_HR | numerical_safety_policy_hr | evidence_class |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Beauty | mistral:7b-instruct | 100 | 0.01 | 0.03 | -0.020408871631207123 | NONE | PAPER_ALIGNED_ADAPTED_RECOVERY |
| Beauty | qwen2:7b-instruct | 100 | 0.0 | 0.0 | 0.0 | NONE | PAPER_ALIGNED_ADAPTED_RECOVERY |
| LastFM | mistral:7b-instruct | 100 | 0.08 | 0.06 | 0.021506205220963463 | NONE | PAPER_ALIGNED_ADAPTED_RECOVERY |
| LastFM | qwen2:7b-instruct | 100 | 0.0 | 0.0 | 0.0 | NONE | PAPER_ALIGNED_ADAPTED_RECOVERY |
| Sports | mistral:7b-instruct | 100 | 0.07 | 0.01 | 0.06252035698133414 | NONE | PAPER_ALIGNED_ADAPTED_RECOVERY |
| Sports | qwen2:7b-instruct | 100 | 0.0 | 0.0 | 0.0 | NONE | PAPER_ALIGNED_ADAPTED_RECOVERY |

## CandDif-NDCG Results

| dataset | model_name | n_users | NDCG_first | NDCG_random | CandDif_NDCG | numerical_safety_policy_ndcg | evidence_class |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Beauty | mistral:7b-instruct | 100 | 0.01 | 0.018175293653079346 | -0.008292157483430252 | NONE | PAPER_ALIGNED_ADAPTED_RECOVERY |
| Beauty | qwen2:7b-instruct | 100 | 0.0 | 0.0 | 0.0 | NONE | PAPER_ALIGNED_ADAPTED_RECOVERY |
| LastFM | mistral:7b-instruct | 100 | 0.08 | 0.0452371901428583 | 0.03708927294780549 | NONE | PAPER_ALIGNED_ADAPTED_RECOVERY |
| LastFM | qwen2:7b-instruct | 100 | 0.0 | 0.0 | 0.0 | NONE | PAPER_ALIGNED_ADAPTED_RECOVERY |
| Sports | mistral:7b-instruct | 100 | 0.07 | 0.003868528072345416 | 0.06869466265343702 | NONE | PAPER_ALIGNED_ADAPTED_RECOVERY |
| Sports | qwen2:7b-instruct | 100 | 0.0 | 0.0 | 0.0 | NONE | PAPER_ALIGNED_ADAPTED_RECOVERY |

## Dataset/Model Interpretation

| dataset | model_name | n_manifest_requests | n_succeeded | n_valid_outputs | n_invalid_outputs | n_backend_failed | n_not_executed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Beauty | mistral:7b-instruct | 200 | 200 | 11 | 189 | 0 | 0 |
| Beauty | qwen2:7b-instruct | 200 | 200 | 1 | 199 | 0 | 0 |
| LastFM | mistral:7b-instruct | 200 | 200 | 26 | 174 | 0 | 0 |
| LastFM | qwen2:7b-instruct | 200 | 200 | 0 | 200 | 0 | 0 |
| Sports | mistral:7b-instruct | 200 | 200 | 13 | 187 | 0 | 0 |
| Sports | qwen2:7b-instruct | 200 | 200 | 1 | 199 | 0 | 0 |

Positive CandDif means FIRST improves the corresponding accuracy relative to RANDOM; values near zero indicate little measured difference; negative values indicate RANDOM outperforms FIRST. No statistical significance is claimed.

## Comparison With Official Evidence

| dataset | model_name | metric | local_adapted_canddif | official_exact_canddif_cells | official_partial_canddif_cells | comparison_class |
| --- | --- | --- | --- | --- | --- | --- |
| Beauty | mistral:7b-instruct | CandDif-HR | -0.020408871631207123 | 6 | 112 | DIRECTIONAL_COMPARISON_ONLY |
| Beauty | mistral:7b-instruct | CandDif-NDCG | -0.008292157483430252 | 6 | 112 | DIRECTIONAL_COMPARISON_ONLY |
| Beauty | qwen2:7b-instruct | CandDif-HR | 0.0 | 6 | 112 | DIRECTIONAL_COMPARISON_ONLY |
| Beauty | qwen2:7b-instruct | CandDif-NDCG | 0.0 | 6 | 112 | DIRECTIONAL_COMPARISON_ONLY |
| LastFM | mistral:7b-instruct | CandDif-HR | 0.021506205220963463 | 0 | 0 | NO_DIRECT_COMPARISON |
| LastFM | mistral:7b-instruct | CandDif-NDCG | 0.03708927294780549 | 0 | 0 | NO_DIRECT_COMPARISON |
| LastFM | qwen2:7b-instruct | CandDif-HR | 0.0 | 0 | 0 | NO_DIRECT_COMPARISON |
| LastFM | qwen2:7b-instruct | CandDif-NDCG | 0.0 | 0 | 0 | NO_DIRECT_COMPARISON |
| Sports | mistral:7b-instruct | CandDif-HR | 0.06252035698133414 | 0 | 48 | DIRECTIONAL_COMPARISON_ONLY |
| Sports | mistral:7b-instruct | CandDif-NDCG | 0.06869466265343702 | 0 | 48 | DIRECTIONAL_COMPARISON_ONLY |
| Sports | qwen2:7b-instruct | CandDif-HR | 0.0 | 0 | 48 | DIRECTIONAL_COMPARISON_ONLY |
| Sports | qwen2:7b-instruct | CandDif-NDCG | 0.0 | 0 | 48 | DIRECTIONAL_COMPARISON_ONLY |

R5B does not force numeric paper/official comparison where model identity, protocol, dataset condition, candidate construction, and output contract are not identical.

## Resolution Of Six P0 Gaps

| dataset | metric | requests_accounted | not_executed | backend_failed | resolution_status |
| --- | --- | --- | --- | --- | --- |
| Beauty | CandDif-HR | 400 | 0 | 0 | RECOVERED_PAPER_ALIGNED_ADAPTED |
| Beauty | CandDif-NDCG | 400 | 0 | 0 | RECOVERED_PAPER_ALIGNED_ADAPTED |
| LastFM | CandDif-HR | 400 | 0 | 0 | RECOVERED_PAPER_ALIGNED_ADAPTED |
| LastFM | CandDif-NDCG | 400 | 0 | 0 | RECOVERED_PAPER_ALIGNED_ADAPTED |
| Sports | CandDif-HR | 400 | 0 | 0 | RECOVERED_PAPER_ALIGNED_ADAPTED |
| Sports | CandDif-NDCG | 400 | 0 | 0 | RECOVERED_PAPER_ALIGNED_ADAPTED |

## Remaining Limitations

The recovered evidence is local and adapted: model identity, preprocessing, candidate generation lineage, and strict JSON index output differ from paper-exact conditions.

## Effect On Global Reproduction Status

The global project status remains `CRITICAL_EXPERIMENTAL_PATH_PROTOCOL_LIMITED`.

## Protection/Provenance

R5B validation checks the branch, HEAD, staged diff, `original_repo/` cleanliness, protected root hashes, manifest SHA, pair contract, runtime availability, request accounting, and output namespace.

## Validation

Validation result: `20 PASS / 0 FAIL`.

| check_id | check_name | status | observed | expected |
| --- | --- | --- | --- | --- |
| R5B-VAL-001 | expected branch | pass | master | master |
| R5B-VAL-002 | expected head | pass | 98c28c5364c40e571765a3b50d4dab8affe80743 | 98c28c5364c40e571765a3b50d4dab8affe80743 |
| R5B-VAL-003 | no staged diff | pass |  | empty |
| R5B-VAL-004 | original repo clean | pass |  | empty |
| R5B-VAL-005 | protected artifacts unchanged | pass | 12/12 | 12/12 |
| R5B-VAL-006 | manifest sha exact | pass | 290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD | 290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD |
| R5B-VAL-007 | manifest row count | pass | 4800 | 4800 |
| R5B-VAL-008 | CandDif row count | pass | 1200 | 1200 |
| R5B-VAL-009 | no History Generation ML1M CandDif | pass | families=['CANDDIF_RECOVERY']; datasets=['Beauty', 'LastFM', 'Sports'] | CandDif ranking only; no ML-1M |
| R5B-VAL-010 | pair count and validity | pass | pairs=600; valid=600 | 600 valid |
| R5B-VAL-011 | material hashes match manifest | pass | prompt=0;candidate=0;history=0 | 0;0;0 |
| R5B-VAL-012 | runtime models available | pass |  | qwen2:7b-instruct and mistral:7b-instruct available |
| R5B-VAL-013 | request accounting sums | pass | rows=1200;terminal=1200;not_executed=0 | rows=1200;terminal+not_executed=1200 |
| R5B-VAL-014 | all requests executed | pass | 0 | 0 |
| R5B-VAL-015 | no backend failures | pass | 0 | 0 |
| R5B-VAL-016 | invalid outputs preserved not hallucination | pass | invalid_outputs=1148 | invalid_outputs>=0 with parse status recorded |
| R5B-VAL-017 | P0 gap set accounted | pass | [["Beauty","CandDif-HR"],["Beauty","CandDif-NDCG"],["LastFM","CandDif-HR"],["LastFM","CandDif-NDCG"],["Sports","CandDif-HR"],["Sports","CandDif-NDCG"]] | [["Beauty","CandDif-HR"],["Beauty","CandDif-NDCG"],["LastFM","CandDif-HR"],["LastFM","CandDif-NDCG"],["Sports","CandDif-HR"],["Sports","CandDif-NDCG"]] |
| R5B-VAL-018 | P0 gap resolution complete | pass | {'RECOVERED_PAPER_ALIGNED_ADAPTED': 6} | all RECOVERED_PAPER_ALIGNED_ADAPTED |
| R5B-VAL-019 | no out-of-scope raw outputs | pass |  | none |
| R5B-VAL-020 | disk free guard | pass | 155088941056 | >=1073741824 |

## What R5C May Do Next

After R5B review/freeze, R5C may address the frozen History-Length recovery subset. R5B does not execute R5C.

Blocked or partial reason: ``
