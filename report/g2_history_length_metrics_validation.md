# G2 History-Length Metrics Validation

- Phase: `g2_history_length_metrics_freeze`
- Decision: `G2_METRICS_FROZEN`
- HEAD before generation: `70d808982645a4b607cef1b3d2444d1febe0d298`
- Scope: frozen G2 analytic universe and retained G1/G2 evidence only; zero Ollama calls; no G3.

## Accounting

- Logical requests: `88050`
- G1 exact reuse: `18660`
- G2 terminal evidence: `69390`
- Strict-valid: `86768`
- Invalid terminal evidence: `1282`
- Backend errors: `0`
- Missing/duplicate/ambiguous evidence: `0` / `0` / `0`

## Cells And Metrics

- Applicable cells: `396`
- Computed metric rows: `5940`
- Denominator rows: `5940`
- Undefined/blocked rows: `984`
- Computed metrics: `hr_at_5;ndcg_at_5;mrr;novelty_at_5;catalog_coverage_at_5;item_coverage_at_5;long_tail_rate_at_5;average_popularity_at_5;duplicate_rate_at_5;candidate_validity_rate;backend_success_rate;strict_parse_valid_rate;invalid_length_rate;duplicate_index_rate;invalid_candidate_reference_rate`
- Blocked by frozen contract, not substituted: `popularity_bias_at_5;intra_list_diversity_at_5`

## Denominator Contract

- Utility: planned/applicable users; non-strict terminal responses count as zero utility.
- Beyond-utility: strict-valid users; novelty additionally uses a no-smoothing positive-train-support subset.
- Coverage: strict-valid users define the exposure set; catalog coverage denominator is train-seen catalog; item coverage denominator is the strict-valid cell candidate union.
- Structural quality: strict-valid users.
- Reliability: planned/applicable users.

## Membership Summary

| membership | rows | cells | G1 reuse | G2 terminal | strict-valid | invalid |
|---|---:|---:|---:|---:|---:|---:|
| FOODCOM_TRUE_ZERO_HISTORY_DIAGNOSTIC | 4362 | 6 | 4362 | 0 | 4343 | 19 |
| LONG_HISTORY_B2_GE20 | 24396 | 60 | 12198 | 12198 | 23825 | 571 |
| PRIMARY_B1_GE10 | 71490 | 150 | 14298 | 57192 | 70674 | 816 |
| SECONDARY_FULL_CURVE_GE20 | 73188 | 180 | 12198 | 60990 | 72033 | 1155 |

## Validation

- Passing checks: `21` / `21`
- Strict-valid rows with absent positive in ranked output: `34162`
- Strict-valid rows with at least one zero train-count top-5 item for novelty: `5350`
- Undefined status counts: `{"BLOCKED_BY_FROZEN_CONTRACT":792,"DEFINED_SUBSET_USED":192}`

## Artifacts

- `results/tables/g2_history_length_metric_denominators.csv` SHA256 `ed2b86c30b3c2f7b581159d73b84e909084b61b1200be630f44cc5587f7f3371`
- `results/tables/g2_history_length_metric_results.csv` SHA256 `4d4e8e3dd9e160556d2788f62684b3068fa93b4a24cefac61ae40df616d5e9f0`
- `results/tables/g2_history_length_metric_undefined.csv` SHA256 `05ad0ad1f4c8acb68fda989a2d551559e25e04d68bb5fe789175e8a78e139f99`
- `results/tables/g2_history_length_metric_validation.csv` SHA256 `1ed8bd27bb3b326e38b164d301baa5115607a259b4e8997639c4e4098ac250a7`
