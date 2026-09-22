# G4 F1BZQ Statistical Analysis

- Phase: g4_f1bzq_statistical_analysis
- Decision token: G4_STATISTICAL_ANALYSIS_FROZEN
- Source: frozen G4 metric artifacts and frozen G4 analytic universe only
- Ollama/API calls: 0
- P-values: not computed
- Holm correction: not applied; metadata only because no frozen p-value method exists

## Frozen Plan

- Template rows: 75/75
- Family counts: {"g1_vs_g4_pipeline_descriptive":30,"g4_model_pair":30,"g4_prompt_strategy":15}
- Inferential templates: 45
- Descriptive G1-vs-G4 templates: 30

## Outputs

- Paired contribution rows: 145188
- Statistical result rows: 315
- Inferential result rows: 225
- Descriptive result rows: 90

## CI Decisions

- Overall inferential CI decisions: {"CI_SUPPORTED_NEGATIVE":24,"CI_SUPPORTED_POSITIVE":29,"NON_SIGNIFICANT_CI_INCLUDES_ZERO":172}
- Prompt-strategy family: {"CI_SUPPORTED_NEGATIVE":13,"CI_SUPPORTED_POSITIVE":2,"NON_SIGNIFICANT_CI_INCLUDES_ZERO":60}
- Model-pair family: {"CI_SUPPORTED_NEGATIVE":11,"CI_SUPPORTED_POSITIVE":27,"NON_SIGNIFICANT_CI_INCLUDES_ZERO":112}
- beauty / NON_SIGNIFICANT_CI_INCLUDES_ZERO: 45
- foodcom / NON_SIGNIFICANT_CI_INCLUDES_ZERO: 45
- lastfm / CI_SUPPORTED_NEGATIVE: 14
- lastfm / CI_SUPPORTED_POSITIVE: 15
- lastfm / NON_SIGNIFICANT_CI_INCLUDES_ZERO: 16
- ml1m / CI_SUPPORTED_NEGATIVE: 5
- ml1m / CI_SUPPORTED_POSITIVE: 10
- ml1m / NON_SIGNIFICANT_CI_INCLUDES_ZERO: 30
- sports / CI_SUPPORTED_NEGATIVE: 5
- sports / CI_SUPPORTED_POSITIVE: 4
- sports / NON_SIGNIFICANT_CI_INCLUDES_ZERO: 36

## Denominator Notes

- Unconditional utility comparisons use all planned users with invalid terminal responses set to zero.
- Conditional HR/NDCG comparisons use positive-present users only, with invalid terminal responses set to zero within that subset.
- Food.com conditional comparisons have paired n=4 and are flagged in the results table.
- Candidate recall is upstream dataset-level evidence and is not tested across model/prompt variants.
- G1-vs-G4 rows are descriptive pipeline-level comparisons only; no causal ranking-vs-reranking inference is made.

## Validation

- PASS: 33
- FAIL: 0
- Status: PASS

No final scientific synthesis or relation update is included in this freeze.
