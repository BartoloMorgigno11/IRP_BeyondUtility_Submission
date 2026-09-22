# G4 F1BZQ Metrics Freeze

- Phase: g4_f1bzq_metrics_freeze
- Decision token: G4_METRICS_FROZEN
- Source: frozen G4 analytic universe only
- Ollama/API calls: 0
- Statistics/bootstrap: not run

## Scope

- Logical rows: 29292
- Users: 4882
- Cells: 30
- Computed metric rows: 630
- Undefined/blocked metric rows: 510

## Candidate Recall

- beauty: 0.049494949494949494 (49/990)
- sports: 0.031390134529147982 (28/892)
- ml1m: 0.193 (193/1000)
- lastfm: 0.46899999999999997 (469/1000)
- foodcom: 0.0040000000000000001 (4/1000)

## Reliability Accounting

- Planned/backend calls represented: 29292
- Strict-valid: 28986
- Structural invalid: 306
- invalid_length: 255
- duplicate_indices: 51
- Backend errors: 0

## Undefined Or Blocked

- beyond_utility_g1_g2_compatible / UNDEFINED_FROZEN_ANALYTIC_UNIVERSE_INPUT_MISSING: 300
- excluded / EXCLUDED_BY_FROZEN_G4_CONTRACT: 150
- reliability_efficiency / UNDEFINED_FROZEN_ANALYTIC_UNIVERSE_INPUT_MISSING: 60

## Validation

- PASS: 70
- FAIL: 0
- Status: PASS

No scientific synthesis, ranking-vs-reranking interpretation, or statistical testing is included in this freeze.
