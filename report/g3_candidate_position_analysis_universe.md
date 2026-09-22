# G3 Candidate-Position Analytic Universe Freeze

## Scope
- Decision token: `G3_ANALYTIC_UNIVERSE_FROZEN`.
- Source: frozen G3 retained evidence only (`results/manifests/g3_candidate_position_execution_retention_manifest.json`).
- Zero Ollama/API calls, no metric computation, no statistical analysis, no protocol/config/runner edits, and no evidence rewrites were performed.

## Universe
- Logical rows: `12000`.
- Pairs: `6000` dataset x model x prompt x user FIRST/RANDOM pairs.
- Dimensions: `5` datasets, `3` models, `2` prompts, `200` users per dataset, `2` order conditions.
- FIRST/RANDOM rows: `FIRST=6000`, `RANDOM=6000`.

## Evidence Resolution
- Request ledger rows resolved: `12000`.
- Terminal rows resolved: `12000`.
- Raw files resolved: `12000`.
- Strict-valid: `11834`.
- Invalid terminal scientific evidence: `166` (`INVALID_LENGTH=114`, `INVALID_DUPLICATE=51`, `HALLUCINATION_OR_OUT_OF_CANDIDATE=1`).
- Backend errors: `0`.

## Counts
- `GRAND_TOTAL ALL`: rows `12000`, strict-valid `11834`, invalid `166`, backend errors `0`
- `BY_ORDER FIRST`: rows `6000`, strict-valid `5904`, invalid `96`, backend errors `0`
- `BY_ORDER RANDOM`: rows `6000`, strict-valid `5930`, invalid `70`, backend errors `0`
- `BY_DATASET beauty`: rows `2400`, strict-valid `2376`, invalid `24`, backend errors `0`
- `BY_DATASET sports`: rows `2400`, strict-valid `2359`, invalid `41`, backend errors `0`
- `BY_DATASET ml1m`: rows `2400`, strict-valid `2362`, invalid `38`, backend errors `0`
- `BY_DATASET lastfm`: rows `2400`, strict-valid `2358`, invalid `42`, backend errors `0`
- `BY_DATASET foodcom`: rows `2400`, strict-valid `2379`, invalid `21`, backend errors `0`
- `BY_MODEL qwen2:7b-instruct`: rows `4000`, strict-valid `3966`, invalid `34`, backend errors `0`
- `BY_MODEL mistral:7b-instruct`: rows `4000`, strict-valid `3915`, invalid `85`, backend errors `0`
- `BY_MODEL gemma2:9b`: rows `4000`, strict-valid `3953`, invalid `47`, backend errors `0`
- `BY_PROMPT BASE`: rows `6000`, strict-valid `5987`, invalid `13`, backend errors `0`
- `BY_PROMPT LLMRANK`: rows `6000`, strict-valid `5847`, invalid `153`, backend errors `0`

## Pairing Validation
- Pairing defects: `0`.
- Candidate/history/hash mismatches: `0`.
- Missing evidence: `0`.
- Orphan evidence: `0`.
- Duplicate evidence IDs: `0`.
- Ambiguous evidence defects: `0`.

## Denominator Semantics
- Utility: planned paired users; invalid terminal response contributes zero.
- Reliability: planned requests.
- CandDif: computed from paired FIRST/RANDOM cell accuracies.
- Beyond-utility: only if already frozen/applicable to G3, using strict-valid users or metric-specific defined-user subsets.

## Outputs
- `results/tables/g3_candidate_position_analysis_universe.csv`
- `results/tables/g3_candidate_position_analysis_pairs.csv`
- `results/tables/g3_candidate_position_analysis_counts.csv`
- `results/tables/g3_candidate_position_denominator_contract.csv`
- `results/tables/g3_candidate_position_analysis_validation.csv`
- `results/manifests/g3_candidate_position_analysis_universe_freeze.json`

G3_ANALYTIC_UNIVERSE_FROZEN
