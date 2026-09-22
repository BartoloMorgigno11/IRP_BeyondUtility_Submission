# G4 F1BZQ Analytic Universe Freeze

## Scope
- Decision token: `G4_ANALYTIC_UNIVERSE_FROZEN`.
- Source: frozen G4 retained evidence only (`results/manifests/g4_f1bzq_execution_retention_manifest.json`).
- Zero Ollama/API calls, no metric computation, no statistical analysis, no protocol/config/runner/F1BZQ-pool edits, and no evidence rewrites were performed.

## Universe
- Logical rows: `29292`.
- Unique users: `4882`.
- Cells: `30` dataset x model x prompt cells.
- Dimensions: `5` datasets, `3` models, `2` prompts.

## Evidence Resolution
- Request ledger rows resolved: `29292`.
- Terminal rows resolved: `29292`.
- Raw files resolved: `29292`.
- Strict-valid: `28986`.
- Structural invalid: `306` (`INVALID_LENGTH=255`, `INVALID_DUPLICATE=51`).
- Backend errors: `0`.

## Candidate-Stage Denominators
- `beauty`: planned users `990`, positive-present `49`, positive-absent `941`, rows `5940`, strict-valid `5876`, invalid `64`
- `sports`: planned users `892`, positive-present `28`, positive-absent `864`, rows `5352`, strict-valid `5307`, invalid `45`
- `ml1m`: planned users `1000`, positive-present `193`, positive-absent `807`, rows `6000`, strict-valid `5936`, invalid `64`
- `lastfm`: planned users `1000`, positive-present `469`, positive-absent `531`, rows `6000`, strict-valid `5907`, invalid `93`
- `foodcom`: planned users `1000`, positive-present `4`, positive-absent `996`, rows `6000`, strict-valid `5960`, invalid `40`

## Denominator Contract
- Utility (`HR@5`, `NDCG@5`, `MRR`): all planned users; invalid terminal response contributes zero.
- Candidate recall: all users.
- Conditional utility (`conditional_hr_at_5`, `conditional_ndcg_at_5`): positive-present users only; invalid terminal response contributes zero within that subset.
- Beyond-utility: strict-valid users, with metric-specific defined subsets where required.
- Reliability: all planned requests.

## Validation
- Validation passed: `true`.
- Mismatches/defects: `0` failing checks.
- Candidate mapping, shared pool/history across model-prompt variants, canonical IDs, model identities, raw/terminal evidence, and denominator counts all passed when validation is true.

## Outputs
- `src/analysis/g4_f1bzq_analytic_universe.py`
- `results/tables/g4_f1bzq_analytic_universe.csv`
- `results/tables/g4_f1bzq_analytic_cell_counts.csv`
- `results/tables/g4_f1bzq_analytic_denominator_counts.csv`
- `results/tables/g4_f1bzq_analytic_denominator_contract.csv`
- `results/tables/g4_f1bzq_analytic_validation.csv`
- `results/manifests/g4_f1bzq_analytic_universe_freeze.json`
- `report/g4_f1bzq_analytic_universe_freeze.md`
- `report/notes.md`

G4_ANALYTIC_UNIVERSE_FROZEN
