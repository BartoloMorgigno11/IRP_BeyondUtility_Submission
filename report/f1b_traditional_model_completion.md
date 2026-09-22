# F1B Traditional Recommender Completion Gate

Decision token: `F1B_TRADITIONAL_COMPLETION_BLOCKED`

## Summary
- F1B is blocked before training and before deterministic baseline recomputation.
- Blocker: Current repository evidence contains incompatible trainable seed/config lineages: original-paper Beauty/Sports archives reference seeds 0-4, local LastFM F6D2 adapted training uses seeds 42-46 and selected reranking seeds 44-46, and Plan E pilot artifacts use single seed 42 with superseded cohorts. F1B must not choose a cross-dataset training protocol implicitly.

## Reuse Audit
- Traditional F0 rows audited: `35`.
- Protocol-identical reusable traditional outputs: `0`.

## Training Plan
- Planned trainable seed slots recorded: `100`.
- Training runs executed: `0`.

## Methods
- `BM25-like/content` rows: `5`.
- `BPRMF` rows: `5`.
- `GRU4Rec` rows: `5`.
- `LightGCN` rows: `5`.
- `MostPop` rows: `5`.
- `Random` rows: `5`.
- `SASRec` rows: `5`.

## Datasets
- `beauty` methods: `7`.
- `foodcom` methods: `7`.
- `lastfm` methods: `7`.
- `ml1m` methods: `7`.
- `sports` methods: `7`.

## Result Cells
- `BLOCKED_EXECUTION` = `588`.
- `DEGENERATE_PROTOCOL_INPUT` = `7`.

## Reranking
- Candidate recall rows: `30`.
- Future requests unlocked: `0`.

## Integrity
- Active datasets: `beauty,sports,ml1m,lastfm,foodcom`.
- Video Games rows: `0`.
- F0 budget unchanged: `true`.
- Food.com hashes unchanged: `true`.
- H0 hashes unchanged: `true`.

## Validation
- Passed: `true`.

## F1BR Addendum
- F1BR resolved the seed/config protocol ambiguity without training.
- Random, MostPop, and BM25-like/content are ready to execute; trainable rows are ready to train under the F1BR frozen protocol.
- F1B completion remains not executed; reranking remains locked until first-stage rankings and candidate recall exist.
