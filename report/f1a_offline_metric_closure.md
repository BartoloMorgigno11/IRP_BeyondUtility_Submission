# F1A Offline Metric Closure

Decision token: `F1A_OFFLINE_METRIC_CLOSURE_VALIDATED`

## Metric inventory
- Metrics accounted: `39`.
- Implemented or ready formulas: `36`.

## Offline results
- Offline result rows computed in F1A: `287`.
- Recommendation-dependent final cells remain readiness-classified until final raw outputs exist.

## Readiness
- `BLOCKED_GLOBAL_PROTOCOL` = `372`
- `COMPUTED_F1A` = `286`
- `IMPLEMENTATION_READY_AWAITING_RAW` = `810`
- `NOT_APPLICABLE` = `285`
- `READY_AFTER_TRAINING_RAW` = `1725`
- `READY_FROM_NEW_RAW` = `5460`

## Blockers
- `popreo_at_5` / `ALL`: `BLOCKED_GLOBAL_PROTOCOL` - F1A re-audit of paper/original/frozen E1/D0A/F0: popularity groups recoverable; zero-group handling and averaging policy irreducibly missing.
- `dpd_at_5` / `ALL`: `BLOCKED_GLOBAL_PROTOCOL` - F1A re-audit of paper/original/frozen E1/D0A/F0: active/inactive grouping described but threshold scope, normalization, and zero-denominator handling irreducibly missing.
- `ks_sample_representativeness` / `beauty`: `BLOCKED_DATA_REQUIREMENT` - selected H0 cohort exists but full eligible-population train-history vector is not materialized as a current F0/F1A artifact
- `ks_sample_representativeness` / `sports`: `BLOCKED_DATA_REQUIREMENT` - selected H0 cohort exists but full eligible-population train-history vector is not materialized as a current F0/F1A artifact
- `ks_sample_representativeness` / `ml1m`: `BLOCKED_DATA_REQUIREMENT` - selected H0 cohort exists but full eligible-population train-history vector is not materialized as a current F0/F1A artifact
- `ks_sample_representativeness` / `lastfm`: `BLOCKED_DATA_REQUIREMENT` - selected H0 cohort exists but full eligible-population train-history vector is not materialized as a current F0/F1A artifact

## Completeness
- Original four and Food.com are separately materialized in F1A completeness tables.

## Validation
- Validation passed: `true`.
## F1AR Addendum

- Original-four K-S population vectors were reconstructed offline.
- Eight original-four K-S diagnostics were added from F1AR: primary and history200 for Beauty, Sports, ML1M, and LastFM.
- Beauty/Sports/ML1M primary K-S comparisons use identical-size sample/population definitions, so KS=0 and p=1 are structural and must not be overinterpreted as independent representativeness evidence.
- F1ARR recovered the PopREO and DPD global protocols; remaining global metric blockers = 0.
## F1ARR Addendum

- PopREO/DPD protocol recovered from paper plus released executable code; final metric cells await canonical recommendation outputs where not already available.
- Global PopREO and DPD blockers were removed from F1A remaining blockers.
- Degenerate active/inactive or popularity-group inputs are classified at result-cell level as DEGENERATE_PROTOCOL_INPUT; no smoothing, merging, or epsilon repair is introduced.
