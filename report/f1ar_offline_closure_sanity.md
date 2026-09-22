# F1AR Offline Closure Sanity

Decision token: `F1A_FINAL_OFFLINE_CLOSURE_VALIDATED`

## K-S Recoverability
- Original-four datasets audited: `4`.
- Original-four K-S result rows computed: `8`.
- Evidence class remains reconstructed/adapted diagnostic, not paper-exact representativeness.
- Beauty/Sports/ML1M primary K-S comparisons use identical-size sample/population definitions, so KS=0 and p=1 are structural and must not be overinterpreted as independent representativeness evidence.

## Final Blockers
None; F1ARR recovered PopREO/DPD global protocols. Degenerate protocol-input cells remain explicit in the F1ARR degenerate-cell table.

## Cost Wording
- `paid_api_marginal_monetary_cost`: No paid API provider is used by the frozen local-Ollama execution plan; provider marginal monetary cost is not a reportable metric.
- `total_computational_cost`: Local execution has nonzero compute time and resource use; do not describe total computational cost as zero.
- `hardware_electricity_cost`: Hardware depreciation and electricity cost are not measured by F1A/F1AR and must not be inferred.

## Validation
- Passed: `true`.
## F1ARR Addendum

- F1ARR supersedes the F1AR PopREO/DPD blocker classification: both protocols are recovered from the paper and released repository code.
- F1AR K-S closure remains unchanged; PopREO/DPD readiness is now governed by F1ARR cell-level audits.
