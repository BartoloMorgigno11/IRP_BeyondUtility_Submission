# phase_f6l_lastfm_corrective_output_contract_v4_pilot

- Decision token: `f6l_lastfm_corrective_output_contract_v4_pilot_passed_full_rerun_ready`
- Run id: `20260804_123438`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_v4_pilot_20260804_123438`
- Classification: `corrective_output_contract_v4_regression_and_holdout_pilot`
- Completed: 40/40
- V4 strict-valid: 40/40
- Acceptance checks: 16/16 pass.
- Regression users: `640, 709, 1333, 1749`
- Holdout users: `223, 387, 2090, 208, 895, 609`
- Parser: `phase_f6l_lastfm_opaque_candidate_token_json_parser_v4` hash `5527024992F87984078F46F15491F1E8CB6EB7DF6659559416EBD67F8A098854`.
- Contract hash: `6E06E4DD3D8CD5EB332324B4FDEDAEA67943950698ED244F16C20B7BAE28714F`.
- Full 4000-run command prepared: yes, not executed

## Contract

V4 uses opaque candidate tokens C01-C20 and requires exactly one JSON object with a `ranking` list of 10 distinct authorized strings. Numeric indices without `C`, titles, markdown, explanations, extra keys, unknown tokens, and duplicates are rejected.

## Evidence Boundary

Validity diagnostics are technical pilot evidence. HR@5/NDCG@5 smoke rows are marked `pilot_only_not_scientific_evidence` and do not update RQs.

## Files

- V3 failure audit: `results/tables/phase_f6l_lastfm_v3_out_of_range_failure_audit.csv`
- Structured support: `results/tables/phase_f6l_lastfm_corrective_output_contract_v4_pilot_structured_mode_support.csv`
- Manifest: `results/tables/phase_f6l_lastfm_corrective_output_contract_v4_pilot_manifest.csv`
- Preflight: `results/tables/phase_f6l_lastfm_corrective_output_contract_v4_pilot_preflight.csv`
- Parse results: `results/tables/phase_f6l_lastfm_corrective_output_contract_v4_pilot_parse_results.csv`
- Regression/holdout validity: `results/tables/phase_f6l_lastfm_corrective_output_contract_v4_pilot_regression_holdout_validity.csv`
- Acceptance: `results/tables/phase_f6l_lastfm_corrective_output_contract_v4_pilot_acceptance.csv`
