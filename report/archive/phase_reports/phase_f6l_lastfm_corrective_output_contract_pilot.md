# phase_f6l_lastfm_corrective_output_contract_pilot

- Decision token: `f6l_lastfm_corrective_output_contract_pilot_failed_redesign_required`
- Run id: `20260804_115330`
- Raw output dir: `results/llm/phase_f6l_lastfm_corrective_output_contract_pilot_20260804_115330`
- Scope: 40-request corrective local output-contract pilot only.
- Original 4000-output run remains separate as `prompt_parser_interface_fragility` evidence.
- V3 strict-valid: 33/40
- Acceptance checks: 11/14 pass.
- Parser: `phase_f6l_lastfm_candidate_index_json_parser_v3` hash `4C00EFF1FA4CE3EE706B1F51B641E070643F1A8ACE3D8D48CEA4AD2D62D3D741`.
- Contract hash: `938CFC785E81C666B5182905543050E29B1C3021D07FA3B8D47CDCC30F47B50C`.
- Full 4000-run command prepared: no

## Output Contract

The pilot prompts require exactly one JSON object of the form `{"ranking":[3,17,1,9,6,12,4,20,8,11]}`. The ranking list must contain exactly 10 distinct integers in the inclusive range 1-20. Titles, item ids, markdown, explanations, extra keys, and extra text are rejected.

## Evidence Boundary

Validity and contract diagnostics are pilot evidence. HR@5/NDCG@5 rows are technical smoke checks only and are marked `pilot_only_not_scientific_evidence`; they do not update research-question conclusions.

## Files

- Manifest: `results/tables/phase_f6l_lastfm_corrective_output_contract_pilot_manifest.csv`
- Preflight: `results/tables/phase_f6l_lastfm_corrective_output_contract_pilot_preflight.csv`
- Parse results: `results/tables/phase_f6l_lastfm_corrective_output_contract_pilot_parse_results.csv`
- Acceptance: `results/tables/phase_f6l_lastfm_corrective_output_contract_pilot_acceptance.csv`
- Smoke metrics: `results/tables/phase_f6l_lastfm_corrective_output_contract_pilot_smoke_metrics.csv`
