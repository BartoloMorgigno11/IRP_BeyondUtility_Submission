# FINALIZATION F6A - Authoritative Paper-Scope Rebaseline Audit

# 1. Executive decision
- Decisione: `finalization_f6a_complete`.
- F6A supersedes taxonomy/accounting artifacts without editing F2-F5 outputs.

# 2. Authoritative paper scope
- Dataset paper: Amazon All Beauty, Sports & Outdoors, MovieLens-1M, LastFM.
- LLM paper: gpt-3.5-turbo-0125, gpt-4o-mini-2024-07-18, gpt-4o-2024-08-06, claude-3-haiku-20240307, llama-3-70b-instruct, qwen-2-7b-instruct, mistral-7b-instruct-v0.2.
- Tradizionali paper: BM25, MostPop, BPRMF, GRU4Rec, SASRec, LightGCN.

# 3. Confirmed taxonomy errors
- LastFM non e extension-only; e un dataset paper originale bloccato localmente.
- Qwen2 e Mistral non sono local-only extensions; sono famiglie LLM paper con run locali adattati.
- F5AF model inventory 5/5 e superseded: il conto corretto delle famiglie LLM paper e 7/7 identificate.

# 4. LastFM status
- Status: `original_paper_component_blocked`.
- Interactions locali: `92834`; catalog rows: `17632`.
- Mancano candidate set, ground truth e split leave-one-out validato.

# 5. Qwen2 status
- `original_paper_component_locally_adapted`; invalid rows `26`.

# 6. Mistral status
- `original_paper_component_locally_adapted`; invalid rows `0`.

# 7. True extension inventory
- Estensioni inventariate: `9`.

# 8. Additional-dataset requirement
- Fifth evaluation dataset: `no`; extra-dataset requirement: `not currently met` if mandatory.

# 9. Current coverage defects
- Denominatore storico: `73`; deve restare storico finche F6A non completa la ricostruzione row-level.

# 10. Proposed coverage taxonomy
- Coverage families separate: `13`.

# 11. Provisional corrected coverage
- Dataset evaluation-ready: `3/4`.
- LLM paper identified: `7/7`.
- Exact LLM version coverage: `0/7`.
- Atomic numeric cells: `187`; protocol-exact numeric replication: `0/187`.

# 12. Missing-evidence rebaseline
- Current operational missing count after F5E: `37`; materialized F3/F4 source rows audited here: `42`.
- Provisional corrected missing count remains `not_safely_derivable` until F6A row-level registry is consumed as authoritative.

# 13. Previous artifacts affected
- Affected artifacts listed: `17`.

# 14. Previous artifacts still valid
- F5B-F5E metric computations, F5E calibration extension-only status, F5AF 187 atomic cells, canonical local LLM execution evidence, F2 protected input hashes.

# 15. Scientific consequences
- Replace local-extension wording for Qwen/Mistral with paper-family/adapted-local wording.

# 16. Exam consequences
- Metric extension is defensible; extra dataset is not currently satisfied.

# 17. Corrective implementation package design
- This package writes new `finalization_f6a_*` tables plus this report.

# 18. Required output files for implementation
- `results/tables/finalization_f6a_authoritative_paper_scope.csv`
- `results/tables/finalization_f6a_paper_component_inventory.csv`
- `results/tables/finalization_f6a_dataset_taxonomy.csv`
- `results/tables/finalization_f6a_model_taxonomy.csv`
- `results/tables/finalization_f6a_prompt_taxonomy.csv`
- `results/tables/finalization_f6a_metric_dimension_taxonomy.csv`
- `results/tables/finalization_f6a_lastfm_reproduction_status.csv`
- `results/tables/finalization_f6a_qwen2_mistral_alignment.csv`
- `results/tables/finalization_f6a_extension_inventory.csv`
- `results/tables/finalization_f6a_additional_dataset_requirement.csv`
- `results/tables/finalization_f6a_coverage_family_definitions.csv`
- `results/tables/finalization_f6a_corrected_component_coverage.csv`
- `results/tables/finalization_f6a_missing_evidence_rebaseline.csv`
- `results/tables/finalization_f6a_affected_artifact_registry.csv`
- `results/tables/finalization_f6a_claim_corrections.csv`
- `results/tables/finalization_f6a_validation.csv`
- `report/finalization_f6a_paper_scope_rebaseline.md`

# 19. Required tests and validations
- Validation pass: `21`; fail: `0`.

# 20. Risks
- PDF visual verification remains limited; DOCX is secondary structured evidence.

# 21. Recommended next phase
- Use F6A artifacts as superseding taxonomy evidence, then update user-facing README/report only in a later F6B.

# 22. Final decision
- `finalization_f6a_complete`.

# Final details
- Branch: `plan-d-extensions`.
- HEAD: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`.
- Protected inputs: `checked=1144;changed=0`.
- Affected artifact count: `17`.
- Blockers: LastFM candidate/ground-truth/split; exact model versions; original prompts/candidate/cohort/protocol; API/hardware for blocked models.
