# E3 Food.com Data Integrity

POST-F6M DATASET-5 EXTENSION EVIDENCE

## Final Decision

EXTENSION_E3_FOODCOM_ACQUISITION_BLOCKED

## Repository Precondition

- branch: master
- HEAD: 955c5397330123424dd9dcb0ed87ae4f32eacceb
- Git: ## master...origin/master [ahead 6]
 M .gitignore
 M report/notes.md
?? configs/foodcom_extension.yaml
?? report/extension_e3_foodcom_data_integrity.md
?? results/tables/extension_e3_foodcom_acquisition_manifest.csv
?? results/tables/extension_e3_foodcom_cohort_statistics.csv
?? results/tables/extension_e3_foodcom_core_statistics.csv
?? results/tables/extension_e3_foodcom_decision_summary.csv
?? results/tables/extension_e3_foodcom_duplicate_audit.csv
?? results/tables/extension_e3_foodcom_forensic_traces.csv
?? results/tables/extension_e3_foodcom_mapping_audit.csv
?? results/tables/extension_e3_foodcom_metric_readiness.csv
?? results/tables/extension_e3_foodcom_model_readiness.csv
?? results/tables/extension_e3_foodcom_preprocessing_contract.csv
?? results/tables/extension_e3_foodcom_protected_inputs.csv
?? results/tables/extension_e3_foodcom_raw_schema.csv
?? results/tables/extension_e3_foodcom_raw_statistics.csv
?? results/tables/extension_e3_foodcom_requirement_realization.csv
?? results/tables/extension_e3_foodcom_split_statistics.csv
?? results/tables/extension_e3_foodcom_temporal_tie_audit.csv
?? src/data/prepare_foodcom_extension.py
?? tests/test_extension_e3_foodcom_data_integrity.py
- runtime: .\.venv_consolidation_candidate\Scripts\python.exe, Python 3.12.10
- original_repo: ## main...origin/main

## Dataset Release

- exact release: UCSD Food.com Recipe & Review Data, linked official Kaggle endpoint shuyangli94/food-com-recipes-and-user-interactions
- authors/source: Bodhisattwa Prasad Majumder, Shuyang Li, Jianmo Ni, Julian McAuley; UCSD McAuley Lab
- acquisition mechanism: manual staging under data/raw/foodcom/source/
- terms/restrictions: raw data remains local and ignored; redistribute via official source links only

## Result

- reason: Manual source directory is missing: data/raw/foodcom/source
- run_started_at: 2026-08-17T16:17:13+00:00
- processed_hash: 

## Tables

- results/tables/extension_e3_foodcom_acquisition_manifest.csv
- results/tables/extension_e3_foodcom_raw_schema.csv
- results/tables/extension_e3_foodcom_raw_statistics.csv
- results/tables/extension_e3_foodcom_duplicate_audit.csv
- results/tables/extension_e3_foodcom_temporal_tie_audit.csv
- results/tables/extension_e3_foodcom_preprocessing_contract.csv
- results/tables/extension_e3_foodcom_core_statistics.csv
- results/tables/extension_e3_foodcom_mapping_audit.csv
- results/tables/extension_e3_foodcom_split_statistics.csv
- results/tables/extension_e3_foodcom_cohort_statistics.csv
- results/tables/extension_e3_foodcom_forensic_traces.csv
- results/tables/extension_e3_foodcom_metric_readiness.csv
- results/tables/extension_e3_foodcom_requirement_realization.csv
- results/tables/extension_e3_foodcom_model_readiness.csv
- results/tables/extension_e3_foodcom_protected_inputs.csv
- results/tables/extension_e3_foodcom_decision_summary.csv

## Interpretation

Is Food.com now a deterministic, provenance-aware, scientifically validated Dataset-5 input that is ready for the complete paper/local-reproduction benchmark pipeline?

NO

## Next Recommended Action

stage the official UCSD-linked Food.com Kaggle archive under data/raw/foodcom/source/, then rerun E3

## Scope Boundary

- no model training
- no model scoring
- no LLM/API calls
- no candidate-pool experiment
- no Calibration result computation
