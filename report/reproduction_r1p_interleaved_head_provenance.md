# R1P Interleaved Head Provenance

Label: POST-F6M TOTAL REPRODUCIBILITY CLOSURE EVIDENCE

Decision: `R1P_HEAD_ACCEPTABLE_WITH_PROVENANCE_ADDENDUM`

## Scope

R1P audited commit `092f6862295402389a4aa20a1e4ef2d8b66da0b0` without
rewriting Git history and without executing R2, resuming E3, downloading
Food.com, training models, running inference, or modifying
`original_repo/EvaLLMasRecommender`.

Expected pre-R1 baseline: `955c5397330123424dd9dcb0ed87ae4f32eacceb`.

Actual HEAD at R1/R1P inspection:
`092f6862295402389a4aa20a1e4ef2d8b66da0b0`.

`955c5397330123424dd9dcb0ed87ae4f32eacceb` is the direct parent of
`092f6862295402389a4aa20a1e4ef2d8b66da0b0`, so the discrepancy is exactly one
interleaved commit.

## Commit Provenance

Commit: `092f6862295402389a4aa20a1e4ef2d8b66da0b0`

Message: `reproducibility final metrics`

Author and committer: `BartoloMorgigno11 <bartolomorgignolav@gmail.com>`

Author/commit date: `2026-08-17T18:24:00+02:00`

Classification: `SAFE_BUT_PREVIOUSLY_UNREPORTED_COMMIT`, with one comparison
artifact requiring this provenance addendum.

Scientific consequence: retain the commit without history rewriting. The E3
content is a documented acquisition-blocked implementation/evidence state, not
a completed Food.com experiment. The comparison CSV is safe evidence, but its
standalone regeneration path is not encoded in the commit.

## File Classification

| File | Commit action | Classification | Finding |
|---|---:|---|---|
| `.gitignore` | M | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | Adds Food.com processed-data ignore and E3 table exception. |
| `configs/foodcom_extension.yaml` | A | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | Manual-staging E3 config; API calls disabled. |
| `src/data/prepare_foodcom_extension.py` | A | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | E3 integrity runner; docstring states no download, training, scoring, LLM inference, metric computation, or `original_repo` modification. |
| `tests/test_extension_e3_foodcom_data_integrity.py` | A | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | Synthetic and missing-archive E3 validation. |
| `report/extension_e3_foodcom_data_integrity.md` | A | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | Records `EXTENSION_E3_FOODCOM_ACQUISITION_BLOCKED`. |
| `report/notes.md` | M | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | Append-only E3 blocker note; prior claims not rewritten. |
| `results/tables/extension_e3_foodcom_acquisition_manifest.csv` | A | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | Records missing `data/raw/foodcom/source`. |
| `results/tables/extension_e3_foodcom_cohort_statistics.csv` | A | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | `not_run` blocker row. |
| `results/tables/extension_e3_foodcom_core_statistics.csv` | A | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | `not_run` blocker row. |
| `results/tables/extension_e3_foodcom_decision_summary.csv` | A | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | Decision token is `EXTENSION_E3_FOODCOM_ACQUISITION_BLOCKED`; commit policy `NONE`. |
| `results/tables/extension_e3_foodcom_duplicate_audit.csv` | A | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | `not_run` blocker row. |
| `results/tables/extension_e3_foodcom_forensic_traces.csv` | A | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | `not_run` placeholder row. |
| `results/tables/extension_e3_foodcom_mapping_audit.csv` | A | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | `not_run` blocker row. |
| `results/tables/extension_e3_foodcom_metric_readiness.csv` | A | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | 29 readiness rows remain `UNRESOLVED` because acquisition/preprocessing did not complete. |
| `results/tables/extension_e3_foodcom_model_readiness.csv` | A | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | 7 model readiness rows remain unresolved. |
| `results/tables/extension_e3_foodcom_preprocessing_contract.csv` | A | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | Documents intended protocol choices; no processed data evidence. |
| `results/tables/extension_e3_foodcom_protected_inputs.csv` | A | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | `not_run` blocker row. |
| `results/tables/extension_e3_foodcom_raw_schema.csv` | A | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | `not_run` blocker row. |
| `results/tables/extension_e3_foodcom_raw_statistics.csv` | A | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | `not_run` blocker row. |
| `results/tables/extension_e3_foodcom_requirement_realization.csv` | A | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | 23 requirement rows remain unresolved where acquisition/preprocessing is missing. |
| `results/tables/extension_e3_foodcom_split_statistics.csv` | A | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | `not_run` blocker row. |
| `results/tables/extension_e3_foodcom_temporal_tie_audit.csv` | A | `CONSISTENT_WITH_DOCUMENTED_BLOCKED_E3` | `not_run` blocker row. |
| `final_metric_paper_comparison.csv` | A | `SAFE_BUT_PROVENANCE_INCOMPLETE_COMPARISON_ARTIFACT` | Derived comparison table over existing paper/local evidence; no exact replications claimed; no E3/Food.com source rows; no dedicated regeneration script found in the commit. |

## E3 Reconciliation

E3 consistency: YES.

The committed E3 report and decision table state
`EXTENSION_E3_FOODCOM_ACQUISITION_BLOCKED` because
`data/raw/foodcom/source` was missing. The E3 notes explicitly say there was no
dataset download, no Kaggle credential/tooling use, no substitution dataset, no
real Food.com preprocessing, no training, no model scoring, no LLM/API call, no
candidate-pool experiment, and no `original_repo` modification.

The committed E3 tests were re-run during R1P:

```text
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 .\.venv_consolidation_candidate\Scripts\python.exe -B -m pytest tests/test_extension_e3_foodcom_data_integrity.py tests/test_config.py tests/test_schemas.py -q -p no:cacheprovider --basetemp results/logs/r1p_e3_provenance_validation_tmp
23 passed
```

The E3 runner was not re-executed during R1P because it would rewrite existing
timestamped E3 evidence files. The blocker state was verified from committed
tables and filesystem state.

## Data And Execution Absence

Food.com acquisition state: BLOCKED.

Food.com raw data committed: NO.

Food.com processed data committed: NO.

Training/inference contained in commit `092f686...`: NO.

Commit `092f686...` does not add files under `data/raw/foodcom`,
`data/processed/foodcom`, `results/checkpoints`, `results/models`,
`results/predictions`, `results/llm`, or new `results/raw_outputs` paths.

## final_metric_paper_comparison.csv Audit

Classification: `SAFE_BUT_PROVENANCE_INCOMPLETE_COMPARISON_ARTIFACT`.

Rows: 187.

Every `paper_cell_id` maps to an existing `contract_id` in
`results/tables/paper_atomic_contract.csv`. All rows have
`paper_value_source_status=secondary_structured_source_pdf_unverified`.

Comparison-status counts:

| Status | Count |
|---|---:|
| `local_value_available_but_protocol_incompatible` | 44 |
| `no_local_final_metric_value_available` | 139 |
| `paper_value_missing_in_contract` | 4 |

Exact numeric replication rows: 0.

Rows sourcing E3/Food.com evidence: 0.

The 158-row `requested_metric_paper_comparison_values.csv` pre-existing root
artifact is a subset of this table by `paper_cell_id`. The committed CSV does
not change a scientific conclusion; it reinforces that local values are either
protocol-incompatible, unavailable, or not applicable. The incomplete part is
operational provenance: the commit does not include a dedicated script or
manifest that regenerates this exact CSV.

## Frozen-Phase Protection

F6M protected artifact changes in `092f686...`: NO.

E1 protected artifact changes in `092f686...`: NO.

E2 protected artifact changes in `092f686...`: NO.

`original_repo/EvaLLMasRecommender` changes in `092f686...`: NO.

`original_repo/EvaLLMasRecommender` current status: `## main...origin/main`.

## Pre-Existing Root Artifact SHA256 Manifest

These files remain root-level untracked artifacts and were not staged by R1P.

| File | Exists | Tracked | Git status | SHA256 |
|---|---:|---:|---|---|
| `final_metric_paper_comparison_values.md` | YES | NO | `??` | `fd0209c410761e08b5ae7189911a2e80f5ac58ad3ad0d18e3ea030a34397c23f` |
| `original_repo_metric_file_inventory.csv` | YES | NO | `??` | `267de0f3be059419202c41df017e39b883b29974f1bd3a9bcb80b5e365916b78` |
| `original_repo_metric_values_decoded.csv` | YES | NO | `??` | `7e84facf5d00260c4fbe8bf9a6fa849ea4b7f38b88cef01ec2a67d8828a505c6` |
| `original_repo_metric_values_decoded_summary.csv` | YES | NO | `??` | `dfcf6dc472ceb1a19781022bef6b19904609798ed55ab53122cffe929452524f` |
| `original_repo_metric_values_decoded_wide.csv` | YES | NO | `??` | `6d27c08b553f162284b002bb133185d1bec87a90438c9b80de88a8a4de8fdd51` |
| `original_repo_requested_metric_values_decoded.csv` | YES | NO | `??` | `268e33138f04d2c3e03bdd98da0267418374d2551bc8b5c5dbd3800df46b3bc0` |
| `requested_metric_candidate_position_bias_values.csv` | YES | NO | `??` | `02f279cc7c1a5d3e719554ddc8e46040f76f1038a99639df3dc07a9a7e7f400d` |
| `requested_metric_generation_involved_performance_values.csv` | YES | NO | `??` | `9d19e1635cb6a39e963f1eb07f515951ea924097f0f84ac3011e049b444d45b6` |
| `requested_metric_history_length_sensitivity_values.csv` | YES | NO | `??` | `48452372e0ae79706d98ab226b4272df1b13982ac3fbe68aaf255d96aea9e4a4` |
| `requested_metric_novelty_values.csv` | YES | NO | `??` | `fa4efb7c9e142875478573a3e31bd7d3773a3ec3ce8c911c78de348ea8792b87` |
| `requested_metric_paper_comparison_values.csv` | YES | NO | `??` | `a112adb8010a960be4c4ecb27a6498d4e3ab3c0c272ea212369ee6bd491072b4` |
| `requested_metric_values_summary.csv` | YES | NO | `??` | `87c5082be5cdc2ff3d19a153eaf7ab56b776791d207b86811f56c2c555f3c929` |

## Decision

`092f686...` can be retained without rewriting history, but R1 freeze should
treat it as the documented committed baseline and include this R1P report as
the provenance addendum.

Next recommended action: rerun the R1 freeze gate using `092f686...` as the
documented committed baseline, include this report in the R1 freeze commit, and
then proceed to R2. Do not execute R2 in R1P.
