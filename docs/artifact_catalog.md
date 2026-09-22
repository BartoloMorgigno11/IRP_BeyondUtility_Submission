# Artifact Catalog

This catalog identifies the committed foreground evidence for the full research repository. Raw LLM response dumps and external ZIP archives remain outside normal Git.

## Current State

- Reproduction status: `LOCAL_ADAPTED_REPRODUCTION_COMPLETE_NOT_PAPER_EXACT`
- F6M reproduction closure: complete and frozen.
- G1-G4 extension: complete and frozen.
- Delivery branch: `main`

## Primary Reports

- `report/final_scientific_report.md`: final narrative scientific summary.
- `report/phase_f6m_reproduction_closure.md`: reproduction-only F6M closeout.
- `report/g1_primary_ranking_execution.md`: G1 execution protocol and evidence.
- `report/g2_history_length_scientific_synthesis.md`: G2 history-length synthesis.
- `report/g3_candidate_position_scientific_synthesis.md`: G3 candidate-position synthesis.
- `report/g4_f1bzq_scientific_synthesis.md`: G4 reranking synthesis.
- `report/GITHUB_PROVENANCE_MAP.md`: old-to-new commit provenance for the reconstructed publishable history.
- Historical phase logs are intentionally excluded from the curated submission package.

## Primary F6M Tables

- `results/tables/phase_f6m_artifact_index.csv`
- `results/tables/phase_f6m_master_results_long.csv`
- `results/tables/phase_f6m_master_results_wide.csv`
- `results/tables/phase_f6m_master_reproducibility_matrix.csv`
- `results/tables/phase_f6m_research_question_matrix.csv`
- `results/tables/phase_f6m_paper_claim_evidence_matrix.csv`
- `results/tables/phase_f6m_limitations_register.csv`
- `results/tables/phase_f6m_lastfm_final_status.csv`
- `results/tables/phase_f6m_reproduction_extension_boundary.csv`

## Extension Foreground Tables

- G1 compact metrics: `results/tables/g1_primary_ranking_*.csv`
- G2 history-length outputs: `results/tables/g2_history_length_*.csv`
- G3 candidate-position outputs: `results/tables/g3_candidate_position_*.csv`
- G4 F1BZQ outputs: `results/tables/g4_f1bzq_*.csv`
- F1BZQ reranking pools and source outputs: `results/raw_outputs/f1bzq_*` where tracked.

Some wildcard families contain ignored local-only raw CSVs. Use `git ls-files` or the manifests to identify committed foreground files.

## Packaged CSV Artifacts

The following CSV evidence is committed as deterministic gzip packages:

- `results/tables/g2_history_length_request_universe.csv.gz`
- `results/tables/g2_history_length_execution_order.csv.gz`
- `results/tables/g4_f1bzq_statistical_paired_contributions.csv.gz`
- `results/tables/g1_primary_ranking_per_request_stats.csv.gz`
- `results/tables/g1_primary_ranking_strict_parsing.csv.gz`
- `results/tables/g1_primary_ranking_recommendations.csv.gz`
- `results/tables/g1_primary_ranking_request_universe.csv.gz`
- `results/tables/g1_primary_ranking_raw_response_inventory.csv.gz`
- `results/tables/g1_primary_ranking_execution_state.csv.gz`
- `results/tables/g1_primary_ranking_physical_execution_order.csv.gz`

Manifest:

- `results/manifests/github_packaged_artifacts.json`

Restore and verify:

```powershell
.\.venv_audit_recovery\Scripts\python.exe scripts\restore_packaged_artifacts.py --verify-package
.\.venv_audit_recovery\Scripts\python.exe scripts\restore_packaged_artifacts.py --restore
.\.venv_audit_recovery\Scripts\python.exe scripts\restore_packaged_artifacts.py --verify-restored
```

## Primary Manifests

- `results/manifests/phase_f6m_reproduction_closure_manifest.json`
- `results/manifests/g1_primary_ranking_execution_manifest.json`
- `results/manifests/g2_history_length_scientific_synthesis_manifest.json`
- `results/manifests/g3_candidate_position_scientific_synthesis_manifest.json`
- `results/manifests/g4_f1bzq_scientific_synthesis_manifest.json`
- `results/manifests/github_packaged_artifacts.json`
- `results/manifests/github_commit_provenance_map.csv`

## Historical Evidence

Historical phase reports remain useful lineage records, but their status statements must be read in the phase context in which they were written. F6M-only documents that say the future extension had not started are historical reproduction-closure statements and are superseded for repository-wide status by the frozen G1-G4 extension artifacts.

## External Or Ignored Evidence

The following families are not normal GitHub artifacts:

- raw and external datasets
- raw LLM response directories for G1-G4
- external archive ZIP bundles
- model checkpoints and local trainer states
- prediction dumps
- Python runtime bundles, caches, logs, and local temp files

See `docs/external_archive_guide.md` for the raw evidence policy.

## LastFM Note

The LastFM `314/1000` positive-present value is a reranking candidate-pool coverage diagnostic only. It is not a ranking denominator and is not the primary reranking metric denominator.
