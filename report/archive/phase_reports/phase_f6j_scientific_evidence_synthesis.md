# Phase F6J Scientific Evidence Synthesis

Decision token: `f6j_scientific_evidence_synthesis_complete`

F6J is a synthesis phase. It does not run training, inference, API calls, downloads, package installation, model pulls, raw-output repair, Git staging, or commits. It closes the research-question readout using existing validated CSVs and manifests only.

## Cleanup Preflight

The exact requested preflight command was run before synthesis:

```bash
$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\.venv_audit_recovery\Scripts\python.exe -B -m pytest -q --tb=short -ra -p no:cacheprovider --basetemp results\logs\phase_f6j_cleanup_preflight_tmp -m "cleanup_current and not unsafe_external" tests
```

Result: `140 passed, 1698 deselected in 437.76s`.

Cleanup is therefore treated as terminally closed for F6J. The stale earlier failure state is not current F6J evidence.

## Evidence Inputs

Primary scientific inputs are existing generated artifacts:

- `results/tables/phase_f6f2_research_question_closure.csv`
- `results/tables/phase_f6f1_metric_exactness_matrix.csv`
- `results/tables/phase_f6f1_llm_vs_baseline.csv`
- `results/tables/phase_f6f1_ranking_vs_reranking.csv`
- `results/tables/phase_f6f1_aplt_results.csv`
- `results/tables/phase_f6f1_popularity_novelty_results.csv`
- `results/tables/phase_f6f1_serendipity_results.csv`
- `results/tables/phase_f6e4c_cell_summary.csv`
- `results/tables/phase_f6e4c_strategy_paired_comparison.csv`
- `results/tables/phase_f6e4c_invalid_output_audit.csv`
- `results/tables/current_reproduction_status_matrix.csv`
- `results/tables/finalization_f5b_ild_results.csv`
- `results/tables/finalization_f5e_calibration_results.csv`
- `results/tables/finalization_f6d2_aggregate_utility_metrics.csv`
- `results/tables/finalization_f6d2_seed_utility_metrics.csv`
- `results/tables/finalization_f6d4a_utility_metrics.csv`
- `results/tables/finalization_f6d4a_history_length_sensitivity.csv`
- `results/tables/finalization_f6d4a_candidate_position_bias.csv`
- `results/tables/phase_f6g3_cell_summary.csv`
- `results/tables/phase_f6g3_reranking_metrics.csv`
- `results/tables/plan_d_qwen2_ml1m_candidate_position_bias_pairwise_delta.csv`
- `results/tables/plan_e_api_cost_scenarios.csv`
- `results/tables/plan_e_local_runtime_estimates.csv`

No source is promoted beyond its recorded exactness class.

## Research-Question Closure

The authoritative row-level closure is `results/tables/phase_f6j_research_question_evidence_matrix.csv`.

Status counts:

- `supported`: 4
- `contradicted`: 3
- `mixed/context-dependent`: 4
- `unresolved`: 3

Closed as supported in the adapted or extension setting: RQ1 utility underperformance, RQ3 long-tail/APLT, RQ9 hallucination and validity separation, RQ12 structured indexed output.

Closed as contradicted in the adapted setting: RQ5 serendipity, RQ10 reranking advantage, RQ11 LLMRank advantage.

Closed as mixed or context-dependent: RQ2 dataset/domain effects, RQ4 novelty, RQ6 history length, RQ7 candidate-position bias.

Left unresolved: RQ8 profile-generation effectiveness, RQ13 LastFM reranking LLM cells, RQ14 current cost and runtime efficiency claims.

## Reproduction Boundary

The project remains a partial reproducibility and extension project. It supports repository-derived, adapted, diagnostic, and extension claims. It does not support exact numerical replication wording.

The separation is encoded in `results/tables/phase_f6j_paper_reproduction_extension_matrix.csv`, which includes exact numeric reproduction as unavailable, adapted reproduction for local baseline and LLM utility evidence, original extensions for beyond-utility diagnostics, diagnostic experiments for sensitivity analyses, paper-unspecified protocol rows, and blocked LastFM reranking cells.

## Claim Registry

Reportable and prohibited wording is recorded in `results/tables/phase_f6j_scientific_claim_registry.csv`.

The safe high-level claim is:

The local evidence supports a qualified adapted reproduction and extension analysis of the paper's scientific questions. It does not support exact numerical reproduction of the paper. Local LLMs underperform traditional baselines on most paired utility evidence, long-tail/APLT behavior is supported in the adapted setting, novelty is partially supported, serendipity and reranking advantage are contradicted under the local adapted definitions, validity must be reported separately from utility, and LastFM reranking plus current cost/runtime claims remain unresolved.

## Limitations

Threats and limitations are recorded in `results/tables/phase_f6j_limitations_and_threats.csv`.

The most important limits are exactness, candidate-pool differences, parser validity, reconstructed metric inputs, missing LastFM reranking raw responses, non-causal dataset and candidate-position evidence, unresolved profile-generation workflows, and non-current cost placeholders.

## Validation

Commands run for F6J validation:

```bash
$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\.venv_audit_recovery\Scripts\python.exe -B -m pytest -q --tb=short -ra -p no:cacheprovider --basetemp results\logs\phase_f6j_scientific_evidence_tmp tests\test_phase_f6j_scientific_evidence_synthesis.py
```

Result: `11 passed`.

```bash
$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONPATH='.'; .\.venv_audit_recovery\Scripts\python.exe -B -m pytest -q --tb=short -ra -p no:cacheprovider --basetemp results\logs\phase_f6j_cleanup_post_artifact_tmp -m "cleanup_current and not unsafe_external" tests
```

Result after F6J guard registration: `151 passed, 1698 deselected`.

Adjacent evidence checks:

- F6F2 scientific readout: `25 passed`.
- F6F1 beyond-utility metrics: `13 passed`.
- F6E4C scientific consolidation: `10 passed`.
- F6G3 LastFM reranking top-10 freeze: `8 passed` with `PYTHONPATH=.`.
- F6D2 LastFM trainable model substantive checks: `15 passed, 1 deselected`; the deselected check is a stale historical protected-hash invariant.
- F6D4A LastFM LLM ranking: `18 passed`.
- F5B ILD core checks: `8 passed, 2 deselected`; the deselected checks are stale historical protected-hash or validation-baseline invariants.
- F5E calibration core checks: `6 passed, 2 deselected`; the deselected checks are stale historical protected-hash or validation-baseline invariants.
- F6I2J/F6I2K/F6I2KR/F6I2L current guard subset: `38 passed`.

The focused F6J test checks schema, uniqueness, evidence references, reproduction/adaptation/extension separation, numeric/status coherence, deterministic ordering, protected-input hashes, absence of new F6J raw LLM outputs, cleanup-current preflight recording, and absence of stale current failure tokens in the F6J canonical handoff/report.
