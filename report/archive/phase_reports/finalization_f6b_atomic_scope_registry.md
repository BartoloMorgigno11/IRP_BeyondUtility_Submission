# 1. Executive summary
- Decisione: `finalization_f6b_complete`.
- Componenti paper atomici: `250`; non numerici: `63`; numeric cells: `187`.

# 2. Authoritative scope
- Scope paper: 4 dataset, 2 task, 7 LLM, 6 modelli tradizionali, 3 strategie prompt.

# 3. Atomicity methodology
- F6B crea righe atomiche solo per requisiti indipendentemente verificabili; non usa prodotti cartesiani arbitrari.

# 4. Dataset coverage
- Dataset identification: `4/4`.
- Evaluation-ready: `3/4`.

# 5. Task coverage
- Task implementation/execution: `2/2` and `2/2`.

# 6. Model-family coverage
- LLM-family local-run coverage: `2/7`.
- Traditional-model coverage: `6/6`.

# 7. Exact-version coverage
- Exact LLM-version coverage: `0/7`.

# 8. Prompt-strategy coverage
- Prompt implementation: `3/3`; exact equivalence: `0/3`.

# 9. Metric coverage
- Metric implementation: `12/12`; execution: `12/12`.

# 10. Analysis-dimension coverage
- Analysis-dimension support: `7/8`.

# 11. Paper-observation verification
- Verified exact/functionally: `0/7`.

# 12. Numeric-cell coverage
- Atomic extraction: `187/187`.
- Local mapping: `44/187`.
- Protocol-exact numeric replication: `0/187`.

# 13. True extension inventory
- Extension rows: `10`; LastFM is not an extension.

# 14. Fifth-dataset status
- Fifth dataset exists: `no`; fifth dataset requirement met: `no_if_mandatory`.

# 15. Corrected coverage dashboard
- Coverage families: `19`.

# 16. Why one headline percentage is unsafe or how it was derived
- Flat overall component percentage is `not_applicable`: rows overlap across component, prompt, model, metric, observation and numeric-cell families.

# 17. Missing-evidence rebaseline
- Source rows audited: `42`; corrected rows: `44`; active gap rows: `27`.

# 18. Experiment priority roadmap
- 1. PDF/manual verification of paper numeric evidence (`F6C`).
- 2. LastFM evaluation readiness (`F6C`).
- 3. Qwen2/Mistral exact-equivalence audit (`F6C`).
- 4. Original prompt-family completion plan (`F7`).
- 5. PopREO denominator policy recovery (`F6C`).
- 6. Candidate-first exact prompt condition (`F7`).
- 7. Generated-profile paired controls (`F7`).
- 8. Open-weight large-model feasibility (`F8`).
- 9. Closed-model exact execution decision (`F8`).
- 10. Additional dataset extension (`F9`).

# 19. Previous artifacts superseded
- Supersession rows: `17`.

# 20. Previous artifacts still authoritative
- F5AF atomic cells, F5B-F5E metric computations, canonical local LLM execution evidence, F2 protected hashes remain authoritative for their scoped evidence.

# 21. Claim corrections
- Claim corrections: `10`.

# 22. Regression timeout resolution
- Regression rows: `7`; inconclusive timeout rows: `2`.

# 23. Anomalies and limitations
- PDF page verification remains limited; exact model/prompt/candidate/cohort equivalence remains absent.

# 24. Tests and validation
- Validation pass: `30`; fail: `0`.

# 25. Protected inputs
- `checked=1144;changed=0`.

# 26. Final decision
- `finalization_f6b_complete`.

# 27. Recommended next phase
- F6C should perform PDF page confirmation, LastFM readiness planning, and exact-equivalence audits before any new experiment.

# Inventory summary
- Components by type: `{'numeric_result_cell': 187, 'model_family': 13, 'metric': 12, 'analysis_dimension': 8, 'model_version': 7, 'paper_observation': 7, 'dataset': 4, 'prompt_strategy': 3, 'cohort_protocol': 3, 'task': 2, 'candidate_protocol': 2, 'appendix_component': 2}`.
- Paper observations total: `7`.
