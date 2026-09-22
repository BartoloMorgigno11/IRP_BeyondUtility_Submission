# Finalization F2 Runtime Verification

## Executive Summary

- Decision: `finalization_f2_complete`.
- Python gate: `pass`.
- Safe test tiers executed: `3`.
- Safe test tier failures: `0`.
- Protected input hash failures: `0`.
- Canonical registry rows: `314`.
- Apparent result conflict groups: `82`.

## Environment

- Official interpreter candidate: `.venv/Scripts/python.exe`.
- `.venv/pyvenv.cfg` records Python `3.12.10`; `environment.yml` records `python=3.10`.
- The mismatch is documented as an environment divergence, not corrected in F2.

## Test Execution

- `LEVEL0`: status `pass`, return code `0`, static tests `111`.
- `LEVEL1`: status `pass`, return code `0`, static tests `32`.
- `LEVEL2`: status `pass`, return code `0`, static tests `114`.

## Protected Inputs

- Protected paths hashed before tests: `1144`.
- All unchanged: `true`.
- Hash algorithm: `sha256`.

## Canonical Registry And Conflicts

- Registry fields include cohort, candidate set, evaluation policy and denominator policy.
- MostPop ML-1M Plan B and Plan E values are classified as non-conflicting when the cohort differs.
- A conflict is true only when the natural comparison fields match and the values differ.

## Authoritative Sources

- Rank `1` `results/llm/canonical_local_full_matrix/requests.jsonl`: raw evidence for LLM requests.
- Rank `1` `results/llm/canonical_local_full_matrix/responses.jsonl`: raw evidence for LLM responses.
- Rank `2` `results/tables/canonical_local_llm_full_matrix_cell_metrics.csv`: LLM cell utility metrics.
- Rank `2` `results/tables/canonical_local_llm_utility_metrics.csv`: LLM utility policy table.
- Rank `2` `results/tables/traditional_baselines_metrics.csv`: traditional baseline metrics on Plan E2 manifests.
- Rank `2` `results/tables/plan_e_bprmf_all_datasets_metrics.csv`: Plan E BPRMF metrics.
- Rank `2` `results/tables/plan_e_gru4rec_all_datasets_metrics.csv`: Plan E GRU4Rec metrics.
- Rank `2` `results/tables/plan_e_sasrec_all_datasets_metrics.csv`: Plan E SASRec metrics.

## Worktree

- Classified worktree/artifact rows: `621`.
- F2 did not stage, commit, checkout, reset, clean, download, call APIs, call Ollama, or train models.

## Completion Criteria

- Interpreter identified: yes if Python gate is pass.
- Dependencies inventoried: yes.
- LEVEL 0-2 safe tests: pass only if every tier return code is zero.
- Protected inputs unchanged: pass only if every hash validation row is unchanged.
- Existing outputs overwritten: no, enforced by F2 output guard.

## Next Action

Review the generated registry, conflicts and artifact classification before staging any F2 files.
