# Final Full Report Audit Through Frozen G4

- Audit timestamp: 2026-09-06T15:05:39+02:00
- Starting HEAD: `8df8c60e39d758049041cb37ba9bf22385a7ec62`
- Scope: full LaTeX manuscript under `report/latex/Reproduction_Study_of___Beyond_Utility__Evaluating_LLM_as_Recommender`
- Scientific evidence source: frozen repository artifacts only
- Inference status: no Ollama/API calls, no new experiments, no metric recomputation, no new statistical tests
- Experimental status: reproduction, G1, G2, G3, and G4 are complete/frozen; no further experimental phase is planned for this manuscript

## Evidence Used

- Reproduction closure and reconciliation: `report/phase_f6m_reproduction_closure.md`, `report/reproduction_final_scientific_reconciliation.md`, and associated R1--R5 reports.
- G1 execution and metrics: `report/g1_primary_ranking_execution.md`, `results/tables/g1_primary_ranking_*.csv`, `results/manifests/g1_primary_ranking_*.json`.
- G2 retention, universe, metrics, statistics, and synthesis: `report/g2_history_length_*.md`, `results/tables/g2_history_length_*.csv`, `results/manifests/g2_history_length_*.json`.
- G3 protocol, retention, universe, metrics, statistics, and synthesis: `report/g3_candidate_position_*.md`, `results/tables/g3_candidate_position_*.csv`, `results/manifests/g3_candidate_position_*.json`.
- G4 protocol, retention, universe, metrics, statistics, and final synthesis: `report/g4_f1bzq_*.md`, `results/tables/g4_f1bzq_*.csv`, `results/manifests/g4_f1bzq_*.json`.
- Final G4 synthesis authority: `report/g4_f1bzq_scientific_synthesis.md`, `results/tables/g4_f1bzq_synthesis_*.csv`, `results/manifests/g4_f1bzq_scientific_synthesis_manifest.json`.

## Chapter Audit

| Manuscript part | Status | Corrections / decision |
|---|---|---|
| Title and front matter | UPDATED | Abstract replaced with final through-G4 abstract. It separates reproduction from extension, summarizes G1--G4 findings, and avoids raw-count overload. |
| Introduction | UPDATED | Scope and contributions now state the completed four-part extension: G1 primary ranking, G2 history length, G3 candidate position, and G4 F1BZQ re-ranking. No extension phase is retroactively promoted to reproduction. |
| Background | OK | No stale phase status, broken citations, or G4-specific numerical claims found. |
| Target Study | OK | Preserves target-paper description and does not insert local G1--G4 extension conditions into the historical paper protocol. |
| Methodology | OK | Existing evidence hierarchy remains valid: frozen machine artifacts override prose summaries. |
| Experimental Setup | OK | Reproduction setup remains scoped to the local reproduction/R5 recovery. Extension protocols stay in the Extension chapter and appendices. |
| Reproduction Results | OK | Reproduction results remain separate from extension results. No G4 extension result is inserted into reproduction-only tables. |
| Reconciliation | OK | Reproduction verdict remains paper-aligned adapted reproduction, not paper-exact replication. |
| Discussion | UPDATED | Re-ranking discussion updated to reflect G4 candidate-recall bottleneck and descriptive-only G1-vs-G4 pipeline comparisons. |
| Threats to Validity | UPDATED | Added complete-study limitations for Gemma2/Food.com extension status, G4 F1BZQ adapted pools, low G4 candidate recall, Food.com conditional `n=4`, CI-only statistics, no Holm/p-values where unfrozen, and descriptive-only G1-vs-G4 comparisons. |
| Reproduction Conclusions | OK | Preserves reproduction-only verdict and does not claim that G1--G4 reproduce the historical paper matrix. |
| Extension | UPDATED | Integrated frozen G4 objective, dimensions, execution accounting, candidate recall, denominator semantics, internal CI counts, prompt/model conclusions, Food.com `n=4` diagnostic status, and G1-vs-G4 descriptive-only policy. |
| Overall Conclusions | UPDATED | Rewritten as final global synthesis through G4, covering reproduction limits, G1 model/prompt dependence, G2 non-monotonic history effects, G3 position sensitivity, G4 candidate-generation bottleneck, implications, limitations, and future work. |
| Appendices | UPDATED | Updated environment, model identity, command, candidate, metric, result-table, failure-taxonomy, git-provenance, validation, inventory, and phase-cross-reference notes through G4. |
| Bibliography | OK | Static citation-key validation found 205 citation uses, 30 BibTeX entries, 0 missing citation keys, and 0 duplicate BibTeX keys. |

## Frozen Numerical Integration

### Reproduction

- Final reproduction verdict remains: broad paper-aligned adapted reproduction, not demonstrated paper-exact numerical replication.
- Missing/non-identical evidence remains explicitly identified: historical model snapshots, sampled cohorts, candidate identities/orderings, raw generated outputs, runtime state, and some metric-support artifacts.
- No reproduction-only table was changed to include G1--G4 extension data.

### G1

- Objective: cross-dataset / cross-model / prompt-conditioned primary LLM ranking.
- Datasets: Beauty, Sports, MovieLens-1M, LastFM, Food.com.
- Models: `qwen2:7b-instruct`, `mistral:7b-instruct`, `gemma2:9b`.
- Prompts: BASE and LLMRANK.
- Logical requests: 29,292.
- Complete cells: 30/30.
- Reliability totals retained from frozen G1 tables: Qwen2 9,764 planned / 9,695 valid / 69 invalid; Mistral 9,764 / 9,585 / 179; Gemma2 9,764 / 9,744 / 20.
- Prompt reliability retained: BASE 14,646 planned / 14,599 valid / 47 invalid; LLMRANK 14,646 / 14,425 / 221.
- Failure classes retained: 238 invalid-length responses and 30 duplicate-index responses.

### G2

- Objective: history-length sensitivity under nested paired cohorts.
- Logical requests: 88,050.
- Exact G1 reuse: 18,660.
- New G2 terminal/backend evidence: 69,390.
- Analytic strict-valid rows: 86,768.
- Analytic invalid rows: 1,282.
- Backend errors: 0.
- Frozen statistical method: paired-user bootstrap mean delta, 10,000 resamples, seed 20260901, two-sided 95 percent percentile CI, no primary p-value method.
- Primary `H=10-H=0` synthesis retained: descriptive 80 / 1 / 9; CI 57 positive / 1 negative / 32 NS.
- Long `H=20-H=10` synthesis retained: descriptive 27 / 4 / 59; CI 3 positive / 18 negative / 69 NS.
- Manuscript wording rejects universal monotonic-history and universal `H=10` optimum claims.

### G3

- Objective: candidate-position/order sensitivity.
- Design: FIRST versus RANDOM over five datasets, three models, two prompts, 200 deterministic users per dataset.
- Logical/backend requests: 12,000.
- Order accounting: 6,000 FIRST and 6,000 RANDOM.
- Response validity: 11,834 strict-valid, 166 invalid, 0 backend errors.
- Statistical method: paired-user bootstrap mean delta `FIRST_minus_RANDOM`, 10,000 resamples, seed 20260901, two-sided 95 percent percentile CI.
- Coverage: 90/90 FIRST-vs-RANDOM utility comparisons.
- CI-supported synthesis retained: overall 85 positive / 1 negative / 4 NS.
- CandDif retained as descriptive only; no CandDif significance test, no p-values, no Holm correction.
- Manuscript wording rejects universal LLMRANK reduction of position bias.

### G4

- Objective: LLM re-ranking over frozen F1BZQ balanced 20-item candidate pools.
- Scientific status: adapted extension, not paper-exact traditional-model re-ranking reproduction.
- Candidate source: F1BZQ only; no superseded F1BR/F1BZC pools and no old F6M response reuse.
- Dimensions: five datasets, three models, BASE + LLMRANK, 4,882 users, 29,292 logical requests, 30 cells.
- Execution accounting: 28,986 strict-valid, 306 structural invalid, 255 invalid-length, 51 duplicate-index, 0 backend errors.
- Candidate recall: Beauty 49/990; Sports 28/892; MovieLens-1M 193/1,000; LastFM 469/1,000; Food.com 4/1,000.
- Denominator semantics: candidate recall uses all planned users; unconditional HR@5/NDCG@5/MRR use all planned users with invalid and positive-absent rows as zero; conditional HR@5/NDCG@5 use positive-present users only with invalid rows as zero inside that subset.
- Internal G4 statistics: 225 inferential rows; 29 positive, 24 negative, 172 NS.
- Prompt family: 2 positive / 13 negative / 60 NS.
- Model-pair family: 27 positive / 11 negative / 112 NS.
- Dataset family counts retained: Beauty 0/0/45; Sports 4/5/36; MovieLens-1M 10/5/30; LastFM 15/14/16; Food.com 0/0/45.
- G1-vs-G4 rows: 90 descriptive pipeline-level rows only because candidate sources differ; no causal re-ranking effect claim is made.

## Global Checks

- Static LaTeX input check: PASS, 13/13 `\input{}` files present.
- Static label check: PASS, 197 labels, 0 duplicate labels.
- Static reference check: PASS, 52 reference uses, 0 missing labels.
- Static bibliography check: PASS, 205 citation uses, 30 BibTeX entries, 0 missing citation keys, 0 duplicate BibTeX keys, 0 unused BibTeX entries.
- Static environment-balance check: PASS, all counted `\begin{...}` / `\end{...}` environments balanced.
- Stale-status scan: PASS. No remaining manuscript occurrences assert that G4 is unfinished, unreported, not started, or future work.
- Reproduction/extension separation: PASS. G1--G4/Gemma2/Food.com extension statements are confined to extension-facing sections and are not inserted into the reproduction-only experiment matrix.
- Statistical wording: PASS. Manuscript uses bootstrap-CI wording where frozen, does not invent p-values, does not apply Holm where no frozen p-value method exists, and does not call NS rows proof of no effect.
- Candidate-recall wording: PASS. G4 distinguishes candidate-stage recall, unconditional utility, and positive-present conditional utility.
- G1-vs-G4 wording: PASS. All G1-vs-G4 references are descriptive pipeline-level only.
- Table/figure review: PASS. No raw metric/statistical dump was added. G4 is represented through compact summary tables for execution reliability, candidate recall, and CI decision counts. No new figure was necessary.
- Frozen artifact integrity: PASS by scope. No frozen scientific artifact, config, runner, evidence file, metric table, statistical table, or synthesis artifact outside the LaTeX/audit report set was edited.

## Build Validation

- Local build command considered: `pdflatex main.tex`, `bibtex main`, `pdflatex main.tex`, `pdflatex main.tex`.
- Build status: BLOCKED IN LOCAL ENVIRONMENT.
- Reason: `pdflatex`, `bibtex`, `latexmk`, `tectonic`, `xelatex`, and `lualatex` are not available on PATH. `chktex` and `lacheck` are also unavailable.
- Existing PDF status: no PDF file is present under the LaTeX report directory.
- Consequence: source-level LaTeX/citation/reference validation was completed, but regenerated PDF build, page count, and visual inspection could not be produced in this environment.

## Remaining Limitations And Future Work

- Remaining limitations are scientific, not unfinished experimental sections: missing historical artifacts, local model/backend substitution, candidate-pool dependence, structural invalidity, G4 adapted F1BZQ pools, low G4 candidate recall, small conditional denominators, and CI-only statistics without frozen p-values for some phases.
- Intentional future work is evidence-oriented: release or recover raw LLM responses, exact sampled cohorts, candidate-pool snapshots, traditional-model checkpoints, prompt templates, parser code, model/backend versions, and metric-support data.
- No experimental section remains marked unfinished.
