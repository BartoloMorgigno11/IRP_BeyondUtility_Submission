# Final Scientific Report: Beyond Utility Reproducibility Closure

## Abstract

This report is the final scientific closeout for the local reproducibility wrapper around *Beyond Utility: Evaluating LLM as Recommender* (arXiv `2411.00331`). The project does not claim complete exact numerical replication. It closes the reproduction core with an auditable distinction between exact, reconstructed, adapted, operationalized, diagnostic, blocked, and future-work evidence. The final F6M closure layer is generated from existing CSV/JSON artifacts only.

## Final Status

REPRODUCIBILITY CORE - CLOSED.

EXTENSION G1-G4 - COMPLETE/FROZEN.

F6M performed no LLM/API calls, no recommender training, no downloads, no raw-output repair, no deduplication, and no full LastFM extension. Later G1-G4 extension phases are separate frozen evidence layers and do not alter the F6M reproduction exactness accounting. The authoritative generated F6M closeout report is `report/phase_f6m_reproduction_closure.md`; the provenance manifest is `results/manifests/phase_f6m_reproduction_closure_manifest.json`.

## Scope And Terminology

Exact reproduction means the local artifact aligns with the paper across the material dimensions recorded by the project. Reconstruction means a paper-like counterpart exists with known artifact or preprocessing differences. Adaptation means a local interface, model, prompt, parser, denominator, or execution condition differs but remains scientifically related. Operationalization means a paper dataset/task is evaluated under an explicit local protocol. Diagnostic and extension evidence never increases the exact reproduction numerator.

Older accounting freezes remain binding: strict exact component reproduction is `9 / 73 = 12.328767%`, maximum defensible paper-component coverage is `68.493151%`, and exact numerical paper-result-cell coverage is `0.0%`. F6M closes the package; it does not promote non-exact rows to exact reproduction.

## LastFM Completion

LastFM is a paper dataset in this package. F6M supersedes older "LastFM incomplete" wording with a narrower, evidence-backed statement: LastFM local LLM ranking and reranking are complete under the 1000-user operationalized local protocol, not under an exact full paper protocol.

Ranking V4 is frozen with decision token `f6l_lastfm_v4_ranking_metrics_consolidated_and_frozen_for_reranking` and freeze hash `D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC`.

Reranking V4.1 is complete with decision token `f6l_lastfm_v4_reranking_continuation_v4_1_complete`, `4000 / 4000` aggregate final responses, `1600 / 1600` continuation final responses, and `1571 / 29 / 29` strict-valid, structural-invalid, and duplicate-token-invalid continuation counts.

## LastFM Primary Results

Primary LastFM ranking fail-strict HR@5/NDCG@5:

- Qwen2 base ranking: `0.458`, `0.323647507835`
- Qwen2 LLMRank ranking: `0.419`, `0.308222730898`
- Mistral base ranking: `0.307`, `0.223073398036`
- Mistral LLMRank ranking: `0.319`, `0.23287621762`

Primary LastFM reranking fail-strict HR@5/NDCG@5:

- Qwen2 base-derived reranking: `0.143`, `0.101374569024`
- Qwen2 LLMRank-focused reranking: `0.143`, `0.102618646645`
- Mistral base-derived reranking: `0.154`, `0.116216279854`
- Mistral LLMRank-focused reranking: `0.151`, `0.101497110768`

These values are read from `results/tables/phase_f6m_master_results_wide.csv`, which is generated from F6L source CSVs. The reranking positive-present coverage is `314 / 1000 = 0.314`; it is diagnostic only and does not replace the all-1000 primary fail-strict denominator. It does not apply to ranking.

## Research Conclusions

- The project supports a local, adapted reproduction-wrapper claim, not complete exact numerical paper replication.
- Local Qwen2 and Mistral results are substitute-model evidence and must not be described as paper-model reproduction.
- LastFM ranking and reranking are complete for the operationalized 1000-user local protocol.
- Reranking and ranking remain separate tasks with different candidate-pool and output-contract semantics.
- Current LastFM reranking-minus-ranking comparison rows are descriptive; F6M does not invent paired-test p-values.
- Profile-generation effectiveness remains unresolved and unclaimed.
- The G1-G4 extension is complete and frozen as adapted local evidence, not paper-exact replication.
- G1 versus G4 comparisons are descriptive only.
- G4 reranking performance is bounded by candidate recall; Food.com conditional G4 evidence is a small positive-present diagnostic subset.
- Longer histories are not universally better, and candidate ordering materially affects local LLM recommender behavior.
- LLMSRec was not executed as a faithful paper reproduction.

## Limitations

Exact paper numerical replication remains blocked by unavailable original raw outputs, exact processed artifacts, model snapshots, candidate ordering, and runtime conditions. Local strict JSON-index output improves auditability but differs from free-form paper prompting. LastFM uses an operationalized 1000-user local cohort in the F6M closure. Later extension evidence is separately labelled and does not become exact paper evidence. Positive-present-only reranking metrics are useful diagnostics but not primary utility.

## Artifact Map

Primary closeout artifacts:

- `results/tables/phase_f6m_master_reproducibility_matrix.csv`
- `results/tables/phase_f6m_master_results_long.csv`
- `results/tables/phase_f6m_master_results_wide.csv`
- `results/tables/phase_f6m_research_question_matrix.csv`
- `results/tables/phase_f6m_paper_claim_evidence_matrix.csv`
- `results/tables/phase_f6m_limitations_register.csv`
- `results/tables/phase_f6m_lastfm_final_status.csv`
- `results/tables/phase_f6m_statistical_assessment.csv`
- `results/tables/phase_f6m_reproduction_extension_boundary.csv`
- `results/manifests/phase_f6m_reproduction_closure_manifest.json`
- `report/g1_primary_ranking_execution.md`
- `report/g2_history_length_scientific_synthesis.md`
- `report/g3_candidate_position_scientific_synthesis.md`
- `report/g4_f1bzq_scientific_synthesis.md`
- `results/manifests/github_packaged_artifacts.json`
- `report/GITHUB_PROVENANCE_MAP.md`

## Final Boundary

The final reproduction package should be read from the F6M closure artifacts and prior source tables named by the F6M manifest. Extension results belong to the later G1-G4 frozen evidence layers and should be interpreted under their adapted local protocols. The full research repository is ready for GitHub publish review after packaging validation; the professor-facing curated submission branch has not been created.
