# R4 Three-Way Beyond-Utility Reconciliation

## Final Decision

`REPRODUCTION_R4_THREE_WAY_RECONCILIATION_COMPLETE`

## Purpose

R4 reconciles what the paper reports, what frozen official author artifacts support, and what the frozen local reproduction produced. It performs no new inference, training, data download, candidate generation, or recovery execution.

## Frozen evidence layers

- Paper layer: `paper_atomic_contract.csv` and `extension_e1_metric_contract.csv`.
- Official layer: frozen R1/R2 tables, especially `reproduction_r2_official_metric_calculability_master.csv`.
- Local layer: frozen R3 tables, especially `reproduction_r3_all_metric_cells_master.csv` and R3 specialized Beyond-Utility tables.

## Three-way methodology

Rows are reconciled at the finest recovered granularity. Model, task, history, candidate-position, generation, profile, cohort, and cutoff fields are not forced together when they differ. Official values and local values remain separate columns.

## Three-way accounting

- paper metric cells: 1361
- official evidence cells: 1266
- local evidence cells: 5965
- numeric agreement rows: 0
- validation: 20 pass / 0 fail

## Utility baseline

HR@5 and NDCG@5 serve as context. R2 independently recalculates compatible official utility rows, and R3 contains local utility rows under explicit denominator policies. Because official models and local models do not share exact identity, R4 treats utility as adapted/contextual rather than exact paper-official-local protocol closure.

## Beyond-Utility executive summary

- SelfInformation: ADAPTED_REPRODUCTION=4; remaining gaps=0.
- Serendipity: ADAPTED_REPRODUCTION=4; remaining gaps=0.
- ACLT/APLT: ADAPTED_REPRODUCTION=4; remaining gaps=4.
- ARP: ADAPTED_REPRODUCTION=4; remaining gaps=0.
- PopREO: BLOCKED_MISSING_INPUT=4; remaining gaps=4.
- ItemCoverage: ADAPTED_REPRODUCTION=4; remaining gaps=0.
- Shannon: BLOCKED_MISSING_INPUT=4; remaining gaps=4.
- Jaccard: BLOCKED_MISSING_INPUT=4; remaining gaps=4.
- Gini: BLOCKED_MISSING_INPUT=4; remaining gaps=4.
- DPD: BLOCKED_MISSING_INPUT=4; remaining gaps=4.
- Jain: BLOCKED_MISSING_INPUT=4; remaining gaps=4.
- History Length Sensitivity: DIAGNOSTIC_REPRODUCTION_ONLY=4; remaining gaps=4.
- Candidate Position Bias: DIAGNOSTIC_REPRODUCTION_ONLY=4; remaining gaps=6.
- Hallucination: ADAPTED_REPRODUCTION=4; remaining gaps=0.
- Generation-Involved Performance: ADAPTED_REPRODUCTION=2; BLOCKED_MISSING_OUTPUT=2; remaining gaps=4.

## SelfInformation

Local SelfInformation is adapted because paper unique-user self-information and local/train-frequency primitives differ. It supports novelty interpretation but not exact scalar parity.

## Serendipity

Serendipity remains adapted because baseline/reference dependence differs. Local MostPop-reference evidence is interpretable but not paper-exact.

## ACLT/APLT

Paper APLT and official/local ACLT naming are not merged silently. R4 treats them as semantically related long-tail metrics with adapted comparability.

## ARP

ARP is locally covered under train-popularity semantics and supports popularity trend discussion, with official values mostly stored-only.

## PopREO

PopREO has four local gaps caused by missing support inputs/groups; no local value is substituted.

## ItemCoverage

ItemCoverage is separated from local CatalogCoverage and treated as the strongest diversity reproduction family.

## Shannon

Shannon remains blocked by category/distribution support inputs.

## Jaccard

Jaccard/OIC semantics require official candidate-overlap support; generic ILD is not substituted.

## Gini

Gini remains blocked by exposure denominator/support-input uncertainty.

## DPD

DPD requires cold/hot or active/inactive user group definitions; R4 does not invent membership.

## Jain

Jain remains missing locally and adds fairness completeness but is lower priority than LLM-specific recovery.

## History Length Sensitivity

Local history curves are diagnostic/binned or natural grids; no paper-grid interpolation is fabricated.

## Candidate Position Bias

Local evidence supports position sensitivity only diagnostically. Paper-compatible CandDif is a P0 recovery target.

## Hallucination

R4 separates textual hallucination, candidate violations, and structural invalidity. Constrained outputs change the meaning of the comparison.

## Generation-Involved Performance

Downstream generation-involved recommendation output exists for Beauty/Sports but four paper-relevant cells remain missing.

## Dataset synthesis

### Beauty
- utility: ADAPTED_BUT_SUPPORTIVE.
- novelty: ADAPTED_BUT_SUPPORTIVE.
- popularity: MIXED_RESULT.
- diversity: MIXED_RESULT.
- fairness: NOT_REPRODUCED.
- history_sensitivity: DIAGNOSTIC_SUPPORT_ONLY.
- candidate_position: DIAGNOSTIC_SUPPORT_ONLY.
- hallucination: ADAPTED_BUT_SUPPORTIVE.
- generation_involved: ADAPTED_BUT_SUPPORTIVE.

### Sports
- utility: ADAPTED_BUT_SUPPORTIVE.
- novelty: ADAPTED_BUT_SUPPORTIVE.
- popularity: MIXED_RESULT.
- diversity: MIXED_RESULT.
- fairness: NOT_REPRODUCED.
- history_sensitivity: DIAGNOSTIC_SUPPORT_ONLY.
- candidate_position: DIAGNOSTIC_SUPPORT_ONLY.
- hallucination: ADAPTED_BUT_SUPPORTIVE.
- generation_involved: ADAPTED_BUT_SUPPORTIVE.

### ML-1M
- utility: ADAPTED_BUT_SUPPORTIVE.
- novelty: ADAPTED_BUT_SUPPORTIVE.
- popularity: MIXED_RESULT.
- diversity: MIXED_RESULT.
- fairness: NOT_REPRODUCED.
- history_sensitivity: DIAGNOSTIC_SUPPORT_ONLY.
- candidate_position: DIAGNOSTIC_SUPPORT_ONLY.
- hallucination: ADAPTED_BUT_SUPPORTIVE.
- generation_involved: NOT_REPRODUCED.

### LastFM
- utility: ADAPTED_BUT_SUPPORTIVE.
- novelty: ADAPTED_BUT_SUPPORTIVE.
- popularity: MIXED_RESULT.
- diversity: MIXED_RESULT.
- fairness: NOT_REPRODUCED.
- history_sensitivity: DIAGNOSTIC_SUPPORT_ONLY.
- candidate_position: DIAGNOSTIC_SUPPORT_ONLY.
- hallucination: ADAPTED_BUT_SUPPORTIVE.
- generation_involved: NOT_REPRODUCED.

## Paper claim reconciliation

- PARTIALLY_SUPPORTED: LLM recommender utility should be assessed with HR@5 and NDCG@5, but utility is not the whole evaluation. Interpretation: Utility is reproduced as context, not as exact paper-protocol closure.
- QUALITATIVELY_SUPPORTED: Accuracy alone is insufficient because novelty, popularity, diversity, fairness, and LLM-specific behavior change the scientific interpretation. Interpretation: The multi-metric conclusion is scientifically supported even without exact scalar equality.
- PARTIALLY_SUPPORTED: LLMs exhibit candidate-position sensitivity. Interpretation: Local evidence supports position sensitivity qualitatively, but exact local CandDif remains a P0 gap.
- PARTIALLY_SUPPORTED: Recommendation behavior changes with user-history length. Interpretation: Trend evidence is diagnostic and protocol-qualified.
- PROTOCOL_DEPENDENT: LLM recommendation outputs may hallucinate items. Interpretation: Comparable only for free-text paper-style runs; structured invalidity is separate evidence.
- PARTIALLY_SUPPORTED: Generated profiles can affect downstream recommendation performance. Interpretation: Beauty/Sports evidence is informative; four-dataset closure would require targeted recovery.
- QUALITATIVELY_SUPPORTED: Novelty/popularity conclusions require metrics such as SelfInformation, Serendipity, APLT/ACLT, ARP, and PopREO. Interpretation: The direction of novelty/popularity analysis is supported under adapted semantics.
- PROTOCOL_DEPENDENT: Diversity and fairness require separate analysis from utility. Interpretation: The need for these families is supported; exact scalar reproduction is not.

## Strong reproductions

No Beyond-Utility family is upgraded to paper-exact local reproduction. ItemCoverage and utility are the strongest adapted/contextual reproductions.

## Adapted but supportive results

SelfInformation, Serendipity, ACLT/APLT, ARP, ItemCoverage, Hallucination, and Beauty/Sports generation-involved performance are scientifically informative under adapted semantics.

## Diagnostic-only results

Candidate-position bias and history-length sensitivity provide diagnostic local support but are not paper-compatible scalar reproductions.

## Contradictions / divergent evidence

R4 finds no clean contradiction. Divergence is dominated by protocol, model, support-input, and output-format differences.

## Protocol-limited comparisons

The global status remains `CRITICAL_EXPERIMENTAL_PATH_PROTOCOL_LIMITED` because preprocessing, candidate ordering, model identity, output format, and supporting metadata differences remain material.

## 54-gap prioritization

P0_MUST_CLOSE=6; P1_HIGH_VALUE=8; P2_USEFUL_BUT_OPTIONAL=32; P3_REDUNDANT_OR_LOW_VALUE=8

## ML-1M CandDif P0 status

`ML1M_CANDDIF_P0_STATUS = IN_SCOPE_ALREADY_ACCOUNTED`

ML-1M has diagnostic CandDif rows but no paper-compatible local CandDif. R4 records `remaining_canddif_gaps=0` for ML-1M, so ML-1M is accounted outside the six P0 CandDif recovery rows and its diagnostics are not upgraded to the paper FIRST vs RANDOM contract.

## Experiments worth recovering

- R4-GAP-0002: Beauty CandDif-HR (P0_MUST_CLOSE) - Run paired candidate-position outputs only for the missing paper-compatible cells.
- R4-GAP-0003: Beauty CandDif-NDCG (P0_MUST_CLOSE) - Run paired candidate-position outputs only for the missing paper-compatible cells.
- R4-GAP-0005: Beauty Generation-involved profile recommendation (P1_HIGH_VALUE) - Recover only the missing controlled paper-grid outputs if R5 is approved.
- R4-GAP-0007: Beauty History-length sensitivity (P1_HIGH_VALUE) - Recover only the missing controlled paper-grid outputs if R5 is approved.
- R4-GAP-0016: LastFM CandDif-HR (P0_MUST_CLOSE) - Run paired candidate-position outputs only for the missing paper-compatible cells.
- R4-GAP-0017: LastFM CandDif-NDCG (P0_MUST_CLOSE) - Run paired candidate-position outputs only for the missing paper-compatible cells.
- R4-GAP-0019: LastFM Generation-involved profile recommendation (P1_HIGH_VALUE) - Recover only the missing controlled paper-grid outputs if R5 is approved.
- R4-GAP-0021: LastFM History-length sensitivity (P1_HIGH_VALUE) - Recover only the missing controlled paper-grid outputs if R5 is approved.
- R4-GAP-0031: ML-1M Generation-involved profile recommendation (P1_HIGH_VALUE) - Recover only the missing controlled paper-grid outputs if R5 is approved.
- R4-GAP-0033: ML-1M History-length sensitivity (P1_HIGH_VALUE) - Recover only the missing controlled paper-grid outputs if R5 is approved.
- R4-GAP-0042: Sports CandDif-HR (P0_MUST_CLOSE) - Run paired candidate-position outputs only for the missing paper-compatible cells.
- R4-GAP-0043: Sports CandDif-NDCG (P0_MUST_CLOSE) - Run paired candidate-position outputs only for the missing paper-compatible cells.
- R4-GAP-0045: Sports Generation-involved profile recommendation (P1_HIGH_VALUE) - Recover only the missing controlled paper-grid outputs if R5 is approved.
- R4-GAP-0047: Sports History-length sensitivity (P1_HIGH_VALUE) - Recover only the missing controlled paper-grid outputs if R5 is approved.

## Recovery request count validation

- CandDif: raw=2400; duplicates=1200; existing reusable=0; unique new=1200; status=PLANNING_ESTIMATE.
- History: raw=2400; duplicates=0; existing reusable=0; unique new=2400; status=PLANNING_ESTIMATE.
- Generation: raw=800; duplicates=0; existing reusable=0; unique new=800; status=PLANNING_ESTIMATE.
- Total proposed unique new requests: 4400.
- `R4_RECOVERY_COUNT_STATUS = ESTIMATED_NOT_YET_DEDUPLICATED`.

The 4400 figure is a family-level planning estimate after known CandDif HR/NDCG reuse. It is not proof of an exact deduplicated prompt/user/candidate request manifest.

## Experiments not worth recovering first

P2/P3 gaps are mostly offline support-input, redundant diagnostic, or lower-marginal-value completeness work. They should not trigger broad new inference.

## New inference decision

`TARGETED_NEW_INFERENCE_RECOMMENDED`

New inference is recommended only for P0/P1 LLM-specific gaps if the project wants paper-compatible local CandDif, exact history-grid evidence, or missing generation-involved downstream outputs. It is not required to make a defensible protocol-limited reproduction claim.

## Implications for final reproduction closure

R4 answers the scientific reproduction question: the Beyond-Utility conclusions are mostly adapted/protocol-qualified and qualitatively supported, while exact local paper-protocol closure remains limited to targeted recoveries rather than all 54 gaps.
