# R1 Original Repository Result Recovery

Label: POST-F6M TOTAL REPRODUCIBILITY CLOSURE EVIDENCE

Decision: `REPRODUCTION_R1_OFFICIAL_ARTIFACT_RECOVERY_COMPLETE`

## Scope

R1 exhaustively inventories scientific result artifacts currently present in
`original_repo/EvaLLMasRecommender` without modifying or extracting into the
official checkout. It recovers official artifact evidence only. It does not
claim paper-cell equivalence and does not reopen F6M.

## Metric Serialization Contract

Official metric vector order recovered from `eva/codes/llm_metrics.py`:

`HR, NDCG, ACLT, ARP, PopREO, Serendipity, SelfInformation, Gini, DPD, Jain, ItemCoverage, Shannon, Jaccard, Hallucination`

The official serialization writes a headerless one-value-per-row CSV through
`pd.DataFrame([final_result]).T.to_csv(...)`. ACLT is preserved under the
official code name and related to paper APLT/long-tail wording in the contract
table.

## Artifact Recovery Summary

- Direct artifacts inventoried: 26
- ZIP archives found: 3
- ZIP members inspected: 364
- Official `_5.csv` artifacts found: 105
- Fully decoded CSVs: 105
- Partially decoded CSVs: 0
- Unresolved/format-conflict CSVs: 0
- Total official metric values decoded: 1470
- JSON artifacts inventoried: 226

## Dataset Evidence

- Beauty: RESULT_ARTIFACT_PRESENT (Official result archive or result member found.)
- Sports: RESULT_ARTIFACT_PRESENT (Official result archive or result member found.)
- ML-1M: REFERENCE_ONLY (README references MovieLens-1M/raw data, but no official result artifact was found.)
- LastFM: REFERENCE_ONLY (README references LastFM/raw data, but no official result artifact was found.)

## Beauty

Models, tasks, history lengths, profile modes, and candidate/order conditions
are recorded in the decoded tables and artifact inventory. Beauty is the
largest official result source in R1.

## Sports

Sports has official result evidence, but it does not mirror Beauty exactly.
Coverage differences are recorded by source member and condition.

## ML-1M and LastFM

No official ML-1M or LastFM result artifact was recovered from the official
checkout during R1. README references are preserved as reference-only evidence.
Local adapted artifacts outside `original_repo` are not used as substitutes.

## History-Length Sensitivity

Recovered rows: 462. These rows are official artifact values
grouped by filename-evidenced history length. R1 does not map them to paper
figures.

## Generation/Profile

Recovered rows: 546. Recommendation model and profile
generator are kept separate.

## Candidate Position / Order

Recovered evidence rows: 403. Official artifacts expose
candidate/order variants and direct HR/NDCG values, but CandDif is not directly
serialized in the `_5.csv` vectors. R2 must handle pairing and formula mapping.

## Hallucination

Official hallucination values are decoded from metric position 14. This is the
official title-to-item matching hallucination semantics, not local structural
validity.

## Metric-Family Coverage

Coverage status counts: CODE_ONLY=18, OFFICIAL_VALUES_PRESENT=18

## Unresolved Artifacts

- None

## Interpretation

The official repository contains substantially more usable Beauty/Sports result
evidence than represented by the historical F6M closure. This does not by
itself eliminate the `CRITICAL_EXPERIMENTAL_PATH_PROTOCOL_LIMITED`
classification because paper-cell identity, preprocessing artifacts, exact
cohorts, candidate pools, and model/runtime equivalence still require R2
reconciliation.
