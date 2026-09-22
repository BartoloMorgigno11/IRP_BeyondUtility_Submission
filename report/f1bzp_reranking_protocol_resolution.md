# F1BZP Reranking Protocol Resolution

Decision token: `F1BZP_ADAPTED_POOL_PROTOCOL_FROZEN`

## Checkpoint-Representable Catalog

- Audited dataset/model cells: `20`.
- Ready under model-representable catalog policy: `20` / `20`.
- Prior F1BZC raw-catalog blocker reassessment: `{"PRESERVED_READY_CELL":4,"RESOLVED_BY_MODEL_REPRESENTABLE_POLICY":16}`.
- The retrievable universe is the raw metadata catalog intersected with the frozen checkpoint item vocabulary.
- OOV raw metadata items are not scored and do not require invented embeddings or retraining.

## Target Representability

- Rows: `19528`.
- Status counts: `{"REPRESENTABLE_TARGET":19528}`.
- If a future target is outside the representable catalog, it is a Candidate Recall miss and is never forced into the vocabulary.

## Paper And Artifact Evidence

- Paper evidence rows: `4`.
- Top-5 x four-model classification: `PLAUSIBLE_BUT_UNRECOVERED`.
- Table 5 consistency classification: `INCONCLUSIVE_CONSISTENCY_CHECK`.
- Local evidence recovers the 20-item pool size and four source models, but not source depth, order, duplicate handling, or refill mechanics.

## Superseded F1BR Pool Component

- Defect: `F1BR_PROTOCOL_DEFECT_DISCOVERED_POST_EXECUTION`.
- F1BZC showed the frozen F1BR concatenation policy fills attempted LastFM pools entirely from BPRMF.
- Only the pool aggregation component is superseded; the source seed policy and other F1BR decisions remain frozen.

## Frozen Adapted Policy

- Policy: `ADAPTED_BALANCED_MULTI_MODEL_RERANKING_POOL`.
- Algorithm: `BALANCED_RANK_ROUND_ROBIN_UNIQUE_20`.
- Model order: `["BPRMF","GRU4Rec","SASRec","LightGCN"]`.
- Requested source depth: `20`.
- Actual max depth used: `PROTOCOL_ONLY_NOT_EXECUTED`.
- Users unable to fill 20: `PROTOCOL_ONLY_NOT_EXECUTED`.
- The policy visits models by rank round, skips duplicates, preserves each source ranking, and stops at 20 unique items.
- It is adapted, not paper-exact, and does not use target identity or positive forcing.

## Corrective Scope

- Scope rows: `20`.
- Real blockers: `{"NONE":20}`.
- Readiness statuses: `{"READY_FOR_FUTURE_PROTOCOL_CONTROLLED_REGENERATION":20}`.
- F1BZP generates no authoritative pools; future execution must regenerate source top20 over the representable catalog first.

## Validation

- Checks: `30`.
- Failures: `0`.
