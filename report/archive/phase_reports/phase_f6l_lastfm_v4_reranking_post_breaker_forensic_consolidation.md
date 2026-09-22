# F6L LastFM V4 Reranking Post-Breaker Forensic Consolidation

Decision token: `f6l_lastfm_v4_reranking_post_breaker_blocked_pending_evidence`.

## Scope

- No inference, Ollama call, API call, detached process, raw overwrite, response repair, deduplication, or contract rewrite was performed.
- Audited run: `20260806_034027`.
- Operational stop token: `f6l_lastfm_v4_reranking_stopped_by_systemic_duplicate_regression`.
- Ranking baseline freeze hash: `D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC`.

## Integrity

- Planned requests: `4000`.
- Final completed parser rows: `2400`.
- Strict-valid + structural-invalid: `2352 + 48 = 2400`.
- Duplicate-token invalid: `48`.
- Malformed JSON: `0`; unknown token: `0`; hard structural invalid: `0`.
- Non-final transport timeouts: `48`.
- Pending final-response requests: `1600`.
- Pending manifest ordinals are exactly 2401-4000: `false`.
- Pending manifest ordinal ranges: `2063;2065;2067;2069;2071;2073;2075;2077;2079;2081;2083;2085;2087;2089;2091;2093;2095;2097;2099;2101;2103;2105;2107;2109;2111;2113;2115;2117;2119;2121;2123;2125;2173;2175;2177;2179;2181;2183;2185;2187;2189;2191;2193;2195;2197;2291;2293;2295;2449-4000`.
- Raw partial artifact manifest SHA-256: `AB6449BD7BE8B8288A6FB0DB5FB3F3439F616662AE93FF572065120D2B6BE030`.
- Baseline audit: `pass`.
- original_repo status: `clean`.

## Breaker

- Window 12 is contractually failed: `11/200 = 0.055`, threshold `>0.05`.
- The failed window is the first and only closed non-overlap window above 5%.
- The evidence supports a local/cell-specific duplicate-token spike, not a global systemic regression claim.

## Decision

- Final post-breaker token: `f6l_lastfm_v4_reranking_post_breaker_blocked_pending_evidence`.
- Reason: the residual final-response set is exact but not the requested contiguous manifest interval `2401-4000`; it includes 48 earlier non-final timeout attempts plus 1552 never-attempted ordinals.
- A versioned continuation needs human policy on whether those 48 timeout attempts are part of the continuation manifest. No V4.1 executable proposal was generated.

## Required Tables

- Cell coverage: `results/tables/phase_f6l_lastfm_v4_reranking_post_breaker_cell_coverage.csv`.
- Failure taxonomy: `results/tables/phase_f6l_lastfm_v4_reranking_post_breaker_failure_taxonomy.csv`.
- Window analysis: `results/tables/phase_f6l_lastfm_v4_reranking_post_breaker_window_analysis.csv`.
- Metrics: `results/tables/phase_f6l_lastfm_v4_reranking_post_breaker_metrics.csv`.
- Statistical assessment: `results/tables/phase_f6l_lastfm_v4_reranking_post_breaker_statistical_assessment.csv`.
- Decision matrix: `results/tables/phase_f6l_lastfm_v4_reranking_post_breaker_decision_matrix.csv`.
- Summary: `results/manifests/phase_f6l_lastfm_v4_reranking_post_breaker_summary.json`.
- Hashes: `results/manifests/phase_f6l_lastfm_v4_reranking_post_breaker_hashes.json`.
