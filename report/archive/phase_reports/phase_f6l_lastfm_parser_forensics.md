# Phase F6L LastFM Parser Forensics

- Decision token: `f6l_lastfm_ranking_reparse_complete_with_conservative_parser_recovery`.
- Run ID audited: `20260802_184502`.
- No inference, reranking, full-cohort, API calls, downloads, training, or raw-output mutation was performed.
- Parser V2 hash: `AFCF8936A3C8A6133A2EA76C18E2BDA36C2FA1668D66EC60E262A78EEB1F906F`.
- Raw protection: `pass`.

## Dominant Failure Taxonomy

- `mistral:7b-instruct` `base/OpenP5 title-list` `textual_ranking_not_single_bracketed_list`: `845`.
- `qwen2:7b-instruct` `LLMRank recency` `textual_ranking_not_single_bracketed_list`: `844`.
- `mistral:7b-instruct` `LLMRank recency` `textual_ranking_not_single_bracketed_list`: `749`.
- `qwen2:7b-instruct` `base/OpenP5 title-list` `textual_ranking_not_single_bracketed_list`: `742`.
- `mistral:7b-instruct` `LLMRank recency` `textual_ranking_not_single_bracketed_list`: `244`.
- `qwen2:7b-instruct` `base/OpenP5 title-list` `textual_ranking_not_single_bracketed_list`: `200`.
- `qwen2:7b-instruct` `LLMRank recency` `textual_ranking_not_single_bracketed_list`: `146`.
- `mistral:7b-instruct` `base/OpenP5 title-list` `textual_ranking_not_single_bracketed_list`: `126`.

## Conservative V2 Recovery

- `qwen2:7b-instruct` `base/OpenP5 title-list`: V2 valid `252/1000`; recovery `0.252`.
- `qwen2:7b-instruct` `LLMRank recency`: V2 valid `396/1000`; recovery `0.396`.
- `mistral:7b-instruct` `base/OpenP5 title-list`: V2 valid `249/1000`; recovery `0.249`.
- `mistral:7b-instruct` `LLMRank recency`: V2 valid `367/1000`; recovery `0.367`.

## Blind Audit Sample

- Deterministic audit sample rows: `23`.
- Holdout rows: `3977`.
- Reported examples are structural/redacted and keyed by raw response hash; positive item, positive position, HR, and NDCG were not used to define parser rules.

## RQ Correction

- `F6L-LASTFM-RQ-UTILITY`: `unresolved_for_original_paper_sample` - V2 recovered enough complete cases for local-only complete-case estimates.
- `F6L-LASTFM-RQ-PROMPT`: `mixed_prompt_dependent` - Only V2-valid complete cases can compare prompts.
- `F6L-LASTFM-RQ-BEYOND-UTILITY`: `unresolved_for_original_paper_sample` - Beyond-utility metrics require valid recommendation rows; invalid rows are not imputed in the main scientific table.

## Artifacts

- `results/tables/phase_f6l_lastfm_parser_failure_taxonomy.csv`
- `results/tables/phase_f6l_lastfm_parser_failure_summary.csv`
- `results/tables/phase_f6l_lastfm_reparse_v2_results.csv`
- `results/tables/phase_f6l_lastfm_reparse_v2_complete_case_metrics.csv`
- `results/manifests/phase_f6l_lastfm_parser_forensics_manifest.json`