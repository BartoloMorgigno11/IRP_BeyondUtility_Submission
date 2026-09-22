# F6G2 Output-Length Contract Resolution

## Decision

Conflict outcome: `D_repository_derived_top10_paper_unspecified`.

Operational decision: the maximum-fidelity executable contract is a 20-item candidate pool, a 10-item returned recommendation list, and top-5 metrics. This is `repository_derived_not_paper_specified`, not paper-explicit.

F6G1/F6G2 full-20 request/shard manifests are therefore superseded for execution. They must not be edited in place. A new corrective offline freeze should produce new top-10 request and shard manifests before any LastFM re-ranking inference is attempted.

No inference was run during this audit.

## Authority-Ordered Evidence

1. Main paper. Section 3.2 defines ranking and re-ranking as selecting top-K recommendations from a candidate set, but does not assign a numeric output-list length. Section 4.1.3 is explicit that ranking uses 1 positive plus 19 negatives, and that re-ranking also uses candidate pools of size 20. Table 5 reports re-ranking utility at HR@5 and NDCG@5.

2. Appendix and supplementary material. Appendix A defines metrics over top-K recommendation lists and includes NDCG@K, candidate-position-bias metrics, and hallucination over K, but does not set K as the output-list length. It does not resolve top-10 versus top-20 output length.

3. Original repository templates. Base/OpenP5 and LLMSRec-style templates request `top 10` from the candidates. LLMRank templates request ranking all 20 candidate products. This creates a template-level LLMRank conflict, not a paper-explicit output contract.

4. Original parser/evaluation code. `llms.py` system messages require choosing 10 items. `output_refine.py` normalizes to `target_len`, and `zero_shot.py` calls it with `10`. `llm_metrics.py` defaults to `cand_len=20`, `metric_len=5`, and checks/parses LLM lists at length 10, truncating or padding to 10 before evaluation.

5. Official released request/output files. Released prompt/result paths use `candidate20`; saved metric CSVs use suffix `_5.csv`. Released Base and multi-candidate re-ranking outputs are evaluated by the top-10 parser. LLMRank released prompts include rank-20 user text, but their official metric artifacts still route through the same top-10 evaluator.

6. F6E1 prompt freeze. `phase_f6e1_prompt_strategy_spec.csv` and `phase_f6e1_parser_contract.csv` record candidate count 20 and official parser expected count 10 for Base, LLMRank, and LLMSRec recommendation steps. The LLMRank note already records that prompt text allows more than the parser consumes.

7. F6E3 exact textual pilot. The exact textual prompt pilot preserved paper-like prompt wording but did not establish a successful top-20 executable parser contract.

8. F6E3B indexed adapter. `phase_f6e3b_request_manifest.csv` freezes `json_ranking_1based_top10_strict_v1` with `candidate_count=20`. This is an adapted extension aligned to the official executable top-10 parser behavior.

9. F6E4A/F6E4B request contract. F6E4A/F6E4B inherit the F6E3B strict top-10 indexed adapter and should remain unchanged.

10. F6G1 logical/request/shard manifests. F6G1 future requests require JSON arrays with min/max 20 and parser `indexed_json_top20_unique_candidates`. That is not the official executable contract recovered here.

11. F6G2 and F6G2R reports/manifests. F6G2 and F6G2R correctly blocked inference because F6G1 top-20 conflicts with prior top-10 adapter evidence.

## Recovered Contract

The explicit matrix is in `results/tables/phase_f6g2_output_length_contract_resolution.csv`.

Summary:

| Task | Strategy | Candidate Pool | Requested Output | Parser Expected | Metric Cutoff | Complete Pool Required |
| --- | --- | ---: | --- | ---: | ---: | --- |
| Ranking | Base/OpenP5 | 20 | 10 | 10 | 5 | No |
| Ranking | LLMRank | 20 | 20 in user template, 10 in executable wrapper/parser | 10 | 5 | Template says yes; executable contract no |
| Ranking | LLMSRec | 20 | 10 | 10 | 5 | No |
| Re-ranking | Base/multi-candidate | 20 | 10 | 10 | 5 | No |
| Re-ranking | LLMRank | 20 | 20 if template-only, 10 under shared executable parser | 10 | 5 | Template says yes; executable contract no |
| Re-ranking | LLMSRec | 20 | 10 | 10 | 5 | No |

Ranking and re-ranking use the same recovered executable output length: 10.

Base, LLMRank, and LLMSRec use the same recovered executable parser length: 10. LLMRank remains separately flagged because its prompt text says to rank all 20 candidates.

The paper and repository agree on candidate pool size 20 and top-5 reported metrics. The paper does not explicitly specify the returned output-list length, so the top-10 decision is repository-derived and must not be labeled paper-explicit.

## F6G1/F6G2 Disposition

The F6G1/F6G2 top-20 request contract is `incorrect_or_superseded_for_execution` under this audit.

Do not modify frozen F6G1/F6G2 artifacts. Do not modify F6E3B/F6E4A/F6E4B.

F6G1 manifests cannot be reused unchanged because the request schema requires exactly 20 indices and names the top-20 parser. The logical candidate pools and source-provenance evidence can be reused as input evidence for a new freeze, but new request and shard manifests must be generated with:

- `candidate_pool_size=20`
- `requested_output_length=10`
- `parser_expected_length=10`
- strict unique one-based candidate indices in range 1..20
- `metric_cutoff_K=5`
- explicit label `repository_derived_not_paper_specified`

Until that corrective offline freeze exists, LastFM re-ranking inference remains blocked.

## Sources

- arXiv HTML v1, Sections 3.2, 4.1.3, 4.8, Appendix A: https://arxiv.org/html/2411.00331v1
- `original_repo/EvaLLMasRecommender/eva/templates/template.txt`
- `original_repo/EvaLLMasRecommender/eva/templates/llmrank.txt`
- `original_repo/EvaLLMasRecommender/eva/templates/llmrank_rf.txt`
- `original_repo/EvaLLMasRecommender/eva/codes/llms.py`
- `original_repo/EvaLLMasRecommender/eva/codes/output_refine.py`
- `original_repo/EvaLLMasRecommender/eva/codes/zero_shot.py`
- `original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py`
- `original_repo/EvaLLMasRecommender/eva/codes/metrics.sh`
- `original_repo/EvaLLMasRecommender/llm_results/beauty.zip`
- `original_repo/EvaLLMasRecommender/llm_results/sports.zip`
- `results/tables/phase_f6e1_prompt_strategy_spec.csv`
- `results/tables/phase_f6e1_parser_contract.csv`
- `results/tables/phase_f6e3b_request_manifest.csv`
- `results/tables/phase_f6e4a_request_manifest.csv`
- `results/tables/phase_f6g1_future_request_manifest.csv`
- `results/tables/phase_f6g2_missing_or_blocked.csv`
- `results/tables/phase_f6g2r_missing_or_blocked.csv`
