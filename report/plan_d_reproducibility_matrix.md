# Plan D Reproducibility Matrix and Feasibility Ledger

## Executive summary

This matrix separates exact paper reproduction claims from reconstructed evaluation, diagnostic subsets, conditional LLM work, and Plan D extensions. The project now has real Plan D diagnostic_subset and extended_subset utility results for BPRMF, LightGCN, SASRec, and GRU4Rec on the utility-evaluable reconstructed datasets: ML-1M, Amazon Beauty, and Amazon Sports.

This is not an exact numerical reproduction of the original paper. It is a reconstructed reproduction and extension because official preprocessed artifacts and exact paper candidate pools are unavailable. Extended_subset results are real and reproducible within this repository, but they must not be described as full-scope or exact paper results.

LastFM is explicitly documented and not forgotten. It remains `extension_only_no_valid_utility_evaluation` because this reconstructed project does not have a valid comparable chronological split, candidate set, and ground-truth setup for HR@5/NDCG@5 evaluation. No fake LastFM utility metrics were generated.

D5B added an LLM model registry and backend feasibility audit before any LLM experiment. D5C0 verified Ollama and `qwen2:7b-instruct`; D5C1 ran exactly one successful toy local smoke inference. D5D completed reusable prompt templates, strict parser utilities, and a no-inference ML-1M manifest with 20 ready rows. D5E ran the bounded 20-user local Qwen2/Ollama ML-1M ranking diagnostic. D5E.1 analyzed the outputs without new inference and found perfect response/parse/schema rates but hallucination_rate `0.370000`; candidate adherence is the main Qwen2 failure mode. This remains diagnostic_subset work, not exact paper reproduction, and closed-source API reproduction remains conditional on explicit approval, keys, availability, and budget. D5F completed candidate position-bias diagnostics. D5G completed a strict candidate-ID-only prompt and deterministic post-filter diagnostic as diagnostic_subset work; it is not exact reproduction. D5H was initially conditional because Mistral was not locally available, then passed a local `mistral:7b-instruct` smoke test after the user manually installed the model; D5H is a diagnostic_subset local model availability/smoke-test phase, not exact reproduction and not full LLM reproduction. D5I compared Qwen2 and Mistral on the same strict-prompt ML-1M diagnostic subset; Qwen2 was not rerun. D5J completed local LLM reranking as diagnostic_subset work. D5K completed local LLM history-length sensitivity as diagnostic_subset work. D5L built the final Plan D reproducibility package and hygiene audit without new inference.

## Status labels

| Label | Meaning |
|---|---|
| `DONE` | Implemented in this project with real local artifacts, but not necessarily an exact paper reproduction. |
| `PARTIAL` | Some required inputs or behaviors are available, but the paper component is incomplete. |
| `DIAGNOSTIC_ONLY` | Implemented only for bounded diagnostic subsets, not full reproduction. |
| `FEASIBLE` | Can be implemented next with current local resources and careful scoping. |
| `CONDITIONAL` | Depends on a user decision, backend, API budget, model, or recovered artifact. |
| `BLOCKED` | Cannot be completed honestly with current local artifacts. |
| `EXTENSION` | Added by this project beyond the original paper scope or as engineering support. |
| `NOT_ATTEMPTED` | Not yet implemented and not currently scheduled. |

Implementation levels use `exact`, `reconstructed`, `diagnostic`, `planned`, `blocked`, `extension`, and `not_attempted`.

## Table 1: Paper components vs current project status

| Component | Paper intent | Current status | Level | Main gap |
|---|---|---|---|---|
| Official code repository | Reference implementation | `DONE` | `exact` | Code availability does not imply exact metric reproduction. |
| Official preprocessed artifacts | Exact splits, candidates, targets, mappings | `BLOCKED` | `blocked` | Official paper-ready artifacts are not available locally. |
| Ranking task | Rank held-out targets in candidate sets | `DONE` | `reconstructed` | Uses reconstructed candidate sets, not exact paper pools. |
| Re-ranking task | LLM reranks recommender-generated pools | `DIAGNOSTIC_ONLY` | `diagnostic` | D5J completed local open-model reranking over reconstructed 20-user ML-1M candidate-order reference; not exact paper pools. |
| Candidate-set evaluation | Evaluate only supplied candidates | `DONE` | `reconstructed` | Candidate provenance differs from official artifacts. |
| Candidate pool construction | Match paper candidate pools | `PARTIAL` | `reconstructed` | Official pool construction is incomplete. |
| Reproducibility blockers | Identify missing artifacts and dependencies | `DONE` | `reconstructed` | Blockers remain for exact preprocessing and LLM backend availability. |
| Final report | Synthesize claims and limitations | `FEASIBLE` | `planned` | D5L final package is generated; final narrative can now be assembled from CSV evidence. |

The full component matrix is stored in `results/tables/plan_d_paper_component_matrix.csv`.

## Table 2: Model-by-model reproduction status

| Model or backend | Status | Level | Notes |
|---|---|---|---|
| MostPop | `DONE` | `reconstructed` | Implemented and tested as a stable reconstructed baseline. |
| Random | `EXTENSION` | `extension` | Deterministic sanity baseline beyond the paper target set. |
| BM25 / ContentBM25 | `PARTIAL` | `reconstructed` | Available where metadata and candidate inputs support it. |
| BPRMF | `DONE` | `reconstructed` | Implemented with diagnostic_subset and extended_subset results for ML-1M, Amazon Beauty, and Amazon Sports. |
| LightGCN | `DONE` | `reconstructed` | Implemented with diagnostic_subset and extended_subset results for ML-1M, Amazon Beauty, and Amazon Sports. |
| SASRec | `DONE` | `reconstructed` | Implemented with diagnostic_subset and extended_subset results for ML-1M, Amazon Beauty, and Amazon Sports. |
| GRU4Rec | `DONE` | `reconstructed` | Implemented with diagnostic_subset and extended_subset results for ML-1M, Amazon Beauty, and Amazon Sports. |
| gpt-3.5-turbo-0125 | `conditional_legacy_api` | `exact_only_if_api_model_available` | Legacy/conditional OpenAI API model; do not use without explicit approval and API availability verification. |
| gpt-4o-mini-2024-07-18 | `feasible_with_api_budget` | `exact_api_conditional` | Best first optional OpenAI API diagnostic if the user approves API budget. |
| gpt-4o-2024-08-06 | `feasible_with_api_budget` | `exact_api_conditional` | Stronger but more expensive optional OpenAI API path; only for small diagnostics if approved. |
| claude-3-haiku-20240307 | `blocked_exact_model_retired` | `blocked_exact_replacement_only` | Exact Claude Haiku model is retired; any current Claude replacement would be non-exact. |
| llama-3-70b-instruct | `blocked_local_hardware` | `blocked_local_conditional_cloud` | Not realistic locally on RTX 4060 8GB; cloud or provider API required for practical evaluation. |
| qwen-2-7b-instruct | `diagnostic_completed_mitigation_analyzed` | `local_open_model_near_exact` | D5E-D5I include Qwen2 ranking, validity, position-bias, strict-prompt, and Qwen-vs-Mistral comparison diagnostics; D5I reused Qwen HR@5 `0.550000` without rerunning Qwen2. |
| mistral-7b-instruct-v0.2 | `diagnostic_completed_local_comparison` | `local_open_model_near_exact` | D5H smoke test passed after manual install; D5I ran local Mistral strict-prompt diagnostic with HR@5 `0.100000` and hallucination_rate `0.000000`. |
| Ollama | `diagnostic_completed` | `planned` | D5E used local Ollama for 20 Qwen2 ML-1M ranking requests; no remote API was called. |
| transformers | `BLOCKED` | `blocked` | D5.0 found `transformers` unavailable. |
| llama_cpp | `BLOCKED` | `blocked` | D5.0 found `llama_cpp` unavailable. |

The detailed D5B LLM registry is stored in `results/tables/plan_d_llm_model_registry.csv`, the local backend audit in `results/tables/plan_d_llm_backend_audit.csv`, and the D5C-D5K roadmap in `results/tables/plan_d_llm_task_roadmap.csv`.

## Table 3: Dataset-by-dataset status

| Dataset | Status | Level | Available local evidence | Limitation |
|---|---|---|---|---|
| Amazon Beauty | `PARTIAL` | `reconstructed` | Reconstructed candidates, ground truth, interactions, catalog, and 4-method extended_subset utility metrics. | Official preprocessed paper artifacts are unavailable. |
| Amazon Sports and Outdoors | `PARTIAL` | `reconstructed` | Reconstructed candidates, ground truth, interactions, catalog, and 4-method extended_subset utility metrics. | Official preprocessed paper artifacts are unavailable. |
| MovieLens-1M | `PARTIAL` | `reconstructed` | Reconstructed candidates, ground truth, interactions, catalog, and 4-method extended_subset utility metrics. | Exact paper preprocessing is not confirmed. |
| LastFM | `BLOCKED` | `blocked` | Reconstructed interactions/catalog and explicit coverage rows. | Excluded from HR@5/NDCG@5 ranking because no valid comparable chronological split/candidate/ground-truth setup exists. |

## Table 4: Metrics and evaluation status

| Metric or dimension | Status | Level | Current support |
|---|---|---|---|
| HR@5 | `DONE` | `reconstructed` | Tested and used by Plan D neural baselines and global extended summaries. |
| NDCG@5 | `DONE` | `reconstructed` | Tested and used by Plan D neural baselines and global extended summaries. |
| Novelty | `EXTENSION` | `extension` | Implemented in D1 beyond-utility metrics. |
| Popularity / long-tail behavior | `EXTENSION` | `extension` | Implemented in D1 beyond-utility metrics. |
| Candidate validity | `EXTENSION` | `extension` | D5E.1 measured mean candidate validity `0.630000` and micro candidate validity `0.630000` for Qwen2. |
| Duplicate rate | `EXTENSION` | `extension` | Implemented for Plan D beyond-utility and available for future LLM outputs. |
| Candidate position bias | `DIAGNOSTIC_ONLY` | `diagnostic` | D5F completed Qwen2 candidate-order robustness diagnostics over 20 ML-1M users and 3 variants. |
| Strict candidate-ID prompt and post-filter mitigation | `DIAGNOSTIC_ONLY` | `diagnostic` | D5G completed a diagnostic_subset prompt mitigation analysis; not exact reproduction and not full LLM reproduction. |
| Qwen2 vs Mistral local strict-prompt comparison | `DIAGNOSTIC_ONLY` | `diagnostic` | D5I compared local open models on 20 ML-1M strict-prompt instances; not exact reproduction. |
| Local LLM re-ranking diagnostic | `DIAGNOSTIC_ONLY` | `diagnostic` | D5J completed Qwen2 and Mistral reranking diagnostics; Qwen HR@5 `0.050000`, Mistral HR@5 `0.050000`. |
| History length sensitivity | `DIAGNOSTIC_ONLY` | `diagnostic` | D5K completed Qwen2 and Mistral local history-length sensitivity diagnostics; Qwen2 medium HR@5 `0.450000`, Mistral medium HR@5 `0.100000`. |
| Hallucination | `PARTIAL` | `extension` | D5E.1 confirmed hallucination_rate `0.370000`; D5G tested strict prompting and post-filtering, with strict hallucination_rate `0.000000`. |
| Output validity and parsing robustness | `PARTIAL` | `extension` | D5E.1 found response/parse/schema/exact-top-k rates all `1.000000` but mean candidate validity `0.630000`. |
| Runtime and cost | `FEASIBLE` | `planned` | Should be recorded for any future LLM diagnostic run. |

LLM ranking now has a bounded D5E local Qwen2 diagnostic result. Reranking and history-length sensitivity remain pending and must use real D5E or future LLM outputs, never fabricated outputs.

## Table 5: Extended neural baseline findings

Generated artifacts:

- `results/tables/plan_d_global_extended_neural_baselines_summary.csv`
- `results/tables/plan_d_global_extended_dataset_model_coverage.csv`
- `results/tables/plan_d_global_neural_diagnostic_vs_extended.csv`
- `report/plan_d_global_extended_neural_baselines.md`

The global utility summary has 12 rows, the dataset/model coverage table has 16 rows, LastFM has 4 explicit coverage rows, and the diagnostic-vs-extended comparison has 48 rows.

Dataset rankings by HR@5 and NDCG@5:

- ML-1M: SASRec > GRU4Rec > LightGCN > BPRMF.
- Amazon Beauty: GRU4Rec > LightGCN > BPRMF > SASRec.
- Amazon Sports: GRU4Rec > LightGCN > BPRMF > SASRec.

Global HR@5 top 5:

1. `ml1m / sasrec` — HR@5 = `0.3333333333333333`
2. `ml1m / gru4rec` — HR@5 = `0.2931547619047619`
3. `ml1m / lightgcn` — HR@5 = `0.259`
4. `ml1m / bprmf` — HR@5 = `0.243`
5. `amazon_beauty / gru4rec` — HR@5 = `0.159`

Interpretation:

- SASRec is strongest on ML-1M extended_subset.
- GRU4Rec is strongest on the reconstructed Amazon datasets.
- LightGCN consistently beats BPRMF on Amazon Beauty and Amazon Sports.
- SASRec performs poorly on the Amazon datasets despite strong ML-1M performance.
- The reconstructed Amazon subsets are sparse, with about 7k train interactions for 1000 users, which likely affects sequence-heavy transformer performance.

## Table 6: Remaining feasible work

| Work item | Feasibility | Decision |
|---|---|---|
| Install or confirm Ollama backend | `available` | D5C0 found `ollama` on PATH. |
| Pull `qwen2:7b-instruct` manually | `available_locally` | D5C0 found `qwen2:7b-instruct` in `ollama list`; no pull was executed by Codex. |
| D5C Qwen smoke test | `smoke_test_passed` | Exactly one toy local smoke inference was run; no dataset ranking experiment was run. |
| LLM prompt templates and parser | `completed_no_inference` | D5D prepared templates, parser utilities, and 20 ML-1M manifest rows without LLM inference. |
| Qwen ML-1M ranking diagnostic | `completed_diagnostic_subset_validity_analyzed` | D5E HR@5 `0.150000`, NDCG@5 `0.100889`; D5E.1 valid-only HR@5 `0.150000`, NDCG@5 `0.100889`. |
| Mistral local comparison diagnostic | `completed_diagnostic_subset` | D5I compared local Qwen2 and Mistral on the same 20 ML-1M strict-prompt instances. |
| Candidate position bias over real LLM outputs | `completed_diagnostic_subset` | D5F ran reverse_order, positive_first, and positive_last candidate-order diagnostics with local Qwen2. |
| Strict candidate-ID prompt and post-filter mitigation | `completed_diagnostic_subset` | D5G ran 20 local Qwen2 strict-prompt requests and deterministic original/strict post-filter analyses. |
| LLM reranking diagnostic | `completed_diagnostic_subset` | D5J completed Qwen2 and Mistral local reranking diagnostics against candidate_order_reference. |
| History length sensitivity | `completed_diagnostic_subset` | D5K varied short/medium/long history windows while holding candidate sets fixed. |
| Final report integration | `feasible_now` | Use generated CSVs and avoid invented metric values. |
| Optional full-scope feasibility audit | `feasible_with_compute` | Planning only unless explicitly approved. |

## Table 7: Blocked or conditional work

| Work item | Status | Reason |
|---|---|---|
| Exact official Beauty/Sports reproduction | `blocked_missing_artifacts` | Official paper-ready preprocessed artifacts are missing. |
| LastFM utility evaluation | `extension_only_no_valid_utility_evaluation` | Valid comparable candidates, chronological split, and ground truth are not available locally. |
| Local LLM diagnostic ranking | `completed_diagnostic_subset` | D5E-D5K produced bounded local LLM diagnostic outputs; no exact or full-scope LLM reproduction is claimed. |
| Local LLM diagnostic reranking | `completed_diagnostic_subset` | D5J completed Qwen2 and Mistral local reranking diagnostics against candidate_order_reference. |
| Closed API LLM reproduction | `feasible_with_api_budget` | Requires explicit API access and budget approval. |
| Llama-3-70B-style local evaluation | `blocked_local_hardware` | Not feasible locally on RTX 4060 8GB; cloud or provider API required. |
| Claude-3-Haiku exact reproduction | `blocked_exact_model_retired` | Exact `claude-3-haiku-20240307` is retired; replacement would be non-exact. |

## Recommended next steps

1. Review the D5L final reproducibility package.
2. Prepare exam presentation tables from generated CSV files.
3. Optionally run an OpenAI API diagnostic only if budget is explicitly approved.
4. Finalize the report.
