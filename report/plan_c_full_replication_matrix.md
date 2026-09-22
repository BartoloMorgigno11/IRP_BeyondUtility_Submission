# Plan C Full-Paper Replication Matrix

## 1. Purpose And Boundary

Plan C is a maximum-fidelity replication planning track for *Beyond Utility:
Evaluating LLM as Recommender*. It is separate from the completed Plan A and
Plan B tracks.

Plan A remains a closed readiness-aware reproducibility audit of the original
Beauty and Sports artifacts. It found that full numerical reproduction is
blocked because the original preprocessed data files are unavailable from the
official README-linked Dropbox artifacts. Plan B remains a completed
MovieLens-1M reconstruction/extension track with real metrics, but those
metrics are not exact reproduction of the original Beauty/Sports experiments.

Plan C does not claim exact reproduction. Its goal is to map every dataset,
task, model family, prompt strategy, metric dimension, and artifact needed for
a full-paper replication before implementing further code.

Machine-readable tables for this phase are:

- `results/tables/plan_c_replication_matrix.csv`
- `results/tables/plan_c_required_artifacts.csv`
- `results/tables/plan_c_models_metrics_tasks_inventory.csv`

No APIs were called, no datasets were downloaded, no models were trained, no
metrics were computed, and no files under `original_repo/` were modified.

## 2. Evidence Inspected

The C0 inventory is based on local evidence only:

- the local paper PDF: `paper/2411.00331v1.pdf`;
- the upstream checkout under `original_repo/EvaLLMasRecommender/`;
- upstream README, prompt templates, evaluation scripts, API wrappers, shell
  command lists, and ZIP manifests;
- the local Beauty and Sports LLM output ZIPs;
- the local Beauty SOTA ZIP;
- existing project documentation for Plan A and Plan B;
- existing project configs and generated diagnostic tables.

The local paper text confirms the full target scope: four datasets, seven LLMs,
three prompting strategies, six traditional baselines, ranking and re-ranking
tasks, utility, novelty, fairness/diversity-related metrics, and the LLM-specific
dimensions of history length sensitivity, candidate position bias,
generation-involved performance, and hallucination.

## 3. Status Taxonomy

Plan C uses the following status labels:

| Status | Meaning |
|---|---|
| `exact_ready` | All exact inputs, code, model version, outputs, and protocol details are available locally for that component. |
| `local_artifact_available` | A relevant local artifact exists, but exact full-paper reproduction may still need additional data, mappings, or protocol validation. |
| `reconstructible_from_public_raw` | A close reconstruction is feasible from public raw files, but equivalence to the paper is not guaranteed without original preprocessing details and seeds. |
| `requires_manual_download` | The component requires the user to place external raw or model files locally; Codex must not download them. |
| `requires_api` | The component requires external model API access and explicit approval in a future phase. |
| `requires_training` | The component requires training or rerunning traditional recommendation models. |
| `blocked_missing_artifact` | A required original artifact is missing or unavailable. |
| `blocked_unavailable_model` | The exact model or weights are not locally available. |
| `blocked_unclear_protocol` | Enough artifacts may exist to attempt something, but exact protocol details are insufficient. |
| `extension_only` | The component belongs to a reconstruction or extension track, not exact paper reproduction. |

No full-paper numerical result is currently `exact_ready`.

## 4. Dataset Matrix

| Dataset | Paper role | Local availability | Original repo availability | Expected raw files | Expected processed files | Candidate and catalog needs | Split protocol | Exact reproduction status |
|---|---|---|---|---|---|---|---|---|
| Amazon All Beauty | Original paper dataset; short-history Amazon domain. | Cached LLM outputs and Beauty SOTA output ZIP are local; canonical data files are absent. | README links a Beauty preprocessed ZIP, but the project records it as unavailable/deleted. | Amazon 2014 Beauty review data. | `sequential_data.txt`, `negative_samples.txt`, `datamaps.json`, `meta.json`. | Candidate set requires one positive plus negatives; catalog requires title/category mapping through datamaps and metadata. | Paper uses 5-core filtering and leave-one-out. | `blocked_missing_artifact`: core preprocessed files are missing. |
| Sports & Outdoors | Original paper dataset; short-history Amazon domain. | Cached LLM outputs are local; Sports SOTA archive and canonical data files are absent. | README links a Sports preprocessed ZIP, but the project records it as unavailable/deleted. | Amazon 2014 Sports review data. | `sequential_data.txt`, `negative_samples.txt`, `datamaps.json`, `meta.json`. | Same as Beauty; Sports SOTA recommendation files are also missing locally. | Paper uses 5-core filtering and leave-one-out. | `blocked_missing_artifact`: core preprocessed files and Sports SOTA outputs are missing. |
| MovieLens-1M | Original paper dataset; longer-history movie domain. | Raw and Plan B processed files exist locally, but they are Plan B reconstruction artifacts. | Original README references raw MovieLens-1M under `raw_data`; no upstream processed ML-1M artifacts are local. | `ratings.dat`, `movies.dat`, optionally `users.dat`. | Paper-equivalent 5-core sequences, negatives, item maps, sample users, and candidate sets are not local. | Titles are required as item text; candidate sets must match the paper's sampling and shuffle protocol. | Paper states 5-core filtering, leave-one-out, and small sampled evaluation. | `reconstructible_from_public_raw`: close reconstruction is feasible, but Plan B is not exact paper equivalence. |
| LastFM | Original paper dataset; longer-history music domain. | No local raw or processed LastFM data in this project. | Original README references LastFM raw data under `raw_data`; no processed artifacts are local. | HetRec 2011 LastFM user-artist interactions and artist metadata. | Paper-equivalent 5-core sequences, negatives, item maps, sample users, and candidate sets are not local. | Artist titles/names are required as item text; candidate sets and split must match paper protocol. | Paper states 5-core filtering, leave-one-out, and small sampled evaluation. | `requires_manual_download`: possible later reconstruction, exact equivalence currently unverified. |

## 5. Task Matrix

| Task | Candidate construction | Required inputs | Expected outputs | Local scripts/artifacts | Missing artifacts | Reproducibility status |
|---|---|---|---|---|---|---|
| Ranking | For each sampled test user, combine the held-out positive item with randomly sampled negative items not in the user's history. Paper implementation uses 20 candidates in the main setting. | User histories, held-out target, negative samples, title/catalog mappings, prompt templates, LLM outputs or model outputs. | Ranked top-k item lists, parsed item IDs, HR@5, NDCG@5, and beyond-utility metrics. | Prompt templates, cached Beauty/Sports LLM result JSONs, upstream `llm_metrics.py`, project parser and utility metric code. | Canonical `sequential_data.txt`, `negative_samples.txt`, `datamaps.json`, `meta.json`. | `blocked_missing_artifact` for exact Beauty/Sports; `extension_only` for Plan B ML-1M. |
| Re-ranking | Candidate pool is formed from top-k outputs of trained traditional recall models, specifically BPRMF, GRU4Rec, SASRec, and LightGCN in the paper. | Trained recall model outputs, user histories, held-out targets, item mappings, prompt templates, LLM outputs. | Re-ranked candidate lists and the same metric dimensions as ranking. | Cached Beauty/Sports LLM reranking JSONs exist for some GPT-3.5 variants; Beauty SOTA output ZIP exists. | Full paper candidate pools for all datasets/models, Sports SOTA outputs, ML-1M/LastFM SOTA outputs, canonical target/mapping files. | `blocked_missing_artifact`; reranking exactness also risks `requires_training` if recall outputs cannot be recovered. |

## 6. Prompting And LLM Strategy Matrix

| Strategy | Repo evidence | Inputs required | Output format | API/local model requirement | Status |
|---|---|---|---|---|---|
| Base/off-the-shelf prompt | `eva/templates/template.txt` and cached `results_*candidate20*_sample.json` files. | User history text, candidate title list, output formatting instruction. | Python-list-like item titles in JSON `result` fields. | API or local LLM needed to rerun; cached outputs exist for some legacy models. | `local_artifact_available` for saved outputs; exact metric computation remains blocked. |
| OpenP5-style template strategy | Paper states that OpenP5 sequential recommendation templates were adapted; repo templates include base, no-history, category, rating, and candidate-first variants. | Same as base plus variant-specific metadata such as category or rating text. | Ranked title lists. | API/local LLM needed to rerun. | `local_artifact_available` for templates and selected cached outputs. |
| LLMRank strategy | `eva/templates/llmrank.txt`, `llmrank_rf.txt`, and cached LLMRank result JSONs for selected Beauty/Sports settings. | Ordered history, candidate list, optionally most recent item. | Ranked title lists, often with chain-of-thought style prompting constraints. | API/local LLM needed to rerun. | `local_artifact_available` for selected artifacts; full-paper coverage is incomplete. |
| LLMSRec / in-context strategy | `template_in_context.txt`, `template_demonstration.txt`, and `template_fewshot.txt` indicate in-context/demonstration prompting. | Demonstration users, last-history item, current user history, candidates, title mappings. | Ranked title lists. | API/local LLM needed to rerun. | `blocked_unclear_protocol`: templates are local, but exact demonstration selection must be validated. |
| User profile generation | `template_user_profile_generate.txt`, `template_profile.txt`, `template_profile_history.txt`, profile result variants in LLM ZIPs. | User history, profile generation model outputs, candidate list, optional history plus generated profile. | Generated profile text and later ranked title lists. | API/local LLM required to regenerate profiles. | `local_artifact_available` for selected saved profile variants; exact rerun is `requires_api`. |

## 7. LLM Model Matrix

| Model | Open-source or closed/API | Local exact version availability | API access required | Cost/risk | Local substitute candidate | Status |
|---|---|---|---|---|---|---|
| `gpt-3.5-turbo-0125` | Closed/API | No exact local model; legacy GPT-3.5 cached outputs exist in Beauty/Sports ZIPs. | Yes for rerun. | API cost, version drift, deprecation risk. | Cached GPT-3.5 outputs for artifact audit only. | `local_artifact_available` for saved outputs; exact rerun `requires_api`. |
| `gpt-4o-mini-2024-07-18` | Closed/API | No local model or matching cached outputs found. | Yes. | API cost and version availability risk. | None exact; later could use saved outputs only if obtained. | `requires_api`. |
| `gpt-4o-2024-08-06` | Closed/API | No local model or matching cached outputs found; legacy GPT-4 outputs appear in ZIP names. | Yes. | Higher API cost and version drift risk. | Legacy cached GPT-4 artifacts are not exact GPT-4o. | `requires_api`. |
| `claude-3-haiku-20240307` | Closed/API | No local model or matching cached outputs found. | Yes. | API access, pricing, and version retention risk. | None exact. | `requires_api`. |
| `llama-3-70b-instruct` | Open weights, heavy local inference | No local weights; cached `llama3` outputs exist for selected Beauty/Sports settings. | Not necessarily, but local inference infrastructure or hosted endpoint would be needed. | High memory/compute if local; hosted inference cost if remote. | Smaller local Llama-family model would be a substitute, not exact. | `blocked_unavailable_model` for exact rerun; selected cached outputs are local. |
| `qwen-2-7b-instruct` | Open weights | No local weights or cached outputs found. | Not required if weights manually supplied. | Moderate local compute and weight-management risk. | None exact in checkout. | `blocked_unavailable_model`. |
| `mistral-7b-instruct-v0.2` | Open weights | No local weights or cached outputs found. | Not required if weights manually supplied. | Moderate local compute and tokenizer/version risk. | None exact in checkout. | `blocked_unavailable_model`. |
| `glm-4` | Closed/API, repo legacy artifact | No local model; cached `results_glm4_*` files exist in Beauty/Sports ZIPs. | Yes for rerun. | API/provider access and key-management risk. | Cached GLM4 outputs for artifact audit only. | `local_artifact_available` for saved outputs; exact rerun `requires_api`. |
| `gpt-4` legacy repo label | Closed/API, repo legacy artifact | No local model; cached `results_gpt4_*` files exist for selected settings. | Yes for rerun. | API version drift; not the same as paper `gpt-4o`. | Cached GPT-4 outputs for artifact audit only. | `local_artifact_available` for saved outputs; not paper-exact. |
| `llama-2` legacy repo label | Open weights, repo legacy artifact | No local weights; cached Sports `llama2` outputs exist. | Not required if weights manually supplied. | Weight and inference setup risk. | Cached Sports outputs only. | `local_artifact_available` for saved outputs; not paper-exact. |

## 8. Traditional Model Matrix

| Model | Local original implementation availability | Config availability | Dataset requirements | Training requirement | Cost estimate | Output format | Metrics supported | Status |
|---|---|---|---|---|---|---|---|---|
| BM25 | Upstream `ReChorus` implementation tree is not populated locally; Beauty BM25 cached outputs exist in SOTA ZIP. | No complete local configs found. | Item text, histories, candidates, targets. | Not required for Beauty cached output; required for missing datasets. | Low to moderate. | `rec-BM25-history10-shuffleFalse-seed*.csv` TSV with `user_id`, `rec_items`. | Original `llm_metrics.py` can compute full metric vector if data files exist. | `local_artifact_available` for Beauty output; full paper `requires_training` or output recovery. |
| MostPop | Implemented inside upstream `llm_metrics.py`; also implemented locally for Plan B. | Original uses script arguments, not full config. | Training interactions and candidate sets. | No heavy training. | Low. | Ranked item ID lists. | HR, NDCG, novelty/serendipity references. | `blocked_missing_artifact` for Beauty/Sports exact; `extension_only` for Plan B. |
| BPRMF | Beauty cached outputs exist; populated training implementation/configs not local. | No complete local configs found. | Train sequences, sampled test users, candidates, ground truth. | Required for missing datasets/seeds. | Moderate. | `rec-BPRMF-history10-shuffleFalse-seed*.csv` TSV. | Full original metric vector if canonical data exists. | `requires_training` for full paper; Beauty output is local but unevaluable without data. |
| GRU4Rec | Beauty cached outputs exist; implementation/configs not local. | No complete local configs found. | Sequential training data and candidate/target files. | Required for missing datasets/seeds. | Moderate to high. | `rec-GRU4Rec-history10-shuffleFalse-seed*.csv` TSV. | Full original metric vector if canonical data exists. | `requires_training` for full paper; Beauty output is local but unevaluable without data. |
| SASRec | Beauty cached outputs exist; implementation/configs not local. | No complete local configs found. | Sequential training data and candidate/target files. | Required for missing datasets/seeds. | Moderate to high. | `rec-SASRec-history10-shuffleFalse-seed*.csv` TSV. | Full original metric vector if canonical data exists. | `requires_training` for full paper; Beauty output is local but unevaluable without data. |
| LightGCN | Beauty cached outputs exist; implementation/configs not local. | No complete local configs found. | User-item graph from train interactions, candidates, targets. | Required for missing datasets/seeds. | Moderate to high. | `rec-LightGCN-history10-shuffleFalse-seed*.csv` TSV. | Full original metric vector if canonical data exists. | `requires_training` for full paper; Beauty output is local but unevaluable without data. |

## 9. Metrics And Evaluation Dimensions

| Metric or dimension | Definition or role | Original script/function | Required inputs | Task/model scope | Implemented locally | Exact paper computation status |
|---|---|---|---|---|---|---|
| HR@5 | Utility hit rate at top 5. | `calHR` in `llm_metrics.py`; local `utility_metrics.py`. | Ground truth and ranked recommendations. | Ranking and re-ranking, all models. | Yes. | `blocked_missing_artifact` for Beauty/Sports; `extension_only` for Plan B. |
| NDCG@5 | Rank-sensitive utility at top 5. | `calNDCG`; local `utility_metrics.py`. | Ground truth ranks and recommendations. | Ranking and re-ranking, all models. | Yes. | `blocked_missing_artifact` for Beauty/Sports; `extension_only` for Plan B. |
| MRR | Reciprocal rank utility. | `calMRR`. | Ground truth ranks. | Ranking and re-ranking. | Not currently a primary project metric. | `blocked_missing_artifact`. |
| APLT/ACLT | Long-tail exposure/popularity novelty measure. | `calACLT`; paper calls APLT. | Training item frequencies and recommendations. | All ranked outputs. | Not exact-paper implemented. | `blocked_missing_artifact`. |
| ARP | Average recommendation popularity. | `calARP`. | Training item frequencies and recommendations. | All ranked outputs. | Not exact-paper implemented. | `blocked_missing_artifact`. |
| PopREO | Popularity-related equal opportunity diagnostic. | `calPopREO`. | Hits, target popularity quantiles, training frequencies. | All ranked outputs. | Not exact-paper implemented. | `blocked_missing_artifact`. |
| Serendipity | Correct recommendations not made by MostPop. | `calSerendipity`. | Model hits and MostPop hits. | All ranked outputs. | Not exact-paper implemented. | `blocked_missing_artifact`. |
| SelfInformation | Novelty/information score from item frequency. | `calSelfInfo`. | Training frequencies and recommendations. | All ranked outputs. | Not exact-paper implemented. | `blocked_missing_artifact`. |
| Gini | Item exposure fairness/inequality over recommendations. | `calcFairMetric`. | Top-k recommended item distribution. | All ranked outputs. | Not exact-paper implemented. | `blocked_missing_artifact`. |
| Jain | User-side fairness over NDCG distribution. | `calJain`. | Per-user hit/rank values. | All ranked outputs. | Not exact-paper implemented. | `blocked_missing_artifact`. |
| DPD | Difference between cold and hot user group performance. | `calDPDFreq`. | User history lengths and per-user NDCG. | All ranked outputs. | Partially related local position/history utilities only. | `blocked_missing_artifact`. |
| ItemCoverage | Candidate-pool coverage by top-k recommendations. | `calItemCoverage`; local catalog coverage exists but differs. | Candidate pools and recommendations. | All ranked outputs. | Local coverage extension implemented; exact paper metric not identical. | `blocked_missing_artifact` for paper; `extension_only` locally. |
| Shannon entropy | Category diversity of recommendations. | `calMeanShannon`. | Item categories and recommendations. | All ranked outputs. | Not exact-paper implemented. | `blocked_missing_artifact`. |
| Jaccard | Pairwise overlap diversity over recommendation lists. | `calMeanJaccard`. | Recommendations and candidate pools. | All ranked outputs. | Not exact-paper implemented. | `blocked_missing_artifact`. |
| Hallucination rate | Fraction of generated recommendations that do not match valid items. | `calHallucination`; local hallucination utilities. | Catalog/item title mappings and parsed LLM outputs. | LLM outputs. | Yes, with explicit match types. | `blocked_missing_artifact` for Beauty/Sports. |
| History length sensitivity | Compare accuracy under truncated histories. | Prompt variants and `metrics.sh` history commands. | Histories, candidate sets, model outputs for each history length. | LLM and traditional comparison. | Readiness utilities only. | `blocked_missing_artifact` plus `requires_api` for reruns. |
| Candidate position bias | Compare positive-first versus random candidate ordering after inverse-log transform. | Prompt variants and paper formula; local position-bias readiness utilities. | Candidate positions, ground truth, repeated outputs. | Primarily LLM outputs. | Readiness utilities only. | `blocked_missing_artifact` plus `requires_api` for reruns. |
| Generation-involved performance | Compare no-profile, profile-only, and profile-plus-history settings. | Profile templates and result variants. | Generated profiles, recommendation outputs, ground truth. | LLM outputs. | Not exact-paper implemented. | `local_artifact_available` for selected saved outputs; exact evaluation blocked. |

## 10. Artifact Matrix

The required artifact inventory is saved in
`results/tables/plan_c_required_artifacts.csv`. The highest-impact blockers are:

- preprocessed Beauty and Sports archives from the README Dropbox links;
- `sequential_data.txt`;
- `negative_samples.txt`;
- `datamaps.json`;
- `meta.json`;
- Sports, ML-1M, and LastFM full SOTA output sets;
- exact paper candidate pools and sampled user sets for all four datasets;
- exact model access for closed LLMs and local weights for open models.

The local artifacts that are useful but insufficient for full metrics are:

- Beauty/Sports LLM ZIPs with prompt/result JSONs and upstream metric vectors;
- Beauty SOTA ZIP with BM25, BPRMF, GRU4Rec, LightGCN, SASRec, and NRMS output
  files for selected seeds;
- upstream prompt templates;
- upstream evaluation scripts;
- Plan B ML-1M reconstruction files, which are extension artifacts.

## 11. Main Findings

1. The full paper cannot be reproduced exactly from the current checkout.
   The missing preprocessed data files block the target labels, candidate sets,
   catalog/title mappings, item categories, popularity statistics, and many
   beyond-utility metrics.
2. Saved Beauty/Sports LLM outputs are locally available, but they are not
   enough to compute exact paper metrics without the missing data and mapping
   files.
3. The local Beauty SOTA ZIP provides useful traditional baseline outputs, but
   those outputs also require `sequential_data.txt` and related mappings for
   exact evaluation.
4. The paper's full model scope is broader than the saved local artifacts. It
   includes closed models that require API access and open models whose weights
   are not local.
5. MovieLens-1M is locally reconstructible through Plan B, but Plan B uses a
   documented reconstruction protocol and must remain `extension_only` unless
   the exact paper preprocessing, sample users, candidate sets, and model
   outputs are recovered.

## 12. Exact-Ready, Blocked, And Reconstructible Components

Exact-ready full-paper numerical components: none.

Local-artifact-available components:

- upstream README, scripts, and prompt templates;
- Beauty and Sports cached LLM result archives;
- Beauty SOTA output archive;
- selected upstream metric CSV vectors inside LLM ZIPs, treated only as
  upstream reference artifacts;
- Plan B ML-1M reconstruction artifacts, marked as extension-only.

Blocked components:

- exact Beauty/Sports metrics due to missing preprocessed data files;
- exact Sports traditional baseline evaluation due to missing Sports SOTA
  outputs and missing canonical data;
- exact ML-1M and LastFM paper replication due to missing paper-equivalent
  processed artifacts and candidate/sample files;
- exact closed-LLM reruns due to API requirements;
- exact open-LLM reruns due to unavailable local weights and inference setup;
- exact traditional model reruns due to missing populated ReChorus
  implementation/configs and training requirements.

Reconstructible components:

- Amazon Beauty and Sports may be reconstructed from public raw Amazon data,
  but exact equivalence is uncertain without author preprocessing details and
  seeds.
- MovieLens-1M is already reconstructed for Plan B, but not paper-exact.
- LastFM is likely reconstructible after manual raw data placement, but exact
  paper preprocessing must be specified first.

## 13. Recommended Next Step

The next Plan C phase should be C1: artifact recovery and validation protocol.
Before adding more code, the project should define a checklist for recovering
or manually supplying paper-equivalent processed artifacts, validating their
schemas and checksums, and mapping them into canonical project files without
modifying `original_repo/`.

The minimum artifact set for C1 is:

- `sequential_data.txt`;
- `negative_samples.txt`;
- `datamaps.json`;
- `meta.json`;
- sampled user lists;
- full SOTA recommendation outputs for each dataset needed by ranking and
  re-ranking;
- exact prompt/output JSONs for each LLM/model/strategy cell to be evaluated.

Only after that artifact validation should Plan C implement converters or
compute metrics.
