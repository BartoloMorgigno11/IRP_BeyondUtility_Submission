# G4 F1BZQ Final Scientific Synthesis

- Phase: g4_f1bzq_scientific_synthesis
- Decision token: G4_SCIENTIFIC_SYNTHESIS_FROZEN
- Source: frozen G4 analytic, metric, and statistical artifacts only
- No Ollama/API calls, no new tests, no new bootstrap/statistical tests, no report rewrite
- Statistical wording: bootstrap CI only; no p-values; Holm not applied; NS means CI includes zero, not proof of no effect

## 1. Candidate-Stage Bottleneck

| Dataset | Positive Present | Users | Candidate Recall@20 | Mean Uncond HR@5 | Conditional n | Mean Cond HR@5 | Assessment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| beauty | 49 | 990 | 0.049494949494949494 | 0.022390572390572391 | 49 | 0.45238095238095238 | candidate_stage_primary_with_model_prompt_dependent_reranking |
| sports | 28 | 892 | 0.031390134529147982 | 0.011584454409566519 | 28 | 0.36904761904761907 | candidate_stage_primary_with_model_prompt_dependent_reranking |
| ml1m | 193 | 1000 | 0.193 | 0.067833333333333343 | 193 | 0.35146804835924006 | both_candidate_stage_ceiling_and_reranking_quality |
| lastfm | 469 | 1000 | 0.46899999999999997 | 0.28783333333333333 | 469 | 0.61371712864250172 | candidate_ceiling_present_but_reranking_quality_is_informative |
| foodcom | 4 | 1000 | 0.0040000000000000001 | 0.002 | 4 | 0.5 | candidate_stage_primary_conditional_n4_diagnostic_only |

Candidate recall is the upstream probability that the held-out positive appears in the frozen 20-item F1BZQ pool. Unconditional HR@5/NDCG@5/MRR measure the whole candidate-pool plus reranking pipeline over all planned users, with invalid terminal responses and positive-absent users contributing zero. Conditional HR@5/NDCG@5 measure reranker quality only among positive-present users, again with invalid terminal responses as zero inside that subset. Low unconditional utility must therefore not be attributed only to the LLM when the positive is absent from the pool.

Dataset reading: Beauty, Sports, and Food.com are dominated by candidate-stage failure; ML1M has both a material candidate ceiling and reranking limitations; LastFM has the least severe candidate bottleneck and the most informative reranking signal.

## 2. Internal G4 Statistics

- Frozen inferential rows: 225.
- Overall CI decisions: positive 29, negative 24, non-significant 172.
- Prompt strategy: positive 2, negative 13, NS 60.
- Model pair: positive 27, negative 11, NS 112.
- Dataset counts: Beauty 0/0/45, Sports 4/5/36, ML1M 10/5/30, LastFM 15/14/16, Food.com 0/0/45.

All inferential claims below are based on frozen percentile bootstrap CIs for paired user-level mean deltas. No p-values or corrected decisions are reported.

## 3. Prompt Effect

LLMRANK does not have a consistent cross-dataset advantage. The supported improvements are isolated to Qwen on LastFM HR@5 and conditional HR@5. Supported worsens occur for Qwen on Sports across all five utility/conditional metrics, for Mistral on LastFM across all five metrics, and for Gemma2 on LastFM NDCG@5, MRR, and conditional NDCG@5. Beauty, ML1M, and Food.com have no BASE-vs-LLMRANK CI-supported differences in the frozen prompt family.

| Dataset | Model | Metric | n | LLMRANK-BASE | CI Low | CI High | Effect |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sports | qwen2:7b-instruct | conditional_hr_at_5 | 28 | -0.14285714285714285 | -0.2857142857142857 | -0.035714285714285712 | LLMRANK_supported_worsening_vs_BASE |
| sports | qwen2:7b-instruct | conditional_ndcg_at_5 | 28 | -0.1388161716869479 | -0.25573422934655804 | -0.040390348341837766 | LLMRANK_supported_worsening_vs_BASE |
| sports | qwen2:7b-instruct | hr_at_5 | 892 | -0.0044843049327354259 | -0.0089686098654708519 | -0.0011210762331838565 | LLMRANK_supported_worsening_vs_BASE |
| sports | qwen2:7b-instruct | mrr | 892 | -0.0038103245782617977 | -0.0078694546764894301 | -0.00068862774930600118 | LLMRANK_supported_worsening_vs_BASE |
| sports | qwen2:7b-instruct | ndcg_at_5 | 892 | -0.0043574583040745987 | -0.0085481987669188651 | -0.0011210762331838565 | LLMRANK_supported_worsening_vs_BASE |
| lastfm | qwen2:7b-instruct | conditional_hr_at_5 | 469 | 0.061833688699360338 | 0.019189765458422176 | 0.1044776119402985 | LLMRANK_supported_improvement_over_BASE |
| lastfm | qwen2:7b-instruct | hr_at_5 | 1000 | 0.029000000000000001 | 0.0089999999999999993 | 0.049000000000000002 | LLMRANK_supported_improvement_over_BASE |
| lastfm | mistral:7b-instruct | conditional_hr_at_5 | 469 | -0.14925373134328357 | -0.20255863539445629 | -0.093816631130063971 | LLMRANK_supported_worsening_vs_BASE |
| lastfm | mistral:7b-instruct | conditional_ndcg_at_5 | 469 | -0.1836874999318992 | -0.22402989724457914 | -0.1440966025714123 | LLMRANK_supported_worsening_vs_BASE |
| lastfm | mistral:7b-instruct | hr_at_5 | 1000 | -0.070000000000000007 | -0.095000000000000001 | -0.043999999999999997 | LLMRANK_supported_worsening_vs_BASE |
| lastfm | mistral:7b-instruct | mrr | 1000 | -0.087206746031746038 | -0.10615982142857143 | -0.068782589285714305 | LLMRANK_supported_worsening_vs_BASE |
| lastfm | mistral:7b-instruct | ndcg_at_5 | 1000 | -0.086149437468060711 | -0.10591267307726503 | -0.066462440123147837 | LLMRANK_supported_worsening_vs_BASE |
| lastfm | gemma2:9b | conditional_ndcg_at_5 | 469 | -0.043177393879942294 | -0.068900683202370208 | -0.016557669764074256 | LLMRANK_supported_worsening_vs_BASE |
| lastfm | gemma2:9b | mrr | 1000 | -0.02411468253968254 | -0.035288412698412694 | -0.013076289682539682 | LLMRANK_supported_worsening_vs_BASE |
| lastfm | gemma2:9b | ndcg_at_5 | 1000 | -0.020250197729692936 | -0.032620592672849121 | -0.0077768839570644122 | LLMRANK_supported_worsening_vs_BASE |

The 60 NS prompt rows are not evidence of no prompt effect; they only mean the frozen CI includes zero.

## 4. Model Effect

Model differences are dataset-dependent. The largest supported differences are conditional utility differences in LastFM and ML1M. Gemma2 is often higher than Mistral in LastFM/ML1M, while Qwen is higher than Mistral in several LLMRANK rows. Qwen-vs-Gemma2 is mixed: Gemma2 is higher on several HR/NDCG rows, but one LastFM LLMRANK MRR row supports Qwen over Gemma2. No model is consistently superior across datasets, prompts, and metrics.

| Dataset | Prompt | Comparison | Metric | n | Second-First | CI Low | CI High | Effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lastfm | LLMRANK | mistral:7b-instruct_vs_gemma2:9b | conditional_hr_at_5 | 469 | 0.21961620469083157 | 0.16844349680170576 | 0.27078891257995735 | second_model_supported_higher_than_first |
| ml1m | LLMRANK | mistral:7b-instruct_vs_gemma2:9b | conditional_hr_at_5 | 193 | 0.20725388601036268 | 0.11917098445595854 | 0.30051813471502592 | second_model_supported_higher_than_first |
| lastfm | LLMRANK | qwen2:7b-instruct_vs_mistral:7b-instruct | conditional_ndcg_at_5 | 469 | -0.17944243134242058 | -0.2208723169790886 | -0.13875144525658839 | first_model_supported_higher_than_second |
| lastfm | LLMRANK | qwen2:7b-instruct_vs_mistral:7b-instruct | conditional_hr_at_5 | 469 | -0.17270788912579957 | -0.22601279317697229 | -0.11940298507462686 | first_model_supported_higher_than_second |
| lastfm | LLMRANK | mistral:7b-instruct_vs_gemma2:9b | conditional_ndcg_at_5 | 469 | 0.16430573557389283 | 0.127667988127242 | 0.20077213275013772 | second_model_supported_higher_than_first |
| ml1m | BASE | mistral:7b-instruct_vs_gemma2:9b | conditional_hr_at_5 | 193 | 0.16062176165803108 | 0.077720207253886009 | 0.24352331606217617 | second_model_supported_higher_than_first |
| ml1m | LLMRANK | qwen2:7b-instruct_vs_mistral:7b-instruct | conditional_hr_at_5 | 193 | -0.14507772020725387 | -0.23834196891191708 | -0.05181347150259067 | first_model_supported_higher_than_second |
| lastfm | BASE | qwen2:7b-instruct_vs_gemma2:9b | conditional_hr_at_5 | 469 | 0.11727078891257996 | 0.074626865671641784 | 0.15991471215351813 | second_model_supported_higher_than_first |
| ml1m | LLMRANK | mistral:7b-instruct_vs_gemma2:9b | conditional_ndcg_at_5 | 193 | 0.11287877705070559 | 0.05419563835398275 | 0.17189027221302824 | second_model_supported_higher_than_first |
| sports | LLMRANK | qwen2:7b-instruct_vs_gemma2:9b | conditional_ndcg_at_5 | 28 | 0.10961869435452168 | 0.024859965412241686 | 0.19890440864023598 | second_model_supported_higher_than_first |
| lastfm | LLMRANK | mistral:7b-instruct_vs_gemma2:9b | hr_at_5 | 1000 | 0.10299999999999999 | 0.078 | 0.128 | second_model_supported_higher_than_first |
| ml1m | LLMRANK | qwen2:7b-instruct_vs_mistral:7b-instruct | conditional_ndcg_at_5 | 193 | -0.089332846612125705 | -0.15232316417093791 | -0.025147716452375044 | first_model_supported_higher_than_second |

Do not overgeneralize from LastFM: it supplies the strongest conditional denominator and the densest set of supported CIs.

## 5. Conditional Reranking Quality

| Dataset | Positive-Present n | Guard |
| --- | --- | --- |
| beauty | 49 |  |
| sports | 28 |  |
| ml1m | 193 |  |
| lastfm | 469 |  |
| foodcom | 4 | diagnostic_n4 |

Conditional HR@5/NDCG@5 are only defined on these positive-present users. Food.com has n=4; its conditional rows are diagnostic only. Beauty and Sports have extremely low candidate recall, so their poor end-to-end utility is primarily candidate-stage limited even though conditional reranking sometimes recovers hits. ML1M reflects both candidate-stage loss and reranking quality differences. LastFM shows both a real candidate ceiling and meaningful reranker/model/prompt variation.

## 6. Beyond-Utility And Reliability

Only item_coverage_at_5 and oic_at_5 are informative frozen beyond-utility metrics for G4; they are strict-valid, candidate-union/pairwise summaries, not full-catalog claims. Other beyond-utility metrics remain undefined or excluded exactly as frozen.

| Dataset | Metric | Mean | Min | Max |
| --- | --- | --- | --- | --- |
| beauty | item_coverage_at_5 | 0.46502659960726578 | 0.41748942172073344 | 0.49572649572649574 |
| beauty | oic_at_5 | 0.16944266107997893 | 0.13450955917535209 | 0.24598969228313064 |
| foodcom | item_coverage_at_5 | 0.46609047349432969 | 0.40874895514070769 | 0.48844333054859368 |
| foodcom | oic_at_5 | 0.19688308144517344 | 0.13032632632632632 | 0.2313873873873874 |
| lastfm | item_coverage_at_5 | 0.61035269438448492 | 0.57647058823529407 | 0.67055393586005829 |
| lastfm | oic_at_5 | 0.10694869871466774 | 0.086232525110280622 | 0.14108997102958887 |
| ml1m | item_coverage_at_5 | 0.57346333608263622 | 0.50120772946859904 | 0.62318840579710144 |
| ml1m | oic_at_5 | 0.092772470460303194 | 0.07168048048048048 | 0.1196 |
| sports | item_coverage_at_5 | 0.47248963505059999 | 0.4296875 | 0.53634577603143418 |
| sports | oic_at_5 | 0.26705781548860635 | 0.22720730593607305 | 0.34482040132264347 |

Reliability accounting: planned 29292, strict-valid 28986, invalid 306, invalid_length 255, duplicate_indices 51, backend errors 0.

Descriptively, Mistral has more structural invalids than Qwen2 or Gemma2, and LLMRANK has more invalids than BASE. No reliability inferential test was frozen.

| Model | Strict-Valid Rate | Strict | Planned |
| --- | --- | --- | --- |
| gemma2:9b | 0.99631298648095046 | 9728 | 9764 |
| mistral:7b-instruct | 0.97839000409668164 | 9553 | 9764 |
| qwen2:7b-instruct | 0.99395739451044651 | 9705 | 9764 |

| Prompt | Strict-Valid Rate | Strict | Planned |
| --- | --- | --- | --- |
| BASE | 0.99685921070599481 | 14600 | 14646 |
| LLMRANK | 0.98224771268605759 | 14386 | 14646 |

## 7. G1 Vs G4

The 90 frozen G1-vs-G4 rows are descriptive only. G1 and G4 use different candidate sources, so these are pipeline-level differences, not a pure causal estimate of a reranking effect. All 90 descriptive deltas are negative for G4 relative to G1 in the frozen table.

| Metric | Mean G4-G1 | Min | Max |
| --- | --- | --- | --- |
| hr_at_5 | -0.27085150307257932 | -0.44955156950672648 | -0.096999999999999975 |
| mrr | -0.1745844024605559 | -0.27753265356964907 | -0.097964682539682546 |
| ndcg_at_5 | -0.17827692290128755 | -0.30412273078016233 | -0.080977464538889932 |

The useful pattern is that the F1BZQ 20-item reranking pipeline is much more ceiling-limited than the G1 ranking pipeline. This should be read as candidate-source and pipeline evidence, not as a claim that reranking itself causally reduces utility.

## 8. Scientific Takeaway

- The F1BZQ candidate generator is a major bottleneck, especially for Beauty, Sports, and Food.com; ML1M is also materially ceiling-limited; LastFM is the least bottlenecked.
- When the positive is present, the LLM reranker can recover useful utility, most clearly in LastFM and in selected Beauty/ML1M cells, but recovery is model- and prompt-dependent.
- LLMRANK does not consistently help; frozen CIs show isolated Qwen LastFM gains and more supported worsens than gains overall.
- Model effects are dataset-dependent; Gemma2 often improves over Mistral in LastFM/ML1M, but no model is uniformly superior.
- Strongest evidence: LastFM, then ML1M. Weakest evidence: Food.com due n=4 conditional denominator, and Beauty for internal CI support.
- G4 adds a frozen reranking-pipeline diagnosis that G1/G2/G3 did not provide: it separates F1BZQ candidate-pool ceiling from conditional LLM reranking quality and supplies paired internal prompt/model CIs.

## 9. Validation

- PASS: 29
- FAIL: 0
- Status: PASS

No final report rewrite, relation update, new experiments, new tests, or inference calls are included in this synthesis freeze.
