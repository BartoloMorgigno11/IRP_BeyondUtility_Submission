# G2 Nested History-Length Sensitivity Protocol

## PRECONDITION
- Required checkpoint: `7d32400` / `Complete and retain G1 primary ranking evidence`.
- Observed HEAD: `7d32400 Complete and retain G1 primary ranking evidence`.
- G1 primary ranking and G1 retention manifests are validated. No G0/G1 artifact is modified by this refreeze.

## DESIGN DECISION
- `ADOPT_DESIGN_B` from `G2_COHORT_DESIGN_REVIEW_COMPLETE` replaces the prior single GE20 primary design before any G2 inference.
- B1 is primary short/medium history sensitivity; B2 is a separate long-history extension; the GE20 full curve is secondary.

## B1 PRIMARY COHORT
- Reporting view `PRIMARY_B1_GE10`; eligibility train history >= `10`; H=0,1,2,5,10.
- Paired users by dataset: `{"beauty":181,"foodcom":61,"lastfm":981,"ml1m":1000,"sports":160}`.

## B2 LONG-HISTORY COHORT
- Reporting view `LONG_HISTORY_B2_GE20`; eligibility train history >= `20`; H=10,20.
- Paired users by dataset: `{"beauty":38,"foodcom":35,"lastfm":932,"ml1m":993,"sports":35}`.

## SECONDARY FULL CURVE
- Reporting view `SECONDARY_FULL_CURVE_GE20`; GE20 users; H=0,1,2,5,10,20.
- It reuses B1/B2 canonical request identities and creates zero duplicate inference calls.

## FOOD.COM
- Food.com true-zero users: `727`.
- Warm GE10 users may enter B1; warm GE20 users may enter B2/secondary. True-zero users remain outside B1/B2 paired warm-user estimands.
- Controlled warm H0 rows use `CONTROLLED_H0`; true-zero diagnostic rows use `TRUE_ZERO_HISTORY`; true-zero H>0 is `NOT_APPLICABLE`.

## LLMRANK H0
- LLMRANK H=0 status: `FROZEN_LOCAL_EQUIVALENT`.
- The frozen prompt renderer emits the no-recency sentence and does not fabricate a last item.

## REQUEST UNIVERSE
- Deduplicated applicable request rows: `88050`.
- A GE20/H10 request can belong to B1, B2, and secondary views, but keeps one canonical identity.
Dataset totals:
dataset,planned_applicable_requests,g1_exact_reuse_requests,new_future_ollama_calls_required
beauty,5658,1086,4572
sports,5010,960,4050
ml1m,35958,6000,29958
lastfm,35022,5886,29136
foodcom,6402,4728,1674

## G1 REUSE
- Exact G1 terminal evidence reuse: `18660` requests.
- Reuse requires exact dataset, user, model, prompt, history content/order/hash, candidate content/order/hash, target, generation contract, parser contract, prompt hash, and model digest.
- B1 H10 rows reuse G1 for GE10 users; Food.com true-zero H0 rows reuse G1.

## NEW CALL ACCOUNTING
- Future new Ollama calls required: `69390`.
- Not-applicable true-zero H>0 user conditions: `21810`.
Model totals:
model,planned_applicable_requests,g1_exact_reuse_requests,new_future_ollama_calls_required
qwen2:7b-instruct,29350,6220,23130
mistral:7b-instruct,29350,6220,23130
gemma2:9b,29350,6220,23130
Prompt totals:
prompt_strategy,planned_applicable_requests,g1_exact_reuse_requests,new_future_ollama_calls_required
BASE,44025,9330,34695
LLMRANK,44025,9330,34695

## STATISTICAL PLAN
- B1 primary paired comparisons: H0-H1, H0-H5, H0-H10, H1-H2, H2-H5, H5-H10.
- B2 long-history comparison: H10-H20.
- Secondary GE20 full-curve comparisons: H0-H20 and H5-H20.
- Multiplicity, CI, effect-size, alpha, and fail-strict denominator policies remain pre-outcome frozen.

## REPRESENTATIVENESS
Eligibility counts:
dataset,users_ge_1,users_ge_2,users_ge_5,users_ge_10,users_ge_20,primary_cohort_status,long_history_cohort_status
beauty,990,990,668,181,38,PRIMARY_B1_FROZEN,LONG_HISTORY_B2_FROZEN_WITH_LOW_SHARE_REPRESENTATIVENESS_WARNING
sports,892,892,626,160,35,PRIMARY_B1_FROZEN,LONG_HISTORY_B2_FROZEN_WITH_LOW_SHARE_REPRESENTATIVENESS_WARNING
ml1m,1000,1000,1000,1000,993,PRIMARY_B1_FROZEN,LONG_HISTORY_B2_FROZEN
lastfm,1000,1000,998,981,932,PRIMARY_B1_FROZEN,LONG_HISTORY_B2_FROZEN
foodcom,273,184,107,61,35,PRIMARY_B1_FROZEN_WITH_LOW_SHARE_REPRESENTATIVENESS_WARNING,LONG_HISTORY_B2_FROZEN_WITH_LOW_SHARE_REPRESENTATIVENESS_WARNING
B1 selected view:
dataset,users,share_of_g1_users,status
beauty,181,0.182828,PRIMARY_B1_FROZEN
sports,160,0.179372,PRIMARY_B1_FROZEN
ml1m,1000,1.000000,PRIMARY_B1_FROZEN
lastfm,981,0.981000,PRIMARY_B1_FROZEN
foodcom,61,0.061000,PRIMARY_B1_FROZEN_WITH_LOW_SHARE_REPRESENTATIVENESS_WARNING
B2 long-history view:
dataset,users,share_of_g1_users,status
beauty,38,0.038384,LONG_HISTORY_B2_FROZEN_WITH_LOW_SHARE_REPRESENTATIVENESS_WARNING
sports,35,0.039238,LONG_HISTORY_B2_FROZEN_WITH_LOW_SHARE_REPRESENTATIVENESS_WARNING
ml1m,993,0.993000,LONG_HISTORY_B2_FROZEN
lastfm,932,0.932000,LONG_HISTORY_B2_FROZEN
foodcom,35,0.035000,LONG_HISTORY_B2_FROZEN_WITH_LOW_SHARE_REPRESENTATIVENESS_WARNING

## RESEARCH QUESTIONS
- RQ-G2.1 through RQ-G2.6 now refer primarily to B1 short/medium sensitivity.
- `RQ-G2.LONG` covers H10 versus H20 among GE20 users.
- The Food.com controlled-zero versus true-zero diagnostic RQ is preserved.

## RETENTION
- Retention classes are preserved: TRACKED_SCIENTIFIC_FOREGROUND, TRACKED_PROVENANCE_MANIFEST, EXTERNALIZED_RAW_EVIDENCE, EXTERNALIZED_PER_REQUEST_AUDIT.
- Future G2 raw responses and per-request execution audit remain externalized and must be referenced by tracked manifests with stable SHA256 hashes before any commit.

## TESTS
- Focused tests are in `tests/test_extension_g2_history_length_sensitivity_protocol.py`.

## FILES
- `configs/g2_history_length_sensitivity.yaml`
- `src/extension/g2_history_length_sensitivity_protocol.py`
- `results/tables/g2_history_length_eligibility.csv`
- `results/tables/g2_history_length_cohort.csv`
- `results/tables/g2_history_length_applicability.csv`
- `results/tables/g2_history_length_g1_reuse_audit.csv`
- `results/tables/g2_history_length_prior_evidence_audit.csv`
- `results/tables/g2_history_length_request_universe.csv`
- `results/tables/g2_history_length_request_accounting.csv`
- `results/tables/g2_history_length_research_questions.csv`
- `results/tables/g2_history_length_statistical_plan.csv`
- `results/tables/g2_history_length_representativeness.csv`
- `results/manifests/g2_history_length_protocol_freeze.json`
- `report/g2_history_length_protocol.md`

## PRIOR HISTORY EVIDENCE
source_id,exists,source_rows,reuse_status,reusable_g2_terminal_evidence
R5C_HISTORY_LENGTH_ORIGINAL,true,2400,COMPARATIVE_PRIOR_EVIDENCE,false
R5C_HISTORY_LENGTH_CORRECTED,true,2400,COMPARATIVE_PRIOR_EVIDENCE,false
PLAN_D_ML1M_HISTORY_LENGTH,true,120,COMPARATIVE_PRIOR_EVIDENCE,false
R1_OFFICIAL_HISTORY_LENGTH,true,462,COMPARATIVE_PRIOR_EVIDENCE,false
R4_HISTORY_LENGTH_RECONCILIATION,true,4,COMPARATIVE_PRIOR_EVIDENCE,false
- Prior history-length evidence remains comparative only, not G2 reusable terminal evidence.

## GIT STATUS
- This refreeze does not commit automatically. Use `git status --short --untracked-files=all` for the current tree.

## NEXT STAGE
- G2B may execute only after this nested protocol is reviewed. No G2B/G3/G4/G5 execution is started here.

G2_NESTED_HISTORY_PROTOCOL_FROZEN
