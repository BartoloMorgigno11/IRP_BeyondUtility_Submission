# Phase F6F2 Scientific Readout

## 1. Initial repository state
Branch `plan-d-extensions`, HEAD `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`, staged files `none`. Tracked dirty files remain `M README.md
 M report/dataset_audit.md
 M report/notes.md`.

## 2. Environment used
`C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_repro_system_site\Scripts\python.exe` with classification `reconstructed_venv_with_system_site_packages`.

## 3. Frozen-input verification
Rows `59`; changed `0`, missing `0`.

## 4. F6F1 scientific-hash verification
`4bf9391b835dad47c80f41b9d41b130f64ac3f4776561cc19bf0a3d044ffb37c`.

## 5. Metric exactness readout
Rows `14`; no blocked metric is promoted to exact.

## 6. APLT findings
Rows `270`. APLT uses the F6F1 q80 train-frequency policy.

## 7. Serendipity findings
Rows `254`. Serendipity remains reconstructed MostPop-hit evidence.

## 8. Popularity and novelty findings
Rows `1924`. Novelty, average popularity, APLT, and coverage stay separate.

## 9. Candidate-position findings
Rows `3083`. Position evidence is observational; full CandDif remains blocked.

## 10. CandDif status
Only diagnostic or blocked CandDif evidence is allowed; no causal claim is made.

## 11. Ranking versus re-ranking findings
Rows `97`; LastFM re-ranking remains excluded.

## 12. LLM versus baseline findings
Rows `1008` with paired intersections and bootstrap CIs from F6F1.

## 13. Base versus LLMRank findings
F6E4C paired strategy rows are reported in the strategy/model readout.

## 14. Qwen2 versus Mistral findings
F6E4C paired model rows are reported with no universal superiority claim.

## 15. Research-question closure
Rows `12` with exactly one classification per RQ.

## 16. Claim registry
Rows `10` with allowed and prohibited wording.

## 17. Remaining replication gaps
Rows `14`.

## 18. F6G1 requirements
Rows `12`.

## 19. Implementation files
`src/analysis/plan_f6f2_scientific_readout.py`, `src/experiments/build_phase_f6f2_scientific_readout.py`, `configs/phase_f6f2_scientific_readout.yaml`.

## 20. Tests
{"suites":[{"command":"C:\\Users\\Bartolo\\Desktop\\PROGETTI\\IRP_BeyondUtility_Project\\.venv_repro_system_site\\Scripts\\python.exe -m pytest tests/test_plan_f6f2_scientific_readout.py -q","failed":0,"passed":25,"skipped":0,"status":"pass","suite":"F6F2 tests"},{"command":"C:\\Users\\Bartolo\\Desktop\\PROGETTI\\IRP_BeyondUtility_Project\\.venv_repro_system_site\\Scripts\\python.exe -m pytest tests/test_plan_f6f1_beyond_utility_metrics.py -q","failed":0,"passed":13,"skipped":0,"status":"pass","suite":"F6F1 tests"},{"command":"C:\\Users\\Bartolo\\Desktop\\PROGETTI\\IRP_BeyondUtility_Project\\.venv_repro_system_site\\Scripts\\python.exe -m pytest tests/test_plan_f6f1r_environment_recovery.py -q","failed":0,"passed":22,"skipped":0,"status":"pass","suite":"F6F1R tests"},{"command":"C:\\Users\\Bartolo\\Desktop\\PROGETTI\\IRP_BeyondUtility_Project\\.venv_repro_system_site\\Scripts\\python.exe -m pytest tests/test_plan_f6e4c_scientific_consolidation.py -q","failed":0,"passed":10,"skipped":0,"status":"pass","suite":"F6E4C tests"},{"command":"C:\\Users\\Bartolo\\Desktop\\PROGETTI\\IRP_BeyondUtility_Project\\.venv_repro_system_site\\Scripts\\python.exe -m pytest tests/test_finalization_f6d4a_protected_input_guard.py -q","failed":0,"passed":1,"skipped":0,"status":"pass","suite":"protected input guard"},{"command":"C:\\Users\\Bartolo\\Desktop\\PROGETTI\\IRP_BeyondUtility_Project\\.venv_repro_system_site\\Scripts\\python.exe -m pytest tests/test_canonical_llm_contract.py -q","failed":0,"passed":16,"skipped":0,"status":"pass","suite":"minimal parser contract"}],"total_failed":0,"total_passed":87,"total_skipped":0,"unsafe_exclusions":["inference","ollama","network","cuda","training","downloads","full_suite"]}

## 21. Excluded unsafe tests
inference, ollama, network, cuda, training, downloads, full_suite

## 22. Determinism
F6F2 writes sorted CSV/JSON outputs and separates runtime metadata from scientific content.

## 23. Protected-input verification
Input invariance changed `0`, missing `0`.

## 24. Scientific freeze manifest
`results/manifests/phase_f6f2_scientific_readout_freeze.json`.

## 25. Missing or blocked evidence
Rows `6`.

## 26. Final decision
f6f2_scientific_readout_complete_with_documented_blocks

## 27. Recommended next action
proceed_to_lastfm_reranking_reconstruction

## 28. Final git state
No staging, commit, reset, clean, checkout, or frozen-output mutation is performed.

## 29. Files created or modified
Only new `phase_f6f2_*` outputs plus F6F2 config/source/test files.

## 30. Remaining limitations
F6F2 is a scientific readout over frozen adapted evidence; it does not create new predictions or repair blocked exact paper evidence.

## 31. Key numeric summary
F6F2-FIND-001 HR@5 mostly_negative mean_delta=-0.38094244075586359; F6F2-FIND-002 NDCG@5 mostly_negative mean_delta=-0.30048159276239816; F6F2-FIND-003 APLT@5 mostly_positive mean_delta=0.2769661412552144; F6F2-FIND-004 Novelty SelfInformation@5 mostly_positive mean_delta=1.1578879806034692; F6F2-FIND-005 Serendipity@5 mostly_negative mean_delta=-0.052480385457720102; F6F2-FIND-006 HR@5 mostly_negative mean_delta=-0.27048433174674225; F6F2-FIND-007 hr_at5 mostly_negative mean_delta=-0.026687497265426845; F6F2-FIND-008 hr_at5 mostly_positive mean_delta=0.044314054898508468
