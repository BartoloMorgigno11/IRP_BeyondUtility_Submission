# Neural Baseline Environment and Feasibility Audit

## 1. Scope and provenance

- Phase 3B audit-only task for BPRMF, GRU4Rec, SASRec and LightGCN.
- Branch: `plan-d-extensions`.
- Commit: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests`.
- Generated tables: `results/tables/neural_environment_inventory.csv`, `neural_hardware_inventory.csv`, `neural_baseline_feasibility_matrix.csv`, `neural_baseline_gaps.csv`, `neural_baseline_execution_roadmap.csv`.

## 2. Executive summary

- One existing environment, `repo_venv_cuda`, can import all four local Plan D neural modules and exposes CUDA through PyTorch.
- No neural baseline is ready for controlled Plan E2 execution today because the current runners target Plan C `candidates.csv`/`ground_truth.csv`, not the Phase 2C Plan E2 manifests.
- BPRMF is the safest first target because it has the simplest data contract and no sequence or graph construction risk.
- LastFM remains data/evaluation blocked for all four baselines.

## 3. Safety boundaries

- No installs, downloads, environment modifications, training, scoring, checkpoint creation, LLM inference, Ollama invocation, or `original_repo/` edits were performed.
- Import probes were limited to package/module imports and tiny tensor checks for environment visibility.

## 4. Hardware inventory

- Hardware is recorded in `results/tables/neural_hardware_inventory.csv`.
- Detected GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB reported VRAM, driver 595.79, driver CUDA 13.2, compute capability 8.9.
- CPU/RAM: Intel Core i9-14900HX, 24 physical cores, 32 logical processors, 32 GiB installed RAM.

## 5. Python environment inventory

- Environment details are in `results/tables/neural_environment_inventory.csv`.
- `repo_venv_cuda` is the only environment with CUDA-enabled PyTorch and successful local Plan D imports.
- Anaconda base is suitable for audit scripting but lacks `torch`.

## 6. CUDA and framework availability

- `nvidia-smi` reported CUDA 13.2 at the driver level.
- `.venv` PyTorch reports `torch==2.6.0+cu124`, `torch.version.cuda==12.4`, CUDA available, one device, and a successful tiny CUDA tensor probe.
- TensorFlow is present in several conda envs, but current local baseline code is PyTorch.

## 7. Dependency-file assessment

- `requirements.txt` and `environment.yml` declare broad `torch>=2.0`; they do not capture the existing `.venv` package state.
- `environment.yml` declares Python 3.10, while `.venv` is Python 3.12.10.
- Official `original_repo/EvaLLMasRecommender/requirements.txt` pins `torch==1.13.1`, `numpy==1.22.0`, and old `pandas==0.24.2`, so exact official environment parity is not locally demonstrated.

## 8. Official-repository neural baseline inventory

- `original_repo/EvaLLMasRecommender/eva/codes/metrics.sh` and `random.sh` call `llm_metrics.py --sota True` for BPRMF, GRU4Rec, SASRec and LightGCN.
- `llm_metrics.py` reads precomputed SOTA recommendation CSVs from `../../sota_results/{dataset}/rec-{model}-...csv` and uses `sequential_data.txt` plus `negative_samples.txt` for evaluation.
- `original_repo/EvaLLMasRecommender/ReChorus` is locally empty, so official trainable model code is not available in this checkout.
- `original_repo/EvaLLMasRecommender/sota_results/beauty_his10.zip` contains Beauty precomputed recommendation CSVs for all four baselines, but no model checkpoints.

## 9. Current custom-repository neural baseline inventory

- Local PyTorch Plan D implementations exist: `src/baselines/plan_d_bprmf.py`, `plan_d_gru4rec.py`, `plan_d_sasrec.py`, `plan_d_lightgcn.py`.
- Runners exist under `src/experiments/run_plan_d_*.py` and write metrics/recommendations when executed; they were not executed in this audit.
- Historical Plan D summaries under `results/tables/plan_d_*_training_summary.csv` provide prior execution evidence only.

## 10. BPRMF implementation assessment

- Matrix-factorization embeddings with BPR loss and deterministic negative sampler are implemented.
- Candidate scoring exists through `rank_bprmf_candidates`, but only for prepared Plan C candidate DataFrames.
- Feasibility is environment-ready but Plan E2 adapter missing.

## 11. GRU4Rec implementation assessment

- GRU encoder with candidate dot-product scoring and train-history sequence construction is implemented.
- It requires non-empty ordered train histories; timestamp ordering is used when available.
- Feasibility is environment-ready but Plan E2 adapter and sequence-cohort checks are missing.

## 12. SASRec implementation assessment

- Small Transformer encoder with padding index 0, max sequence length 50, two heads and two layers by default is implemented.
- It has the highest local runtime risk among Plan D extended-subset logs.
- Feasibility is environment-ready but adapter and sequence validity checks are missing.

## 13. LightGCN implementation assessment

- Sparse normalized bipartite graph propagation with BPR loss is implemented.
- Graph can be derived from training interactions in memory, but no Plan E2 adapter exists.
- Feasibility is environment-ready with graph-memory checks required before execution.

## 14. Beauty feasibility

- Plan E2 item coverage against datamaps/catalog: 1.0; train-known item coverage: 0.999285568483.
- Ground truth, negative samples, timestamps and Plan E2 ranking/reranking manifests exist.
- Feasible after adapter approval; not exact paper reproduction.

## 15. Sports feasibility

- Plan E2 item coverage against datamaps/catalog: 1.0; train-known item coverage: 0.999589153657.
- Ground truth, negative samples, timestamps and Plan E2 manifests exist.
- Feasible after adapter approval, with title gaps irrelevant to neural scoring but relevant to LLM comparisons.

## 16. ML-1M feasibility

- Plan E2 item coverage against datamaps/catalog: 1.0; train-known item coverage: 0.999187652315.
- Complete Plan E2 full/matched cohorts and prior Plan D neural runtime evidence exist.
- Best first dataset for BPRMF controlled pilot.

## 17. LastFM feasibility

- LastFM has interactions/catalog/datamaps but no Plan E2 candidate sets, ground truth, negative samples or sequential_data.txt.
- All four neural baselines are data/evaluation blocked for LastFM.

## 18. Training-data and leakage assessment

- Local Plan D code trains from reconstructed Plan C train interactions and evaluates candidates separately.
- For Phase 2C Plan E2 execution, adapter code must preserve train/test separation and must not derive rankings from held-out labels.

## 19. User/item mapping compatibility

- Beauty, Sports and ML-1M Plan E2 raw IDs map completely through reconstructed datamaps/catalogs.
- A few Plan E2 candidate items are absent from train interactions, so cold/untrained candidate behavior must be explicit before execution.

## 20. Plan E2 candidate-scoring compatibility

- Existing scorers can rank supplied candidate DataFrames after training.
- Existing runners do not read `ranking_candidate_sets.csv` or `experiment_cohort_membership.csv`; this is the primary adapter gap.

## 21. Checkpoint inventory

- No `.pt`, `.pth`, `.ckpt`, `.onnx`, `.pkl`, or `.joblib` model checkpoint was found outside ignored virtualenv paths.
- Beauty SOTA ZIP contains precomputed recommendation CSVs, not reusable checkpoints.

## 22. Import and test probe findings

- `.venv` imports all four Plan D modules successfully.
- Anaconda base import probes fail with `ModuleNotFoundError: torch` for all four modules.
- No neural tests were run because the task prohibits neural training/scoring; import-only probes were sufficient for this audit.

## 23. Runtime and GPU-memory assessment

- Plan D extended-subset logs show 5-epoch, 1000-user CUDA runs in seconds-to-minutes, depending on baseline and dataset.
- Full Plan E2-aligned execution is estimated qualitatively only; BPRMF is under-one-hour on the detected GPU once adapter gaps are resolved.
- SASRec is the highest runtime-risk baseline because of Transformer sequence processing.

## 24. Exact-paper alignment limitations

- Exact local support is not available for any of the four baselines: official trainable implementation is absent, exact official processed data/checkpoints are absent, and local Plan D code is adapted.
- Matrix `paper_alignment` is therefore `adapted` for Beauty/Sports/ML-1M and `not_aligned` for LastFM.

## 25. Required environment changes

- None are executed in Phase 3B.
- If `.venv` is accepted, no immediate environment change is required for import capability.
- A future reproducible environment spec should be prepared before controlled execution because dependency files do not pin the working `.venv` state.

## 26. Required code adapters

- A bounded Plan E2 adapter is required to read canonical candidate rows, map raw IDs, score all 20 candidates, and write new neural rankings/metrics without mutating Phase 2/3A artifacts.
- Cold/untrained candidate-item policy must be explicit.

## 27. Required data transformations

- For Beauty/Sports/ML-1M, graph and sequence tensors can be derived in memory from existing reconstructed interactions.
- LastFM requires a separate approved candidate/ground-truth construction phase and remains blocked here.

## 28. Baseline execution priority

- Priority order is recorded in `results/tables/neural_baseline_execution_roadmap.csv`.
- BPRMF first, then LightGCN, then GRU4Rec/SASRec, with LastFM left blocked.

## 29. Recommended first neural-baseline task

- Controlled BPRMF Plan E2 adapter implementation audit/fix, without training until separately approved.

## 30. Stop conditions and risk controls

- Stop on any candidate/cohort mutation, missing mapping, non-finite loss, CUDA OOM, unexpected output writes, checkpoint creation, or historical artifact change.
- Matched metrics must always be filtered from the same full-cohort prediction rows.

## 31. Canonical Phase 2 and Phase 3A files to preserve

- Preserve `results/tables/ranking_candidate_sets.csv`, `experiment_cohort_membership.csv`, `candidate_item_title_map.csv`, and all Phase 3A traditional baseline CSVs.
- Preserve historical Plan D outputs as evidence only.

## 32. Commands executed

- Provenance: `git status --short --branch`, `git branch --show-current`, `git log -1 --format="%H %s"`.
- Hardware: `nvidia-smi`, NVIDIA query, CIM OS/CPU/RAM, `Get-PSDrive`.
- Environment: `where.exe`, `Get-Command`, direct Python version/import probes, `conda env list`.
- Repository: `rg`, `Get-Content`, `Select-String`, archive member listing with `tar -tf`.
- Output: generated the five CSVs and this report; validation command recorded in final response.

## 33. Recommended single next task

- Controlled BPRMF Plan E2 adapter implementation audit/fix, without model training or project-data inference until separately approved.
