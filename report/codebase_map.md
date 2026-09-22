# Original Repository Codebase Map

Inspected repository: `original_repo/EvaLLMasRecommender`

This map is based on read-only inspection. No files under `original_repo/` were
modified, no archive was extracted, and no training or API script was run.

## Top-Level Structure

```text
original_repo/EvaLLMasRecommender/
+-- README.md
+-- requirements.txt
+-- assets/
+-- eva/
|   +-- codes/
|   +-- templates/
+-- llm_results/
|   +-- beauty.zip
|   +-- sports.zip
+-- ReChorus/
+-- sota_results/
    +-- beauty_his10.zip
```

## Main Directories

- `assets/`: contains `architecture.jpg` for the framework diagram.
- `eva/codes/`: contains prompt execution, output parsing, metric calculation,
  and command-runner scripts.
- `eva/templates/`: contains prompt templates for ranking, re-ranking, profile,
  few-shot, candidate-first, no-history, and LLMRank-style variants.
- `llm_results/`: contains compressed saved prompts, saved LLM outputs, samples,
  and small unlabeled metric CSVs for Beauty and Sports.
- `sota_results/`: contains a compressed Beauty traditional-baseline result
  archive.
- `ReChorus/`: present in this checkout but appears empty from file inspection.
  The README identifies it as the traditional-model code location, but no
  runnable ReChorus files are available in the current tree.

## Dataset Locations

The README says preprocessed data should live under:

```text
original_repo/EvaLLMasRecommender/data/beauty/
original_repo/EvaLLMasRecommender/data/sports/
```

It also links external Dropbox ZIPs for Beauty and Sports. Those expanded data
folders are not present in the current checkout.

The original metric script expects these files:

```text
data/<dataset>/datamaps.json
data/<dataset>/meta.json
data/<dataset>/sequential_data.txt
data/<dataset>/negative_samples.txt
```

The README also mentions `raw_data/`, but `raw_data/` is not present in this
checkout. Because of that, any local reproduction path that needs item metadata,
interaction histories, negative samples, or MostPop should first run a dataset
audit after obtaining or unpacking the expected preprocessed data.

## LLM Outputs

Saved LLM artifacts are compressed:

- `llm_results/beauty.zip`: 224 entries: 150 JSON, 71 CSV, 2 TXT.
- `llm_results/sports.zip`: 113 entries: 76 JSON, 34 CSV, 2 TXT.

The archives include:

- `sample.txt`: 200 sampled users.
- `sample_1k_all.txt` for Beauty and `sample_1k.txt` for Sports: 1,000-user
  sample files.
- `test_*.json`: prompt/input files.
- `results_*.json`: saved model responses.
- `results_*_5.csv`: upstream metric CSVs for top-5 evaluation. These CSVs are
  small and have no headers.

Observed JSON record schema:

```text
prompt      prompt text shown to the LLM
target      held-out item title
user        user id
cand_list   candidate item ids as strings
template    prompt template id
result      saved model output; present only in results_*.json
```

Observed LLM model families:

- Beauty: `glm4`, `gpt3.5`, `gpt4`, `llama3`.
- Sports: `glm4`, `gpt3.5`, `gpt4`, `llama2`, `llama3`.

Useful saved-output variants:

- Main ranking: `results_<model>_seed0_test_history10_candidate20_shuffle_sample.json`
- Positive candidate at first / unshuffled candidates:
  `results_<model>_seed0_test_history10_candidate20_sample.json`
- Randomized history order:
  `results_<model>_seed0_test_history10_shuffle_candidate20_shuffle_sample.json`
- History sensitivity: `history0`, `history1`, `history2`, `history3`,
  `history4`, `history5`, `history10`, `history15`, `history20`.
- Candidate-size variants: `candidate20` and `candidate100`.
- Re-ranking-like variant named in upstream script:
  `test_history10_multi_candidate20_shuffle_sample_1k` and corresponding
  `results_*_multi_candidate20_shuffle_sample_1k`.
- Profile variants:
  `profile_gpt3.5`, `profile_gpt4`, `profile_glm4`, with and without
  `_history`.
- LLMRank variants: `llmrank` and `llmrank_rf`.

Uncertainty: the paper-level meaning of each filename should be confirmed
against the paper and prompt templates before reporting experimental claims.
The names above reflect inspected filenames and comments in upstream scripts.

## Baseline Outputs

Available compressed traditional-output archive:

```text
sota_results/beauty_his10.zip
```

It contains 26 tab-separated CSV files under `beauty_his10/`. Observed schema:

```text
user_id    rec_items
1          [6421, 3408, ...]
```

Observed model files:

- `BM25`: seeds 0-4
- `BPRMF`: seeds 0-4
- `GRU4Rec`: seeds 0-4
- `LightGCN`: seeds 0-4
- `SASRec`: seeds 0-4
- `NRMS`: seed 0 only

No Sports SOTA archive was present in this checkout. `random.sh` references
Sports SOTA evaluation commands, so Sports baseline outputs may require a
separate download or regeneration. Regeneration would be outside the minimal
safe path unless explicitly requested.

## Metrics

Primary metric implementation:

```text
eva/codes/llm_metrics.py
```

The script computes and prints:

- Accuracy: `HR`, `NDCG`, `MRR`
- Popularity bias: `ARP`, `ACLT`, `PopREO`
- Fairness: `Gini`, `Jain`, `DPD`
- Diversity: `ItemCoverage`, `Shannon`, `Jaccard`
- Novelty: `Serendipity`, `SelfInformation`
- Hallucination: `Hallucination`

The saved `*_5.csv` files appear to contain one metric value per line, without
headers. The order is defined in `final_result` in `llm_metrics.py`:

```text
HR, NDCG, ACLT, ARP, PopREO, Serendipity, SelfInformation, Gini,
DPD, Jain, ItemCoverage, Shannon, Jaccard, Hallucination
```

Important implementation notes:

- LLM titles are mapped to item ids through `data/<dataset>/meta.json` and
  `data/<dataset>/datamaps.json`.
- Unmatched recommendations map to item id `0`; hallucination is calculated as
  the fraction of recommendation ids equal to `0`.
- Upstream `ItemCoverage` is coverage over the sampled candidate pool, not
  necessarily full catalog coverage. Our project extension should implement and
  label Catalog Coverage@K separately.
- Exact, normalized, and fuzzy matching are not clearly separated in the
  upstream metrics. Our wrapper should report them separately.
- Upstream metric CSVs have no provenance columns and should not be used as
  final project tables without conversion into the required schema.

## Ranking and Re-Ranking Scripts

Useful files:

- `eva/codes/llm_metrics.py`: reads saved LLM or SOTA outputs and computes
  metrics. It is useful as reference logic, but should not be run directly
  against `original_repo/` in this project because it writes metric CSVs beside
  input files when `save` is true.
- `eva/codes/output_refine.py`: parses list-like LLM outputs and pads/truncates
  to target length. Useful reference for robust output parsing tests.
- `eva/codes/metrics.sh`: command list for many saved-output evaluations,
  including profile, re-ranking, SOTA, LLMRank, and history-sensitivity runs.
- `eva/codes/random.sh`: command list for Sports SOTA metric runs.
- `eva/codes/random_test.py`: repeats shell commands and extracts
  `final_results` from stdout. Treat as an experiment runner, not as a library.
- `eva/codes/zero_shot.py`: calls LLM APIs to fill prompt results and retry
  invalid outputs. This is expensive and should not be run by default.
- `eva/codes/llms.py`: OpenAI/ZhipuAI API wrappers.
- `eva/codes/generate_prompt.ipynb`: prompt-generation notebook. It was not
  executed during this inspection.

Ranking appears to use candidate lists such as `candidate20` or `candidate100`.
The upstream `metrics.sh` labels `test_history10_multi_candidate20_shuffle_sample_1k`
as "rerank exp."; this should be treated as the available re-ranking-style
saved-output path unless confirmed more precisely.

## Useful Files for This Project

Best direct references:

- `llm_results/beauty.zip`
- `llm_results/sports.zip`
- `sota_results/beauty_his10.zip`
- `eva/codes/llm_metrics.py`
- `eva/codes/output_refine.py`
- `eva/templates/template.txt`
- `eva/templates/template_candidate_first.txt`
- `eva/templates/template_wohistory.txt`
- `eva/templates/llmrank.txt`
- `eva/templates/llmrank_rf.txt`

Useful schema sources:

- Prompt JSONs inside `llm_results/*.zip`
- Result JSONs inside `llm_results/*.zip`
- Baseline CSVs inside `sota_results/beauty_his10.zip`

## Risky or Expensive Parts

Do not run these as part of the minimal reproduction:

- `eva/codes/zero_shot.py`: calls OpenAI/ZhipuAI APIs and writes result JSONs.
- `eva/codes/llms.py`: contains live API calls. It also references `ZhipuAI`
  without an observed import in the file, so GLM execution may fail without
  patching.
- `eva/codes/random_test.py`: executes shell commands through
  `subprocess.check_output(..., shell=True)`.
- `eva/codes/metrics.sh` and `eva/codes/random.sh`: batch command scripts that
  can write metric CSVs under the upstream result folders.
- Any ReChorus or SOTA training path: no runnable ReChorus files were present
  in the current checkout, and training would be outside the current phase.

Additional caution:

- `llm_metrics.py` uses relative paths such as `../../data/...` and
  `../../llm_results/...`, so behavior depends on current working directory.
- `llm_metrics.py` defaults to `save=True` and writes CSV outputs next to the
  evaluated result files. Its boolean CLI arguments use `type=bool`, which is
  error-prone for values like `--save False`.
- The SOTA branch appears brittle: the main script passes `results_path` into a
  function parameter named `dataset`. This should be validated before relying on
  direct upstream SOTA execution.
- Upstream dependencies are old and heavy (`torch==1.13.1`,
  `pandas==0.24.2`, `numpy==1.22.0`) and are not needed for a lightweight
  parser/evaluation wrapper.

## Recommended Minimal Reproduction Path

1. Keep `original_repo/` read-only.
2. Build local loaders under `src/` that read ZIP contents without extracting
   into `original_repo/`.
3. For LLM outputs, start with:

   ```text
   llm_results/beauty.zip:
   beauty/results_gpt3.5_seed0_test_history10_candidate20_shuffle_sample.json

   llm_results/sports.zip:
   sports/results_gpt3.5_seed0_test_history10_candidate20_shuffle_sample.json
   ```

4. Parse each JSON record into the project recommendation schema, retaining
   `dataset`, `model_name`, `user_id`, `rank`, `recommended_item_text`,
   `raw_output`, `source_file`, `prompt_variant`, `candidate_ordering`,
   `history_length`, and `run_id`.
5. For candidate-level evaluation, use each record's `target`, `cand_list`, and
   candidate titles embedded in the prompt. Mark any candidate-title extraction
   uncertainty explicitly and test the parser on fixed examples.
6. Download or otherwise locate the expected preprocessed `data/<dataset>/`
   files before implementing MostPop, full catalog coverage, or exact
   title-to-item-id catalog matching.
7. Use `sota_results/beauty_his10.zip` only as an optional Beauty traditional
   baseline source after the dataset audit confirms user ids and ground truth
   align with the archive.
8. Compute local HR@5, NDCG@5, hallucination, and Catalog Coverage@5 in
   `src/evaluation/`, with tests and provenance columns required by
   `AGENTS.md`.
9. Save all project-generated metrics under `results/metrics/` and all analysis
   under `report/`; do not write derived files into `original_repo/`.

## Current Blockers and Uncertainties

- The upstream preprocessed `data/` directory is missing in this checkout.
- Sports traditional baseline outputs are referenced but not present.
- ReChorus is present as a directory but appears empty.
- Prompt parsing can recover candidate titles from text, but robust
  item-id/catalog matching requires the missing metadata files.
- The exact correspondence between every filename variant and paper experiment
  should be confirmed before final reporting.
