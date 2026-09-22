# Original Artifact Map

Scope: this map inspects `original_repo/EvaLLMasRecommender` and the links
referenced by its README to identify artifacts that could unlock real Beauty
and Sports reproduction metrics after a later conversion phase.

No files under `original_repo/` were modified. No datasets were downloaded, no
APIs were called, no models were trained, no metrics were computed, and no
project configs were changed in this phase.

## Inspected Sources

- Project documentation and configs: `PROJECT_SPEC.md`, `AGENTS.md`,
  `CODEX_PROMPTS.md`, `README.md`, `report/codebase_map.md`,
  `report/dataset_audit.md`, `report/quality_review.md`,
  `configs/beauty.yaml`, and `configs/sports.yaml`.
- Project readiness code: dataset audit, loaders, schemas, reproduction,
  coverage, position-bias, history-length, and final pipeline scripts.
- Full local original repository tree under `original_repo/EvaLLMasRecommender`.
- Original evaluation and prompt scripts:
  - `original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py`
  - `original_repo/EvaLLMasRecommender/eva/codes/generate_prompt.ipynb`
  - `original_repo/EvaLLMasRecommender/eva/codes/metrics.sh`
  - `original_repo/EvaLLMasRecommender/eva/codes/random.sh`
  - `original_repo/EvaLLMasRecommender/eva/codes/zero_shot.py`
- Local ZIP archives, inspected without extraction:
  - `original_repo/EvaLLMasRecommender/llm_results/beauty.zip`
  - `original_repo/EvaLLMasRecommender/llm_results/sports.zip`
  - `original_repo/EvaLLMasRecommender/sota_results/beauty_his10.zip`

## Local Original Tree

The local original repository contains:

```text
original_repo/EvaLLMasRecommender/
|-- README.md
|-- requirements.txt
|-- assets/
|   `-- architecture.jpg
|-- eva/
|   |-- codes/
|   |   |-- generate_prompt.ipynb
|   |   |-- llm_metrics.py
|   |   |-- llms.py
|   |   |-- metrics.sh
|   |   |-- output_refine.py
|   |   |-- random.sh
|   |   |-- random_test.py
|   |   `-- zero_shot.py
|   `-- templates/
|       |-- llmrank.txt
|       |-- llmrank_rf.txt
|       |-- template.txt
|       |-- template_candidate_first.txt
|       |-- template_category.txt
|       |-- template_demonstration.txt
|       |-- template_fewshot.txt
|       |-- template_in_context.txt
|       |-- template_profile.txt
|       |-- template_profile_history.txt
|       |-- template_rating.txt
|       |-- template_user_profile_generate.txt
|       `-- template_wohistory.txt
|-- llm_results/
|   |-- beauty.zip
|   `-- sports.zip
|-- ReChorus/
`-- sota_results/
    `-- beauty_his10.zip
```

No expanded `data/`, `raw_data/`, expanded `llm_results/beauty/`,
expanded `llm_results/sports/`, expanded `sota_results/beauty_his10/`, or
Sports SOTA archive is present locally.

## Original Evaluation Dependency Chain

The most important original code path is
`original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py`.

- `read_meta(dataset)` reads `../../data/{dataset}/datamaps.json` and
  `../../data/{dataset}/meta.json`. It uses `id2item` from `datamaps.json`
  plus `asin` and `title` from `meta.json` to map item titles to internal item
  ids and item ids to categories.
- The main block reads sampled users from `../../llm_results/{dataset}/sample.txt`.
- It reads `../../data/{dataset}/sequential_data.txt`. The local code comment
  says each row is `1: user_id; 2-n: user_history item_id`. The code uses
  `x[1:-1]` as training/history interactions and `x[-1]` as the test target.
- It reads `../../data/{dataset}/negative_samples.txt`. For candidate ranking,
  the code builds the candidate list as the target item from
  `sequential_data.txt` plus negative items from `negative_samples.txt`, unless
  a prompt-specific candidate list is supplied.
- LLM result JSON records are expected to include `result`, `target`, and
  `user`; observed local result records also include `prompt`, `cand_list`, and
  `template`.
- SOTA recommendation files are read from
  `../../sota_results/{dataset}/rec-{model}-history{history_max}-shuffle{shuffle}-seed{random_seed}.csv`
  with tab-separated columns `user_id` and `rec_items`.
- SOTA utility targets still come from `sequential_data.txt`, not from the SOTA
  output files.

`generate_prompt.ipynb` documents prompt construction and references
`sequential_data.txt`, `negative_samples.txt`, `datamaps.json`, and `meta.json`.
Some notebook cells reference these files under `../../llm_results/{dataset}/`,
while `llm_metrics.py` references `../../data/{dataset}/`. The local ZIPs do
not contain those data files in either location, so this path discrepancy is an
ambiguity to resolve only after the README-linked data is manually obtained.

## README-Linked Data

The original README references preprocessed data downloads:

- Beauty preprocessed data:
  `https://www.dropbox.com/scl/fi/t744vcqf8zmoyjhoj41yl/beauty.zip?rlkey=oojaz38gpjqjhnmrlo33el5b0&st=2wn7qlta&dl=0`
- Sports preprocessed data:
  `https://www.dropbox.com/scl/fi/uedey8lf34kbv3vzer7no/sports.zip?rlkey=9ekgqctdlp1g1au627cxox2m4&st=e0sosryh&dl=0`

### External artifact availability check

Context for this phase records that manual browser access to both README-linked
Dropbox items reports deleted files, and curl attempts returned HTML/error
pages rather than valid ZIP archives. No network check, download, or artifact
recovery was run in this phase.

- Beauty Dropbox link: unavailable/deleted.
- Sports Dropbox link: unavailable/deleted.

Consequence: `sequential_data.txt`, `negative_samples.txt`, `datamaps.json`,
and `meta.json` cannot currently be recovered from the README-linked
preprocessed archives.

Impact: `ground_truth_path`, `candidates_path`, `catalog_path`, and
`interactions_path` remain unavailable unless these artifacts are obtained from
the authors or reconstructed from valid raw data in a separate, documented
phase.

Based on the original scripts, the expected useful members of the unavailable
preprocessed archives are likely:

- `sequential_data.txt`
- `negative_samples.txt`
- `datamaps.json`
- `meta.json`

The README also references `./raw_data` for raw Amazon Beauty and Sports 2014
data and preprocessing. The raw data directory is not present locally. Raw data
is not required if the preprocessed data can be manually supplied and validated.

## ZIP Inventory Summary

Local archive inspection, without extraction:

| Archive | Local | Key contents |
|---|---:|---|
| `llm_results/beauty.zip` | yes | 224 members: 150 JSON files, 71 `_5.csv` metric CSV files, 2 sample TXT files |
| `llm_results/sports.zip` | yes | 113 members: 76 JSON files, 34 `_5.csv` metric CSV files, 2 sample TXT files |
| `sota_results/beauty_his10.zip` | yes | 26 recommendation TSV files for BM25, BPRMF, GRU4Rec, LightGCN, SASRec seeds 0-4 and NRMS seed 0 |
| `sota_results/sports_his10.zip` | no | Sports SOTA outputs are referenced by `random.sh` but not present locally |

The LLM JSON prompt files have records with keys:

```text
prompt, target, user, cand_list, template
```

The LLM JSON result files have records with keys:

```text
prompt, target, user, cand_list, template, result
```

The Beauty SOTA recommendation files have tab-separated columns:

```text
user_id, rec_items
```

The local `_5.csv` files inside LLM ZIPs appear to be original metric output
vectors without headers. They are useful as upstream reference artifacts, but
they should not be treated as project-generated real metrics unless a later
phase explicitly maps their row order and provenance.

## Beauty Artifact Map

| Question | Artifact(s) | Local/readable now | Confidence | Conversion notes and risks |
|---|---|---:|---|---|
| User histories | `original_repo/EvaLLMasRecommender/data/beauty/sequential_data.txt` | no | high | README-linked download is unavailable; author-provided artifact or raw-data reconstruction is required. Convert each sequence row into canonical `interactions` rows; preserve item order through `timestamp` or rank. Risk: exact recovered path and delimiter must be validated. |
| User histories, sampled prompt text | `llm_results/beauty.zip!beauty/test_*.json` and `results_*.json` `prompt` fields | yes | medium | Prompts contain rendered history text for sampled users. Useful for audit/prompt reproduction, but not a complete canonical interaction file and not reliable for item-id histories without parsing text. |
| Train/test split | `data/beauty/sequential_data.txt` | no | high | Original code treats all items except last as train/history and last as test target. Convert split labels deterministically after recovery. |
| Ground-truth target item per user | Last item in `data/beauty/sequential_data.txt`; also `target` title in LLM JSON | partial | high for sequence, medium for JSON title | Best canonical source is last item id from sequence file. LLM JSON target titles can be converted only if `datamaps.json` plus `meta.json` are available and title mapping is unambiguous. |
| Negative samples/candidate sets | `data/beauty/negative_samples.txt`; `llm_results/beauty.zip!beauty/test_*candidate*.json` `cand_list`; `results_*candidate*.json` `cand_list` | partial | high | Full negative sample file is missing. Prompt/result JSON `cand_list` can produce sampled canonical candidates because it stores item ids as strings, but only for the prompt users/variants. |
| Item catalog metadata | `data/beauty/meta.json` | no | high | README-linked download is unavailable; author-provided artifact or raw-data reconstruction is required. Convert to canonical catalog with item id, ASIN, title, categories when paired with `datamaps.json`. |
| Item id to title mapping | `data/beauty/datamaps.json` plus `data/beauty/meta.json` | no | high | Original code maps `id2item` internal id -> ASIN, then ASIN -> title from metadata. Must handle missing titles and duplicate titles explicitly. |
| ASIN to item id mapping | `data/beauty/datamaps.json` | no | high | Expected keys include `item2id` and `id2item` from notebook output. Validate exact JSON after recovery. |
| LLM prompts | `llm_results/beauty.zip!beauty/test_*.json` | yes | high | Can convert to prompt/input audit records and sampled candidates. Do not call APIs. |
| LLM outputs | `llm_results/beauty.zip!beauty/results_*.json` | yes | high | Can normalize recommendations by text now; item-id matching requires catalog mappings. Current config targets `results_gpt3.5_seed0_test_history10_candidate20_shuffle_sample.json`. |
| Traditional baseline outputs | `sota_results/beauty_his10.zip!beauty_his10/rec-*.csv` | yes | high | Can normalize recommendation ids and rank. Utility still blocked until canonical ground truth from `sequential_data.txt` exists. |
| Original metric vectors | `llm_results/beauty.zip!beauty/results_*_5.csv` | yes | medium | Upstream reference only. Do not copy as project metrics without schema/provenance mapping. |
| README-referenced but absent | Expanded `data/beauty/`, `raw_data/`, expanded LLM/SOTA dirs | no | high | README-linked preprocessed Dropbox item is unavailable/deleted. Recovery now requires author-provided artifacts or raw-data reconstruction. |

### Beauty Config Fields Fillable After Conversion

- `paths.interactions_path`: fill from converted `sequential_data.txt`.
- `paths.catalog_path`: fill from converted `meta.json` joined to `datamaps.json`.
- `paths.candidates_path`: fill from converted `negative_samples.txt` for the
  full candidate pool, or from LLM JSON `cand_list` for sampled prompt
  candidates if explicitly labeled as prompt-sample candidates.
- `paths.ground_truth_path`: fill from the last item in each converted
  `sequential_data.txt` row, aligned to the sampled users when evaluating
  sampled outputs.
- `paths.llm_outputs_path`: already points to the local Beauty LLM ZIP.
- `paths.baseline_outputs_path`: already points to the local Beauty SOTA ZIP.

## Sports Artifact Map

| Question | Artifact(s) | Local/readable now | Confidence | Conversion notes and risks |
|---|---|---:|---|---|
| User histories | `original_repo/EvaLLMasRecommender/data/sports/sequential_data.txt` | no | high | README-linked download is unavailable; author-provided artifact or raw-data reconstruction is required. Convert sequence rows into canonical `interactions`; preserve chronological order. |
| User histories, sampled prompt text | `llm_results/sports.zip!sports/test_*.json` and `results_*.json` `prompt` fields | yes | medium | Prompt text is local for sampled users, but it is not a complete canonical interaction source. |
| Train/test split | `data/sports/sequential_data.txt` | no | high | Original code uses all but last item as train/history and last item as test target. |
| Ground-truth target item per user | Last item in `data/sports/sequential_data.txt`; also `target` title in LLM JSON | partial | high for sequence, medium for JSON title | Best canonical source is sequence file. LLM JSON target titles need catalog mappings before item-id utility evaluation. |
| Negative samples/candidate sets | `data/sports/negative_samples.txt`; `llm_results/sports.zip!sports/test_*candidate*.json` `cand_list`; `results_*candidate*.json` `cand_list` | partial | high | Full negative sample file is missing. Local `cand_list` can create sampled candidate records for prompt variants. |
| Item catalog metadata | `data/sports/meta.json` | no | high | README-linked download is unavailable; author-provided artifact or raw-data reconstruction is required. Convert to canonical catalog with item id, ASIN, title, categories when paired with `datamaps.json`. |
| Item id to title mapping | `data/sports/datamaps.json` plus `data/sports/meta.json` | no | high | Needed to map LLM title outputs and targets to item ids. |
| ASIN to item id mapping | `data/sports/datamaps.json` | no | high | Expected keys include `item2id` and `id2item` from notebook output. Validate after recovery. |
| LLM prompts | `llm_results/sports.zip!sports/test_*.json` | yes | high | Can convert to prompt/input audit records and sampled candidates. |
| LLM outputs | `llm_results/sports.zip!sports/results_*.json` | yes | high | Can normalize recommendation text now; item-id matching remains blocked until catalog mappings are available. Current config targets `results_gpt3.5_seed0_test_history10_candidate20_shuffle_sample.json`. |
| Traditional baseline outputs | Expected `sota_results/sports_his10.zip` or expanded `sota_results/sports/rec-*.csv` | no | medium | `random.sh` references Sports SOTA evaluation commands, but no Sports SOTA archive is local and the README does not provide a direct Sports SOTA link. |
| Original metric vectors | `llm_results/sports.zip!sports/results_*_5.csv` | yes | medium | Upstream reference only. Do not copy as project metrics without schema/provenance mapping. |
| README-referenced but absent | Expanded `data/sports/`, `raw_data/`, expanded LLM dirs, Sports SOTA outputs | no | high | README-linked preprocessed Dropbox item is unavailable/deleted. Recovery now requires author-provided artifacts or raw-data reconstruction. Sports SOTA source remains uncertain. |

### Sports Config Fields Fillable After Conversion

- `paths.interactions_path`: fill from converted `sequential_data.txt`.
- `paths.catalog_path`: fill from converted `meta.json` joined to `datamaps.json`.
- `paths.candidates_path`: fill from converted `negative_samples.txt` for the
  full candidate pool, or from LLM JSON `cand_list` for sampled prompt
  candidates if explicitly labeled as prompt-sample candidates.
- `paths.ground_truth_path`: fill from the last item in each converted
  `sequential_data.txt` row, aligned to sampled users.
- `paths.llm_outputs_path`: already points to the local Sports LLM ZIP.
- `paths.baseline_outputs_path`: remains null until a real Sports SOTA output
  artifact is manually supplied or another valid baseline output is generated
  from configured interactions in a later phase.

## Artifact Details And Conversion Plan

The machine-readable inventory is saved at
`results/tables/original_artifact_inventory.csv`. It records each artifact or
artifact family with path, dataset, likely role, local existence, expected
schema/format, confidence, conversion route, and risks.

High-confidence conversion routes for a later phase:

1. Convert `sequential_data.txt` to canonical interactions and ground truth.
   Each row should become ordered interaction rows. The last item should become
   `ground_truth_item_id`; preceding items become train/history interactions.
2. Convert `negative_samples.txt` to canonical candidates. Preserve
   `candidate_position`; include the positive target from the corresponding
   sequence row as position 1 only if reproducing the original construction.
3. Convert `datamaps.json` and `meta.json` to a canonical catalog and mapping
   table. Preserve internal item id, ASIN, title, and categories where present.
4. Convert LLM `test_*.json` files to prompt/input records and sampled
   candidate sets. Convert LLM `results_*.json` files to normalized
   recommendation outputs, preserving raw text and match type.
5. Convert Beauty SOTA `rec-*.csv` files to canonical recommendation outputs.
   Do not compute utility until the ground-truth table is available.

## Risks And Ambiguities

- The core preprocessed files are referenced by README but are not local, and
  the README-linked Dropbox items are currently unavailable/deleted. Their
  exact paths and formats can only be validated if equivalent author-provided
  artifacts are recovered or the data is reconstructed from raw sources.
- `generate_prompt.ipynb` references data files under `llm_results/{dataset}/`,
  while `llm_metrics.py` references `data/{dataset}/`. Treat this as a path
  ambiguity, not as evidence that those files exist in local LLM ZIPs.
- LLM JSON `target` values are titles, not canonical item ids. Using them for
  utility requires a validated title-to-id mapping. Duplicate titles or missing
  titles must be handled explicitly.
- LLM JSON `cand_list` values are stringified Python lists of item ids. They are
  usable after safe parsing but only for the sampled prompt variants.
- Original `_5.csv` files are headerless upstream metric vectors. They should
  not be merged into `results/metrics/` as project metrics without a separate
  provenance and schema mapping phase.
- Beauty SOTA recommendations are local, but the full target labels required to
  evaluate them are still missing until `sequential_data.txt` is available.
- Sports SOTA outputs are referenced by scripts but are missing locally. No
  Sports SOTA path should be filled in configs until a real artifact is present.

## Recommended Minimal Next Step

Do not change configs yet. Prompt 18.5B should be deferred unless the missing
preprocessed artifacts are supplied by the authors or reconstructed from valid
raw data in a separate phase. The next safe data phase, once artifacts exist,
should place them outside `original_repo/`, then implement a converter that
reads only local files and writes canonical CSVs under
`data/processed/{dataset}/`. After validating schemas, configs can be updated
to point to:

- `data/processed/{dataset}/interactions.csv`
- `data/processed/{dataset}/catalog.csv`
- `data/processed/{dataset}/candidates.csv`
- `data/processed/{dataset}/ground_truth.csv`

Only after those files exist should the reproduction pipeline compute HR@5,
NDCG@5, hallucination, and coverage from project-generated CSVs.
