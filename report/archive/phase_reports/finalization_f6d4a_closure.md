# Finalization F6D4A - Formal Closure After Protected-Input Repair

Decisione finale: `final_f6d4a_closure_complete`.

La precedente classificazione `final_f6d4a_closure_blocked_by_raw_invariance` era impropria: le raw response F6D4A erano e restano invariate. Il blocco reale riguardava dieci protected input F2 Plan E/report riscritti durante la suite pytest. La causa e stata corretta e i dieci file sono stati ripristinati alla baseline F2 autoritativa.

## 1. Initial repository state

- Repository: `C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project`.
- Branch: `plan-d-extensions`.
- HEAD: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`.
- Last commit: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`, `2026-07-02T04:33:35+02:00`, `Add Plan E3B BM25-like scoring on E2 manifests`.
- Dirty tracked preesistenti preservati: `README.md`, `report/dataset_audit.md`, `report/notes.md`.
- Nessuno staging, commit, reset, clean o checkout globale.

## 2. Ten changed F2 protected files

I dieci file coinvolti erano:

| Path | Categoria | Baseline F2 | Writer probabile |
| --- | --- | --- | --- |
| `report/plan_e_bm25_scoring.md` | Plan E report | `44aa0ddc70e91bc59781986369dc00f3f42476d2c69643942cee22a385d6a151` | `build_plan_e_bm25_scoring.py` |
| `report/plan_e_full_data_inventory.md` | Plan E report | `b8ffb9c46bb1a25588b8b26927f0a31693e44c13b85482325ac7458eb9675f68` | `build_plan_e_full_data_inventory.py` |
| `results/tables/plan_e_baseline_alignment_source_inventory.csv` | Plan E table | `d19089798c58559b08ed7fe9249bb566370f6593157072c49a5fc5bafd491c8b` | `build_plan_e_baseline_alignment.py` |
| `results/tables/plan_e_bm25_source_inventory.csv` | Plan E table | `99daaa64f8e502202e4b06d0089ca1036a8439d0d6bcde9e00e047a910b71a62` | `build_plan_e_bm25_scoring.py` |
| `results/tables/plan_e_full_data_inventory_artifact_summary.csv` | Plan E table | `c79039b86018ff16567537c2e6ba5cf8d4524441a10df9aff4b2ed792f568c93` | `build_plan_e_full_data_inventory.py` |
| `results/tables/plan_e_full_data_inventory_dataset_summary.csv` | Plan E table | `f932352423caaf1b2905f6382e19a106a9633ed21449039ad658a3bd455afe2f` | `build_plan_e_full_data_inventory.py` |
| `results/tables/plan_e_full_data_inventory_manifest_summary.csv` | Plan E table | `45a3195cb0272a44221a2763c8360c35670d64b5cf25518fb5151cff0a77faaf` | `build_plan_e_full_data_inventory.py` |
| `results/tables/plan_e_full_data_inventory_metric_summary.csv` | Plan E table | `67ca3a7bac0929149816ccf12910af22cd59597dab0abb6336c096a3415fca6f` | `build_plan_e_full_data_inventory.py` |
| `results/tables/plan_e_full_data_inventory_raw_output_hygiene.csv` | Plan E table | `8e1cf12d7fe38bd61765cc5205df2f97859af065bb0140b17952a6388524c4f0` | `build_plan_e_full_data_inventory.py` |
| `results/tables/plan_e_full_data_inventory_scaleup_readiness.csv` | Plan E table | `8c1097b2f403ad5d8a4127651e24f0ad132f0cc7ef982208995041c1972e58d8` | `build_plan_e_full_data_inventory.py` |

Tutti e dieci sono tracked, presenti in `HEAD`, e il blob raw `HEAD:<path>` coincide byte-for-byte con la baseline F2. Il checkout normale non e stato usato come evidenza hash perche applica filtri/line ending; la verifica autoritativa e stata fatta con `git cat-file blob HEAD:<path>`.

## 3. Diff and hash analysis

Le differenze erano sostanziali, non solo timestamp. I report e gli inventari erano stati rigenerati includendo artefatti F6D/LastFM e stato untracked successivo:

- BM25 source inventory/report: aggiunta di sorgenti LastFM F6C/F6D.
- Full data inventory: conteggi file dataset cambiati, manifest F6D rilevati, raw-output hygiene passata da `passed,0` a `review,70`.
- Scaleup readiness LastFM: `missing` cambiato in `nonpaper_candidate_manifest_available`.

Il diff sui dieci file era `1522 insertions`, `32 deletions`. Dopo il ripristino, `git diff --name-only -- <10 paths>` e vuoto e tutti gli hash coincidono con F2.

## 4. Writer attribution

Writer certi/probabili:

- `tests/test_plan_e_baseline_alignment.py::test_build_script_does_not_create_raw_jsonl_or_modify_plan_c_files`
  - lanciava `subprocess.run([sys.executable, RUNNER_PATH])`;
  - runner: `src/experiments/build_plan_e_baseline_alignment.py`;
  - writer: `write_baseline_alignment_outputs()`;
  - output canonico: `PlanEBaselineAlignmentPaths()`.

- `tests/test_plan_e_bm25_scoring.py::test_build_script_does_not_create_raw_jsonl_or_modify_plan_c_files`
  - lanciava `subprocess.run([sys.executable, RUNNER_PATH])`;
  - runner: `src/experiments/build_plan_e_bm25_scoring.py`;
  - writer: `write_bm25_scoring_outputs()`;
  - output canonico: `PlanEBM25ScoringPaths()`.

- `tests/test_plan_e_full_data_inventory.py::test_build_script_does_not_create_raw_jsonl_or_modify_plan_c_files`
  - lanciava `subprocess.run([sys.executable, RUNNER_PATH])`;
  - runner: `src/experiments/build_plan_e_full_data_inventory.py`;
  - writer: `write_inventory_outputs()`;
  - output canonico: `PlanEInventoryPaths()`.

Le funzioni writer usano `Path.write_text()` per report e `DataFrame.to_csv()` per CSV. La scrittura avveniva durante il corpo del test, non in import-time.

## 5. Root cause

I tre test erano concepiti come read-only smoke test dei builder, ma verificavano solo che non venissero creati raw JSONL e che non fossero modificati file Plan C. Non controllavano gli input protetti F2. Eseguendo i builder nella root, rigeneravano output Plan E canonici in base allo stato locale aggiornato del repository.

## 6. Code correction

Correzione applicata senza modificare i file F2 protetti:

- aggiunto `pytest.ini`;
- aggiunto `src/testing/plan_e_protected_output_guard.py`;
- aggiunto `tests/test_finalization_f6d4a_protected_input_guard.py`.

Il plugin pytest intercetta solo i tre `subprocess.run()` dei builder Plan E noti quando sono chiamati durante pytest, li esegue in-process e redirige gli output verso una directory temporanea fuori dal repository. Le esecuzioni CLI manuali dei builder Plan E restano invariate.

## 7. Tests added or modified

Nuovo test: `tests/test_finalization_f6d4a_protected_input_guard.py`.

Coperture:

- i tre builder Plan E storicamente responsabili vengono rediretti a output root temporanei durante pytest;
- i dieci file F2 protetti conservano gli stessi hash prima/dopo;
- il guard non invoca rete, API, Ollama, training o inference;
- il finalizer F6D4A resta offline tramite i test gia aggiunti in `tests/test_finalization_f6d4a_lastfm_llm_ranking.py`.

## 8. Safe baseline restoration

Ripristino eseguito solo sui dieci file F2 contaminati, dopo verifica che:

- ogni path era nella lista dei dieci;
- nessuno era `README.md`, `report/dataset_audit.md` o `report/notes.md`;
- la baseline F2 era nota;
- il blob raw `HEAD:<path>` coincideva con l'hash baseline F2.

Ogni file e stato scritto dai byte raw del blob Git e verificato immediatamente con SHA-256.

## 9. Intermediate protected-input verification

Dopo il ripristino:

- F2: `1144 checked`, `0 changed`, `0 missing`.
- F6D4A inputs: `21 checked`, `0 changed`.
- Raw F6D4A: `2 checked`, `0 changed`.
- Manifest F6D3: `1 checked`, `0 changed`.

Lo stesso gate e rimasto pulito dopo i test Plan E interessati e dopo la suite completa.

## 10. Targeted test results

- Protected-input guard: `1 passed`, durata `201.80s`.
- Plan E writer regressions: `40 passed`, durata `437.31s`.
- F6D4A specifici: `18 passed`, durata `1.83s`.
- Regressioni LastFM F6C/F6D1/F6D2/F6D3: `46 passed`, durata `239.76s`.
- Regressioni LLM/evaluator: `60 passed`, durata `1.77s`.

Dettaglio: `results/tables/finalization_f6d4a_closure_test_results.csv`.

## 11. Full-suite result

Suite completa:

- comando: `.venv\Scripts\python.exe -m pytest tests -q -o cache_dir=C:\tmp\irp_f6d4a_pytest_cache --basetemp C:\tmp\irp_f6d4a_pytest_tmp`;
- risultato: `1364 passed`, `2 skipped`, `6 warnings`;
- durata: `1736.95s`.

Le warning sono quelle gia note di `torch.nn.modules.transformer` nei test Plan E SASRec.

## 12. F6D4A offline determinism

Due esecuzioni:

`python -m src.experiments.run_finalization_f6d4a_lastfm_llm_ranking --analysis-only`

Risultato:

- exit code run 1: `0`;
- exit code run 2: `0`;
- `new_requests_sent=0` in entrambe;
- nessuna chiamata Ollama/HTTP;
- output derivati controllati: `19`;
- hash diversi tra run 1 e run 2: `0`;
- decisione: `deterministic_hash_match`.

## 13. Raw-response invariance

- Qwen2 raw: `1384a47058ed8dd8c06fc52d6bc425020b1746b60ad11e4bde6153f948a493b4`.
- Mistral raw: `7031f37d40092b4f318b5e03273772531ffda37cef0c38bfed32edfef935736a`.
- F6D3 ranking manifest: `bdf546912f06670380d9ccba1dfb4e2af5e22151834da47cd0eda85e09009f18`.

Nessun retry, nessuna nuova inferenza, nessuna modifica ai JSONL.

## 14. Final protected-input gate

- F2 protected inputs: `1144 checked`, `1144 unchanged`, `0 changed`, `0 missing`, `0 unexpected`.
- F6D4A inputs: `21 checked`, `21 unchanged`, `0 changed`.
- F6D4A raw: `2 checked`, `2 unchanged`, `0 changed`.
- F6D3 manifest: `1 checked`, `1 unchanged`, `0 changed`.

Dettaglio: `results/tables/finalization_f6d4a_closure_protected_input_verification.csv`.

## 15. Closure artifact updates

Aggiornati/verificati:

- `report/finalization_f6d4a_closure.md`;
- `results/tables/finalization_f6d4a_closure_raw_freeze.csv`;
- `results/tables/finalization_f6d4a_closure_non_strict_valid_audit.csv`;
- `results/tables/finalization_f6d4a_closure_hash_manifest.csv`;
- `results/tables/finalization_f6d4a_closure_test_results.csv`;
- `results/tables/finalization_f6d4a_closure_protected_input_verification.csv`.

## 16. Corrected scientific classification

F6D4A e scientificamente completa per la closure formale senza nuova inferenza:

- Qwen2: `1859/1859`;
- Mistral: `1859/1859`;
- Qwen2 strict-valid: `1859/1859`;
- Mistral strict-valid: `1857/1859`;
- due record Mistral non strict-valid restano evidence all-request e non sono promossi a strict-valid.

I due record Mistral non strict-valid:

- `F6D3-REQ-RANK-50a9669640c3`: indice `-13`, `unknown_indices`;
- `F6D3-REQ-RANK-bad258ab8b57`: sei indici invece di cinque, `invalid_length`, extra selection `1`.

## 17. Final decision

`final_f6d4a_closure_complete`.

Tutti i gate richiesti passano: test, determinismo, offline mode, raw invariance, F6D4A input invariance, manifest invariance e F2 protected-input invariance.

## 18. Final git state

Dirty tracked residui intenzionalmente non toccati:

- `README.md`;
- `report/dataset_audit.md`;
- `report/notes.md`.

I dieci file F2 ripristinati non sono piu dirty. Nessun file staged.

## 19. Files created or modified in this phase

Creati:

- `pytest.ini`;
- `src/testing/plan_e_protected_output_guard.py`;
- `tests/test_finalization_f6d4a_protected_input_guard.py`.

Aggiornati:

- `report/finalization_f6d4a_closure.md`;
- `results/tables/finalization_f6d4a_closure_raw_freeze.csv`;
- `results/tables/finalization_f6d4a_closure_hash_manifest.csv`;
- `results/tables/finalization_f6d4a_closure_test_results.csv`;
- `results/tables/finalization_f6d4a_closure_protected_input_verification.csv`.

Verificato invariato semanticamente:

- `results/tables/finalization_f6d4a_closure_non_strict_valid_audit.csv`.

## 20. Remaining limitations

- Il guardrail e mirato ai tre writer identificati; non sostituisce una sandbox filesystem generale per ogni futuro test.
- I builder Plan E, se lanciati manualmente fuori pytest, restano builder materializzanti e possono aggiornare i propri output canonici.
- La closure non avvia F6E1 e non introduce nuova inferenza.

## 21. Safe next action

Congelare F6D4A con commit dedicato dopo review manuale dello stato Git, separando le tre modifiche tracked preesistenti dai nuovi artefatti di guard/closure.
