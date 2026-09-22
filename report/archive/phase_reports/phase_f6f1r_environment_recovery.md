# Phase F6F1R Environment Recovery

## 1. Initial repository state
Branch `plan-d-extensions`, HEAD `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`, staged files `none`. Pre-existing tracked dirty files remain `README.md, report/dataset_audit.md, report/notes.md`.

## 2. Broken `.venv` diagnosis
Classification `windowsapps_alias_resolution`. `.venv` was inspected only and not modified.

## 3. Python interpreter inventory
Rows written: `8`.

## 4. F6F1 interpreter identity
F6F1 manifest evidence records WindowsApps Store Python 3.12.10; exact prior process path is reconstructed from local evidence, not invented.

## 5. Dependency-contract inventory
Rows written: `6` from local dependency files and import probes.

## 6. Selected stable base interpreter
C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\python3.12.exe

## 7. Parallel environment creation
`.venv_repro` was attempted first; fallback `.venv_repro_system_site` was used only if isolated imports were unavailable.

## 8. Environment classification
reconstructed_venv_with_system_site_packages

## 9. Environment safety gates
Rows written: `15`.

## 10. Frozen-input verification
Rows written: `60`; changed/missing counts are in the table and manifest.

## 11. Targeted tests
Rows written: `12`.

## 12. Excluded unsafe tests
inference, ollama, network, cuda, training, downloads, full_suite

## 13. Isolated F6F1 rebuild method
F6F1R-specific wrapper calls existing F6F1 pure analysis with in-memory output path redirection under `results/environment/phase_f6f1r_isolated_f6f1`.

## 14. Cross-environment artifact comparison
Rows written: `14`.

## 15. Scientific hash comparison
Official `4bf9391b835dad47c80f41b9d41b130f64ac3f4776561cc19bf0a3d044ffb37c`, isolated `4bf9391b835dad47c80f41b9d41b130f64ac3f4776561cc19bf0a3d044ffb37c`.

## 16. Runtime metadata differences
Manifest/report runtime metadata is separated from scientific content; CSV outputs require byte identity.

## 17. Pip freeze and dependency provenance
`results/environment/phase_f6f1r_pip_freeze.txt` and `phase_f6f1r_pip_list.json` are written.

## 18. Determinism
F6F1R writes deterministic sorted CSV/JSON artifacts; runtime metadata is reported in the manifest.

## 19. Protected-input verification
Protected and direct frozen inputs are included in `results\tables\phase_f6f1r_input_integrity.csv`.

## 20. Environment freeze manifest
`results\manifests\phase_f6f1r_environment_freeze.json`.

## 21. Missing or blocked evidence
Rows written: `1`.

## 22. Final decision
f6f1r_environment_recovery_complete_with_runtime_metadata_differences

## 23. Recommended next action
keep_existing_environment_state_frozen

## 24. Final git state
No git add, commit, reset, clean, checkout, or frozen output overwrite is performed.

## 25. Files created or modified
Only new F6F1R source/test/report/table/environment/manifest files plus the new parallel venv directory.

## 26. Remaining limitations
If the selected environment uses system-site packages, it is not a fully isolated dependency rebuild until a local wheel/cache bundle is reconstructed.
