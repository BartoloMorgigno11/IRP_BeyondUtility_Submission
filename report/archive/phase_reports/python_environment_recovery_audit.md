# Python Environment Recovery Audit

## 1. Executive diagnosis

Decision: `repaired_clone_validated`.

The existing `.venv` was broken because `pyvenv.cfg` pointed to a Microsoft Store/App Execution Alias base interpreter under `WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0`, which exists as an alias/path but cannot launch in this session. The `.venv` launcher itself was not corrupt: its SHA-256 matched the working repaired-clone launcher.

A separate repaired clone was created at `.venv_audit_recovery` by copying `.venv` and changing only the clone's `pyvenv.cfg` to use the local CPython 3.12.10 base at `results/environment/cpython-3.12.10-amd64/python.exe`.

## 2. Initial Git state

- Branch: `plan-d-extensions`
- HEAD: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`
- Initial status command returned `0`.
- Worktree was dirty before this recovery phase and remained dirty by design.

## 3. Exact interpreter error

The broken `.venv` command failed earlier with an inability to create a process using the WindowsApps Store Python path. Direct checks in this phase confirmed the user-local WindowsApps target is a zero-byte alias and the Program Files WindowsApps package executable is not launchable from this session.

## 4. Broken `.venv` structure

- `.venv/pyvenv.cfg` records Python `3.12.10` and `include-system-site-packages = false`.
- Static package metadata entries: `32`.
- Compiled package evidence is present via wheel tags and RECORD entries, including `cp312-cp312-win_amd64` packages.
- Critical packages in the broken environment include NumPy, pandas, PyYAML, pytest, matplotlib, and torch.

## 5. Launcher hashes

| Path | SHA-256 |
| --- | --- |
| `.venv/Scripts/python.exe` | `0B471133E110CFB53A061CAD528CE8E517D7B9AC41A0A396C39AD795A487FC14` |
| `.venv_audit_recovery/Scripts/python.exe` | `0B471133E110CFB53A061CAD528CE8E517D7B9AC41A0A396C39AD795A487FC14` |
| `results/environment/cpython-3.12.10-amd64/python.exe` | `4D6F5F81A4BCA11191C4C7C6B43632694D0A4CE74E068619D8FDC161D469859A` |

## 6. Static installed-package inventory

Static inventory was written to `results/environment/broken_venv_static_package_inventory.csv`. Important recovered versions:

- `numpy==2.5.0`
- `pandas==3.0.3`
- `PyYAML==6.0.3`
- `pytest==9.1.1`
- `matplotlib==3.11.0`
- `torch==2.6.0+cu124`

## 7. Repository dependency contract

Dependency contract rows were written to `results/environment/project_dependency_contract.csv`. The local project specs are broad (`requirements.txt`) and `environment.yml` specifies `python=3.10`, while the broken environment is Python 3.12.10. The official repository requirements are older and not equivalent to the local recovery environment.

## 8. Local Python candidates

Candidate inventory was written to `results/environment/python_environment_candidate_inventory.csv`. Summary:

- Original Store/App base: present as alias/path but not launchable, classified `unavailable`.
- Local CPython 3.12.10 under `results/environment/cpython-3.12.10-amd64`: launchable, classified compatible base.
- `.venv_f6g2_py312`: launchable but smaller dependency set, evidence only.
- Anaconda Python 3.12.4: launchable but not exact 3.12.10.
- `.venv_audit_recovery`: launchable repaired clone.

## 9. Recovery option selected

Selected option: repaired clone. The original `.venv` was preserved untouched. `.venv_audit_recovery` was created as a copy, and only `.venv_audit_recovery/pyvenv.cfg` was modified.

## 10. Validation results

Validation rows were written to `results/environment/recovered_environment_validation.csv`.

- Interpreter launch: `pass`
- Critical imports: `pass`
- Torch/CUDA visibility: `pass`
- Package inventory comparison: `pass`

Torch validation only imported torch and queried `torch.cuda.is_available()`; no training was run.

## 11. Environment comparison

Comparison rows were written to `results/environment/environment_recovery_comparison.csv`.

- Broken static packages: `32`
- Recovered importlib packages: `32`
- Name/version differences: `0`
- Critical differences: `0`

Environment-equivalence verdict: `defensible_repaired_clone_exact_package_versions`.

## 12. Scientific defensibility

Scientifically defensible for the future forensic audit: `true`.

This verdict means the repaired clone preserves the original `.venv` package tree and validated critical imports under the same Python 3.12.10 ABI. It does not prove that exact original wheel sources are fully recoverable.

## 13. Proposed interpreter for future audit

`C:\Users\Bartolo\Desktop\PROGETTI\IRP_BeyondUtility_Project\.venv_audit_recovery\Scripts\python.exe`

This path still requires explicit approval before changing the forensic-audit interpreter contract.

## 14. Commands executed

- `git status --short --branch`
- `git rev-parse HEAD`
- `PowerShell environment and Python discovery commands`
- `Copy-Item -LiteralPath .\.venv -Destination .\.venv_audit_recovery -Recurse -Force:$false`
- `apply_patch update .venv_audit_recovery/pyvenv.cfg`
- `.\.venv_audit_recovery\Scripts\python.exe validation commands with PYTHONNOUSERSITE=1`
- `.\.venv_audit_recovery\Scripts\python.exe generated recovery artifacts`

## 15. Files created or modified

Created artifacts:

- `report/python_environment_recovery_audit.md`
- `results/environment/python_environment_candidate_inventory.csv`
- `results/environment/broken_venv_static_package_inventory.csv`
- `results/environment/project_dependency_contract.csv`
- `results/environment/environment_recovery_comparison.csv`
- `results/manifests/python_environment_recovery_snapshot.json`
- `results/environment/recovered_environment_package_inventory.csv`
- `results/environment/recovered_environment_validation.csv`
- `.venv_audit_recovery/`

Modified files:

- `.venv_audit_recovery/pyvenv.cfg` only.

No existing `.venv` file was modified.

## 16. Stop conditions and exclusions

No package installation, internet access, dataset/model download, global install, Git config change, training, LLM inference, Ollama invocation, project test suite, staging, or commit was performed.

## 17. Final Git state

Final Git state after recovery artifact creation:

- Branch remains `plan-d-extensions`.
- HEAD remains `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`.
- `git status --porcelain=v1` row count after recovery: `1134`.
- The repository remains dirty, as it was before recovery.
- Expected new untracked recovery entries include `.venv_audit_recovery/`, `report/python_environment_recovery_audit.md`, and the new recovery CSV/JSON files under `results/environment/` and `results/manifests/`.

This phase did not stage or commit anything.
