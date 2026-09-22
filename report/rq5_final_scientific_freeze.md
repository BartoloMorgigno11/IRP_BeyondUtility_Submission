# RQ5 final scientific package: reproduction contract

This file documents artifacts and commands, not manuscript claims. The machine-readable scientific synthesis is `results/rq5_final_scientific_freeze/scientific_synthesis.json`.

## Authority

Corrected full is the foreground matched utility experiment. R5D V2 supports structural/pipeline feasibility and absolute Profile+History utility; it has no matched no-profile counterfactual. The systematic extension is supporting evidence and reuses corrected observations. History V2 is experimentally distinct. Historical F6L statistics, old history analysis and superseded pilots must not be substituted for corrected matched evidence.

## Complete offline inputs

All selected raw/model outputs are preserved byte for byte inside versioned deterministic ZIP files under `results/rq5_final_scientific_freeze/`. `package_inventory.csv` records each logical member, family, role, original branch/status, hash, size and reproduction role. These archives require no external acquisition. Original source files and historical Windows paths inside them are forensic provenance only. Operational code never executes archived experiment scripts and never resolves paths in other worktrees. No new inference is supported by the final producer.

The systematic profile reuse audit is generated from archived profile manifests, source profiles and hashes, at both the package path and the declared delivery path `results/tables/rq5_primary_systematic_profile_reuse_audit.csv`.

## V2 metric contract

Terminal-invalid rankings contribute zero to HR, NDCG, APLT and serendipity. Missing popularity evidence raises an error; it is not converted to zero. APLT uses unique top-5 IDs belonging to the positive-count train support below the linear q80 threshold, divided by five.

MostPop sorts by descending train frequency and then ascending **exact canonical internal_item_id string**, using Unicode lexicographic ordering. For example, tied ID `10` precedes `2`; no numeric coercion or candidate-order tie break occurs. The identity is checked against the bundled canonical catalog. Serendipity is HR@5 multiplied by one minus the deterministic MostPop hit. This is a local adapted definition, not paper-exact reproduction. Historical 47/200 and audit-observed 56/200 are provenance observations, never target values.

`history_v2/migration_manifest.json` records old/new hashes, affected requests/cells and the final deterministic hit count. No raw output was changed.

## Reproduction from a clean clone

Use Python 3.12 and install `configs/rq5_freeze_requirements.txt` in a fresh environment. No API credentials, network inference, original worktrees or caches are required.

```bash
python -B src/analysis/rq5_final_freeze.py --config configs/rq5_final_freeze.yaml --mode verify
python -B src/analysis/rq5_final_freeze.py --config configs/rq5_final_freeze.yaml --mode reproduce --output-dir results/rq5_reproduction_check
python -B src/analysis/rq5_final_freeze.py --config configs/rq5_final_freeze.yaml --mode figures
python -B -m pytest tests/test_rq5_final_freeze.py -p no:cacheprovider
```

`reproduce` reparses raw rankings, verifies matched inputs and profile dependencies, recomputes metrics and paired bootstrap statistics, and byte-compares derived analytics against the freeze. It writes only to a separate output directory. `verify` checks artifact and archive member SHA256 values. Figure generation reads authoritative CSVs only, uses deterministic metadata and records actual source/image hashes. Graphical bytes may require the pinned renderer/fonts; numeric verification is separate.

Historical bootstrap algorithms and ordering are preserved for corrected/systematic utility: 10,000 paired resamples, seed 20260901, linear percentile 95% intervals. There are no p-values or retrospective multiplicity corrections. New V2 history contrasts use the same declared bootstrap contract with corrected per-user contributions.

## Freeze commit identity

`freeze_manifest.json` records the base/source commits separately. The actual commit containing its exact bytes is recorded in `commit_provenance.json` by a subsequent provenance-only commit. This avoids falsely treating the pre-freeze checkout as the freeze commit and avoids impossible self-referential hashes. `clean_room_validation.json` records validation of the tracked package export.

## Scope and limitations

No semantic preference-preservation validation exists. Profile-only substitution and profile-plus-history addition are different estimands. Effects apply only to the specified local cohorts, models, generated profiles and delivered prompts. Bootstrap signals are exploratory and unadjusted. Reused observations are not independent replications. Other research questions and manuscript files are outside this change.

To repeat the isolated tracked-export validation, run `python -B src/analysis/rq5_clean_room_check.py --destination <new-empty-location-outside-project-worktrees> --commit <freeze-commit>`. The helper exports only committed manifest-listed paths, copies the pinned runtime dependency closure without bytecode caches, blocks original-worktree reads and socket connections, and checks analytic/figure reproduction plus RQ5 tests.
