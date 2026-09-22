# GitHub Readiness Cleanup Audit

Generated: `2026-08-14T09:55:07.338370+00:00`

## Baseline

- Branch: `plan-d-extensions`
- Cleanup base HEAD at audit generation: `62ab50924ab36c65b8fd8e03dfffb0a11267c3d1`
- Frozen reproduction commit: `db22d3c9f92f258ca3264cb588ba4ed1c5f918b7`
- Decision token: `f6m_reproduction_closure_complete_with_documented_limitations`
- Status entries before cleanup classification: `0`
- Cleanup audit self-refresh paths excluded from status baseline: `3`
- Ignored entries included from collapsed Git status: `1514`
- Inventory rows: `1514`
- Inventory size represented: `17785359864` bytes
- `original_repo/` dirty in publishable status: `no`

## Category Counts

| category | rows |
| --- | ---: |
| `E_HISTORICAL_AUDIT_EVIDENCE` | 1324 |
| `G_DUPLICATE` | 3 |
| `H_GENERATED_REBUILDABLE` | 43 |
| `I_LOCAL_ONLY` | 124 |
| `J_LARGE_EXTERNAL_DATA` | 20 |

## Recommended Action Counts

| recommended_action | rows |
| --- | ---: |
| `delete_after_approval` | 2 |
| `ignore_or_archive_by_reference` | 1324 |
| `ignore_or_archive_if_unique` | 43 |
| `ignore_or_delete_after_local_review` | 124 |
| `ignore_or_externalize` | 20 |
| `migrate_to_report_or_results_archive_then_delete` | 1 |

## GitHub Suitability Counts

| github_suitability | rows |
| --- | ---: |
| `exclude_from_github` | 147 |
| `exclude_from_github_unless_selected` | 1367 |

## Git Status Counts

| git_status | rows |
| --- | ---: |
| `!!` | 1514 |

## Cleanup Decisions

- Keep and commit source, tests, configs, F6M foreground tables, current report entrypoints, and F6M-aligned docs.
- Archive superseded foreground reports and old Plan C plots only after hash verification.
- Ignore or externalize raw datasets, processed bulk data, raw LLM responses, predictions, model checkpoints, local environments, and runtime logs.
- Leave the transitional root archive as ignored local material unless a separate explicit recursive-deletion approval is given.
- Do not start any extension, inference, model training, download, or result repair as part of cleanup.

## Unknown Status

- Relevant unknown rows: `0`
- No relevant unknown rows remain after rule-based classification.

## Final Validation Requirements

- `git status --short` must contain only intentional staged/unstaged changes until commits are created.
- Run focused tests for modified helper/source bundles and F6M closure.
- Run secret and operational absolute-path scans before final commit.
- Confirm `original_repo/` remains unmodified.

## Machine-Readable Inventory

- `results/tables/repository_cleanup_inventory.csv`
