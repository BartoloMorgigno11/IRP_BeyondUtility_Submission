# Canonical Item-Title Map Audit

## 1. Scope and provenance

- Audit timestamp: 2026-07-10T20:57:50+02:00
- Phase: 2B canonical item-title map reconciliation
- Candidate-set input: `results/tables/ranking_candidate_sets.csv`
- Python executable: `C:\ProgramData\anaconda3\python.exe` with `PYTHONNOUSERSITE=1`
- Network/API/model/training activity: none

## 2. Executive summary

- Built `results/tables/item_title_map.csv` with 812916 dataset-item rows.
- Built `results/tables/item_title_mapping_issues.csv` with 99 aggregate issue rows.
- Built `results/tables/item_title_coverage_summary.csv` with 14 candidate-set summaries.
- ML-1M and LastFM title sources are complete in the verified catalogs.
- Beauty and Sports still have candidate rows with at least one missing title.
- No missing title was filled with an item ID or generated placeholder.

## 3. Input candidate-set verification

- Row count: 76121
- Candidate-count distribution: `{20: 6080, 100: 70041}`
- Row validation status distribution: `{'valid': 60551, 'valid_with_warnings': 15570}`
- Candidate item lists were JSON-decodable and candidate counts matched decoded list lengths.
- The Phase 2A candidate-set file was read only and was not regenerated.

## 4. Title-source inventory

### Amazon Beauty
- `data/reconstructed/plan_c/amazon_beauty/catalog.csv` | origin=reconstructed | id_field=item_id | title_field=title | namespace=amazon_asin_string | priority=10 | records=259204 | usable_titles=258760 | used=true | Current canonical pipeline catalog has direct ASIN item IDs.
- `data/external/plan_c/meta_Beauty.json.gz` | origin=raw | id_field=asin | title_field=title | namespace=amazon_asin_string | priority=30 | records=0 | usable_titles=0 | used=gap_check_only | Inspected as raw metadata; loaded separately only for unresolved candidate ASIN checks to avoid duplicating the reconstructed catalog.
- `data/external/plan_c/meta_Beauty.json.gz` | origin=raw | id_field=asin | title_field=title | namespace=amazon_asin_string | priority=30 | records=7 | usable_titles=0 | used=true | Targeted raw metadata check for 7 unresolved candidate ASINs; archive was streamed read-only and not extracted.
- `original_repo/EvaLLMasRecommender/sota_results/beauty_his10.zip` | origin=upstream_archive | id_field=not_used | title_field=not_used | namespace=unverified_prompt_or_score_payload | priority=not_used | records=0 | usable_titles=0 | used=false | contains recommender output IDs or scores but no title metadata

### Amazon Sports
- `data/reconstructed/plan_c/amazon_sports/catalog.csv` | origin=reconstructed | id_field=item_id | title_field=title | namespace=amazon_asin_string | priority=10 | records=532197 | usable_titles=529895 | used=true | Current canonical pipeline catalog has direct ASIN item IDs.
- `data/external/plan_c/meta_Sports_and_Outdoors.json.gz` | origin=raw | id_field=asin | title_field=title | namespace=amazon_asin_string | priority=30 | records=0 | usable_titles=0 | used=gap_check_only | Inspected as raw metadata; loaded separately only for unresolved candidate ASIN checks to avoid duplicating the reconstructed catalog.
- `data/external/plan_c/meta_Sports_and_Outdoors.json.gz` | origin=raw | id_field=asin | title_field=title | namespace=amazon_asin_string | priority=30 | records=91 | usable_titles=0 | used=true | Targeted raw metadata check for 91 unresolved candidate ASINs; archive was streamed read-only and not extracted.

### LastFM
- `data/reconstructed/plan_c/lastfm/catalog.csv` | origin=reconstructed | id_field=item_id | title_field=title | namespace=lastfm_artist_id_string | priority=10 | records=17632 | usable_titles=17632 | used=true | LastFM has catalog titles but no canonical candidate rows in Phase 2A.
- `data/external/plan_c/hetrec2011-lastfm-2k.zip::artists.dat` | origin=raw_archive | id_field=id | title_field=name | namespace=lastfm_artist_id_string | priority=20 | records=17632 | usable_titles=17632 | used=true | Read-only ZIP member streaming; archive was not extracted.

### ML-1M
- `data/processed/ml1m/catalog.csv` | origin=processed | id_field=item_id | title_field=title | namespace=movie_id_string | priority=10 | records=3883 | usable_titles=3883 | used=true | Plan B processed catalog directly matches ML-1M candidate item IDs.
- `data/reconstructed/plan_c/ml1m/catalog.csv` | origin=reconstructed | id_field=item_id | title_field=title | namespace=movie_id_string | priority=20 | records=3883 | usable_titles=3883 | used=true | Plan C catalog is secondary to the processed ML-1M catalog for Phase 2B.
- `data/raw/ml1m/movies.dat` | origin=raw | id_field=movie_id | title_field=movie_title | namespace=movie_id_string | priority=30 | records=3883 | usable_titles=3883 | used=true | Raw manually provided ML-1M file confirms processed titles.
- `data/external/plan_c/ml-1m.zip::ml-1m/movies.dat` | origin=raw_archive | id_field=movie_id | title_field=movie_title | namespace=movie_id_string | priority=40 | records=3883 | usable_titles=3883 | used=true | Read-only ZIP member streaming; archive was not extracted.

### multiple
- `original_repo/EvaLLMasRecommender/llm_results/beauty.zip` | origin=upstream_archive | id_field=not_used | title_field=not_used | namespace=unverified_prompt_or_score_payload | priority=not_used | records=0 | usable_titles=0 | used=false | contains prompt/result text with item titles but no authoritative candidate item_id title map
- `original_repo/EvaLLMasRecommender/llm_results/sports.zip` | origin=upstream_archive | id_field=not_used | title_field=not_used | namespace=unverified_prompt_or_score_payload | priority=not_used | records=0 | usable_titles=0 | used=false | contains prompt/result text with item titles but no authoritative candidate item_id title map

## 5. Identifier namespaces and mapping chains

- Amazon Beauty and Amazon Sports candidate items use raw ASIN strings. Reconstructed `datamaps.json` files verify ASIN to internal integer IDs but no internal-ID join is required for Phase 2B.
- ML-1M candidate items use MovieLens movie IDs as strings. Processed and reconstructed catalogs join directly on `item_id`; raw `movies.dat` confirms the same namespace.
- LastFM catalog items use artist IDs as strings. Phase 2A has no LastFM candidate-set rows, so LastFM is catalog-only in this audit.
- Upstream LLM prompt archives and SOTA result archives were inspected as provenance but are not authoritative title maps because their title payloads are not a verified item-ID catalog.

## 6. Reconciliation methodology

- Item IDs are preserved as strings with leading zeros intact.
- Title cleanup only trims whitespace, collapses repeated whitespace, decodes simple JSON string values, and treats explicit null-like tokens as missing.
- Titles are selected by deterministic dataset-specific source priority.
- Genuine title conflicts are retained in `alternate_titles` as JSON arrays and recorded in the issue table.
- Unresolved titles remain empty in `canonical_title`.

## 7. Dataset-specific source priorities

- Amazon Beauty: reconstructed Plan C catalog first; raw staged Amazon metadata second.
- Amazon Sports: reconstructed Plan C catalog first; raw staged Amazon metadata second.
- ML-1M: processed Plan B catalog first; reconstructed Plan C catalog second; raw `movies.dat` third; staged Plan C ZIP member fourth.
- LastFM: reconstructed Plan C catalog first; staged HetRec ZIP `artists.dat` second.

## 8. Canonical-map statistics

| dataset | rows | candidate_items | resolved | missing | conflicted |
| --- | --- | --- | --- | --- | --- |
| Amazon Beauty | 259204 | 12101 | 258760 | 444 | 0 |
| Amazon Sports | 532197 | 18357 | 529895 | 2302 | 0 |
| LastFM | 17632 | 0 | 17632 | 0 | 1 |
| ML-1M | 3883 | 3883 | 3883 | 0 | 0 |

## 9. Candidate-set title coverage

| dataset | candidate_sets | rows | rows_with_missing_titles | occurrence_coverage |
| --- | --- | --- | --- | --- |
| Amazon Beauty | 3.0 | 24363.0 | 1251.0 | 0.999440 |
| Amazon Sports | 3.0 | 37598.0 | 14319.0 | 0.994943 |
| ML-1M | 8.0 | 14160.0 | 0.0 | 1.000000 |

## 10. Positive-item title coverage

| dataset | positive_items | titled_positive_items | positive_title_coverage |
| --- | --- | --- | --- |
| Amazon Beauty | 9748.0 | 9741.0 | 0.999282 |
| Amazon Sports | 14472.0 | 14410.0 | 0.995716 |
| ML-1M | 5183.0 | 5183.0 | 1.000000 |

## 11. Missing-title findings

- Amazon Beauty: 7 missing-title or missing-source issue rows.
- Amazon Sports: 91 missing-title or missing-source issue rows.
- LastFM: 0 missing-title or missing-source issue rows.

## 12. Conflicting-title findings

- Aggregate conflicting-title issue rows: 1.
- Conflicts do not block map generation because the selected title follows documented source priority and alternates are retained.

## 13. Beauty findings

- Canonical map rows: 259204.
- Candidate-set item rows in map: 12101.
- Resolved title rows: 258760.
- Missing or unverifiable title rows: 444.
- Rows with missing titles in candidate sets: 1251.
- Candidate occurrence title coverage: 0.999440.
- Positive item title coverage: 0.999282.
- Issue counts: `{'missing_title': 7}`.

## 14. Sports findings

- Canonical map rows: 532197.
- Candidate-set item rows in map: 18357.
- Resolved title rows: 529895.
- Missing or unverifiable title rows: 2302.
- Rows with missing titles in candidate sets: 14319.
- Candidate occurrence title coverage: 0.994943.
- Positive item title coverage: 0.995716.
- Issue counts: `{'missing_title': 91}`.

## 15. ML-1M findings

- Canonical map rows: 3883.
- Candidate-set item rows in map: 3883.
- Resolved title rows: 3883.
- Missing or unverifiable title rows: 0.
- Rows with missing titles in candidate sets: 0.
- Candidate occurrence title coverage: 1.000000.
- Positive item title coverage: 1.000000.
- Issue counts: `{}`.

## 16. LastFM findings

- Canonical map rows: 17632.
- Candidate-set item rows in map: 0.
- Resolved title rows: 17632.
- Missing or unverifiable title rows: 0.
- Rows with missing titles in candidate sets: 0.
- Candidate occurrence title coverage: 0.000000.
- Positive item title coverage: 0.000000.
- Issue counts: `{'conflicting_titles': 1}`.
- LastFM has catalog title coverage but no Phase 2A canonical candidate-set rows.

## 17. Plan C / Plan D / Plan E consistency

- Plan C reconstructed 100-candidate pools join directly to reconstructed catalogs for Beauty, Sports, and ML-1M.
- Plan D ML-1M Qwen2 diagnostic candidate manifests use the same ML-1M movie-ID namespace.
- Plan E 1000-user 20-candidate manifests use the same dataset namespaces as their Plan C/processed sources.
- Existing Plan C/D/E outputs were read as sources or candidate-set provenance only and were not modified.

## 18. Remaining blocking issues

- Candidate rows with missing Beauty or Sports titles remain blocked because matching reconstructed catalog records and targeted raw metadata records have empty or missing title values.
- No unresolved title is safe to fill from descriptions, categories, item IDs, prompt text, or LLM inference.
- Upstream prompt/result archives are not authoritative item-title maps for future prompt construction.

## 19. Canonical title-map path for future experiments

- Use `results/tables/item_title_map.csv` as the authoritative Phase 2B title source.
- Future prompt construction should join on `(dataset, item_id)` and must handle empty `canonical_title` as unresolved.

## 20. Artifacts that should not be used as authoritative title maps

- `original_repo/EvaLLMasRecommender/llm_results/*.zip`: prompt/result payloads, not verified item-title catalogs.
- `original_repo/EvaLLMasRecommender/sota_results/beauty_his10.zip`: recommender outputs, not title metadata.
- `normalized_text`, `description`, `categories`, `brand`, and item IDs: not title fields for unresolved titles.

## 21. Test and validation commands

- `$env:PYTHONNOUSERSITE='1'; & 'C:\ProgramData\anaconda3\python.exe' --version`
- `$env:PYTHONNOUSERSITE='1'; & 'C:\ProgramData\anaconda3\python.exe' src/experiments/build_item_title_map_audit.py`
- `$env:PYTHONNOUSERSITE='1'; & 'C:\ProgramData\anaconda3\python.exe' -m pytest tests/test_item_title_map_reconciliation.py tests/test_dataset_candidate_set_validation.py`
- validated 812916 item_title_map rows
- validated 99 item_title_mapping_issues rows
- validated 14 item_title_coverage_summary rows
- all 34341 unique candidate items have title-map rows

## 22. Recommended single next task

Resolve remaining Beauty and Sports candidate-title gaps by auditing whether already-staged raw metadata contains missing ASINs under another verified repository-local source or mapping chain, without using web access or generated titles.
