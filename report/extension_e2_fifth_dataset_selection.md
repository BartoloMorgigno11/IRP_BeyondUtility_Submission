# E2 Fifth Dataset Selection

POST-F6M EXTENSION DESIGN EVIDENCE

## Final Decision

EXTENSION_E2_FIFTH_DATASET_SELECTED

PRIMARY DATASET: Food.com

BACKUP DATASET: Steam reviews/interactions

Can the selected fifth dataset receive the complete applicable paper/local-reproduction evaluation BEFORE our original extensions?

YES

## Repository Precondition / Recovery State

- branch: master
- HEAD at recovery: 07e239ce1ede2807a40ea3064554febcf941e1db
- Git state at recovery: clean; master...origin/master [ahead 5]
- original_repo/EvaLLMasRecommender state: clean; main...origin/main
- Interrupted-run artifact recovery: no extension_e2_* tracked or untracked artifacts found; no files retained; no incomplete extension E2 evidence; no unrelated changes existed.

## Frozen E1 Contract Counts

- metric contract rows: 29
- four-dataset coverage rows: 26
- fifth-dataset requirements: 23
- inference-requirement rows: 8
- dataset-selection scorecard criteria: 25

## Authoritative Source Inventory

| source_id | title | url | notes |
| --- | --- | --- | --- |
| S_BOOKCROSSING_GROUPLENS | GroupLens Book-Crossing dataset mirror | https://grouplens.org/datasets/book-crossing/ | Institutional mirror of Cai-Nicolas Ziegler's Book-Crossing data with user, book and rating tables. |
| S_BOOKCROSSING_MLDATA | Mldata Book-Crossing schema/summary mirror | https://www.mldata.io/dataset-details/book_crossing/ | Trusted dataset catalog summary listing 278858 users, 271379 books and 1149780 ratings. |
| S_FOOD_BERD | BERD Food.com Recipes and Interactions catalog record | https://berd-platform.de/records/dmajj-zp851 | Institutional record for the Food.com release; lists open dataset status, 180K+ recipes, 700K+ reviews, 18-year span, CSV tables and variables including RAW_interactions date/rating/review and RAW_recipes name/tags/ingredients. |
| S_FOOD_MCAULEY | UCSD Food.com Recipes and Interactions | https://cseweb.ucsd.edu/~jmcauley/datasets.html#foodcom | Author-maintained academic release for Food.com recipes and interactions; links to download and citation. |
| S_GOODREADS_UCSD | UCSD Book Graph / Goodreads official release | https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/UCSD%20Book%20Graph.html | Official UCSD release with book, interaction, review, shelf and genre-subset files; states academic use only and no redistribution or commercial use. |
| S_GOOGLE_LOCAL_2021 | UCSD Google Local 2021 dataset | https://cseweb.ucsd.edu/~jmcauley/datasets.html#google_local | Author-maintained academic Google Local release with 666M reviews and business metadata; useful context but too broad for an unsupported restaurant-only convenience slice. |
| S_GOOGLE_RESTAURANTS_MCAULEY | UCSD Google Restaurants dataset | https://cseweb.ucsd.edu/~jmcauley/datasets.html#google_restaurants | Author-maintained academic restaurant release; states full/subset sizes and review fields, but the visible page does not explicitly establish restaurant name/title fields. |
| S_STEAM_BERD | BERD Steam dataset catalog record | https://berd-platform.de/records/t0ce0-a0708 | Institutional record for the McAuley Steam release; lists open dataset status, 1.43 GB scale, and item metadata variables including genres, tags and title. |
| S_STEAM_MCAULEY | UCSD Steam Video Game and Bundle Data | https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data | Author-maintained academic release; states 7,793,069 reviews, 2,567,538 users and 15,474 games. |
| S_YELP_DOCS | Yelp public business data documentation | https://docs.developer.yelp.com/docs/resources-categories | Yelp platform documentation confirms business categories and business entities, but it is not a substitute for Open Dataset license terms. |
| S_YELP_OPEN | Yelp Open Dataset official page | https://business.yelp.com/data/resources/open-dataset/ | Official source page exists, but the web audit could not retrieve current terms/content. Treat exact current license/registration terms as unresolved until manual access. |

## Candidate Inventory

| candidate_id | candidate_name | exact_release | access_mechanism | academic_terms | dataset_size | sequential_classification | llm_metadata_classification | calibration_classification | scale_classification | selection_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| goodreads | Goodreads | UCSD Book Graph Goodreads release; use an official genre subset only after E3 verifies size and metadata coverage. | Google Drive files from official UCSD Book Graph page. | Academic use only; no redistribution; no commercial use. | large multi-file release; official page lists book metadata, interactions, reviews, shelves and genre subsets. | SEQUENTIAL_ADAPTABLE | LLM_METADATA_STRONG | CALIBRATION_ADEQUATE | VERY_LARGE_REQUIRES_SUBSET | REJECTED_FINALIST |
| bookcrossing | BookCrossing / Book-Crossing | Cai-Nicolas Ziegler Book-Crossing dataset as mirrored by GroupLens. | downloadable zip via institutional mirrors. | public research dataset mirror; exact redistribution license not recovered in the official page during this audit. | small/moderate classic recommender dataset. | SEQUENTIAL_BLOCKING | LLM_METADATA_WEAK | CALIBRATION_BLOCKING | SMALL | REJECTED_HARD_GATE |
| yelp | Yelp Open Dataset | Yelp Open Dataset current educational/open dataset release; exact current version must be captured during manual access. | official Yelp Open Dataset web form/download, not Kaggle reuploads. | UNKNOWN in this audit because current official terms were not retrievable by web tool. | large; public summaries report about 6.99M reviews and 150K businesses, but exact current official count remains to be verified. | SEQUENTIAL_STRONG | LLM_METADATA_ADEQUATE | CALIBRATION_STRONG | LARGE_BUT_MANAGEABLE | REJECTED_ACCESS_UNRESOLVED |
| steam | Steam reviews/interactions | UCSD Steam Video Game and Bundle Data; use Version 2 review data plus item metadata from the same release. | direct academic dataset links from UCSD McAuley page; BERD catalog mirrors descriptive metadata. | open academic dataset; cite Kang and McAuley 2018 and related release references. | 7,793,069 reviews, 2,567,538 users, 15,474 games; BERD record lists about 1.43 GB. | SEQUENTIAL_STRONG | LLM_METADATA_STRONG | CALIBRATION_STRONG | LARGE_BUT_MANAGEABLE | BACKUP_DATASET |
| foodcom | Food.com | Food.com Recipes and Interactions, published 2019 Version v2; use RAW_recipes.csv, RAW_interactions.csv and official train/validation/test split files. | UCSD McAuley page links to the official dataset download; BERD records the dataset as an open dataset with external resource access. | open academic dataset record; cite Majumder et al. 2019; redistribution should remain via source links, not committed raw data. | BERD records 180K+ recipes, 700K+ reviews, and 18 years of user interactions; 0.89 GB. | SEQUENTIAL_STRONG | LLM_METADATA_STRONG | CALIBRATION_STRONG | MODERATE | PRIMARY_DATASET |
| google_restaurants | Google Restaurants / Google Local restaurant release | UCSD Google Restaurants dataset from Personalized Showcases; evaluate the official restaurant full release and official subset only, not an arbitrary Google Local category slice. | direct links from UCSD McAuley dataset page. | open academic release; cite Liu et al. 2023. | full: 65779 restaurants, 1010915 users, 1771463 reviews; official subset: 30238 restaurants, 37720 users, 108289 reviews. | SEQUENTIAL_WEAK | LLM_METADATA_WEAK | CALIBRATION_ADEQUATE | MODERATE | REJECTED_FIELD_UNRESOLVED |

## Access and Licensing Comparison

| candidate_name | academic_terms | paid_or_free | registration_or_manual_request | acquisition_reproducibility | source_ids |
| --- | --- | --- | --- | --- | --- |
| Goodreads | Academic use only; no redistribution; no commercial use. | free academic access | manual Google Drive download; some detailed data may require author contact. | reproducible with manual UCSD Google Drive intake and no redistribution. | S_GOODREADS_UCSD |
| BookCrossing / Book-Crossing | public research dataset mirror; exact redistribution license not recovered in the official page during this audit. | free | none observed on mirrors. | reproducible mirror download, but official source page availability is weaker than UCSD releases. | S_BOOKCROSSING_GROUPLENS;S_BOOKCROSSING_MLDATA |
| Yelp Open Dataset | UNKNOWN in this audit because current official terms were not retrievable by web tool. | free if accepted through official access flow; no paid requirement established. | manual form/terms acceptance likely required; exact requirement UNKNOWN. | blocked from final selection until current terms and exact release are captured. | S_YELP_OPEN;S_YELP_DOCS |
| Steam reviews/interactions | open academic dataset; cite Kang and McAuley 2018 and related release references. | free | none established for UCSD download. | reproducible UCSD academic download. | S_STEAM_MCAULEY;S_STEAM_BERD |
| Food.com | open academic dataset record; cite Majumder et al. 2019; redistribution should remain via source links, not committed raw data. | free | Kaggle/external-resource account or manual intake may be required; document in E3. | reproducible with documented external-resource/manual intake and no raw data committed. | S_FOOD_MCAULEY;S_FOOD_BERD |
| Google Restaurants / Google Local restaurant release | open academic release; cite Liu et al. 2023. | free | none established for UCSD download. | reproducible UCSD academic download; avoid unsupported restaurant-only slices from broad Google Local. | S_GOOGLE_RESTAURANTS_MCAULEY;S_GOOGLE_LOCAL_2021 |

## Scale Comparison

| candidate_name | scale_classification | raw_download_scale | official_subset_policy | compute_rationale | risk |
| --- | --- | --- | --- | --- | --- |
| Goodreads | VERY_LARGE_REQUIRES_SUBSET | VERY_LARGE_REQUIRES_SUBSET | children, comics_graphic, fantasy_paranormal, history_biography, mystery_thriller_crime, poetry, romance and young_adult official subsets. | large multi-file release; official page lists book metadata, interactions, reviews, shelves and genre subsets. | Not selected because compute/access restrictions and multi-date policy complexity are worse than Food.com for the same Role A coverage. |
| BookCrossing / Book-Crossing | SMALL | SMALL | none needed for size, none recovered as official scientific subset. | small/moderate classic recommender dataset. | Fails the frozen E1 chronological/sequential gate and lacks authoritative category/genre metadata. |
| Yelp Open Dataset | LARGE_BUT_MANAGEABLE | LARGE_BUT_MANAGEABLE | official city/category filters not treated as scientific subsets unless documented by Yelp. | large; public summaries report about 6.99M reviews and 150K businesses, but exact current official count remains to be verified. | Not selected because current official access/license/version details could not be established authoritatively in E2. |
| Steam reviews/interactions | LARGE_BUT_MANAGEABLE | LARGE_BUT_MANAGEABLE | no smaller official scientific subset selected; use full release unless E3 shows compute risk. | 7,793,069 reviews, 2,567,538 users, 15,474 games; BERD record lists about 1.43 GB. | Selected as backup; weaker than Food.com only because the review/recommendation signal requires more domain adaptation. |
| Food.com | MODERATE | MODERATE | use official train/validation/test split files; do not invent a convenience subset in E2. | BERD records 180K+ recipes, 700K+ reviews, and 18 years of user interactions; 0.89 GB. | primary manageable with no arbitrary subset |
| Google Restaurants / Google Local restaurant release | MODERATE | MODERATE | official restaurant subset provided by the release. | full: 65779 restaurants, 1010915 users, 1771463 reviews; official subset: 30238 restaurants, 37720 users, 108289 reviews. | Fails E2 certainty for item-title/name and timestamp gates; do not use broad Google Local restaurant slicing as a shortcut. |

## Temporal / Sequential Comparison

| candidate_name | timestamps | chronology | history_depth | sequential_classification |
| --- | --- | --- | --- | --- |
| Goodreads | detailed review records expose date_added, date_updated, read_at and started_at fields; chronology requires choosing and freezing one timestamp policy. | available but policy-sensitive because multiple date fields exist. | strong at full scale; subset-specific depth must be verified in E3. | SEQUENTIAL_ADAPTABLE |
| BookCrossing / Book-Crossing | NOT_SUPPORTED: canonical schema is users, books and ratings without an authoritative timestamp/order field. | sequential parity is blocked; file order must not be used as chronology. | rating histories exist but cannot be chronologically ordered. | SEQUENTIAL_BLOCKING |
| Yelp Open Dataset | review date field is a standard Open Dataset field in documented schema summaries. | review events are chronologically usable after mapping business reviews to interactions. | likely strong at full scale; must be verified after official acquisition. | SEQUENTIAL_STRONG |
| Steam reviews/interactions | review date fields are available in release previews; freeze exact field in E3. | review chronology supports leave-one-out and sequential models after date normalization. | large enough for history-length bins after filtering; exact bins to verify in E3. | SEQUENTIAL_STRONG |
| Food.com | RAW_interactions date field; official processed files also include ordered items in user histories. | strong chronological support using explicit interaction date and official split/order files. | official interaction histories and splits exist; exact post-filter depth to verify in E3. | SEQUENTIAL_STRONG |
| Google Restaurants / Google Local restaurant release | timestamps are expected from Google review data lineage, but the visible restaurant page sample does not explicitly list a timestamp field. | chronological defensibility remains UNKNOWN until raw fields are inspected in E3. | official subset may be shallow; full release likely adequate but must be verified. | SEQUENTIAL_WEAK |

## Metadata / LLM Comparison

| candidate_name | item_title_name | categories_genres_tags | additional_metadata | llm_metadata_classification | calibration_classification |
| --- | --- | --- | --- | --- | --- |
| Goodreads | book title/title_without_series in book metadata. | popular shelves and official genre subset files. | authors, publication fields, ratings counts, shelves and review text. | LLM_METADATA_STRONG | CALIBRATION_ADEQUATE |
| BookCrossing / Book-Crossing | book title field exists. | NOT_SUPPORTED authoritatively; no genre/tag/category field in the canonical schema. | ISBN, author, publication year, publisher and image URLs. | LLM_METADATA_WEAK | CALIBRATION_BLOCKING |
| Yelp Open Dataset | business names are expected in business records, but official current schema was not directly retrieved here. | business categories are central Yelp metadata. | business attributes, stars, review text, check-ins/tips/photos depending on release. | LLM_METADATA_ADEQUATE | CALIBRATION_STRONG |
| Steam reviews/interactions | title field in Steam game metadata. | genres, tags and specs in game metadata. | developer, publisher, price, release date, early access, categories/specs/tags. | LLM_METADATA_STRONG | CALIBRATION_STRONG |
| Food.com | RAW_recipes name field. | recipe tags, ingredients, techniques and contributor/submitted metadata. | minutes, nutrition, steps, description, ingredients, n_steps and n_ingredients. | LLM_METADATA_STRONG | CALIBRATION_STRONG |
| Google Restaurants / Google Local restaurant release | UNKNOWN from the visible authoritative restaurant page; item-name/title gate cannot be passed in E2. | restaurant/business metadata is present; exact category-name fields must be verified. | review text, rating, picture IDs and restaurant metadata per the official page. | LLM_METADATA_WEAK | CALIBRATION_ADEQUATE |

## Complete 29-Row Metric Compatibility Summary

| candidate_id | candidate_name | NOT_SUPPORTED | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_WITH_ADAPTATION | UNKNOWN |
| --- | --- | --- | --- | --- | --- |
| goodreads | Goodreads | 0 | 0 | 29 | 0 |
| bookcrossing | BookCrossing / Book-Crossing | 11 | 14 | 4 | 0 |
| yelp | Yelp Open Dataset | 0 | 0 | 29 | 0 |
| steam | Steam reviews/interactions | 0 | 8 | 21 | 0 |
| foodcom | Food.com | 0 | 8 | 21 | 0 |
| google_restaurants | Google Restaurants / Google Local restaurant release | 0 | 23 | 3 | 3 |

| metric_id | metric_name | goodreads | bookcrossing | yelp | steam | foodcom | google_restaurants |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hr_at_5 | HR@5 | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| ndcg_at_5 | NDCG@5 | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| mrr_at_5_printed_only | MRR printed by official script | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| self_information_at_5 | SelfInformation@5 | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| serendipity_at_5 | Serendipity@5 | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| arp_at_5 | ARP@5 | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| aplt_at_5 | APLT@5 | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| popreo_at_5 | PopREO@5 | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| item_coverage_at_5 | ItemCoverage@5 | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| oic_at_5 | Overlap Item Coverage@5 | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| shannon_entropy_at_5 | Shannon category entropy@5 | SUPPORTED_WITH_ADAPTATION | NOT_SUPPORTED | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| gini_at_5 | Gini@5 | SUPPORTED_WITH_ADAPTATION | NOT_SUPPORTED | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| dpd_at_5 | DPD@5 | SUPPORTED_WITH_ADAPTATION | NOT_SUPPORTED | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| jain_index_at_5 | Jain's Index@5 | SUPPORTED_WITH_ADAPTATION | NOT_SUPPORTED | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| hallucination_rate_at_5 | Hallucination rate@5 | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | UNKNOWN |
| canddif_hr | CandDif-HR | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION |
| canddif_ndcg | CandDif-NDCG | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION |
| history_length_sensitivity | History-length sensitivity | SUPPORTED_WITH_ADAPTATION | NOT_SUPPORTED | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION |
| generation_involved_profile_matrix | Generation-involved profile recommendation | SUPPORTED_WITH_ADAPTATION | NOT_SUPPORTED | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | UNKNOWN |
| ks_sample_representativeness | K-S sample representativeness gate | SUPPORTED_WITH_ADAPTATION | NOT_SUPPORTED | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| catalog_coverage_at_5 | Catalog Coverage@5 | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| intra_list_diversity_at_5 | Intra-List Diversity@5 | SUPPORTED_WITH_ADAPTATION | NOT_SUPPORTED | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| calibration_at_5 | Calibration@5 | SUPPORTED_WITH_ADAPTATION | NOT_SUPPORTED | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| strict_parse_validity_rate | Strict parse validity rate | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | UNKNOWN |
| duplicate_rate_at_5 | Duplicate rate@5 | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| candidate_recall_at_n | Candidate Recall@N | SUPPORTED_WITH_ADAPTATION | NOT_SUPPORTED | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| conditional_hr_ndcg_at_5 | Conditional HR/NDCG@5 | SUPPORTED_WITH_ADAPTATION | NOT_SUPPORTED | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| normalized_popularity_shift_lift | Normalized popularity, popularity shift, popularity lift | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING |
| cost_token_latency_summary | Cost/token/latency summary | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_WITH_ADAPTATION | SUPPORTED_AFTER_STANDARD_PREPROCESSING |

## Complete 23-Requirement Compatibility Summary

| candidate_id | candidate_name | FAIL | OPTIONAL | PASS | PASS_WITH_DOCUMENTED_ADAPTATION | UNKNOWN |
| --- | --- | --- | --- | --- | --- | --- |
| goodreads | Goodreads | 0 | 1 | 0 | 22 | 0 |
| bookcrossing | BookCrossing / Book-Crossing | 9 | 0 | 0 | 14 | 0 |
| yelp | Yelp Open Dataset | 0 | 1 | 0 | 21 | 1 |
| steam | Steam reviews/interactions | 0 | 1 | 16 | 6 | 0 |
| foodcom | Food.com | 0 | 1 | 14 | 8 | 0 |
| google_restaurants | Google Restaurants / Google Local restaurant release | 0 | 0 | 0 | 18 | 5 |

| requirement | mandatory | goodreads | bookcrossing | yelp | steam | foodcom | google_restaurants |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Public, reproducible dataset access and license | yes | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | UNKNOWN | PASS | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION |
| User-item interaction rows with stable user_id and item_id | yes | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS | PASS | PASS_WITH_DOCUMENTED_ADAPTATION |
| Chronological order or timestamps | yes | PASS_WITH_DOCUMENTED_ADAPTATION | FAIL | PASS_WITH_DOCUMENTED_ADAPTATION | PASS | PASS | UNKNOWN |
| Item titles or names | yes | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS | PASS | UNKNOWN |
| Item metadata categories/tags/genres | yes_if_claimed | PASS_WITH_DOCUMENTED_ADAPTATION | FAIL | PASS_WITH_DOCUMENTED_ADAPTATION | PASS | PASS | PASS_WITH_DOCUMENTED_ADAPTATION |
| Train-only item popularity counts | yes | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS | PASS | PASS_WITH_DOCUMENTED_ADAPTATION |
| Unique-user item popularity counts | yes | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS | PASS | PASS_WITH_DOCUMENTED_ADAPTATION |
| Ground-truth positive target per sampled user | yes | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS | PASS | PASS_WITH_DOCUMENTED_ADAPTATION |
| Ranking candidate pools with one positive and 19 negatives | yes | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS | PASS | PASS_WITH_DOCUMENTED_ADAPTATION |
| Reranking candidate pools from BPRMF, GRU4Rec, SASRec, and LightGCN top-K outputs | yes_for_reranking | PASS_WITH_DOCUMENTED_ADAPTATION | FAIL | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION |
| Candidate order and positive position metadata | yes | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS | PASS | PASS_WITH_DOCUMENTED_ADAPTATION |
| History length bins with sufficient depth | yes | PASS_WITH_DOCUMENTED_ADAPTATION | FAIL | PASS_WITH_DOCUMENTED_ADAPTATION | PASS | PASS | PASS_WITH_DOCUMENTED_ADAPTATION |
| Aligned MostPop recommendation/hit vector | yes | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS | PASS | PASS_WITH_DOCUMENTED_ADAPTATION |
| Popularity quintile groups with zero-denominator policy | yes | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS | PASS | PASS_WITH_DOCUMENTED_ADAPTATION |
| User activity groups based on median history length | yes | PASS_WITH_DOCUMENTED_ADAPTATION | FAIL | PASS_WITH_DOCUMENTED_ADAPTATION | PASS | PASS | PASS_WITH_DOCUMENTED_ADAPTATION |
| Catalog item universe with exposure denominator policy | yes | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS | PASS | PASS_WITH_DOCUMENTED_ADAPTATION |
| Free-text/title-list LLM outputs or paper-equivalent parser outputs | yes_for_paper_hallucination | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | UNKNOWN |
| Profile-generation outputs and dependency manifest | yes | PASS_WITH_DOCUMENTED_ADAPTATION | FAIL | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | UNKNOWN |
| Traditional model training-ready split and configs | yes_if_reproducing_traditional_matrix | PASS_WITH_DOCUMENTED_ADAPTATION | FAIL | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | UNKNOWN |
| Raw output manifests, parser provenance, and invalid-output policy | yes | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION |
| Candidate retrieval and aggregation intermediate artifacts | yes_later | PASS_WITH_DOCUMENTED_ADAPTATION | FAIL | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION |
| Positive-injection intervention records | yes_later | PASS_WITH_DOCUMENTED_ADAPTATION | FAIL | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION | PASS_WITH_DOCUMENTED_ADAPTATION |
| Token, latency, and cost logs | optional | OPTIONAL | PASS_WITH_DOCUMENTED_ADAPTATION | OPTIONAL | OPTIONAL | OPTIONAL | PASS_WITH_DOCUMENTED_ADAPTATION |

## Traditional-Model Compatibility

| model_name | goodreads | bookcrossing | yelp | steam | foodcom | google_restaurants |
| --- | --- | --- | --- | --- | --- | --- |
| Random | DIRECTLY_COMPATIBLE | DIRECTLY_COMPATIBLE | DIRECTLY_COMPATIBLE | DIRECTLY_COMPATIBLE | DIRECTLY_COMPATIBLE | DIRECTLY_COMPATIBLE |
| MostPop | COMPATIBLE_AFTER_STANDARD_MAPPING | COMPATIBLE_AFTER_STANDARD_MAPPING | COMPATIBLE_AFTER_STANDARD_MAPPING | COMPATIBLE_AFTER_STANDARD_MAPPING | COMPATIBLE_AFTER_STANDARD_MAPPING | COMPATIBLE_AFTER_STANDARD_MAPPING |
| BM25 | COMPATIBLE_AFTER_STANDARD_MAPPING | COMPATIBLE_AFTER_STANDARD_MAPPING | COMPATIBLE_AFTER_STANDARD_MAPPING | COMPATIBLE_AFTER_STANDARD_MAPPING | COMPATIBLE_AFTER_STANDARD_MAPPING | UNKNOWN |
| BPRMF | COMPATIBLE_AFTER_STANDARD_MAPPING | COMPATIBLE_AFTER_STANDARD_MAPPING | COMPATIBLE_AFTER_STANDARD_MAPPING | COMPATIBLE_AFTER_STANDARD_MAPPING | COMPATIBLE_AFTER_STANDARD_MAPPING | COMPATIBLE_AFTER_STANDARD_MAPPING |
| GRU4Rec | COMPATIBLE_WITH_NONTRIVIAL_ADAPTATION | INCOMPATIBLE | COMPATIBLE_WITH_NONTRIVIAL_ADAPTATION | COMPATIBLE_AFTER_STANDARD_MAPPING | COMPATIBLE_AFTER_STANDARD_MAPPING | UNKNOWN |
| SASRec | COMPATIBLE_WITH_NONTRIVIAL_ADAPTATION | INCOMPATIBLE | COMPATIBLE_WITH_NONTRIVIAL_ADAPTATION | COMPATIBLE_AFTER_STANDARD_MAPPING | COMPATIBLE_AFTER_STANDARD_MAPPING | UNKNOWN |
| LightGCN | COMPATIBLE_AFTER_STANDARD_MAPPING | COMPATIBLE_AFTER_STANDARD_MAPPING | COMPATIBLE_AFTER_STANDARD_MAPPING | COMPATIBLE_AFTER_STANDARD_MAPPING | COMPATIBLE_AFTER_STANDARD_MAPPING | COMPATIBLE_AFTER_STANDARD_MAPPING |

## Paper-Specific Analysis Compatibility

Ranking, reranking, history-length sensitivity, candidate-position bias, profile generation, hallucination, novelty, popularity, diversity and fairness are represented by the metric matrix above. Food.com supports all applicable paper analyses before local extensions, provided E3 generates Dataset 5 candidate pools, traditional model outputs, raw LLM outputs and parser manifests. Steam is the backup with the same Role A path but a stronger need to document review/recommendation-signal adaptation.

## Calibration Compatibility

| candidate_name | categories_genres_tags | calibration_classification | role_b_assessment |
| --- | --- | --- | --- |
| Goodreads | popular shelves and official genre subset files. | CALIBRATION_ADEQUATE | strong LLM and calibration domain if shelf/genre coverage survives preprocessing. |
| BookCrossing / Book-Crossing | NOT_SUPPORTED authoritatively; no genre/tag/category field in the canonical schema. | CALIBRATION_BLOCKING | weak for calibration and sequential pipeline extensions. |
| Yelp Open Dataset | business categories are central Yelp metadata. | CALIBRATION_STRONG | strong calibration and business-category extension domain if access is resolved. |
| Steam reviews/interactions | genres, tags and specs in game metadata. | CALIBRATION_STRONG | strong metadata, candidate aggregation and calibration support. |
| Food.com | recipe tags, ingredients, techniques and contributor/submitted metadata. | CALIBRATION_STRONG | best Role B support for calibration, candidate recall and pipeline decompositions after Role A. |
| Google Restaurants / Google Local restaurant release | restaurant/business metadata is present; exact category-name fields must be verified. | CALIBRATION_ADEQUATE | promising restaurant category/geography domain after raw-field verification. |

## Pipeline-Aware Extension Compatibility

Food.com and Steam support source top-100 rankings, multiple recommenders, Round-Robin, RRF, Candidate Recall@10/20/50/100, conditional and end-to-end HR/NDCG@5, oracle candidate-availability ceilings, and controlled positive injection after Role A. These extensions remain secondary and must be executed only after Dataset 5 paper/local-reproduction parity artifacts are frozen.

## Full 25-Criterion Scorecard

| criterion | category | priority_class | goodreads | bookcrossing | yelp | steam | foodcom | google_restaurants |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Public availability | access | HARD_GATE | ADEQUATE | ADEQUATE | WEAK | STRONG | STRONG | STRONG |
| License/access terms | access | HARD_GATE | ADEQUATE | WEAK | UNKNOWN | ADEQUATE | ADEQUATE | ADEQUATE |
| Reproducible download | access | HARD_GATE | WEAK | ADEQUATE | WEAK | STRONG | ADEQUATE | STRONG |
| User-item interactions | schema | HARD_GATE | STRONG | STRONG | STRONG | STRONG | STRONG | STRONG |
| Timestamps/order | schema | HARD_GATE | ADEQUATE | BLOCKING | STRONG | STRONG | STRONG | UNKNOWN |
| Item titles | LLM compatibility | HARD_GATE | STRONG | STRONG | ADEQUATE | STRONG | STRONG | UNKNOWN |
| Item metadata/categories | beyond-utility | MEDIUM_PRIORITY | ADEQUATE | BLOCKING | STRONG | STRONG | STRONG | ADEQUATE |
| History depth | protocol | HARD_GATE | STRONG | WEAK | STRONG | ADEQUATE | STRONG | ADEQUATE |
| Post-filter user count | scale | HARD_GATE | STRONG | ADEQUATE | ADEQUATE | STRONG | ADEQUATE | ADEQUATE |
| Post-filter item count | scale | HARD_GATE | STRONG | ADEQUATE | ADEQUATE | ADEQUATE | STRONG | ADEQUATE |
| Interaction density | scale | MEDIUM_PRIORITY | ADEQUATE | WEAK | ADEQUATE | ADEQUATE | ADEQUATE | ADEQUATE |
| Long-tail properties | beyond-utility | HARD_GATE | STRONG | ADEQUATE | STRONG | ADEQUATE | STRONG | ADEQUATE |
| Leave-one-out compatibility | protocol | HARD_GATE | ADEQUATE | BLOCKING | STRONG | STRONG | STRONG | UNKNOWN |
| Paper metric compatibility | protocol | HARD_GATE | ADEQUATE | BLOCKING | ADEQUATE | STRONG | STRONG | WEAK |
| Profile generation compatibility | LLM compatibility | HARD_GATE | STRONG | BLOCKING | ADEQUATE | STRONG | STRONG | UNKNOWN |
| Hallucination evaluation compatibility | LLM compatibility | HARD_GATE | STRONG | ADEQUATE | ADEQUATE | STRONG | STRONG | UNKNOWN |
| Candidate-position experiment compatibility | protocol | HARD_GATE | ADEQUATE | WEAK | ADEQUATE | STRONG | STRONG | ADEQUATE |
| Calibration compatibility | extension | MEDIUM_PRIORITY | ADEQUATE | BLOCKING | STRONG | STRONG | STRONG | ADEQUATE |
| BPRMF compatibility | traditional models | MEDIUM_PRIORITY | ADEQUATE | ADEQUATE | ADEQUATE | STRONG | STRONG | ADEQUATE |
| GRU4Rec compatibility | traditional models | MEDIUM_PRIORITY | ADEQUATE | BLOCKING | ADEQUATE | STRONG | STRONG | UNKNOWN |
| SASRec compatibility | traditional models | MEDIUM_PRIORITY | ADEQUATE | BLOCKING | ADEQUATE | STRONG | STRONG | UNKNOWN |
| LightGCN compatibility | traditional models | MEDIUM_PRIORITY | ADEQUATE | ADEQUATE | ADEQUATE | STRONG | STRONG | ADEQUATE |
| Compute cost | runtime | HARD_GATE | WEAK | STRONG | ADEQUATE | ADEQUATE | STRONG | ADEQUATE |
| LLM prompt suitability | LLM compatibility | HARD_GATE | STRONG | ADEQUATE | STRONG | STRONG | STRONG | UNKNOWN |
| Domain novelty relative to Beauty/Sports/Movies/Music | scientific value | MEDIUM_PRIORITY | ADEQUATE | ADEQUATE | STRONG | STRONG | STRONG | STRONG |

## Hard-Gate Results

| candidate_name | result | reason |
| --- | --- | --- |
| Goodreads | PASS_WITH_DOCUMENTED_ADAPTATION | strong paper-framework support after official-subset and timestamp-policy freeze; compute-heavy. |
| BookCrossing / Book-Crossing | FAIL | fails mandatory timestamp/order and metadata gates. |
| Yelp Open Dataset | UNKNOWN | schema likely supports Role A after review-as-interaction adaptation, but access/license provenance is unresolved. |
| Steam reviews/interactions | PASS | strong Role A backup with full paper metric support after standard review-to-interaction mapping. |
| Food.com | PASS | best Role A balance: explicit interactions, dates, titles, rich metadata, official splits, manageable scale. |
| Google Restaurants / Google Local restaurant release | UNKNOWN | not selectable until item-name and timestamp fields are authoritatively confirmed. |

## Pareto Analysis

| candidate_name | pareto_class | rationale | sensitivity_note |
| --- | --- | --- | --- |
| Goodreads | PARETO_FRONT | Strong paper/LLM coverage but access and compute costs prevent it from dominating Food.com or Steam. | primary changes only under an extreme book-domain or scale-emphasis scenario. |
| BookCrossing / Book-Crossing | DOMINATED | Dominated because BookCrossing fails chronology and calibration while offering only compute convenience. | never wins under paper-framework or sequential emphasis. |
| Yelp Open Dataset | DOMINATED | Dominated by Food.com/Steam until current official license/version access is resolved. | could re-enter only after manual access proves terms and exact schema. |
| Steam reviews/interactions | PARETO_FRONT | Strong access, scale, timestamps and metadata; selected as backup. | can win if the project prioritizes direct UCSD download and game-domain novelty over Food.com moderation. |
| Food.com | PARETO_FRONT | Best balance across Role A completeness, sequential fidelity, metadata richness, calibration and compute. | stable winner under paper-framework, sequential, metadata, extension and compute emphasis. |
| Google Restaurants / Google Local restaurant release | DOMINATED | Dominated until title/name and timestamp fields are authoritatively confirmed. | could become competitive only after E3-style raw-field inspection, which E2 does not perform. |

## Sensitivity Analysis

- Paper-framework parity emphasis: Food.com remains primary; Steam remains backup.
- Sequential fidelity emphasis: Food.com and Steam remain strongest; BookCrossing is eliminated.
- Compute emphasis: Food.com wins; BookCrossing is smaller but fails hard gates.
- Metadata emphasis: Food.com, Steam and Goodreads are strongest; Food.com remains best because metadata is rich at moderate scale.
- Extension compatibility emphasis: Food.com remains primary because recipe tags, ingredients and techniques are directly usable for calibration and candidate aggregation.
- Access/licensing emphasis: Steam can become primary if the project prioritizes direct UCSD download over Food.com external-resource/manual intake.

No single candidate strictly dominates every dimension. Food.com is the stable selection under all scientific priorities except an access-only preference for Steam.

## Rejected Candidates and Exact Reasons

| candidate_name | selection_status | reason |
| --- | --- | --- |
| Goodreads | REJECTED_FINALIST | Not selected because compute/access restrictions and multi-date policy complexity are worse than Food.com for the same Role A coverage. |
| BookCrossing / Book-Crossing | REJECTED_HARD_GATE | Fails the frozen E1 chronological/sequential gate and lacks authoritative category/genre metadata. |
| Yelp Open Dataset | REJECTED_ACCESS_UNRESOLVED | Not selected because current official access/license/version details could not be established authoritatively in E2. |
| Steam reviews/interactions | BACKUP_DATASET | Selected as backup; weaker than Food.com only because the review/recommendation signal requires more domain adaptation. |
| Google Restaurants / Google Local restaurant release | REJECTED_FIELD_UNRESOLVED | Fails E2 certainty for item-title/name and timestamp gates; do not use broad Google Local restaurant slicing as a shortcut. |

## Primary Dataset Release

Food.com Recipes and Interactions, published 2019 Version v2; use RAW_recipes.csv, RAW_interactions.csv and official train/validation/test split files.

Use the official raw recipe and interaction tables plus official split files. Do not invent an arbitrary convenience subset in E2 or E3; if E3 discovers a compute issue, first evaluate documented official split artifacts and report the limitation.

## Scientific Risks / Limitations of Primary

- Exact raw user count and post-filter history-depth distribution must be verified after authorized E3 acquisition.
- External-resource/manual intake must be documented with checksums and no raw data committed.
- Repeated user-recipe interactions, if present, need a deterministic deduplication policy before split construction.
- All paper/local metrics still require new Dataset 5 candidate pools, traditional model outputs and LLM/parser manifests; E2 selects data only.

## Repository Postcondition

- E2 evidence artifacts only.
- No dataset downloaded.
- No preprocessing, training, LLM inference or API call.
- Frozen E1 artifacts and original_repo remain out of scope.
