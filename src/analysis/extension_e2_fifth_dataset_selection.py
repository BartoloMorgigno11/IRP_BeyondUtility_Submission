"""Build E2 fifth-dataset selection evidence artifacts.

This module is documentation/table generation only. It records authoritative
source evidence and compatibility decisions for selecting a fifth dataset
against the frozen E1 metric contract. It does not download datasets, train
models, preprocess data, run LLM inference, or modify frozen E1 artifacts.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
REPORT_PATH = PROJECT_ROOT / "report" / "extension_e2_fifth_dataset_selection.md"

EVIDENCE_LABEL = "POST-F6M EXTENSION DESIGN EVIDENCE"
DECISION_TOKEN = "EXTENSION_E2_FIFTH_DATASET_SELECTED"
PRIMARY_DATASET = "foodcom"
BACKUP_DATASET = "steam"

E1_METRIC_PATH = RESULTS_DIR / "extension_e1_metric_contract.csv"
E1_REQUIREMENTS_PATH = RESULTS_DIR / "extension_e1_fifth_dataset_requirements.csv"
E1_SCORECARD_PATH = RESULTS_DIR / "extension_e1_dataset_selection_scorecard_template.csv"

CANDIDATE_IDS = [
    "goodreads",
    "bookcrossing",
    "yelp",
    "steam",
    "foodcom",
    "google_restaurants",
]

COMPATIBILITY_VALUES = {
    "SUPPORTED_DIRECTLY",
    "SUPPORTED_AFTER_STANDARD_PREPROCESSING",
    "SUPPORTED_WITH_ADAPTATION",
    "NOT_SUPPORTED",
    "UNKNOWN",
}
GATE_VALUES = {"PASS", "PASS_WITH_DOCUMENTED_ADAPTATION", "FAIL", "UNKNOWN", "OPTIONAL"}
MODEL_VALUES = {
    "DIRECTLY_COMPATIBLE",
    "COMPATIBLE_AFTER_STANDARD_MAPPING",
    "COMPATIBLE_WITH_NONTRIVIAL_ADAPTATION",
    "INCOMPATIBLE",
    "UNKNOWN",
}
SEQUENTIAL_VALUES = {
    "SEQUENTIAL_STRONG",
    "SEQUENTIAL_ADAPTABLE",
    "SEQUENTIAL_WEAK",
    "SEQUENTIAL_BLOCKING",
}
LLM_METADATA_VALUES = {
    "LLM_METADATA_STRONG",
    "LLM_METADATA_ADEQUATE",
    "LLM_METADATA_WEAK",
    "LLM_METADATA_BLOCKING",
}
CALIBRATION_VALUES = {
    "CALIBRATION_STRONG",
    "CALIBRATION_ADEQUATE",
    "CALIBRATION_WEAK",
    "CALIBRATION_BLOCKING",
}
SCALE_VALUES = {
    "SMALL",
    "MODERATE",
    "LARGE_BUT_MANAGEABLE",
    "VERY_LARGE_REQUIRES_SUBSET",
    "COMPUTATIONALLY_RISKY",
}
RATING_VALUES = {"STRONG", "ADEQUATE", "WEAK", "BLOCKING", "UNKNOWN", "SECONDARY"}
PARETO_VALUES = {"PARETO_FRONT", "DOMINATED", "DOMINANT_IF_ANY"}

MODEL_NAMES = ["Random", "MostPop", "BM25", "BPRMF", "GRU4Rec", "SASRec", "LightGCN"]


def joined(values: Sequence[str]) -> str:
    return ";".join(values)


SOURCES: dict[str, dict[str, str]] = {
    "S_GOODREADS_UCSD": {
        "title": "UCSD Book Graph / Goodreads official release",
        "url": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/UCSD%20Book%20Graph.html",
        "notes": "Official UCSD release with book, interaction, review, shelf and genre-subset files; states academic use only and no redistribution or commercial use.",
    },
    "S_BOOKCROSSING_GROUPLENS": {
        "title": "GroupLens Book-Crossing dataset mirror",
        "url": "https://grouplens.org/datasets/book-crossing/",
        "notes": "Institutional mirror of Cai-Nicolas Ziegler's Book-Crossing data with user, book and rating tables.",
    },
    "S_BOOKCROSSING_MLDATA": {
        "title": "Mldata Book-Crossing schema/summary mirror",
        "url": "https://www.mldata.io/dataset-details/book_crossing/",
        "notes": "Trusted dataset catalog summary listing 278858 users, 271379 books and 1149780 ratings.",
    },
    "S_YELP_OPEN": {
        "title": "Yelp Open Dataset official page",
        "url": "https://business.yelp.com/data/resources/open-dataset/",
        "notes": "Official source page exists, but the web audit could not retrieve current terms/content. Treat exact current license/registration terms as unresolved until manual access.",
    },
    "S_YELP_DOCS": {
        "title": "Yelp public business data documentation",
        "url": "https://docs.developer.yelp.com/docs/resources-categories",
        "notes": "Yelp platform documentation confirms business categories and business entities, but it is not a substitute for Open Dataset license terms.",
    },
    "S_STEAM_MCAULEY": {
        "title": "UCSD Steam Video Game and Bundle Data",
        "url": "https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data",
        "notes": "Author-maintained academic release; states 7,793,069 reviews, 2,567,538 users and 15,474 games.",
    },
    "S_STEAM_BERD": {
        "title": "BERD Steam dataset catalog record",
        "url": "https://berd-platform.de/records/t0ce0-a0708",
        "notes": "Institutional record for the McAuley Steam release; lists open dataset status, 1.43 GB scale, and item metadata variables including genres, tags and title.",
    },
    "S_FOOD_MCAULEY": {
        "title": "UCSD Food.com Recipes and Interactions",
        "url": "https://cseweb.ucsd.edu/~jmcauley/datasets.html#foodcom",
        "notes": "Author-maintained academic release for Food.com recipes and interactions; links to download and citation.",
    },
    "S_FOOD_BERD": {
        "title": "BERD Food.com Recipes and Interactions catalog record",
        "url": "https://berd-platform.de/records/dmajj-zp851",
        "notes": "Institutional record for the Food.com release; lists open dataset status, 180K+ recipes, 700K+ reviews, 18-year span, CSV tables and variables including RAW_interactions date/rating/review and RAW_recipes name/tags/ingredients.",
    },
    "S_GOOGLE_RESTAURANTS_MCAULEY": {
        "title": "UCSD Google Restaurants dataset",
        "url": "https://cseweb.ucsd.edu/~jmcauley/datasets.html#google_restaurants",
        "notes": "Author-maintained academic restaurant release; states full/subset sizes and review fields, but the visible page does not explicitly establish restaurant name/title fields.",
    },
    "S_GOOGLE_LOCAL_2021": {
        "title": "UCSD Google Local 2021 dataset",
        "url": "https://cseweb.ucsd.edu/~jmcauley/datasets.html#google_local",
        "notes": "Author-maintained academic Google Local release with 666M reviews and business metadata; useful context but too broad for an unsupported restaurant-only convenience slice.",
    },
}


CANDIDATES: list[dict[str, str]] = [
    {
        "candidate_id": "goodreads",
        "candidate_name": "Goodreads",
        "exact_release": "UCSD Book Graph Goodreads release; use an official genre subset only after E3 verifies size and metadata coverage.",
        "access_mechanism": "Google Drive files from official UCSD Book Graph page.",
        "academic_terms": "Academic use only; no redistribution; no commercial use.",
        "paid_or_free": "free academic access",
        "registration_or_manual_request": "manual Google Drive download; some detailed data may require author contact.",
        "dataset_size": "large multi-file release; official page lists book metadata, interactions, reviews, shelves and genre subsets.",
        "users": "876145 users reported for full Goodreads interaction corpus by the official release.",
        "items": "2360655 books reported for the full release.",
        "interactions": "228648342 user-book interactions and 15739967 detailed reviews reported by the official release.",
        "timestamps": "detailed review records expose date_added, date_updated, read_at and started_at fields; chronology requires choosing and freezing one timestamp policy.",
        "chronology": "available but policy-sensitive because multiple date fields exist.",
        "item_title_name": "book title/title_without_series in book metadata.",
        "categories_genres_tags": "popular shelves and official genre subset files.",
        "additional_metadata": "authors, publication fields, ratings counts, shelves and review text.",
        "interaction_values": "shelf interactions, explicit ratings, review text and review dates depending on file.",
        "history_depth": "strong at full scale; subset-specific depth must be verified in E3.",
        "duplicate_behavior": "UNKNOWN before acquisition; Goodreads shelves/reviews may require per-user item deduplication policy.",
        "official_subsets": "children, comics_graphic, fantasy_paranormal, history_biography, mystery_thriller_crime, poetry, romance and young_adult official subsets.",
        "raw_download_scale": "VERY_LARGE_REQUIRES_SUBSET",
        "acquisition_reproducibility": "reproducible with manual UCSD Google Drive intake and no redistribution.",
        "sequential_classification": "SEQUENTIAL_ADAPTABLE",
        "llm_metadata_classification": "LLM_METADATA_STRONG",
        "calibration_classification": "CALIBRATION_ADEQUATE",
        "scale_classification": "VERY_LARGE_REQUIRES_SUBSET",
        "role_a_assessment": "strong paper-framework support after official-subset and timestamp-policy freeze; compute-heavy.",
        "role_b_assessment": "strong LLM and calibration domain if shelf/genre coverage survives preprocessing.",
        "selection_status": "REJECTED_FINALIST",
        "rejected_reason": "Not selected because compute/access restrictions and multi-date policy complexity are worse than Food.com for the same Role A coverage.",
        "source_ids": joined(["S_GOODREADS_UCSD"]),
    },
    {
        "candidate_id": "bookcrossing",
        "candidate_name": "BookCrossing / Book-Crossing",
        "exact_release": "Cai-Nicolas Ziegler Book-Crossing dataset as mirrored by GroupLens.",
        "access_mechanism": "downloadable zip via institutional mirrors.",
        "academic_terms": "public research dataset mirror; exact redistribution license not recovered in the official page during this audit.",
        "paid_or_free": "free",
        "registration_or_manual_request": "none observed on mirrors.",
        "dataset_size": "small/moderate classic recommender dataset.",
        "users": "278858 users reported by trusted dataset catalogs.",
        "items": "271379 books reported by trusted dataset catalogs.",
        "interactions": "1149780 ratings reported by trusted dataset catalogs.",
        "timestamps": "NOT_SUPPORTED: canonical schema is users, books and ratings without an authoritative timestamp/order field.",
        "chronology": "sequential parity is blocked; file order must not be used as chronology.",
        "item_title_name": "book title field exists.",
        "categories_genres_tags": "NOT_SUPPORTED authoritatively; no genre/tag/category field in the canonical schema.",
        "additional_metadata": "ISBN, author, publication year, publisher and image URLs.",
        "interaction_values": "explicit and implicit book ratings.",
        "history_depth": "rating histories exist but cannot be chronologically ordered.",
        "duplicate_behavior": "UNKNOWN.",
        "official_subsets": "none needed for size, none recovered as official scientific subset.",
        "raw_download_scale": "SMALL",
        "acquisition_reproducibility": "reproducible mirror download, but official source page availability is weaker than UCSD releases.",
        "sequential_classification": "SEQUENTIAL_BLOCKING",
        "llm_metadata_classification": "LLM_METADATA_WEAK",
        "calibration_classification": "CALIBRATION_BLOCKING",
        "scale_classification": "SMALL",
        "role_a_assessment": "fails mandatory timestamp/order and metadata gates.",
        "role_b_assessment": "weak for calibration and sequential pipeline extensions.",
        "selection_status": "REJECTED_HARD_GATE",
        "rejected_reason": "Fails the frozen E1 chronological/sequential gate and lacks authoritative category/genre metadata.",
        "source_ids": joined(["S_BOOKCROSSING_GROUPLENS", "S_BOOKCROSSING_MLDATA"]),
    },
    {
        "candidate_id": "yelp",
        "candidate_name": "Yelp Open Dataset",
        "exact_release": "Yelp Open Dataset current educational/open dataset release; exact current version must be captured during manual access.",
        "access_mechanism": "official Yelp Open Dataset web form/download, not Kaggle reuploads.",
        "academic_terms": "UNKNOWN in this audit because current official terms were not retrievable by web tool.",
        "paid_or_free": "free if accepted through official access flow; no paid requirement established.",
        "registration_or_manual_request": "manual form/terms acceptance likely required; exact requirement UNKNOWN.",
        "dataset_size": "large; public summaries report about 6.99M reviews and 150K businesses, but exact current official count remains to be verified.",
        "users": "UNKNOWN exact current official count during this audit.",
        "items": "business count reported by public dataset summaries; exact current official count UNKNOWN until manual access.",
        "interactions": "review rows can be operationalized as user-business interactions.",
        "timestamps": "review date field is a standard Open Dataset field in documented schema summaries.",
        "chronology": "review events are chronologically usable after mapping business reviews to interactions.",
        "item_title_name": "business names are expected in business records, but official current schema was not directly retrieved here.",
        "categories_genres_tags": "business categories are central Yelp metadata.",
        "additional_metadata": "business attributes, stars, review text, check-ins/tips/photos depending on release.",
        "interaction_values": "star ratings and review text.",
        "history_depth": "likely strong at full scale; must be verified after official acquisition.",
        "duplicate_behavior": "UNKNOWN before acquisition.",
        "official_subsets": "official city/category filters not treated as scientific subsets unless documented by Yelp.",
        "raw_download_scale": "LARGE_BUT_MANAGEABLE",
        "acquisition_reproducibility": "blocked from final selection until current terms and exact release are captured.",
        "sequential_classification": "SEQUENTIAL_STRONG",
        "llm_metadata_classification": "LLM_METADATA_ADEQUATE",
        "calibration_classification": "CALIBRATION_STRONG",
        "scale_classification": "LARGE_BUT_MANAGEABLE",
        "role_a_assessment": "schema likely supports Role A after review-as-interaction adaptation, but access/license provenance is unresolved.",
        "role_b_assessment": "strong calibration and business-category extension domain if access is resolved.",
        "selection_status": "REJECTED_ACCESS_UNRESOLVED",
        "rejected_reason": "Not selected because current official access/license/version details could not be established authoritatively in E2.",
        "source_ids": joined(["S_YELP_OPEN", "S_YELP_DOCS"]),
    },
    {
        "candidate_id": "steam",
        "candidate_name": "Steam reviews/interactions",
        "exact_release": "UCSD Steam Video Game and Bundle Data; use Version 2 review data plus item metadata from the same release.",
        "access_mechanism": "direct academic dataset links from UCSD McAuley page; BERD catalog mirrors descriptive metadata.",
        "academic_terms": "open academic dataset; cite Kang and McAuley 2018 and related release references.",
        "paid_or_free": "free",
        "registration_or_manual_request": "none established for UCSD download.",
        "dataset_size": "7,793,069 reviews, 2,567,538 users, 15,474 games; BERD record lists about 1.43 GB.",
        "users": "2567538 users reported by official UCSD page.",
        "items": "15474 games reported by official UCSD page.",
        "interactions": "7793069 reviews reported by official UCSD page; review/user-item records can be mapped to interactions.",
        "timestamps": "review date fields are available in release previews; freeze exact field in E3.",
        "chronology": "review chronology supports leave-one-out and sequential models after date normalization.",
        "item_title_name": "title field in Steam game metadata.",
        "categories_genres_tags": "genres, tags and specs in game metadata.",
        "additional_metadata": "developer, publisher, price, release date, early access, categories/specs/tags.",
        "interaction_values": "review/recommendation signal, playtime/hours fields depending on file.",
        "history_depth": "large enough for history-length bins after filtering; exact bins to verify in E3.",
        "duplicate_behavior": "UNKNOWN before acquisition; user-game review duplicates must be collapsed deterministically.",
        "official_subsets": "no smaller official scientific subset selected; use full release unless E3 shows compute risk.",
        "raw_download_scale": "LARGE_BUT_MANAGEABLE",
        "acquisition_reproducibility": "reproducible UCSD academic download.",
        "sequential_classification": "SEQUENTIAL_STRONG",
        "llm_metadata_classification": "LLM_METADATA_STRONG",
        "calibration_classification": "CALIBRATION_STRONG",
        "scale_classification": "LARGE_BUT_MANAGEABLE",
        "role_a_assessment": "strong Role A backup with full paper metric support after standard review-to-interaction mapping.",
        "role_b_assessment": "strong metadata, candidate aggregation and calibration support.",
        "selection_status": "BACKUP_DATASET",
        "rejected_reason": "Selected as backup; weaker than Food.com only because the review/recommendation signal requires more domain adaptation.",
        "source_ids": joined(["S_STEAM_MCAULEY", "S_STEAM_BERD"]),
    },
    {
        "candidate_id": "foodcom",
        "candidate_name": "Food.com",
        "exact_release": "Food.com Recipes and Interactions, published 2019 Version v2; use RAW_recipes.csv, RAW_interactions.csv and official train/validation/test split files.",
        "access_mechanism": "UCSD McAuley page links to the official dataset download; BERD records the dataset as an open dataset with external resource access.",
        "academic_terms": "open academic dataset record; cite Majumder et al. 2019; redistribution should remain via source links, not committed raw data.",
        "paid_or_free": "free",
        "registration_or_manual_request": "Kaggle/external-resource account or manual intake may be required; document in E3.",
        "dataset_size": "BERD records 180K+ recipes, 700K+ reviews, and 18 years of user interactions; 0.89 GB.",
        "users": "exact authoritative user count UNKNOWN before acquisition; raw and processed files include stable user_id/u fields.",
        "items": "180K+ recipes reported by BERD.",
        "interactions": "700K+ reviews/interactions reported by BERD; RAW_interactions contains user_id, recipe_id, date, rating and review.",
        "timestamps": "RAW_interactions date field; official processed files also include ordered items in user histories.",
        "chronology": "strong chronological support using explicit interaction date and official split/order files.",
        "item_title_name": "RAW_recipes name field.",
        "categories_genres_tags": "recipe tags, ingredients, techniques and contributor/submitted metadata.",
        "additional_metadata": "minutes, nutrition, steps, description, ingredients, n_steps and n_ingredients.",
        "interaction_values": "ratings and review text.",
        "history_depth": "official interaction histories and splits exist; exact post-filter depth to verify in E3.",
        "duplicate_behavior": "UNKNOWN before acquisition; repeated user-recipe reviews must be frozen deterministically if present.",
        "official_subsets": "use official train/validation/test split files; do not invent a convenience subset in E2.",
        "raw_download_scale": "MODERATE",
        "acquisition_reproducibility": "reproducible with documented external-resource/manual intake and no raw data committed.",
        "sequential_classification": "SEQUENTIAL_STRONG",
        "llm_metadata_classification": "LLM_METADATA_STRONG",
        "calibration_classification": "CALIBRATION_STRONG",
        "scale_classification": "MODERATE",
        "role_a_assessment": "best Role A balance: explicit interactions, dates, titles, rich metadata, official splits, manageable scale.",
        "role_b_assessment": "best Role B support for calibration, candidate recall and pipeline decompositions after Role A.",
        "selection_status": "PRIMARY_DATASET",
        "rejected_reason": "not rejected",
        "source_ids": joined(["S_FOOD_MCAULEY", "S_FOOD_BERD"]),
    },
    {
        "candidate_id": "google_restaurants",
        "candidate_name": "Google Restaurants / Google Local restaurant release",
        "exact_release": "UCSD Google Restaurants dataset from Personalized Showcases; evaluate the official restaurant full release and official subset only, not an arbitrary Google Local category slice.",
        "access_mechanism": "direct links from UCSD McAuley dataset page.",
        "academic_terms": "open academic release; cite Liu et al. 2023.",
        "paid_or_free": "free",
        "registration_or_manual_request": "none established for UCSD download.",
        "dataset_size": "full: 65779 restaurants, 1010915 users, 1771463 reviews; official subset: 30238 restaurants, 37720 users, 108289 reviews.",
        "users": "1010915 users full; 37720 users official subset.",
        "items": "65779 restaurants full; 30238 restaurants official subset.",
        "interactions": "1771463 reviews full; 108289 reviews official subset.",
        "timestamps": "timestamps are expected from Google review data lineage, but the visible restaurant page sample does not explicitly list a timestamp field.",
        "chronology": "chronological defensibility remains UNKNOWN until raw fields are inspected in E3.",
        "item_title_name": "UNKNOWN from the visible authoritative restaurant page; item-name/title gate cannot be passed in E2.",
        "categories_genres_tags": "restaurant/business metadata is present; exact category-name fields must be verified.",
        "additional_metadata": "review text, rating, picture IDs and restaurant metadata per the official page.",
        "interaction_values": "ratings, review text and image links.",
        "history_depth": "official subset may be shallow; full release likely adequate but must be verified.",
        "duplicate_behavior": "UNKNOWN before acquisition.",
        "official_subsets": "official restaurant subset provided by the release.",
        "raw_download_scale": "MODERATE",
        "acquisition_reproducibility": "reproducible UCSD academic download; avoid unsupported restaurant-only slices from broad Google Local.",
        "sequential_classification": "SEQUENTIAL_WEAK",
        "llm_metadata_classification": "LLM_METADATA_WEAK",
        "calibration_classification": "CALIBRATION_ADEQUATE",
        "scale_classification": "MODERATE",
        "role_a_assessment": "not selectable until item-name and timestamp fields are authoritatively confirmed.",
        "role_b_assessment": "promising restaurant category/geography domain after raw-field verification.",
        "selection_status": "REJECTED_FIELD_UNRESOLVED",
        "rejected_reason": "Fails E2 certainty for item-title/name and timestamp gates; do not use broad Google Local restaurant slicing as a shortcut.",
        "source_ids": joined(["S_GOOGLE_RESTAURANTS_MCAULEY", "S_GOOGLE_LOCAL_2021"]),
    },
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, str]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: str(row.get(column, "")) for column in columns})


def _candidate(candidate_id: str) -> dict[str, str]:
    for candidate in CANDIDATES:
        if candidate["candidate_id"] == candidate_id:
            return candidate
    raise KeyError(candidate_id)


def _metric_status(candidate_id: str, metric: Mapping[str, str]) -> tuple[str, str]:
    metric_id = metric["metric_id"]
    family = metric["family"]
    candidate = _candidate(candidate_id)

    if candidate_id == "bookcrossing":
        if metric_id in {
            "history_length_sensitivity",
            "generation_involved_profile_matrix",
            "calibration_at_5",
            "shannon_entropy_at_5",
            "intra_list_diversity_at_5",
        }:
            return "NOT_SUPPORTED", "No authoritative chronology and no category/tag/genre feature source."
        if metric_id in {"canddif_hr", "canddif_ndcg", "strict_parse_validity_rate", "duplicate_rate_at_5"}:
            return "SUPPORTED_WITH_ADAPTATION", "Candidate lists and parser diagnostics can be built, but lack of chronology blocks full parity."
        if metric_id in {
            "gini_at_5",
            "dpd_at_5",
            "jain_index_at_5",
            "ks_sample_representativeness",
            "candidate_recall_at_n",
            "conditional_hr_ndcg_at_5",
        }:
            return "NOT_SUPPORTED", "Sequential/user-activity or full pipeline requirements are not defensibly supported."
        return "SUPPORTED_AFTER_STANDARD_PREPROCESSING", "Utility/popularity metrics can be computed over ratings, but the dataset cannot win Role A."

    if candidate_id == "google_restaurants":
        if metric_id in {
            "hallucination_rate_at_5",
            "generation_involved_profile_matrix",
            "strict_parse_validity_rate",
        }:
            return "UNKNOWN", "Visible authoritative restaurant page does not establish item title/name fields needed for paper-like prompts."
        if metric_id in {"history_length_sensitivity", "canddif_hr", "canddif_ndcg"}:
            return "SUPPORTED_WITH_ADAPTATION", "Likely possible after raw timestamp/name verification, but not proven enough for selection."
        return "SUPPORTED_AFTER_STANDARD_PREPROCESSING", "Review interactions and restaurant metadata are available, subject to raw-field verification."

    if candidate_id == "yelp":
        if metric_id in {"cost_token_latency_summary"}:
            return "SUPPORTED_WITH_ADAPTATION", "Requires new local inference logs after official access is resolved."
        return "SUPPORTED_WITH_ADAPTATION", "Schema likely supports the metric after review-as-interaction mapping, but current access/license version is unresolved."

    if family == "local_extension" or "LLM" in family:
        return "SUPPORTED_WITH_ADAPTATION", "Requires Dataset 5 prompts/raw outputs or extension-stage artifacts after Role A."

    if metric_id in {
        "popreo_at_5",
        "oic_at_5",
        "shannon_entropy_at_5",
        "gini_at_5",
        "dpd_at_5",
        "jain_index_at_5",
        "ks_sample_representativeness",
    }:
        return "SUPPORTED_WITH_ADAPTATION", "Data primitives exist; E1 already records unresolved/local implementation policy details."

    if candidate_id == "goodreads":
        return "SUPPORTED_WITH_ADAPTATION", "Multiple date fields and very large official files require a documented subset/timestamp policy."

    return "SUPPORTED_AFTER_STANDARD_PREPROCESSING", f"{candidate['candidate_name']} exposes the data primitives required by the frozen metric row."


MODEL_COMPATIBILITY: dict[str, dict[str, tuple[str, str]]] = {
    "goodreads": {
        "Random": ("DIRECTLY_COMPATIBLE", "Candidate pools can be sampled once split is frozen."),
        "MostPop": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "Train-only item popularity from interactions."),
        "BM25": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "Book titles/authors/shelves can form item text."),
        "BPRMF": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "Implicit or explicit interactions after deduplication."),
        "GRU4Rec": ("COMPATIBLE_WITH_NONTRIVIAL_ADAPTATION", "Multiple date fields require chronological policy."),
        "SASRec": ("COMPATIBLE_WITH_NONTRIVIAL_ADAPTATION", "Multiple date fields require chronological policy."),
        "LightGCN": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "User-book graph after preprocessing."),
    },
    "bookcrossing": {
        "Random": ("DIRECTLY_COMPATIBLE", "Candidate pools can be sampled."),
        "MostPop": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "Popularity from ratings."),
        "BM25": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "Book title/author text available."),
        "BPRMF": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "Ratings can become implicit feedback."),
        "GRU4Rec": ("INCOMPATIBLE", "No authoritative chronology/timestamp."),
        "SASRec": ("INCOMPATIBLE", "No authoritative chronology/timestamp."),
        "LightGCN": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "User-book graph from ratings."),
    },
    "yelp": {
        "Random": ("DIRECTLY_COMPATIBLE", "Candidate pools can be sampled after access."),
        "MostPop": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "Business popularity from train reviews."),
        "BM25": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "Business names/categories/reviews can form item text."),
        "BPRMF": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "User-business review graph."),
        "GRU4Rec": ("COMPATIBLE_WITH_NONTRIVIAL_ADAPTATION", "Review-as-interaction semantics and current release must be frozen."),
        "SASRec": ("COMPATIBLE_WITH_NONTRIVIAL_ADAPTATION", "Review-as-interaction semantics and current release must be frozen."),
        "LightGCN": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "User-business graph from reviews."),
    },
    "steam": {
        "Random": ("DIRECTLY_COMPATIBLE", "Candidate pools can be sampled."),
        "MostPop": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "Game popularity from train review/interactions."),
        "BM25": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "Game title/genres/tags/specs can form item text."),
        "BPRMF": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "User-game feedback graph."),
        "GRU4Rec": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "Review date supports sequence construction after field freeze."),
        "SASRec": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "Review date supports sequence construction after field freeze."),
        "LightGCN": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "User-game graph from interactions."),
    },
    "foodcom": {
        "Random": ("DIRECTLY_COMPATIBLE", "Candidate pools can be sampled."),
        "MostPop": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "Recipe popularity from train interactions."),
        "BM25": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "Recipe names/tags/ingredients/descriptions can form item text."),
        "BPRMF": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "User-recipe ratings/reviews as feedback."),
        "GRU4Rec": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "Explicit interaction date and processed histories support sequence construction."),
        "SASRec": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "Explicit interaction date and processed histories support sequence construction."),
        "LightGCN": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "User-recipe graph from interactions."),
    },
    "google_restaurants": {
        "Random": ("DIRECTLY_COMPATIBLE", "Candidate pools can be sampled after field verification."),
        "MostPop": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "Restaurant popularity from train reviews."),
        "BM25": ("UNKNOWN", "Visible source does not establish restaurant name/title field."),
        "BPRMF": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "User-restaurant ratings/reviews as feedback."),
        "GRU4Rec": ("UNKNOWN", "Timestamp field not explicit on visible restaurant page."),
        "SASRec": ("UNKNOWN", "Timestamp field not explicit on visible restaurant page."),
        "LightGCN": ("COMPATIBLE_AFTER_STANDARD_MAPPING", "User-restaurant graph from interactions."),
    },
}


def _model_compatibility_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for candidate in CANDIDATES:
        candidate_id = candidate["candidate_id"]
        for model_name in MODEL_NAMES:
            status, rationale = MODEL_COMPATIBILITY[candidate_id][model_name]
            rows.append(
                {
                    "evidence_label": EVIDENCE_LABEL,
                    "candidate_id": candidate_id,
                    "candidate_name": candidate["candidate_name"],
                    "model_name": model_name,
                    "compatibility_status": status,
                    "rationale": rationale,
                    "source_ids": candidate["source_ids"],
                }
            )
    return rows


def _gate_status(candidate_id: str, requirement: Mapping[str, str]) -> tuple[str, str]:
    name = requirement["requirement"]
    mandatory = requirement["mandatory"]

    if candidate_id == "bookcrossing":
        if name in {
            "Chronological order or timestamps",
            "Item metadata categories/tags/genres",
            "Reranking candidate pools from BPRMF, GRU4Rec, SASRec, and LightGCN top-K outputs",
            "History length bins with sufficient depth",
            "User activity groups based on median history length",
            "Traditional model training-ready split and configs",
            "Candidate retrieval and aggregation intermediate artifacts",
            "Positive-injection intervention records",
        }:
            return "FAIL", "Canonical schema lacks timestamps/order and category metadata; sequential Role A parity fails."
        if name in {"Profile-generation outputs and dependency manifest"}:
            return "FAIL", "Profile-generation histories cannot be chronologically truncated."
        return "PASS_WITH_DOCUMENTED_ADAPTATION", "Can support simple rating/title metrics, but not the full E1 contract."

    if candidate_id == "yelp" and name == "Public, reproducible dataset access and license":
        return "UNKNOWN", "Current official terms/version were not retrievable during E2."

    if candidate_id == "google_restaurants":
        if name in {
            "Chronological order or timestamps",
            "Item titles or names",
            "Free-text/title-list LLM outputs or paper-equivalent parser outputs",
            "Profile-generation outputs and dependency manifest",
            "Traditional model training-ready split and configs",
        }:
            return "UNKNOWN", "Visible authoritative page does not explicitly prove the required title/timestamp fields."
        return "PASS_WITH_DOCUMENTED_ADAPTATION", "Official restaurant release likely supports the primitive after raw-field verification."

    if name == "Token, latency, and cost logs":
        return "OPTIONAL", "Not a paper metric; create only when new inference is later run."

    if candidate_id == "goodreads":
        if name in {
            "Chronological order or timestamps",
            "Reranking candidate pools from BPRMF, GRU4Rec, SASRec, and LightGCN top-K outputs",
            "History length bins with sufficient depth",
            "Traditional model training-ready split and configs",
        }:
            return "PASS_WITH_DOCUMENTED_ADAPTATION", "Multiple date fields and scale require documented policy/subset freeze."
        return "PASS_WITH_DOCUMENTED_ADAPTATION", "Required primitives are present, with academic-access and scale caveats."

    if candidate_id == "steam":
        if name in {
            "Traditional model training-ready split and configs",
            "Reranking candidate pools from BPRMF, GRU4Rec, SASRec, and LightGCN top-K outputs",
            "Free-text/title-list LLM outputs or paper-equivalent parser outputs",
            "Profile-generation outputs and dependency manifest",
            "Candidate retrieval and aggregation intermediate artifacts",
            "Positive-injection intervention records",
        }:
            return "PASS_WITH_DOCUMENTED_ADAPTATION", "New Dataset 5 artifacts must be generated; raw fields support them."
        return "PASS", "Raw release evidence supports the required primitive."

    if candidate_id == "foodcom":
        if name in {
            "Public, reproducible dataset access and license",
            "Reranking candidate pools from BPRMF, GRU4Rec, SASRec, and LightGCN top-K outputs",
            "Free-text/title-list LLM outputs or paper-equivalent parser outputs",
            "Profile-generation outputs and dependency manifest",
            "Traditional model training-ready split and configs",
            "Raw output manifests, parser provenance, and invalid-output policy",
            "Candidate retrieval and aggregation intermediate artifacts",
            "Positive-injection intervention records",
        }:
            return "PASS_WITH_DOCUMENTED_ADAPTATION", "Requires E3 manual intake or later generated outputs, but dataset primitives support it."
        return "PASS", "Raw and official split files support the required primitive."

    if mandatory == "optional":
        return "OPTIONAL", "Optional support artifact."
    return "PASS_WITH_DOCUMENTED_ADAPTATION", "Default documented adaptation."


def _hard_gate_rows(requirements: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for candidate in CANDIDATES:
        for requirement in requirements:
            status, evidence = _gate_status(candidate["candidate_id"], requirement)
            rows.append(
                {
                    "evidence_label": EVIDENCE_LABEL,
                    "candidate_id": candidate["candidate_id"],
                    "candidate_name": candidate["candidate_name"],
                    "requirement": requirement["requirement"],
                    "mandatory": requirement["mandatory"],
                    "requirement_class": requirement["requirement_class"],
                    "gate_status": status,
                    "evidence": evidence,
                    "source_ids": candidate["source_ids"],
                }
            )
    return rows


SCORECARD_RATINGS: dict[str, dict[str, tuple[str, str]]] = {
    "goodreads": {
        "Public availability": ("ADEQUATE", "Public UCSD page, but academic-only and manual files."),
        "License/access terms": ("ADEQUATE", "Terms are explicit but restrictive: academic only, no redistribution/commercial use."),
        "Reproducible download": ("WEAK", "Google Drive/manual intake and possible author contact for detailed files."),
        "User-item interactions": ("STRONG", "Large user-book interaction and review corpus."),
        "Timestamps/order": ("ADEQUATE", "Multiple review date fields; must freeze chronology policy."),
        "Item titles": ("STRONG", "Book title metadata."),
        "Item metadata/categories": ("ADEQUATE", "Shelves/genres exist but need coverage audit."),
        "History depth": ("STRONG", "Full corpus is deep; subset depth must be verified."),
        "Post-filter user count": ("STRONG", "Full scale far exceeds 1000 users."),
        "Post-filter item count": ("STRONG", "Large catalog and official subsets."),
        "Interaction density": ("ADEQUATE", "Likely sparse but large enough."),
        "Long-tail properties": ("STRONG", "Large catalog supports long-tail grouping."),
        "Leave-one-out compatibility": ("ADEQUATE", "Possible after timestamp-field policy."),
        "Paper metric compatibility": ("ADEQUATE", "Supports primitives with scale/date adaptation."),
        "Profile generation compatibility": ("STRONG", "Book histories are prompt-suitable."),
        "Hallucination evaluation compatibility": ("STRONG", "Title catalog supports matching."),
        "Candidate-position experiment compatibility": ("ADEQUATE", "Candidate pools can be built."),
        "Calibration compatibility": ("ADEQUATE", "Genres/shelves can support features after audit."),
        "BPRMF compatibility": ("ADEQUATE", "Graph training possible."),
        "GRU4Rec compatibility": ("ADEQUATE", "Sequential training possible after chronology policy."),
        "SASRec compatibility": ("ADEQUATE", "Sequential training possible after chronology policy."),
        "LightGCN compatibility": ("ADEQUATE", "Graph training possible."),
        "Compute cost": ("WEAK", "Very large; likely requires official subset."),
        "LLM prompt suitability": ("STRONG", "Book titles/authors are prompt-friendly."),
        "Domain novelty relative to Beauty/Sports/Movies/Music": ("ADEQUATE", "Books are new but adjacent to media domains."),
    },
    "bookcrossing": {
        "Public availability": ("ADEQUATE", "Institutional mirror exists."),
        "License/access terms": ("WEAK", "Exact original redistribution terms not recovered."),
        "Reproducible download": ("ADEQUATE", "Small mirror download."),
        "User-item interactions": ("STRONG", "Stable users/books/ratings."),
        "Timestamps/order": ("BLOCKING", "No authoritative timestamp/order field."),
        "Item titles": ("STRONG", "Book title field."),
        "Item metadata/categories": ("BLOCKING", "No authoritative category/genre/tag field."),
        "History depth": ("WEAK", "Ratings exist but cannot be time-ordered."),
        "Post-filter user count": ("ADEQUATE", "Large enough in raw count."),
        "Post-filter item count": ("ADEQUATE", "Large enough in raw count."),
        "Interaction density": ("WEAK", "Sparse and old."),
        "Long-tail properties": ("ADEQUATE", "Book popularity distribution can be computed."),
        "Leave-one-out compatibility": ("BLOCKING", "Chronological leave-one-out unsupported."),
        "Paper metric compatibility": ("BLOCKING", "Sequential/history metrics fail."),
        "Profile generation compatibility": ("BLOCKING", "No chronological histories."),
        "Hallucination evaluation compatibility": ("ADEQUATE", "Titles support string matching."),
        "Candidate-position experiment compatibility": ("WEAK", "Candidate lists possible but Role A fails."),
        "Calibration compatibility": ("BLOCKING", "No feature distribution source."),
        "BPRMF compatibility": ("ADEQUATE", "Ratings graph usable."),
        "GRU4Rec compatibility": ("BLOCKING", "No chronology."),
        "SASRec compatibility": ("BLOCKING", "No chronology."),
        "LightGCN compatibility": ("ADEQUATE", "Graph usable."),
        "Compute cost": ("STRONG", "Small."),
        "LLM prompt suitability": ("ADEQUATE", "Titles only."),
        "Domain novelty relative to Beauty/Sports/Movies/Music": ("ADEQUATE", "Books are new but media-adjacent."),
    },
    "yelp": {
        "Public availability": ("WEAK", "Official page exists but current content/terms not retrievable by web audit."),
        "License/access terms": ("UNKNOWN", "Current official educational/research terms must be accepted and archived manually."),
        "Reproducible download": ("WEAK", "Manual official access required; avoid Kaggle copies."),
        "User-item interactions": ("STRONG", "Review rows map to user-business interactions."),
        "Timestamps/order": ("STRONG", "Review dates support chronology."),
        "Item titles": ("ADEQUATE", "Business names expected but current schema not fetched."),
        "Item metadata/categories": ("STRONG", "Business categories support calibration."),
        "History depth": ("STRONG", "Large review corpus likely sufficient."),
        "Post-filter user count": ("ADEQUATE", "Likely sufficient; exact current count unresolved."),
        "Post-filter item count": ("ADEQUATE", "Likely sufficient; exact current count unresolved."),
        "Interaction density": ("ADEQUATE", "Large sparse local-business graph."),
        "Long-tail properties": ("STRONG", "Business popularity long tail likely strong."),
        "Leave-one-out compatibility": ("STRONG", "Review chronology supports split."),
        "Paper metric compatibility": ("ADEQUATE", "Requires review-as-interaction adaptation."),
        "Profile generation compatibility": ("ADEQUATE", "Business-review histories are prompt-suitable."),
        "Hallucination evaluation compatibility": ("ADEQUATE", "Business names support matching after schema verification."),
        "Candidate-position experiment compatibility": ("ADEQUATE", "Candidate lists can be built."),
        "Calibration compatibility": ("STRONG", "Business categories are strong feature source."),
        "BPRMF compatibility": ("ADEQUATE", "Graph training possible."),
        "GRU4Rec compatibility": ("ADEQUATE", "Chronological reviews possible."),
        "SASRec compatibility": ("ADEQUATE", "Chronological reviews possible."),
        "LightGCN compatibility": ("ADEQUATE", "Graph training possible."),
        "Compute cost": ("ADEQUATE", "Large but manageable."),
        "LLM prompt suitability": ("STRONG", "Local-business names/categories suit prompts."),
        "Domain novelty relative to Beauty/Sports/Movies/Music": ("STRONG", "Local-business domain is novel."),
    },
    "steam": {
        "Public availability": ("STRONG", "UCSD academic release with direct data links."),
        "License/access terms": ("ADEQUATE", "Academic citation terms; do not redistribute raw data."),
        "Reproducible download": ("STRONG", "Direct UCSD release and institutional catalog record."),
        "User-item interactions": ("STRONG", "Large user-game review/interactions."),
        "Timestamps/order": ("STRONG", "Review dates support chronological histories."),
        "Item titles": ("STRONG", "Game titles in metadata."),
        "Item metadata/categories": ("STRONG", "Genres/tags/specs support features."),
        "History depth": ("ADEQUATE", "Large enough after filtering; exact bins in E3."),
        "Post-filter user count": ("STRONG", "Official user count far exceeds sample needs."),
        "Post-filter item count": ("ADEQUATE", "15K games sufficient for candidate pools."),
        "Interaction density": ("ADEQUATE", "Large sparse graph."),
        "Long-tail properties": ("ADEQUATE", "Catalog supports popularity groups."),
        "Leave-one-out compatibility": ("STRONG", "Timestamped review events."),
        "Paper metric compatibility": ("STRONG", "All primitives available after standard mapping."),
        "Profile generation compatibility": ("STRONG", "Game histories and metadata are prompt-suitable."),
        "Hallucination evaluation compatibility": ("STRONG", "Title catalog supports matching."),
        "Candidate-position experiment compatibility": ("STRONG", "Candidate pools can be built."),
        "Calibration compatibility": ("STRONG", "Genres/tags are strong feature source."),
        "BPRMF compatibility": ("STRONG", "Graph feedback compatible."),
        "GRU4Rec compatibility": ("STRONG", "Timestamped sequences compatible."),
        "SASRec compatibility": ("STRONG", "Timestamped sequences compatible."),
        "LightGCN compatibility": ("STRONG", "Graph compatible."),
        "Compute cost": ("ADEQUATE", "Large but manageable."),
        "LLM prompt suitability": ("STRONG", "Game titles/tags prompt well."),
        "Domain novelty relative to Beauty/Sports/Movies/Music": ("STRONG", "Games are novel relative to current four."),
    },
    "foodcom": {
        "Public availability": ("STRONG", "Author-maintained UCSD page and BERD open dataset record."),
        "License/access terms": ("ADEQUATE", "Academic citation and source-link redistribution policy required."),
        "Reproducible download": ("ADEQUATE", "External-resource/manual intake may be required, but source is authoritative."),
        "User-item interactions": ("STRONG", "RAW_interactions has user_id and recipe_id."),
        "Timestamps/order": ("STRONG", "Explicit date field and processed histories."),
        "Item titles": ("STRONG", "Recipe name field."),
        "Item metadata/categories": ("STRONG", "Tags, ingredients and techniques are rich features."),
        "History depth": ("STRONG", "Official user histories/splits exist; exact bins in E3."),
        "Post-filter user count": ("ADEQUATE", "Exact user count to verify, but 700K+ interactions support 1000-user sampling."),
        "Post-filter item count": ("STRONG", "180K+ recipes."),
        "Interaction density": ("ADEQUATE", "Moderate user-recipe graph."),
        "Long-tail properties": ("STRONG", "Large recipe catalog supports long-tail groups."),
        "Leave-one-out compatibility": ("STRONG", "Date and official split/order files support it."),
        "Paper metric compatibility": ("STRONG", "All primitives available with manageable scale."),
        "Profile generation compatibility": ("STRONG", "Recipe histories are prompt-suitable."),
        "Hallucination evaluation compatibility": ("STRONG", "Recipe-name catalog supports matching."),
        "Candidate-position experiment compatibility": ("STRONG", "Candidate pools can be built."),
        "Calibration compatibility": ("STRONG", "Tags/ingredients/techniques fit JSD/Hellinger features."),
        "BPRMF compatibility": ("STRONG", "User-recipe graph compatible."),
        "GRU4Rec compatibility": ("STRONG", "Timestamped histories compatible."),
        "SASRec compatibility": ("STRONG", "Timestamped histories compatible."),
        "LightGCN compatibility": ("STRONG", "Graph compatible."),
        "Compute cost": ("STRONG", "Moderate size."),
        "LLM prompt suitability": ("STRONG", "Recipe titles and ingredients suit prompts."),
        "Domain novelty relative to Beauty/Sports/Movies/Music": ("STRONG", "Recipes are novel relative to current four."),
    },
    "google_restaurants": {
        "Public availability": ("STRONG", "UCSD academic release exists."),
        "License/access terms": ("ADEQUATE", "Academic citation terms."),
        "Reproducible download": ("STRONG", "Direct UCSD release."),
        "User-item interactions": ("STRONG", "User-restaurant reviews/ratings."),
        "Timestamps/order": ("UNKNOWN", "Timestamp not explicit on visible restaurant page."),
        "Item titles": ("UNKNOWN", "Restaurant name/title not explicit on visible page."),
        "Item metadata/categories": ("ADEQUATE", "Restaurant metadata exists but exact fields need verification."),
        "History depth": ("ADEQUATE", "Full release likely enough; official subset may be shallow."),
        "Post-filter user count": ("ADEQUATE", "Full and subset user counts are official."),
        "Post-filter item count": ("ADEQUATE", "Full and subset restaurant counts are official."),
        "Interaction density": ("ADEQUATE", "Review graph likely sparse."),
        "Long-tail properties": ("ADEQUATE", "Restaurant popularity groups possible after verification."),
        "Leave-one-out compatibility": ("UNKNOWN", "Timestamp field unresolved."),
        "Paper metric compatibility": ("WEAK", "Title/timestamp gates unresolved."),
        "Profile generation compatibility": ("UNKNOWN", "Depends on restaurant names and chronological histories."),
        "Hallucination evaluation compatibility": ("UNKNOWN", "Depends on restaurant name field."),
        "Candidate-position experiment compatibility": ("ADEQUATE", "Candidate lists possible after field verification."),
        "Calibration compatibility": ("ADEQUATE", "Restaurant metadata likely supports features."),
        "BPRMF compatibility": ("ADEQUATE", "Graph compatible."),
        "GRU4Rec compatibility": ("UNKNOWN", "Timestamp unresolved."),
        "SASRec compatibility": ("UNKNOWN", "Timestamp unresolved."),
        "LightGCN compatibility": ("ADEQUATE", "Graph compatible."),
        "Compute cost": ("ADEQUATE", "Moderate full release; small official subset."),
        "LLM prompt suitability": ("UNKNOWN", "Depends on name/title fields."),
        "Domain novelty relative to Beauty/Sports/Movies/Music": ("STRONG", "Restaurants are novel."),
    },
}


def _priority(weight: str, mandatory_gate: str) -> str:
    if mandatory_gate == "yes":
        return "HARD_GATE"
    if weight == "high":
        return "HIGH_PRIORITY"
    if weight == "medium":
        return "MEDIUM_PRIORITY"
    return "SECONDARY"


def _scorecard_rows(scorecard: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for candidate in CANDIDATES:
        ratings = SCORECARD_RATINGS[candidate["candidate_id"]]
        for criterion in scorecard:
            rating, evidence = ratings[criterion["criterion"]]
            rows.append(
                {
                    "evidence_label": EVIDENCE_LABEL,
                    "candidate_id": candidate["candidate_id"],
                    "candidate_name": candidate["candidate_name"],
                    "criterion": criterion["criterion"],
                    "category": criterion["category"],
                    "e1_weight": criterion["weight"],
                    "e1_mandatory_gate": criterion["mandatory_gate"],
                    "priority_class": _priority(criterion["weight"], criterion["mandatory_gate"]),
                    "rating": rating,
                    "evidence": evidence,
                    "source_ids": candidate["source_ids"],
                }
            )
    return rows


def _metric_compatibility_rows(metrics: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for candidate in CANDIDATES:
        for metric in metrics:
            status, rationale = _metric_status(candidate["candidate_id"], metric)
            rows.append(
                {
                    "evidence_label": EVIDENCE_LABEL,
                    "candidate_id": candidate["candidate_id"],
                    "candidate_name": candidate["candidate_name"],
                    "metric_id": metric["metric_id"],
                    "family": metric["family"],
                    "metric_name": metric["metric_name"],
                    "e1_requirement_classification": metric["fifth_dataset_requirement_classification"],
                    "compatibility_status": status,
                    "rationale": rationale,
                    "source_ids": candidate["source_ids"],
                }
            )
    return rows


def _pareto_rows() -> list[dict[str, str]]:
    evidence = {
        "goodreads": (
            "PARETO_FRONT",
            "Strong paper/LLM coverage but access and compute costs prevent it from dominating Food.com or Steam.",
            "primary changes only under an extreme book-domain or scale-emphasis scenario.",
        ),
        "bookcrossing": (
            "DOMINATED",
            "Dominated because BookCrossing fails chronology and calibration while offering only compute convenience.",
            "never wins under paper-framework or sequential emphasis.",
        ),
        "yelp": (
            "DOMINATED",
            "Dominated by Food.com/Steam until current official license/version access is resolved.",
            "could re-enter only after manual access proves terms and exact schema.",
        ),
        "steam": (
            "PARETO_FRONT",
            "Strong access, scale, timestamps and metadata; selected as backup.",
            "can win if the project prioritizes direct UCSD download and game-domain novelty over Food.com moderation.",
        ),
        "foodcom": (
            "PARETO_FRONT",
            "Best balance across Role A completeness, sequential fidelity, metadata richness, calibration and compute.",
            "stable winner under paper-framework, sequential, metadata, extension and compute emphasis.",
        ),
        "google_restaurants": (
            "DOMINATED",
            "Dominated until title/name and timestamp fields are authoritatively confirmed.",
            "could become competitive only after E3-style raw-field inspection, which E2 does not perform.",
        ),
    }
    rows = []
    for candidate in CANDIDATES:
        pareto_class, rationale, sensitivity = evidence[candidate["candidate_id"]]
        rows.append(
            {
                "evidence_label": EVIDENCE_LABEL,
                "candidate_id": candidate["candidate_id"],
                "candidate_name": candidate["candidate_name"],
                "pareto_class": pareto_class,
                "rationale": rationale,
                "sensitivity_note": sensitivity,
                "source_ids": candidate["source_ids"],
            }
        )
    return rows


def _compute_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for candidate in CANDIDATES:
        rows.append(
            {
                "evidence_label": EVIDENCE_LABEL,
                "candidate_id": candidate["candidate_id"],
                "candidate_name": candidate["candidate_name"],
                "scale_classification": candidate["scale_classification"],
                "raw_download_scale": candidate["raw_download_scale"],
                "official_subset_policy": candidate["official_subsets"],
                "compute_rationale": candidate["dataset_size"],
                "risk": (
                    "primary manageable with no arbitrary subset"
                    if candidate["candidate_id"] == PRIMARY_DATASET
                    else candidate["rejected_reason"]
                ),
                "source_ids": candidate["source_ids"],
            }
        )
    return rows


def _decision_rows() -> list[dict[str, str]]:
    return [
        {
            "evidence_label": EVIDENCE_LABEL,
            "decision_token": DECISION_TOKEN,
            "primary_dataset_id": PRIMARY_DATASET,
            "primary_dataset_name": _candidate(PRIMARY_DATASET)["candidate_name"],
            "backup_dataset_id": BACKUP_DATASET,
            "backup_dataset_name": _candidate(BACKUP_DATASET)["candidate_name"],
            "primary_release": _candidate(PRIMARY_DATASET)["exact_release"],
            "backup_release": _candidate(BACKUP_DATASET)["exact_release"],
            "can_receive_complete_applicable_paper_local_reproduction_before_extensions": "YES",
            "selection_rationale": "Food.com has the strongest combined Role A support: explicit user-recipe interactions, dates, recipe titles, rich tags/ingredients/techniques, official split artifacts, manageable scale, and strong later calibration support. Steam is the backup because it is also strong but requires more review/recommendation-signal adaptation.",
            "scope_boundary": "No E3 acquisition, no preprocessing, no training, no LLM inference, no dataset download.",
        }
    ]


def _markdown_table(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [str(row.get(column, "")).replace("\n", " ").replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _matrix(
    rows: Sequence[Mapping[str, str]],
    row_key: str,
    value_key: str,
    prefix_columns: Sequence[str],
) -> list[dict[str, str]]:
    by_key: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row[row_key]
        if key not in by_key:
            by_key[key] = {column: row[column] for column in prefix_columns}
        by_key[key][row["candidate_id"]] = row[value_key]
    return list(by_key.values())


def _candidate_summary_rows() -> list[dict[str, str]]:
    columns = [
        "candidate_id",
        "candidate_name",
        "exact_release",
        "access_mechanism",
        "academic_terms",
        "paid_or_free",
        "registration_or_manual_request",
        "dataset_size",
        "users",
        "items",
        "interactions",
        "timestamps",
        "chronology",
        "item_title_name",
        "categories_genres_tags",
        "additional_metadata",
        "interaction_values",
        "history_depth",
        "duplicate_behavior",
        "official_subsets",
        "raw_download_scale",
        "acquisition_reproducibility",
        "sequential_classification",
        "llm_metadata_classification",
        "calibration_classification",
        "scale_classification",
        "role_a_assessment",
        "role_b_assessment",
        "selection_status",
        "rejected_reason",
        "source_ids",
    ]
    return [{column: candidate[column] for column in columns} | {"evidence_label": EVIDENCE_LABEL} for candidate in CANDIDATES]


def _source_rows() -> list[dict[str, str]]:
    return [
        {"source_id": source_id, **source}
        for source_id, source in sorted(SOURCES.items())
    ]


def _status_counts(rows: Sequence[Mapping[str, str]], group_key: str, status_key: str) -> list[dict[str, str]]:
    counter: dict[str, Counter[str]] = {}
    for row in rows:
        counter.setdefault(row[group_key], Counter())[row[status_key]] += 1
    statuses = sorted({row[status_key] for row in rows})
    output = []
    for candidate_id in CANDIDATE_IDS:
        counts = counter.get(candidate_id, Counter())
        record = {"candidate_id": candidate_id, "candidate_name": _candidate(candidate_id)["candidate_name"]}
        record.update({status: str(counts.get(status, 0)) for status in statuses})
        output.append(record)
    return output


def _report(
    metrics: Sequence[Mapping[str, str]],
    requirements: Sequence[Mapping[str, str]],
    scorecard: Sequence[Mapping[str, str]],
    metric_rows: Sequence[Mapping[str, str]],
    gate_rows: Sequence[Mapping[str, str]],
    model_rows: Sequence[Mapping[str, str]],
    scorecard_rows: Sequence[Mapping[str, str]],
    pareto_rows: Sequence[Mapping[str, str]],
    compute_rows: Sequence[Mapping[str, str]],
) -> str:
    metric_matrix = _matrix(metric_rows, "metric_id", "compatibility_status", ["metric_id", "metric_name"])
    requirement_matrix = _matrix(gate_rows, "requirement", "gate_status", ["requirement", "mandatory"])
    scorecard_matrix = _matrix(scorecard_rows, "criterion", "rating", ["criterion", "category", "priority_class"])
    model_matrix = _matrix(model_rows, "model_name", "compatibility_status", ["model_name"])

    return "\n\n".join(
        [
            "# E2 Fifth Dataset Selection",
            EVIDENCE_LABEL,
            "## Final Decision\n\n"
            f"{DECISION_TOKEN}\n\n"
            f"PRIMARY DATASET: {_candidate(PRIMARY_DATASET)['candidate_name']}\n\n"
            f"BACKUP DATASET: {_candidate(BACKUP_DATASET)['candidate_name']}\n\n"
            "Can the selected fifth dataset receive the complete applicable paper/local-reproduction evaluation BEFORE our original extensions?\n\nYES",
            "## Repository Precondition / Recovery State\n\n"
            "- branch: master\n"
            "- HEAD at recovery: 07e239ce1ede2807a40ea3064554febcf941e1db\n"
            "- Git state at recovery: clean; master...origin/master [ahead 5]\n"
            "- original_repo/EvaLLMasRecommender state: clean; main...origin/main\n"
            "- Interrupted-run artifact recovery: no extension_e2_* tracked or untracked artifacts found; no files retained; no incomplete extension E2 evidence; no unrelated changes existed.",
            "## Frozen E1 Contract Counts\n\n"
            f"- metric contract rows: {len(metrics)}\n"
            "- four-dataset coverage rows: 26\n"
            f"- fifth-dataset requirements: {len(requirements)}\n"
            "- inference-requirement rows: 8\n"
            f"- dataset-selection scorecard criteria: {len(scorecard)}",
            "## Authoritative Source Inventory\n\n"
            + _markdown_table(_source_rows(), ["source_id", "title", "url", "notes"]),
            "## Candidate Inventory\n\n"
            + _markdown_table(
                _candidate_summary_rows(),
                [
                    "candidate_id",
                    "candidate_name",
                    "exact_release",
                    "access_mechanism",
                    "academic_terms",
                    "dataset_size",
                    "sequential_classification",
                    "llm_metadata_classification",
                    "calibration_classification",
                    "scale_classification",
                    "selection_status",
                ],
            ),
            "## Access and Licensing Comparison\n\n"
            + _markdown_table(
                _candidate_summary_rows(),
                [
                    "candidate_name",
                    "academic_terms",
                    "paid_or_free",
                    "registration_or_manual_request",
                    "acquisition_reproducibility",
                    "source_ids",
                ],
            ),
            "## Scale Comparison\n\n"
            + _markdown_table(
                compute_rows,
                [
                    "candidate_name",
                    "scale_classification",
                    "raw_download_scale",
                    "official_subset_policy",
                    "compute_rationale",
                    "risk",
                ],
            ),
            "## Temporal / Sequential Comparison\n\n"
            + _markdown_table(
                _candidate_summary_rows(),
                ["candidate_name", "timestamps", "chronology", "history_depth", "sequential_classification"],
            ),
            "## Metadata / LLM Comparison\n\n"
            + _markdown_table(
                _candidate_summary_rows(),
                [
                    "candidate_name",
                    "item_title_name",
                    "categories_genres_tags",
                    "additional_metadata",
                    "llm_metadata_classification",
                    "calibration_classification",
                ],
            ),
            "## Complete 29-Row Metric Compatibility Summary\n\n"
            + _markdown_table(
                _status_counts(metric_rows, "candidate_id", "compatibility_status"),
                ["candidate_id", "candidate_name"]
                + sorted({row["compatibility_status"] for row in metric_rows}),
            )
            + "\n\n"
            + _markdown_table(metric_matrix, ["metric_id", "metric_name"] + CANDIDATE_IDS),
            "## Complete 23-Requirement Compatibility Summary\n\n"
            + _markdown_table(
                _status_counts(gate_rows, "candidate_id", "gate_status"),
                ["candidate_id", "candidate_name"] + sorted({row["gate_status"] for row in gate_rows}),
            )
            + "\n\n"
            + _markdown_table(requirement_matrix, ["requirement", "mandatory"] + CANDIDATE_IDS),
            "## Traditional-Model Compatibility\n\n"
            + _markdown_table(model_matrix, ["model_name"] + CANDIDATE_IDS),
            "## Paper-Specific Analysis Compatibility\n\n"
            "Ranking, reranking, history-length sensitivity, candidate-position bias, profile generation, hallucination, novelty, popularity, diversity and fairness are represented by the metric matrix above. Food.com supports all applicable paper analyses before local extensions, provided E3 generates Dataset 5 candidate pools, traditional model outputs, raw LLM outputs and parser manifests. Steam is the backup with the same Role A path but a stronger need to document review/recommendation-signal adaptation.",
            "## Calibration Compatibility\n\n"
            + _markdown_table(
                _candidate_summary_rows(),
                ["candidate_name", "categories_genres_tags", "calibration_classification", "role_b_assessment"],
            ),
            "## Pipeline-Aware Extension Compatibility\n\n"
            "Food.com and Steam support source top-100 rankings, multiple recommenders, Round-Robin, RRF, Candidate Recall@10/20/50/100, conditional and end-to-end HR/NDCG@5, oracle candidate-availability ceilings, and controlled positive injection after Role A. These extensions remain secondary and must be executed only after Dataset 5 paper/local-reproduction parity artifacts are frozen.",
            "## Full 25-Criterion Scorecard\n\n"
            + _markdown_table(scorecard_matrix, ["criterion", "category", "priority_class"] + CANDIDATE_IDS),
            "## Hard-Gate Results\n\n"
            + _markdown_table(
                [
                    {
                        "candidate_name": candidate["candidate_name"],
                        "result": (
                            "PASS"
                            if candidate["candidate_id"] in {"foodcom", "steam"}
                            else "FAIL"
                            if candidate["candidate_id"] == "bookcrossing"
                            else "PASS_WITH_DOCUMENTED_ADAPTATION"
                            if candidate["candidate_id"] == "goodreads"
                            else "UNKNOWN"
                        ),
                        "reason": candidate["role_a_assessment"],
                    }
                    for candidate in CANDIDATES
                ],
                ["candidate_name", "result", "reason"],
            ),
            "## Pareto Analysis\n\n"
            + _markdown_table(pareto_rows, ["candidate_name", "pareto_class", "rationale", "sensitivity_note"]),
            "## Sensitivity Analysis\n\n"
            "- Paper-framework parity emphasis: Food.com remains primary; Steam remains backup.\n"
            "- Sequential fidelity emphasis: Food.com and Steam remain strongest; BookCrossing is eliminated.\n"
            "- Compute emphasis: Food.com wins; BookCrossing is smaller but fails hard gates.\n"
            "- Metadata emphasis: Food.com, Steam and Goodreads are strongest; Food.com remains best because metadata is rich at moderate scale.\n"
            "- Extension compatibility emphasis: Food.com remains primary because recipe tags, ingredients and techniques are directly usable for calibration and candidate aggregation.\n"
            "- Access/licensing emphasis: Steam can become primary if the project prioritizes direct UCSD download over Food.com external-resource/manual intake.\n\n"
            "No single candidate strictly dominates every dimension. Food.com is the stable selection under all scientific priorities except an access-only preference for Steam.",
            "## Rejected Candidates and Exact Reasons\n\n"
            + _markdown_table(
                [
                    {
                        "candidate_name": candidate["candidate_name"],
                        "selection_status": candidate["selection_status"],
                        "reason": candidate["rejected_reason"],
                    }
                    for candidate in CANDIDATES
                    if candidate["candidate_id"] != PRIMARY_DATASET
                ],
                ["candidate_name", "selection_status", "reason"],
            ),
            "## Primary Dataset Release\n\n"
            f"{_candidate(PRIMARY_DATASET)['exact_release']}\n\n"
            "Use the official raw recipe and interaction tables plus official split files. Do not invent an arbitrary convenience subset in E2 or E3; if E3 discovers a compute issue, first evaluate documented official split artifacts and report the limitation.",
            "## Scientific Risks / Limitations of Primary\n\n"
            "- Exact raw user count and post-filter history-depth distribution must be verified after authorized E3 acquisition.\n"
            "- External-resource/manual intake must be documented with checksums and no raw data committed.\n"
            "- Repeated user-recipe interactions, if present, need a deterministic deduplication policy before split construction.\n"
            "- All paper/local metrics still require new Dataset 5 candidate pools, traditional model outputs and LLM/parser manifests; E2 selects data only.",
            "## Repository Postcondition\n\n"
            "- E2 evidence artifacts only.\n"
            "- No dataset downloaded.\n"
            "- No preprocessing, training, LLM inference or API call.\n"
            "- Frozen E1 artifacts and original_repo remain out of scope.",
        ]
    ) + "\n"


def build_artifacts() -> None:
    metrics = _read_csv(E1_METRIC_PATH)
    requirements = _read_csv(E1_REQUIREMENTS_PATH)
    scorecard = _read_csv(E1_SCORECARD_PATH)

    metric_rows = _metric_compatibility_rows(metrics)
    gate_rows = _hard_gate_rows(requirements)
    model_rows = _model_compatibility_rows()
    scorecard_rows = _scorecard_rows(scorecard)
    pareto_rows = _pareto_rows()
    compute_rows = _compute_rows()

    _write_csv(
        RESULTS_DIR / "extension_e2_dataset_candidates.csv",
        _candidate_summary_rows(),
        [
            "evidence_label",
            "candidate_id",
            "candidate_name",
            "exact_release",
            "access_mechanism",
            "academic_terms",
            "paid_or_free",
            "registration_or_manual_request",
            "dataset_size",
            "users",
            "items",
            "interactions",
            "timestamps",
            "chronology",
            "item_title_name",
            "categories_genres_tags",
            "additional_metadata",
            "interaction_values",
            "history_depth",
            "duplicate_behavior",
            "official_subsets",
            "raw_download_scale",
            "acquisition_reproducibility",
            "sequential_classification",
            "llm_metadata_classification",
            "calibration_classification",
            "scale_classification",
            "role_a_assessment",
            "role_b_assessment",
            "selection_status",
            "rejected_reason",
            "source_ids",
        ],
    )
    _write_csv(
        RESULTS_DIR / "extension_e2_metric_compatibility.csv",
        metric_rows,
        [
            "evidence_label",
            "candidate_id",
            "candidate_name",
            "metric_id",
            "family",
            "metric_name",
            "e1_requirement_classification",
            "compatibility_status",
            "rationale",
            "source_ids",
        ],
    )
    _write_csv(
        RESULTS_DIR / "extension_e2_model_compatibility.csv",
        model_rows,
        [
            "evidence_label",
            "candidate_id",
            "candidate_name",
            "model_name",
            "compatibility_status",
            "rationale",
            "source_ids",
        ],
    )
    _write_csv(
        RESULTS_DIR / "extension_e2_dataset_scorecard.csv",
        scorecard_rows,
        [
            "evidence_label",
            "candidate_id",
            "candidate_name",
            "criterion",
            "category",
            "e1_weight",
            "e1_mandatory_gate",
            "priority_class",
            "rating",
            "evidence",
            "source_ids",
        ],
    )
    _write_csv(
        RESULTS_DIR / "extension_e2_dataset_hard_gates.csv",
        gate_rows,
        [
            "evidence_label",
            "candidate_id",
            "candidate_name",
            "requirement",
            "mandatory",
            "requirement_class",
            "gate_status",
            "evidence",
            "source_ids",
        ],
    )
    _write_csv(
        RESULTS_DIR / "extension_e2_dataset_pareto.csv",
        pareto_rows,
        [
            "evidence_label",
            "candidate_id",
            "candidate_name",
            "pareto_class",
            "rationale",
            "sensitivity_note",
            "source_ids",
        ],
    )
    _write_csv(
        RESULTS_DIR / "extension_e2_dataset_compute_estimates.csv",
        compute_rows,
        [
            "evidence_label",
            "candidate_id",
            "candidate_name",
            "scale_classification",
            "raw_download_scale",
            "official_subset_policy",
            "compute_rationale",
            "risk",
            "source_ids",
        ],
    )
    _write_csv(
        RESULTS_DIR / "extension_e2_dataset_selection_decision.csv",
        _decision_rows(),
        [
            "evidence_label",
            "decision_token",
            "primary_dataset_id",
            "primary_dataset_name",
            "backup_dataset_id",
            "backup_dataset_name",
            "primary_release",
            "backup_release",
            "can_receive_complete_applicable_paper_local_reproduction_before_extensions",
            "selection_rationale",
            "scope_boundary",
        ],
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        _report(metrics, requirements, scorecard, metric_rows, gate_rows, model_rows, scorecard_rows, pareto_rows, compute_rows),
        encoding="utf-8",
    )


def validate_artifacts() -> None:
    metrics = _read_csv(E1_METRIC_PATH)
    requirements = _read_csv(E1_REQUIREMENTS_PATH)
    scorecard = _read_csv(E1_SCORECARD_PATH)
    metric_rows = _read_csv(RESULTS_DIR / "extension_e2_metric_compatibility.csv")
    gate_rows = _read_csv(RESULTS_DIR / "extension_e2_dataset_hard_gates.csv")
    scorecard_rows = _read_csv(RESULTS_DIR / "extension_e2_dataset_scorecard.csv")
    model_rows = _read_csv(RESULTS_DIR / "extension_e2_model_compatibility.csv")
    candidates = _read_csv(RESULTS_DIR / "extension_e2_dataset_candidates.csv")
    decision = _read_csv(RESULTS_DIR / "extension_e2_dataset_selection_decision.csv")

    if len(candidates) != len(CANDIDATE_IDS):
        raise ValueError("candidate count mismatch")
    if len(metric_rows) != len(metrics) * len(CANDIDATE_IDS):
        raise ValueError("metric compatibility row count mismatch")
    if len(gate_rows) != len(requirements) * len(CANDIDATE_IDS):
        raise ValueError("hard-gate row count mismatch")
    if len(scorecard_rows) != len(scorecard) * len(CANDIDATE_IDS):
        raise ValueError("scorecard row count mismatch")
    if len(model_rows) != len(MODEL_NAMES) * len(CANDIDATE_IDS):
        raise ValueError("model compatibility row count mismatch")
    if len(decision) != 1:
        raise ValueError("selection decision must have exactly one row")

    primary = decision[0]["primary_dataset_id"]
    backup = decision[0]["backup_dataset_id"]
    if primary == backup or primary != PRIMARY_DATASET or backup != BACKUP_DATASET:
        raise ValueError("primary/backup selection mismatch")

    for row in metric_rows:
        if row["compatibility_status"] not in COMPATIBILITY_VALUES:
            raise ValueError(f"invalid metric status: {row}")
        if not row["rationale"] or not row["source_ids"]:
            raise ValueError(f"missing metric rationale/provenance: {row}")
    for row in gate_rows:
        if row["gate_status"] not in GATE_VALUES:
            raise ValueError(f"invalid gate status: {row}")
        if not row["evidence"] or not row["source_ids"]:
            raise ValueError(f"missing gate evidence/provenance: {row}")
    for row in scorecard_rows:
        if row["rating"] not in RATING_VALUES:
            raise ValueError(f"invalid scorecard rating: {row}")
    for row in model_rows:
        if row["compatibility_status"] not in MODEL_VALUES:
            raise ValueError(f"invalid model status: {row}")

    primary_gate_rows = [row for row in gate_rows if row["candidate_id"] == PRIMARY_DATASET]
    failing_primary = [row for row in primary_gate_rows if row["gate_status"] in {"FAIL", "UNKNOWN"}]
    if failing_primary:
        raise ValueError(f"primary has failing mandatory gates: {failing_primary}")


def main() -> None:
    build_artifacts()
    validate_artifacts()


if __name__ == "__main__":
    main()
