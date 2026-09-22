from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.dataset_candidate_set_validation import (
    DATASET_STATISTICS_COLUMNS,
    POSITIVE_POSITIONS_COLUMNS,
    RANKING_CANDIDATE_SETS_COLUMNS,
    CandidateSetValidationError,
    DatasetContext,
    ListCandidateArtifact,
    RowCandidateArtifact,
    canonicalize_list_candidate_artifact,
    canonicalize_row_candidate_artifact,
    derive_positive_positions,
    load_dataset_context,
    parse_candidate_items,
    write_csv,
)


LOGGER = logging.getLogger("build_dataset_candidate_set_audit")
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORT_DIR = PROJECT_ROOT / "report"
DATASET_STATS_PATH = TABLES_DIR / "dataset_statistics.csv"
RANKING_CANDIDATES_PATH = TABLES_DIR / "ranking_candidate_sets.csv"
POSITIVE_POSITIONS_PATH = TABLES_DIR / "positive_positions.csv"
REPORT_PATH = REPORT_DIR / "dataset_audit.md"


@dataclass(frozen=True)
class RawArtifactFamily:
    dataset: str
    artifact_status: str
    artifact_origin: str
    interaction_path: str = ""
    title_mapping_path: str = ""
    num_users: str | int = ""
    num_items: str | int = ""
    num_interactions: str | int = ""
    filtering_protocol: str = ""
    split_protocol: str = ""
    paper_alignment: str = ""
    blocking_reason: str = ""
    notes: str = ""


@dataclass(frozen=True)
class NonCanonicalCandidateArtifact:
    path: str
    dataset: str
    source_plan: str
    reason: str
    notes: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical dataset and ranking candidate-set audit outputs."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Accepted for CLI consistency; Phase 2A explicitly enumerates verified local artifacts.",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")
    try:
        outputs = build_dataset_candidate_set_audit(PROJECT_ROOT)
        write_outputs(outputs)
        validate_written_outputs()
    except (CandidateSetValidationError, OSError, ValueError, pd.errors.ParserError) as exc:
        LOGGER.error("%s", exc)
        return 2

    candidate_sets = outputs["candidate_sets"]
    positions = outputs["positive_positions"]
    dataset_stats = outputs["dataset_statistics"]
    invalid_rows = int((candidate_sets["row_validation_status"] == "invalid").sum())
    position_invalid = int((positions["position_validation_status"] == "invalid").sum())
    print("Phase 2A dataset and candidate-set audit complete")
    print("offline_only=true")
    print("api_calls=false")
    print("downloads=false")
    print("model_pulls=false")
    print("llm_inference=false")
    print("recommender_training=false")
    print(f"dataset_statistics_rows={len(dataset_stats)}")
    print(f"canonical_candidate_rows={len(candidate_sets)}")
    print(f"invalid_candidate_rows={invalid_rows}")
    print(f"positive_position_rows={len(positions)}")
    print(f"invalid_positive_position_rows={position_invalid}")
    print(f"dataset_statistics={_relative(DATASET_STATS_PATH)}")
    print(f"ranking_candidate_sets={_relative(RANKING_CANDIDATES_PATH)}")
    print(f"positive_positions={_relative(POSITIVE_POSITIONS_PATH)}")
    print(f"report={_relative(REPORT_PATH)}")
    return 0


def build_dataset_candidate_set_audit(project_root: Path = PROJECT_ROOT) -> dict[str, pd.DataFrame | list[NonCanonicalCandidateArtifact]]:
    contexts = build_dataset_contexts(project_root)
    row_artifacts, list_artifacts = candidate_artifacts()

    frames: list[pd.DataFrame] = []
    artifact_metadata: list[dict[str, Any]] = []
    for artifact in row_artifacts:
        context = contexts.get(artifact.context_key)
        frame = canonicalize_row_candidate_artifact(artifact, project_root=project_root, context=context)
        frames.append(frame)
        artifact_metadata.append(_artifact_metadata(artifact, frame))

    for artifact in list_artifacts:
        context = contexts.get(artifact.context_key)
        frame = canonicalize_list_candidate_artifact(artifact, project_root=project_root, context=context)
        frames.append(frame)
        artifact_metadata.extend(_list_artifact_metadata(artifact, frame))

    candidate_sets = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=RANKING_CANDIDATE_SETS_COLUMNS)
    )
    candidate_sets = candidate_sets.sort_values(["candidate_set_id", "user_id"], kind="mergesort").reset_index(drop=True)
    positive_positions = derive_positive_positions(candidate_sets)
    dataset_statistics = build_dataset_statistics(contexts, candidate_sets, artifact_metadata, project_root)
    report = build_report(
        dataset_statistics=dataset_statistics,
        candidate_sets=candidate_sets,
        positive_positions=positive_positions,
        noncanonical_artifacts=noncanonical_candidate_artifacts(),
    )
    return {
        "dataset_statistics": dataset_statistics,
        "candidate_sets": candidate_sets,
        "positive_positions": positive_positions,
        "report_frame": pd.DataFrame([{"content": report}]),
        "noncanonical_artifacts": noncanonical_candidate_artifacts(),
    }


def write_outputs(outputs: dict[str, pd.DataFrame | list[NonCanonicalCandidateArtifact]]) -> None:
    write_csv(outputs["dataset_statistics"], DATASET_STATS_PATH, DATASET_STATISTICS_COLUMNS)  # type: ignore[arg-type]
    write_csv(outputs["candidate_sets"], RANKING_CANDIDATES_PATH, RANKING_CANDIDATE_SETS_COLUMNS)  # type: ignore[arg-type]
    write_csv(outputs["positive_positions"], POSITIVE_POSITIONS_PATH, POSITIVE_POSITIONS_COLUMNS)  # type: ignore[arg-type]
    report = str(outputs["report_frame"].iloc[0]["content"])  # type: ignore[index,union-attr]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def build_dataset_contexts(project_root: Path) -> dict[str, DatasetContext]:
    return {
        "amazon_beauty_plan_c": load_dataset_context(
            dataset="Amazon Beauty",
            interaction_path=project_root / "data/reconstructed/plan_c/amazon_beauty/interactions.csv",
            ground_truth_path=project_root / "data/reconstructed/plan_c/amazon_beauty/ground_truth.csv",
            catalog_path=project_root / "data/reconstructed/plan_c/amazon_beauty/catalog.csv",
            datamaps_path=project_root / "data/reconstructed/plan_c/amazon_beauty/datamaps.json",
            negative_samples_path=project_root / "data/reconstructed/plan_c/amazon_beauty/negative_samples.txt",
            raw_paths=(
                project_root / "data/external/plan_c/reviews_Beauty_5.json.gz",
                project_root / "data/external/plan_c/meta_Beauty.json.gz",
            ),
            artifact_origin="Plan C public raw reconstruction",
            artifact_status="reconstructed",
            filtering_protocol="minimum 2 interactions per user from public raw 5-core source",
            split_protocol="chronological leave-one-out",
            candidate_protocol="1 positive + 99 deterministic negatives from interaction_items",
            notes="Generated from public raw Amazon Beauty files; not official paper processed artifact.",
        ),
        "amazon_sports_plan_c": load_dataset_context(
            dataset="Amazon Sports",
            interaction_path=project_root / "data/reconstructed/plan_c/amazon_sports/interactions.csv",
            ground_truth_path=project_root / "data/reconstructed/plan_c/amazon_sports/ground_truth.csv",
            catalog_path=project_root / "data/reconstructed/plan_c/amazon_sports/catalog.csv",
            datamaps_path=project_root / "data/reconstructed/plan_c/amazon_sports/datamaps.json",
            negative_samples_path=project_root / "data/reconstructed/plan_c/amazon_sports/negative_samples.txt",
            raw_paths=(
                project_root / "data/external/plan_c/reviews_Sports_and_Outdoors_5.json.gz",
                project_root / "data/external/plan_c/meta_Sports_and_Outdoors.json.gz",
            ),
            artifact_origin="Plan C public raw reconstruction",
            artifact_status="reconstructed",
            filtering_protocol="minimum 2 interactions per user from public raw 5-core source",
            split_protocol="chronological leave-one-out",
            candidate_protocol="1 positive + 99 deterministic negatives from interaction_items",
            notes="Generated from public raw Amazon Sports files; not official paper processed artifact.",
        ),
        "ml1m_plan_c": load_dataset_context(
            dataset="ML-1M",
            interaction_path=project_root / "data/reconstructed/plan_c/ml1m/interactions.csv",
            ground_truth_path=project_root / "data/reconstructed/plan_c/ml1m/ground_truth.csv",
            catalog_path=project_root / "data/reconstructed/plan_c/ml1m/catalog.csv",
            datamaps_path=project_root / "data/reconstructed/plan_c/ml1m/datamaps.json",
            negative_samples_path=project_root / "data/reconstructed/plan_c/ml1m/negative_samples.txt",
            raw_paths=(project_root / "data/external/plan_c/ml-1m.zip",),
            artifact_origin="Plan C public raw reconstruction",
            artifact_status="reconstructed",
            filtering_protocol="minimum 2 interactions per user from MovieLens-1M",
            split_protocol="chronological leave-one-out",
            candidate_protocol="1 positive + 99 deterministic negatives from interaction_items",
            notes="Generated from staged MovieLens-1M archive; not verified as exact paper candidate file.",
        ),
        "ml1m_processed": load_dataset_context(
            dataset="ML-1M",
            interaction_path=project_root / "data/processed/ml1m/interactions.csv",
            ground_truth_path=project_root / "data/processed/ml1m/ground_truth.csv",
            catalog_path=project_root / "data/processed/ml1m/catalog.csv",
            raw_paths=(
                project_root / "data/raw/ml1m/ratings.dat",
                project_root / "data/raw/ml1m/movies.dat",
                project_root / "data/raw/ml1m/users.dat",
            ),
            artifact_origin="Plan B local processed MovieLens-1M",
            artifact_status="processed",
            filtering_protocol="minimum 2 interactions per user from MovieLens-1M",
            split_protocol="chronological leave-one-out",
            candidate_protocol="1 positive + 99 deterministic negatives from catalog",
            notes="Processed local ML-1M files; candidate protocol is adapted 100-candidate evaluation.",
        ),
        "lastfm_plan_c": load_dataset_context(
            dataset="LastFM",
            interaction_path=project_root / "data/reconstructed/plan_c/lastfm/interactions.csv",
            catalog_path=project_root / "data/reconstructed/plan_c/lastfm/catalog.csv",
            datamaps_path=project_root / "data/reconstructed/plan_c/lastfm/datamaps.json",
            raw_paths=(project_root / "data/external/plan_c/hetrec2011-lastfm-2k.zip",),
            artifact_origin="Plan C public raw reconstruction",
            artifact_status="extension_only",
            filtering_protocol="observed interactions and catalog only",
            split_protocol="not available; no chronological timestamp split",
            candidate_protocol="no validated ranking candidates",
            blocking_reason="No ground_truth.csv, candidates.csv, sequential_data.txt, or negative_samples.txt were written for LastFM.",
            notes="LastFM is not usable for current ranking candidate-set validation.",
        ),
    }


def candidate_artifacts() -> tuple[list[RowCandidateArtifact], list[ListCandidateArtifact]]:
    row_artifacts = [
        RowCandidateArtifact(
            candidate_set_id="plan_b_ml1m_processed_100_candidate_pool",
            source_path=Path("data/processed/ml1m/candidates.csv"),
            source_plan="Plan B",
            dataset="ML-1M",
            context_key="ml1m_processed",
            expected_candidate_count=100,
            paper_alignment="adapted_100_candidate_pool",
            row_format="row_per_candidate",
        ),
        RowCandidateArtifact(
            candidate_set_id="plan_c_ml1m_reconstructed_100_candidate_pool",
            source_path=Path("data/reconstructed/plan_c/ml1m/candidates.csv"),
            source_plan="Plan C",
            dataset="ML-1M",
            context_key="ml1m_plan_c",
            expected_candidate_count=100,
            paper_alignment="adapted_100_candidate_pool",
            row_format="row_per_candidate",
        ),
        RowCandidateArtifact(
            candidate_set_id="plan_c_amazon_beauty_reconstructed_100_candidate_pool",
            source_path=Path("data/reconstructed/plan_c/amazon_beauty/candidates.csv"),
            source_plan="Plan C",
            dataset="Amazon Beauty",
            context_key="amazon_beauty_plan_c",
            expected_candidate_count=100,
            paper_alignment="adapted_100_candidate_pool",
            row_format="row_per_candidate",
        ),
        RowCandidateArtifact(
            candidate_set_id="plan_c_amazon_sports_reconstructed_100_candidate_pool",
            source_path=Path("data/reconstructed/plan_c/amazon_sports/candidates.csv"),
            source_plan="Plan C",
            dataset="Amazon Sports",
            context_key="amazon_sports_plan_c",
            expected_candidate_count=100,
            paper_alignment="adapted_100_candidate_pool",
            row_format="row_per_candidate",
        ),
        *_plan_e_artifacts(),
    ]
    list_artifacts = [
        ListCandidateArtifact(
            candidate_set_id_prefix="plan_d_qwen2_ml1m_ranking_diagnostic_openp5_base",
            source_path=Path("results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_manifest.csv"),
            source_plan="Plan D",
            dataset="ML-1M",
            context_key="ml1m_plan_c",
            expected_candidate_count=20,
            paper_alignment="diagnostic_20_candidate_pool",
            row_format="row_per_user_json_list",
        ),
        ListCandidateArtifact(
            candidate_set_id_prefix="plan_d_qwen2_ml1m_candidate_position_bias",
            source_path=Path("results/tables/plan_d_qwen2_ml1m_candidate_position_bias_manifest.csv"),
            source_plan="Plan D",
            dataset="ML-1M",
            context_key="ml1m_plan_c",
            expected_candidate_count=20,
            paper_alignment="diagnostic_20_candidate_position_variant",
            row_format="row_per_user_json_list",
            stored_position_column="variant_positive_position",
            candidate_set_suffix_column="variant",
        ),
    ]
    return row_artifacts, list_artifacts


def _plan_e_artifacts() -> list[RowCandidateArtifact]:
    specs = [
        ("plan_e_ml1m_1000user_ranking_manifest", "ML-1M", "ml1m_plan_c", "results/tables/plan_e_ml1m_1000user_ranking_manifest.csv"),
        ("plan_e_ml1m_1000user_reranking_manifest", "ML-1M", "ml1m_plan_c", "results/tables/plan_e_ml1m_1000user_reranking_manifest.csv"),
        ("plan_e_beauty_1000user_ranking_manifest", "Amazon Beauty", "amazon_beauty_plan_c", "results/tables/plan_e_beauty_1000user_ranking_manifest.csv"),
        ("plan_e_beauty_1000user_reranking_manifest", "Amazon Beauty", "amazon_beauty_plan_c", "results/tables/plan_e_beauty_1000user_reranking_manifest.csv"),
        ("plan_e_sports_1000user_ranking_manifest", "Amazon Sports", "amazon_sports_plan_c", "results/tables/plan_e_sports_1000user_ranking_manifest.csv"),
        ("plan_e_sports_1000user_reranking_manifest", "Amazon Sports", "amazon_sports_plan_c", "results/tables/plan_e_sports_1000user_reranking_manifest.csv"),
    ]
    return [
        RowCandidateArtifact(
            candidate_set_id=candidate_set_id,
            source_path=Path(path),
            source_plan="Plan E2",
            dataset=dataset,
            context_key=context_key,
            expected_candidate_count=20,
            paper_alignment="paper_like_reconstructed_20_candidate_pool",
            row_format="row_per_candidate_manifest",
            item_column="item_id",
            positive_column="is_positive",
        )
        for candidate_set_id, dataset, context_key, path in specs
    ]


def noncanonical_candidate_artifacts() -> list[NonCanonicalCandidateArtifact]:
    return [
        NonCanonicalCandidateArtifact(
            path="results/tables/plan_d_llm_ml1m_reranking_manifest.csv",
            dataset="ML-1M",
            source_plan="Plan D",
            reason="not_canonicalized_no_candidate_items_column",
            notes="Contains candidate order signatures and positive item IDs but not candidate item lists.",
        ),
        NonCanonicalCandidateArtifact(
            path="results/tables/plan_d_qwen2_ml1m_strict_prompt_manifest.csv",
            dataset="ML-1M",
            source_plan="Plan D",
            reason="not_canonicalized_no_candidate_items_column",
            notes="Contains signatures for diagnostic prompt variants but not candidate item lists.",
        ),
        NonCanonicalCandidateArtifact(
            path="results/tables/plan_d_qwen_vs_mistral_ml1m_manifest.csv",
            dataset="ML-1M",
            source_plan="Plan D",
            reason="not_canonicalized_no_candidate_items_column",
            notes="Contains signatures for model comparison diagnostics but not candidate item lists.",
        ),
        NonCanonicalCandidateArtifact(
            path="results/tables/plan_d_llm_ml1m_history_length_manifest.csv",
            dataset="ML-1M",
            source_plan="Plan D",
            reason="not_canonicalized_no_candidate_items_column",
            notes="Contains signatures for history-length diagnostics but not candidate item lists.",
        ),
        NonCanonicalCandidateArtifact(
            path="original_repo/EvaLLMasRecommender/llm_results/beauty.zip",
            dataset="Amazon Beauty",
            source_plan="upstream original_repo",
            reason="not_canonicalized_embedded_llm_outputs_without_positive_item_id",
            notes="JSON records contain cand_list and target title, but no authoritative positive item ID field in the current canonical schema.",
        ),
        NonCanonicalCandidateArtifact(
            path="original_repo/EvaLLMasRecommender/llm_results/sports.zip",
            dataset="Amazon Sports",
            source_plan="upstream original_repo",
            reason="not_canonicalized_embedded_llm_outputs_without_positive_item_id",
            notes="JSON records contain cand_list and target title, but no authoritative positive item ID field in the current canonical schema.",
        ),
    ]


def build_dataset_statistics(
    contexts: dict[str, DatasetContext],
    candidate_sets: pd.DataFrame,
    artifact_metadata: list[dict[str, Any]],
    project_root: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(_raw_artifact_rows(project_root))
    rows.append(_context_stats_row("LastFM", contexts["lastfm_plan_c"], "", "", "extension_only_no_candidate_set", 0, "not_aligned_blocked"))
    for metadata in artifact_metadata:
        context = contexts[metadata["context_key"]]
        rows.append(
            _context_stats_row(
                metadata["dataset"],
                context,
                metadata["candidate_set_id"],
                metadata["source_path"],
                metadata["artifact_status"],
                metadata["candidate_count"],
                metadata["paper_alignment"],
                num_evaluated_users=metadata["num_evaluated_users"],
                title_mapping_coverage=_title_coverage_for_subset(candidate_sets, metadata["candidate_set_id"], context),
                notes=metadata["notes"],
            )
        )
    rows.extend(_upstream_artifact_rows())
    return pd.DataFrame(rows, columns=DATASET_STATISTICS_COLUMNS).sort_values(
        ["dataset", "artifact_status", "candidate_manifest_path"],
        kind="mergesort",
    )


def build_report(
    *,
    dataset_statistics: pd.DataFrame,
    candidate_sets: pd.DataFrame,
    positive_positions: pd.DataFrame,
    noncanonical_artifacts: list[NonCanonicalCandidateArtifact],
) -> str:
    total_rows = len(candidate_sets)
    valid_rows = int((candidate_sets["row_validation_status"] == "valid").sum())
    warning_rows = int((candidate_sets["row_validation_status"] == "valid_with_warnings").sum())
    invalid_rows = int((candidate_sets["row_validation_status"] == "invalid").sum())
    duplicate_rows = int((candidate_sets["candidate_items_unique"] == "false").sum())
    missing_positive_rows = int((candidate_sets["positive_in_candidates"] == "false").sum())
    position_invalid = int((positive_positions["position_validation_status"] == "invalid").sum())
    mapping_failures = int(
        (
            (candidate_sets["user_mapping_valid"] == "false")
            | (candidate_sets["positive_item_mapping_valid"] == "false")
            | (candidate_sets["all_candidate_items_mapping_valid"] == "false")
        ).sum()
    )
    title_failures = int((candidate_sets["all_candidate_titles_available"] == "false").sum())
    gt_mismatches = int((candidate_sets["ground_truth_match"] == "false").sum())
    history_failures = int((candidate_sets["negative_history_exclusion_valid"] == "false").sum())
    unverifiable = int(
        candidate_sets[
            [
                "user_mapping_valid",
                "positive_item_mapping_valid",
                "all_candidate_items_mapping_valid",
                "all_candidate_titles_available",
                "ground_truth_match",
                "negative_history_exclusion_valid",
            ]
        ].eq("not_verifiable_from_current_artifacts").sum().sum()
    )
    candidate_inventory = _candidate_inventory_table(candidate_sets)
    position_summary = _position_summary_table(positive_positions)
    dataset_summary = _dataset_summary_table(dataset_statistics)
    noncanonical_table = _noncanonical_table(noncanonical_artifacts)

    return f"""# Dataset and Candidate-Set Audit

## 1. Scope and provenance

This Phase 2A audit builds an offline canonical view of dataset and ranking candidate-set artifacts already present in the repository. It uses only local files and does not call APIs, download data, pull models, invoke Ollama, train recommenders, or run LLM inference.

The outputs are:

- `results/tables/dataset_statistics.csv`
- `results/tables/ranking_candidate_sets.csv`
- `results/tables/positive_positions.csv`
- `report/dataset_audit.md`

The audit preserves candidate order and item IDs as found in the source artifacts. It does not infer or synthesize missing candidates, titles, mappings, positives, or statistics.

## 2. Executive summary

- Canonical candidate rows generated: {total_rows}
- Valid rows: {valid_rows}
- Valid with title warnings: {warning_rows}
- Invalid rows: {invalid_rows}
- Positive-position rows generated: {len(positive_positions)}
- Position validation failures: {position_invalid}
- Candidate-set families canonicalized: {candidate_sets["candidate_set_id"].nunique()}
- Dataset-statistics rows: {len(dataset_statistics)}

Current paper-like ranking support is strongest for the Plan E2 20-candidate manifests for Amazon Beauty, Amazon Sports, and ML-1M. These are reconstructed paper-like artifacts, not exact paper candidate files.

## 3. Canonical terminology

- original artifact: an author-provided artifact from the original repository or paper release whose exact provenance is preserved.
- reconstructed artifact: an artifact rebuilt from public raw data by local project code.
- derived artifact: an artifact derived from another local project artifact, such as Plan E2 manifests built from Plan C candidates.
- paper-like candidate set: a reconstructed or derived candidate set matching a paper-level shape, such as 1 positive plus 19 negatives, without claiming exact original sampling.
- adapted candidate set: a valid local candidate set with a different protocol from the paper, such as 1 positive plus 99 negatives.
- diagnostic candidate set: a small local diagnostic set prepared for debugging prompts or sensitivity checks.
- candidate count: the number of candidate item IDs present for one user in one candidate-set row.
- positive position: the index of the positive item inside the candidate list.
- zero-based position: a position where the first candidate is 0.
- one-based position: a position where the first candidate is 1.

## 4. Dataset-by-dataset audit

### Beauty

Amazon Beauty has staged public raw artifacts and a Plan C reconstructed dataset with interactions, ground truth, catalog, datamaps, negative samples, and 100-candidate rows. Plan E2 provides ranking and reranking 20-candidate manifests for 1000 users each. The upstream original repository also contains Beauty LLM output archives, but those archives are not promoted to canonical candidate sets because the embedded records expose target titles rather than authoritative positive item IDs.

### Sports

Amazon Sports has staged public raw artifacts and a Plan C reconstructed dataset with interactions, ground truth, catalog, datamaps, negative samples, and 100-candidate rows. Plan E2 provides ranking and reranking 20-candidate manifests for 1000 users each. The upstream Sports LLM output archive is recorded as evidence but is not canonicalized as a candidate set for the same reason as Beauty.

### ML-1M

ML-1M has local raw files, a Plan B processed artifact family, a Plan C reconstructed artifact family, 100-candidate files in both families, and Plan E2 20-candidate ranking and reranking manifests. Plan D also contains parseable 20-candidate diagnostic candidate lists for Qwen2 ranking and candidate-position bias checks. Other Plan D manifest files contain only signatures and are not canonicalized.

### LastFM

LastFM has a staged raw archive and a Plan C extension-only reconstructed interaction/catalog family. It does not have current ground truth, negative samples, sequential data, or candidate sets, so it is not usable for ranking candidate-set validation in this phase.

### additional datasets

No additional dataset with parseable ranking candidate artifacts was found beyond Amazon Beauty, Amazon Sports, ML-1M, and LastFM.

## 5. Candidate-set inventory

{candidate_inventory}

Non-canonical or unparseable candidate-like artifacts:

{noncanonical_table}

## 6. Twenty-candidate versus one-hundred-candidate protocols

The 20-candidate protocol appears in Plan E2 manifests and selected Plan D diagnostics. Plan E2 uses 1 positive plus 19 negatives for 1000 users per dataset-task pair. Plan D 20-candidate files are diagnostic ML-1M prompt or position-bias artifacts, not authoritative paper reproduction artifacts.

The 100-candidate protocol appears in Plan C reconstructed candidate files and the Plan B processed ML-1M candidate file. These use 1 positive plus 99 negatives and are adapted or reconstructed evaluation sources, not 20-candidate paper-like files.

No other parseable canonical candidate size was found in the canonicalized rows.

## 7. Validation findings

- Valid rows: {valid_rows}
- Valid with warnings: {warning_rows}
- Invalid rows: {invalid_rows}
- Rows with duplicate candidates: {duplicate_rows}
- Rows with missing positive item in candidates: {missing_positive_rows}
- Rows with positive-position mismatches: {position_invalid}
- Rows with mapping failures: {mapping_failures}
- Rows with incomplete title coverage: {title_failures}
- Rows with ground-truth mismatches: {gt_mismatches}
- Rows with negative-history leakage: {history_failures}
- Unverifiable mapping/title/ground-truth/history checks: {unverifiable}

Positive-position summary:

{position_summary}

## 8. Plan C / Plan D / Plan E consistency

Plan C candidate files remain the authoritative reconstructed 100-candidate sources for Amazon Beauty, Amazon Sports, and ML-1M. They are internally compatible with their corresponding reconstructed interactions, ground truth, catalogs, datamaps, and negative-sample files.

Plan D parseable candidate manifests are diagnostic and small-scale. They are useful for local LLM and position-bias diagnostics, but should not be treated as authoritative paper-like candidate pools for future baseline comparisons.

Plan E2 manifests are derived from the Plan C candidate sources and are the current paper-like 20-candidate artifacts to use for later non-API paper-like ranking experiments. They are compatible with Plan C mappings and ground truth, but remain reconstructed rather than exact original paper candidate files.

Earlier Beauty/Sports configured audit files that pointed only to upstream output zips are superseded for candidate-set validation by the Plan C and Plan E2 artifacts listed here.

## 9. Paper alignment

Aligned in shape:

- Plan E2 Beauty, Sports, and ML-1M ranking/reranking manifests use 20 candidates with 1 positive and 19 negatives.

Not exact paper reproduction:

- The Plan E2 manifests are reconstructed from local Plan C/processed candidate sources.
- Official original candidate files and exact user/negative sampling evidence remain unavailable.
- Upstream LLM output archives are not canonical candidate manifests in this phase.

Adapted or diagnostic:

- Plan C and Plan B 100-candidate files are adapted 1+99 candidate pools.
- Plan D manifests are diagnostic 20-candidate prompt/sensitivity artifacts.

## 10. Blocking issues

- Official original paper candidate files are still missing.
- LastFM lacks ranking ground truth and candidates.
- Upstream Beauty/Sports LLM archives contain embedded candidate lists but do not expose a directly validated positive item ID field for canonical candidate-set rows.
- Several Plan D manifests contain only candidate-order signatures, not candidate lists.
- Any remaining title coverage gaps require title-map reconciliation rather than silent filling.

## 11. Canonical artifacts to use going forward

Use these for paper-like 20-candidate future phases:

- `results/tables/plan_e_beauty_1000user_ranking_manifest.csv`
- `results/tables/plan_e_beauty_1000user_reranking_manifest.csv`
- `results/tables/plan_e_sports_1000user_ranking_manifest.csv`
- `results/tables/plan_e_sports_1000user_reranking_manifest.csv`
- `results/tables/plan_e_ml1m_1000user_ranking_manifest.csv`
- `results/tables/plan_e_ml1m_1000user_reranking_manifest.csv`

Use these for adapted 100-candidate reconstructed validation:

- `data/reconstructed/plan_c/amazon_beauty/candidates.csv`
- `data/reconstructed/plan_c/amazon_sports/candidates.csv`
- `data/reconstructed/plan_c/ml1m/candidates.csv`
- `data/processed/ml1m/candidates.csv`

Use these for diagnostics only:

- `results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_manifest.csv`
- `results/tables/plan_d_qwen2_ml1m_candidate_position_bias_manifest.csv`

## 12. Artifacts that should no longer be treated as authoritative

Do not treat these as authoritative canonical candidate sets:

- `original_repo/EvaLLMasRecommender/llm_results/beauty.zip`
- `original_repo/EvaLLMasRecommender/llm_results/sports.zip`
- Plan D manifests that contain candidate-order signatures without candidate item lists
- Older Beauty/Sports dataset audit summaries that only report upstream archive availability

These artifacts should not be deleted; they remain useful provenance or diagnostic evidence.

## 13. Tests and validation commands

Required output validation should check exact headers, non-empty CSV outputs, JSON-decodable `candidate_items`, unique `(candidate_set_id, user_id)` keys, candidate counts matching decoded lists, and positive-position rows derived from `ranking_candidate_sets.csv`.

The implementation used by this report is offline and based on:

- `src/analysis/dataset_candidate_set_validation.py`
- `src/experiments/build_dataset_candidate_set_audit.py`
- `tests/test_dataset_candidate_set_validation.py`

## 14. Recommended next task

Perform title-map reconciliation planning for the canonical candidate sets, using only existing catalogs and without generating new model outputs or running baselines.
"""


def validate_written_outputs() -> None:
    dataset_stats = pd.read_csv(DATASET_STATS_PATH, dtype=str).fillna("")
    candidate_sets = pd.read_csv(RANKING_CANDIDATES_PATH, dtype=str).fillna("")
    positions = pd.read_csv(POSITIVE_POSITIONS_PATH, dtype=str).fillna("")

    _require_columns(dataset_stats, DATASET_STATISTICS_COLUMNS, DATASET_STATS_PATH)
    _require_columns(candidate_sets, RANKING_CANDIDATE_SETS_COLUMNS, RANKING_CANDIDATES_PATH)
    _require_columns(positions, POSITIVE_POSITIONS_COLUMNS, POSITIVE_POSITIONS_PATH)
    if dataset_stats.empty:
        raise CandidateSetValidationError("dataset_statistics.csv is empty")
    if candidate_sets.empty:
        raise CandidateSetValidationError("ranking_candidate_sets.csv is empty")
    if positions.empty:
        raise CandidateSetValidationError("positive_positions.csv is empty")

    keys = candidate_sets[["candidate_set_id", "user_id"]]
    if keys.duplicated().any():
        duplicated = keys.loc[keys.duplicated()].head(3).to_dict("records")
        raise CandidateSetValidationError(f"ranking_candidate_sets.csv has duplicate candidate_set_id/user_id keys: {duplicated}")
    for _, row in candidate_sets.iterrows():
        items = parse_candidate_items(row["candidate_items"])
        if int(row["candidate_count"]) != len(items):
            raise CandidateSetValidationError("candidate_count does not match decoded candidate_items")
    allowed_tri = {"true", "false", "not_verifiable_from_current_artifacts"}
    for column in [
        "candidate_items_unique",
        "positive_in_candidates",
        "user_mapping_valid",
        "positive_item_mapping_valid",
        "all_candidate_items_mapping_valid",
        "all_candidate_titles_available",
        "ground_truth_match",
        "negative_history_exclusion_valid",
    ]:
        invalid = set(candidate_sets[column]) - allowed_tri
        if invalid:
            raise CandidateSetValidationError(f"{column} has invalid status values: {sorted(invalid)}")
    if len(positions) != len(candidate_sets):
        raise CandidateSetValidationError("positive_positions.csv row count differs from ranking_candidate_sets.csv")
    position_keys = set(map(tuple, positions[["candidate_set_id", "user_id"]].itertuples(index=False, name=None)))
    candidate_keys = set(map(tuple, keys.itertuples(index=False, name=None)))
    if position_keys != candidate_keys:
        raise CandidateSetValidationError("positive_positions.csv keys do not match ranking_candidate_sets.csv")


def _artifact_metadata(artifact: RowCandidateArtifact, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "candidate_set_id": artifact.candidate_set_id,
        "dataset": artifact.dataset,
        "context_key": artifact.context_key,
        "source_path": artifact.source_path.as_posix(),
        "source_plan": artifact.source_plan,
        "artifact_status": _artifact_status(artifact.source_plan, artifact.paper_alignment),
        "candidate_count": artifact.expected_candidate_count,
        "paper_alignment": artifact.paper_alignment,
        "num_evaluated_users": frame["user_id"].nunique(),
        "notes": f"{artifact.row_format}; source_plan={artifact.source_plan}",
    }


def _list_artifact_metadata(artifact: ListCandidateArtifact, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for candidate_set_id, group in frame.groupby("candidate_set_id", sort=True):
        rows.append(
            {
                "candidate_set_id": candidate_set_id,
                "dataset": artifact.dataset,
                "context_key": artifact.context_key,
                "source_path": artifact.source_path.as_posix(),
                "source_plan": artifact.source_plan,
                "artifact_status": "diagnostic_manifest",
                "candidate_count": artifact.expected_candidate_count,
                "paper_alignment": artifact.paper_alignment,
                "num_evaluated_users": group["user_id"].nunique(),
                "notes": f"{artifact.row_format}; source_plan={artifact.source_plan}",
            }
        )
    return rows


def _context_stats_row(
    dataset: str,
    context: DatasetContext,
    candidate_set_id: str,
    candidate_path: str,
    artifact_status: str,
    candidate_count: int | str,
    paper_alignment: str,
    *,
    num_evaluated_users: int | str = "",
    title_mapping_coverage: str = "",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "artifact_status": artifact_status,
        "artifact_origin": context.artifact_origin,
        "interaction_path": _maybe_relative(context.interaction_path),
        "user_mapping_path": _maybe_relative(context.datamaps_path),
        "item_mapping_path": _maybe_relative(context.datamaps_path),
        "title_mapping_path": _maybe_relative(context.catalog_path),
        "train_path": _maybe_relative(context.interaction_path),
        "validation_path": "",
        "test_path": _maybe_relative(context.ground_truth_path),
        "ground_truth_path": _maybe_relative(context.ground_truth_path),
        "negative_samples_path": _maybe_relative(context.negative_samples_path),
        "candidate_manifest_path": candidate_path,
        "num_users": context.num_users,
        "num_items": context.num_items,
        "num_interactions": context.num_interactions,
        "num_eligible_users": context.num_eligible_users,
        "num_evaluated_users": num_evaluated_users,
        "filtering_protocol": context.filtering_protocol,
        "split_protocol": context.split_protocol,
        "candidate_protocol": _candidate_protocol(candidate_count),
        "candidate_count": candidate_count,
        "title_mapping_coverage": title_mapping_coverage,
        "paper_alignment": paper_alignment,
        "blocking_reason": context.blocking_reason,
        "notes": notes or context.notes,
    }


def _raw_artifact_rows(project_root: Path) -> list[dict[str, Any]]:
    stats = _plan_c_stats(project_root)
    raw_families = [
        RawArtifactFamily(
            dataset="Amazon Beauty",
            artifact_status="raw_staged",
            artifact_origin="data/external/plan_c public raw files",
            interaction_path="data/external/plan_c/reviews_Beauty_5.json.gz",
            title_mapping_path="data/external/plan_c/meta_Beauty.json.gz",
            num_users=stats.get("amazon_beauty", {}).get("raw_users", ""),
            num_items=stats.get("amazon_beauty", {}).get("raw_items", ""),
            num_interactions=stats.get("amazon_beauty", {}).get("raw_interactions", ""),
            filtering_protocol="not applied in raw artifact",
            split_protocol="not applied in raw artifact",
            paper_alignment="raw_source_for_reconstruction_not_exact_processed_artifact",
            notes="Raw staged source used by Plan C reconstruction.",
        ),
        RawArtifactFamily(
            dataset="Amazon Sports",
            artifact_status="raw_staged",
            artifact_origin="data/external/plan_c public raw files",
            interaction_path="data/external/plan_c/reviews_Sports_and_Outdoors_5.json.gz",
            title_mapping_path="data/external/plan_c/meta_Sports_and_Outdoors.json.gz",
            num_users=stats.get("amazon_sports", {}).get("raw_users", ""),
            num_items=stats.get("amazon_sports", {}).get("raw_items", ""),
            num_interactions=stats.get("amazon_sports", {}).get("raw_interactions", ""),
            filtering_protocol="not applied in raw artifact",
            split_protocol="not applied in raw artifact",
            paper_alignment="raw_source_for_reconstruction_not_exact_processed_artifact",
            notes="Raw staged source used by Plan C reconstruction.",
        ),
        RawArtifactFamily(
            dataset="ML-1M",
            artifact_status="raw_staged",
            artifact_origin="data/raw/ml1m and data/external/plan_c/ml-1m.zip",
            interaction_path="data/raw/ml1m/ratings.dat",
            title_mapping_path="data/raw/ml1m/movies.dat",
            num_users=stats.get("ml1m", {}).get("raw_users", ""),
            num_items=stats.get("ml1m", {}).get("raw_items", ""),
            num_interactions=stats.get("ml1m", {}).get("raw_interactions", ""),
            filtering_protocol="not applied in raw artifact",
            split_protocol="not applied in raw artifact",
            paper_alignment="raw_source_for_reconstruction_not_exact_processed_artifact",
            notes="Local raw ML-1M files and staged Plan C archive are present.",
        ),
        RawArtifactFamily(
            dataset="LastFM",
            artifact_status="raw_staged",
            artifact_origin="data/external/plan_c/hetrec2011-lastfm-2k.zip",
            interaction_path="data/external/plan_c/hetrec2011-lastfm-2k.zip",
            num_users=stats.get("lastfm", {}).get("raw_users", ""),
            num_items=stats.get("lastfm", {}).get("raw_items", ""),
            num_interactions=stats.get("lastfm", {}).get("raw_interactions", ""),
            filtering_protocol="not applied in raw artifact",
            split_protocol="not applied in raw artifact",
            paper_alignment="extension_only_raw_source",
            blocking_reason="No validated ranking split or candidate set in current artifacts.",
            notes="Raw LastFM source exists but current project has no ranking candidate artifact.",
        ),
    ]
    return [
        {
            "dataset": row.dataset,
            "artifact_status": row.artifact_status,
            "artifact_origin": row.artifact_origin,
            "interaction_path": row.interaction_path,
            "user_mapping_path": "",
            "item_mapping_path": "",
            "title_mapping_path": row.title_mapping_path,
            "train_path": "",
            "validation_path": "",
            "test_path": "",
            "ground_truth_path": "",
            "negative_samples_path": "",
            "candidate_manifest_path": "",
            "num_users": row.num_users,
            "num_items": row.num_items,
            "num_interactions": row.num_interactions,
            "num_eligible_users": "",
            "num_evaluated_users": "",
            "filtering_protocol": row.filtering_protocol,
            "split_protocol": row.split_protocol,
            "candidate_protocol": "not_applicable",
            "candidate_count": "",
            "title_mapping_coverage": "",
            "paper_alignment": row.paper_alignment,
            "blocking_reason": row.blocking_reason,
            "notes": row.notes,
        }
        for row in raw_families
    ]


def _upstream_artifact_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset, path in [
        ("Amazon Beauty", "original_repo/EvaLLMasRecommender/llm_results/beauty.zip"),
        ("Amazon Sports", "original_repo/EvaLLMasRecommender/llm_results/sports.zip"),
    ]:
        rows.append(
            {
                "dataset": dataset,
                "artifact_status": "upstream_output_archive_not_canonicalized",
                "artifact_origin": "original_repo upstream LLM output archive",
                "interaction_path": "",
                "user_mapping_path": "",
                "item_mapping_path": "",
                "title_mapping_path": "",
                "train_path": "",
                "validation_path": "",
                "test_path": "",
                "ground_truth_path": "",
                "negative_samples_path": "",
                "candidate_manifest_path": path,
                "num_users": "",
                "num_items": "",
                "num_interactions": "",
                "num_eligible_users": "",
                "num_evaluated_users": "",
                "filtering_protocol": "not_verifiable_from_current_artifacts",
                "split_protocol": "not_verifiable_from_current_artifacts",
                "candidate_protocol": "embedded candidate list in LLM output archive",
                "candidate_count": "20 observed in file names and sample records",
                "title_mapping_coverage": "not_verifiable_from_current_artifacts",
                "paper_alignment": "upstream_evidence_not_canonical_candidate_set",
                "blocking_reason": "Positive item ID is not directly available in canonical form; target is a title string.",
                "notes": "Recorded as provenance; not used as authoritative candidate set in Phase 2A.",
            }
        )
    return rows


def _candidate_inventory_table(candidate_sets: pd.DataFrame) -> str:
    if candidate_sets.empty:
        return "No parseable candidate sets were canonicalized."
    summary = (
        candidate_sets.groupby(["candidate_set_id", "source_path", "source_plan", "dataset"], sort=True)
        .agg(
            users=("user_id", "nunique"),
            candidate_count=("candidate_count", lambda values: ";".join(sorted({str(value) for value in values}))),
            valid_rows=("row_validation_status", lambda values: int((values == "valid").sum())),
            warning_rows=("row_validation_status", lambda values: int((values == "valid_with_warnings").sum())),
            invalid_rows=("row_validation_status", lambda values: int((values == "invalid").sum())),
        )
        .reset_index()
    )
    return _markdown_table(summary, ["candidate_set_id", "source_plan", "dataset", "users", "candidate_count", "valid_rows", "warning_rows", "invalid_rows"], max_rows=30)


def _position_summary_table(positive_positions: pd.DataFrame) -> str:
    summary = (
        positive_positions.groupby(["candidate_set_id", "stored_position_base", "position_validation_status"], sort=True)
        .size()
        .reset_index(name="rows")
    )
    return _markdown_table(summary, ["candidate_set_id", "stored_position_base", "position_validation_status", "rows"], max_rows=30)


def _dataset_summary_table(dataset_statistics: pd.DataFrame) -> str:
    return _markdown_table(dataset_statistics, ["dataset", "artifact_status", "candidate_manifest_path", "candidate_count", "paper_alignment", "blocking_reason"], max_rows=30)


def _noncanonical_table(noncanonical_artifacts: list[NonCanonicalCandidateArtifact]) -> str:
    frame = pd.DataFrame([artifact.__dict__ for artifact in noncanonical_artifacts])
    return _markdown_table(frame, ["path", "dataset", "source_plan", "reason"], max_rows=20)


def _title_coverage_for_subset(candidate_sets: pd.DataFrame, candidate_set_id: str, context: DatasetContext) -> str:
    subset = candidate_sets.loc[candidate_sets["candidate_set_id"] == candidate_set_id]
    if subset.empty:
        return ""
    title_by_item = context.title_by_item or {}
    if not title_by_item:
        return "not_verifiable_from_current_artifacts"
    available = 0
    total = 0
    for _, row in subset.iterrows():
        items = parse_candidate_items(row["candidate_items"])
        total += len(items)
        available += sum(1 for item in items if str(title_by_item.get(item, "")).strip())
    return f"{available / total:.6f}" if total else ""


def _plan_c_stats(project_root: Path) -> dict[str, dict[str, str]]:
    path = project_root / "results/tables/plan_c_reconstructed_dataset_stats.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype=str).fillna("")
    return {
        str(row["dataset"]): {column: str(row[column]) for column in frame.columns}
        for _, row in frame.iterrows()
    }


def _artifact_status(source_plan: str, paper_alignment: str) -> str:
    if source_plan == "Plan E2":
        return "derived_plan_e2_manifest"
    if source_plan == "Plan C":
        return "reconstructed_plan_c_candidate_set"
    if source_plan == "Plan B":
        return "processed_plan_b_candidate_set"
    if "diagnostic" in paper_alignment:
        return "diagnostic_manifest"
    return "candidate_set"


def _candidate_protocol(candidate_count: int | str) -> str:
    if candidate_count == "":
        return "not_applicable"
    try:
        count = int(candidate_count)
    except ValueError:
        return str(candidate_count)
    if count == 20:
        return "1 positive + 19 negatives"
    if count == 100:
        return "1 positive + 99 negatives"
    return f"1 positive + {count - 1} negatives"


def _require_columns(frame: pd.DataFrame, expected: list[str], path: Path) -> None:
    actual = list(frame.columns)
    if actual != expected:
        raise CandidateSetValidationError(f"{_relative(path)} header mismatch: expected {expected}, got {actual}")


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int) -> str:
    if frame.empty:
        return "| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns) + " |"
    shown = frame.loc[:, columns].head(max_rows)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(_escape(str(row[column])) for column in columns) + " |")
    if len(frame) > max_rows:
        lines.append("| " + " | ".join("..." for _ in columns) + " |")
    return "\n".join(lines)


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _maybe_relative(path: Path | None) -> str:
    if path is None:
        return ""
    return _relative(path)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
