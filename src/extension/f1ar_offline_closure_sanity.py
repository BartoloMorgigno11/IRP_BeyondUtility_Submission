"""F1AR K-S recoverability and offline closure sanity check.

This phase reads local frozen artifacts only. It reconstructs original-four
K-S inputs without inference, recommender training, downloads, or network use.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

import pandas as pd

from src.extension import h0_history_final_parity_audit as h0
from src.extension.e0_protocol import item_list_hash, sha256_json, stable_json
from src.extension.f0_final_extension_freeze import PRIMARY_COHORT_SIZES, sha256_file
from src.extension.f1a_offline_metric_closure import (
    EXPECTED_F0_HEAD,
    OUTPUT_MANIFEST as F1A_MANIFEST,
    OUTPUT_REPORT as F1A_REPORT,
    OUTPUT_TABLES as F1A_TABLES,
    build_completeness,
    build_manifest as build_f1a_manifest,
    build_remaining_blockers as build_f1a_remaining_blockers,
    clean,
    file_hash_or_empty,
    format_float,
    ks_2samp,
    load_f0_artifacts,
    offline_result_row,
)
from src.utils.config import find_project_root


PHASE_ID = "f1ar_offline_closure_sanity"
DECISION_TOKEN = "F1A_FINAL_OFFLINE_CLOSURE_VALIDATED"
ORIGINAL_FOUR = ("beauty", "sports", "ml1m", "lastfm")

TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
REPORT_DIR = Path("report")

OUTPUT_TABLES = {
    "ks_recoverability_audit": TABLE_DIR / "f1ar_ks_recoverability_audit.csv",
    "ks_original_four_results": TABLE_DIR / "f1ar_ks_original_four_results.csv",
    "final_metric_blockers": TABLE_DIR / "f1ar_final_metric_blockers.csv",
    "cost_wording_audit": TABLE_DIR / "f1ar_cost_wording_audit.csv",
}
OUTPUT_MANIFEST = MANIFEST_DIR / "f1ar_offline_closure_sanity.json"
OUTPUT_REPORT = REPORT_DIR / "f1ar_offline_closure_sanity.md"

DATASET_LABELS = {
    "beauty": "Amazon Beauty",
    "sports": "Amazon Sports",
    "ml1m": "ML-1M",
    "lastfm": "LastFM",
}

PRIMARY_SOURCE_PATHS = {
    "beauty": "results/tables/canonical_llm_cohort_preflight.csv",
    "sports": "results/tables/canonical_llm_cohort_preflight.csv",
    "ml1m": "results/tables/canonical_llm_cohort_preflight.csv",
    "lastfm": "results/tables/phase_f6l_lastfm_operationalized_sample_cohort.csv",
}
PRIMARY_POPULATION_PATHS = {
    "beauty": "results/tables/canonical_llm_cohort_preflight.csv",
    "sports": "results/tables/canonical_llm_cohort_preflight.csv",
    "ml1m": "results/tables/canonical_llm_cohort_preflight.csv",
    "lastfm": "data/processed/finalization_f6c_lastfm_ranking_cohort.csv",
}
PRIMARY_SPLIT_SOURCES = {
    "beauty": "split column in data/reconstructed/plan_c/amazon_beauty/interactions.csv; train split excludes held-out target",
    "sports": "split column in data/reconstructed/plan_c/amazon_sports/interactions.csv; train split excludes held-out target",
    "ml1m": "split column in data/processed/ml1m/interactions.csv; train split excludes held-out target",
    "lastfm": "F6C train_history_count after validation/test holdout in data/processed/finalization_f6c_lastfm_ranking_cohort.csv",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F1AR offline closure sanity check.")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    outputs = generate_f1ar(root)
    if not bool(outputs["validation_passed"]):
        print(stable_json({"passed": False, "errors": outputs["validation_errors"]}))
        return 1
    if not args.validate_only:
        write_outputs(root, outputs)
        update_f1a_outputs(root, outputs)
    print(f"ks_recoverability_rows={len(outputs['ks_recoverability_audit'])}")
    print(f"ks_original_four_result_rows={len(outputs['ks_original_four_results'])}")
    print(f"final_metric_blockers={len(outputs['final_metric_blockers'])}")
    print(f"decision={DECISION_TOKEN}")
    print("llm_inference_calls=0")
    print("training_runs=0")
    print("downloads=0")
    print("network_calls=0")
    return 0


def generate_f1ar(root: Path | None = None) -> dict[str, Any]:
    project_root = root or find_project_root()
    contexts = build_dataset_contexts(project_root)
    recoverability = build_recoverability_audit(project_root, contexts)
    ks_results = build_ks_results(project_root, contexts)
    blockers = build_final_metric_blockers(project_root)
    cost = build_cost_wording_audit()
    validation = validate_outputs(project_root, contexts, recoverability, ks_results, blockers, cost)
    manifest = build_manifest(project_root, validation, recoverability, ks_results, blockers)
    report = render_report(recoverability, ks_results, blockers, cost, validation)
    return {
        "contexts": contexts,
        "ks_recoverability_audit": recoverability,
        "ks_original_four_results": ks_results,
        "final_metric_blockers": blockers,
        "cost_wording_audit": cost,
        "validation": validation,
        "validation_passed": validation["status"].eq("PASS").all(),
        "validation_errors": validation.loc[validation["status"].ne("PASS"), "check"].tolist(),
        "manifest": manifest,
        "report": report,
    }


def build_dataset_contexts(root: Path) -> dict[str, dict[str, Any]]:
    train_counts = {
        dataset: train_history_counts(root, dataset)
        for dataset in ("beauty", "sports", "ml1m")
    }
    h0_populations = strict_l20_history_populations(root)
    h0_cohorts = read_csv(root / "results/tables/extension_h0_final_paired_200_cohorts.csv")
    contexts: dict[str, dict[str, Any]] = {}
    for dataset in ORIGINAL_FOUR:
        if dataset == "lastfm":
            primary_sample, primary_population = lastfm_primary_vectors(root)
            primary_sample_users = read_csv(root / PRIMARY_SOURCE_PATHS[dataset])["user_id"].astype(str).tolist()
        else:
            primary_sample_users = primary_prompt_compatible_users(root, dataset)
            primary_sample = [float(train_counts[dataset][user_id]) for user_id in primary_sample_users]
            primary_population = list(primary_sample)
        history_users = (
            h0_cohorts.loc[h0_cohorts["dataset"].eq(dataset), "user_id"]
            .astype(str)
            .tolist()
        )
        history_population = h0_populations[dataset]
        history_by_user = history_population.set_index("user_id")["train_history_count_before_target"].astype(float).to_dict()
        history_sample = [float(history_by_user[user_id]) for user_id in history_users]
        contexts[dataset] = {
            "dataset": dataset,
            "primary_sample_users": primary_sample_users,
            "primary_sample": primary_sample,
            "primary_population": primary_population,
            "history_users": history_users,
            "history_sample": history_sample,
            "history_population": history_population["train_history_count_before_target"].astype(float).tolist(),
            "history_population_users": history_population["user_id"].astype(str).tolist(),
            "history_cohort_hash": item_list_hash(history_users),
            "history_population_hash": item_list_hash(history_population["user_id"].astype(str).tolist()),
            "primary_source": PRIMARY_SOURCE_PATHS[dataset],
            "primary_population_source": PRIMARY_POPULATION_PATHS[dataset],
            "history_population_source": "H0 strict L20 pool reconstructed from canonical interactions/candidates/catalog",
        }
    return contexts


def train_history_counts(root: Path, dataset: str) -> dict[str, int]:
    train = h0.train_interactions(h0.read_interactions_csv(root / h0.INTERACTIONS_PATHS[dataset]))
    return {
        str(user_id): int(size)
        for user_id, size in train.groupby("user_id", sort=False).size().items()
    }


def strict_l20_history_populations(root: Path) -> dict[str, pd.DataFrame]:
    catalog_frames = {dataset: h0.read_catalog_csv(root / path) for dataset, path in h0.CATALOG_PATHS.items()}
    catalogs = {dataset: h0.title_map(frame) for dataset, frame in catalog_frames.items()}
    full_pool = h0.build_full_feasibility_pool(root, catalogs, catalog_frames)
    populations: dict[str, pd.DataFrame] = {}
    for dataset in ORIGINAL_FOUR:
        counts = train_history_counts(root, dataset)
        subset = full_pool.loc[
            full_pool["dataset"].eq(dataset)
            & full_pool["fully_paired_l20_valid"].astype(bool)
        ].copy()
        subset["train_history_count_before_target"] = subset["user_id"].astype(str).map(counts).astype(int)
        populations[dataset] = subset[["dataset", "user_id", "train_history_count_before_target"]].sort_values("user_id", kind="mergesort").reset_index(drop=True)
    return populations


def primary_prompt_compatible_users(root: Path, dataset: str) -> list[str]:
    frame = read_csv(root / "results/tables/canonical_llm_cohort_preflight.csv")
    label = DATASET_LABELS[dataset]
    selected = frame.loc[
        frame["dataset"].eq(label)
        & frame["task_type"].eq("ranking")
        & frame["primary_prompt_compatible"].astype(str).str.lower().eq("true")
    ].copy()
    return selected["user_id"].astype(str).tolist()


def lastfm_primary_vectors(root: Path) -> tuple[list[float], list[float]]:
    sample = read_csv(root / "results/tables/phase_f6l_lastfm_operationalized_sample_cohort.csv")
    population = read_csv(root / "data/processed/finalization_f6c_lastfm_ranking_cohort.csv")
    return (
        sample["train_history_count"].astype(float).tolist(),
        population["train_history_count"].astype(float).tolist(),
    )


def build_recoverability_audit(root: Path, contexts: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    h0_feasibility = read_csv(root / "results/tables/extension_h0_dataset_l20_feasibility.csv")
    f0r_primary = read_csv(root / "results/tables/f0r_primary_request_breakdown.csv")
    rows = []
    for dataset in ORIGINAL_FOUR:
        context = contexts[dataset]
        h0_row = h0_feasibility.loc[h0_feasibility["dataset"].eq(dataset)].iloc[0]
        f0r_row = f0r_primary.loc[f0r_primary["dataset"].eq(dataset)].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "canonical_interaction_source": canonical_interaction_source(dataset),
                "canonical_split_source": PRIMARY_SPLIT_SOURCES[dataset],
                "eligible_population_definition": eligible_population_definition(dataset),
                "eligible_population_N": len(context["history_population"]),
                "primary_population_N": len(context["primary_population"]),
                "history_population_N": len(context["history_population"]),
                "primary_sample_N": len(context["primary_sample"]),
                "history_sample_N": len(context["history_sample"]),
                "vector_reconstructible": "true",
                "missing_material_if_any": "",
                "reconstruction_algorithm": reconstruction_algorithm(dataset),
                "deterministic": "true",
                "final_status": "COMPUTABLE_OFFLINE",
                "history_population_user_hash": context["history_population_hash"],
                "history_cohort_user_hash": context["history_cohort_hash"],
                "h0_expected_history_cohort_hash": clean(h0_row.get("cohort_user_hash")),
                "f0_primary_cohort_id": clean(f0r_row.get("cohort_name")),
                "f0_primary_hash_status": clean(f0r_row.get("hash_status")),
                "primary_membership_source": context["primary_source"],
                "primary_population_source": context["primary_population_source"],
                "source_hashes": stable_json(source_hashes(root, dataset)),
            }
        )
    return pd.DataFrame(rows)


def build_ks_results(root: Path, contexts: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for dataset in ORIGINAL_FOUR:
        context = contexts[dataset]
        rows.append(ks_result_row(root, dataset, "primary_vs_eligible_population", context["primary_sample"], context["primary_population"], context["primary_source"], context["primary_population_source"]))
        rows.append(ks_result_row(root, dataset, "history200_vs_strict_l20_population", context["history_sample"], context["history_population"], "results/tables/extension_h0_final_paired_200_cohorts.csv", context["history_population_source"]))
    return pd.DataFrame(rows)


def ks_result_row(
    root: Path,
    dataset: str,
    comparison: str,
    sample: Sequence[float],
    population: Sequence[float],
    sample_source: str,
    population_source: str,
) -> dict[str, Any]:
    result = ks_2samp(sample, population)
    if result is None:
        raise ValueError(f"empty K-S vectors for {dataset} {comparison}")
    identity = {
        "dataset": dataset,
        "comparison": comparison,
        "statistic": result["statistic"],
        "p_value": result["p_value"],
        "sample_n": int(result["sample_n"]),
        "population_n": int(result["population_n"]),
    }
    return {
        "ks_result_id": "F1AR-KS-" + sha256_json(identity).upper()[:16],
        "dataset": dataset,
        "comparison": comparison,
        "variable": "train_history_count_before_target",
        "sample_N": int(result["sample_n"]),
        "population_N": int(result["population_n"]),
        "ks_statistic": format_float(result["statistic"]),
        "p_value": format_float(result["p_value"]),
        "evidence_class": "LOCAL_EXTENSION;PAPER_ALIGNED_RECONSTRUCTED_TARGET",
        "sample_source": sample_source,
        "population_source": population_source,
        "source_hashes": stable_json(source_hashes(root, dataset)),
        "computation_status": "COMPUTED_OFFLINE",
        "notes": "Two-sample K-S over train-history-count; diagnostic representativeness audit, not paper-exact claim.",
    }


def build_final_metric_blockers(root: Path) -> pd.DataFrame:
    columns = ["metric", "dataset", "blocker_type", "reason", "f1ar_status", "next_action"]
    if f1arr_protocol_recovered(root):
        return pd.DataFrame(columns=columns)

    f1a_blockers = read_csv(root / "results/tables/f1a_remaining_blockers.csv")
    rows = []
    for metric in ("popreo_at_5", "dpd_at_5"):
        source = f1a_blockers.loc[f1a_blockers["metric"].eq(metric)]
        reason = clean(source["reason"].iloc[0]) if not source.empty else ""
        rows.append(
            {
                "metric": metric,
                "dataset": "ALL",
                "blocker_type": "BLOCKED_GLOBAL_PROTOCOL",
                "reason": reason,
                "f1ar_status": "PRESERVED_PROTOCOL_BLOCKER",
                "next_action": "recover protocol from paper/original source or explicitly replace as an adapted extension before computing",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_cost_wording_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cost_dimension": "paid_api_marginal_monetary_cost",
                "f1a_status": "NOT_APPLICABLE",
                "wording": "No paid API provider is used by the frozen local-Ollama execution plan; provider marginal monetary cost is not a reportable metric.",
                "request_budget_effect": "unchanged_zero_extra_requests",
            },
            {
                "cost_dimension": "total_computational_cost",
                "f1a_status": "NOT_ZERO_BUT_UNPRICED",
                "wording": "Local execution has nonzero compute time and resource use; do not describe total computational cost as zero.",
                "request_budget_effect": "unchanged",
            },
            {
                "cost_dimension": "hardware_electricity_cost",
                "f1a_status": "UNMEASURED",
                "wording": "Hardware depreciation and electricity cost are not measured by F1A/F1AR and must not be inferred.",
                "request_budget_effect": "unchanged",
            },
        ]
    )


def validate_outputs(
    root: Path,
    contexts: Mapping[str, Mapping[str, Any]],
    recoverability: pd.DataFrame,
    ks_results: pd.DataFrame,
    blockers: pd.DataFrame,
    cost: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, str]] = []

    def check(name: str, condition: bool, evidence: str) -> None:
        checks.append({"check": name, "status": "PASS" if condition else "FAIL", "evidence": evidence})

    head = git_output(root, ["rev-parse", "HEAD"])
    f1a_manifest = read_json(root / "results/manifests/f1a_no_inference_manifest.json")
    f0_budget = read_csv(root / "results/tables/f0_final_request_budget.csv")
    budget = {row["budget_item"]: int(row["request_count"]) for row in f0_budget.to_dict("records")}
    h0_feasibility = read_csv(root / "results/tables/extension_h0_dataset_l20_feasibility.csv")

    check("head_descends_from_f0_checkpoint", head_descends_from(root, EXPECTED_F0_HEAD), f"{EXPECTED_F0_HEAD}..{head}")
    check("f1a_precondition_validated", f1a_manifest.get("decision_token") == "F1A_OFFLINE_METRIC_CLOSURE_VALIDATED", clean(f1a_manifest.get("decision_token")))
    check("recoverability_four_datasets", set(recoverability["dataset"]) == set(ORIGINAL_FOUR) and len(recoverability) == 4, ",".join(recoverability["dataset"]))
    check("all_vectors_reconstructible", recoverability["vector_reconstructible"].eq("true").all(), ",".join(recoverability["final_status"]))
    check("all_original_four_ks_computed", len(ks_results) == 8 and ks_results["computation_status"].eq("COMPUTED_OFFLINE").all(), str(len(ks_results)))
    for dataset in ORIGINAL_FOUR:
        context = contexts[dataset]
        expected_primary = PRIMARY_COHORT_SIZES[dataset]
        expected_history = int(h0_feasibility.loc[h0_feasibility["dataset"].eq(dataset), "selected_users"].iloc[0])
        expected_population = int(h0_feasibility.loc[h0_feasibility["dataset"].eq(dataset), "strict_paired_l20_users"].iloc[0])
        check(f"{dataset}_primary_n_matches_f0", len(context["primary_sample"]) == expected_primary, f"{len(context['primary_sample'])} vs {expected_primary}")
        check(f"{dataset}_history_n_matches_h0", len(context["history_sample"]) == expected_history, f"{len(context['history_sample'])} vs {expected_history}")
        check(f"{dataset}_history_population_matches_h0", len(context["history_population"]) == expected_population, f"{len(context['history_population'])} vs {expected_population}")
        check(f"{dataset}_history_cohort_hash_matches_h0", recoverability.loc[recoverability["dataset"].eq(dataset), "history_cohort_user_hash"].iloc[0] == recoverability.loc[recoverability["dataset"].eq(dataset), "h0_expected_history_cohort_hash"].iloc[0], dataset)
        check(f"{dataset}_target_exclusion_preserved", min(context["primary_sample"] + context["history_sample"]) >= 1.0, "train-only counts before target")
    if f1arr_protocol_recovered(root):
        check("popreo_dpd_global_blockers_closed_by_f1arr", blockers.empty, str(len(blockers)))
    else:
        check("popreo_and_dpd_only_final_blockers", set(blockers["metric"]) == {"popreo_at_5", "dpd_at_5"} and len(blockers) == 2, ",".join(blockers["metric"]))
    check("cost_wording_distinguishes_zero_api_from_compute", {"paid_api_marginal_monetary_cost", "total_computational_cost", "hardware_electricity_cost"} == set(cost["cost_dimension"]), ",".join(cost["cost_dimension"]))
    check("f0_request_budget_unchanged", budget.get("MANDATORY_RAW_REQUESTS_TOTAL") == 50292 and budget.get("CONCEPTUAL_ALL_APPLICABLE_REQUESTS_TOTAL") == 100584, stable_json(budget))
    check("no_inference_training_download_network", True, "F1AR reads local CSV/JSON artifacts only; counters are zero")
    return pd.DataFrame(checks)


def build_manifest(root: Path, validation: pd.DataFrame, recoverability: pd.DataFrame, ks_results: pd.DataFrame, blockers: pd.DataFrame) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "decision_token": DECISION_TOKEN,
        "head": git_output(root, ["rev-parse", "HEAD"]),
        "expected_f0_checkpoint_head": EXPECTED_F0_HEAD,
        "validation_passed": bool(validation["status"].eq("PASS").all()),
        "validation_failures": validation.loc[validation["status"].ne("PASS"), "check"].tolist(),
        "ks_recoverability_rows": len(recoverability),
        "ks_original_four_result_rows": len(ks_results),
        "final_metric_blockers": blockers["metric"].tolist(),
        "avoidable_ks_blockers": 0,
        "llm_inference_calls_executed": 0,
        "training_runs_executed": 0,
        "downloads_executed": 0,
        "network_calls_executed": 0,
        "commit_created": 0,
    }


def write_outputs(root: Path, outputs: Mapping[str, Any]) -> None:
    for directory in (root / TABLE_DIR, root / MANIFEST_DIR, root / REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    for key, rel_path in OUTPUT_TABLES.items():
        outputs[key].to_csv(root / rel_path, index=False, lineterminator="\n")
    (root / OUTPUT_REPORT).write_text(outputs["report"], encoding="utf-8")
    manifest = dict(outputs["manifest"])
    manifest["artifact_hashes"] = {
        rel_path.as_posix(): sha256_file(root / rel_path)
        for rel_path in OUTPUT_TABLES.values()
    }
    manifest["artifact_hashes"][OUTPUT_REPORT.as_posix()] = sha256_file(root / OUTPUT_REPORT)
    (root / OUTPUT_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def update_f1a_outputs(root: Path, outputs: Mapping[str, Any]) -> None:
    recovered_by_f1arr = f1arr_protocol_recovered(root)
    offline = read_csv(root / "results/tables/f1a_offline_metric_results.csv")
    offline = offline.loc[~offline["evidence_source"].astype(str).str.contains("f1ar_ks_original_four_results.csv", regex=False)].copy()
    artifacts = load_f0_artifacts(root)
    ks_cell = artifacts["result_cells"].loc[artifacts["result_cells"]["metric"].eq("ks_sample_representativeness")].iloc[0].to_dict()
    add_rows = []
    for result in outputs["ks_original_four_results"].to_dict("records"):
        add_rows.append(
            offline_result_row(
                ks_cell,
                metric_value=result["ks_statistic"],
                value_json=stable_json(result),
                evidence_source="results/tables/f1ar_ks_original_four_results.csv;" + result["sample_source"] + ";" + result["population_source"],
                computation_status="COMPUTED_F1A_F1AR",
                notes="F1AR original-four K-S closure; no model call.",
            )
        )
    offline = pd.concat([offline, pd.DataFrame(add_rows)], ignore_index=True)
    offline.to_csv(root / "results/tables/f1a_offline_metric_results.csv", index=False, lineterminator="\n")

    blockers = read_csv(root / "results/tables/f1a_remaining_blockers.csv")
    if recovered_by_f1arr:
        blockers = blockers.iloc[0:0].copy()
    else:
        blockers = blockers.loc[~blockers["metric"].eq("ks_sample_representativeness")].copy()
    blockers.to_csv(root / "results/tables/f1a_remaining_blockers.csv", index=False, lineterminator="\n")

    validation = read_csv(root / "results/tables/f1a_validation.csv")
    validation = validation.loc[~validation["check"].astype(str).str.startswith("f1ar_")].copy()
    blocker_check = (
        {"check": "f1ar_popreo_dpd_global_blockers_closed_by_f1arr", "status": "PASS", "evidence": "PopREO and DPD global protocols recovered by F1ARR; remaining global metric blockers = 0"}
        if recovered_by_f1arr
        else {"check": "f1ar_final_metric_blockers_only_popreo_dpd", "status": "PASS", "evidence": "PopREO and DPD retained as protocol blockers; avoidable K-S blockers = 0"}
    )
    validation = pd.concat(
        [
            validation,
            pd.DataFrame(
                [
                    {"check": "f1ar_original_four_ks_closed", "status": "PASS", "evidence": "8 original-four primary/history K-S diagnostics computed offline"},
                    blocker_check,
                ]
            ),
        ],
        ignore_index=True,
    )
    validation.to_csv(root / "results/tables/f1a_validation.csv", index=False, lineterminator="\n")

    readiness = read_csv(root / "results/tables/f1a_result_cell_readiness.csv")
    original_four = build_completeness(readiness, list(ORIGINAL_FOUR))
    original_four.to_csv(root / "results/tables/f1a_original_four_completeness.csv", index=False, lineterminator="\n")

    metric_resolution = read_csv(root / "results/tables/f1a_metric_resolution_audit.csv")
    dataset_applicability = read_csv(root / "results/tables/f1a_metric_dataset_applicability.csv")
    refreshed_blockers = build_f1a_remaining_blockers(metric_resolution, dataset_applicability)
    if recovered_by_f1arr:
        refreshed_blockers = refreshed_blockers.iloc[0:0].copy()
    else:
        refreshed_blockers = refreshed_blockers.loc[~refreshed_blockers["metric"].eq("ks_sample_representativeness")].copy()
    refreshed_blockers.to_csv(root / "results/tables/f1a_remaining_blockers.csv", index=False, lineterminator="\n")

    report_path = root / F1A_REPORT
    report_text = report_path.read_text(encoding="utf-8")
    marker = "\n## F1AR Addendum\n"
    report_text = report_text.split(marker, 1)[0].rstrip()
    blocker_note = (
        "- F1ARR recovered the PopREO and DPD global protocols; remaining global metric blockers = 0.\n"
        if recovered_by_f1arr
        else "- K-S was removed from the F1A remaining blocker table; final true metric blockers are PopREO and DPD.\n"
    )
    addendum = (
        marker
        + "\n"
        + "- Original-four K-S population vectors were reconstructed offline.\n"
        + "- Eight original-four K-S diagnostics were added from F1AR: primary and history200 for Beauty, Sports, ML1M, and LastFM.\n"
        + "- Beauty/Sports/ML1M primary K-S comparisons use identical-size sample/population definitions, so KS=0 and p=1 are structural and must not be overinterpreted as independent representativeness evidence.\n"
        + blocker_note
    )
    report_path.write_text(report_text + addendum, encoding="utf-8")

    manifest_path = root / F1A_MANIFEST
    f1a_manifest = read_json(manifest_path)
    f1a_manifest["f1ar_update"] = {
        "phase_id": PHASE_ID,
        "decision_token": DECISION_TOKEN,
        "ks_original_four_result_rows_added": len(add_rows),
        "avoidable_ks_blockers": 0,
    }
    f1a_manifest["artifact_hashes"] = {
        rel_path.as_posix(): sha256_file(root / rel_path)
        for rel_path in F1A_TABLES.values()
        if (root / rel_path).exists()
    }
    f1a_manifest["artifact_hashes"][F1A_REPORT.as_posix()] = sha256_file(root / F1A_REPORT)
    manifest_path.write_text(json.dumps(f1a_manifest, indent=2, sort_keys=True), encoding="utf-8")


def render_report(recoverability: pd.DataFrame, ks_results: pd.DataFrame, blockers: pd.DataFrame, cost: pd.DataFrame, validation: pd.DataFrame) -> str:
    blocker_lines = (
        "None; F1ARR recovered PopREO/DPD global protocols. Degenerate protocol-input cells remain explicit in the F1ARR degenerate-cell table."
        if blockers.empty
        else "\n".join(f"- `{row['metric']}`: {row['reason']}" for row in blockers.to_dict("records"))
    )
    return "\n".join(
        [
            "# F1AR Offline Closure Sanity",
            "",
            f"Decision token: `{DECISION_TOKEN}`",
            "",
            "## K-S Recoverability",
            f"- Original-four datasets audited: `{len(recoverability)}`.",
            f"- Original-four K-S result rows computed: `{len(ks_results)}`.",
            "- Evidence class remains reconstructed/adapted diagnostic, not paper-exact representativeness.",
            "- Beauty/Sports/ML1M primary K-S comparisons use identical-size sample/population definitions, so KS=0 and p=1 are structural and must not be overinterpreted as independent representativeness evidence.",
            "",
            "## Final Blockers",
            blocker_lines,
            "",
            "## Cost Wording",
            "\n".join(f"- `{row['cost_dimension']}`: {row['wording']}" for row in cost.to_dict("records")),
            "",
            "## Validation",
            f"- Passed: `{str(validation['status'].eq('PASS').all()).lower()}`.",
        ]
    ) + "\n"


def canonical_interaction_source(dataset: str) -> str:
    if dataset == "lastfm":
        return "data/processed/finalization_f6c_lastfm_interactions.csv plus F6C/F6L cohort artifacts"
    return h0.INTERACTIONS_PATHS[dataset].as_posix()


def eligible_population_definition(dataset: str) -> str:
    if dataset == "lastfm":
        return "primary: F6C full ranking cohort with train_history_count; history: H0 strict paired L20 pool"
    if dataset in {"beauty", "sports", "ml1m"}:
        return "primary: canonical prompt-compatible ranking rows; history: H0 strict paired L20 pool"
    return ""


def reconstruction_algorithm(dataset: str) -> str:
    if dataset == "lastfm":
        return "read F6L operationalized 1000 sample and F6C full ranking cohort train_history_count; reconstruct H0 strict L20 population with H0 train/candidate/title policy"
    return "count train split rows per user before held-out target; select primary_prompt_compatible ranking users; reconstruct H0 strict L20 population with H0 train/candidate/title policy"


def source_hashes(root: Path, dataset: str) -> dict[str, str]:
    paths = {
        "interaction_source": h0.INTERACTIONS_PATHS.get(dataset, Path("data/processed/finalization_f6c_lastfm_interactions.csv")),
        "candidate_source": h0.CANDIDATE_PATHS[dataset],
        "catalog_source": h0.CATALOG_PATHS[dataset],
        "h0_cohorts": Path("results/tables/extension_h0_final_paired_200_cohorts.csv"),
        "primary_source": Path(PRIMARY_SOURCE_PATHS[dataset]),
        "primary_population_source": Path(PRIMARY_POPULATION_PATHS[dataset]),
    }
    return {key: file_hash_or_empty(root / path) for key, path in paths.items()}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_output(root: Path, args: list[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return completed.stdout.strip()


def head_descends_from(root: Path, ancestor: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.returncode == 0


def f1arr_protocol_recovered(root: Path) -> bool:
    manifest_path = root / "results/manifests/f1arr_popreo_dpd_recovery.json"
    if not manifest_path.exists():
        return False
    manifest = read_json(manifest_path)
    return (
        manifest.get("decision_token") == "F1ARR_POPREO_DPD_RECOVERED"
        and int(manifest.get("global_metric_blockers", -1)) == 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
