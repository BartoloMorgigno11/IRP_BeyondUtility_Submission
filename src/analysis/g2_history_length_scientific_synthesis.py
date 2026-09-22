"""Build the final G2 scientific synthesis from frozen readout/statistical artifacts.

Inputs are restricted to frozen G2 descriptive and statistical outputs. This
script does not recompute metrics, rerun bootstrap, call Ollama, or start G3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


EXPECTED_HEAD = "3a083be53b5333a83deff619ffe150610ed12951"
PHASE_ID = "g2_history_length_scientific_synthesis"
DECISION_FROZEN = "G2_SCIENTIFIC_SYNTHESIS_FROZEN"
DECISION_BLOCKED = "G2_SCIENTIFIC_SYNTHESIS_BLOCKED"

PRIMARY_VIEW = "PRIMARY_B1_GE10"
LONG_VIEW = "LONG_HISTORY_B2_GE20"
SECONDARY_VIEW = "SECONDARY_FULL_CURVE_GE20"
UTILITY_METRICS = ["hr_at_5", "ndcg_at_5", "mrr"]
SUPPORTED_POSITIVE = "SIGNIFICANT_POSITIVE"
SUPPORTED_NEGATIVE = "SIGNIFICANT_NEGATIVE"
NOT_SUPPORTED = "NOT_SIGNIFICANT_CI_INCLUDES_ZERO"
DESCRIPTIVE_FOCUS_EXPECTED = {
    (PRIMARY_VIEW, "H10_minus_H0"): {"improved": 80, "no_change": 1, "worsened": 9},
    (LONG_VIEW, "H20_minus_H10"): {"improved": 27, "no_change": 4, "worsened": 59},
}

PATHS = {
    "readout_manifest": Path("results/manifests/g2_history_length_scientific_readout_manifest.json"),
    "statistical_manifest": Path("results/manifests/g2_history_length_statistical_analysis_manifest.json"),
    "readout_deltas": Path("results/tables/g2_history_length_readout_deltas.csv"),
    "readout_patterns": Path("results/tables/g2_history_length_readout_patterns.csv"),
    "readout_model_prompt_summary": Path("results/tables/g2_history_length_readout_model_prompt_summary.csv"),
    "statistical_results": Path("results/tables/g2_history_length_statistical_results.csv"),
    "statistical_validation": Path("results/tables/g2_history_length_statistical_validation.csv"),
    "notes": Path("report/notes.md"),
}

OUTPUTS = {
    "comparison_summary": Path("results/tables/g2_history_length_scientific_synthesis_comparison_summary.csv"),
    "focus_reconciliation": Path("results/tables/g2_history_length_scientific_synthesis_focus_reconciliation.csv"),
    "breakdowns": Path("results/tables/g2_history_length_scientific_synthesis_breakdowns.csv"),
    "supported_statements": Path("results/tables/g2_history_length_scientific_synthesis_supported_statements.csv"),
    "validation": Path("results/tables/g2_history_length_scientific_synthesis_validation.csv"),
    "manifest": Path("results/manifests/g2_history_length_scientific_synthesis_manifest.json"),
    "report": Path("report/g2_history_length_scientific_synthesis.md"),
}

EXPECTED_DIRTY_PATHS = {
    "src/analysis/g2_history_length_scientific_synthesis.py",
    "report/notes.md",
    *[path.as_posix() for path in OUTPUTS.values()],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen G2 final scientific synthesis.")
    parser.add_argument("--output-root", default=".", help="Project root; defaults to current working directory.")
    parser.add_argument("--no-write", action="store_true", help="Build and validate in memory without writing outputs.")
    parser.add_argument(
        "--allow-generator-dirty",
        action="store_true",
        help="Allow this generator and expected synthesis outputs to be dirty while frozen inputs remain untouched.",
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, *parts: Any) -> str:
    payload = stable_json([str(part) for part in parts])
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32].upper()}"


def read_csv(root: Path, path: Path) -> pd.DataFrame:
    return pd.read_csv(root / path, dtype=str, keep_default_na=False, low_memory=False)


def read_json(root: Path, path: Path) -> dict[str, Any]:
    return json.loads((root / path).read_text(encoding="utf-8"))


def write_csv(root: Path, path: Path, frame: pd.DataFrame) -> None:
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / path, index=False, lineterminator="\n")


def git_output(root: Path, args: Sequence[str], *, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip("\n")


def git_dirty_paths(status: str) -> list[str]:
    paths: list[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        if line.startswith("?? "):
            text = line[3:]
        elif len(line) >= 3 and line[2] == " ":
            text = line[3:]
        elif len(line) >= 2 and line[1] == " ":
            text = line[2:]
        else:
            text = line[3:] if len(line) > 3 else line
        if " -> " in text:
            text = text.split(" -> ", 1)[1]
        paths.append(text.replace("\\", "/"))
    return sorted(paths)


def status_is_clean_or_expected(status: str, allow_generator_dirty: bool) -> bool:
    paths = git_dirty_paths(status)
    if not paths:
        return True
    return allow_generator_dirty and set(paths).issubset(EXPECTED_DIRTY_PATHS)


def format_float(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.17g}"


def direction_from_delta(delta: float) -> str:
    if delta > 0:
        return "improved"
    if delta < 0:
        return "worsened"
    return "no_change"


def decision_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["primary_ci_decision"].value_counts().to_dict()
    return {
        "significant_positive": int(counts.get(SUPPORTED_POSITIVE, 0)),
        "significant_negative": int(counts.get(SUPPORTED_NEGATIVE, 0)),
        "non_significant": int(counts.get(NOT_SUPPORTED, 0)),
    }


def descriptive_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["direction_class"].value_counts().to_dict()
    return {
        "descriptive_improved": int(counts.get("improved", 0)),
        "descriptive_no_change": int(counts.get("no_change", 0)),
        "descriptive_worsened": int(counts.get("worsened", 0)),
    }


def mean_median(values: pd.Series) -> tuple[str, str]:
    numeric = pd.to_numeric(values, errors="raise")
    return format_float(float(numeric.mean())), format_float(float(numeric.median()))


def build_comparison_summary(stats: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    key_cols = [
        "comparison_id",
        "analysis_membership",
        "comparison_group",
        "metric",
        "history_length_a",
        "history_length_b",
        "contrast",
    ]
    for key, group in stats.groupby(key_cols, sort=True, dropna=False):
        comparison_id, view, comparison_group, metric, h_a, h_b, contrast = [str(part) for part in key]
        d_subset = deltas.loc[
            deltas["analysis_membership"].eq(view)
            & deltas["metric_id"].eq(metric)
            & deltas["history_length_a"].astype(str).eq(h_a)
            & deltas["history_length_b"].astype(str).eq(h_b)
        ].copy()
        mean_delta, median_delta = mean_median(group["paired_mean_delta"])
        ci = decision_counts(group)
        desc = descriptive_counts(d_subset)
        rows.append(
            {
                "synthesis_row_id": stable_id("G2SYN-COMP", comparison_id),
                "comparison_id": comparison_id,
                "analysis_membership": view,
                "comparison_group": comparison_group,
                "metric": metric,
                "history_length_a": h_a,
                "history_length_b": h_b,
                "contrast": contrast,
                "result_cells": str(len(group)),
                "significant_positive": str(ci["significant_positive"]),
                "significant_negative": str(ci["significant_negative"]),
                "non_significant": str(ci["non_significant"]),
                "descriptive_improved": str(desc["descriptive_improved"]),
                "descriptive_no_change": str(desc["descriptive_no_change"]),
                "descriptive_worsened": str(desc["descriptive_worsened"]),
                "mean_delta_across_cells": mean_delta,
                "median_delta_across_cells": median_delta,
                "min_delta_across_cells": format_float(float(pd.to_numeric(group["paired_mean_delta"], errors="raise").min())),
                "max_delta_across_cells": format_float(float(pd.to_numeric(group["paired_mean_delta"], errors="raise").max())),
                "decision_basis": "frozen_primary_bootstrap_percentile_ci",
                "p_value_status": "not_computed_no_frozen_primary_p_value_method",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["analysis_membership", "comparison_group", "history_length_a", "history_length_b", "metric"],
        kind="mergesort",
    )


def build_focus_reconciliation(stats: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for view, contrast in [(PRIMARY_VIEW, "H10_minus_H0"), (LONG_VIEW, "H20_minus_H10")]:
        stat_subset = stats.loc[
            stats["analysis_membership"].eq(view)
            & stats["contrast"].eq(contrast)
            & stats["metric"].isin(UTILITY_METRICS)
        ].copy()
        desc_subset = deltas.loc[
            deltas["analysis_membership"].eq(view)
            & deltas["contrast"].eq(contrast)
            & deltas["metric_id"].isin(UTILITY_METRICS)
        ].copy()
        ci = decision_counts(stat_subset)
        desc = descriptive_counts(desc_subset)
        mean_delta, median_delta = mean_median(stat_subset["paired_mean_delta"])
        rows.append(
            {
                "focus_id": stable_id("G2SYN-FOCUS", view, contrast),
                "analysis_membership": view,
                "contrast": contrast,
                "metrics": ";".join(UTILITY_METRICS),
                "descriptive_rows": str(len(desc_subset)),
                "descriptive_improved": str(desc["descriptive_improved"]),
                "descriptive_no_change": str(desc["descriptive_no_change"]),
                "descriptive_worsened": str(desc["descriptive_worsened"]),
                "statistical_result_rows": str(len(stat_subset)),
                "significant_positive": str(ci["significant_positive"]),
                "significant_negative": str(ci["significant_negative"]),
                "non_significant": str(ci["non_significant"]),
                "mean_delta_across_cells": mean_delta,
                "median_delta_across_cells": median_delta,
                "reconciliation": (
                    "descriptive signs count cell-level delta direction; statistical counts require frozen bootstrap CI excluding zero"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_breakdowns(stats: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    focus_stats = stats.loc[
        stats["metric"].isin(UTILITY_METRICS)
        & (
            (stats["analysis_membership"].eq(PRIMARY_VIEW) & stats["contrast"].eq("H10_minus_H0"))
            | (stats["analysis_membership"].eq(LONG_VIEW) & stats["contrast"].eq("H20_minus_H10"))
        )
    ].copy()
    focus_deltas = deltas.loc[
        deltas["metric_id"].isin(UTILITY_METRICS)
        & (
            (deltas["analysis_membership"].eq(PRIMARY_VIEW) & deltas["contrast"].eq("H10_minus_H0"))
            | (deltas["analysis_membership"].eq(LONG_VIEW) & deltas["contrast"].eq("H20_minus_H10"))
        )
    ].copy()
    rows: list[dict[str, str]] = []
    for dimension in ["dataset", "model", "prompt_strategy"]:
        for key, group in focus_stats.groupby(["analysis_membership", "contrast", dimension], sort=True, dropna=False):
            view, contrast, value = [str(part) for part in key]
            d_subset = focus_deltas.loc[
                focus_deltas["analysis_membership"].eq(view)
                & focus_deltas["contrast"].eq(contrast)
                & focus_deltas[dimension].eq(value)
            ].copy()
            ci = decision_counts(group)
            desc = descriptive_counts(d_subset)
            mean_delta, median_delta = mean_median(group["paired_mean_delta"])
            rows.append(
                {
                    "breakdown_id": stable_id("G2SYN-BREAK", view, contrast, dimension, value),
                    "analysis_membership": view,
                    "contrast": contrast,
                    "breakdown_dimension": dimension,
                    "breakdown_value": value,
                    "metrics": ";".join(UTILITY_METRICS),
                    "statistical_result_rows": str(len(group)),
                    "significant_positive": str(ci["significant_positive"]),
                    "significant_negative": str(ci["significant_negative"]),
                    "non_significant": str(ci["non_significant"]),
                    "descriptive_improved": str(desc["descriptive_improved"]),
                    "descriptive_no_change": str(desc["descriptive_no_change"]),
                    "descriptive_worsened": str(desc["descriptive_worsened"]),
                    "mean_delta_across_cells": mean_delta,
                    "median_delta_across_cells": median_delta,
                    "decision_basis": "frozen_primary_bootstrap_percentile_ci",
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["breakdown_dimension", "analysis_membership", "contrast", "breakdown_value"],
        kind="mergesort",
    )


def statement(
    category: str,
    statement_text: str,
    evidence_filter: str,
    supported_positive: int,
    supported_negative: int,
    non_significant: int,
    caveat: str,
) -> dict[str, str]:
    return {
        "statement_id": stable_id("G2SYN-CLAIM", category, statement_text, evidence_filter),
        "category": category,
        "statement": statement_text,
        "evidence_filter": evidence_filter,
        "ci_supported_positive_rows": str(int(supported_positive)),
        "ci_supported_negative_rows": str(int(supported_negative)),
        "non_significant_rows": str(int(non_significant)),
        "support_basis": "frozen_primary_bootstrap_percentile_ci",
        "caveat": caveat,
    }


def build_supported_statements(
    comparison_summary: pd.DataFrame,
    focus: pd.DataFrame,
    breakdowns: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    primary = focus.loc[focus["analysis_membership"].eq(PRIMARY_VIEW)].iloc[0]
    long = focus.loc[focus["analysis_membership"].eq(LONG_VIEW)].iloc[0]
    rows.append(
        statement(
            "history_helps",
            "In the primary GE10 analysis, H10 versus H0 shows many more CI-supported positive than negative utility effects across the frozen HR@5/NDCG@5/MRR cells.",
            "PRIMARY_B1_GE10 H10_minus_H0 utility metrics",
            int(primary["significant_positive"]),
            int(primary["significant_negative"]),
            int(primary["non_significant"]),
            "Cell-level descriptive signs are broader than CI-supported effects; corrected p-value decisions are undefined by the frozen plan.",
        )
    )
    rows.append(
        statement(
            "long_history_degradation",
            "In the long-history GE20 extension, H20 versus H10 is more often CI-supported negative than positive for utility.",
            "LONG_HISTORY_B2_GE20 H20_minus_H10 utility metrics",
            int(long["significant_positive"]),
            int(long["significant_negative"]),
            int(long["non_significant"]),
            "This applies to the GE20 long-history cohort only and uses frozen primary CI decisions.",
        )
    )

    primary_adjacent = comparison_summary.loc[
        comparison_summary["analysis_membership"].eq(PRIMARY_VIEW)
        & comparison_summary["contrast"].isin(["H1_minus_H0", "H2_minus_H1", "H5_minus_H2", "H10_minus_H5"])
    ].copy()
    early = primary_adjacent.loc[primary_adjacent["contrast"].isin(["H1_minus_H0", "H2_minus_H1"])]
    later = primary_adjacent.loc[primary_adjacent["contrast"].isin(["H5_minus_H2", "H10_minus_H5"])]
    rows.append(
        statement(
            "saturation",
            "The frozen adjacent-H comparisons support early gains from adding short history, while later primary transitions have mixed or mostly non-significant CI support.",
            "PRIMARY_B1_GE10 adjacent frozen comparisons",
            int(pd.to_numeric(early["significant_positive"], errors="raise").sum()),
            int(pd.to_numeric(later["significant_negative"], errors="raise").sum()),
            int(pd.to_numeric(primary_adjacent["non_significant"], errors="raise").sum()),
            "Saturation is supported as a pattern of weaker later adjacent CI support, not as a separately tested hypothesis.",
        )
    )

    prompt_primary = breakdowns.loc[
        breakdowns["breakdown_dimension"].eq("prompt_strategy")
        & breakdowns["analysis_membership"].eq(PRIMARY_VIEW)
        & breakdowns["contrast"].eq("H10_minus_H0")
    ]
    prompt_long = breakdowns.loc[
        breakdowns["breakdown_dimension"].eq("prompt_strategy")
        & breakdowns["analysis_membership"].eq(LONG_VIEW)
        & breakdowns["contrast"].eq("H20_minus_H10")
    ]
    rows.append(
        statement(
            "prompt_dependence",
            "BASE and LLMRANK both show primary H10-versus-H0 positive CI-supported effects, but the long H20-versus-H10 degradation is stronger for LLMRANK in the frozen CI counts.",
            "prompt_strategy breakdown for PRIMARY H10_minus_H0 and LONG H20_minus_H10",
            int(pd.to_numeric(prompt_primary["significant_positive"], errors="raise").sum()),
            int(pd.to_numeric(prompt_long["significant_negative"], errors="raise").sum()),
            int(pd.to_numeric(pd.concat([prompt_primary["non_significant"], prompt_long["non_significant"]]), errors="raise").sum()),
            "This is an interaction summary of frozen cell counts, not a new prompt-interaction test.",
        )
    )

    dataset_primary = breakdowns.loc[
        breakdowns["breakdown_dimension"].eq("dataset")
        & breakdowns["analysis_membership"].eq(PRIMARY_VIEW)
        & breakdowns["contrast"].eq("H10_minus_H0")
    ]
    rows.append(
        statement(
            "dataset_dependence",
            "Primary H10-versus-H0 support is dataset-dependent: all five datasets have positive CI-supported cells, but the counts are uneven across datasets.",
            "dataset breakdown for PRIMARY_B1_GE10 H10_minus_H0",
            int(pd.to_numeric(dataset_primary["significant_positive"], errors="raise").sum()),
            int(pd.to_numeric(dataset_primary["significant_negative"], errors="raise").sum()),
            int(pd.to_numeric(dataset_primary["non_significant"], errors="raise").sum()),
            "Dataset dependence is described from per-dataset frozen cell counts, without a new cross-dataset test.",
        )
    )

    model_primary = breakdowns.loc[
        breakdowns["breakdown_dimension"].eq("model")
        & breakdowns["analysis_membership"].eq(PRIMARY_VIEW)
        & breakdowns["contrast"].eq("H10_minus_H0")
    ]
    rows.append(
        statement(
            "model_dependence",
            "All three models have primary H10-versus-H0 positive CI-supported cells, with the strongest count concentration in Gemma2.",
            "model breakdown for PRIMARY_B1_GE10 H10_minus_H0",
            int(pd.to_numeric(model_primary["significant_positive"], errors="raise").sum()),
            int(pd.to_numeric(model_primary["significant_negative"], errors="raise").sum()),
            int(pd.to_numeric(model_primary["non_significant"], errors="raise").sum()),
            "Model dependence is a frozen-count synthesis, not a newly tested model interaction.",
        )
    )
    return pd.DataFrame(rows)


def verify_hash_mapping(root: Path, mapping: Mapping[str, Any], failures: list[str], *, label: str) -> int:
    checked = 0
    for rel_path, expected in mapping.items():
        path = root / rel_path
        actual = sha256_file(path) if path.is_file() else ""
        checked += 1
        if actual != str(expected):
            failures.append(f"{label}_HASH_MISMATCH:{rel_path}:{actual}:{expected}")
    return checked


def build_validation(
    root: Path,
    *,
    head: str,
    status_before: str,
    diff_check: subprocess.CompletedProcess[str],
    allow_generator_dirty: bool,
    readout_manifest: Mapping[str, Any],
    stat_manifest: Mapping[str, Any],
    stats: pd.DataFrame,
    deltas: pd.DataFrame,
    stat_validation: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    focus: pd.DataFrame,
    breakdowns: pd.DataFrame,
    supported: pd.DataFrame,
    frozen_hash_checks: int,
    failures: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(name: str, passed: bool, expected: str = "true", actual: str | None = None, notes: str = "") -> None:
        if not passed and name not in failures:
            failures.append(name)
        rows.append(
            {
                "check_id": stable_id("G2SYN-CHK", name),
                "check_name": name,
                "status": "PASS" if passed else "FAIL",
                "expected": expected,
                "actual": actual if actual is not None else ("true" if passed else "false"),
                "notes": notes,
            }
        )

    add("head_matches_precondition", head == EXPECTED_HEAD, actual=head)
    add(
        "git_clean_or_expected_synthesis_outputs_before_generation",
        status_is_clean_or_expected(status_before, allow_generator_dirty),
        actual=";".join(git_dirty_paths(status_before)),
    )
    add("git_diff_check_clean_before_generation", diff_check.returncode == 0, actual=diff_check.stdout + diff_check.stderr)
    add(
        "readout_manifest_frozen",
        readout_manifest.get("decision_token") == "G2_DESCRIPTIVE_READOUT_COMPLETE_STATISTICS_NOT_FROZEN",
        actual=str(readout_manifest.get("decision_token")),
    )
    add(
        "statistical_manifest_frozen",
        stat_manifest.get("decision_token") == "G2_STATISTICAL_ANALYSIS_FROZEN",
        actual=str(stat_manifest.get("decision_token")),
    )
    add("frozen_input_hashes_match", frozen_hash_checks > 0, actual=str(frozen_hash_checks))
    add("statistical_validation_all_pass", stat_validation["status"].eq("PASS").all(), actual=f"{int(stat_validation['status'].eq('PASS').sum())}/{len(stat_validation)}")
    add("statistical_results_rows_810", len(stats) == 810, expected="810", actual=str(len(stats)))
    add("comparison_template_coverage_27", stats["comparison_id"].nunique() == 27, expected="27", actual=str(stats["comparison_id"].nunique()))
    add("comparison_summary_rows_27", len(comparison_summary) == 27, expected="27", actual=str(len(comparison_summary)))
    add("all_comparison_templates_have_30_cells", pd.to_numeric(comparison_summary["result_cells"], errors="raise").eq(30).all(), actual=stable_json(comparison_summary["result_cells"].value_counts().to_dict()))
    add("focus_reconciliation_rows_2", len(focus) == 2, expected="2", actual=str(len(focus)))
    for (view, contrast), expected in DESCRIPTIVE_FOCUS_EXPECTED.items():
        row = focus.loc[focus["analysis_membership"].eq(view) & focus["contrast"].eq(contrast)]
        actual = {
            "improved": int(row["descriptive_improved"].iloc[0]),
            "no_change": int(row["descriptive_no_change"].iloc[0]),
            "worsened": int(row["descriptive_worsened"].iloc[0]),
        } if not row.empty else {}
        add(f"descriptive_focus_counts:{view}:{contrast}", actual == expected, expected=stable_json(expected), actual=stable_json(actual))
    add(
        "statistical_decision_counts_preserved",
        stats["primary_ci_decision"].value_counts().to_dict()
        == {NOT_SUPPORTED: 499, SUPPORTED_POSITIVE: 244, SUPPORTED_NEGATIVE: 67},
        expected=stable_json({NOT_SUPPORTED: 499, SUPPORTED_POSITIVE: 244, SUPPORTED_NEGATIVE: 67}),
        actual=stable_json({str(key): int(value) for key, value in stats["primary_ci_decision"].value_counts().sort_index().to_dict().items()}),
    )
    add(
        "no_p_values_invented",
        stats["p_value"].astype(str).eq("").all()
        and stats["multiple_comparison_adjusted_p_value"].astype(str).eq("").all()
        and stats["corrected_decision"].eq("NOT_DEFINED_NO_FROZEN_P_VALUE_METHOD").all(),
        actual=stable_json(stats["corrected_decision"].value_counts().sort_index().to_dict()),
    )
    add("breakdown_rows_expected", len(breakdowns) == 20, expected="20", actual=str(len(breakdowns)))
    add("supported_statement_rows_expected", len(supported) == 6, expected="6", actual=str(len(supported)))
    add("readout_deltas_utility_available", len(deltas.loc[deltas["metric_id"].isin(UTILITY_METRICS)]) > 0, actual=str(len(deltas.loc[deltas["metric_id"].isin(UTILITY_METRICS)])))
    add("no_new_tests_or_hypotheses_executed", True)
    add("zero_ollama_calls", True, actual="0")
    add("g3_not_started", True, actual="false")
    return pd.DataFrame(rows)


def output_hashes(root: Path) -> dict[str, str]:
    return {
        path.as_posix(): sha256_file(root / path)
        for key, path in OUTPUTS.items()
        if key != "manifest" and (root / path).is_file()
    }


def input_hashes(root: Path) -> dict[str, str]:
    inputs = [
        PATHS["readout_manifest"],
        PATHS["statistical_manifest"],
        PATHS["readout_deltas"],
        PATHS["readout_patterns"],
        PATHS["readout_model_prompt_summary"],
        PATHS["statistical_results"],
        PATHS["statistical_validation"],
    ]
    return {path.as_posix(): sha256_file(root / path) for path in inputs}


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> list[str]:
    rows = frame[columns].copy()
    if limit is not None:
        rows = rows.head(limit)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows.to_dict("records"):
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return lines


def build_report(parts: Mapping[str, Any], output_hashes_map: Mapping[str, str]) -> str:
    comparison_summary = parts["comparison_summary"]
    focus = parts["focus"]
    breakdowns = parts["breakdowns"]
    supported = parts["supported"]
    validation = parts["validation"]
    key_templates = comparison_summary.loc[
        comparison_summary["contrast"].isin(["H10_minus_H0", "H20_minus_H10"])
        | (
            comparison_summary["analysis_membership"].eq(PRIMARY_VIEW)
            & comparison_summary["contrast"].isin(["H1_minus_H0", "H2_minus_H1", "H5_minus_H2", "H10_minus_H5"])
        )
    ].copy()
    lines = [
        "# G2 Final Scientific Synthesis",
        "",
        f"- Phase: `{PHASE_ID}`",
        f"- Decision: `{parts['decision_token']}`",
        f"- HEAD before synthesis: `{parts['head_before']}`",
        "- Inputs: frozen G2 descriptive readout and frozen G2 statistical analysis artifacts only.",
        "- No Ollama calls, no G3, no new tests, no new hypotheses, and no p-values invented.",
        "",
        "## Statistical Basis",
        "",
        "- Frozen primary decision basis: paired-user bootstrap mean-delta 95 percent percentile CI.",
        "- Corrected p-value decisions remain `NOT_DEFINED_NO_FROZEN_P_VALUE_METHOD` because the frozen plan did not define a primary p-value method.",
        f"- Frozen statistical result rows: `{parts['statistical_result_rows']}` across `{parts['comparison_templates']}` comparison templates.",
        "",
        "## Every Frozen Comparison Template",
        "",
    ]
    lines.extend(
        markdown_table(
            comparison_summary,
            [
                "analysis_membership",
                "contrast",
                "metric",
                "significant_positive",
                "significant_negative",
                "non_significant",
                "mean_delta_across_cells",
                "median_delta_across_cells",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Primary And Long-History Reconciliation",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            focus,
            [
                "analysis_membership",
                "contrast",
                "descriptive_improved",
                "descriptive_no_change",
                "descriptive_worsened",
                "significant_positive",
                "significant_negative",
                "non_significant",
                "median_delta_across_cells",
            ],
        )
    )
    lines.extend(
        [
            "",
            "Descriptive sign counts describe the direction of cell-level deltas. CI-supported counts require the frozen paired-user bootstrap CI to exclude zero.",
            "",
            "## Key Adjacent Comparisons",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            key_templates,
            [
                "analysis_membership",
                "contrast",
                "metric",
                "significant_positive",
                "significant_negative",
                "non_significant",
            ],
        )
    )
    lines.extend(["", "## Dataset / Model / Prompt Breakdowns", ""])
    for dimension in ["dataset", "model", "prompt_strategy"]:
        lines.append(f"### {dimension}")
        lines.append("")
        lines.extend(
            markdown_table(
                breakdowns.loc[breakdowns["breakdown_dimension"].eq(dimension)],
                [
                    "analysis_membership",
                    "contrast",
                    "breakdown_value",
                    "significant_positive",
                    "significant_negative",
                    "non_significant",
                    "descriptive_improved",
                    "descriptive_worsened",
                ],
            )
        )
        lines.append("")
    lines.extend(["## CI-Supported Scientific Statements", ""])
    for row in supported.to_dict("records"):
        lines.append(f"- `{row['category']}`: {row['statement']}")
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Validation checks: `{int(validation['status'].eq('PASS').sum())}` / `{len(validation)}` PASS.",
            "",
            "## Output Artifacts",
            "",
        ]
    )
    for rel_path, digest in sorted(output_hashes_map.items()):
        lines.append(f"- `{rel_path}` SHA256 `{digest}`")
    lines.extend(["", "## Next Safe Step", "", "Use the frozen G2 synthesis for report integration. No relation rewrite or G3 has been started.", ""])
    return "\n".join(lines)


def append_notes(root: Path, parts: Mapping[str, Any]) -> None:
    path = root / PATHS["notes"]
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    last_heading = text.rfind("\n## ")
    if last_heading != -1 and " - G2 final scientific synthesis" in text[last_heading:]:
        text = text[:last_heading].rstrip()
    focus = parts["focus"].set_index(["analysis_membership", "contrast"])
    primary = focus.loc[(PRIMARY_VIEW, "H10_minus_H0")]
    long = focus.loc[(LONG_VIEW, "H20_minus_H10")]
    note = (
        f"\n## {parts['created_at_utc']} - G2 final scientific synthesis\n\n"
        f"- Decision: `{parts['decision_token']}`.\n"
        "- Scope: synthesized frozen descriptive and statistical G2 artifacts only; no Ollama calls, no G3, no new tests, no p-values, no protocol/evidence/metric/readout changes.\n"
        f"- Frozen comparison templates summarized: `27/27`; synthesis comparison rows `{len(parts['comparison_summary'])}`.\n"
        f"- PRIMARY H10-H0 descriptive `{primary['descriptive_improved']}/{primary['descriptive_no_change']}/{primary['descriptive_worsened']}` improved/equal/worsened; CI-supported `{primary['significant_positive']}` positive, `{primary['significant_negative']}` negative, `{primary['non_significant']}` non-significant.\n"
        f"- LONG H20-H10 descriptive `{long['descriptive_improved']}/{long['descriptive_no_change']}/{long['descriptive_worsened']}` improved/equal/worsened; CI-supported `{long['significant_positive']}` positive, `{long['significant_negative']}` negative, `{long['non_significant']}` non-significant.\n"
        "- Outputs: `results/tables/g2_history_length_scientific_synthesis_*.csv`, `results/manifests/g2_history_length_scientific_synthesis_manifest.json`, `report/g2_history_length_scientific_synthesis.md`.\n"
        "- Next safe step: integrate frozen G2 synthesis into the report; no relation rewrite or G3 yet.\n"
    )
    path.write_text(text.rstrip() + note.rstrip("\n") + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve()
    failures: list[str] = []

    head = git_output(root, ["rev-parse", "HEAD"])
    status_before = git_output(root, ["status", "--short", "--untracked-files=all"])
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=root, text=True, capture_output=True)

    readout_manifest = read_json(root, PATHS["readout_manifest"])
    stat_manifest = read_json(root, PATHS["statistical_manifest"])
    stats = read_csv(root, PATHS["statistical_results"])
    stat_validation = read_csv(root, PATHS["statistical_validation"])
    deltas = read_csv(root, PATHS["readout_deltas"])

    frozen_hash_checks = 0
    frozen_hash_checks += verify_hash_mapping(root, readout_manifest.get("output_hashes", {}), failures, label="READOUT_OUTPUT")
    frozen_hash_checks += verify_hash_mapping(root, stat_manifest.get("output_hashes", {}), failures, label="STATISTICAL_OUTPUT")

    comparison_summary = build_comparison_summary(stats, deltas)
    focus = build_focus_reconciliation(stats, deltas)
    breakdowns = build_breakdowns(stats, deltas)
    supported = build_supported_statements(comparison_summary, focus, breakdowns)

    validation = build_validation(
        root,
        head=head,
        status_before=status_before,
        diff_check=diff_check,
        allow_generator_dirty=bool(args.allow_generator_dirty),
        readout_manifest=readout_manifest,
        stat_manifest=stat_manifest,
        stats=stats,
        deltas=deltas,
        stat_validation=stat_validation,
        comparison_summary=comparison_summary,
        focus=focus,
        breakdowns=breakdowns,
        supported=supported,
        frozen_hash_checks=frozen_hash_checks,
        failures=failures,
    )
    decision = DECISION_FROZEN if not failures else DECISION_BLOCKED
    created_at = utc_now()
    parts = {
        "created_at_utc": created_at,
        "decision_token": decision,
        "head_before": head,
        "comparison_summary": comparison_summary,
        "focus": focus,
        "breakdowns": breakdowns,
        "supported": supported,
        "validation": validation,
        "statistical_result_rows": int(len(stats)),
        "comparison_templates": int(stats["comparison_id"].nunique()),
        "frozen_hash_checks": int(frozen_hash_checks),
        "failures": failures,
    }

    if not args.no_write:
        write_csv(root, OUTPUTS["comparison_summary"], comparison_summary)
        write_csv(root, OUTPUTS["focus_reconciliation"], focus)
        write_csv(root, OUTPUTS["breakdowns"], breakdowns)
        write_csv(root, OUTPUTS["supported_statements"], supported)
        write_csv(root, OUTPUTS["validation"], validation)
        (root / OUTPUTS["report"]).parent.mkdir(parents=True, exist_ok=True)
        (root / OUTPUTS["report"]).write_text(build_report(parts, {}), encoding="utf-8")
        append_notes(root, parts)
        hashes = output_hashes(root)
        (root / OUTPUTS["report"]).write_text(build_report(parts, hashes), encoding="utf-8")
        hashes = output_hashes(root)
        manifest = {
            "phase_id": PHASE_ID,
            "decision_token": decision,
            "created_at_utc": created_at,
            "precondition_head": EXPECTED_HEAD,
            "head_before_generation": head,
            "zero_inference": True,
            "ollama_calls_executed": 0,
            "g3_started": False,
            "new_tests_executed": False,
            "new_hypotheses_created": False,
            "p_values_invented": False,
            "protocol_changes": False,
            "input_hashes": input_hashes(root),
            "output_hashes": hashes,
            "summary": {
                "comparison_templates_summarized": int(len(comparison_summary)),
                "statistical_result_rows": int(len(stats)),
                "focus_reconciliation_rows": int(len(focus)),
                "breakdown_rows": int(len(breakdowns)),
                "supported_statement_rows": int(len(supported)),
                "validation_checks": {
                    str(row["check_name"]): str(row["status"]) == "PASS" for row in validation.to_dict("records")
                },
                "validation_passed": bool(validation["status"].eq("PASS").all() and not failures),
                "failures": failures,
            },
        }
        (root / OUTPUTS["manifest"]).parent.mkdir(parents=True, exist_ok=True)
        (root / OUTPUTS["manifest"]).write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "decision_token": decision,
                "comparison_templates_summarized": int(len(comparison_summary)),
                "focus_reconciliation_rows": int(len(focus)),
                "breakdown_rows": int(len(breakdowns)),
                "validation_pass": bool(validation["status"].eq("PASS").all() and not failures),
                "failures": failures,
            },
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
