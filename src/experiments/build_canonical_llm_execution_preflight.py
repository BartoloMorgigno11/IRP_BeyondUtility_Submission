"""Build Phase 5A canonical LLM prompt/parser/execution preflight artifacts."""

from __future__ import annotations

from src.audit import common_git

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.canonical_llm_execution_preflight import (  # noqa: E402
    OUTPUT_PATHS,
    build_all_frames,
    execution_environment,
    file_sha256,
    hash_protected_inputs,
    validate_frames,
    validate_report,
    write_frames,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    initial_hashes = hash_protected_inputs(root)
    frames = build_all_frames(root)
    write_frames(root, frames)
    report = render_report(root, frames, execution_environment(root), len(initial_hashes))
    report_path = root / OUTPUT_PATHS["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    validate_frames(frames)
    validate_report(report_path)
    final_hashes = hash_protected_inputs(root)
    protected_unchanged = initial_hashes == final_hashes
    output_hashes = {path: file_sha256(root / path) for key, path in OUTPUT_PATHS.items() if key != "report"}
    output_hashes[OUTPUT_PATHS["report"]] = file_sha256(report_path)

    for key, frame in frames.items():
        print(f"{key}_rows={len(frame)}")
    print(f"protected_inputs_hashed={len(initial_hashes)}")
    print(f"protected_inputs_unchanged={str(protected_unchanged).lower()}")
    for relative, digest in sorted(output_hashes.items()):
        print(f"sha256 {relative} {digest}")
    return 0 if protected_unchanged else 1


def git_value(root: Path, args: list[str]) -> str:
    return common_git.git_stdout_no_check(args, root, timeout=10, stdout_or_stderr=True)


def render_report(root: Path, frames: Mapping[str, pd.DataFrame], env: str, protected_count: int) -> str:
    cohort = frames["cohort_preflight"]
    render = frames["prompt_rendering_audit"]
    matrix = frames["execution_matrix"]
    model = frames["model_inventory"]
    parser = frames["parser_contract"]
    policies = frames["policy_options"]
    blocked = frames["blocked_or_conditional"]
    strict_count = int(cohort["eligible_strict_title_primary"].eq("true").sum())
    prompt_ready = int(cohort["primary_prompt_compatible"].eq("true").sum())
    strict_by_dataset = cohort.loc[cohort["eligible_strict_title_primary"].eq("true")].groupby("dataset").size().to_dict()
    render_valid = int(render["rendering_valid"].eq("true").sum())
    request_total = int(matrix["total_planned_request_count"].astype(int).sum())
    token_min = int(matrix["estimated_token_min_total"].astype(int).sum())
    token_max = int(matrix["estimated_token_max_total"].astype(int).sum())
    conditional_models = model.loc[model["availability_status"].str.contains("conditional|blocked", regex=True), "model_family"].tolist()
    local_models = model.loc[model["allowed_for_future_execution"].eq("true"), "model_family"].tolist()
    recommended_policies = policies.loc[policies["recommendation_status"].eq("recommended"), ["ambiguity_category", "policy_id"]].to_dict("records")

    lines = [
        "# Canonical LLM Prompt, Parsing and Execution Contract Preflight",
        "",
        "## 1. Scope and provenance",
        f"- Branch: `{git_value(root, ['branch', '--show-current'])}`.",
        f"- Commit: `{git_value(root, ['log', '-1', '--format=%H %s'])}`.",
        f"- Environment: `{env}`.",
        f"- Protected input files hashed: `{protected_count}`.",
        "",
        "## 2. Executive summary",
        f"- Full Plan E2 LLM candidate layer: `{len(cohort)}` rows.",
        f"- Candidate strict-title rows: `{strict_count}` with dataset counts `{strict_by_dataset}`.",
        f"- Prompt-render-valid rows after rendered-history title checks: `{prompt_ready}`.",
        "- No LLM execution, API call, Ollama invocation, or new recommendation result was produced.",
        "",
        "## 3. Safety boundaries",
        "- Phase 5A freezes contracts only. It writes contract/audit tables and does not create request or response files.",
        "",
        "## 4. Completed baseline and metric layers",
        "- Phase 4B metric matrices are treated as protected inputs and are not regenerated here.",
        "",
        "## 5. Paper LLM experiment inventory",
        "- Paper families represented: GPT-3.5, GPT-4o-mini, GPT-4o, Claude 3 Haiku, and Llama3-70B-Instruct.",
        "",
        "## 6. Official prompt inventory",
        "- Official templates under `original_repo/EvaLLMasRecommender/eva/templates/` request title-based recommendation lists, often top 10.",
        "",
        "## 7. Local prompt inventory",
        "- Plan D local prompts use strict JSON and candidate IDs for bounded diagnostics.",
        "",
        "## 8. Prompt-definition conflicts",
        "- Official title-list prompts conflict with local strict JSON ID prompts; Phase 5A freezes an adapted index-JSON contract.",
        "",
        "## 9. Ranking task semantics",
        "- Ranking uses train-only chronological history plus the persisted 20 candidates and requests top-5 ordered candidate indices.",
        "",
        "## 10. Reranking task semantics",
        "- Reranking treats the supplied candidate order as an initial order and requests a full 20-index permutation.",
        "",
        "## 11. User-history contract",
        "- History source is train split only, ordered by timestamp then source row, with held-out positive excluded from rendered history.",
        "",
        "## 12. Candidate-rendering contract",
        "- Candidate rendering preserves the exact 20-item order and displays one-based index labels with titles.",
        "",
        "## 13. Candidate-order preservation",
        "- Candidate order hashes are recorded for every strict-title prompt audit row.",
        "",
        "## 14. Positive-label concealment",
        f"- Positive-label exposure checks passed for `{render_valid}` valid prompt renderings.",
        "",
        "## 15. Ranking prompt contract",
        "- `llm_rank_index_json_top5_v1` is ready as an adapted contract.",
        "",
        "## 16. Reranking prompt contract",
        "- `llm_rerank_index_json_full20_v1` is ready as an adapted contract.",
        "",
        "## 17. Generated-profile prompt contract",
        "- Profile generation remains blocked/ambiguous and is audit-only.",
        "",
        "## 18. Output-format contract",
        "- Required output is a strict JSON object with `ranked_candidate_indices`.",
        "",
        "## 19. Strict parser contract",
        f"- Strict parser rows: `{len(parser.loc[parser['parser_mode'].eq('strict_primary_parser')])}`.",
        "",
        "## 20. Diagnostic salvage-parser contract",
        "- Salvage parser rows are diagnostic only and cannot overwrite strict primary results.",
        "",
        "## 21. Invalid-output taxonomy",
        "- Parse validity, candidate validity, output length, duplicate indices, and unknown indices are separate conditions.",
        "",
        "## 22. Candidate validity",
        "- Every output index must map to one of the supplied 20 candidates.",
        "",
        "## 23. Hallucination definition",
        "- Primary hallucination means an output index outside the supplied candidate set or an item not recoverable from candidate membership.",
        "",
        "## 24. Duplicate-output handling",
        "- Duplicates invalidate strict output; no duplicate removal is permitted for primary results.",
        "",
        "## 25. Retry policy",
        "- Scientific retry count is frozen to 0; infrastructure retries are also 0 in the frozen contract.",
        "",
        "## 26. Model inventory",
        f"- Model inventory rows: `{len(model)}`.",
        "",
        "## 27. Paper models",
        "- Closed/API and large paper models are represented but not executable in this offline phase.",
        "",
        "## 28. Local extension models",
        f"- Locally tested extension models allowed for future execution: `{local_models}`.",
        "",
        "## 29. Closed-API conditional models",
        "- GPT and Claude rows are conditional or blocked because no provider query is allowed in Phase 5A.",
        "",
        "## 30. Retired or unavailable models",
        "- Claude 3 Haiku is recorded as blocked/conditional from local Plan D registry evidence.",
        "",
        "## 31. Hardware-blocked models",
        "- Llama3-70B-Instruct is blocked for local execution on the project target hardware.",
        "",
        "## 32. Execution-parameter contract",
        f"- Execution parameter rows: `{len(frames['execution_parameter_contract'])}`.",
        "",
        "## 33. Deterministic adapted settings",
        "- Local adapted execution uses temperature 0, JSON response format, no streaming, concurrency 1, and raw-response retention.",
        "",
        "## 34. Canonical LLM cohort",
        f"- Full rows: `{len(cohort)}`; strict candidate-title rows: `{strict_count}`; prompt-compatible rows: `{prompt_ready}`.",
        "",
        "## 35. Title-coverage exclusions",
        "- Candidate title exclusions reconcile to 186 rows; rendered-history title exclusions are recorded separately.",
        "",
        "## 36. Prompt rendering validation",
        f"- Prompt rendering audit rows: `{len(render)}`; valid renderings: `{render_valid}`.",
        "",
        "## 37. Existing-response parser audit",
        f"- Existing parser audit rows: `{len(frames['existing_parser_audit'])}`; existing responses were not regenerated.",
        "",
        "## 38. Canonical execution matrix",
        f"- Execution matrix rows: `{len(matrix)}`.",
        "",
        "## 39. Request-count inventory",
        f"- Planned request count across all represented cells: `{request_total}`. This is a preflight count only.",
        "",
        "## 40. Token-envelope estimates",
        f"- Estimated token envelope total across represented cells: `{token_min}` to `{token_max}` using character/whitespace estimates.",
        "",
        "## 41. Nondeterminism extension",
        "- Exactly one result-independent bounded-repeat policy is recommended.",
        "",
        "## 42. Candidate-position-bias extension",
        "- Position bias is a separate diagnostic layer with fixed permutations, not part of canonical single-order utility.",
        "",
        "## 43. History-length extension",
        "- History-length diagnostics are separate from canonical execution and use fixed variants.",
        "",
        "## 44. Generated-profile extension",
        "- Profile generation remains blocked until prompt, model, reuse, parser, and leakage policies are frozen.",
        "",
        "## 45. Paper-aligned execution cells",
        "- Paper-model cells are included as blocked or conditional; no paper-exact LLM execution is ready offline.",
        "",
        "## 46. Adapted local execution cells",
        "- Qwen2 and Mistral local cells are contract-ready but not executed.",
        "",
        "## 47. Blocked execution cells",
        f"- Blocked/conditional issues recorded: `{len(blocked)}`; affected model families include `{conditional_models}`.",
        "",
        "## 48. Required raw-response retention",
        "- Future execution must retain raw responses, parsed outputs, attempt metadata, and prompt hashes.",
        "",
        "## 49. Reproducibility manifest requirements",
        "- Future manifests must include model identifier, prompt/parser/execution contract IDs, row IDs, hashes, and attempt number.",
        "",
        "## 50. Cost-analysis boundary",
        "- No monetary cost table or current price estimate is produced in Phase 5A.",
        "",
        "## 51. Exact-reproduction limitations",
        "- Exact paper prompts, provider availability, retry semantics, and title-list parsing remain insufficiently aligned for exact reproduction.",
        "",
        "## 52. Policy decisions",
        f"- Recommended policy IDs: `{recommended_policies}`.",
        "",
        "## 53. Execution stop conditions",
        "- Stop on candidate mutation, positive-label exposure, parser ambiguity, provider retry hiding, or protected-input mutation.",
        "",
        "## 54. Tests",
        "- Synthetic tests cover prompt registry, history rendering, candidate rendering, parsers, request counts, token estimates, and policy rules.",
        "",
        "## 55. Determinism validation",
        "- The builder is deterministic when output hashes match across consecutive runs.",
        "",
        "## 56. Protected-input immutability",
        "- Protected hashes are compared before and after writing Phase 5A outputs.",
        "",
        "## 57. Commands executed",
        "- Exact commands are recorded in the final response.",
        "",
        "## 58. Whether Phase 5A succeeded",
        "- Phase 5A succeeds when schemas validate, counts reconcile, tests pass, output hashes are deterministic, and protected inputs remain unchanged.",
        "",
        "## 59. LLM execution readiness assessment",
        "- Local Qwen2 and Mistral execution is contract-ready for prompt-compatible rows, with candidate/history title exclusions documented.",
        "",
        "## 60. Recommended single next task",
        "Phase 5B — canonical local Qwen2 and Mistral ranking/reranking execution on frozen strict-title cohorts",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
