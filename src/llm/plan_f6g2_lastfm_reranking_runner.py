"""Local runner shell for Phase F6G2 LastFM re-ranking.

The runner always delegates to the offline F6G2 analysis gate first. Local
Ollama generation is reachable only when that gate reports an executable
top-10 request contract. With the current frozen F6G1 artifacts, the gate
blocks before any generation or raw JSONL append.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.analysis import plan_f6g2_lastfm_reranking_execution_analysis as analysis


DEFAULT_CONFIG_PATH = analysis.DEFAULT_CONFIG_PATH


def append_raw_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(analysis.stable_json(record) + "\n")


def render_prompt_from_request(row: Mapping[str, Any], *, requested_count: int = 10) -> str:
    candidate_labels = str(row.get("indexed_candidate_labels", "")).replace("|", "\n")
    history = str(row.get("history_item_ids", ""))
    return (
        "Task: rerank the supplied 20 candidate items for the user's next interaction.\n"
        "The input candidate order is the persisted initial ranking to be reranked.\n"
        f"Return exactly {requested_count} unique candidate indices, ordered from most to least likely, "
        "as one JSON object with exactly one key named ranking.\n"
        "The JSON value must be an array of integers. Every integer must be between 1 and 20.\n"
        "Do not return all 20 candidates. Do not explain. Do not include Markdown or extra fields.\n\n"
        f"User history item IDs:\n{history}\n\n"
        f"Initial candidate order:\n{candidate_labels}"
    )


def call_local_ollama(model: str, prompt: str, *, timeout_seconds: int = 120) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    completed = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )
    finished_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return {
        "success": completed.returncode == 0,
        "raw_response": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "returncode": completed.returncode,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "duration_seconds": time.perf_counter() - started,
    }


def execute_missing_requests(root: Path, config: Mapping[str, Any], *, shard_id: str = "") -> int:
    frames = analysis.load_frames(root, config)
    contract = analysis.contract_reconciliation(frames, config)
    if not contract["execution_allowed"]:
        return 0

    raw_records, duplicates, errors = analysis.load_raw_records(root, config)
    if duplicates or errors:
        raise RuntimeError(f"F6G2 raw integrity gate failed before execution: duplicates={duplicates}; errors={errors}")

    requests = list(frames["requests"])
    lookup = analysis.request_shard_lookup(requests, frames["shards"])
    if shard_id:
        allowed = {request_id for request_id, shard in lookup.items() if shard["shard_id"] == shard_id}
        requests = [row for row in requests if str(row.get("request_id", "")) in allowed]
    missing = analysis.plan_resume([str(row["request_id"]) for row in requests], raw_records)
    request_by_id = {str(row["request_id"]): row for row in requests}
    written = 0
    for request_id in missing:
        row = request_by_id[request_id]
        prompt = render_prompt_from_request(row, requested_count=int(config["execution_contract"]["required_requested_ranking_length"]))
        result = call_local_ollama(
            str(row["model"]),
            prompt,
            timeout_seconds=int(config["runtime"]["runtime_parameters"]["timeout_seconds"]),
        )
        raw_path = root / analysis.raw_path_for_model(config, str(row["model"]))
        append_raw_record(
            raw_path,
            {
                "request_id": request_id,
                "logical_instance_id": row.get("logical_instance_id", ""),
                "model": row.get("model", ""),
                "strategy": row.get("strategy", ""),
                "candidate_order_hash": row.get("candidate_order_hash", ""),
                "prompt_sha256": analysis.sha256_text(prompt),
                "transport_status": "success" if result["success"] else "failure",
                "raw_response": result["raw_response"],
                "stderr": result["stderr"],
                "returncode": result["returncode"],
                "started_at_utc": result["started_at_utc"],
                "finished_at_utc": result["finished_at_utc"],
                "duration_seconds": analysis.fmt(result["duration_seconds"]),
            },
        )
        written += 1
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase F6G2 contract-gated LastFM re-ranking.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to F6G2 YAML config.")
    parser.add_argument("--preflight-only", action="store_true", help="Run gates and write preflight/freeze artifacts.")
    parser.add_argument("--integrity-only", action="store_true", help="Run input/raw integrity gates and write artifacts.")
    parser.add_argument("--analysis-only", action="store_true", help="Analyze existing raw responses without model calls.")
    parser.add_argument("--execute", action="store_true", help="Execute all missing requests only after gates pass.")
    parser.add_argument("--resume", action="store_true", help="Resume missing requests only after gates pass.")
    parser.add_argument("--shard-id", default="", help="Restrict execution/resume to a frozen shard ID.")
    args = parser.parse_args(argv)

    mode = (
        "execute"
        if args.execute
        else "resume"
        if args.resume
        else "analysis-only"
        if args.analysis_only
        else "integrity-only"
        if args.integrity_only
        else "preflight-only"
    )
    root = Path.cwd()
    config = analysis.load_config(root / args.config)
    manifest = analysis.run_phase(root, config, mode=mode)
    if manifest["decision"] == analysis.DECISION_CONTRACT:
        print(
            analysis.stable_json(
                {
                    "phase": analysis.PHASE_ID,
                    "decision": manifest["decision"],
                    "recommendation": manifest["recommended_next_action"],
                    "scientific_content_hash": manifest["scientific_content_hash"],
                    "generated_requests": 0,
                }
            )
        )
        return 0
    if args.execute or args.resume:
        written = execute_missing_requests(root, config, shard_id=args.shard_id)
        manifest = analysis.run_phase(root, config, mode="analysis-only")
    else:
        written = 0
    print(
        analysis.stable_json(
            {
                "phase": analysis.PHASE_ID,
                "decision": manifest["decision"],
                "recommendation": manifest["recommended_next_action"],
                "scientific_content_hash": manifest["scientific_content_hash"],
                "generated_requests": written,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
