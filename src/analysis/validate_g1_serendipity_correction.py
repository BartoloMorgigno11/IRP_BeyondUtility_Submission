"""Verify only the ported G1 serendipity correction; no inference or broad metric audit."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import pandas as pd
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def git_blob(root: Path, reference: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{reference}:{path}"], cwd=root)


def validate(root: Path, config_path: Path) -> dict[str, Any]:
    """Check exact imported bytes, the 30 affected cells and paired contingency evidence."""
    config = yaml.safe_load((root / config_path).read_text())
    if config["allow_api_calls"]:
        raise ValueError("Offline validation only")
    base, source = config["base_commit"], config["source_fix_reference"]
    paths = config["paths"]
    old_bytes = git_blob(root, base, paths["artifact"])
    new_bytes = (root / paths["artifact"]).read_bytes()
    if new_bytes != git_blob(root, source, paths["artifact"]):
        raise ValueError("Corrected bytes differ from the source fix")
    keys = ["dataset", "model_name", "prompt_variant", "metric"]
    old = pd.read_csv(io.BytesIO(old_bytes), keep_default_na=False).set_index(keys).sort_index()
    new = pd.read_csv(io.BytesIO(new_bytes), keep_default_na=False).set_index(keys).sort_index()
    pd.testing.assert_index_equal(old.index, new.index)
    if not old.index.is_unique:
        raise ValueError("Duplicate result cells")
    changes = old["value"].ne(new["value"])
    affected = new.loc[changes]
    if len(affected) != config["affected_cells"] or set(affected.index.get_level_values("metric")) != {config["affected_metric"]}:
        raise ValueError("Unexpected changed metric cells")
    pd.testing.assert_frame_equal(old.loc[~changes], new.loc[~changes])
    pd.testing.assert_frame_equal(old.drop(columns=["value", "result_id"]), new.drop(columns=["value", "result_id"]))
    cells = pd.read_csv(root / paths["cells"])
    users = pd.read_csv(root / paths["contributions"])
    group_keys = keys[:-1]
    if users.duplicated(group_keys + ["user_id"]).any():
        raise ValueError("Duplicate user contributions")
    if not users["candidate_alignment_status"].eq("MATCH").all():
        raise ValueError("Unaligned MostPop evidence")
    recomputed = users["llm_hit_at_5"] * (1 - users["mostpop_hit_at_5_correct"])
    if not np.array_equal(recomputed, users["serendipity_contribution_correct"]):
        raise ValueError("Contribution identity mismatch")
    derived = users.assign(recomputed=recomputed).groupby(group_keys)["recomputed"].mean().sort_index()
    corrected = pd.to_numeric(new.xs(config["affected_metric"], level="metric")["value"].sort_index(), errors="raise")
    if not derived.index.equals(corrected.index) or not np.allclose(derived, corrected, rtol=0, atol=1e-12):
        raise ValueError("Corrected values disagree with user contributions")
    expected_cells = cells.set_index(group_keys)["recomputed_correct_serendipity_at_5"].sort_index()
    if not np.allclose(expected_cells, corrected, rtol=0, atol=1e-12):
        raise ValueError("Corrected values disagree with cell audit")
    protected = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", base], cwd=root).decode().splitlines()
    protected = [p for p in protected if (p.startswith("results/tables/g1_") and p != paths["artifact"]) or p.startswith("results/manifests/g1_") or p.startswith("results/llm/") or p.startswith("results/raw_outputs/")]
    for p in protected:
        # Git's path filters also account for inherited line-ending attributes.
        expected = subprocess.check_output(["git", "rev-parse", f"{base}:{p}"], cwd=root).strip()
        actual = subprocess.check_output(["git", "hash-object", "--path=" + p, p], cwd=root).strip()
        if expected != actual:
            raise ValueError("Protected inference/accounting artifact changed: " + p)
    return {"status":"PASS", "pre_fix_artifact_hash":hashlib.sha256(old_bytes).hexdigest(),
            "post_fix_artifact_hash":hashlib.sha256(new_bytes).hexdigest(),
            "bug_description":"bool('False') is True; explicit bool_value parsing restores MostPop ground-truth semantics.",
            "affected_metric":config["affected_metric"], "affected_cells":int(changes.sum()),
            "unchanged_beyond_utility_cells":int((~changes).sum()),
            "raw_inference_unchanged":True, "request_accounting_unchanged":True,
            "deterministic_recomputation_only":True, "source_fix_reference":source,
            "source_branch":"submission/prof", "source_base_commit":base,
            "protected_artifacts_verified":len(protected), "user_contributions_verified":len(users),
            "historical_manifests_policy":"Preserved unchanged as pre-correction provenance; this artifact records the correction successor.",
            "validation_command":"python -B src/analysis/validate_g1_serendipity_correction.py --config configs/g1_serendipity_correction.yaml",
            "scope":"Port and deterministic verification of the existing correction only; no G1 metric-contract audit.",
            "inference_calls":0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/g1_serendipity_correction.yaml"))
    args = parser.parse_args()
    print(json.dumps(validate(ROOT, args.config), indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
