"""Resolve historical R1 root artifacts after their provenance relocation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


RELOCATION_DIR = Path("results/provenance/r1_metric_recovery")
RELOCATION_MANIFEST = Path("results/manifests/r1_root_artifact_relocation.csv")
EXPECTED_RELOCATION_COUNT = 13


@dataclass(frozen=True)
class RelocatedArtifact:
    historical_path: str
    current_path: str
    sha256: str
    size_bytes: int


def relocation_map(project_root: Path) -> dict[str, RelocatedArtifact]:
    manifest = project_root / RELOCATION_MANIFEST
    if not manifest.is_file():
        return {}

    records: dict[str, RelocatedArtifact] = {}
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["action"] != "MOVED":
                continue
            historical = row["old_path"].replace("\\", "/")
            current = row["new_path"].replace("\\", "/")
            expected_current = (RELOCATION_DIR / historical).as_posix()
            if Path(historical).name != historical or current != expected_current:
                raise ValueError(f"Invalid R1 metric-recovery relocation: {historical} -> {current}")
            if historical in records:
                raise ValueError(f"Duplicate R1 metric-recovery relocation: {historical}")
            records[historical] = RelocatedArtifact(
                historical_path=historical,
                current_path=current,
                sha256=row["sha256"].lower(),
                size_bytes=int(row["size_bytes"]),
            )
    if len(records) != EXPECTED_RELOCATION_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_RELOCATION_COUNT} R1 metric-recovery relocations, found {len(records)}"
        )
    return records


def resolve_r1_metric_recovery_artifact(project_root: Path, historical_path: str | Path) -> Path:
    """Resolve only manifest-listed historical root artifacts to current storage."""
    text = str(historical_path).replace("\\", "/")
    record = relocation_map(project_root).get(text)
    return project_root / (record.current_path if record else text)


def relocation_record(project_root: Path, historical_path: str | Path) -> RelocatedArtifact | None:
    return relocation_map(project_root).get(str(historical_path).replace("\\", "/"))
