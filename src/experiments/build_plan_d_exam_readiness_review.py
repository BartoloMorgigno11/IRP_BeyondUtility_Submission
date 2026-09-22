from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_d_exam_readiness_review import (
    CHECKLIST_COLUMNS,
    CLAIM_AUDIT_COLUMNS,
    ORAL_POINTS_COLUMNS,
    REPORT_DIR,
    TABLES_DIR,
    build_claim_safety_audit,
    build_exam_readiness_checklist,
    build_exam_readiness_report,
    build_oral_defense_key_points,
)


LOGGER = logging.getLogger(__name__)

CHECKLIST_PATH = TABLES_DIR / "plan_d_exam_readiness_checklist.csv"
CLAIM_AUDIT_PATH = TABLES_DIR / "plan_d_claim_safety_audit.csv"
ORAL_POINTS_PATH = TABLES_DIR / "plan_d_oral_defense_key_points.csv"
REPORT_PATH = REPORT_DIR / "plan_d_exam_readiness_review.md"
NOTES_PATH = REPORT_DIR / "notes.md"


def build_review() -> dict[str, object]:
    checklist = build_exam_readiness_checklist()
    claim_audit = build_claim_safety_audit()
    oral_points = build_oral_defense_key_points()
    report = build_exam_readiness_report(checklist, claim_audit, oral_points)
    return {
        "checklist": checklist,
        "claim_audit": claim_audit,
        "oral_points": oral_points,
        "report": report,
    }


def write_review(review: dict[str, object]) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    checklist = review["checklist"]
    claim_audit = review["claim_audit"]
    oral_points = review["oral_points"]
    report = review["report"]

    checklist.to_csv(CHECKLIST_PATH, index=False, columns=CHECKLIST_COLUMNS)
    claim_audit.to_csv(CLAIM_AUDIT_PATH, index=False, columns=CLAIM_AUDIT_COLUMNS)
    oral_points.to_csv(ORAL_POINTS_PATH, index=False, columns=ORAL_POINTS_COLUMNS)
    REPORT_PATH.write_text(str(report), encoding="utf-8")
    _update_notes()


def _update_notes() -> None:
    section = """
## 2026-07-01 - Plan D D5M: Final Report Critical Review and Exam Readiness Audit

### Changes

- Built the D5M exam-readiness checklist, claim-safety audit, oral-defense key points, and concise review report.
- Reviewed the D5L final package, reproducibility matrix, notes, final tables, blocked-items table, and hygiene audit.
- No new inference, API call, download, model pull, recommender training, LastFM run, destructive cleanup, or metric recomputation occurred.

### Outputs Generated

- `results/tables/plan_d_exam_readiness_checklist.csv`
- `results/tables/plan_d_claim_safety_audit.csv`
- `results/tables/plan_d_oral_defense_key_points.csv`
- `report/plan_d_exam_readiness_review.md`

### Result

- Final oral positioning should be: reconstructed reproduction plus local diagnostic/extended subset analysis.
- D5E-D5K remain diagnostic_subset local LLM diagnostics, not exact or full LLM reproduction.
- The review flags stale D5H blocked wording in the matrix as a documentation wording issue to avoid in oral presentation.
"""
    _replace_or_append_section(NOTES_PATH, "## 2026-07-01 - Plan D D5M: Final Report Critical Review and Exam Readiness Audit", section)


def _replace_or_append_section(path: Path, header: str, section: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if header in text:
        text = _replace_section_text(text, header, section.strip() + "\n")
    else:
        text = text.rstrip() + "\n\n" + section.strip() + "\n"
    path.write_text(text, encoding="utf-8")


def _replace_section_text(text: str, header: str, replacement: str) -> str:
    start = text.find(header)
    if start == -1:
        return text.rstrip() + "\n\n" + replacement.strip() + "\n"
    next_start = text.find("\n## ", start + 1)
    if next_start == -1:
        return text[:start] + replacement.strip() + "\n"
    return text[:start] + replacement.strip() + "\n" + text[next_start:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Plan D D5M exam-readiness review artifacts.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    review = build_review()
    write_review(review)
    LOGGER.info("Wrote %s", REPORT_PATH)


if __name__ == "__main__":
    main()
