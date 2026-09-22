"""CLI wrapper for F6I2O YAML config consolidation audit."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plan_f6i2o_yaml_config_consolidation_after_explicit_approval import (
    CONFIG_PATH,
    build,
)


def main(argv: list[str] | None = None) -> int:
    result = build(CONFIG_PATH, write_outputs_enabled=True)
    print(result.decision_token)
    print(f"tables={len(result.tables)}")
    print(f"manifests={len(result.manifests)}")
    return 0 if result.validation_passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
