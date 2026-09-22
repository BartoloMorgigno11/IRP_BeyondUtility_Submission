from pathlib import Path


def test_prompt_00_scaffold_directories_exist() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    expected_dirs = [
        "src/data",
        "src/baselines",
        "src/prompts",
        "src/evaluation",
        "src/experiments",
        "src/utils",
        "configs",
        "data/raw",
        "data/processed",
        "data/samples",
        "results/raw_outputs",
        "results/metrics",
        "results/tables",
        "results/plots",
        "tests",
        "report",
    ]

    missing = [path for path in expected_dirs if not (repo_root / path).is_dir()]

    assert missing == []
