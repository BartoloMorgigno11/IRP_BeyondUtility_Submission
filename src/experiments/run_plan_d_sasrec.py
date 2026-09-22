"""Run the Plan D SASRec sequential recommendation baseline.

Quick diagnostic run from the repository root:

    python src/experiments/run_plan_d_sasrec.py --dataset ml1m --epochs 1 --max-users 100 --max-train-interactions 5000 --scope diagnostic_subset --skip-plots --verbose

The runner builds user histories only from reconstructed Plan C training
interactions and evaluates only against reconstructed Plan C candidate sets.
It does not modify Plan A/B/C outputs, rerun Plan C baselines, download data,
or call APIs.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from src.baselines.plan_d_bprmf import KNOWN_DATASETS
from src.baselines.plan_d_sasrec import SASRecConfig, run_plan_d_sasrec


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate Plan D SASRec on reconstructed Plan C data.")
    parser.add_argument("--dataset", required=True, choices=list(KNOWN_DATASETS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Subset user cap. Defaults to 1000 only when scope=diagnostic_subset.",
    )
    parser.add_argument(
        "--max-train-interactions",
        type=int,
        default=None,
        help="Subset training-interaction cap. Defaults to 50000 only when scope=diagnostic_subset.",
    )
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--scope", choices=["full", "diagnostic_subset", "extended_subset"], default="diagnostic_subset")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/reconstructed/plan_c"),
        help="Project-relative reconstructed Plan C data root.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Project-relative results root.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    max_users = args.max_users
    max_train_interactions = args.max_train_interactions
    if args.scope == "diagnostic_subset":
        max_users = 1000 if max_users is None else max_users
        max_train_interactions = 50_000 if max_train_interactions is None else max_train_interactions
    config = SASRecConfig(
        dataset=args.dataset,
        scope=args.scope,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        max_seq_len=args.max_seq_len,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        max_users=max_users,
        max_train_interactions=max_train_interactions,
        patience=args.patience,
        requested_device=args.device,
    )
    result = run_plan_d_sasrec(
        config=config,
        data_root=args.data_root,
        results_root=args.results_root,
        project_root=PROJECT_ROOT,
        skip_plots=args.skip_plots,
    )
    logging.info("Saved metrics to %s", result["metrics_path"].relative_to(PROJECT_ROOT))
    logging.info("Saved recommendations to %s", result["recommendations_path"].relative_to(PROJECT_ROOT))
    logging.info("Saved training summary to %s", result["summary_path"].relative_to(PROJECT_ROOT))
    for plot_path in result["plot_paths"]:
        logging.info("Saved plot to %s", plot_path.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
