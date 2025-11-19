"""Placeholder script to turn benchmark CSVs into plots.

Expected usage:
    python scripts/plot_benchmarks.py benchmarks/run.csv

Replace this stub with your preferred plotting logic once you have stable metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("csv", type=Path, help="Path to benchmark CSV")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/plot.png"), help="Where to write the plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"TODO: implement plotting for {args.csv}. Save output to {args.output}."
    )


if __name__ == "__main__":
    main()
