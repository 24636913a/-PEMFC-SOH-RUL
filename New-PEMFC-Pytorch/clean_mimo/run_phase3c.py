"""Command-line entry point for Phase-3C H=24 history-length diagnostic."""
from __future__ import annotations

import argparse
from pathlib import Path

from phase3c_experiment import run_phase3c


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FC1/FC2 H=24 N=15/24/48/72 development diagnostics.")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "outputs_phase3c")
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    result = run_phase3c(args.output_root, args.data_root)
    print(f"Phase-3C complete: H={result['H']}, N={result['history_lengths']}; no final test was evaluated.")


if __name__ == "__main__":
    main()
