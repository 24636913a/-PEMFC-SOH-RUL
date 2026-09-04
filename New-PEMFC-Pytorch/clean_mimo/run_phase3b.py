"""Command-line entry point for the development-only Phase-3B horizon sweep."""
from __future__ import annotations

import argparse
from pathlib import Path

from phase3b_experiment import run_phase3b


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FC1/FC2 H=4/8/12/24 causal horizon diagnostics.")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "outputs_phase3b")
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    result = run_phase3b(args.output_root, args.data_root)
    print(f"Phase-3B complete for {result['stacks']} and H={result['horizons']}; no final test was evaluated.")


if __name__ == "__main__":
    main()
