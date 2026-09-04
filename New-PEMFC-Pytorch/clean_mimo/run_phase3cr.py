"""Entry point for the frozen Phase-3C-R matched-origin history diagnostic."""
from __future__ import annotations

import argparse
from pathlib import Path

from phase3cr_experiment import run_phase3cr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched-origin FC1/FC2 H=24 N=15/24/48 development diagnostics.")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "outputs_phase3cr")
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    result = run_phase3cr(args.output_root, args.data_root)
    print(f"Phase-3C-R complete: H={result['H']}, N={result['history_lengths']}; no final-test evaluation.")


if __name__ == "__main__":
    main()
