"""Command-line entry point for the frozen development-only FC2 Phase 3A."""
from __future__ import annotations

import argparse
from pathlib import Path

from phase3a_experiment import run_phase3a_fc2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FC2 Phase-3A causal decomposition on development folds only.")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "outputs_phase3a")
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    result = run_phase3a_fc2(args.output_root, args.data_root)
    print(f"Phase-3A complete: validation-selected tau={result['validation_selected_tau_hours']} h; no final test was evaluated.")


if __name__ == "__main__":
    main()
