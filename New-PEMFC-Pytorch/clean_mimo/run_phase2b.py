"""Command-line entry point for the frozen FC2 Phase-2b validation ladder."""
from __future__ import annotations

import argparse
from pathlib import Path

from phase2b_experiment import run_phase2b_fc2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FC2 Phase-2b R1/R2/R3 expanding validation only.")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "outputs_phase2b")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_phase2b_fc2(args.output_root, args.data_root, args.seed)
    print(f"Phase-2b complete: FC2 seed={args.seed}; validation-selected feature set={result['validation_selected_feature_set']}")


if __name__ == "__main__":
    main()
