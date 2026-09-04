"""Convenience entry point: FC2 runs exactly the same Phase-1 protocol as FC1."""
from __future__ import annotations

import argparse
from pathlib import Path

from train import _parse_seeds, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clean Phase-1 MIMO on FC2.")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--include-residual", action="store_true")
    args = parser.parse_args()
    print(run_experiment("FC2", args.output_root, _parse_seeds(args.seeds), args.data_root, args.include_residual).to_string(index=False))


if __name__ == "__main__":
    main()
