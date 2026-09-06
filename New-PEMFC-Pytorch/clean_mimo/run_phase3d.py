"""Entry point for Phase 3D absolute versus origin-relative history diagnostic."""
from __future__ import annotations

import argparse
from pathlib import Path

from phase3d_experiment import run_phase3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase-3D FC1/FC2 H=24 N=15 representation diagnostic.")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "outputs_phase3d")
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    result = run_phase3d(args.output_root, args.data_root)
    print(f"Phase-3D complete: H={result['H']}, N={result['N']}; no final-test evaluation.")


if __name__ == "__main__":
    main()
