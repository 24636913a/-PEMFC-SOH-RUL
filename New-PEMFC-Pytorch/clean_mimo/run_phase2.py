"""Run one predeclared Phase-2 seed-42 absolute/residual experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

from phase2_experiment import run_phase2_seed42


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-2 causal persistence-anchored residual MIMO experiment.")
    parser.add_argument("--fc", choices=("FC1", "FC2"), required=True)
    parser.add_argument("--seed", type=int, default=42, help="Phase-2 diagnostic is predeclared for seed 42 only.")
    parser.add_argument("--feature-set", choices=("utot_only", "voltage_state", "state_load", "full_26"), default="full_26")
    parser.add_argument("--input-mode", choices=("absolute_inputs", "relative_voltage_inputs"), default="absolute_inputs")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "outputs_phase2")
    args = parser.parse_args()
    if args.seed != 42:
        parser.error("This frozen Phase-2 diagnostic permits seed 42 only.")
    result = run_phase2_seed42(args.fc, args.output_root, args.feature_set, args.input_mode, args.data_root, args.seed)
    print(f"Phase 2 completed: {args.fc}, seed {args.seed}; output={args.output_root / args.fc / f'seed_{args.seed}'}")
    print(f"Best validation MSE: {result['best_validation_MSE']}")


if __name__ == "__main__":
    main()
