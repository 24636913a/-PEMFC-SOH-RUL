# P3 — Full450 Epoch Sensitivity

Post-P2 exploratory sensitivity only. Nine fresh G0/G1/G2 × Seed42/52/62 runs, exactly 450 epochs; best-of-450 primary and last450 mandatory. Historical P2 unchanged.

Run from this directory: `python -B scripts/run_p3_training.py --freeze`, then `python -B scripts/run_p3_training.py --run`. Freeze binds all execution code before training. No restart after ledger creation or hard failure.

Development Train/Validation only. Bounded N1 read nrows=3007, then prescribed Development filter. No Assessment/Test/raw/polarization, no whole N1 checksum. Original H preprocessing/runtime and P1 model constructors retained. New initialization compared to P1 and P2; all 450 actual sampler orders compared within seed.

Checkpoint backups are independent directory copies on the same E: volume, not off-device disaster recovery. Original/copy SHA256 must match. Weights and batch journals stay local and ignored, manifests committed. No older phase is writable.

Implementation adapted from frozen P2 source with provenance SHA in the execution manifest; unchanged metric mathematics. Science and technical failure kept separate. No automatic follow-on experiment.

Git storage: stage text uses LF to preserve generated artifact hashes. Already-frozen execution files retain their exact terminal blank lines rather than changing execution SHA after training; the phase-local whitespace policy allows EOF blank lines only, retaining trailing-space/space-before-tab checks. Scientific integrity checks are unchanged.
