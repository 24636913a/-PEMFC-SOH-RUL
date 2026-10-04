# One-shot unblinding procedure

An approved execution must verify hashes, construct both Test P1S inputs, generate both P1S-affine, nominal-age, and raw predictions, then write and SHA256-freeze `inputs/FROZEN_TEST_PREDICTIONS_BEFORE_TARGET_UNBLINDING.csv`. Only then may it read the 2000 h and 2300 h polarization targets. The 2000 h target cannot affect the 2300 h prediction. No refitting, tuning, or Test follow-up is allowed after target unblinding.
