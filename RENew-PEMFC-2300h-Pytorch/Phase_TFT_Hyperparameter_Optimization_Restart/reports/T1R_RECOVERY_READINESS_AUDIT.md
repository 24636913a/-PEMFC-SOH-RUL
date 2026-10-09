# T1-R recovery readiness audit

T1-R is a prospective, separate experiment. It has a different study prefix,
database path, runner module, and initialization lineage from original T1. No
old T1 database, checkpoint, curve, trial parameter, score, optimizer state,
or sampler state is permitted as an input.

This preparation contains no Optuna trial, no model training, no outer
prediction, and no Test read. `T1R_READY_FOR_SEPARATE_EXECUTION` means only
that the frozen code and protocol can await a separate explicit authorization.
