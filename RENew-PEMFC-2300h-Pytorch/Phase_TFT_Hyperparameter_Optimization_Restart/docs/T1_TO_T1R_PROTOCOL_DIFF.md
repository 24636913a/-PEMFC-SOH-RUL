# T1 to T1-R protocol differences

T1-R preserves the model identity, Raw F2 strict-history task, 48-to-24
contract, folds, search space, 16-attempt budget, and Development firewall.
It changes the sampler seed to 27491 and uses a new empty study/database only.

The runner now creates a shuffled DataLoader once per independent fit and
reuses it across epochs. The decision engine requires a strict C < B matched-
budget comparison, treats Seed42 success only as pending multi-seed validation,
and counts failed/pruned/running trials against the fixed attempt budget.
