# Causality contract

For an origin row `t`, every formal augmented feature is a function only of
the 48 observed values `V[t-47:t]`. VMD and EMD may jointly use all 48 values
inside that observed block, so their values are **origin-conditioned causal
block features**, not permanently fixed online values at each earlier physical
timestamp. No decomposition result is written back to the Phase N1 row table.

The production feature API accepts one 48-point voltage history and fixed
parameters; it accepts neither a target, a full dataframe, an origin suffix,
nor an arbitrary-length series. Future-perturbation and prefix-invariance tests
are mandatory before a method can reach training.
