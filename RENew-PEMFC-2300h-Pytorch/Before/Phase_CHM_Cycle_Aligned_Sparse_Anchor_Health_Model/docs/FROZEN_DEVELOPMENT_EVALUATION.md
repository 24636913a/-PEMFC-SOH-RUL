# Frozen CHM Development evaluation

Final V1 has six outer leave-one-anchor-out folds over 300, 600, 850, 1150,
1500, and 1750 h. Every outer fit excludes its held-out anchor from fitting,
profile selection, and early stopping.

Only three predeclared profiles are eligible: A BALANCED `(1, 1, 0.1)`, B
ANCHOR_EMPHASIS `(3, 1, 0.1)`, and C OBSERVATION_EMPHASIS `(1, 3, 0.1)`.
Anchor, observation, and smoothness terms are means, never raw sums. Each
outer fold has five inner anchor holdouts and three seeds; therefore profile
selection consists of exactly 270 inner fits and 18 outer deterministic
refits. A tie within `1e-8` is resolved A, then B, then C.

This phase has executed none of those fits. Bootstrap is excluded from profile
selection and is skipped entirely when deterministic Gates 1–4 fail. The six
outer predictions, their aggregate metrics, and the secondary online diagnostic
remain future Phase CHM-D work under this unchanged protocol.
