# Posthoc correction: training-time accounting

`outputs/SEED_SUMMARY.csv` correctly records `total_training_time_s = 220.1549425` for its stated field: the sum of the three **outer refit** runs. It does not include the already-executed inner selection loops.

`outputs/POSTHOC_TRAINING_TIME_CORRECTION.csv` adds a transparent total-training-flow accounting from the saved per-fold metrics:

| Component | Three-fold time |
| --- | ---: |
| Inner selection | 222.7357605 s |
| Outer refit | 220.1549425 s |
| Total training flow | 442.8907030 s |

This is a documentation correction only. It does not modify `SEED_SUMMARY.csv`, a prediction, checkpoint, loss, gate, protocol SHA, model selection, data-access record, or any historical phase. It involved no data load and no neural-network training.
