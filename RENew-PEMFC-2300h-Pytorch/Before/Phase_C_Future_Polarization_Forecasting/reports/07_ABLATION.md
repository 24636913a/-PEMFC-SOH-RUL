# Minimal ablation

- A0: M1 curve trend (no dynamic representation).
- A1: M2 LSTM representation with the same scalar curve head.
- A2: M3 TimeXer representation with the same scalar curve head.
- A3: M4 concatenated LSTM+TimeXer representations with a fixed strongly regularized head.
- A4: M4 fixed 0.5/0.5 prediction average diagnostic.

The ablation is limited by 2/3/4 history-available transition curves at the three origins. Missing entries are formal `NOT_ESTIMABLE` outcomes.
