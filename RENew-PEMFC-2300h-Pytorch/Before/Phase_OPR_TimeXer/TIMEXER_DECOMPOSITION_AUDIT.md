# TimeXer decomposition audit

The frozen forward pass was evaluated in normalized target space as `Y=B+C`, where `B` is the TimeXer history-backbone output and `C` is the Step-Aligned FiLM correction head. After inverse target scaling, the maximum `B+C−Y` residual is 0.000119209 mV. P1 and P2 base predictions are identical for the same history (`max difference = 0` in scaled space). The RS pointwise identity `Y_RS=(1-g)Y_P1+gY_P2` also passed.

`B` alone is diagnostic only, not a fairly trained baseline: its three-fold mean RMSE is 192.382680 mV, compared with P1 24.442883 mV and P2 21.346566 mV. The checkpoint strongly relies on its condition correction to reach its formal accuracy; this does not by itself establish harmful over-reliance.
