# Exact P1S Test alignment semantics

AHC-TV inherits the exact alignment semantics of the PAA protocol at frozen
SHA256 `f8b5ddf76ec3f0ac40d79e637d7641cfb14d29251821a1f47533e8718910b8a5`.
The fixed state is `221||221.200`; a dwell needs at least 120 wall-clock
seconds and uses the middle 50% of its run. The dwell value is the median of
the U6-excluded active-cell mean voltage, and an interval is the median of its
eligible dwell values.

For each future checkpoint, Reader M supplies the actual polarization start.
The raw builder then finds the last legal state-run end strictly before that
timestamp and records its operating age. An interval is legal only if it has a
complete P1S dwell, all seven conditions, a distinct ID, a measurement end
strictly before the actual characterization start, and a measurement end within
48 operating hours of that raw cutoff. A partial interval additionally requires
the frozen cycle start anchor and cannot infer an unobserved remainder.

The three intervals are selected *before* support evaluation: descending
measurement-end DateTime, then descending measurement-end OpHrs2, then
ascending interval ID. Support, voltage, retention, and prospective target
error cannot replace the third selected interval. The selected checkpoint is
legal only if at least two of the three fall inside all seven frozen
Development-support bounds. Its P1S anchor functional is the median of all
three selected retentions, including a selected out-of-support interval.

The 48 h threshold is inclusive exactly at 48.0 h; a measurement end equal to
the characterization start is excluded.
