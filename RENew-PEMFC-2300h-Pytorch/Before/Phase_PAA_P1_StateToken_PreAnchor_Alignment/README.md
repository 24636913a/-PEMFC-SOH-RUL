# Phase PAA — P1 State-Token Pre-Anchor Alignment Audit

中文：**P1固定子状态预极化锚点对齐审查**。

This is a `POSTHOC_EXPLORATORY`, zero-training Development-only audit of the
pre-frozen P1 substate `221||221.200`. It is a new estimand: a single
state-token measurement summarized once per frozen complete cycle interval, or
once per pre-characterization partial interval for local sparse-anchor
alignment only.

It cannot change `RAW_CYCLE_STRUCTURE_QUALIFIED`, the original dual-point RHA
status `CONDITION_COMMON_SUPPORT_INSUFFICIENT`, or the P1 audit status
`P1_ANCHOR_ALIGNMENT_INCONCLUSIVE`. P1S* Operating-Voltage Retention is not
SOH ground truth and this phase cannot create a continuous SOH label, RUL
model, neural model, or backbone result.

The raw reader has a stricter cutoff than prior audits: it opens only the frozen
source files needed through the final 1750 h checkpoint's pre-characterization
cutoff (`1719.31 h`) and rejects Test raw rows. Polarization remains blocked
until support, long-term drift, joint condition support, matching direction,
and dwell sensitivity pass.

Run only after the protocol commit:

```powershell
$env:PYTHONUTF8 = '1'
python scripts/run_p1_state_token_preanchor_audit.py
python -m unittest discover -s tests -p 'test_*.py'
```
