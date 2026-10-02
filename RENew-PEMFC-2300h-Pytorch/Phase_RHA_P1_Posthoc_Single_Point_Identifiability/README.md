# Phase RHA-P1 — Posthoc Single-Operation-Point Identifiability Audit

中文：**P1 单运行点健康可辨识性事后探索审查**。

This is a `POSTHOC_EXPLORATORY`, zero-training, Development-only audit of the
operation point that was already fixed as the original RHA primary point:
`P1 = 221`. It cannot change the original dual-point RHA decision,
`CONDITION_COMMON_SUPPORT_INSUFFICIENT`, and it cannot establish SOH ground
truth, continuous SOH, RUL, cross-stack generalization, or confirmation.

The audit reuses the 146 Phase-RCR frozen complete cycles, RHA target-run
identity, 120 s stable-dwell rule, middle-50% primary window, last-25%
sensitivity window, reference-cycle convention, and seven condition variables.
No cycle rediscovery, rephasing, new operation-point selection, neural training,
or backbone search is allowed.

The primary representation is named **P1 Cycle Operating-Voltage Retention**:

`H_c,P1 = V_c,P1 / median(V_reference,P1)`.

It is explicitly not called SOH. Raw boundary-crossing rows, if read while
streaming, are disclosed and excluded before every measurement and decision.
Polarization is inaccessible until P1 support, voltage-visibility, matching, and
stable-window gates pass.

Run only after the frozen protocol commit:

```powershell
$env:PYTHONUTF8 = '1'
python scripts/run_p1_single_point_identifiability_audit.py
python -m unittest tests.test_p1_single_point_identifiability_audit
```
