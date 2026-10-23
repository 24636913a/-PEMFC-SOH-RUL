# Non-blocking historical documentation debt

The following Phase MQ issues are deliberately not repaired by Phase RCR:

1. The A0 audit script labels some exclusion counters with inaccurate
   semantics.
2. Attempt 2 has no `ATTEMPT_2_EXECUTION_AUDIT.json`.
3. A0 substatus and phase-level final-status mapping are not unified.

They are historical documentation debt, not an input to raw-cycle discovery.
Phase RCR neither reruns nor changes any MQ artefact.
