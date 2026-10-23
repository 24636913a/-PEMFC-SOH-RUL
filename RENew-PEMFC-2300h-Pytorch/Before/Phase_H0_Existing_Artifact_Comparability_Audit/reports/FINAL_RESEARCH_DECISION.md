# Final research decision

**DESCRIPTIVE_REUSE_PROTOCOL_CONFOUNDED**

The old H0 artifacts can be reused for same-key descriptive error reporting because key, truth, anchor and Development-boundary contracts pass. They cannot be used as a formal causal ablation against SA/P2: independent H0 training, adapter/lifecycle differences, SA joint FiLM correction, future-condition construction, and unavailable exact training-window counts are confounds. `B_joint_only≈192.383 mV` remains a joint-output component diagnostic and says nothing about independently trained H0 ability.

A genuinely matched independent H0 experiment could be justified only in a separately authorized phase. Its unique added information must be: same SA architecture/checkpoint training protocol, same chronological completed-target-purged training windows, same 14-feature/scaler/DeltaV contract, and one preregistered removal of the nine future-condition tensor. This phase does not run it.
