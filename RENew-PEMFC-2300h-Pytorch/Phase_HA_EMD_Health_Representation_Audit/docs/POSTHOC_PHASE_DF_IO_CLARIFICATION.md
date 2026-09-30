# POSTHOC_DOCUMENTATION_CLARIFICATION — Phase DF I/O

This is a historical source-code clarification only.  In the frozen Phase DF
TimesNet adapter, `enc_in = 16` and `c_out = 16`.  The model output route then
uses output channel 0 to form the voltage DeltaV prediction, and the loss is
computed only against that channel's voltage DeltaV target.  The remaining 15
output channels have no direct future-variable supervision.

This clarification changes neither the frozen Phase DF configuration/hash,
forecasting metrics, model weights, nor the conclusion
`EMD_SPECIFIC_ADVANCE`.  It is not used to infer a health mechanism.

