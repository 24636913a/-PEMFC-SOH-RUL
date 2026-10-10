# Phase F Design Revision Record

|Issue|Revision|Reason and evidence|Structural effect|Conclusion effect|
|---|---|---|---|---|
|Inherited Assessment selection|`PHASE_F_ASSESSMENT_INHERITED_SELECTION_BIAS`|Phase-E E2 was selected after exposed Assessment RMSE (7.785821 mV), despite its Validation RMSE (6.064596 mV) exceeding E0 Raw (4.797507 mV).|None|F0–F2 can only be Development exploratory.|
|C1/D0/E0 disagreement|`C1_D0_E0_REPRODUCIBILITY_UNRESOLVED`|Known aligned keys/truths but divergent outputs: 11.445854, 13.169241, 19.318765 mV. No verified cause is asserted.|None|No cross-stage score claim or causal attribution.|
|BiLSTM wording|Corrected|Bidirectional encoding within a wholly observed history window is not intrinsically future leakage. It is excluded here because F1 preserves the native encoder-to-decoder handoff, tests LSTM+causal-convolution complementarity, and avoids reopening encoder-replacement search.|None|No claim that BiLSTM is inherently invalid.|
|TimesNet relationship|`PHASE_F_TFT_ARCHITECTURE_EXPLORATION_ONLY`|TimesNet freeze/backbone-search closure remains in force. This is a user-authorized TFT architecture branch only.|None|F0–F2 cannot automatically replace TimesNet or create a cross-protocol ranking.|
|Gate interpretation|Data-driven fusion only|F2 gate is a scalar mixture based on observed history; it is not a physical constraint or mechanism detector.|F2 only|Gate values are diagnostics, not mechanism evidence.|

No formal model was trained for this report; all evidence is code/source review plus synthetic tensors.
