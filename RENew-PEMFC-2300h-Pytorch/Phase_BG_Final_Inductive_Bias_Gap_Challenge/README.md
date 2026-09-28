# Phase BG — Final Inductive-Bias Gap Challenge

This is the single, user-authorized exception after Phase BS. It closes three
previously incomplete modern inductive-bias categories in a fixed order:
PDFNet, xLSTMTime, then CFPT. It is not an open-ended model search.

The only permitted final outcomes are `TIMESNET_FREEZE`, `PDFNET_PROMOTED`,
`XLSTMTIME_PROMOTED`, `CFPT_PROMOTED_RAW_F2`, and
`CFPT_PROMOTED_KNOWN_TIME_PROTOCOL`. After the state machine stops,
`BACKBONE_SEARCH_CLOSED` applies permanently.

Phase BS is read-only. Phase BG uses its SHA-verified protocol, its verified
window helpers, and its TimesNet Development-only prediction caches. It never
reads Test rows, raw data, or future sensor values.

## Reproducible audit entry points

Run from the repository root. These commands verify the frozen Phase BS
contract, the author-source fingerprints, and the source-only gates. They do
not train or score a new model.

```powershell
python RENew-PEMFC-2300h-Pytorch/Phase_BG_Final_Inductive_Bias_Gap_Challenge/tests/test_protocol_recovery.py
python RENew-PEMFC-2300h-Pytorch/Phase_BG_Final_Inductive_Bias_Gap_Challenge/tests/test_xlstmtime_frozen_contract.py
python RENew-PEMFC-2300h-Pytorch/Phase_BG_Final_Inductive_Bias_Gap_Challenge/tests/test_cfpt_protocol_gate.py
python RENew-PEMFC-2300h-Pytorch/Phase_BG_Final_Inductive_Bias_Gap_Challenge/scripts/run_protocol_recovery.py
python RENew-PEMFC-2300h-Pytorch/Phase_BG_Final_Inductive_Bias_Gap_Challenge/scripts/run_cfpt_protocol_gate.py
```

`run_xlstmtime_resource_gate.py` intentionally exits non-zero when the author
implementation is resource-infeasible. The final state is terminal: the
candidate-training entry point rejects any attempted restart after
`final/FINAL_DECISION.yaml` marks the search closed.
