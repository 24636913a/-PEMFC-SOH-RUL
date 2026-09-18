# R0 correction audit

R0 history was not modified. This R1 audit corrects its interpretation: 21.346566 mV is the formal three-fold **unweighted** P2 RMSE, while 24.475593 mV is a pooled Oracle-common prediction-point diagnostic. Validation E1/E2 have 9/2 independent events, whereas 87/20 are prediction points. Common valid origins are Fold1=233, Fold2=245, Fold3=466, all folds=944. Both windows are consistently stated as 48 history steps=24 h and 24 forecast steps=12 h.

The raw prefix terminal row was inspected before filtering: `OpHrs2=1819.49 h`, under the Development boundary. Evidence: `outputs/audit/r0_correction_audit.csv` and `test_firewall_audit.json`.
