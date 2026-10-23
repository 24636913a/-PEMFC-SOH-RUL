# Polarization Configuration Compatibility

Overall BoL-versus-durability answer: **CONDITIONAL**

## Nominal polarization SOH series

**YES — directly comparable on a per-cell basis.** All 11 nominal polarization files, including BoL, report `Number of cells = 10` and `Membrane active area = 280 cm²`. The recomputed mean of the 10 active voltage channels agrees with `U.S.AveCell [V]`. The source paper states that the durability test uses a 10-cell generic HyFaB stack with 280 cm² active area and a project-specific MEA, and that the nominal RealHyFC conditions apply to polarization curves and the durability test.

Nominal protocol from the source paper: anode/cathode inlet temperature 75 °C, coolant 70 °C, humidity 50%/35%, outlet pressure 160/150 kPaG, stoichiometry 1.5/1.7, 16 load points from 0 to 1.8 A/cm² in descending and ascending order, 2 A/s ramps, 5 min dwell at j≥0.3 A/cm², 2 min below 0.3 A/cm², and 30 min conditioning at 1 A/cm².

Therefore the BoL nominal polarization voltage at the selected j* is a legal reference for the sparse SOH series.

## BoL sensitivity/EIS series

**NO for unqualified absolute stack-level comparison; configuration-aware interpretation is required.** The paper explicitly states that the BoL full-sensitivity/EIS study used a separate 8-cell stack while the durability stack used 10 cells. The released BoL sensitivity CSVs nevertheless export `Number of cells = 10`, creating a metadata-versus-paper conflict. That field is therefore not accepted as proof that the physical BoL sensitivity stack was the 10-cell durability stack. The paper also states that the two stacks used identical MEAs from the same production batch and the same 280 cm² hardware family, but they remain separate physical stacks. Per-cell sensitivity trends may be compared only with an explicit configuration caveat; absolute stack voltage and unqualified BoL-to-2300 h health normalization must not mix these records.

## Raw configuration checks

Polarization cell-count values: `[10.0]`  
Polarization active-area values: `[280.0]`  
Test files omitting the active-area field: `['Polcurve_Nominal_RHFC_2000h.csv', 'Polcurve_Nominal_RHFC_2300h.csv']`; this omission is recorded and not imputed from Test. The 280 cm² protocol value comes from Train+Validation CSV metadata and the source paper.  
BoL sensitivity exported cell-count values: `['10']` (conflicts with the paper's physical 8-cell description)  
2300 h sensitivity cell-count values: `['10']`

Evidence source: original nominal polarization CSV metadata and source paper pages 2–4. No Test performance was used in this configuration decision.
