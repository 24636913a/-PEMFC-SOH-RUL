# Prediction–target column isolation

The future manual execution has two physically separate polarization readers.

`Reader M` calls `pd.read_csv(..., usecols=["DateTime"])` only. It derives
`T_pol,start = min(valid DateTime)` and may write a metadata artifact with the
checkpoint, source basename, valid DateTime count, start timestamp, and the
literal `DateTime` column declaration. It must not hash an entire polarization
file before prediction freeze, because hashing would read target bytes.

After `OPEN_AHC_TV_TEST`, `Stage P` may use Reader M plus Test operating raw
data to construct both 2000 h and 2300 h P1S inputs. It must freeze both
prediction rows and a SHA256 sidecar in one artifact before any target column
is opened.

`Reader T` first requires an exact, externally supplied
`OPEN_AHC_TV_TEST` marker. It then verifies artifact immutability (SHA256),
artifact completeness (the exact 21-column schema with no target-like fields),
frozen-model consistency (recomputed P1S, raw, and nominal-age predictions),
and alignment legality (support, interval identities, and temporal evidence).
Only then may it request `j.S.act [A/cm2]` and the frozen active-cell columns
U1–U5 and U7–U11. U6 remains excluded. It uses the frozen `j*=1.0 ± 0.01`
aggregation and `V_pol_BoL_jstar_1_0 = 0.6989533066 V`; no Test normalization
is refit.

`OPEN_AHC_TV_TEST` is not a security credential. It is an explicit human
execution-authorization marker that prevents a frozen protocol from opening
Test data automatically or by accident. Any future Test raw operating-data
builder must require the same externally supplied marker before reading a
2000/2300 h raw source.

If any prerequisite fails, Reader T fails before opening the polarization file.
