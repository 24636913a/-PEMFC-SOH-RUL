# AHC-TV Target-Blind Alignment Failure Postmortem

This directory diagnoses the frozen AHC-TV Stage-P alignment failure without
opening Test polarization target columns or invoking the one-shot runner.  It
uses only the already-frozen characterization-start metadata, formal raw
operating sources, and frozen Development alignment inputs.

Run from the repository root:

```powershell
$env:PYTHONPATH = 'RENew-PEMFC-2300h-Pytorch\Phase_AHC_TV_One_Time_Late_Life_Test_Protocol\scripts'
python -B RENew-PEMFC-2300h-Pytorch\Phase_AHC_TV_Target_Blind_Alignment_Postmortem\scripts\run_target_blind_alignment_postmortem.py
```

The diagnostic is postmortem-only. It neither alters the aborted one-shot
ledger nor supplies any authority to rerun the Test.
