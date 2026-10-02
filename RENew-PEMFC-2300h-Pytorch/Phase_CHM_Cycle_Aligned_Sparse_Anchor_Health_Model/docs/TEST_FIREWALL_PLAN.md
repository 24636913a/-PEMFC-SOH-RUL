# CHM Test firewall and real read contract

CHM uses exact registry membership, not whole-path substring tests. This avoids
the false positive caused by the legal project parent
`RENew-PEMFC-2300h-Pytorch`.

All repository artifacts are read through `safe_read_frozen_artifact`, all
polarization through `safe_read_development_polarization`, and the sole new raw
source through `safe_read_development_raw_prefix`. The raw file
`Loadcycle_151-162.csv` physically contains Test rows, but CHM parses exactly
its first 268,684 data rows with parser-level `nrows=268684`; it never parses,
filters, or uses a subsequent row. The prefix must remain strictly below
`OpHrs2=1819.997738`.

The registry permits only BoL/300/600/850/1150/1500/1750 polarization files.
It rejects 2000 h, 2300 h, after-Air-Air, EIS, and all unregistered raw paths
before their contents are opened. The integration audit records parsed rows,
not a misleading claim that the mixed physical file is itself pure Development.
