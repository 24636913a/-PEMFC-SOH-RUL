# CHM-D Test firewall — Final V2

CHM-D consumes only `CHM_HOU_TABLE.csv`, `CHM_SPARSE_ANCHOR_TARGETS.csv`,
`CHM_FROZEN_ANCHOR_INTERVAL_MAPPING.csv`, and Final V2 after SHA256 validation.
It does not reconstruct HOU, read a Loadcycle file, or parse a polarization CSV.

Test raw data, Test polarization, Test EIS, 2000 h/2300 h polarization, and
after-Air-Air artifacts are prohibited. The source registry remains a
pre-execution provenance record; it is not permission for CHM-D to read raw or
polarization artifacts. Any runtime constant or input not matching Final V2
must stop with `CHM_PROTOCOL_RUNTIME_MISMATCH`.
