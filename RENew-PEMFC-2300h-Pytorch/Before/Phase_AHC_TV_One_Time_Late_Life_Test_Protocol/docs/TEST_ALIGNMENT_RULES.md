# Test alignment rules

Each checkpoint needs three distinct P1S intervals within 48 **operating**
hours of the final legal raw state-run end before the actual polarization
characterization start. The start is the minimum valid `DateTime` read by the
metadata-only reader; neither nominal checkpoint labels nor filenames are
cutoffs. Measurement ends must be strictly before that start.

Legal intervals are first ranked by measurement-end DateTime descending,
OpHrs2 descending, then interval ID ascending. The first three are fixed before
the frozen seven-dimensional Development support is evaluated; a better
supported fourth interval cannot replace a selected interval. At least two of
the fixed three must be in support. Partial pre-characterization cycles remain
eligible only under the frozen PAA conditions. Any failed checkpoint causes
`TEST_ALIGNMENT_INCONCLUSIVE`; the window, interval count, support bounds, and
state token may not be changed.
