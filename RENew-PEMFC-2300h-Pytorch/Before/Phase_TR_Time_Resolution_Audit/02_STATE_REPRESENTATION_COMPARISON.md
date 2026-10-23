# State representation comparison

States are q=0, 0<q<1, and q=1. Counts below are actual contiguous state events, never overlapping forecast windows.

## Independent events

| resolution_min | state | independent_events | total_duration_h | median_duration_h |\n| --- | --- | --- | --- | --- |\n| 10 | 0<q<1 | 630 | 105.33333333334826 | 0.1666666666666856 |\n| 10 | q=0 | 637 | 1393.3333333333494 | 1.0000000000002274 |\n| 30 | 0<q<1 | 628 | 314.5 | 0.5 |\n| 30 | q=0 | 635 | 1181.5 | 1.0 |

## Ten-minute heterogeneity inside 30-minute bins

| hidden_transition | bins | mean_current_range_A | mean_voltage_range_V |\n| --- | --- | --- | --- |\n| False | 2375 | 91.70425402772993 | 0.03884218011075468 |\n| True | 630 | 171.32201439024388 | 0.4717983707573211 |

A true within-bin state change demonstrates added expression resolution only; it does not itself establish future predictability.

## Boundary-condition summary

| previous_state | state | transitions | mean_current_A | mean_density_A_cm2 | mean_voltage_V |\n| --- | --- | --- | --- | --- | --- |\n| 0<q<1 | q=0 | 628 | 203.4954408936667 | 0.7267568567827745 | 0.7163121255447491 |\n| q=0 | 0<q<1 | 629 | 33.94182910254146 | 0.1212184007166968 | 0.2583518113933535 |
