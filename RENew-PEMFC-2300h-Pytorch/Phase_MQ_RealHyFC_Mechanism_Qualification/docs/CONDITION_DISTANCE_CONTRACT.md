# Condition-distance contract

For each selected operation point, centers and scales are frozen solely from its ten earliest EARLY reference cycles. The pre-registered variables are actual current density, anode-inlet temperature, anode-inlet pressure, anode-inlet RH, anode hydrogen flow, anode stoichiometry and cathode stoichiometry. Variables with Development completeness below 90% cause Gate C to be unresolved; they are not imputed.

For nonconstant variables, `z=(x-median(reference))/IQR(reference)`; a zero IQR falls back to the reference standard deviation; a remaining zero scale marks a variable constant and excludes it from the sum while being reported. The pairwise distance is `sqrt(mean((z_a-z_b)^2))`. Voltage, age, SOH, EMD and any forecast error are excluded.

Aligned comparisons use the same frozen cycle phase and operation point. Controls keep the operation point but remove the phase requirement. The gate is assessed before any residual contrast: median aligned distance must not exceed 0.80 times median control distance, with common ranges reported per variable and age stratum.
