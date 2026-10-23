# 冻结输入槽位

|slot|name|unit|Top10_selected|role|EWMA|
|---|---|---|---|---|---|
|0|cell_voltage_mean|V|True|past only|True|
|1|cell_voltage_std|V|False|past only|False|
|2|cell_voltage_range|V|False|past only|False|
|3|cell_voltage_min|V|True|past only|False|
|4|cell_voltage_max|V|True|past only|False|
|5|actual_current_A_mean|A|False|past only|False|
|6|actual_current_A_std|A|True|past only|False|
|7|actual_current_A_peak_to_peak|A|True|past only|False|
|8|actual_current_density_A_cm2_mean|A/cm2|True|past only|False|
|9|actual_current_density_A_cm2_std|A/cm2|True|past only|False|
|10|actual_current_density_A_cm2_peak_to_peak|A/cm2|True|past only|False|
|11|load_transition_count|count/bin|True|past only|False|
|12|operation_point_transitions|count/bin|False|past only|False|
|13|zero_current_fraction|fraction|True|past only|False|
