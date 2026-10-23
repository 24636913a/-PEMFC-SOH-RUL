# 90% EOL 阈值审计

预注册规则：`SOH_EOL=0.90`，即 `V_EOL=0.90×V_ref,0=0.629057976 V`。95% 阈值 `0.664005641 V` 仅作辅助敏感性检查。判定只使用冻结 j*、同构名义极化曲线、非 after-Air-Air-start 的 reference points；不把任意动态低电压点当作 EOL。

| stage | ageing_hour | after_air_air_start | reference_voltage_V | SOH_U | threshold_eligible | below_90 | below_95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BoL | 0.0 | False | 0.69895331 | 1.0 | True | False | False |
| 300h | 300.0 | False | 0.6882167 | 0.98463902 | True | False | False |
| 600h | 600.0 | False | 0.68474991 | 0.97967904 | True | False | False |
| 850h | 850.0 | False | 0.6833948 | 0.97774028 | True | False | False |
| 1150h | 1150.0 | False | 0.6798183 | 0.97262334 | True | False | False |
| 1500h | 1500.0 | False | 0.68068489 | 0.97386319 | True | False | False |
| 1500h_afterAirAirStart | 1500.0 | True | 0.68460045 | 0.97946521 | False | False | False |
| 1750h | 1750.0 | False | 0.6792323 | 0.97178494 | True | False | False |
| 2000h | 2000.0 | False | 0.67763301 | 0.96949682 | True | False | False |
| 2300h | 2300.0 | False | 0.67798389 | 0.96999884 | True | False | False |
| 2300h_afterAirAirStart | 2300.0 | True | 0.68202761 | 0.97578422 | False | False | False |

90% 结论：`EOL_90_OBSERVED = False`；首次 crossing：`NOT_OBSERVED`；持续确认：`False`。95%：`EOL95_observed = False`。
