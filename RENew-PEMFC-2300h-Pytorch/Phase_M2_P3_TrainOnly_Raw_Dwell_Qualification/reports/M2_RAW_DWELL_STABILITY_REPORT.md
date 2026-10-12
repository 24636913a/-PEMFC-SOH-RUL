# Raw dwell稳定性

4312831个合法Train raw records中恢复7273个>=120s独立token保持片段。先按operation point/set current rounded3/时间gap>5s分段，无voltage-based rephasing；不跨source file拼接。

每dwell以middle50% measurement、至少30samples；等待为hold的25%(>=30s)。同一既有segment内可跨0.5h bin，缺失bin或跨segment不满足候选必要条件。记录原文件、logical record起止、raw/measurement时间、bin/segment、源SHA和协议SHA，详见M2_DWELL_IDENTITY_MANIFEST.csv。

校准仅[0,1051)，证据[1051,2103)；P50相对安静阈值见M2_STABILITY_THRESHOLD_AUDIT.json，不参考任何低高电压差或违约率。相对安静候选8，其中证据子域0。独立120s保持片段不等于physically certified dwell；**0 certified**，可用稳定dwell真实总数仍未知。

`DWELL_FOUND_BUT_QUASI_STEADY_NOT_VERIFIED`。PAA120s/middle50是操作定义，不是热/水/电化学近稳态充分条件。P50是统计设计而非sensor精度或稳态物理容差；没有以安静排序调整准则。缺少独立stabilization、材料状态和hysteresis证据，因此停止在Stage1，不执行极化pair违约率。
