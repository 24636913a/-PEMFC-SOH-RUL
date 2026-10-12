# Raw读取资格

`PASS_PARTIAL_TRAIN_ONLY`。9/17源完成历史内容SHA一致性验证，numeric范围72.78869–1130.87h，共4312831records。不是完整72.5–1310h raw覆盖。

证据链：commit固定的N1 source_file_manifest(实际raw SHA/size) → N1全文件timeline(row count/min/max/时序审计) ↔ RCR独立timeline(row count/range) → 本轮预登记byte[0,historical_size) → 有上限identity读取SHA一致 → 同一byte限额内parse。这是已知全Train源的验证，不以文件名/粗范围替代完整性，也不通过whole-mixed扫描找边界。

F2 bin prefix只消费M1已审计的5976261bytes并复现其2103行逻辑fingerprint；不运行旧3007读器。raw record offset为CSV logical data record，skip_blank_lines=false；quoting/embedded LF由bounded parser处理。每源记录数与首末年龄再逐项一致。没有跨源合并dwell。

精确1310 mixed-source cutoff仍NOT_ESTABLISHED。不能按秒数、均匀行长、OpHrs差或raw_sample_count估算byte offset；也没有读1310后第一条记录。只有已有可信全Train源被授权，不完整scope不妨碍限定子域审查，但限制推广到全部Train。
