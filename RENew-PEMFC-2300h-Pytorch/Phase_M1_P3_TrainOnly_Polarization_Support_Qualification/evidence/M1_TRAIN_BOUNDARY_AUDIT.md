# F2 Train 身份与边界审计

技术边界：PASS。F2 行 `[0,2103)`；bin_start 起点 72.5 h，最后 bin_end 1310.0 h。这是聚合 bin 的身份，非原始最早观测年龄。

1. N1 `finalize_aggregate_table()` 在输出前按 bin_start 全局排序，未按 valid 或 split 删除行。
2. C `load_development()` 仅按 bin_end<1819.997738 过滤并 reset_index，没有删 invalid 行。E/F/H 继承同一顺序与 `[0,2103)`。
3. 本轮不调用旧 C/E loader，也不读3007行。无buffer FileIO只获取header和2103个完整CSV逻辑记录，quoted newline不当作记录边界。没有第2104条记录/下一字节探测。
4. 运行验证所有消费行都满足旧age过滤，所以源prefix→F2 Train是恒等映射；检查严格增序/唯一start、0.5h宽度、首末冻结边界、segment整数/增序与原1380个合法48→24窗口身份。没有重算segment。
5. 行序/筛选证明绑定 N1/C/E 源码hash与E split manifest。后续执行须重新验证这些hash；不借完整N1 Git blob证明内容。

物理读取bytes=5976261。读后仅retain显式列。fingerprint=`85973a4c37f5f39de5adb703ac9913331fca39509886c3cf280e1911303e5d67`，只含2103个Train记录的SAFE_COLUMNS。禁止使用完整源大小/mtime作为完整性证明。

raw来源另外不合格：原PAA默认上限1719.31h、chunk解析、全RCR表读取都超过本轮F2 Train。原120s/middle50%规则只能作为历史语义，不能直接执行。没有可信的每raw文件Train截止byte/record清单，本轮未读任何raw。不能读边界后的第一条raw记录来发现停止点。

Validation、Assessment、Test数值行读取：NO。
