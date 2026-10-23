# 已实际检查的资源
 exact whitelist共有275个路径：31份cell-EIS、34份spatial/reference-EIS、201份新鲜BoL敏感性头部、7份名义极化曲线、ReadMe及原论文。DATA_RESOURCE_INVENTORY.csv记录273份测量资源，SOURCE_CONTENT_MANIFEST.csv同时记录方法文件及全部SHA/字节。
 BoL核心cellEIS为两组各10通道（约50.5A与505A），1150h为10单电池加1复合参考；空间系列各16导出通道加1参考文件00。本地真实ASCII约5–7KB/谱，Git中是LFS指针；275个local内容均匹配LFS OID与大小。git-lfs3.7.1可用，没有执行fetch/pull/全库下载，也未打开混合zip。
 核心真实日历2025-01-08/09与04-24/25，独立新鲜敏感性头部2025-10-31至11-06。BoL名称不能代替物理堆/发布时间。频率、复数阻抗或模相、significance、elapsed时间、导出电压/电流齐备；没有精确durability OpHrs2、发布时刻、完整实际RH/P或有效界面状态。
 温度头部的0.x±x°C或unknown不视为75°C实际电化学温度：官方说明该字符串只有正确配置测温卡才有传感器意义。场景名义条件来自论文，不是每谱内部状态实测证明。
  低负载文件名按ReadMe一般规则相加会给60.4A，但实测头部为约50.495A，记录命名歧义，不用名称覆盖实测值；低负载图约0.18A/cm²由头部I/280得到。EIS_DURABILITY_CALENDAR_BRACKETING表只记录实际日历与前后已观测bin，不插值填造EIS OpHrs2或结果发布时间。
 禁止文件只枚举文件名和大小，写入EXCLUDED_TEST_FILE_METADATA_ONLY.csv；不读内容、不算Test全文件SHA、不解析2300h/2000h值。没有混入其他数据集。
