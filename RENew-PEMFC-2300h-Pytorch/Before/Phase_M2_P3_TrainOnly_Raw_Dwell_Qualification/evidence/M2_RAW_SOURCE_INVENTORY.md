# Raw source inventory

这里只展示历史来源身份metadata和本轮限定读取状态，不含非Train实际观测。

|顺序|源|历史age范围h|当前状态|允许byte_end|
|---|---|---|---|---|
|1|Loadcycle_data/Loadcycle_data/Loadcycle_0-14.csv|72.78869–206.03581|TRAIN_ONLY_SOURCE_VERIFIED|401012788|
|2|Loadcycle_data/Loadcycle_data/Loadcycle_15-25.csv|206.22415–312.15411|TRAIN_ONLY_SOURCE_VERIFIED|319673021|
|3|Loadcycle_data/Loadcycle_data/Loadcycle_26-37.csv|327.81354–437.29321|TRAIN_ONLY_SOURCE_VERIFIED|329967768|
|4|Loadcycle_data/Loadcycle_data/Loadcycle_37-50.csv|437.66711–566.1698|TRAIN_ONLY_SOURCE_VERIFIED|388853998|
|5|Loadcycle_data/Loadcycle_data/Loadcycle_51-70.csv|615.66968–806.33136|TRAIN_ONLY_SOURCE_VERIFIED|572951135|
|6|Loadcycle_data/Loadcycle_data/Loadcycle_70-75.csv|806.33136–854.26343|TRAIN_ONLY_SOURCE_VERIFIED|144686909|
|7|Loadcycle_data/Loadcycle_data/Loadcycle_76-82.csv|882.59088–942.38123|TRAIN_ONLY_SOURCE_VERIFIED|180107151|
|8|Loadcycle_data/Loadcycle_data/Loadcycle_82-84.csv|942.38123–978.16644|TRAIN_ONLY_SOURCE_VERIFIED|108635588|
|9|Loadcycle_data/Loadcycle_data/Loadcycle_85-100.csv|978.27283–1130.87|TRAIN_ONLY_SOURCE_VERIFIED|460122597|
|10|Loadcycle_data/Loadcycle_data/Loadcycle_101-112.csv|1225.01–1330.22|MIXED_SOURCE_NOT_AUTHORIZED|0|
|11|Loadcycle_data/Loadcycle_data/Loadcycle_112-125.csv|1330.22–1463.38|MIXED_SOURCE_NOT_AUTHORIZED|0|
|12|Loadcycle_data/Loadcycle_data/Loadcycle_126-133.csv|1482.18–1557.08|MIXED_SOURCE_NOT_AUTHORIZED|0|
|13|Loadcycle_data/Loadcycle_data/Loadcycle_134-150.csv|1557.09–1719.31|MIXED_SOURCE_NOT_AUTHORIZED|0|
|14|Loadcycle_data/Loadcycle_data/Loadcycle_151-162.csv|1759.77–1866.77|MIXED_SOURCE_NOT_AUTHORIZED|0|
|15|Loadcycle_data/Loadcycle_data/Loadcycle_162-175.csv|1866.77–1998.17|MIXED_SOURCE_NOT_AUTHORIZED|0|
|16|Loadcycle_data/Loadcycle_data/Loadcycle_176-190.csv|2017.0–2161.63|MIXED_SOURCE_NOT_AUTHORIZED|0|
|17|Loadcycle_data/Loadcycle_data/Loadcycle_191-200.csv|2162.27–2256.8|MIXED_SOURCE_NOT_AUTHORIZED|0|

前9个全Train源用已登记历史完整字节extent核验SHA，与历史N1 source_file_manifest完全一致；其范围先由N1+RCR穷尽式timeline/count audit固定，不是扫描年龄发现截止。
Loadcycle_101-112跨1225.01–1330.22h，未发现可信的1310精确byte/record前缀；其余7源全在F2 Train之后。后8源0内容字节读取。
source_file_manifest中的historical full SHA是历史既存证据。本轮只认证9个已证明全Train文件，不计算任何mixed/outside-Train文件hash。
历史N1未发现age reversals不等于严格增加；本轮另记录age plateau。UTF-8-sig/semicolon为历史declared并在已授权源解析；未对非授权源探测BOM/引号/编码。
