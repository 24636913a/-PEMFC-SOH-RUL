# 数据划分

|region|row_start|row_stop_exclusive|windows|forecast_queries|use|
|---|---|---|---|---|---|
|Train|0|2103|1380|33120|historical fitting only|
|Validation|2103|2404|205|4920|reused Development; primary comparison and screening|
|Assessment|2404|3005|378|9072|reused Development; existing I2 summary only|
