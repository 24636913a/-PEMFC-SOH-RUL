# 数据清单（Data inventory）

递归扫描了 `525` 个实际源文件；清单在 `outputs/data_inventory.csv`。其中可读 `525` 个、不可读 `0` 个。按数据类型汇总：

| data_type | files | total_MB |
| --- | --- | --- |
| EIS or electrochemical spectrum | 68 | 0.415 |
| Loadcycle durability measurement | 17 | 5527.005 |
| paper / experiment protocol | 1 | 10.875 |
| partial sensitivity test | 8 | 1008.061 |
| polarization curve | 11 | 1098.964 |
| spatially resolved EIS | 419 | 2.812 |
| text / metadata / spectrum | 1 | 0.001 |

原始文件只读取，不写回；大 CSV 均采用 `chunksize=200000`。
