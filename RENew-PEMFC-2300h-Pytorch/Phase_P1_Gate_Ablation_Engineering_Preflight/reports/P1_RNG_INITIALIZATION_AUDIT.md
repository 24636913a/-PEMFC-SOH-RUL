# 初始化、RNG 与采样验收

状态：PASS。Seed42/52/62，全部是 fresh random initialization，无历史 checkpoint 热启动。

每个 Seed 的 G0 fresh 核心模板、隔离 RNG 下 Seed+100000 的原始 G2 门控模板在三组中共享。所有构造调用保存并恢复 CPU/CUDA/Python/NumPy RNG；正序、逆序构造逐点一致。这个 gate RNG offset 只隔离初始化流，不构成模型调参。

创建和复制完成后，各组恢复同一 run-seed RNG；独立 DataLoader generator 以 run seed 初始化。真实 PyTorch DataLoader 对合成索引 0..1379 的前两个完整 epoch 逐项核对，未读取真实窗口。dropout probe 使用相同 CUDA RNG 且保存/恢复状态，三组 mask 一致。

完整初始核心/门控、CPU/CUDA/Python/NumPy/loader 状态 SHA256，以及两个 epoch 全序列和 SHA256 均保存于 `outputs/P1_SYNTHETIC_MODEL_EVIDENCE.json`。未来授权训练必须逐 epoch 记录实际 origin 顺序并核对共同 epoch；本次索引模拟不能替代未来真实运行审计。H 不宣称 CUDA 严格位级训练确定性。
