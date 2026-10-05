# Historical late-life Test exposure disclosure

Phase N1's `phase_n1_analyze_polarization.py` scanned all 11 nominal
polarization CSVs. This included late-life Test checkpoints, whose values were
parsed and persisted in portions of the N1 SOH-anchor analysis directory and
its figures. The N1 code excluded those checkpoints from `j*` selection and
from its formula/model-selection decisions.

Accordingly, this project must not state that Test data were never accessed
before final validation. The accurate wording is:

> Late-life polarization checkpoints had been parsed and persisted during the
> initial data-foundation phase for labelled future-validation visualization,
> while they were excluded from j* selection and later AHC model
> fitting/selection. AHC-TV therefore uses a protocol-frozen late-life holdout
> evaluation with documented historical exposure rather than a pristine
> never-observed test set.

中文口径：寿命后期极化检查点曾在早期数据基础阶段被解析并作为未来验证数据标记/可视化，但未参与 j* 选择及后续 AHC 校准模型的拟合和选择。因此
AHC-TV 属于具有历史接触记录、但模型与协议事前冻结的寿命后期留出验证，而非严格意义上的“从未观察过”的盲测集。

At AHC-TV runtime, the entire historical N1
`outputs/soh_anchor_analysis/` directory and any Git-history copy of its
Test-derived outputs are prohibited inputs. This disclosure does not taint
the separately frozen PAA Development-only artifacts used by the contract.
