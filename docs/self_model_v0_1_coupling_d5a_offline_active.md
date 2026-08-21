# Self Model v0.1 Coupling-D5A：离线 Active Contract 与 Fake Projection

日期：2026-08-21  
状态：无权重 fake 实现；Coupling-D5B 及以后各门未授权

## 目标

本门将已有静态 Self Store 和 deterministic fake encoder 接入一个明确标记为 fake 的投影器，再将投影结果送入无权重 RWKV-7 残差夹具。它只证明数据契约能够闭合，不证明真实 Self projection 已经存在，也不产生 Self 行为证据。

离线链路为：

`Structured Self State → fake encoder → deterministic fake projection → fake post-FFN callback`

## Fake projection

投影器使用固定 namespace 生成确定性矩阵，把 16 维 fake encoding 映射为 8 维 fake residual。矩阵没有训练，不使用 Torch，不读取模型 embedding 或权重，也不对应真实 2.9B 隐藏维度。输出必须有限，输入对象保持不可变，并记录来源 encoding、矩阵和输出 digest。

## Active contract 检查

- `enabled=false` 与 `scale=0` 完全不调用 callback；
- 两种 OFF 条件在 `forward_one` 和 `forward_seq` 上均与无 callback 基线精确相同；
- active 条件覆盖两个执行路径和固定两层，shape、dtype、device 不变；
- 相同输入重复运行完全确定；
- full scale 的输出改变量大于 half scale；
- identity 字段 swap 和 encoded norm-matched random 均改变 fake projection；
- Self State、EncodedSelf 与来源 recurrent state 不被原位修改。

这些都是 fake fixture 的工程检查，不能外推到真实 RWKV。

## 授权边界

本门不实现项目内真实 active 路径，不导入 RWKV/Torch，不访问权重，不加载或执行模型，不选择真实层，不构造真实 Self projection，不运行 Self 效果实验或 Self Updater。

下一步 Coupling-D5B 只允许在获得新确认后实现项目内真实 shape 的 active 路径并做无模型静态验证；D5A 的通过不会自动授权它。

冻结的下一门确认文本为：

> 确认进入 Self Model v0.1 Coupling-D5B 项目内active路径静态集成与无模型验证；不授权Coupling-D5C/D5D/D5E、RWKV/Torch导入、权重访问、模型加载或执行、真实层选择、真实Self projection构造、Self效果实验、Self Updater或自动重跑。
