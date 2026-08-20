# Self Model v0.1 D4B 真实单次执行与观察授权记录

## 1. 项目负责人授权原文

> 授权执行 Self Model v0.1 D4B 真实2.9B稳态OFF等价门一次，并授权观察本次结果；不授权重跑D4或D4B、自动重跑、D5、active injection或Self效果实验。

该文本与
`configs/development/self_model_v0_1_d4b_real_off_equivalence.json`中的冻结授权文本
逐字一致。

## 2. 授权范围

本授权只允许在最终干净main上创建一次机器授权和single-use claim，加载冻结的
RWKV-7 G1h 2.9B资产，并执行D4B固定21次调用与120项`torch.equal`比较；同时允许
观察这一次运行产生的报告或claim后失败记录。

授权明确不允许：

- 重跑D4或D4B；
- 自动重跑或根据中间结果追加预热；
- 修改调用顺序、比较规则、容差、模型或资产；
- 授权D5、active injection或Self效果实验；
- 根据D4B通过自动作出确认性结论。

## 3. 当前机器状态

记录本文件时只确认人类授权。服务器机器授权和execution claim尚未创建，模型尚未
加载或执行，结果尚不存在。下一步必须先拉取包含本记录的最终main提交并确认工作区
干净，再通过唯一runner单次消费授权。成功或失败均停止并回传原始输出。
