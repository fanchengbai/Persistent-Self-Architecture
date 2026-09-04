# Self Model v0.1 D9-D 失败纯离线 projection/ledger 因果结构诊断

## 目的与边界

D9-D 已在一次性授权下有效完成，但真实 calibration-only projection 未通过预注册的字段因果特异性门。本阶段不修复或重跑 D9-D，只读取已经存在的 authorization、claim、projection、raw ledger、report、integrity，以及冻结 manifest 和源码，回答现有数据能解释到什么程度、哪些问题仍不可识别，以及是否存在科学独立的新研究候选。

诊断器不导入 RWKV/Torch，不访问权重，不加载或执行模型，不修改真实 runner、projection、ledger 或冻结阈值。输出进入新的纯离线诊断目录，不覆盖任何 D9-D 工件，也不创建 authorization 或 claim。

## 证据验证

真实诊断启动前必须逐项验证：

- authorization、claim、projection、raw ledger、report 的文件 SHA-256 与回传冻结值一致；
- authorization、projection、report、integrity 的内部 digest 可重新计算；
- claim 绑定 `main=75de89e273c193c1633c7f5c60d73ce7e38cd8a2`、D9-C 静态报告与 installed source；
- 32 条 calibration capture 和 448 条 held-out pair 严格按 manifest/schedule 排列；
- 每个 held-out observation 的 A/B/C/D 分数有限，margin 可由分数重新计算，96 个 state 组件、projection digest、condition 顺序与 pair order 一致；
- 480 条 ledger 对应 928 次 forward，全部计分路线仍为 persistent wrapper，任何 public、缺失、重复、乱序、非有限或摘要篡改均失败关闭。

## 分析内容

诊断报告将物化：

- 16 个 identity×goal 基础组合各自四代码轮换的 active-zero 与 active-random；
- 七类 contrast 的 rotation-level 分布、pair-order 分层、full-output 分层和输出变化计数；
- identity/goal 各水平的 active 分布，以及 mask/swap 的逐基础组合判定；
- 冻结 projection 的 identity/goal 分支 RMS、组内与跨字段 cosine、16 个 active 和向量 RMS、分支与联合数值 rank；
- 对原冻结 endpoint 的完整重算，并要求与已写 report 逐项一致。

原始 calibration ledger 只记录每次 2560 维 capture 的 SHA-256，没有保存向量。因此可以确认每个 cell 的两个 replicate 是否字节级相同，但不能恢复二者的 L2、cosine、方差或信噪比。诊断必须明确报告 `replicate_numeric_distance_identifiable=false`，不能从最终 projection 或 capture hash 反推数值稳定性。

## 路线审查

当前结果支持“注入机制能改变输出”，不支持“冻结真实 projection 产生可靠字段特异 held-out 因果效果”。以下做法会构成事后修补或 D9-D 重跑，继续禁止：

- 在 D9-D 数据上搜索 projection 增益、目标层或新阈值；
- 复用 D9-D fixture、seed、claim、projection 或结果作为新实验数据；
- 用 synthetic 正控制通过替代真实 projection 主要端点；
- 在同一命名空间重新执行或拆分原联合实验。

存在一个科学上可独立预注册的候选方向：先用全新、语义结构明确的 calibration 与完全独立的 validation 数据建立“表征可识别性门”，要求 identity/goal 在跨 replicate 和 held-out 上可解码，然后才讨论新的因果 projection 实验。该方向必须使用全新 token、fixture、seed、claim、output，不得使用 D9-D 结果作为新实验数据。当前只记录候选，不实现、不授权，也不形成 Self 效果结论。

## 执行顺序

本地只运行专项测试和静态验证，不读取服务器真实工件。代码推送并由服务器拉取后，先运行相同无模型测试与静态验证，再只读执行真实诊断器。输出固定为：

`results/development/self_model_v0_1_d9d_offline_causal_diagnostic_v01/report.json`

服务器返回完整诊断报告后才能关闭本阶段。普通“继续/下一步”不授权新的真实实验、D9-D重跑、D7-D/D7-E、正式测试集、Self效果结论、Self Updater、raw-original或自动重跑。
