# Self Model v0.1 D8-C real numerical identifiability execution authorization

日期：2026-08-31
状态：授权已在最终干净 main 上单次消费；D8-C 有效完成并关闭，禁止重跑

在 D8-C 协议、D8-C-I single-use runner 及服务器无模型静态复验全部闭环后，项目负责人给出配置中冻结的精确授权：

> 授权执行 Self Model v0.1 D8-C 真实2.9B数值可识别性验证一次（固定8次conditioning不计分、24个fixture/288个pair/584次forward、严格确定性策略与完整有序ledger），并授权观察本次结果；不授权重跑D8-C或历史实验、自动重跑、D7-C/D6D重跑、D7-D/D7-E、projection、正式测试集、Self效果结论、Self Updater、raw-original路线。

该授权只覆盖一次 D8-C 真实 2.9B 数值可识别性尝试：同一进程、同一 wrapper，先执行 8 次不计分 conditioning，再对 24 个全新 fixture 的 288 个 pair block 执行 576 次计分 forward，总计严格 584 次。每个 pair 的两个调用必须从同一 fixture prebuilt zero state 的独立克隆开始，并写入完整有序 ledger 后才能计算冻结的 excess-drift endpoint。

服务器 launcher 必须在最终干净 `main` 上检查严格 launcher 环境，现场构造并独占写入机器 authorization。authorization 绑定该提交、D8-C-I runner 静态报告、D8-C 远程报告、模型配置、全部冻结 manifest 和 call-ID digest。通过授权验证及 installed-source 检查后，runner 必须在导入 Torch、访问权重和加载模型前创建并消费唯一 execution claim。成功或任何失败都会消费本次机会，不能覆盖、放宽确定性策略或自动重跑。

本授权允许观察本次工程结果，但只回答 public 与 wrapper-zero 的跨路径数值差异是否超过各自路径内重复性背景。即使主要 endpoint 为正，也不能据此形成 Self Model 效果结论，更不授权 projection、正式测试集、Self Updater、raw-original、D7-D/D7-E 或任何历史实验重跑。

在执行前，本文件只持久化项目负责人的人类授权，不创建机器 authorization、execution claim 或 execution output，不探测 installed source，不访问权重，也不加载或执行模型。真实单次执行随后只能在包含本记录的最终干净 GitHub `main` 上由冻结 launcher 启动；下节记录该条件满足后的唯一执行结果。

## 单次执行结果

授权已在干净 `main=e0ab61a58394e6eaef2567aa3a988afa6e47738c` 上单次消费。机器 authorization、installed-source 探测、single-use claim、固定 2.9B 资产校验和模型加载均有效；584/584 次 forward、8 条 conditioning 与 288 条 scored pair 全部完成，296 行 ledger 和 integrity 验证通过。

冻结主要端点检测到 route-specific excess drift：24/24 fixture 为正，平均 excess drift=`0.0032601490100549004`，99% bootstrap 下界=`0.002858416711489245`。报告状态为 `d8c_real_numerical_identifiability_completed`、`valid=true`，决策仅为非 Self 工程证据；它不证明路径等价或 Self 效果。

claim SHA-256=`85403630…05db`，报告内 digest=`a0dad92b…2ac5`，原始 ledger SHA-256=`72a3e919…8e73`。本次 claim 已消费，D8-C、历史实验和自动重跑均关闭；详细观察见 `docs/self_model_v0_1_d8c_real_observation.md`。
