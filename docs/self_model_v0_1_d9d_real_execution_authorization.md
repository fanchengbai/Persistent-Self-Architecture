# Self Model v0.1 D9-D real within-wrapper causal isolation execution authorization

日期：2026-09-04
状态：项目负责人已逐字授权一次真实 2.9B 联合工程验证；机器 authorization 与 single-use claim 尚未创建

在 D9-A 预注册、D9-B deterministic manifests/fake endpoint、D9-C calibration-only projection contract/single-use 入口及服务器无模型复验全部闭环后，项目负责人给出配置中冻结的精确授权：

> 授权执行 Self Model v0.1 D9-D 真实2.9B within-wrapper causal isolation联合验证一次（同一进程、同一persistent wrapper，固定32次calibration capture后冻结真实projection，再执行64个held-out fixture的448个pair/896次forward，共928次forward），并授权观察本次工程结果；不授权重跑D9-D、D8-C或任何历史实验、自动重跑、D7-D/D7-E、正式测试集、Self效果结论、Self Updater或raw-original路线。

该授权只覆盖一次 D9-D 真实 2.9B 联合工程尝试。运行必须在同一进程和同一 persistent wrapper 中先完成 32 次 calibration capture；projection 只能用这些 capture 拟合并通过 artifact 审计。真实 projection 必须 exclusive-create 并冻结后，入口才可读取 64 条 held-out fixture 与 448-pair schedule，再执行 896 次 held-out forward。总调用数严格为 928。

服务器 launcher 必须在包含本记录的最终干净 GitHub `main` 上检查确定性环境，现场构造并独占写入机器 authorization。authorization 必须绑定该提交、D9-C 静态报告、D9-A/B/C 冻结配置与 manifest、模型配置和 installed-source 现场摘要。通过授权验证及 installed-source 检查后，runner 必须在导入 Torch、访问权重和加载模型前创建并消费唯一 execution claim。成功或任何失败都会消费本次机会，禁止覆盖、修改阈值、拆分运行或自动重跑。

所有 calibration 与 held-out 调用都必须走同一个 persistent wrapper；public 和 raw-original 不得进入计分路线。ledger 必须按冻结顺序完整记录 32 条 capture 与 448 条 pair，共 480 条记录，只有在 928 次 forward 全部完成并通过完整性检查后才能计算冻结 endpoint。

本授权允许观察本次非正式工程结果，但不授权正式测试集或 Self 效果结论。即使字段特异因果门全部通过，结果也只能作为后续研究门的工程候选证据，不能直接证明模型具备 Self Model，更不开放 Self Updater、D7-D/D7-E、D8-C或任何历史实验重跑。

本文件只持久化项目负责人的人类授权，不创建机器 authorization、execution claim、真实 projection 或 execution output，不探测 installed source，不导入 RWKV/Torch，不访问权重，也不加载或执行模型。真实单次执行只能在本记录推送并由服务器拉取后，由冻结 launcher 在最终干净 `main` 上启动。
