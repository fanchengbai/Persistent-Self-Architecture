# Self Model v0.1 D9-D real within-wrapper causal isolation execution authorization

日期：2026-09-04
状态：授权已在最终 main 上单次消费；D9-D 有效完成但预注册因果门失败，禁止重跑

在 D9-A 预注册、D9-B deterministic manifests/fake endpoint、D9-C calibration-only projection contract/single-use 入口及服务器无模型复验全部闭环后，项目负责人给出配置中冻结的精确授权：

> 授权执行 Self Model v0.1 D9-D 真实2.9B within-wrapper causal isolation联合验证一次（同一进程、同一persistent wrapper，固定32次calibration capture后冻结真实projection，再执行64个held-out fixture的448个pair/896次forward，共928次forward），并授权观察本次工程结果；不授权重跑D9-D、D8-C或任何历史实验、自动重跑、D7-D/D7-E、正式测试集、Self效果结论、Self Updater或raw-original路线。

该授权只覆盖一次 D9-D 真实 2.9B 联合工程尝试。运行必须在同一进程和同一 persistent wrapper 中先完成 32 次 calibration capture；projection 只能用这些 capture 拟合并通过 artifact 审计。真实 projection 必须 exclusive-create 并冻结后，入口才可读取 64 条 held-out fixture 与 448-pair schedule，再执行 896 次 held-out forward。总调用数严格为 928。

服务器 launcher 必须在包含本记录的最终干净 GitHub `main` 上检查确定性环境，现场构造并独占写入机器 authorization。authorization 必须绑定该提交、D9-C 静态报告、D9-A/B/C 冻结配置与 manifest、模型配置和 installed-source 现场摘要。通过授权验证及 installed-source 检查后，runner 必须在导入 Torch、访问权重和加载模型前创建并消费唯一 execution claim。成功或任何失败都会消费本次机会，禁止覆盖、修改阈值、拆分运行或自动重跑。

所有 calibration 与 held-out 调用都必须走同一个 persistent wrapper；public 和 raw-original 不得进入计分路线。ledger 必须按冻结顺序完整记录 32 条 capture 与 448 条 pair，共 480 条记录，只有在 928 次 forward 全部完成并通过完整性检查后才能计算冻结 endpoint。

本授权允许观察本次非正式工程结果，但不授权正式测试集或 Self 效果结论。即使字段特异因果门全部通过，结果也只能作为后续研究门的工程候选证据，不能直接证明模型具备 Self Model，更不开放 Self Updater、D7-D/D7-E、D8-C或任何历史实验重跑。

本文件只持久化项目负责人的人类授权，不创建机器 authorization、execution claim、真实 projection 或 execution output，不探测 installed source，不导入 RWKV/Torch，不访问权重，也不加载或执行模型。真实单次执行只能在本记录推送并由服务器拉取后，由冻结 launcher 在最终干净 `main` 上启动。

## 单次执行结果

授权已在 `main=75de89e273c193c1633c7f5c60d73ce7e38cd8a2` 上单次消费。installed RWKV 0.8.32/source lock、launcher与运行期确定性、真实projection冻结、32次calibration capture、448个held-out pair、896次held-out forward及480条ledger记录全部完成；总计928/928次forward，报告`valid=true`。

预注册因果门未通过：active-minus-zero均值为正，但99%下界=`-0.0013022422790527344`；仅`10/16`基础组合为正，真实active未可靠胜过matched random，identity/goal的level、mask和swap门全部失败。synthetic active正控制`64/64`通过，只证明注入机制工作，不证明真实projection具有字段特异Self效果。

single-use claim SHA-256=`2b8a5470…013e`，报告内digest=`2aca70f1…fc1`，integrity digest=`a4a59c5e…2023`。冻结决定为`revise_or_stop_without_self_effect_claim_or_rerun`；D9-D、D8-C、历史实验和自动重跑均关闭。详细观察见`docs/self_model_v0_1_d9d_real_observation.md`。
