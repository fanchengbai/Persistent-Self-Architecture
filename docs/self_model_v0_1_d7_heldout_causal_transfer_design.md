# Self Model v0.1 D7 独立 held-out causal transfer 设计

日期：2026-08-28
状态：无模型预注册设计保持冻结；D7-B manifests 已在单独授权下实现，projection 与真实执行入口仍未实现

## 为什么 D7 不是 D6D 重跑

D6D 的目标是先在一组 non-Core fixture 上完成真实 projection 工程 pilot，但唯一尝试在第一个 capture 的 state 初始化边界失败。修复该边界后重复相同模型、训练/pilot manifests 和 160-call 计划仍属于明确禁止的 D6D 重跑。

D7 改变研究问题：由一个全新 non-Core calibration set 学到的字段分离 Self projection，能否把 identity-specific 与 goal-specific 因果影响迁移到训练时未出现的任务族，同时保持通用能力？它检验的是 held-out task-family transfer，而不是补做 D6D 的工程结果。

D7 不复用 D6D 的 identity/goal key、任务族、fixture、seed、claim、输出或定量结果。D6D 只作为“不得重跑”和“真实协议必须覆盖 `state=None`”的历史边界。

## 数据与调用结构

calibration 使用五个全新 identity key 与五个全新 goal key 的 5×5 网格，共规划 25 次只读 capture；calibration prompt 不得进入能力资格集或 held-out pilot。

held-out 端包含四个全新任务族，每族四个语义案例，每个案例进行 A/B/C/D 四代码轮换，共 64 个 fixture。每个 fixture 固定一次 OFF 预条件和 13 个条件，因此 held-out pilot 规划 896 次 forward；与 25 次 calibration capture 合计为未来单一联合 921 次 forward。

13 个条件包括 OFF、zero、synthetic 正控制、matched、identity/goal/dual swap、identity/goal/dual mask 和 identity/goal/dual norm-matched random。mask/random 在执行前被定义为对照，不允许看到结果后临时改成通过规则。

## 四道顺序门

1. **D7-B 无模型实现门**：只实现全新 manifests、统计器和失败关闭入口；不运行模型。
2. **D7-C 真实协议兼容门**：使用与 held-out payload 无关的固定 token，覆盖 single/sequence、`state=None`/prebuilt、`full_output=false/true` 的八个等价 cell。public 与 wrapper 的 OFF/zero logits 和全部 state 必须 `torch.equal`；另有两次 synthetic active 调用验证32层计数及目标层一次应用。总计未来18次 forward，失败即停。
3. **D7-D prompt-visible 能力门**：在独立 qualification namespace 上执行64条，不使用 projection。四代码边际化联合准确率至少0.80、各任务族至少0.75、identity/goal边际至少0.85、prefix roundtrip为1.0且单一预测代码占比不超过0.50。失败不进入效果 pilot。
4. **D7-E 单一联合 held-out pilot**：在另行授权下同一进程训练并冻结 projection，再加载 held-out payload并执行921次。即使通过也只形成 non-Core engineering evidence，不得直接作 Self 效果结论。

每一道门都需要独立授权、claim 和输出目录；前一门的权限不能升级到后一门。

## 预先固定的因果端点

主要端点是合取门：matched joint accuracy至少0.75；identity swap的注入identity与保留goal均至少0.75；goal swap的注入goal与保留identity均至少0.75；dual swap的注入joint至少0.70。matched相对dual mask的平均log margin及fixture-cluster bootstrap 95%下界必须为正，matched相对dual random平均log margin也必须为正。

机制与安全门同时要求：OFF/zero逐fixture精确率1.0、synthetic目标层应用率1.0、通用能力sentinel保留率至少0.95且相对OFF下降不超过0.05、来源Self State与基础模型实例字典均不改变。

这些阈值在任何 D7 模型执行前冻结，不能事后降低。

## 当前权限

研究问题和无模型预注册设计保持冻结。D7-B 已在后续单独授权下把 calibration 与 held-out manifests、确定性展开和符号 fake runtime 物化；兼容入口、能力runner、projection和pilot仍未实现。没有导入RWKV/Torch、访问权重、加载或执行模型。D6D重跑、D7真实执行、正式测试集、Self效果结论、Self Updater、raw-original及自动重跑全部关闭。

下一步只有完成 D7-B 远程无模型复验后，才能另行确认 D7-C 真实协议兼容设计。
