# Self Model v0.1 D7-C real 2.9B public semantics compatibility observation

日期：2026-08-31

## 执行身份与范围

项目负责人在最终干净 `main` 提交 `665ac40026249fd8f1523aa2cae40486bb427d44` 上运行唯一一次 D7-C 真实 2.9B public 语义兼容门。机器授权精确匹配冻结原文，authorization digest=`50d57789…58af`；single-use claim 状态为 `d7c_single_use_compatibility_execution_claim_consumed`，报告绑定的 claim SHA-256=`fa86ad70…00e1`。

installed source 为 `rwkv==0.8.32`，`model.py` digest=`75482aee…05e0`；固定 2.9B 权重与 tokenizer 校验通过。runner 正常返回 exit code 0，18 次 forward 全部完成，运行约 15.39 秒。exit code 0 只表示 runner 完整生成报告，不表示兼容门通过。

## 通过的工程检查

- 8 个兼容 cell 完整执行：8 次 public OFF、8 次 wrapper zero。
- 2 次 synthetic active 完整执行，总 forward 数严格为 18。
- active 共触发 64 次 layer callback，只在 0-based 第 15 层应用 2 次；两次 active 输出均不同于 zero。
- `state=None` 与 prebuilt 的初始化计数符合冻结计划。
- 基础 RWKV 实例字典未改变，wrapper-owned bindings 与 context 保持稳定。
- 每个 state 输出均有 96 个形状和类型兼容、数值有限的组件。

## 失败的精确等价门

冻结标准要求每个 cell 的 public OFF 与 wrapper zero 在 logits 和全部 state 组件上逐项 `torch.equal`。真实结果为 8/8 cell 均未达到精确相等；每个 cell 只有 4/96 个 state 组件精确相等，虽然 96/96 均保持结构兼容。

| Cell | 路径 | State | Full output | Logits最大绝对误差 | State最大绝对误差 | 精确State组件 |
|---|---|---|---:|---:|---:|---:|
| 01 | one | None | false | 0.11328125 | 0.109375 | 4/96 |
| 02 | one | None | true | 0.109375 | 0.125 | 4/96 |
| 03 | one | prebuilt | false | 0.109375 | 0.125 | 4/96 |
| 04 | one | prebuilt | true | 0.109375 | 0.125 | 4/96 |
| 05 | seq | None | false | 0.03125 | 0.2490997314453125 | 4/96 |
| 06 | seq | None | true | 0.2421875 | 0.08443832397460938 | 4/96 |
| 07 | seq | prebuilt | false | 0.03125 | 0.08443832397460938 | 4/96 |
| 08 | seq | prebuilt | true | 0.2421875 | 0.08443832397460938 | 4/96 |

最终检查中 `all_logits_exact=false`、`all_states_exact=false`、`all_state_inventories_equal=false`。报告状态为 `d7c_real_public_semantics_compatibility_failed`、`valid=false`，报告 digest=`9e22f664908cb477ca006a7af0bb450fd5309ee59719f1a6243755f2ce35233d`。

## 解释边界与决定

本次是一个完整、可解释的真实兼容门失败，不是 runner 崩溃：模型加载、18 次调用、positive control、计数和清理均完成，但 wrapper zero 没有满足预注册的逐位等价标准。现有报告只能确认差异存在，不能在未经独立离线诊断的情况下判定差异来自 GPU 非确定性、public 与instrumented执行路径、调用顺序或其他具体机制。

single-use claim 已消费，D7-C 和自动重跑永久关闭。D7-D、D7-E、projection 实现/构造、正式测试集、Self 效果结论、Self Updater、raw-original 和 D6D 重跑均未授权且保持关闭。本次结果不评价 Self Model 是否有效。

下一步如继续，只能由项目负责人另行确认一个纯离线失败诊断阶段，使用现有 authorization、claim、report 和冻结源码分析差异分布与可能解释；不得访问权重、加载/执行模型、修改真实 runner 或授权修复后重跑。
