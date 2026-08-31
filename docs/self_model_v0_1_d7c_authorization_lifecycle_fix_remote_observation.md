# Self Model v0.1 D7-C authorization lifecycle fix remote observation

日期：2026-08-31

## 观察范围

项目负责人在远程服务器拉取 D7-C 授权生命周期修复后，执行 14 项 D7-C 专项测试和无模型静态入口验证，并回传终端输出。本次观察只验证 create→validate 生命周期修复、锁定源码 inventory、18-call 计划和 synthetic acceptance 的跨环境一致性；不复用此前人类授权，不创建机器授权或 claim，不探测 installed source，不访问 payload/权重，也不加载或执行模型。

## 回传结果

- 14 项 D7-C 专项测试通过，状态为 `OK`；其中新增用例覆盖机器授权出现后的绑定预授权报告验证。
- 静态报告 `valid=true`，状态为 `d7c_real_public_semantics_entry_static_verified`。
- 14 项入口检查、15 项配置检查和 14 类 synthetic acceptance 全部通过。
- 8 个兼容 cell、16 次 public OFF/wrapper zero 等价调用与 2 次 synthetic active 仍严格合计 18 次 forward。
- 4 个 `state=None` cell 各初始化一次 zero-state，4 个 prebuilt-state cell 不重新初始化；output、logits 和 96 个 state 组件全部精确。
- 两次 active 共覆盖 64 次 layer callback，仅在 0-based 第 15 层各应用一次；基础实例字典、owned bindings 和 context 保持稳定。
- 三个 synthetic commitment 未改变：compatibility cells=`fd0a1a02…a70a`、cell reports=`19aaa642…f16`、active reports=`2453408a…4f2`。
- 修正版总报告 digest 为 `f1134f2684f3935a9c1151a8e0b2d263564d9bc36844b0281e3da00b3b56748e`，与本地一致。
- `execution_artifacts_absent=true`，机器授权、claim、report 和 failure 均不存在。
- `installed_source_probed=false`、`rwkv_model_imported=false`、`torch_imported=false`、`weights_accessed=false`、`model_loaded=false`、`model_executed=false`。
- calibration/held-out payload 未访问；projection、D7-D/D7-E、历史重跑、正式测试集、Self 效果、Updater、raw-original 和自动重跑继续关闭。

## 证据边界

回传文本未包含 `git rev-parse HEAD` 或 `git status --short` 的结果，不能仅凭本次回传声称服务器工作树对应提交 `b59bd00` 或完全干净。不过，修正版入口与测试 digest、其余 13 个锁定源 digest、三个 synthetic commitment 和总报告 digest 全部与本地一致，足以证明本轮锁定源码 inventory、create→validate 无模型路径与 synthetic 行为跨环境一致。

## 决定

D7-C 授权生命周期修复的服务器无模型复验闭环。此前 2026-08-31 的逐字人类授权在修复前给出，未被执行或消费，但因锁定源码摘要已改变而不能复用。

下一步只有在最终修正版 `main` 上由项目负责人重新给出配置中同一逐字授权后，才允许创建唯一机器授权、探测 installed source、消费 single-use claim、访问权重并执行一次真实 18-call D7-C 兼容门。该授权仍不开放 D7-D、D7-E、projection 或 Self 效果实验。
