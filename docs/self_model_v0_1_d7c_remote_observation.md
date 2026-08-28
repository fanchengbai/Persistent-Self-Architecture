# Self Model v0.1 D7-C 服务器无模型复验观察

日期：2026-08-28

## 观察范围

项目负责人在远程服务器执行 D7-C 专项测试和无模型静态入口验证，并回传终端输出。本次观察只判断锁定源码 inventory、18-call public 语义兼容计划与 synthetic fake acceptance 是否跨环境一致；不探测 installed source，不访问 calibration/held-out payload 或权重，不导入 RWKV/Torch，不加载或执行模型，也不创建机器授权或 single-use claim。

## 回传结果

- 13 项 D7-C 专项测试通过，状态为 `OK`。
- 静态报告 `valid=true`，状态为 `d7c_real_public_semantics_entry_static_verified`。
- 14 项入口检查、15 项配置检查和 14 类 synthetic acceptance 全部通过。
- 8 个兼容 cell 与未来 18 次 forward 计划完整：8 次 public OFF、8 次 wrapper zero 和 2 次 synthetic active。
- 4 个 `state=None` cell 各初始化一次 zero-state；4 个 prebuilt-state cell 均不重新初始化。
- 每个 cell 的 output、logits 和 96 个 state 组件都在 synthetic fixture 中精确相等；基础实例字典、wrapper-owned bindings 和 context 均保持稳定。
- 两次 synthetic active 共覆盖 64 次 layer callback，只在 0-based 第 15 层各应用一次，并且输出确定地不同于 zero。
- 三个 commitment 与本地一致：compatibility cells=`fd0a1a02…a70a`、cell reports=`19aaa642…f16`、active reports=`2453408a…4f2`。
- 总报告 digest 为 `8f9fb1d99e47488692dad7afb63515927f7a1e49a238c8b447e989c5e5c9d405`，与本地一致。
- `installed_source_probed=false`、`rwkv_model_imported=false`、`torch_imported=false`、`weights_accessed=false`、`model_loaded=false`、`model_executed=false`。
- calibration/held-out payload 未访问，projection 未实现或构造，D7-D/D7-E、D6D 重跑、正式测试集、Self 效果、Self Updater、raw-original 和自动重跑均保持关闭。
- 机器授权和 execution claim 均明确不存在。

## 证据边界

回传文本未包含 `git rev-parse HEAD` 或 `git status --short` 的结果，因此不能仅凭本次回传宣称服务器工作树对应提交 `3693e01` 或完全干净。不过，报告锁定的 15 个源文件 digest、三个 synthetic commitment 和总报告 digest 均与本地冻结输出一致，足以证明本轮 D7-C 源码 inventory 与无模型行为跨环境一致。

## 决定

D7-C 设计与无模型安全入口的服务器复验闭环。该结果只证明未来兼容门的调用计划、初始化语义、比较规则、目标层计数和安全入口在纯 Python fixture 上确定且跨环境一致；它不证明真实 2.9B public/wrapper 等价，不证明 installed source 兼容，不证明 projection 或任何 Self 效果。

下一步只有在项目负责人另行给出配置中冻结的逐字授权后，才允许执行一次 D7-C 真实 2.9B public 语义兼容门。该授权不自动开放 D7-D、D7-E 或任何效果实验。
