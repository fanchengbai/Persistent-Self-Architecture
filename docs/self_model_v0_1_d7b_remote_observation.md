# Self Model v0.1 D7-B 服务器无模型复验观察

日期：2026-08-28

## 观察范围

项目负责人在远程服务器执行 D7-B 专项测试和纯离线验证脚本，并回传终端输出。本次观察只判断冻结源码 inventory、deterministic manifests 与符号 fake runtime 是否跨环境一致；不探测 installed source，不访问权重，不加载或执行模型，也不构造 projection。

## 回传结果

- 18 项专项测试通过，状态为 `OK`。
- D7-B 报告 `valid=true`，状态为 `d7b_manifests_and_symbolic_fake_runtime_verified`。
- 25 条 calibration record、64 条 held-out fixture、896 条 held-out 符号调用及未来 921 次联合调用计数一致。
- 13 个评分条件各出现 64 次；64 次 OFF 预条件与 832 次评分调用完整。
- 15 类 fake acceptance 全部为 `true`，包括 prompt 分离、答案代码平衡、swap/mask 字段路由、输入不变性和完整 ledger。
- expanded calibration、held-out fixture、call plan 与 symbolic ledger 四个 commitment 均与本地一致。
- 总报告 digest 为 `9800d221f13b3351ffb175815fa02e9c0659c3c6f6471ca8b76424a3a59ff683`，与本地一致。
- `installed_source_probed=false`、`rwkv_model_imported=false`、`torch_imported=false`、`weights_accessed=false`、`model_loaded=false`、`model_executed=false`、`projection_implemented=false`、`projection_constructed=false`。
- D7-C/D7-D/D7-E、D6D 重跑、正式测试集、Self 效果结论、Self Updater、raw-original 与自动重跑均保持关闭。

## 证据边界

回传文本未包含 `git rev-parse HEAD` 或 `git status --short` 的结果，因此不能仅凭本次回传宣称服务器工作树对应某一 commit 或完全干净。不过，报告中的 11 个锁定源文件 digest、四个展开 commitment 和总报告 digest 均与本地冻结输出一致，足以证明本轮 D7-B 源码 inventory 和纯离线行为跨环境一致。

## 决定

D7-B 服务器无模型复验闭环。该结果只证明 manifests 与符号 runtime contract 的确定性、完整性和安全边界，不证明真实 public/wrapper 协议兼容，不证明 projection 可训练，也不形成任何 Self 效果证据。

下一步只能在项目负责人单独确认后进入 D7-C 真实协议兼容的设计与无模型安全入口阶段；本次观察不授权 D7-C 执行或任何后续门。
