# Self Model v0.1 D9-C 服务器无模型复验观察

## 观察范围

项目负责人于 2026-09-04 在远程服务器回传 D9-A、D9-B 与 D9-C 纯离线测试及 D9-C 静态验证报告。本观察只核对回传输出，不探测 installed source，不运行模型，不创建真实 projection，也不升级 D9-D 或其他后续权限。

回传内容未显示 `git rev-parse HEAD` 与最终 `git status --short` 的输出。因此本轮可以确认报告中锁定的源码 inventory、配置与 manifest 哈希，以及纯离线行为跨环境一致；不额外声称服务器工作树位于特定提交或处于完全整洁状态。

## 复验结果

- D9-A/B/C 专项测试：42 项，`OK`；
- 报告状态：`d9c_projection_contract_and_single_use_entry_static_verified`；
- 分类：`d9c_calibration_only_projection_contract_and_single_use_entry_static_verified_execution_not_authorized`；
- 静态检查：14/14；projection contract：9/9；projection fake acceptance：13/13；entry fake acceptance：9/9；
- 未来计划：32 次 calibration capture、448 个 held-out pair、896 次 held-out forward，共 928 次 forward；
- call plan 保持 calibration 先行、held-out 全部只走 persistent wrapper、无 public 计分路线；
- projection 在解析 held-out manifest 前冻结，endpoint 只在 projection、held-out 与完整 ledger 后运行；
- 报告 digest：`e9ad2903a5bf703b0eebcc61cdc8d5afb87f27b7838443a406df84e77fc5cc09`，与本地冻结结果一致。

projection 合成验收确认 calibration-only 拟合、artifact 审计、zero、identity/goal mask 与 swap、matched random、输入不变性及缺失/重复/非有限 capture 的失败关闭行为。fake artifact 明确不能作为研究证据。

入口合成验收确认 authorization 字段和摘要绑定、928-call 展开、call ID 唯一性、顺序完整性，以及 missing、duplicate、reordered 和 public route 的失败关闭行为。

## 安全边界与下一门

回传报告确认以下事项均未发生：

- machine authorization、single-use claim、projection、raw ledger、report、failure、integrity 或真实 output 创建；
- installed source 探测、RWKV/Torch 导入、权重访问、模型加载或执行；
- D9-D 真实执行、D8-C 或任何历史重跑；
- D7-D/D7-E、正式测试集、Self 效果结论、Self Updater、raw-original 或自动重跑。

D9-C 服务器无模型复验至此闭环。下一步只有项目负责人给出配置中冻结的逐字 D9-D 单次执行授权后，才能另行持久化人类授权记录；普通“继续/下一步”不构成 D9-D 执行授权。即使未来 D9-D 工程门通过，也不能直接形成正式 Self 效果结论。
