# D8-C 首次远程无模型复验失败观察

## 观察结果

服务器针对 D8-C 首版运行 D8-A、D8-B、D8-C 合计 31 项纯离线测试，得到 2 项 failure、3 项 error。验证脚本因同一前置错误停止，未生成 D8-C 静态报告。回传片段未包含 `git rev-parse HEAD`，因此不补写服务器提交号。

机器授权与 execution claim 均明确不存在；输出中没有 RWKV/Torch、权重、模型加载或 forward 证据。因此这是无模型实现失败，不是 D8-C 真实实验 attempt，也不消费未来单次执行机会。

## 两个根因

1. `REQUIRED_CONFIRMATION` 的分段字符串在 `D8-C` 与“真实执行”之间漏掉一个空格，导致配置中的负责人确认原文无法通过逐字比较。
2. `self_model_v0_1_d8_counterbalanced_schedule.json` 是冻结的设计型 manifest，只保存种子、计数和 commitment，不直接保存 `conditioning_calls` 与 `pair_blocks`。D8-C 误把它当作已展开 schedule，因而生成了 0-call 计划。D8-B 的正确路径是从冻结 D8-A design 调用 `expand_fixtures` 和 `expand_schedule`，得到 8 次 conditioning 与 288 个 pair block，再展开为 584 次调用。

## 修复边界

修复仅恢复逐字确认和复用冻结的 D8-A/D8-B 确定性展开函数，同时让未展开 manifest 直接失败关闭，并新增回归测试。D8-A/D8-B 的 fixture、seed、schedule、commitment、阈值与未来 584-call 计划均不改变。

本轮仍不探测 installed source、不修改真实 runner、不导入 RWKV/Torch、不访问权重或 payload、不加载或执行模型，也不创建 authorization/claim。D8-C 真实执行及所有后续权限继续关闭；修复后只允许重新进行无模型静态复验。
