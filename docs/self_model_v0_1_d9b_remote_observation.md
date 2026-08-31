# Self Model v0.1 D9-B 服务器纯离线复验观察

## 观察范围

项目负责人于 2026-08-31 在远程服务器回传 D9-A 与 D9-B 纯离线测试及 D9-B 验证报告。本观察只核对回传输出，不运行模型，不升级 D9-C/D9-D 权限，也不把 fake 数据当作真实 projection 或 Self 效果证据。

回传内容未显示 `git rev-parse HEAD` 与最终 `git status --short` 的输出。因此本轮可以确认锁定源码 inventory、manifest 哈希和纯离线行为跨环境一致，但不额外声称服务器工作树位于某个提交或处于完全整洁状态。

## 复验结果

- D9-A+D9-B 专项测试：24 项，`OK`；
- 报告状态：`d9b_deterministic_manifests_and_fake_endpoint_verified`；
- 分类：`d9b_deterministic_manifests_and_fake_within_wrapper_causal_endpoint_verified_no_model`；
- 总检查：13/13；fake acceptance：12/12；
- calibration：32 条；held-out：64 条；pair：448 条；ledger：480 条；未来 forward：928 次；
- 四份 manifest 的文件 SHA-256 均与本地冻结值一致；
- calibration、held-out、schedule commitment 均保持冻结；
- 报告 digest：`6fa53a0ae84db81bcb1ec2294876bfdd15d0a9f3717dbcbae621c6110565ac91`，与本地一致。

三类合成情形符合预注册判定边界：字段特异候选通过全部冻结门，但只产生非正式、非 Self 的工程候选结论；同 wrapper 无因果信号情形未通过主要端点；active 与 random 同样变化的非特异情形未通过 random、mask 和 swap 特异性门。所有情形的 `self_effect_conclusion` 均为 `false`。

缺失、重复、乱序、public 路由、非有限值、condition 顺序改变和 calibration/held-out 阶段泄漏均被拒绝。D9-A 阈值未改变，所有计分 pair 仍只使用 `persistent_wrapper` 路径。

## 安全边界与下一门

回传报告确认以下字段全部保持关闭或未发生：

- installed source 探测、真实 runner 修改、执行入口实现；
- projection contract、真实 projection；
- authorization、single-use claim、output；
- RWKV/Torch 导入、权重访问、模型加载或执行；
- D9-C、D9-D、D7-D/D7-E；
- D8-C或任何历史重跑、正式测试集、Self效果结论、Self Updater、raw-original和自动重跑。

D9-B 的服务器纯离线复验至此闭环。下一步如果进入 D9-C，只能在项目负责人单独明确确认后，实现离线 projection contract 与无模型安全入口；普通“继续/下一步”不构成 D9-C 或任何模型执行授权。
