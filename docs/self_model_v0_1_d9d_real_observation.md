# Self Model v0.1 D9-D real within-wrapper causal isolation observation

## 结论

D9-D 真实 2.9B within-wrapper causal isolation 联合验证在单次授权范围内有效完成。runner 在同一进程、同一 persistent wrapper 中先完成 32 次 calibration capture 并冻结真实 projection，再完成 64 个 held-out fixture 的 448 个 pair、896 次 held-out forward；总计 928/928 次 forward、480 条 ledger 记录。报告 `valid=true`，状态为 `d9d_real_within_wrapper_causal_isolation_completed_claim_consumed`。

预注册因果特异性门未通过，冻结决策为 `revise_or_stop_without_self_effect_claim_or_rerun`。active-minus-zero 均值虽为正，但 99% bootstrap 下界为负；基础组合、字段层一致性、matched-random、mask 与 swap 门全部失败。synthetic active 正控制 64/64 通过，说明固定 dispatcher、目标层应用和输出变化机制能够工作，但真实 calibration-only projection 没有产生预注册要求的可靠、字段特异因果效果。

因此本次结果不能形成 Self 效果结论。D9-D single-use claim 已消费，禁止修补后重跑、自动重跑或把当前非显著结果事后改用更宽阈值解释为通过。

## 冻结端点结果

- active-minus-zero mean：`0.0009245872497558594`，均值方向为正；
- active-minus-zero 99% lower bound：`-0.0013022422790527344`，未满足严格大于零；
- positive base cases：`10/16`，低于预注册的至少 `13/16`；
- identity-level minimum positive：`2/4`，低于每层至少 `3/4`；
- goal-level minimum positive：`2/4`，低于每层至少 `3/4`；
- true-minus-matched-random 99% lower bound：`-0.0026636123657226562`，真实 projection 未可靠胜过尺度匹配随机对照；
- identity-mask specificity：`1/16`，低于 `13/16`；
- goal-mask specificity：`2/16`，低于 `13/16`；
- identity-swap following：`6/16`，低于 `12/16`；
- goal-swap following：`8/16`，低于 `12/16`；
- synthetic active changed fixture：`64/64`，通过至少 `60/64` 的正控制门；
- `all_gates_pass=false`，`self_effect_conclusion=false`。

这组结果区分了“注入机制是否能改变输出”和“真实 projection 是否携带可靠字段因果信息”：前者由 synthetic 正控制支持，后者没有得到当前协议支持。不能用正控制通过来替代真实 projection 的主要端点与特异性门。

## 执行与完整性证据

- 运行提交：`75de89e273c193c1633c7f5c60d73ce7e38cd8a2`；
- machine authorization 文件 SHA-256：`92c3db9a8a00795131d21d31cfb7a544af677264f1d57ad1f5170570ef45c616`；authorization digest：`6cfb24172ced90d9da2c730fff09fd0e2e4a3cfae86417615d1a813da8a46d57`；
- single-use claim SHA-256：`2b8a5470b1b0fa277032fe1429c02dba3b45321958d07a0b33e54b901ba5013e`；
- installed RWKV：`0.8.32`；源码 SHA-256：`75482aee89a08d2a8c8dbe628110b317fc8d0974ddffbaa52aa19190667305e0`；
- launcher 与运行期确定性检查全部通过，runtime seed=`29083101`；
- projection parameter digest：`f0b0e5300a7fc07d86d9e1f799ac67c301ce3cddbcc836aebfe454a6229067fc`；artifact digest：`916991012dda5181176ae075f01470b825a40d26c06cece4ca0dcaf11ada28c9`；
- projection artifact 文件 SHA-256：`061168d21d016d9450e72ac501426266351eb4e94db26965fa351173bd072a41`；
- raw ledger SHA-256：`0c1f91baf077e14099a07217b627ce0f465115f0024f6283872737856d997cb2`；
- report 文件 SHA-256：`0f274d8a247f913f381abc60cdb58963fa4240846271d0a7c654f623c6347eea`；报告内 digest：`2aca70f12a37420a94cda524052acea81cb1a00acb5ef7a16b39093665f99fc1`；
- integrity digest：`a4a59c5e5647d6bdb13dd9a102dee6f22939beff9875fabada135b51224c2023`；
- 总耗时约 `121.49` 秒。

回传包含 claim、完成报告与 integrity，但没有再次显示完整 machine authorization payload、最终 `git status --short` 或明确的 failure-file absence。因此授权字段可由报告与 claim 中的摘要绑定确认，运行完成性可由 `valid=true` 和 integrity 确认；不额外声称回传后服务器工作树状态或未展示文件的状态。

## 解释边界与下一门

本次是当前路线首次直接触及真实 calibration-only Self projection 的同路径因果问题。它排除了“persistent wrapper 路径差异”这一已知跨路径混杂，也证明工程注入正控制有效；但当前线性字段分离 projection 没有在 held-out 条件上形成稳定、字段特异、胜过匹配随机对照的因果信号。

下一步只能先做纯离线结果闭环与失败来源/研究路线审查，使用现有 authorization、claim、projection、ledger、report、integrity 和冻结源码，不执行模型、不修改 runner、不构成 D9-D 重跑。任何新的真实实验都必须证明研究问题和数据命名空间具有科学独立性，并取得新的明确授权；普通“继续/下一步”不授权修复或运行。

D8-C及其他历史重跑、D7-D/D7-E、正式测试集、Self 效果结论、Self Updater、raw-original 与自动重跑继续关闭。
