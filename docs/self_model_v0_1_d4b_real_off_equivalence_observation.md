# Self Model v0.1 D4B 真实 2.9B 稳态 OFF 等价门观察

日期：2026-08-20  
性质：对已授权、已完成的唯一真实运行进行只读观察；不构成重跑、D5、active injection 或 Self 效果实验授权。

## 结论

D4B 在冻结的前瞻协议下通过。真实 2.9B 模型完成 21 次调用，24 项同路线比较和 96 项跨路线比较中，logits 与全部 96 个 recurrent-state 组件均逐元素精确相等，最大绝对误差和不等元素数均为 0。

最终报告状态为 `d4b_real_off_equivalence_passed`，`valid=true`，决策效应严格为 `d5_review_candidate_only`。这不等于 D5 已获授权，也不等于 active injection 或 Self Model 效果已经实现或验证。

## 完整性链

- 执行 Git：干净 `main`，提交 `949bfa0e10a984ceb139f20e6861bb320d3fd54d`；
- authorization digest：`6b25d532015199ac08de6a5bb0c5608813657f34cfddcc002cb69497ff913968`，独立复算一致；
- 授权文件 SHA-256 与 claim 绑定一致；
- single-use claim SHA-256 与最终报告绑定一致，claim 已消费；
- 最终报告 digest：`8befb5f4b2ce90241b66aff1f43bce59645d367c14f6594169e9c454fcf36a20`，移除自摘要字段后独立复算一致；
- 报告内所有 runtime-core 检查均为真，没有失败检查。

## 冻结调用与比较盘点

| 项目 | 观察值 |
|---|---:|
| 总调用 | 21 |
| prefix snapshot | 1 |
| 固定预条件 | 4 |
| 拉丁顺序计分调用 | 16 |
| original baseline 调用 | 6 |
| OFF-G1 passthrough 调用 | 5 |
| G0 recompiled-unmodified 调用 | 5 |
| OFF-G2 instrumented 调用 | 5 |
| 同路线比较 | 24 / 24 精确相等 |
| 跨路线比较 | 96 / 96 精确相等 |
| 每次 state 比较组件 | 96 |

全部 21 次输出均被记录；16 次计分调用数量正确。所有比较的 logits 和 state 都满足 shape/dtype/device 兼容及 `torch.equal`，没有容差替代、top-1 替代或跨运行 digest 替代。

运行耗时约 17.18 秒，CUDA 峰值显存为 6,381,519,360 字节。

## 科学解释边界

本结果支持一个限定结论：在 D4B 预先冻结的 prefix 记录、每路线一次固定预条件和 4×4 拉丁计分安排下，original、OFF-G1、G0 和 OFF-G2 已进入相同的可观察稳态，OFF 路径没有产生可检测的 logits 或 recurrent-state 差异。

它不能回写或撤销 D4 的真实失败。D4 在原冻结调用顺序下观察到首调用相关的不稳定；D4A 也记录了 original/G0 的首次瞬态。D4B说明该差异不是固定预条件后持续存在的 OFF-G2 语义差异，但仍不足以唯一定位底层 CUDA、编译、方法重绑定或缓存机制。

本轮没有构造 Self projection，没有实现或执行 active injection，没有运行 Self 效果实验，也没有作确认性研究决定。

## 下一门

下一步只进入 D5 离线审阅候选：先定义 D5 的最小设计、安全边界、消融和独立授权文本。任何 D5 实现、真实模型执行、active injection 或 Self 效果实验都需要项目负责人新的明确授权；D4 与 D4B 均不得自动重跑。
