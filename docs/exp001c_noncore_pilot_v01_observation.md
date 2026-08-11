# EXP-001C 非 Core Pilot v01 观察记录

状态：`development_pilot_observed_closed_no_rerun`

观察授权日期：2026-08-11

范围：仅观察已完成的非 Core development pilot；未授权重跑、正式测试集或正式运行。

## 1. 完整性与边界

- commit：`d20936c76f7ed8c752d30e97ea5f2b54bacd0fe9`
- manifest SHA-256：`5e5d9a8eba2f5fe8b078ad007273e2f4e0040385ee6ab458b00b244ed2a18a8f`
- probe result SHA-256：`7791aad073ad2692a68c9a2b7e0beedb1d19d06f850ac35c52630beaca39a056`
- 记录数：32；条件数：8；每个条件 4 条。
- `formal_test_set_accessed=false`
- `contains_confirmatory_decision=false`
- `automatic_rerun_authorized=false`

观察时重新计算的 probe result SHA-256 与运行摘要完全一致。本记录不改变 EXP-001B
确认性决定，也不把 development pilot 转化为确认性证据。

## 2. 观察结果

所有 32 条记录的最高分答案均为 `A`。每个条件只有目标本来就是 A 的 1/4 条正确，
总体为 8/32（25%，四选一机会水平）。`prompt_visible_reset` 同样为 1/4，因此基础
任务正控制未通过；在该前提下不能解释 recurrent-state 条件的语义保持或损害。

| 条件 | semantic top-1 | 平均 target margin | prefix greedy exact | `>` 平均 rank | newline 匹配 |
|---|---:|---:|---:|---:|---:|
| continuous | 1/4 | -2.860840 | 0/4 | 149.25 | 0/4 |
| restored | 1/4 | -2.860352 | 0/4 | 149.25 | 0/4 |
| swapped_I | 1/4 | -2.990967 | 0/4 | 149.25 | 0/4 |
| swapped_G | 1/4 | -3.267090 | 0/4 | 149.25 | 0/4 |
| swapped_both | 1/4 | -3.288818 | 0/4 | 149.25 | 0/4 |
| reset | 1/4 | -2.117676 | 0/4 | 472.00 | 4/4 |
| prompt_visible_reset | 1/4 | -2.857178 | 0/4 | 765.50 | 0/4 |
| random_matched | 1/4 | -0.255371 | 0/4 | 11650.00 | 0/4 |

forced prefix `">\n"` 的 tokenizer roundtrip 为 32/32，但 greedy exact 为 0/32。
因此格式失败不是 tokenizer 无法表示前缀，而是模型生成分布不服从该前缀协议。
`random_matched` 的 token rank 明显恶化，说明随机状态能扰动 token 分布；由于正控制
已经失败，这不能被解释为有效的语义 assay sensitivity。

`continuous` 与 `restored` 的聚合分数近乎一致，可作为临时 safetensors state
roundtrip 工程路径的支持性证据，但不是语义保持结论。

## 3. 诊断决定

v01 结论是 `revise_positive_control_protocol_before_any_state_inference`：

1. v01 关闭，不自动重跑；
2. 不对 natural-state、swap 或 random 条件作科学解释；
3. v02 改用项目中已验证的 G1 fake-think/chat 模板；
4. 每个语义案例执行完整 A–D code rotation；
5. v02 第一阶段只运行 prompt-visible 正控制；
6. 正控制未达到预先冻结阈值时，不得进入 recurrent-state 阶段；
7. v02 的任何模型执行仍需新的独立授权。
