# Self Model v0.1 Coupling-D5：Active Injection 离线设计审阅

日期：2026-08-21  
状态：仅离线设计；active injection 实现、真实层选择、Self projection、模型加载与效果实验均未授权

## 1. 命名纠正

项目中曾出现两个不同的“D5”：

- 当前 D1–D5 coupling 工程流程中的 **Coupling-D5**，表示 OFF 等价门之后的 active injection 设计；
- 总架构决策中的 **Architecture-D5-Self-Updater**，表示 Stage 4 的确定性受约束 Self 更新器。

二者不是同一步。本轮只审阅 Coupling-D5；Self Updater 继续不在范围内，静态 Self State 仍不可自动更新。

## 2. D4B 允许了什么

D4B 证明了冻结预条件后的 original、OFF-G1、G0 与 OFF-G2 在真实 2.9B 模型上逐元素等价。因此可以开始审阅 active 路径，但不能从该结果推出 active 路径安全、有效或具有 Self 语义。

D4 的首次调用失败和 D4A 的瞬态证据永久保留，不通过重跑改写。Coupling-D5 不需要重跑 D4/D4B。

## 3. Active contract

未来最小 active 路径只允许存在于项目代码，不修改 `site-packages`。注入点固定为 FFN 残差加法之后，同时覆盖 `forward_one` 和 `forward_seq`：

`x_out = x_in + gate × scale × projection(encoded_self)`

约束如下：

- 序列路径对每个位置广播同一 Self residual，使其语义与逐 token 持续注入一致；
- projection 输出最终必须匹配残差的 shape、dtype 和 device；
- 不修改 RWKV 三类 recurrent-state 组件的定义；
- 每个条件使用独立克隆的 recurrent state，Self State 与来源 state 均不可原位修改；
- `enabled=false` 或 `scale=0` 时完全不调用 callback，并继续满足 D4B 的精确 OFF 要求；
- 非有限 projection 必须在残差加法前失败；
- 本轮不选择真实层、scale、projection 参数或真实 Self Encoder。

## 4. 必须分开的五道门

| 门 | 内容 | 模型 | 能否声称 Self 效果 |
|---|---|---:|---:|
| Coupling-D5A | 离线 active contract 与 fake projection | 否 | 否 |
| Coupling-D5B | 项目内 active 路径静态集成和无模型复验 | 否 | 否 |
| Coupling-D5C | 真实 2.9B 非 Core 机制冒烟 | 是，需单次授权 | 否 |
| Coupling-D5D | 非 Core Self 语义效果 pilot | 是，需新的单次授权 | 仅开发性 |
| Coupling-D5E | 正式 Self 效果实验 | 是，需新预注册与授权 | 仅按预注册结论 |

每一门都需要独立确认，前一门通过不能自动启动后一门。

## 5. D5C 只证明机制

D5C 使用显式标记为 synthetic 的固定探针向量，不把它称为 Self representation。它只检查：

- original、OFF、zero-scale 仍精确一致；
- active callback 次数与层 mask 完全一致；
- active 输出有限、可重复，来源输入不变；
- scale 顺序和注入量被记录；
- 不访问正式测试集，也不计算 Self 行为结论。

任何 OFF 不等价、非有限值、非确定性、来源污染或调用数错误都必须停止，不能自动重跑或进入效果 pilot。

## 6. D5D 最小对照

未来非 Core 效果 pilot 至少保留：original、coupling-off、zero-scale、active-correct-Self、identity swap、goal swap、field mask、encoded norm-matched random、scale dose、prompt-visible reference 和通用能力副作用。

真实层、scale 和 encoder 只能在非 Core 开发数据上选择；正式测试集不得用于选择。projection 的训练数据、算法和 seed 必须在效果评估前冻结。推理时禁止把 Self State 序列化进 Prompt。

## 7. 本轮出口

本轮只产生可审计设计包。下一步若继续，只能由项目负责人明确确认 **Coupling-D5A 离线实现**；该确认仍不授权真实 active 路径、RWKV/Torch 导入、权重访问、模型加载、真实层选择或任何效果实验。

冻结的下一门确认文本为：

> 确认进入 Self Model v0.1 Coupling-D5A 离线active contract与fake projection实现；不授权Coupling-D5B/D5C/D5D/D5E、RWKV/Torch导入、权重访问、模型加载或执行、真实层选择、真实Self projection构造、Self效果实验、Self Updater或自动重跑。
