# EXP-001 任务设计：Identity–Goal Binding

> 版本：v0.1  
> 状态：Phase 0 任务草案，尚未冻结、尚未生成正式测试集  
> 日期：2026-07-29  
> 对应研究主张：P1 跨时间因果作用、P2 决策时联合绑定、S1 可读性、S2 持续与衰减、S3 恢复保真  
> 依赖文档：[`definitions.md`](definitions.md)、[`research_claims.md`](research_claims.md)、[`architecture.md`](architecture.md)  
> 适用范围：EXP-001 原生 recurrent state；其任务结构也供后续显式 Self State 实验复用。

## 1. 设计目标

EXP-001 不直接测试“模型有没有完整自我”。它先测试一个更小的必要能力：

> 一个绑定到 Agent 实例的持久约束 \(I\)，和一个需要跨干扰保持的当前目标 \(G\)，能否由历史形成的内部状态带到后续决策，并在同一次选择中共同决定唯一行动？

任务必须同时满足：

1. \(I\) 与 \(G\) 分别可单独测量；
2. 联合任务必须同时使用 \(I+G\)；
3. 当前测试提示不泄漏 \(I\) 或 \(G\) 的取值；
4. 答案位置、标签、措辞和历史长度完全平衡；
5. 可以进行 state swap、reset、random、restore 和干扰实验；
6. 结果可由程序评分，不依赖模型自述或主观评审；
7. Prompt-only 与 Memory-only 可以获得与 state 条件等量的信息；
8. 如果模型连显式条件都不会做，不能把 state-only 失败解释成状态机制失败。

## 2. 为什么替换原来的“形状偏好 × 风险策略”

原设计容易生成和评分，但存在三个问题：

- “喜欢圆形”主要像 Persona 或任意偏好，不像绑定到 Agent 实例的持续约束；
- “保守/激进”带有开放式语义和价值判断，小模型可能因常识偏差而非 state 作答；
- 两个变量都容易被当作角色描述，难以连接到身份约束与目标绑定。

新设计把首轮潜变量改为：

| 变量 | 操作性含义 | 时间尺度 | 不主张的含义 |
|---|---|---|---|
| \(I\) | 绑定到该 Agent 实例的操作域或授权域 | 持久 / 受保护 | 不等于自然人的完整身份 |
| \(G\) | 当前必须完成的任务操作 | 快变量 | 不等于长期人生目标 |

\(I\) 是“这个实例受哪条持续约束”，\(G\) 是“这个实例现在要完成什么”。这仍是合成变量，只用于测试 Persistent Self 的前置机制。

### 2.1 这是否仍然只是“记住两个词”？

在信息内容上，是的：首轮刻意只要求保存两个最小变量。它的研究价值不来自变量复杂，而来自一组更严格的问题：

- 两个变量能否跨干扰保留；
- 能否在同一次决策中联合使用；
- 交换内部 state 后，行为能否按来源方向迁移；
- 这种迁移能否排除当前 Prompt、最近 token 和非特异扰动；
- 能否在恢复和任务切换后复现。

因此，EXP-001 最多证明“recurrent state 是跨时间联合决策的因果载体”。它不能单独证明 Self。若连这个最小问题都不能成立，就没有理由立即构建更复杂的 Self Model。

## 3. 形式化定义

### 3.1 潜变量

\[
I \in \{I_0, I_1\}
\]

\[
G \in \{G_0, G_1\}
\]

四种目标状态：

```text
S_00 = I_0 + G_0
S_01 = I_0 + G_1
S_10 = I_1 + G_0
S_11 = I_1 + G_1
```

### 3.2 当前决策空间

每个测试题提供四个候选行动，完整覆盖笛卡尔积：

```text
O_00 = action(I_0, G_0)
O_01 = action(I_0, G_1)
O_10 = action(I_1, G_0)
O_11 = action(I_1, G_1)
```

正确答案：

\[
y^* = O_{I,G}
\]

四个选项每次随机排列。若只记住 \(I\)，最多只能排除两个选项；若只记住 \(G\)，同样只能排除两个选项。只有同时使用 \(I\) 和 \(G\)，才能确定唯一答案。

### 3.3 因素与干扰变量

每个样本由以下因素定义：

```text
identity_value I
goal_value G
identity_label_pair L_I
goal_label_pair L_G
history_template T_H
test_template T_Q
option_permutation π
history_order O
delay_condition D
distractor_condition X
answer_code_mapping C
generator_seed Z
```

正式数据生成器必须保存完整因子表。

## 4. 双轨任务设计

小型基础模型可能缺少稳定的指令理解能力。为避免把“不会做题”误判为“state 不携带变量”，任务分为两个轨道。

### Track S：合成协议任务

特点：

- 语法短、结构固定；
- 减少常识与社会语义；
- 主要测试状态保持、绑定和干预；
- 适合基础模型或弱 instruction-following 模型。

示意语法：

```text
INSTANCE-CONSTRAINT: domain = dax
ACTIVE-MISSION: operation = mip
```

测试：

```text
Select the valid action for this instance.
A. dax / mip
B. dax / rov
C. kel / mip
D. kel / rov
Answer:
```

实际标签不能直接固定为 `dax/kel/mip/rov`。需要在 tokenizer 调查后，从 tokenization 和基线概率相近的标签池中选择并轮换。

### Track N：自然语言微世界任务

特点：

- 使用容易理解的“授权区域 × 当前任务”；
- 检查结果是否能迁移到较自然的表达；
- 只在 Prompt-visible 能力门通过后作为确认任务。

示例：

```text
历史中的实例约束：
该实例始终被授权操作琥珀区域，而不是青色区域。

历史中的当前目标：
当前任务是检查设备，而不是封存设备。
```

测试时不再提供上述两条事实，只给出：

```text
请选择该实例现在应执行的唯一行动：
A. 检查琥珀区域设备
B. 封存琥珀区域设备
C. 检查青色区域设备
D. 封存青色区域设备
```

“琥珀/青色”“检查/封存”只是示例。正式数据使用多组语义中性、长度近似的标签和操作，并做完整交叉平衡。

### 双轨解释规则

| Track S | Track N | 允许解释 |
|---|---|---|
| 失败 | 失败 | 任务或模型能力不足，不能解释 state |
| 成功 | 失败 | 存在合成状态能力，尚未迁移到自然语言 |
| 成功 | 成功 | 支持跨表达形式的状态作用 |
| 失败 | 成功 | 检查合成语法/tokenization 是否异常 |

## 5. 轨迹形成流程

每条轨迹由四个阶段组成。

### 5.1 通用任务规则

所有条件都先接收同一份规则说明：

```text
每个实例有一个持久约束和一个当前目标。
选择必须同时匹配两者的唯一行动。
```

通用规则不包含当前实例的 \(I\) 或 \(G\) 取值，因此可以在测试提示中重复。

### 5.2 身份约束写入事件

历史中为 Agent 实例设置 \(I\)：

```text
This agent instance is bound to domain <I_value>.
This binding applies until explicitly revoked.
```

要求：

- 使用 `bound`、`authorized`、`assigned` 等多种模板；
- 明确绑定对象是“当前 Agent 实例”，不是世界中其他实体；
- 不使用“喜欢”“性格”“感觉”等 Persona 词汇；
- 对 \(I_0/I_1\) 使用镜像模板；
- 设置过程是实验输入，不称为模型自主形成身份。

### 5.3 当前目标写入事件

历史中设置 \(G\)：

```text
The active mission for this instance is <G_value>.
Keep it active until completion or cancellation.
```

要求：

- \(G_0/G_1\) 语义强度相当；
- 不让其中一个目标天然更安全、更礼貌或更常见；
- 不通过奖惩暗示某个固定答案；
- 目标与身份约束来自两个独立维度。

### 5.4 共同后缀与干扰

完成 I/G 写入后，所有轨迹接收相同后缀。后缀用于：

- 把目标信息推离当前 token；
- 消除不同历史的结尾差异；
- 测量随 token 距离的衰减；
- 插入任务切换或内容匹配干扰。

### 5.5 决策查询

查询只包含：

- 通用选择规则；
- 四个组合选项；
- 固定答案格式。

不得包含：

- 当前 \(I\) 或 \(G\)；
- 对正确区域或操作的代词提示；
- 与历史模板一一对应的特殊短语；
- 不平衡的选项长度或标点。

## 6. 历史顺序与 recency 控制

默认历史顺序不能固定为 `I → G`，否则近期变量 \(G\) 可能更容易保留。

至少包含：

```text
O_0: I → G → common suffix
O_1: G → I → common suffix
O_2: I → neutral event → G → common suffix
O_3: G → neutral event → I → common suffix
```

分析时分别报告：

- \(I\) 维度准确率；
- \(G\) 维度准确率；
- 联合准确率；
- 变量距测试位置的 token 距离；
- order × variable 的交互。

如果只有最近变量有效，应报告 recency effect，而不是 joint binding。

## 7. 任务族

### T0：Prompt-visible 能力门

把 \(I\) 和 \(G\) 都明确放在当前查询中，检查模型能否理解规则并完成四选一组合。

用途：

- 验证模型、模板和答案 token；
- 估计无记忆情况下的任务上限；
- 筛掉模型根本不会做的任务模板。

限制：

- T0 不提供 persistence 或 Self 证据；
- 只有 T0 达到预注册能力门，相关模板才能进入 T1–T6。

### T1：单变量身份约束

当前查询的两个候选行动只在 \(I\) 维度不同，目标维度固定且显式提供。

回答问题：

> state 是否能保留并使用实例身份约束 \(I\)？

### T2：单变量当前目标

当前查询的两个候选行动只在 \(G\) 维度不同，身份维度固定且显式提供。

回答问题：

> state 是否能保留并使用当前目标 \(G\)？

### T3：联合绑定

当前查询提供四个 \(I \times G\) 组合，不显式给出任何目标取值。

回答问题：

> \(I\) 与 \(G\) 是否在同一次决策中共同决定唯一行动？

T1/T2 成功而 T3 失败，记为 weak persistence，不记为 strong persistence。

### T4：延迟与无关干扰

在写入事件和测试之间插入不同长度的共同后缀：

```text
D0: 无额外干扰
D1: 短
D2: 中
D3: 长
D4: 任务切换后返回
```

具体 token 档位在 tokenizer 和上下文窗口确定后冻结。不能先看正式结果再移动档位。

干扰材料包括：

- 与 I/G 无关的中性事实；
- 格式不同的简单任务；
- 相同长度的自然语言；
- 边界 token 或会话分隔符；
- 需要短暂回答、随后回到原任务的子任务。

### T5：内容匹配干扰

在共同后缀中出现与 \(I/G\) 相同的标签，但明确属于其他实体或无关记录。

示例：

```text
An unrelated device log mentions domain dax and operation mip.
This log does not describe the current agent.
```

目的：

- 检查模型读取的是“当前 Agent 的绑定关系”，还是只追随最近出现的词；
- 让 matched-context 条件具有相同标签和近似 token 统计。

### T6：冲突诱导

加入下列干扰之一：

- 另一 Agent 的不同身份约束；
- 已作废的旧目标；
- 未授权来源给出的冲突声明；
- 与当前任务无关的假设性条件。

只有预注册为有效来源的明确撤销或重设事件，才能真的改变 \(I\) 或 \(G\)。其他冲突应进入干扰测试。

### T7：恢复、交换与插值

在同一测试查询上比较：

- original；
- restored；
- reset；
- scale-matched random；
- swapped；
- interpolated；
- layer/channel ablated。

T7 的任务输入与 T3 相同，只改变状态条件。

## 8. 实验条件

### 8.1 原生 state 条件

| 条件 | \(R_t\) 来源 | 当前 Prompt 中的 I/G | 用途 |
|---|---|---|---|
| `continuous` | 原轨迹连续运行 | 无 | 参考条件 |
| `restored` | 原轨迹 checkpoint | 无 | 恢复保真 |
| `reset` | 模型初始 state | 无 | 无状态基线 |
| `random_matched` | 尺度/shape 匹配随机 state | 无 | 非特异扰动基线 |
| `swapped_I` | 同 G、不同 I 的配对轨迹 | 无 | 身份方向迁移 |
| `swapped_G` | 同 I、不同 G 的配对轨迹 | 无 | 目标方向迁移 |
| `swapped_both` | I/G 均不同的轨迹 | 无 | 完整行为迁移 |
| `interpolated` | 两条 state 的加权组合 | 无 | 剂量—响应探索 |
| `ablated` | 指定层/通道被替换或清零 | 无 | 必要性与定位探索 |

原生 \(R_t\) 不能直接做语义字段交换。`swapped_I` 的含义是交换两条除 I 外均匹配的完整轨迹 state；`swapped_G` 同理。

### 8.2 信息基线

| 条件 | 信息提供方式 | 用途 |
|---|---|---|
| `prompt_visible` | 当前 Prompt 明示 I/G | 模型能力上限 |
| `persona_prompt` | 角色式文本描述 I/G | Persona 基线 |
| `memory_only` | 外部 Memory 检索 I/G 记录 | 长期记忆基线 |
| `matched_context` | 出现相同标签但关系无关 | 表面 token 控制 |
| `state_only` | 当前 Prompt 不含 I/G，只保留 \(R_t\) | 核心原生状态条件 |
| `prompt_plus_state` | Prompt 与 \(R_t\) 都提供 | 一致/冲突诊断，不是主条件 |

### 8.3 后续显式 PSA 条件

后续实验复用同一任务，并增加：

- `self_full`：完整 \(S_t\)；
- `self_identity_only`；
- `self_goal_only`；
- `self_field_swapped`；
- `self_gate_off`；
- `self_random_encoded`；
- `output_conditioning_only`。

EXP-001 不实现这些条件，只保证任务可复用。

## 9. State Swap 配对矩阵

四条基础轨迹：

| 来源 | I | G | 目标选项 |
|---|---|---|---|
| \(S_{00}\) | \(I_0\) | \(G_0\) | \(O_{00}\) |
| \(S_{01}\) | \(I_0\) | \(G_1\) | \(O_{01}\) |
| \(S_{10}\) | \(I_1\) | \(G_0\) | \(O_{10}\) |
| \(S_{11}\) | \(I_1\) | \(G_1\) | \(O_{11}\) |

选择性配对：

```text
I-only contrast:
  S_00 ↔ S_10
  S_01 ↔ S_11

G-only contrast:
  S_00 ↔ S_01
  S_10 ↔ S_11

Both contrast:
  S_00 ↔ S_11
  S_01 ↔ S_10
```

配对轨迹必须共享：

- 模板；
- token 长度；
- history order；
- common suffix；
- 测试题；
- option permutation；
- 除目标变量外的所有生成因子。

## 10. 标签设计与 tokenization 控制

标签池在 checkpoint 和 tokenizer 确定后建立。

### 10.1 身份标签要求

- 两个标签语义中性；
- 尽量具有相同 token 数；
- 不带明显情感、价值或安全含义；
- 不与预训练中强关联的角色名绑定；
- 在不同样本中轮换，而不是固定 \(I_0=\) 某个词。

### 10.2 目标标签要求

- 两个操作在语言频率和长度上尽量接近；
- 不使用“帮助/伤害”“安全/危险”等具有默认偏好的配对；
- 不让一个操作天然更常见；
- 在 Track S 使用无语义标签，在 Track N 使用中性操作。

### 10.3 答案代码要求

优先使用 tokenizer 中：

- 单 token；
- 无前导空格歧义；
- 基线 logit 接近；
- 可完全轮换映射；
- 不与选项内容共享 token。

若不存在合适单 token 答案，使用选项序列条件 log-likelihood，不把不同 token 长度的原始概率直接比较。

## 11. 平衡与随机化

完整生成设计至少平衡：

- \(I_0/I_1\)；
- \(G_0/G_1\)；
- 四种联合状态；
- I/G 历史顺序；
- 四个答案位置；
- 答案代码映射；
- 标签对；
- 历史模板；
- 测试模板；
- 干扰类型；
- 干扰长度。

每个高层组合下，四种 \(I \times G\) 状态必须成组生成。不得独立随机采样后再依赖大样本“自然平衡”。

## 12. 数据切分

### 12.1 开发集

用于：

- 验证 Prompt-visible 能力；
- 选择可理解的任务语法；
- 检查 tokenizer；
- 调试输出格式；
- 选择 coupling 层或 probe 超参数。

### 12.2 冻结测试集

必须与开发集隔离：

- 模板措辞；
- 标签组合；
- generator seeds；
- filler 文本；
- 可行时隔离任务微世界主题。

测试集在预注册和代码冻结后生成或解封。

### 12.3 Probe 切分

probe 训练/测试必须按轨迹组切分，不能把同一历史的轻微改写分到两边。至少增加：

- label-pair OOD；
- template OOD；
- history-order OOD；
- shuffled-label baseline。

## 13. 原始测量

本文件定义原始测量，不冻结统计检验和数值阈值。

### 13.1 选项级

- 每个候选选项的 logit 或序列 log-likelihood；
- 正确选项概率；
- 正确选项与最强错误选项的 logit margin；
- 最终选择；
- 格式有效性。

### 13.2 维度级

一个四选一答案可拆成两个维度：

```text
identity_correct
goal_correct
joint_correct = identity_correct AND goal_correct
```

同时报告：

- identity marginal accuracy；
- goal marginal accuracy；
- joint accuracy；
- 只对 I 正确；
- 只对 G 正确；
- 两者都错误。

### 13.3 干预级

- swap 后正确来源方向的概率变化；
- directional transfer indicator；
- restore logits error；
- rollback behavior recovery；
- intervention strength；
- target effect；
- non-target effect；
- 通用能力变化。

### 13.4 持续性

- 从 I/G 写入到测试的 token 距离；
- 每个 delay 档位的目标效应；
- 估计衰减曲线；
- 是否跨会话边界或任务切换；
- weak 与 strong persistence gap。

## 14. 关键派生量

具体估计方法由 `evaluation_protocol.md` 冻结。

### 14.1 Directional Transfer

当 state 从来源 \(s_a\) 换成 \(s_b\) 时，输出分布是否向 \(s_b\) 对应答案移动，而非仅发生任意变化。

概念形式：

\[
\Delta_{\text{transfer}}
=
\text{score}(y_{s_b}\mid R_{s_b})
-
\text{score}(y_{s_b}\mid R_{s_a})
\]

### 14.2 Joint Binding

联合成功不能只由两个边际准确率机械推断。需要比较：

- 实际联合准确；
- I-only 策略；
- G-only 策略；
- 最近变量策略；
- 基于边际表现得到的组合期望。

### 14.3 Specificity

干预目标维度的变化应大于非目标维度变化：

\[
\text{specificity}
=
|\Delta_{\text{target}}|
-
|\Delta_{\text{non-target}}|
\]

### 14.4 Recovery Fidelity

同时测：

- state 数值校验；
- logits 误差；
- 选项分布差异；
- 最终行为一致。

只恢复最终答案而 logits 大幅不同，不应称为完全恢复。

## 15. 能力门与解释门

### Gate A：Prompt-visible 能力

若模型在当前 Prompt 明示 I/G 时仍无法稳定完成任务：

- 该模板不得进入 state-only 确认实验；
- 优先简化语法、检查答案 token 或更换合适 checkpoint；
- 不解释为 state persistence 失败。

### Gate B：恢复保真

若 continuous 与 restored 不能在数值容差内一致：

- 停止所有语义解释；
- 检查 dtype、kernel、边界 token、序列化和模型版本。

### Gate C：单变量

若 T1/T2 均失败：

- 不直接进入 T3；
- 检查是否存在原生 state 衰减、任务理解失败或历史未被有效编码。

### Gate D：联合绑定

只有 T1/T2 有效且 T3 超出 I-only、G-only、recency 策略时，才支持 strong persistence。

### Gate E：特异因果迁移

swap 必须产生来源方向一致的变化，同时非目标能力没有整体崩溃，才支持 P1/P2 的因果结论。

## 16. 泄漏检查清单

每个生成批次自动检查：

- [ ] 测试 Prompt 不包含当前 I/G 取值；
- [ ] 文件名、样本 ID 不编码正确答案；
- [ ] 四个答案位置计数一致；
- [ ] I/G 标签在各答案位置均匀；
- [ ] 历史长度和 token 数差异在允许范围内；
- [ ] 选项长度不预测答案；
- [ ] filler 不包含关系同义词泄漏；
- [ ] matched-context 含相同标签但不含目标绑定关系；
- [ ] memory baseline 与 state 条件的信息字段相同；
- [ ] 生成 seed 与测试答案不产生简单映射；
- [ ] 评分代码不读取实验条件决定“预测”；
- [ ] probe 切分不存在轨迹近重复泄漏。

## 17. 失败模式与允许结论

| 结果 | 允许结论 |
|---|---|
| T0 失败 | 当前模型/模板不具备任务能力 |
| T0 成功，T1/T2 失败 | 未发现单变量跨时间使用证据 |
| Probe 成功，T1/T2 失败 | state 中信息可读，但没有行为使用证据 |
| T1/T2 成功，T3 失败 | weak persistence；没有 joint binding |
| T3 成功，swap 无方向迁移 | 行为相关，但未建立 state 因果迁移 |
| swap 有效但 random 同样有效 | 可能是非特异 state 扰动 |
| state-only 有效但 Memory 同样好 | recurrent state 有作用，未证明独立 Self 价值 |
| Track S 有效、Track N 失败 | 能力局限于合成协议 |
| 效应仅在 D0 有效 | 短期上下文残留 |
| 效应跨干扰、swap、restore 成立 | 原生 state 通过首轮因果载体资格 |

## 18. EXP-001 与后续显式 PSA 的边界

EXP-001 只研究：

- 冻结 RWKV；
- 原生 recurrent state \(R_t\)；
- state 保存、恢复和因果干预；
- 合成身份约束与当前目标的联合任务。

EXP-001 不包含：

- Self Encoder；
- gated coupling；
- 自动 Self Updater；
- 长期偏好形成；
- 能力估计；
- 开放式身份对话。

后续显式 PSA 使用相同任务，比较：

\[
\text{Prompt} \quad vs \quad
\text{Memory} \quad vs \quad
R_t \quad vs \quad
S_t + E_\phi + C_\psi
\]

这样可以避免任务变化与架构变化同时发生。

## 19. 生成器设计要求

后续任务生成器至少需要：

```text
generate_label_pool(tokenizer_spec)
sample_factorial_cell(seed)
render_history(I, G, labels, order, template)
render_common_suffix(delay, distractor, seed)
render_query(labels, option_permutation, answer_mapping)
validate_no_leakage(sample)
score_choice(logits_or_sequence_scores, sample)
export_manifest(sample)
```

每个样本输出：

```text
sample_id
trajectory_group_id
all_generation_factors
history_text_or_tokens
common_suffix
query
ordered_options
correct_option
correct_identity_dimension
correct_goal_dimension
token_counts
split
generator_version
```

在任务设计冻结前不生成正式测试集。

## 20. 尚待共同冻结的决策

### Q1 Track N 的首组语义

候选默认：

- 身份约束：两个语义中性的授权区域；
- 当前目标：两个语义中性的设备操作。

需要确认这些词在选定 tokenizer 和 checkpoint 上没有明显不平衡。

### Q2 Track S 的标签池

必须在 tokenizer 确定后选择。Impl-3 已登记候选池和 tokenizer-only 选择规则：
bare 与 leading-space 两种形式都必须精确 roundtrip、pair 内 token 数相同且
每种形式不超过 4 tokens；按配置中的声明顺序取前两个合格 pair。云端结果
只能形成 Batch 2 冻结候选，当前示例标签仍不能直接进入正式测试。

### Q3 通用规则是否保留在测试 Prompt

建议保留通用组合规则，只隐藏当前 I/G 取值。这样测试的是状态变量，而不是模型是否记得题型说明。

Impl-3 Prompt-visible v0.1 使用历史声明加通用规则时，RWKV-7 0.4B 在 32 条
开发轨迹中固定选择 A。v0.2 的能力门因此改用无持久性要求的显式字段模板：
直接给出 `CURRENT DOMAIN`、`CURRENT OPERATION` 和四个结构化候选，让 T0
只回答“模型是否会做精确二字段匹配”。标准 delay 不进入 T0，仍独立标定；
state-only 模板是否保留通用规则继续在 Batch 2 冻结。

项目负责人已于 2026-07-31 接受建议：state-only Prompt 保留通用组合规则，
只隐藏当前 I/G 的具体值。该决定已写入 Impl-3p 配置；仍需随最终预注册包
计算 digest。

v0.2 仍在全部 32 条轨迹中固定选择 A，因此不再继续按结果修改双字段措辞。
Impl-3b 改为能力分解：先测直接 copy code，再测 single-field matching，最后
引用既有 two-field 结果。这样可以判断当前 checkpoint 是不遵循答案接口、
不会基本查表，还是只缺少组合能力；任何一种失败都不能解释成 state
persistence 失败。

### Q4 身份写入是否需要确认步骤

建议开发阶段比较：

- 单次绑定声明；
- 声明后立即做一次验证选择；
- 多次一致绑定事件。

正式条件只能选一种，不能根据测试结果临时改变。

已新增 Impl-3p 开发门，按预先固定的“最简通过优先”规则比较上述三种方式：
标签边际化准确率至少 `0.80`，每个语义案例必须完整轮换 A–D，且查询不能
修改来源 state。三种都不通过时保持 Revise，不事后降低阈值。

Impl-3p 实际选择 `single_statement`：标签边际化准确率为 31/32
（0.96875），代码级准确率为 0.9296875，32/32 案例四轮完整且来源 state
不变。另两种更强写入均为 32/32，但按预先固定的最简通过规则不能取代
已经通过的单次声明。正式冻结候选因此只写入一次 I/G。

### Q5 干扰 token 档位

Impl-3 先按 tokenizer-only 规则标定标准 delay 候选：在 1–32 个中性 filler
units 中选择与 128 tokens 绝对误差最小者，误差必须不超过 16 tokens；
不使用任务表现参与选择。云端实测 token 数将在 Batch 2 审阅后冻结。

## 21. 任务设计冻结标准

- [x] I/G 潜变量与结论边界已定义；
- [x] 单变量和联合任务已定义；
- [x] 双轨任务和能力门已定义；
- [x] Prompt、Memory、state 和 matched-context 条件已定义；
- [x] swap 配对矩阵已定义；
- [x] 平衡、切分和泄漏规则已定义；
- [x] 原始测量和失败解释已定义；
- [ ] 共同确认 Track N 的首组语义；
- [ ] 确定 checkpoint/tokenizer 后建立标签池；
- [ ] 冻结历史模板、测试模板和通用规则；
- [ ] 冻结干扰档位；
- [x] 起草与 `evaluation_protocol.md` 的样本量、统计模型和阈值映射；
- [ ] 共同审阅并冻结任务—评价映射；
- [ ] 将状态从 Draft 改为 Frozen for Generation。

## 22. 下一步

`docs/evaluation_protocol.md` 已起草。下一步需要共同审阅并冻结：

1. 主要与次要终点；
2. 配对实验单位；
3. 样本量与 power / precision 目标；
4. 多层或混合效应统计模型；
5. directional transfer、joint binding 和 specificity 的正式公式；
6. 多重比较处理；
7. Go / Revise / Stop 数值阈值；
8. 探索集与确认集的边界。
