# EXP-001：RWKV State Persistence & Swap

> 状态：Draft / Blocked by Phase 0 / 待预注册  
> 阶段：Phase 0 → Phase 1  
> 目的：验证 RWKV recurrent state 的可复现操纵能力，并测试历史形成的两个行为约束能否在同一决策中共同生效。

> 执行限制：本文目前只是用于暴露设计缺口的实验草案。完成并共同冻结 `docs/definitions.md`、`docs/research_claims.md`、`docs/architecture.md`、`docs/task_design.md`、`docs/evaluation_protocol.md`、`docs/state_format.md`、`docs/implementation_spec.md` 和工程数值参数前，不启动远程正式实验，也不把任何探索性运行计入研究结果。

## 1. 实验定位

本实验不尝试证明模型拥有 Self。

它只回答三个更基础的问题：

1. RWKV state 能否被可靠保存、恢复和跨进程复现？
2. 不同历史轨迹产生的 state 是否会对相同后续输入产生不同影响？
3. 交换 state 后，行为差异是否随 state 迁移？

只有三项均成立，后续才值得研究 state 中是否存在 Self 候选信息。

## 2. 假设

### H0：恢复保真

在模型、输入、精度和采样配置固定时，保存并恢复 state 后，后续 token logits 与原轨迹一致。

### H1：单约束保持

历史轨迹中形成的行为约束 A 或 B，能够在延迟和干扰后影响对应选择。

### H2：联合约束共同生效

在必须同时应用 A 与 B 的任务中，模型行为显著优于只保留其中一个约束的条件。

### H3：State Swap 行为迁移

对完全相同的后续输入，交换两个轨迹的 state 后，选择分布向 state 来源轨迹的行为方向迁移。

### H4：干预强度响应

State interpolation 或分层消融造成的行为效应随干预强度呈可解释变化，而不是随机跳变。

## 3. 行为变量

首版避免复杂人格，选择两个正交、可程序化评分，并更贴近 PSA 操作性定义的变量：

```text
A：实例身份约束 / 操作域
  A0 = 当前 Agent 实例绑定操作域 I0
  A1 = 当前 Agent 实例绑定操作域 I1

B：当前任务目标 / 操作
  B0 = 当前目标为操作 G0
  B1 = 当前目标为操作 G1
```

形成四种轨迹状态：

```text
S_00 = A0 + B0
S_01 = A0 + B1
S_10 = A1 + B0
S_11 = A1 + B1
```

最终任务提供完整的 `I × G` 四个组合，只有同时使用 A+B 才能确定唯一行动。任务分为合成协议 Track S 和自然语言微世界 Track N；变量、模板、平衡、干扰和泄漏规范以 [`docs/task_design.md`](../../docs/task_design.md) 为准。

## 4. 实验条件

### 4.1 State 条件

- `original`：使用轨迹自身 state；
- `restored`：从磁盘恢复同一 state；
- `reset`：使用模型初始 state；
- `random`：使用尺度匹配的随机 state；
- `swapped`：使用另一轨迹的 state；
- `interpolated`：两个 state 按比例插值；
- `ablated_layer`：清零或替换指定层 state；
- `ablated_channel`：清零或替换指定 head / channel。

### 4.2 信息基线

- `prompt_only`：将 A/B 直接放入当前提示；
- `memory_only`：从外部记录检索 A/B 并注入提示；
- `state_only`：当前提示不出现 A/B，只使用历史 state；
- `prompt_plus_state`：同时提供显式提示和 state；
- `matched_context`：控制当前上下文 token 数和表面形式。

## 5. 任务

### I0：基础设施确定性恢复

给定固定前缀和固定后续输入：

1. 连续运行得到参考 logits；
2. 保存中间 state；
3. 在同一进程恢复并继续；
4. 在新进程恢复并继续；
5. 比较逐 token logits 和输出。

### T0：Prompt-visible 能力门

- 在当前 Prompt 中明确给出 I/G；
- 验证模型是否理解组合规则、选项和答案格式；
- 未通过的模板不得进入 state-only 实验；
- 本任务不提供 persistence 或 Self 证据。

### T1/T2：单约束探测

- T1 只测试实例身份约束 A；
- T2 只测试当前目标 B；
- 记录正确选项概率、margin 和准确率。

### T3：联合约束决策

构造必须同时应用实例身份约束 A 和当前目标 B 才能选对的选项，例如：

```text
选项 1：操作域 I0 + 操作 G0
选项 2：操作域 I0 + 操作 G1
选项 3：操作域 I1 + 操作 G0
选项 4：操作域 I1 + 操作 G1
```

每个 `S_ab` 都对应唯一正确选项。

### T4–T6：延迟、内容匹配与冲突干扰

在形成 A/B 后插入：

- 不同长度的无关文本；
- 与 A/B 内容相似但无关的文本；
- 任务切换；
- 边界 token；
- 矛盾诱导信息。

测量约束信号随距离的衰减和抗干扰能力。

### T7A：交换与恢复

- 在相同测试输入上运行 `S_00` 至 `S_11`；
- 交换任意两个 state；
- 恢复历史快照；
- 检查行为是否迁移或恢复。

### T7B：消融与插值

- 按层、head 和 channel 干预；
- 定位与 A、B、联合任务相关的 state 区域；
- 检查单约束和联合约束是否由相同或不同区域承载。

## 6. 控制混淆

### 内容控制

- 四类历史使用相同 token 数和相同模板；
- 交换变量标签与输出 token，避免固定词语偏差；
- 对身份/目标标签、选项顺序和答案 token 做完全平衡；
- 测试提示不直接出现轨迹标签。

### 最近上下文控制

- 在所有轨迹后添加相同后缀；
- 改变干扰长度；
- 对比 state-only 与当前 Prompt 显式注入。

### 数值控制

- 固定模型和 tokenizer 版本；
- 固定精度、kernel 和设备；
- 同时记录 state RMS、norm、stable rank 和异常值；
- 记录 `<|endoftext|>` 等边界 token；
- 在 fp32 参考实现与加速 kernel 间交叉验证关键样本。

### Probe 控制

- probe 训练集和评估集分离；
- 使用打乱标签和随机 state 基线；
- probe 成功不视为 H2/H3 成立；
- 核心结论必须来自实际 logits 或行为变化。

## 7. 指标

### 基础设施指标

- state 序列化成功率；
- 恢复后的最大 logits 绝对误差；
- 恢复后的 KL divergence；
- 跨进程输出一致率；
- state RMS、norm 和 stable rank。

### 行为指标

- A 单约束准确率；
- B 单约束准确率；
- A+B 联合准确率；
- 正确选项 logit margin；
- swap 后的 behavioral transfer rate；
- rollback recovery fidelity；
- intervention effect size；
- 信号随干扰长度的衰减曲线。

### Weak / Strong Persistence

- Weak：A、B 分别能在单约束任务中被行为表达；
- Strong：A、B 能在联合任务的同一决策中共同决定唯一选项；
- Gap：Weak 高但 Strong 低时，说明模型记得组成部分，却没有把它们共同用于决策。

## 8. 执行顺序与决策门

### Gate 0：环境可运行

- 基础模型和 tokenizer 可加载；
- 固定输入能完成推理；
- state API 的 shape、dtype 和层结构已记录。

失败处理：先解决环境和模型兼容性，不进入行为实验。

### Gate 1：恢复保真

- 同进程和跨进程恢复均达到预注册误差范围；
- state 文件包含模型版本、shape、dtype 和 checksum。

失败处理：检查精度、kernel、边界 token 和序列化，不解释任何语义结果。

### Gate 2：单变量可迁移

- A 或 B 至少一个变量在 `state_only` 条件下显著优于 reset/random；
- swap 后行为方向随 state 来源改变。

失败处理：判断基础模型是否具备任务理解能力；必要时使用更简单的合成序列或小型任务微调。

### Gate 3：联合约束

- 单约束任务有效；
- 联合任务表现不能仅由 A 或 B 单独解释；
- weak / strong persistence 均可计算。

失败处理：记录为“成分可保持但未共同实例化”，不把它包装成成功。

### Gate 4：定位与稳健性

- 在多个模板、答案排列和随机种子上重复；
- 插值或消融产生可重复效应；
- 通用语言建模能力没有因干预完全崩溃。

完成后才进入显式 Self Model 实验。

## 9. 预注册前必须确定

- 具体 RWKV-7 checkpoint；
- 是否存在适合本任务的 instruction-tuned 版本；
- state 序列化格式；
- kernel 与数值精度；
- 测试模板和答案 token；
- 随机种子数量；
- 主要统计检验；
- H0–H4 的数值阈值；
- 开发集与冻结评估集。

## 10. 结果解释

| 结果 | 允许的结论 | 不允许的结论 |
|---|---|---|
| 恢复完全一致 | state 工具链可靠 | 模型具有 Self |
| 可从 state 解码 A/B | state 包含相关信息 | 模型在决策中使用了该信息 |
| swap 后单约束行为迁移 | state 对该行为有因果影响 | 存在长期身份 |
| 联合任务也随 state 迁移 | 两项约束在决策中共同生效 | 已证明一般性的自我模型 |
| 长期轨迹、恢复和基线均通过 | 存在 Persistent Self 候选机制 | 机器具有意识 |

## 11. 下一项实现工作

在确认性实验前按以下顺序完成：

1. 共同冻结 `docs/definitions.md`；
2. 明确首篇研究主张、主要假设和否定条件；
3. 画出实验因果图并列出全部混淆变量；
4. 共同冻结 `docs/architecture.md` 和 `docs/task_design.md`；
5. 共同审阅并冻结 `docs/evaluation_protocol.md`；
6. 共同审阅并冻结 `docs/state_format.md` 和 `docs/implementation_spec.md`；
7. 实现不依赖目标模型的纯逻辑骨架；
8. 确定远程机器、checkpoint、环境和执行入口；
9. 只在开发集运行接口调查、I0 roundtrip 和 T0 能力门；
10. 填写 tokenizer、标准 delay、数值容差和资源参数；
11. 固定任务生成规则、样本量、统计检验和数值阈值；
12. 完成协议审查，将状态从 Draft 改为 Preregistered；
13. 最后才运行 state-only 确认性行为实验。

本机只用于代码、配置、结果分析和文档维护，不作为模型实验或性能结论的运行环境。
