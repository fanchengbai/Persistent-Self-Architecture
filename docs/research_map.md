# PSA 研究地图

> 状态：v0.3，首轮宽检索及两批核心文献精读完成  
> 检索日期：2026-07-29  
> 目的：识别 Persistent Self Architecture（PSA）的直接先导工作、可复用方法、必要基线和真正研究空白。

## 1. 本轮研究问题

首轮检索围绕六个问题展开：

1. RWKV、Mamba 等 recurrent / state-space 模型的内部状态可以保存什么？
2. 如何证明状态中“存在信息”，以及该信息是否真正影响模型行为？
3. 现有 Agent 如何实现长期记忆、Persona 和身份一致性？
4. 是否已有工作实现稳定身份与适应性状态的分层？
5. LLM 的置信度、自我评估和能力边界如何测量？
6. PSA 与上述工作的实质差异在哪里？

## 2. 初步结论

当前文献已经分别覆盖了以下能力：

- 使用外部记忆维持长期交互；
- 使用 Persona Prompt、检索或后处理维持角色一致性；
- 从内部激活中读取或操纵人格、价值和其他高层特征；
- 使用 recurrent state 压缩历史信息；
- 评估模型是否知道自己“知道或不知道”；
- 通过 activation patching、steering 或消融建立局部因果证据。

但首轮检索尚未发现一个成熟工作同时实现并严格比较：

```text
World State / Self State 显式分离
        +
多时间尺度、受约束的 Self Evolution
        +
跨会话持续状态
        +
state swap / rollback / ablation 因果实验
        +
Prompt / Memory / Internal State 等信息量基线
```

因此，PSA 暂定的研究空白不是“让 Agent 有 Persona”或“让 Agent 记得更久”，而是：

> 把可持续、可演化的自我状态变成一个可独立干预的内部变量，并验证它是否对长期行为产生超出 Prompt 与 Memory 的因果作用。

这是首轮检索形成的工作假设，不是最终的新颖性结论；需在精读和引文追踪后再次核验。

## 3. 文献分类

### 3.1 A 类：直接先导工作

这些论文与 PSA 的核心问题直接重叠，应优先精读。

| 优先级 | 论文 | 主要贡献 | 与 PSA 的关系 | 需要重点核验 |
|---|---|---|---|---|
| A1 | [RWKV-7 “Goose” with Expressive Dynamic State Evolution](https://arxiv.org/abs/2503.14456)（2025） | 使用带向量门控和上下文学习率的广义 delta rule；支持固定内存和 state tracking | PSA 初始基础模型与 state 操作对象 | 0.4B 权重的实际 state 结构、保存/恢复接口、跨段使用方式 |
| A1 | [Beyond Static Persona Consistency: Dynamic Persona Coherence in LLM Role-Playing](https://aclanthology.org/2026.acl-long.1336/)（ACL 2026） | 将长期身份、中期状态和短期情绪分层，并抑制 Persona 漂移 | 与 PSA 的多时间尺度 Self State 最接近 | 其状态是否只是外部文本脚手架；是否做了内部状态因果干预 |
| A1 | [Time, Identity and Consciousness in Language Model Agents](https://arxiv.org/abs/2603.09043)（2026，预印本） | 区分“谈论稳定自我”和“在时间上按稳定自我组织”，提出持续性指标 | 可为 PSA 的 Identity Continuity 提供操作指标 | 五个身份指标、Arpeggio/Chord persistence score 的计算与适用范围 |
| A1 | [Identity as Attractor: Geometric Evidence for Persistent Agent Architecture in LLM Activation Space](https://arxiv.org/abs/2604.12016)（2026，预印本） | 探索身份文档是否在激活空间形成 attractor-like geometry | 直接关联隐式 Self 表征与激活空间 | 样本量、控制条件、相关证据与因果证据的界限 |
| A1 | [Your Language Model Secretly Contains Personality Subnetworks](https://arxiv.org/abs/2602.07164)（ICLR 2026） | 通过激活统计和对比剪枝定位 Persona 相关子网络，并做局部恢复实验 | 为隐式 Self probe、persona 消融和因果定位提供方法 | Persona 是否跨时间持续；子网络是否承载身份还是表达风格 |
| A1 | [Psychological Steering of Large Language Models](https://arxiv.org/abs/2604.14463)（2026，预印本） | 用激活注入控制 OCEAN 人格特征，并与 Persona Prompt 比较 | 可作为 Self State 注入方式和 Prompt 基线 | 干预是否长期保持；不同特征之间的耦合和副作用 |
| A1 | [Characterizing Mamba’s Selective Memory using Auto-Encoders](https://arxiv.org/abs/2512.15653)（2025，预印本） | 从 Mamba hidden state 重建输入，分析选择性遗忘内容 | 可直接迁移为 RWKV State 信息保持实验 | 重建探针的容量控制、序列长度与 state 信息丢失类型 |

### 3.2 B 类：状态模型与长期记忆基础

这些论文帮助理解 recurrent state 的能力和边界，但不直接研究 Self。

| 优先级 | 论文 | 可复用内容 | PSA 中的角色 |
|---|---|---|---|
| B1 | [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)（2023） | selective state、选择性复制任务、状态保持/重置机制 | 比较架构；帮助设计“保留什么、遗忘什么”的实验 |
| B1 | [Transformers are SSMs: Structured State Space Duality / Mamba-2](https://arxiv.org/abs/2405.21060)（2024） | SSM 与 attention 的统一视角 | 解释 RWKV/SSM state 与 Transformer 上下文机制的差异 |
| B1 | [Longhorn: State Space Models are Amortized Online Learners](https://openreview.net/forum?id=8jOqCcLzeO)（ICLR 2025） | 把 SSM 解释为在线学习器 | 连接“状态演化”和 inference-time learning |
| B2 | [State-space models can learn in-context by gradient descent](https://openreview.net/forum?id=52XG8eexal)（2024/2025） | 分析 SSM 中类似梯度下降的 in-context learning | 判断 state evolution 是否只是任务内在线学习 |
| B2 | [StableSSM: Alleviating the Curse of Memory](https://openreview.net/forum?id=BwG8hwohU4)（2023） | 长期记忆的稳定性与参数化限制 | 为 state 衰减和长期保持实验提供理论背景 |

### 3.3 C 类：Agent Memory、Persona 与身份基线

这些系统能够表现出“像是持续个体”的行为，但大多依赖外部记忆或提示。它们是 PSA 必须击败或区分的基线。

| 优先级 | 论文 | 主要机制 | 对 PSA 的启示 |
|---|---|---|---|
| C1 | [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)（UIST 2023） | 经验记录、按相关性/时近性/重要性检索、反思和规划 | Self 与 Memory 必须区分；反思摘要可作为 Memory 基线 |
| C1 | [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)（2023） | 分层记忆和虚拟上下文管理 | 外部持久记忆基线；验证 PSA 收益是否只是更好检索 |
| C1 | [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813)（ICLR 2025） | 信息提取、跨会话推理、时间推理、知识更新、拒答五类评估 | 可复用任务结构，但需把“用户记忆”扩展为“智能体自身状态” |
| C1 | [Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427)（2023） | 模块化记忆、内部/外部动作和决策循环 | 用于定位 PSA 在认知架构中的位置 |
| C1 | [Hello Again! LLM-powered Personalized Agent for Long-term Dialogue](https://aclanthology.org/2025.naacl-long.272/)（NAACL 2025） | 事件记忆、动态用户 Persona 和 Agent Persona | 与动态 Persona 模块直接比较 |
| C2 | [PersonaLLM](https://aclanthology.org/2024.findings-naacl.229/)（NAACL Findings 2024） | Big Five Persona 表达及一致性评估 | 可复用人格任务，但不能把自报问卷直接当成 Self 证据 |
| C2 | [LLM Agents in Interaction: Measuring Personality Consistency](https://aclanthology.org/2024.personalize-1.9/)（2024） | 测量 Persona Prompt 在交互中的一致性与语言趋同 | 作为 Persona Prompt 的行为基线 |
| C2 | [Dissociative Identity: Language Model Agents Lack Grounding for Reputation Mechanisms](https://arxiv.org/abs/2605.30169)（2026，预印本） | 指出现有 Agent 身份由多个可替换模块组成，缺乏行为连续性和制裁敏感性 | 为“状态可交换性、不可替代性和身份落地”提供反例框架 |

### 3.4 D 类：因果探针与内部状态干预

这组方法决定 PSA 能否从“相关性”走到“因果性”。

| 优先级 | 论文 | 可复用方法 | 使用注意 |
|---|---|---|---|
| D1 | [Towards Best Practices of Activation Patching in Language Models](https://arxiv.org/abs/2309.16042)（2023） | activation patching、corruption、恢复和指标选择 | 不同 corruption 与指标会导致不同定位结论 |
| D1 | [Latent Causal Probing](https://arxiv.org/abs/2407.13765)（COLM 2024） | 用结构因果模型约束 probe 结论 | 高 probe 准确率不能单独证明模型在使用该变量 |
| D1 | [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)（2023） | 高层表示读取和控制 | 可用于 Self 特征读取与注入，但需增加持久性实验 |
| D1 | [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248)（2023） | 从对比提示构造 activation addition 方向 | 适合作为轻量 state injection 原型 |
| D2 | [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)（2022） | causal tracing、局部知识编辑与泛化测试 | 可借鉴“知道”与“复述”的区分，以及局部编辑的特异性评估 |

### 3.5 E 类：元认知与不确定性

这些工作为 `confidence` 和 `capability_estimate` 字段提供指标，但通常不涉及持续 Self State。

| 优先级 | 论文 | 可复用内容 | 局限 |
|---|---|---|---|
| E1 | [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221)（2022） | P(True)、P(IK)、校准与跨任务泛化 | 自我评估可能只是估计题目难度，而非真正模型特定知识 |
| E1 | [Do Large Language Models Know What They Don’t Know?](https://arxiv.org/abs/2305.18153)（2023） | SelfAware 数据集、可回答/不可回答问题 | 偏输出层行为，不证明存在持续内部自我估计 |
| E1 | [Large Language Models Must Be Taught to Know What They Don’t Know](https://arxiv.org/abs/2406.08391)（2024） | 用少量标注和内部特征训练不确定性估计 | 支持“内部特征优于纯 Prompt”的实验方向 |
| E2 | [Beyond “I Don’t Know”: Discriminating Data and Model Uncertainty](https://aclanthology.org/2026.acl-long.547/)（ACL 2026） | 区分输入歧义和模型能力不足 | 可用于 `confidence` 与 `capability_estimate` 的分离 |
| E2 | [Do LLMs Know What They Know? Measuring Metacognitive Efficiency with Signal Detection Theory](https://arxiv.org/abs/2603.25112)（2026，预印本） | meta-d′ 和 M-ratio，将任务能力与元认知效率分离 | 需核验在小型 RWKV 上的样本和分布要求 |

## 4. 对 PSA 设计的直接影响

### 4.1 Self State 必须与外部记忆分离

Generative Agents、MemGPT、LongMemEval 等说明外部记忆本身就能提高长期一致性。因此，PSA 必须设置：

- Memory-only 基线；
- Persona Prompt 基线；
- Self State 基线；
- Memory + Self State 组合；
- 等信息量和等上下文长度控制。

否则无法判断收益来自 Self 机制还是信息重新注入。

### 4.2 Probe 只能作为相关证据

从 state 中解码出身份、目标或偏好，只能说明信息可被读取。核心结论必须依靠：

- state swap；
- state ablation；
- state interpolation；
- state rollback；
- 层或通道级 patching；
- 干预强度—行为效应曲线。

### 4.3 Self State 应采用多时间尺度

Dynamic Persona Coherence 已明确区分长期身份与短期适应。PSA 若只做一个扁平向量，新颖性和工程合理性都会不足。首版建议：

```text
Protected / very slow:
  identity_anchors

Slow:
  preferences, capability_estimate

Fast:
  active_goals, confidence, uncertainty_conflicts
```

`values`、`emotion`、`curiosity` 暂不进入 v0.1 核心实验；字段边界以 [`definitions.md`](definitions.md) 为准。

### 4.4 元认知不能只测“会不会说不知道”

应同时测：

- 正确率或任务能力；
- 置信度校准；
- 区分数据不确定性与模型不确定性的能力；
- 是否据此采取正确动作：回答、求助、检索、继续推理或拒答；
- Self State 中置信度发生变化后，行为是否按预期迁移。

### 4.5 “身份表现”不等于“身份机制”

Persona 一致、人格问卷分数或自我描述都可能由 Prompt 模仿产生。PSA 必须区分：

1. 说出某种身份；
2. 在不同语境表达相同身份；
3. 在决策中受该身份约束；
4. 在时间上保持约束；
5. 被经验适当更新；
6. 状态干预后约束随之迁移。

## 5. 暂定研究空白

按重要性排序：

### Gap 1：Recurrent state 与 Agent identity 研究尚未充分连接

SSM/RWKV 论文主要关注序列建模、state tracking 和信息压缩；Agent identity 论文主要依赖外部 Prompt、Memory 或 scaffold。两条路线之间存在明显断层。

### Gap 2：缺少面向 Self 的纵向因果实验

Activation steering 能改变当下行为，但通常不研究跨会话持续、经验演化、身份分叉和恢复。PSA 的 fork/swap/rollback 实验可补足这一点。

### Gap 3：稳定性—可塑性缺少字段级约束

现有 Persona 系统开始区分长期与短期状态，但很少把不同更新速率、证据阈值、变更历史和回滚作为统一状态机制。

### Gap 4：Self、Memory 与 Persona 缺少等信息量比较

许多系统只证明自己的脚手架优于无记忆模型，尚不足以证明存在独立 Self 机制。

### Gap 5：元认知状态与行动策略缺少持续耦合

校准研究通常评估一次性置信度。PSA 可以研究能力估计如何跨经验更新，并因果影响求助、探索和拒答策略。

## 6. 精读顺序

### 第一批：决定 PSA 定位

1. RWKV-7 “Goose”
2. Beyond Static Persona Consistency
3. Time, Identity and Consciousness in Language Model Agents
4. Identity as Attractor
5. Your Language Model Secretly Contains Personality Subnetworks
6. Generative Agents
7. MemGPT

### 第二批：决定实验方法

1. Towards Best Practices of Activation Patching
2. Latent Causal Probing
3. Representation Engineering
4. Characterizing Mamba’s Selective Memory
5. LongMemEval

### 第三批：决定元认知指标

1. Language Models (Mostly) Know What They Know
2. Large Language Models Must Be Taught to Know What They Don’t Know
3. Beyond “I Don’t Know”
4. Metacognitive Efficiency with Signal Detection Theory

## 7. 单篇精读记录模板

每篇精读笔记至少包含：

```text
Citation:
Publication status:
Research question:
Operational definitions:
Architecture / method:
Model and data:
Intervention:
Controls:
Metrics:
Main result:
Negative result:
Threats to validity:
Released code/data:
What PSA can reuse:
What PSA must do differently:
Questions raised:
```

## 8. 分工

### Codex 负责

- 继续检索论文并进行前向/后向引文追踪；
- 阅读论文正文，提取定义、实验、指标和局限；
- 维护本文献地图和精读笔记；
- 把文献方法转化为 PSA 的实验协议；
- 检查拟议创新点是否已被已有工作覆盖；
- 整理开源代码、数据集和复现条件；
- 在不需要方向性决策时持续推进。

### 项目负责人负责

- 决定最终研究主张和不能越过的概念边界；
- 确定可使用的硬件、时间和资金预算；
- 决定首要产出是研究论文、可运行原型，还是两者并重；
- 审核 Self State 是否引入价值、情感、好奇心等高争议字段；
- 对阶段结果做 Go / Revise / Stop 决策；
- 决定公开范围、署名、发布和伦理立场。

### 共同负责

- 冻结操作性定义；
- 选择核心假设和主要指标；
- 确定什么证据足以支持或否定结论；
- 审核实验是否存在拟人化、循环定义或评价泄漏；
- 根据负面结果调整理论，而不是只调整实验以追求正面结果。

## 9. 当前不阻塞研究的默认假设

在项目负责人进一步确认前，暂按以下前提推进：

- 研究论文与最小可运行原型并重；
- 优先使用开源模型、论文和数据；
- 计算预算有限，先做冻结模型、推理期干预和小型 probe；
- 不以“意识”作为实验标签或结论；
- Self State 首版只包含身份、目标、偏好、置信度和能力估计；
- 所有核心结论都要求 Prompt、Memory 和 state intervention 对照。

## 10. 下一轮工作

第一批 7 篇论文的精读结果见：

- [第一批核心文献精读笔记](paper_notes/batch_1_core_literature.md)
- [第二批实验方法精读笔记](paper_notes/batch_2_methods.md)
- [PSA 操作性定义与判定标准](definitions.md)
- [PSA 首轮研究主张与可证伪条件](research_claims.md)
- [PSA 理论框架与模型架构](architecture.md)
- [EXP-001 Identity–Goal Binding 任务设计](task_design.md)
- [EXP-001 评价与统计协议](evaluation_protocol.md)
- [PSA 状态与 Checkpoint 格式规范](state_format.md)
- [PSA 实现与远程执行规范](implementation_spec.md)
- [EXP-001：RWKV State Persistence & Swap](../experiments/EXP-001_state_persistence_swap/PROTOCOL.md)

下一轮任务：

1. 追踪第一批论文的参考文献和后续引用，继续核验暂定研究空白。
2. 共同审阅并冻结 `docs/definitions.md`、`docs/research_claims.md` 和 `docs/architecture.md`。
3. 共同审阅并冻结 `docs/task_design.md` 的变量、任务生成和混淆控制。
4. 共同审阅 `docs/evaluation_protocol.md` 的指标、样本量、统计方法、SESOI 和决策门。
5. 共同审阅 `docs/state_format.md` 与 `docs/implementation_spec.md`。
6. 远程 checkpoint/tokenizer 确定后填写剩余工程参数并完成预注册；本机不运行实验。
