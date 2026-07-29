# 第一批核心文献精读笔记

> 版本：v0.1  
> 精读日期：2026-07-29  
> 范围：RWKV state、动态 Persona、身份持续性、内部人格表征与长期记忆基线

## 1. RWKV-7 “Goose” with Expressive Dynamic State Evolution

**论文**

- Peng et al. (2025)
- [arXiv:2503.14456](https://arxiv.org/abs/2503.14456)
- 状态：预印本；模型、数据清单和代码已公开

**研究问题**

如何设计一种保持常数推理内存和单 token 常数计算量，同时拥有更强状态追踪能力的 recurrent language model？

**方法与架构**

RWKV-7 在 RWKV-6 基础上引入广义 delta rule：

- 数据依赖的向量衰减；
- 向量化的 in-context learning rate；
- 不同的 removal key 与 replacement key；
- 非对角、输入依赖的状态转移；
- 每个 head 的 WKV state 为 `64 × 64` 矩阵。

状态更新可以被理解为一种在推理期间发生的快速在线学习：新输入按通道选择性地覆盖、保留或修改已有状态。

**关键实验**

- 语言模型基准与长上下文评估；
- 关联回忆；
- 群乘法 state-tracking 任务；
- WKV state 的 RMS、stable rank 和矩阵可视化；
- 架构消融和初始 token 敏感性分析。

在群乘法任务中，RWKV-7 的状态追踪强于 Transformer、Mamba 和 S4，但略弱于经典 RNN。论文还证明 RWKV-7 可以用常数层数识别所有正则语言。

**限制**

- state-tracking 证据来自合成任务，并未研究身份、目标或偏好等语义变量；
- WKV state 检查主要测数值稳定性和有效秩，没有做语义 probe；
- state update 对数值精度敏感；
- 论文发布的模型是 base model，没有经过 instruction tuning 或偏好对齐；
- 缺少 `<|endoftext|>` 等边界 token 时可能出现首 token 记忆问题；
- 论文未验证跨独立会话保存和恢复 state 后的行为稳定性。

**PSA 可复用内容**

- state 保存、恢复、交换和矩阵级干预的技术对象；
- RMS 与 stable rank 作为状态健康度指标；
- 群乘法 state-tracking 作为基础设施验收任务；
- delta rule 的“通道级选择性更新”可为 Self State 耦合提供机制灵感；
- 数值精度和边界 token 必须进入复现实验配置。

**PSA 必须做得不同**

- 从数值 state-tracking 走向身份、目标和偏好等行为变量；
- 区分“state 中可解码”与“state 被模型用于决策”；
- 增加跨进程恢复、state swap、state ablation 和行为迁移；
- 使用相同后续输入，隔离历史 state 的因果作用；
- 避免将 state 对上下文内容的保存直接称为 Self。

---

## 2. Beyond Static Persona Consistency: Dynamic Persona Coherence in LLM Role-Playing

**论文**

- Qi et al. (ACL 2026)
- [ACL Anthology](https://aclanthology.org/2026.acl-long.1336/)
- 状态：ACL 2026 长文

**研究问题**

如何让角色在长期互动中既保持稳定身份，又能根据累积经历产生合理的情绪变化？

**操作性定义**

Dynamic Persona Coherence 被拆成：

- Identity-Layer Stability：稳定的身份属性；
- Adaptive-Layer Appropriateness：随历史合理演化的心理状态；
- Constrained Expression：动态状态的表达不能违反身份约束。

综合分数使用：

```text
coherence = min(identity_score, adaptive_score)
```

这避免用优秀的短期状态表达抵消严重的身份违背。

**L/M/S 状态模型**

- L 层：固定身份锚点，包括先天特征、学习到的特征和当前处境；
- M 层：中期的意义感和压力适应能力，取值 0–10；
- S 层：短期情感效价，取值 0–10。

不同状态以不同速率更新：

- S 每轮通常变化 1.0–2.0；
- M 每轮通常变化 0.3–0.5；
- L 在全部交互中保持不变。

关键更新方式是：

```text
state_t = clip(state_(t-1) + event_delta, 0, 10)
```

其中 event delta 由 GPT-4o 根据事件和 L 层 Persona 评估。状态随后通过提示和生成模块影响输出。

**闭环组件**

- PCC：按 L、M、S 分别评价输出；
- PCR：只存储高分的情境—状态—回答案例；
- PDS：低分时检索案例并重新生成回答。

**实验**

- 5 个 Persona；
- GPT-4o、Claude-3.5-Sonnet、DeepSeek-V3.2；
- 每个 Persona 约 100–150 轮；
- 5 个随机种子；
- 静态 Persona Prompt、S/M 状态但无 PDS、完整系统三组比较。

完整系统相对静态基线平均提高约 26.8% PCC；其中累积 S/M 跟踪解释了约 84.9% 的收益，PDS 修正贡献其余约 15.1%。

**关键限制**

- L/M/S 是外部显式状态，不是基础模型的 recurrent hidden state；
- 状态变化由另一个 LLM 计算，本质上仍是 scaffold；
- L 层严格冻结，没有研究长期身份更新；
- 没有 state swap、rollback 或内部状态消融；
- 没有等信息量 Memory-only 对照；
- 主要评价依赖 LLM-as-judge，未报告系统性人类评估；
- 任务集中于角色扮演和情绪压力，不等同于一般目标、价值或能力估计；
- “无外部监督”指运行期自动闭环，不代表机制已经内化进基础模型。

**PSA 可复用内容**

- 稳定身份与动态状态分层；
- 不同更新速率；
- `min(stability, adaptation)` 形式的硬约束评分；
- 长轨迹、情绪反转和压力累积任务；
- 静态 Prompt、动态 state、动态 state + corrector 的消融结构。

**PSA 必须做得不同**

- 把 DPC 作为强外部 Self scaffold 基线，而不是 PSA 的最终架构；
- 加入 recurrent state 与外部 Self State 的独立干预；
- 测试交换和删除状态后行为是否迁移；
- 研究能够更新的身份字段，而非永久冻结整个 L 层；
- 加入非情绪任务：长期目标、能力校准、冲突决策；
- 使用程序化任务和客观行为指标降低 judge 偏差。

---

## 3. Time, Identity and Consciousness in Language Model Agents

**论文**

- Perrier & Bennett (2026)
- [arXiv:2603.09043](https://arxiv.org/abs/2603.09043)
- 状态：理论预印本

**研究问题**

一个 Agent 能够分别回忆身份的每个组成部分，是否意味着这些约束在做决定时真正共同生效？

**核心区分**

论文区分：

- **Weak persistence**：身份的各个组成部分在某个时间窗口中分别出现；
- **Strong persistence**：这些组成部分在一个实际决策步骤中共同实例化。

这解释了为什么一个系统可能：

- 能稳定回答“我是谁”；
- 能分别复述角色、目标和约束；
- 却在行动时没有同时受到这些约束。

因此，“说得像一个稳定自我”不等于“按稳定自我组织行动”。

**五项身份指标**

1. Identifiability：是否存在能与其他 Agent 或会话区分的稳定特征；
2. Continuity：身份相关状态是否平滑演化，而不是突然翻转；
3. Consistency：重复身份问题是否得到稳定回答；
4. Persistence：身份组成是否随时间保持，尤其是否在决策点共同生效；
5. Recovery：状态中断后能否恢复目标身份。

**最小评估流程**

1. 将身份拆成可检查的组成部分；
2. 记录每个决策步骤中哪些组成部分处于活动状态；
3. 选择时间窗口；
4. 计算 weak / strong persistence；
5. 同时进行重复自我报告和实际行为测试；
6. 检查“自我报告稳定”与“行动约束生效”是否分离。

**限制**

- 是抽象理论模型，没有真实 Agent 的大规模实证；
- 依赖能够把身份组成映射到可观测寄存器、文本或内部特征；
- 对 RAG、缓存和工具控制器的结论依赖具体实现；
- 时间窗口的选择会影响结果。

**PSA 可复用内容**

- 将 `identity continuity` 拆成五个具体指标；
- 把 strong persistence 设为高于自我报告一致性的证据；
- 首个实验应设计“必须同时应用两个身份约束”的行动任务；
- state snapshot 应记录实际决策点，而不只记录会话结束状态；
- recovery 可直接对应 state restore / rollback 实验。

**PSA 必须做得不同**

- 给 weak / strong persistence 提供神经 state 级操作性实现；
- 通过 swap、ablation 和 intervention 验证共同实例化是否因果影响行为；
- 不将这些指标用于意识判定，只用于身份机制评估。

---

## 4. Identity as Attractor: Geometric Evidence for Persistent Agent Architecture

**论文**

- Vasilenko (2026)
- [arXiv:2604.12016](https://arxiv.org/abs/2604.12016)
- 状态：独立研究者预印本，已预注册并公开代码

**研究问题**

复杂 Agent identity 文档的语义改写，是否会在 LLM 激活空间中收敛到一个相对稳定的区域？

**方法**

- Llama 3.1 8B Instruct；
- 额外使用 Gemma 2 9B 进行跨架构复现；
- 原始 `cognitive_core` 1 份；
- 等义改写 7 份；
- 不同身份但结构相似的 Agent 文档 7 份；
- 在早、中、晚层提取 mean-pooled hidden states；
- 使用 cosine distance、Welch t-test、置换检验、Mann-Whitney U 和 bootstrap。

**主要结果**

- 原始 identity 文档与其改写比不同身份文档形成更紧密的激活簇；
- 部分层中，组内距离随深度下降；
- 简化版 identity 文档比随机等长片段更接近完整 identity 的中心；
- 结构严密的完整文档仍明显优于极简摘要；
- 阅读“关于该身份的论文”会靠近该激活区域，但远不如直接处理完整 identity 文档。

**作者自己明确的边界**

该工作证明的是“attractor-like geometry”，不是严格的动力系统 attractor；主要实验测量激活几何，而非行为。探索性 steering 只提供有限行为证据。

**限制**

- 每个条件样本量很小；
- 复杂 identity 与结构长度仍存在少量混淆；
- 主要是静态前向传播，没有跨时间持续性；
- 没有经验驱动的身份演化；
- 没有交换或删除 identity representation 后的系统行为测试；
- 语义改写聚类是 LLM 的一般性质，并非 identity 独有；
- identity 文档本质上仍然是输入 Prompt。

**PSA 可复用内容**

- 等义 identity 文档与结构匹配控制；
- hidden-state 距离和跨层轨迹；
- 把“知道某身份”与“以该身份运行”作为独立实验条件；
- 在 state probe 中加入长度、结构、语言形式控制；
- 预注册与非参数检验。

**PSA 必须做得不同**

- 研究保存于 recurrent state 的身份，而不是只研究身份文档引起的激活；
- 对相同后续输入进行 state swap 和 ablation；
- 将激活几何与实际选择任务连接；
- 测量经历后轨迹是否持续分化。

---

## 5. Your Language Model Secretly Contains Personality Subnetworks

**论文**

- Ye et al. (ICLR 2026)
- [arXiv:2602.07164](https://arxiv.org/abs/2602.07164)
- 状态：ICLR 2026 会议论文

**研究问题**

预训练 LLM 的参数空间中是否已经存在可被隔离的人格相关子网络？

**方法**

1. 使用少量 Persona 校准数据收集各层激活统计；
2. 用权重幅值与激活频率计算参数重要性；
3. 逐层构造稀疏二值 mask；
4. 对相反 Persona 使用 contrastive pruning 增强分离；
5. 推理时应用 mask，不训练或永久修改原始权重。

**关键因果实验**

作者先用完整 mask 将模型从原有 ENFJ 倾向转向 INFP，再逐个恢复被 mask 的层。如果恢复特定 MLP 模块后某些人格维度回到原方向，则认为该模块对目标人格具有因果必要性。

结果显示：

- 影响分布在多个 MLP block；
- 某些早期和中期 MLP 对特定维度影响明显；
- 单个模块通常是必要但不充分的；
- Persona 子网络并非完全局部化。

**实验边界**

- 研究的是模型参数子网络，不是随时间变化的 recurrent state；
- mask 是静态选择，不能自然表达经验演化；
- Persona 主要用 MBTI、角色和二元行为倾向测量；
- 所谓“因果”主要来自 mask 和单层恢复，不等于长期身份因果机制；
- 更强 Persona 表达可能只是输出倾向更极端；
- 没有长期 fork、rollback 或目标保持实验。

**PSA 可复用内容**

- 对比 Persona 校准集；
- 参数或 state 通道的重要性评分；
- “完整干预 → 单层恢复”的因果定位流程；
- 稀疏度扫描和通用能力副作用评估；
- Prompt、RAG 与内部机制比较。

**PSA 必须做得不同**

- 操作动态 recurrent state，而不是只操作固定参数；
- 测试经历能否形成不同 state 轨迹；
- 检查状态干预是否保持通用任务能力；
- 把人格表达与目标、能力判断和实际行动区分。

---

## 6. Generative Agents: Interactive Simulacra of Human Behavior

**论文**

- Park et al. (UIST 2023)
- [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)
- 状态：UIST 2023 会议论文

**研究问题**

如何利用自然语言记忆、反思和规划，使 LLM Agent 在模拟环境中产生可信的长期行为？

**架构**

- Memory stream：以自然语言保存完整经历；
- Retrieval：结合相关性、时近性和重要性检索记忆；
- Reflection：把低层经历综合成高层推断；
- Planning：生成日程并根据环境重新规划；
- Observation：将环境事件写入记忆。

**证据**

- 25 个 Agent 的小镇模拟；
- 观察到信息传播、关系形成和集体活动；
- 通过观察、规划和反思组件的消融，显示各部分都影响行为可信度。

**限制**

- 主要目标是人类感知的 believability，不是 Self 的因果机制；
- 反思和身份推断都是自然语言记忆；
- World、Other 和 Self 信息混在同一 memory stream 中；
- 没有 state swap 或等信息量内部机制对照；
- 行为可能来自检索到的显式文本，而非持续内部状态。

**PSA 中的角色**

这是必须保留的 Memory + Reflection 基线。PSA 只有在相同信息条件下表现出更稳定或更可控的行为，才能说明 Self State 提供了额外价值。

---

## 7. MemGPT: Towards LLMs as Operating Systems

**论文**

- Packer et al. (2023)
- [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
- 状态：预印本，代码和数据已发布

**研究问题**

如何借鉴虚拟内存，让有限上下文窗口的 LLM 管理远超窗口大小的长期信息？

**架构**

- Main context：当前提示 token，可被模型直接访问；
- External context：窗口外的持久存储；
- Working context：可读写的常驻文本块；
- FIFO queue：近期消息；
- Recall / archival storage：长期检索存储；
- Function executor：让模型通过函数调用移动、检索和编辑信息；
- Memory pressure event：上下文接近上限时触发整理。

记忆编辑和检索由 LLM 根据系统提示自主决定，但每轮实际推理仍只读取拼接后的当前上下文。

**主要证据**

- 长文档问答；
- 多会话对话；
- 嵌套键值检索；
- 展示了外部存储和多次检索对有限上下文的扩展作用。

**限制**

- 所有持久信息最终都必须重新进入 prompt 才能影响基础模型；
- `working context` 仍是外部文本，不是内部 Self State；
- 自主记忆管理由系统提示和函数 schema 驱动；
- 主要研究用户信息和文档记忆；
- 没有显式区分 World Memory 与 Self；
- 没有内部状态级因果干预。

**PSA 中的角色**

MemGPT 是“强外部记忆”基线。比较时需要控制：

- 可访问的信息量；
- 上下文 token 数；
- 检索次数；
- 当前提示中实际出现的信息；
- 额外模型调用成本。

---

## 8. 横向综合

| 工作 | 持续状态 | 状态位置 | 可演化 | 内部因果干预 | 长期身份 | PSA 中的角色 |
|---|---|---|---|---|---|---|
| RWKV-7 | 是 | recurrent WKV state | 随 token 更新 | 未针对 Self | 未研究 | 基础模型 |
| Dynamic Persona Coherence | 是 | 外部数值与 Prompt | 是，规则累计 | 否 | L 层固定 | 最接近的外部 Self 基线 |
| Time & Identity | 理论支持 | 可跨多种 scaffold | 理论描述 | 无实证 | 核心问题 | 指标框架 |
| Identity as Attractor | 单次前向 | Transformer hidden states | 否 | 探索性 steering | 静态 identity 文档 | 表征分析基线 |
| Personality Subnetworks | 参数中长期存在 | 稀疏参数 mask | 否 | 有，mask/恢复 | 静态 Persona | 内部 Persona 基线 |
| Generative Agents | 是 | 外部自然语言记忆 | 反思式更新 | 否 | 间接形成 | Memory + Reflection 基线 |
| MemGPT | 是 | 外部文本和存储 | 自主编辑 | 否 | 间接保持 | 强外部 Memory 基线 |

## 9. 对 PSA 研究主张的修订

首轮精读后，PSA 不宜把下列内容作为主要创新：

- 把身份与短期状态分层；
- 使用外部数值状态维持动态 Persona；
- 用长期记忆保持角色一致性；
- 从激活中读取或操纵人格；
- 仅展示模型能稳定复述身份。

更有辨识度、也更可验证的主张应当是：

> 在 recurrent language model 中，将跨时间持续的 Self State 作为可独立保存和干预的内部变量；通过强对照证明它在实际决策时共同承载身份与目标约束，并产生超出外部 Prompt 和 Memory 的行为效应。

这个主张包含四个不可缺少的部分：

1. **内部状态**：不是每轮重新注入的自然语言描述；
2. **持续与演化**：状态来自历史轨迹并跨边界保存；
3. **决策时共同生效**：不只在自我报告问题中出现；
4. **因果可干预**：交换、消融和恢复会使行为按状态迁移。

## 10. 对首个实验的直接要求

首个实验不应直接尝试完整“人格”。建议使用两个简单、正交、可程序化评分的身份约束，例如：

```text
Identity component A:
  偏好选择圆形物体

Identity component B:
  在风险不明确时选择保守方案
```

训练或上下文经历分别建立四类轨迹：

```text
A0 B0
A0 B1
A1 B0
A1 B1
```

然后使用完全相同的后续输入测试：

1. 自我报告：分别询问 A、B；
2. 单约束决策：只需应用 A 或 B；
3. 联合决策：必须同时应用 A 与 B；
4. state reset；
5. state restore；
6. state swap；
7. 分层或分通道消融。

主要判断：

- 是否能从 state 解码 A、B；
- A、B 是否在联合任务中共同影响行为；
- 交换 state 后联合行为是否随之迁移；
- 恢复旧 state 后是否恢复旧行为；
- 效应是否超出相同内容的 Prompt 和 Memory 基线。

该实验比直接让模型回答“你是谁”更能区分：

- 上下文复述；
- 身份成分的独立记忆；
- 决策时的共同实例化；
- state 的真实因果作用。

