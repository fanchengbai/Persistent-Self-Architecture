# Persistent Self Architecture（PSA）项目计划

> 本文档是 PSA 的行动指南。它把项目简介中的研究设想转化为可执行、可验证、可复现的工作计划。

## 1. 文档信息

- 项目阶段：Phase 0 — Research Design
- 初始基础模型：RWKV-7 0.4B
- 计划原则：小规模验证、因果优先、对照充分、结果可复现、允许否定结论
- 更新方式：每完成一个阶段或发生关键技术决策后更新本文档

## 2. 项目目标

### 2.1 核心研究问题

一个具备持续、自我更新且能因果影响行为的 Self State，是否会使人工智能系统表现出区别于传统无状态 Agent 的稳定行为特征？

### 2.2 总体目标

设计并验证一种由以下部分组成的最小人工智能架构：

```text
World Model
    +
Persistent Self Model
    +
Coupling / Policy
    +
World & Self Update
```

该架构应能够：

1. 跨任务和跨会话保持可识别的身份与目标倾向。
2. 依据经验更新部分 Self State，同时避免无约束漂移。
3. 让 Self State 对选择、规划和自我评估产生可测量的因果作用。
4. 表达自身的不确定性、能力边界和内部冲突。
5. 支持状态保存、恢复、编辑、交换、消融和追踪。

### 2.3 非目标

本项目暂不：

- 判断系统是否具有主观体验或“真正的意识”。
- 把角色提示词、用户画像或普通长期记忆直接等同于 Self。
- 以扩大参数量或训练数据量代替架构验证。
- 在没有基线、对照和消融实验的情况下宣称出现了“自我”。

## 3. 核心概念的操作性定义

为了使研究可验证，项目采用以下工程定义：

### 3.1 World State

描述外部环境、任务、知识、历史事件和未来预测的状态。

### 3.2 Self State

一种跨时间持续存在、可被经验更新，并能系统性影响未来行为的内部状态。v0.1 只研究：

- 受保护 / 极慢：`identity_anchors`
- 慢变量：`preferences`、`capability_estimate`
- 快变量：`active_goals`、`confidence`、`uncertainty_conflicts`

`values`、`emotion`、`curiosity` 等字段暂不进入首版核心实验。完整边界、排除规则和证据等级以 [`docs/definitions.md`](docs/definitions.md) 为准。

### 3.3 Persistent Self

只有同时满足下列条件，才把一个状态视为 Persistent Self 的候选机制：

1. 持续性：跨任务或跨会话保留。
2. 个体性：不同经历形成可区分的状态轨迹。
3. 因果性：干预状态会导致可重复的行为变化。
4. 可更新性：状态能依据经验发生有约束的变化。
5. 决策时绑定：多个身份或目标约束能在同一次决策中共同生效。
6. 可审计与可恢复：能够追踪状态的来源、版本和影响，并通过快照恢复对应行为。

## 4. 研究假设

| 编号 | 假设 | 主要验证方法 | 支持证据 |
|---|---|---|---|
| H1 | RWKV 的 recurrent state 中存在可解码的身份、目标或偏好信息 | State Probe、时间序列分析 | 探针显著优于随机与内容匹配基线 |
| H2 | 模型隐状态对个体行为具有因果影响 | 状态交换、消融、插值、恢复 | 行为特征随状态迁移或按干预强度连续变化 |
| H3 | 显式 Self State 比提示词或普通记忆产生更稳定的长期行为影响 | 与 Prompt、Memory、无 Self 基线对比 | 跨会话一致性和干预效应显著更高 |
| H4 | 不同经验能够使相同基础模型形成不同个体轨迹 | 分叉经历实验 | 多轮后组间差异扩大且组内保持稳定 |
| H5 | 受约束的 Self Update 能在适应性和身份稳定性之间取得平衡 | 漂移测试、恢复测试、冲突任务 | 能适应新证据，同时关键身份字段不过度漂移 |
| H6 | 显式置信度和能力估计可以改善元认知校准 | 置信度校准与拒答实验 | 校准误差下降，恰当求助或继续思考的比例上升 |

所有假设都允许被否定。失败结果应记录为研究结论，而不是通过改变指标定义规避。

## 5. 总体技术架构

```text
Input / Observation
        |
        +--------------------+
        |                    |
        v                    v
   World Encoder        Self Encoder
        |                    |
        v                    v
   World State W_t      Self State S_t
        |                    |
        +---------+----------+
                  |
           Coupling Layer
      (gating / conditioning)
                  |
             Base RWKV
                  |
          Policy / Response
                  |
        Environment Feedback
                  |
        +---------+----------+
        |                    |
        v                    v
 World State Update   Self State Update
```

### 5.1 最小实现边界

第一版只实现能够回答核心因果问题的最小系统：

- 一个可冻结的 RWKV 基础模型
- 一个结构化 Self State
- 一个 Self Encoder
- 一种门控注入机制
- 一个受规则或小型网络控制的 Self Updater
- 状态快照、恢复、交换和消融接口
- 可重复运行的评估脚本

在证明最小机制有效前，不引入多智能体社会、复杂情感系统或大规模持续训练。

## 6. 阶段路线图

### Phase 0：研究设计与架构冻结

目标：把概念转化为可复现实验问题。

主要任务：

1. 完成相关研究地图：RWKV、Mamba/SSM、长期记忆、Agent identity、persona consistency、metacognition、causal representation。
2. 明确 Self、Memory、Persona、World State 的边界。
3. 定义任务集、指标、对照组和统计方法。
4. 确认 RWKV-7 0.4B 的权重、推理环境、显存和运行成本。
5. 建立实验记录、配置和随机种子规范。

交付物：

- `docs/research_map.md`
- `docs/definitions.md`
- `docs/research_claims.md`
- `docs/architecture.md`
- `docs/task_design.md`
- `docs/evaluation_protocol.md`
- `docs/state_format.md`
- `docs/implementation_spec.md`

完成标准：

- 每个核心假设都有对应干预、对照和量化指标。
- World、Memory、原生 recurrent state 和显式 Self State 的边界清楚。
- 架构、任务、指标、统计方法和否定条件能够在看不到正式结果时冻结。
- EXP-001 状态由 Draft 更新为 Preregistered。

### Phase 1：RWKV State 基础设施

目标：能够可靠观察和操纵模型隐状态。

主要任务：

1. 实现 state 捕获、序列化、加载和版本检查。**首版已完成，云端 L3/100 次恢复门已通过。**
2. 实现 state 恢复、交换、插值、噪声扰动和分层消融。**恢复、reset、diff 与完整交换开发门已通过；matched random 已实现待验证，插值和分层消融待后续扩展。**
3. 记录 token、层、时间步和 state 统计量。
4. 建立状态操作前后的输出比较工具。

交付物：

- `src/psa/state/`：状态读写与干预组件
- `experiments/state_smoke_test/`
- 状态格式说明和兼容性测试

完成标准：

- 保存并恢复 state 后，确定性设置下输出可复现。
- 每一种干预都有单元测试和最小示例。
- 实验日志能够追溯模型版本、配置、输入和状态来源。

### Phase 2：Recurrent State 因果载体资格验证

目标：不预设“Self”已经存在，检验原生 recurrent state 是否具备跨时间保存信息、在决策时产生特异因果作用并联合绑定多个约束的资格。

主要实验：

1. **State Probe**：解码预注册的合成身份锚点和目标约束，只作为可读性证据。
2. **State Swap**：交换两个不同经历轨迹的 state。
3. **State Ablation**：删除特定层或通道的状态。
4. **State Interpolation**：在两种状态之间连续插值。
5. **State Persistence**：测试信号随上下文长度和任务切换的衰减。
6. **Content Control**：排除 probe 只读取最近文本内容的可能。

对照组：

- 无 state 或重置 state
- 随机 state
- 打乱标签
- 只提供相同提示词
- 只提供普通记忆摘要
- 内容匹配但身份不同的输入

完成标准：

- 对 H1、H2 给出支持或否定结论。
- 结果在多个随机种子和任务子集上可重复。
- 如果只检测到短期上下文残留，应明确记录为“原生 state 未通过跨时间因果载体资格”，不使用隐式 Self 表述。

### Phase 3：显式 Self Model 最小原型

目标：验证显式 Self State 是否比 Prompt 或 Memory 基线产生更稳定、更可控的因果影响。

主要任务：

1. 实现静态、版本化的 Self Store 和 v0.1 schema。
2. 实现 Self Encoder 与可关闭、可缩放的 gated injection。
3. 实现字段级 mask、交换、随机化和 coupling-off 消融。
4. 对比不同耦合方式：Soft Prefix、层间门控、输出策略条件化。
5. 此阶段由实验直接设置 Self State，不加入自动 Self Updater。

必须比较的系统：

- A：基础模型，无持久状态
- B：基础模型 + Persona Prompt
- C：基础模型 + Memory Retrieval
- D：基础模型 + 原生 recurrent state
- E：基础模型 + 显式 Self State + internal coupling
- E-ablation：E 的字段、编码或 coupling 消融版本

完成标准：

- E 相对 A/B/C/D 在至少一项预注册核心指标上稳定提升。
- 修改、交换或消融 Self State 后出现方向一致、可重复的行为变化。
- Self State 的影响不能完全由提示词长度、最近上下文或记忆内容解释。

### Phase 4：Self Evolution 与个体分化

目标：研究 Self State 如何在长期经验中保持、适应和分化。

主要任务与实验：

1. 实现 Evidence Builder 和受约束 Self Updater。
2. 建立快、慢、受保护字段的证据门槛和更新权限。
3. **Fork Experiment**：从相同模型和初始 Self State 出发，给予不同经历。
4. **Identity Continuity**：跨任务、跨会话和干扰情境测试身份稳定性。
5. **Preference Formation**：观察新偏好是否由重复经验形成并迁移到新任务。
6. **Goal Persistence**：插入无关任务后检查长期目标能否恢复。
7. **Self Correction**：错误能力估计能否因反馈被修正。
8. **Conflict Resolution**：目标、偏好和约束冲突时的决策是否一致且可解释。
9. **State Rollback**：恢复历史快照后，行为是否恢复到对应轨迹。

完成标准：

- 相同起点的智能体因不同经验形成统计上可区分的行为轨迹。
- 关键身份保持稳定，允许更新的字段能响应证据。
- 能区分“合理适应”和“无约束漂移”。

### Phase 5：综合评估与研究结论

目标：形成对 Persistent Self Architecture 的可复现结论。

主要任务：

1. 冻结最终协议，运行完整实验矩阵。
2. 汇总效应量、置信区间、失败案例和替代解释。
3. 复核数据泄漏、提示词混淆和评估器偏差。
4. 发布复现实验、技术报告和负面结果。
5. 决定下一步：继续扩展、修改架构或终止假设。

完成标准：

- 每个核心假设都有明确结论和证据等级。
- 第三方能够依据文档复现关键实验。
- 结论严格限定在实验范围内，不外推为意识证明。

## 7. 实验设计规范

### 7.1 基本实验单元

每个实验至少记录：

- 实验编号与假设编号
- 模型、权重和代码版本
- Self State 初始值及来源
- 输入、环境和任务版本
- 干预类型与强度
- 随机种子与采样参数
- 原始输出和结构化指标
- 运行时间与资源消耗
- 异常、失败和人工判断说明

### 7.2 因果实验模板

```text
固定：
  模型权重、任务、输入、采样参数、随机种子

仅改变：
  Self State 或指定 state 组件

观察：
  行为选择、目标坚持、偏好一致性、置信度、错误率

验证：
  效应是否可重复、是否随干预强度变化、是否存在替代解释
```

### 7.3 评价指标

核心指标：

- **Identity Consistency**：跨时间身份相关回答或行为的一致性
- **Goal Retention**：经过干扰后仍保持或恢复长期目标的比例
- **Behavioral Transfer**：交换 Self State 后行为特征的迁移程度
- **Intervention Effect Size**：Self State 干预造成的行为效应量
- **Trajectory Divergence**：不同经验分支之间的轨迹差异
- **Adaptation Score**：Self State 对新证据的合理更新程度
- **Self Drift**：无充分证据时 Self State 的变化幅度
- **Calibration Error**：置信度与实际正确率之间的误差
- **Recovery Fidelity**：恢复快照后行为的复原程度

辅助指标：

- 任务正确率
- 输出稳定性
- 状态存储成本
- 推理延迟
- 训练或适配成本

具体公式、阈值和主要指标必须在正式实验前写入 `docs/evaluation_protocol.md`，避免根据结果临时改变成功标准。

### 7.4 统计与复现要求

- 报告均值、离散程度、效应量和置信区间，不只报告单次样例。
- 对关键实验使用多个随机种子和多个任务模板。
- 分离开发集与最终评估集。
- 保存原始结果，不覆盖失败运行。
- 自动生成汇总表，人工评分须保留评分标准和盲评信息。

## 8. 数据与任务集

初始任务集应保持小而可解释，覆盖：

1. 稳定偏好选择：在内容变化时保持相同偏好原则。
2. 长期目标：完成多阶段任务并抵抗无关干扰。
3. 身份追踪：跨会话识别自身历史和当前状态。
4. 能力校准：对可答、不可答和需进一步信息的问题做区分。
5. 冲突决策：处理目标、偏好和价值之间的权衡。
6. 经验分叉：不同反馈历史导致不同后续选择。

任务样例、生成方式和版本信息统一放在 `data/tasks/`。测试集在指标和架构冻结前不用于调参。

## 9. 建议的仓库结构

```text
Persistent-Self-Architecture/
├─ README.md
├─ PROJECT_OVERVIEW.md
├─ PROJECT_PLAN.md
├─ docs/
│  ├─ research_map.md
│  ├─ definitions.md
│  ├─ research_claims.md
│  ├─ architecture.md
│  ├─ task_design.md
│  ├─ evaluation_protocol.md
│  ├─ state_format.md
│  ├─ implementation_spec.md
│  ├─ experiment_log.md
│  └─ decision_log.md
├─ configs/
├─ data/
│  └─ tasks/
├─ src/
│  ├─ model/
│  ├─ state/
│  ├─ self_model/
│  ├─ coupling/
│  └─ evaluation/
├─ experiments/
├─ tests/
└─ results/
```

`results/` 中的大型权重、缓存和原始输出应根据体积决定是否进入版本控制，并记录生成方式。

## 10. 工作优先级

### P0：立即执行

1. 审阅并冻结术语和操作性定义。
2. 审阅并冻结研究主张与模型架构。
3. 冻结 EXP-001 的 A/B 变量、任务生成和混淆控制。
4. 审阅评价、状态格式和实现规范，形成冻结候选。
5. 实现不依赖目标模型的纯逻辑骨架和验证器。
6. 确定远程环境与 checkpoint，完成接口调查、数值 roundtrip 和 Prompt-visible 开发门。**环境、接口调查和内存 roundtrip 已完成；磁盘/跨进程 roundtrip 待运行。**
7. 填写工程参数并完成最终预注册。
8. 只在预注册后运行确认性 state 实验。

### P1：基线完成后执行

1. State Probe、消融和插值。
2. 显式 Self State schema。
3. Self Encoder 与 gated injection 原型。
4. Prompt、Memory 与 Self State 对比实验。

### P2：确认有研究信号后执行

1. 长期 Self Evolution。
2. 多分支个体轨迹。
3. 更复杂的元认知与冲突解决。
4. 扩展模型规模或比较其他 SSM。

## 11. 首个迭代周期

当前第一个短周期已完成从“概念”到“远程模型接口门通过”的闭环，正在执行
SafeTensors checkpoint 与跨进程恢复开发门。开发门只校准接口、tokenizer、
数值容差和任务能力；不运行确认集。

### 迭代目标

形成一个定义清楚、可证伪、对照充分，并能在远程机器上执行的首个实验设计。

### 迭代任务

1. 冻结 Self、Memory、Persona 和 Persistent Self 的操作性边界。
2. 明确主研究问题、主要假设和可接受的否定结论。
3. 画出实验因果图，列出干预变量、结果变量和混淆因素。
4. 固定 Prompt、Memory、reset、random 和 matched-context 对照。
5. 冻结任务生成规则、开发/测试分离和答案平衡方式。
6. 起草主要指标、统计方法、样本量和数值阈值。
7. 审查设计草案后进入远程开发门。
8. 根据不含确认结果的工程信息填写参数并完成最终预注册。

### 迭代验收

- 关键术语没有循环定义，结论措辞与证据等级对应。
- 每个主要假设都有干预、对照、指标和明确否定条件。
- 因果图中的主要后门路径均有控制或限制说明。
- 协议可以在看不到正式测试结果的情况下冻结。
- 实验草案状态从 Draft 变为 Preregistered 后，才进入实现周期。

## 12. 风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| 把上下文记忆误认为 Self | probe 读取到最近文本内容 | 使用内容匹配、上下文清除和延迟测试 |
| 把 Persona Prompt 效果误认为机制创新 | 显式 Self 与提示词效果相同 | 设置等信息量 Prompt 和 Memory 基线 |
| state 变化不代表因果作用 | 能解码但干预无效 | 将交换、消融和插值作为必要证据 |
| Self State 快速漂移 | 身份或价值随少量输入改变 | 字段分层、更新阈值、证据累计和回滚 |
| Self State 完全僵化 | 无法根据纠错经验调整 | 区分快慢变量并设置可解释更新规则 |
| 评估依赖主观样例 | 只展示少数“有趣”对话 | 使用预注册任务、批量运行和盲评 |
| 计算资源不足 | 无法完成大规模训练 | 冻结基础模型，优先 probe、小模块和推理实验 |
| 概念外推过度 | 将行为特征称为意识 | 使用操作性术语，明确证据边界 |
| 结果不可复现 | 运行环境或采样不稳定 | 固定版本、种子、配置并保存原始输出 |

## 13. 决策门

每个阶段结束时进行一次 Go / Revise / Stop 决策：

- **Go**：核心效应可重复，进入下一阶段。
- **Revise**：信号存在但混淆较大，修改实验或架构后重测。
- **Stop**：关键假设在充分对照下被否定，记录负面结论并停止该路线。

关键决策：

1. 原生 RWKV state 是否通过跨时间因果载体资格验证？
2. 显式 Self State 是否优于等信息量 Prompt 或 Memory？
3. Self State 的收益是否来自真正的持续性和因果机制？
4. 架构收益是否足以支持更长周期或更大规模实验？

## 14. 项目管理规范

### 14.1 每次实验前

- 写明假设、主要指标、对照组和预期结果。
- 分配唯一实验编号。
- 冻结配置和任务版本。

### 14.2 每次实验后

- 保存原始输出、聚合指标和环境信息。
- 记录支持证据、反对证据和替代解释。
- 将意外结果加入待验证列表，不直接修改理论。

### 14.3 每个迭代结束

- 更新 `docs/experiment_log.md`。
- 在 `docs/decision_log.md` 记录重要选择及理由。
- 更新本文档的阶段状态和下一周期 P0 任务。
- 清理无法复现的结论，保留失败实验记录。

## 15. 项目完成的判断标准

项目不以“证明意识”为完成条件。满足以下任一结果，都可以形成有价值的阶段性结论：

### 结果 A：支持

原生 recurrent state 或显式 Self State 在严格对照下表现出持续、个体化、可更新和因果影响；显式方案进一步优于 Prompt、Memory 与原生 state 基线。

### 结果 B：有限支持

Self State 只在部分任务、时间尺度或耦合机制下有效，明确其适用边界。

### 结果 C：否定

在充分实验后，Self State 没有超出上下文记忆、Persona Prompt 或普通状态机制的独立价值。

无论得到哪种结果，最终产出都应包括：

- 可复现实验代码
- 明确的操作性定义
- 完整对照和消融
- 正面与负面结果
- 对核心假设的证据化结论

## 16. 下一步行动

继续完成 Phase 0，不启动实验：

1. 共同审阅并冻结 `docs/definitions.md`。
2. 共同审阅 `docs/research_claims.md` 与 `docs/architecture.md`。
3. 共同审阅并冻结 `docs/task_design.md` 的变量、任务和混淆控制。
4. 共同审阅 `docs/evaluation_protocol.md`，冻结指标、样本量、统计方法、SESOI 和阈值。
5. 共同审阅 `docs/state_format.md` 与 `docs/implementation_spec.md`。
6. 确定远程 checkpoint/tokenizer，填写工程参数并完成预注册。
7. 完成设计审查后，再实现纯逻辑骨架和远程 state 保存/恢复闭环。
