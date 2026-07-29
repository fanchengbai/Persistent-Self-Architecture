# 第二批方法论文精读笔记

> 状态：v1.0  
> 日期：2026-07-29  
> 目的：为 PSA 冻结“什么算相关证据、什么算因果证据、如何避免把记忆残留误认为 Self”的方法标准。

## 1. Towards Best Practices of Activation Patching in Language Models

- 论文：[arXiv:2309.16042](https://arxiv.org/abs/2309.16042)
- 研究问题：activation patching 的 corruption、评价指标和滑动窗口等方法选择，会不会改变机制解释结论？
- 基本方法：
  1. clean run：运行干净输入并缓存激活；
  2. corrupted run：运行受控扰动输入；
  3. patched run：把 clean activation 恢复到 corrupted run 的指定位置，观察输出恢复程度。
- 关键发现：
  - 高斯噪声可能把输入推到分布外并破坏多个内部机制；对称 token 替换通常更适合作为语义对照。
  - probability、logit difference、KL divergence 可能给出不同结论。
  - 只看目标 token 概率可能漏掉“负向组件”；细粒度定位宜优先使用 logit difference。
  - 多层滑动窗口的结果不能简单等同于各层单独 patch 的效应之和。
- 对 PSA 的直接约束：
  - 优先使用成对、等长度、分布内的历史轨迹作为 clean/corrupted 条件。
  - 主要决策指标采用预注册的 logit difference，同时报告概率和 KL divergence 的敏感性分析。
  - state swap、layer patch、channel patch 必须分开报告，不能把窗口效应解释为单层因果效应。
  - 任何“定位到 Self 区域”的结论都必须披露 corruption 和指标选择。
- 局限：该方法能定位某次前向计算中的因果贡献，但本身不能证明跨时间的身份持续性。

## 2. Latent Causal Probing: A Formal Perspective on Probing with Causal Models of Data

- 论文：[arXiv:2407.13765](https://arxiv.org/abs/2407.13765)
- 发表状态：COLM 2024
- 研究问题：如何区分“语言模型表示了某变量”与“探针自己学会了该变量”？
- 基本方法：
  - 先为数据生成过程建立结构因果模型（SCM）；
  - 把目标概念定义为 SCM 中的潜变量；
  - 通过反事实数据生成机制和因果中介分析，隔离经过语言模型表示的路径，排除探针容量带来的捷径。
- 关键发现：
  - 高 probe accuracy 可能主要来自 probe 的表达能力，而非模型内部表示。
  - 原始 probe 分数与因果中介测量可能不相关，甚至方向相反。
  - 复杂探针并非天然无效；关键是反事实基线和中介路径是否成立。
- 对 PSA 的直接约束：
  - 在训练 probe 前，必须先写出目标变量、生成机制、允许变化的因素和保持不变的因素。
  - probe 数据优先采用程序化生成的成对反事实任务，避免从自然对话标签直接推断“Self”。
  - probe 只提供“可读性”证据；swap、ablation、restore 才能提升到因果证据。
  - 若无法构造可信的反事实世界，就不得把 probe 结果写成“模型内部存在该 Self 概念”。
- 局限：真实自然语言中的身份和目标没有唯一、已知的数据生成 SCM，因此该方法更适合 PSA 的合成任务和受控微世界，不适合直接验证开放域“自我”。

## 3. Representation Engineering: A Top-Down Approach to AI Transparency

- 论文：[arXiv:2310.01405](https://arxiv.org/abs/2310.01405)
- 研究问题：能否以群体级表示为分析单位，读取并控制诚实、效用、风险、情绪等高层概念与功能？
- 基本方法：
  - Representation Reading：设计刺激和任务、收集神经活动、用线性模型构造 reading vector。
  - LAT 基线：常以成对活动差做 PCA，取第一主成分作为读取方向。
  - Representation Control：使用 reading vector、输入相关的 contrast vector 或 LoRRA 改变内部表示。
- 证据分级：
  1. Correlation：表示能预测目标变量；
  2. Manipulation：增强或抑制表示会改变行为；
  3. Termination：移除表示会损害对应功能；
  4. Recovery：移除后重新注入表示可恢复功能。
- 对 PSA 的直接约束：
  - Self 特征可先用 representation reading 探索，但不能由读取准确率直接下因果结论。
  - 至少组合 manipulation、termination、recovery 三类实验。
  - 需要报告控制对通用能力和非目标行为的副作用，避免把整体模型损坏误判为特异性效应。
  - reading vector 是输入无关的统一方向，可能不适合情境依赖的 Self；应与 state swap 和输入相关 contrast vector 比较。
- 局限：论文关注高层表示在一次推理中的读取与控制，没有建立跨会话持续、经验更新和身份轨迹的充分标准。

## 4. Characterizing Mamba's Selective Memory using Auto-Encoders

- 论文：[arXiv:2512.15653](https://arxiv.org/abs/2512.15653)
- 研究问题：固定大小的 Mamba hidden state 会选择性保留或遗忘哪些输入信息？
- 基本方法：
  - 冻结预训练 Mamba encoder；
  - 用最后的 SSM state 和 convolutional state 初始化 decoder；
  - 让 decoder 重建原输入；
  - 用 token omission rate 和 ROUGE F1 衡量可恢复信息。
- 实验范围：Mamba 130M–1.4B，序列长度 4–256。
- 关键发现：
  - 序列越长，重建质量整体越差；较早 token 比近期 token 更容易被遗忘。
  - 数字、变量、组织实体和非标准美式英语方言更容易丢失。
  - 信息遗忘与预训练语料中的 token 频率存在部分关联。
- 对 PSA 的直接约束：
  - recurrent state 不是无损历史容器，不能假定身份、目标或偏好会自然长期保留。
  - 应分别测量“状态中可恢复的信息”和“决策时实际使用的信息”。
  - 历史位置、token 类型、训练频率和序列长度都是必要分层变量。
  - 精确标识符、数字和版本号不适合作为唯一身份信号；它们可能测到架构的 token 记忆偏差。
- 局限：
  - decoder 错误与 encoder state 的真实信息丢失存在混淆；
  - 只研究 Mamba 家族和较短序列；
  - 没有下游任务或因果使用测试；
  - 训练 decoder 的成本并不低，1.4B 实验曾需 48GB A6000 上运行数周。

## 5. 跨论文方法结论

### 5.1 PSA 证据阶梯

| 等级 | 证据 | 允许的结论 |
|---|---|---|
| E0 | 模型自述“我是谁/我想要什么” | 只说明生成了相应文本 |
| E1 | 行为跨模板保持一致 | 存在稳定行为现象 |
| E2 | 内部状态可被 probe 或重建 | 状态携带可读相关信息 |
| E3 | swap / ablation / interpolation 改变行为 | 状态对目标行为有因果贡献 |
| E4 | termination 后 restore 能恢复目标功能，且副作用受控 | 候选机制具有必要性或充分性证据 |
| E5 | 跨时间 fork / swap / rollback 成立，并优于等信息量 Prompt 与 Memory | 可称为 Persistent Self 候选机制 |

E0–E2 不得单独支持 “Persistent Self” 结论。

### 5.2 首批实验的方法规范

1. 先定义变量和因果图，再生成任务和训练 probe。
2. 尽量使用成对、等长度、分布内的反事实历史。
3. 主指标预注册为目标选项的 logit difference；概率与 KL divergence 作为敏感性分析。
4. 同时报告 target effect、通用任务能力和非目标变量变化。
5. 单层、窗口、完整 state 的干预分开解释。
6. 至少使用多个模板、答案排列、随机种子和独立测试集。
7. 任何读取结果都必须由行为干预验证；任何行为干预都必须与 Prompt、Memory 和 state-reset 基线比较。

## 6. 对 PSA 研究主张的收缩

当前不主张“找到了自我向量”或“recurrent state 天然形成身份”。首阶段只检验：

> 在受控任务中，跨时间保存的内部状态是否同时携带并因果使用身份锚点与目标约束；这种作用是否不能被当前提示、普通记忆检索或短期文本残留充分解释。

