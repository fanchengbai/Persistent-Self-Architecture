# EXP-001 评价与统计协议

> 版本：v0.1  
> 状态：Phase 0 统计草案，尚未预注册、尚未查看正式测试结果  
> 日期：2026-07-29  
> 对应任务：[`task_design.md`](task_design.md)  
> 对应协议：[`EXP-001 PROTOCOL`](../experiments/EXP-001_state_persistence_swap/PROTOCOL.md)  
> 目标：在实验运行前冻结终点、实验单位、样本量、统计方法、实质效应阈值和 Go / Revise / Stop 规则。

## 1. 基本原则

评价遵循：

1. **因果问题优先**：主要终点来自 state 的配对交换和恢复，不来自模型自述。
2. **未采样分数优先**：主要分析使用候选答案的 logits 或序列 log-likelihood，不以单次随机采样文本作为主要数据。
3. **配对设计优先**：同一任务、标签、模板和选项排列下，只改变 state 来源。
4. **实验单位正确**：一个完整的四状态配对组才是独立单位；同一轨迹产生的多个查询不是独立样本。
5. **效应量与区间优先**：同时报告点估计、置信区间、原始分布和最小有意义效应，不只报告 p 值。
6. **正面与否定对称**：没有显著差异不等于证明无效；否定结论需要等效性检验或足够窄的区间。
7. **确认与探索分离**：layer/channel 定位、插值形状和事后模板分析不能用于挽救失败的主要终点。
8. **结果不可改协议**：正式测试集解封后，不修改主要终点、方向、排除规则或阈值。

这些选择参考 NLP 中对依赖样本、统计功效、配对检验和实质显著性的讨论：

- [The Hitchhiker’s Guide to Testing Statistical Significance in NLP](https://aclanthology.org/P18-1128/)
- [With Little Power Comes Great Responsibility](https://aclanthology.org/2020.emnlp-main.745/)
- [Exact Paired-Permutation Testing for Structured Test Statistics](https://aclanthology.org/2022.naacl-main.360/)
- [Equivalence Tests: A Practical Primer](https://doi.org/10.1177/1948550617697177)

## 2. 研究问题与分析层级

### 2.1 基础设施问题

- I0：保存与恢复的 state 是否在固定环境中数值保真？

### 2.2 能力问题

- C0：模型在当前 Prompt 明示 \(I/G\) 时是否能理解组合任务？

### 2.3 主要因果问题

- P1-I：改变 state 中的实例身份约束来源，identity 维度分数是否按来源方向改变？
- P1-G：改变 state 中的当前目标来源，goal 维度分数是否按来源方向改变？
- P2：包含 \(I+G\) 的 state 是否能在同一次决策中选择唯一联合行动？

### 2.4 次要问题

- S1：I/G 是否可从 state 中被受限 probe 读出？
- S2：效应如何随 token 距离、任务切换和干扰类型衰减？
- S3：state restore 和 rollback 能否恢复 logits 与行为？
- S4：state 作用是否优于 reset、random 和 matched-context？
- S5：Track S 的结果能否迁移到 Track N？

### 2.5 探索问题

- 哪些层、head 或 channel 对效应贡献更大？
- interpolation 是否呈近似单调剂量—响应？
- 哪类 token、模板或干扰导致异常衰减？
- probe 与行为因果效应是否相关？

探索结果只能生成新假设。

## 3. 数据层级与独立实验单位

### 3.1 最小观测

一次模型 continuation 在一个固定 state 和固定 query 下产生：

```text
四个选项的 score
选项概率
最终 argmax 选择
输出格式状态
native state 统计
运行元数据
```

### 3.2 轨迹

一条轨迹对应一个确定的：

```text
I value
G value
history template
history order
label pair
common suffix
delay / distractor
generator seed
```

### 3.3 四状态配对组

一个独立 factorial group 同时包含：

```text
S_00
S_01
S_10
S_11
```

四条轨迹除 I/G 取值和必要的镜像标签外，其余生成因素完全匹配。

### 3.4 主要实验单位

主要统计分析单位是 factorial group，而不是：

- token；
- 单个选项；
- 单次 continuation；
- 同一 state 上的多个测试模板；
- 同一历史的多个轻微改写。

若同一 group 产生多个观测，先在 group 内构造预注册 contrast，再跨 group 推断。

## 4. 选项分数与基本符号

每个 query 有四个组合选项。记未归一化选项分数为：

\[
\ell_{00}, \ell_{01}, \ell_{10}, \ell_{11}
\]

如果答案是单 token，\(\ell\) 为对应 token logit；如果答案为多 token 序列：

\[
\ell_{ig}
=
\sum_k \log p(y_{ig,k}\mid y_{ig,<k}, x, R)
\]

主要分析不做长度归一化，因为正式答案代码应尽量等长。若 tokenizer 限制导致长度不同，则在解封前冻结是否使用每 token 平均 log-likelihood，并在所有条件一致使用。

选项概率：

\[
p_{ig}
=
\frac{\exp(\ell_{ig})}
{\sum_{a,b}\exp(\ell_{ab})}
\]

## 5. 维度分数

### 5.1 Identity log-odds

\[
q_I
=
\operatorname{logsumexp}(\ell_{10},\ell_{11})
-
\operatorname{logsumexp}(\ell_{00},\ell_{01})
\]

正值偏向 \(I_1\)，负值偏向 \(I_0\)。

### 5.2 Goal log-odds

\[
q_G
=
\operatorname{logsumexp}(\ell_{01},\ell_{11})
-
\operatorname{logsumexp}(\ell_{00},\ell_{10})
\]

正值偏向 \(G_1\)，负值偏向 \(G_0\)。

### 5.3 对齐分数

对状态 \(S_{ig}\)，定义：

\[
s_I(i)=2i-1,\qquad s_G(g)=2g-1
\]

\[
A_I=s_I(i)q_I
\]

\[
A_G=s_G(g)q_G
\]

正的 \(A_I/A_G\) 表示输出分布与 state 来源的 I/G 值方向一致。

### 5.4 联合 log-margin

对正确联合选项 \((i,g)\)：

\[
J_{ig}
=
\ell_{ig}
-
\operatorname{logsumexp}
\{\ell_{ab}:(a,b)\neq(i,g)\}
\]

\(J>0\) 表示正确联合选项的分数高于三个错误选项的总竞争质量。

### 5.5 行为结果

```text
identity_correct = 选中选项的 I 维度正确
goal_correct     = 选中选项的 G 维度正确
joint_correct    = identity_correct AND goal_correct
```

## 6. 主要终点

### E1：Identity Directional Transfer

对 group \(j\)，在 G 固定时比较 I 来源：

\[
D_{I,j}
=
\frac{1}{2}
\sum_{g\in\{0,1\}}
\left[
q_I(R_{1g})-q_I(R_{0g})
\right]
\]

若 identity 信息随 state 来源迁移，\(D_I>0\)。

### E2：Goal Directional Transfer

\[
D_{G,j}
=
\frac{1}{2}
\sum_{i\in\{0,1\}}
\left[
q_G(R_{i1})-q_G(R_{i0})
\right]
\]

若 goal 信息随 state 来源迁移，\(D_G>0\)。

### E3：Joint Binding

每个 group 的联合分数：

\[
\bar{J}_j
=
\frac{1}{4}
\sum_{i,g}J_{ig}(R_{ig})
\]

联合准确：

\[
Acc^{joint}_j
=
\frac{1}{4}
\sum_{i,g}
\mathbb{1}
\left[
\arg\max_{a,b}\ell_{ab}(R_{ig})=(i,g)
\right]
\]

P2 不能只靠 E3 单独通过。它同时要求 E1、E2 成立，且联合准确超过单变量策略的理论上限。

## 7. 主要基线对比

### 7.1 Reset advantage

对每个目标 \(i,g\)，比较来源 state 与 reset state：

\[
\Delta J_{\text{reset}}
=
\mathbb{E}_{i,g}
\left[
J_{ig}(R_{ig})-J_{ig}(R_{\text{reset}})
\right]
\]

### 7.2 Random-state advantage

\[
\Delta J_{\text{random}}
=
\mathbb{E}_{i,g}
\left[
J_{ig}(R_{ig})-J_{ig}(R_{\text{random-matched}})
\right]
\]

random state 必须与真实 state 的 shape、dtype 和预注册尺度统计匹配。

### 7.3 Matched-context advantage

matched-context 包含相同标签和相近 token 统计，但不包含当前 Agent 的 I/G 绑定关系：

\[
\Delta J_{\text{matched}}
=
\mathbb{E}_{i,g}
\left[
J_{ig}(R_{ig})-J_{ig}(R_{\text{matched}})
\right]
\]

### 7.4 Prompt-normalized retention

Prompt-visible 提供任务能力参照：

\[
RR_I
=
\frac{\mathbb{E}[D_I^{state}]}
{\mathbb{E}[D_I^{prompt}]}
\]

\[
RR_G
=
\frac{\mathbb{E}[D_G^{state}]}
{\mathbb{E}[D_G^{prompt}]}
\]

只在分母通过能力门且远离零时报告。比率使用跨 group 的均值之比及 cluster bootstrap 区间，不计算不稳定的逐样本比率。

## 8. 特异性终点

I-only contrast 不应同等改变 G：

\[
N_{I,j}
=
\frac{1}{2}
\sum_g
\left[
q_G(R_{1g})-q_G(R_{0g})
\right]
\]

\[
Spec_{I,j}
=
|D_{I,j}|-|N_{I,j}|
\]

类似地：

\[
Spec_{G,j}
=
|D_{G,j}|-|N_{G,j}|
\]

主要报告带符号目标对比和非目标绝对变化。若目标与非目标一起无差别变化，只能说明 state 造成一般扰动。

## 9. 恢复保真终点

### 9.1 必须完全一致

- checkpoint manifest；
- 模型、tokenizer 和代码版本；
- state shape 与 dtype；
- tensor checksum；
- tokenized continuation input。

### 9.2 数值终点

- 最大 logits 绝对误差；
- logits RMSE；
- 候选分布 KL divergence；
- top-1 token 一致率；
- 完整生成一致率。

### 9.3 数值容差冻结规则

由于容差依赖 dtype、kernel 和设备，不在架构未知时拍定统一常数。采用以下不看语义结果的工程校准：

1. 在开发集上选择与 I/G 无关的固定短序列；
2. 完成至少 100 次 continuous vs restore roundtrip；
3. 记录每项误差的经验分布；
4. 容差定义为：

```text
max(工程绝对下限, 开发 roundtrip 误差 99.9% 分位数的 10 倍)
```

5. 在正式测试解封前，把实际数值写入本节并冻结；
6. 测试中任何超出容差的 run 标记为 infrastructure failure，不进行语义解释。

同一软件和硬件配置若理论上应 bitwise deterministic，则优先要求 bitwise equality。

## 10. 能力门

### 10.1 Track S

Prompt-visible 条件必须同时达到：

- joint accuracy 的 95% cluster bootstrap 下界不低于 0.80；
- identity 与 goal marginal accuracy 下界均不低于 0.90；
- 格式有效率不低于 0.99；
- 四个答案位置的准确率差异不出现预注册的严重不平衡。

### 10.2 Track N

使用相同标准。若 Track S 通过而 Track N 未通过：

- Track S 仍可作为主要确认任务；
- Track N 只报告能力限制；
- 不把 Track N state-only 结果用于否定 P1/P2。

### 10.3 模板级资格

模板先在开发集资格审查。正式测试只能使用在不查看 state-only 结果时已通过能力门的模板族。

## 11. 最小有意义效应（SESOI）

为避免把极小但统计显著的 logit 变化称为机制证据，v0.1 采用以下默认阈值：

| 终点 | 默认 SESOI | 解释 |
|---|---:|---|
| \(D_I\) | 0.50 log-odds | state 来源至少产生可辨认的 identity 方向变化 |
| \(D_G\) | 0.50 log-odds | state 来源至少产生可辨认的 goal 方向变化 |
| \(\bar{J}\) 相对 reset/random | 0.50 log-margin | 联合选项获得实质优势 |
| joint accuracy | 高于 0.50 至少 0.10 | 超过只知道一个变量时的 0.50 上限 |
| \(Spec_I,Spec_G\) | 0.25 | 目标变化明显大于非目标变化 |
| prompt-normalized retention | 0.20 | 保留显式 Prompt 信号的至少五分之一 |

这些数值属于研究判断，不来自测试结果。共同审阅时可以修改一次；正式测试解封后不得修改。

## 12. 样本量设计

### 12.1 Confirmatory Core Set

Track S 默认使用：

```text
N = 320 个独立 factorial groups
每组包含 S_00、S_01、S_10、S_11
```

在近似配对正态、方向性检验、总体 \(\alpha=0.05\)、三个主要终点进行保守校正的条件下，320 个独立组大致可为标准化配对效应 \(d_z\approx0.20\) 提供约 90% 功效。正式实现前使用任务生成器和开发集 nuisance variance 做模拟复核，但不得把 N 下调到 320 以下。

### 12.2 Track N

只有通过能力门才进入确认性迁移测试：

```text
N = 320 个独立 factorial groups
```

Track N 被定义为外部表达迁移终点，不与 Track S 合并制造更大的伪样本量。

### 12.3 Delay Set

默认五个 delay 档位：

```text
每档至少 80 个独立 factorial groups
总计至少 400 groups / track
```

Delay Set 与 Core Set 使用不同 generator seeds。若计算预算不足，优先保证 Core Set；delay 曲线可作为第二批确认实验，但必须在运行前整体冻结。

### 12.4 Probe Set

probe 的样本量和训练预算独立：

- 至少 512 条训练轨迹；
- 至少 256 条 ID 测试轨迹；
- 至少 256 条 label/template OOD 测试轨迹；
- 四种 I/G 状态平衡；
- 按 trajectory group 切分。

probe 是次要证据，不得占用主要因果实验的测试集。

### 12.5 样本量变更

只允许因明确计算预算在正式运行前整体修改，并记录理由。开始读取确认集结果后：

- 不追加样本追逐显著性；
- 不因趋势“接近阈值”延长运行；
- 基础设施失败导致的预注册重跑不计为追加样本，但必须保留失败记录。

## 13. 主要统计分析

### 13.1 点估计

对 E1–E3：

1. 先在每个 factorial group 内计算 contrast；
2. 再对 group-level contrast 求均值、中位数和分位数；
3. 报告原始散点或分布图；
4. 不把 group 内四条轨迹当作四个独立样本。

### 13.2 置信区间

主要区间采用：

- factorial group 为重采样单位；
- 10,000 次 cluster bootstrap；
- 95% BCa interval；
- 固定 bootstrap seed；
- 同一个 bootstrap draw 同时重采样所有配对条件。

若 BCa 因离散或退化分布不可用，使用 percentile cluster bootstrap，并明确报告降级。

### 13.3 显著性检验

E1、E2 和 E3 的主要比较使用 group-level 配对置换或符号翻转检验：

- 方向在预注册中明确，因此使用单侧检验；
- 小样本可精确枚举时使用 exact；
- 否则使用至少 100,000 次 Monte Carlo permutation；
- 固定 permutation seed；
- 报告未经校正和校正后 p 值。

### 13.4 多重比较

主要 family：

```text
E1 Identity Transfer
E2 Goal Transfer
E3 Joint Binding
```

对三项使用 Holm correction，family-wise \(\alpha=0.05\)。

下列结果单独标记为 secondary/exploratory，不与主要 family 混合：

- delay 各档位；
- layer/channel；
- 多个 probe；
- Track N 子模板；
- interpolation 多个 alpha；
- 个别标签分析。

### 13.5 实质效应判定

“支持”需要同时满足：

- Holm 校正后方向性检验通过；
- 点估计达到 SESOI；
- 95% CI 下界大于 0；
- 基线和特异性要求通过。

仅 p 值通过但效应未达到 SESOI，报告为“可检测但不足以支持机制主张”。

## 14. 联合绑定判定

P2 必须同时满足：

1. E1 通过；
2. E2 通过；
3. joint accuracy 的 95% 下界至少为 0.60；
4. \(\bar{J}\) 的均值至少达到 0.50，且 95% 下界大于 0；
5. joint accuracy 显著高于：
   - I-only 策略；
   - G-only 策略；
   - 最近变量策略；
   - 固定答案位置策略；
6. `swapped_both` 后联合答案向来源状态迁移；
7. random/reset 不能产生同等效果。

若 T1/T2 有效但以上任一核心条件失败，结论为 weak persistence。

## 15. 特异性与非目标能力

### 15.1 变量特异性

默认要求：

- \(Spec_I\) 点估计至少 0.25，95% 下界大于 0；
- \(Spec_G\) 点估计至少 0.25，95% 下界大于 0。

若 state 的变化同时无差别推动 I/G，只报告“整体状态影响”，不报告字段或维度特异性。

### 15.2 通用能力副指标

每批 state 干预同时运行一组与 I/G 无关的短任务：

- 固定 token 续写；
- 简单词法或格式选择；
- 与任务标签无关的四选一控制；
- 基础语言建模 loss/perplexity 摘要（实现允许时）。

架构损伤预警：

- 控制任务准确率下降超过 5 个百分点；
- 平均目标 token log-prob 下降超过开发基线的预注册容差；
- 格式失败率增加超过 2 个百分点；
- state norm/RMS 超出开发分布的 99.9% 范围。

出现预警时，目标效应必须标记为可能非特异，不能直接计入 Go。

## 16. Delay 与持久性分析

### 16.1 主要表示

对每个 delay \(d\) 报告：

- \(D_I(d)\)；
- \(D_G(d)\)；
- joint accuracy；
- \(\bar{J}(d)\)；
- 相对 D0 的 retention fraction；
- token 距离而不是只报告“轮数”。

### 16.2 模型

确认性分析优先使用预先指定的 delay 分类因素，不强制假定指数衰减。

探索性分析可以拟合：

\[
E(d)=E_0\exp(-\lambda d)+b
\]

并报告半衰期：

\[
d_{1/2}=\frac{\ln 2}{\lambda}
\]

若拟合不稳定或非单调，不报告单一半衰期，用分层点估计代替。

### 16.3 Persistent 的最低时间要求

“跨时间”至少要求在预注册标准 delay 和一次任务切换条件下仍达到主要效应门槛。只在 D0 成立的效应归类为 immediate context effect。

具体标准 delay 的 token 数在 checkpoint/context 调查后、测试解封前写入。

## 17. Probe 评价

### 17.1 任务

分别预测：

- I；
- G；
- I×G 四分类。

### 17.2 模型限制

首版 probe 优先：

- logistic regression；
- 线性 SVM 或等价线性分类器；
- 固定正则化搜索空间；
- 不使用大型非线性 decoder 作为主要 probe。

### 17.3 对照

- shuffled labels；
- random state；
- matched-context；
- label-pair OOD；
- template OOD；
- history-order OOD。

### 17.4 指标

- balanced accuracy；
- macro F1；
- AUROC（二分类）；
- calibration；
- OOD performance；
- 与行为 DTE 的 group-level 相关。

probe 成功只达到 E2 证据。无论准确率多高，都不能替代 E1–E3。

## 18. 等效性与否定结论

### 18.1 不能这样解释

```text
p > 0.05
→ 没有效应
```

这是不允许的。

### 18.2 实质零区间

对 \(D_I/D_G\)，默认等效区间：

\[
[-0.50,\;0.50]
\]

对 joint accuracy 相对 0.50 ceiling 的优势，默认等效区间：

\[
[-0.10,\;0.10]
\]

使用 TOST 或等价的区间判定。只有效应区间足够窄并落在实质零区间内，才可以报告“未发现达到最小有意义幅度的效应”。

### 18.3 三种结果

| 结果 | 解释 |
|---|---|
| 支持最小效应，排除零 | 有意义的正效应 |
| 排除最小效应，支持等效 | 实质上不足 |
| 两者都不能排除 | 不确定，需要更多精度或修改设计 |

## 19. 敏感性分析

以下分析预注册但不替代主要分析：

1. probability 而非 logit/log-likelihood；
2. KL divergence；
3. median 与 trimmed mean；
4. percentile bootstrap；
5. mixed-effects model；
6. 只保留格式完全有效的样本；
7. Track S / Track N 分开；
8. 不同 history order；
9. 不同 label pool；
10. 去掉最短 delay。

若敏感性分析与主要分析方向冲突，结论降级并解释，不选择最有利的指标。

## 20. 层级模型

作为次要确认分析，拟合 mixed-effects model。

### 20.1 连续终点

对 group-level log-odds 或 log-margin：

```text
effect ~ state_condition
       + history_order
       + delay
       + track
       + state_condition:track
       + (1 | history_template)
       + (1 | label_pair)
       + (1 | factorial_group)
```

若模型奇异或不收敛，简化随机效应结构，并完整报告。

### 20.2 二元终点

对 `joint_correct` 使用 logistic mixed-effects model。不得把四个相关状态当作无层级的独立 Bernoulli 样本。

主要结论仍以预注册 group-level contrast 为准；层级模型用于检查泛化和异质性。

## 21. 排除、失败与重跑

### 21.1 允许排除

- checkpoint checksum 不匹配；
- state shape/dtype 不兼容；
- 运行中断导致没有完整输出；
- tokenizer 或模型版本错误；
- 生成器验证发现 Prompt 泄漏；
- 任务文件损坏；
- 超出已冻结数值恢复容差。

### 21.2 不允许排除

- 模型答错；
- state swap 没有效果；
- 输出概率很低；
- 某模板表现差但格式有效；
- outlier 使均值变差；
- 结果不符合理论。

### 21.3 格式失败

主要 logits 分析仍可进行时保留样本。行为准确率中：

- 无法解析的自由文本记为行为错误；
- 若直接从候选 logits 评分，则另报格式失败，不删除该 trial。

### 21.4 重跑

技术失败可以使用完全相同的：

```text
sample_id
checkpoint
configuration
generator seed
```

重跑。原失败记录不得删除。不能更换 seed 直到得到有利结果。

## 22. 缺失数据

- 不插补模型输出；
- 报告每个条件的计划数、成功数、失败数、排除数；
- 若某条件基础设施失败超过 1%，暂停批次并调查；
- 若排除造成 factorial group 不完整，该 group 不进入主要配对分析，但保留在失败报告；
- 缺失机制和受影响条件必须公开。

## 23. Go / Revise / Stop 决策

### 23.1 Gate 0：基础设施

**Go**

- restore manifest/checksum 正确；
- 所有正式 run 在数值容差内；
- 基础设施失败率不超过 1%。

**Revise**

- 存在可定位的 dtype、kernel 或序列化问题。

**Stop**

- 无法实现可靠 state restore；停止语义实验。

### 23.2 Gate 1：任务能力

**Go**

- Track S Prompt-visible 通过第 10 节能力门。

**Revise**

- Track S 失败但错误可归因于模板、tokenization 或答案格式；只允许使用开发集修改。

**Stop**

- 合理模板和合适 checkpoint 均无法完成最小组合任务；停止该模型路线。

### 23.3 Gate 2：单变量因果迁移

**Go**

- E1、E2 均通过统计与 SESOI 门槛；
- 相对 reset、random、matched-context 均有优势；
- \(Spec_I/Spec_G\) 通过；
- 通用能力无严重损伤。

**Revise**

- 只有一个变量通过；
- 效应存在但未达 SESOI；
- 区间过宽；
- 特异性失败；
- 只在 D0 有效。

**Stop**

- 能力门和基础设施均通过，但 E1/E2 的等效性检验表明效应均小于 SESOI。

### 23.4 Gate 3：联合绑定

**Go**

- 满足第 14 节全部条件。

**Revise**

- T1/T2 有效但 T3 不能超过单变量或 recency 策略。

**Stop**

- 在充分精度下，joint advantage 与零或单变量 ceiling 等效。

### 23.5 Gate 4：原生 state 载体资格

只有 Gate 0–3 全部 Go，才记录：

> 原生 recurrent state 在该模型、任务和时间尺度上通过首轮跨时间联合决策因果载体资格。

这仍不是 Persistent Self 结论。

## 24. 结果报告模板

每个终点至少报告：

```text
estimand
N factorial groups
mean
median
standard deviation / IQR
95% confidence interval
raw p-value
Holm-adjusted p-value
SESOI
equivalence result
baseline comparisons
exclusions / failures
sensitivity consistency
allowed conclusion
```

必须同时发布：

- 全部 group-level contrasts；
- 不含敏感内容的原始模型分数；
- 配置和 generator manifest；
- 排除与重跑日志；
- 失败模板和负面结果；
- 绘图和统计脚本；
- 软件、模型、tokenizer 和硬件版本。

## 25. 防止研究者自由度

正式测试前冻结：

- 主要终点和方向；
- SESOI；
- N；
- 标准 delay；
- 模板和标签池；
- 主要 bootstrap/permutation seeds；
- 排除规则；
- 多重比较 family；
- Go / Revise / Stop；
- 探索分析清单。

正式测试后新增分析必须标记：

```text
Post-hoc / Exploratory
```

不得用 post-hoc 结果改写预注册主要主张。

## 26. 实验批次顺序

```text
Batch 0  工程 roundtrip，仅校准恢复容差
Batch 1  开发集 Prompt-visible 能力门
Batch 2  冻结 generator、模板、标签、标准 delay
Batch 3  预注册并生成/解封 Core Set
Batch 4  Track S 主要确认实验
Batch 5  Track N 迁移实验（仅在能力门通过时）
Batch 6  Delay Set
Batch 7  Probe 与 layer/channel 探索
```

Batch 4 完成前不查看 Batch 6/7 结果来修改主要假设。

## 27. 预注册前仍需填写

- [ ] 远程模型 checkpoint 和 revision；
- [ ] tokenizer 与答案代码；
- [ ] Track S 标签池；
- [ ] Track N 首组语义；
- [ ] 标准 delay 的确切 token 数；
- [ ] restore 工程绝对下限；
- [ ] 100 次开发 roundtrip 得到的最终数值容差；
- [ ] 通用能力控制任务；
- [ ] generator 版本和正式 seeds；
- [ ] 320 groups 的模拟功效复核；
- [ ] 统计实现库与版本；
- [ ] 共同审阅 SESOI；
- [ ] 将本文状态改为 Preregistered；
- [ ] 为预注册版本计算不可变 checksum。

在这些项目完成前，不运行确认性 Batch 4。

