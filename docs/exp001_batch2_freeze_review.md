# EXP-001 Batch 2 参数冻结审阅单

> 版本：v0.3
> 状态：Impl-3q 与 Impl-3r 均为有效 Hold；Impl-3r-a 已将问题定位到正式历史模板稳定性；不是最终预注册文件
> 日期：2026-07-31
> 进入条件：Impl-3m、Impl-3n-b、Impl-3o 均已通过

## 1. 这一步在做什么

开发阶段已经回答了两个工程问题：

1. 选定模型能否完成 EXP-001 所需的显式二字段匹配；
2. 模型的 recurrent state 能否可靠保存、恢复、重置、交换和构造公平的随机对照。

Batch 2 不再调模型，也不运行正式实验。它把开发阶段得到的工程事实写成一份不可随正式结果改变的实验合同。

本文件把参数分成三类：

- **已有证据，可以冻结**；
- **建议默认值，但需要共同确认**；
- **仍缺证据，不能冻结**。

## 2. 开发门闭合情况

| 开发门 | 作用 | 结果 | Batch 2 含义 |
|---|---|---|---|
| Impl-3l | 四代码轮换后检查模型是否真正理解二字段语义 | 32/32 语义案例正确 | 可以固定代码轮换平均读出 |
| Impl-3m | 独立进程恢复 2.9B recurrent state | L3；100/100 容差与 top-1 通过 | checkpoint 能跨进程可靠恢复 |
| Impl-3n | 首次直接运行状态操作门 | reset state 误差超限，保留为失败 | 不能删除或改写这份失败 |
| Impl-3n-a | 定位 reset 超限是否持续发生 | 确认为首次形状调用异常 | 允许预先声明一次同形状预热 |
| Impl-3n-b | 单次预热后完整复验 | diff/reset/swap 全部通过 | 状态操作工程门闭合 |
| Impl-3o | 尺度匹配随机状态复验 | 所有条件通过 | random_matched 对照工程门闭合 |

Impl-3o 的关键证据：

- `component_count=96`；
- `continuation_shape_warmup_count=1`；
- 同 seed 逐位复现；
- 不同 seed 可区分；
- 逐组件尺度匹配有效；
- `max_relative_l2_error=2.8687819151988067e-05`，低于 `0.01` 上限；
- 随机状态续算有效；
- tokenizer roundtrip 有效；
- 构造和续算没有修改来源状态；
- 总门 `valid=true`。

## 3. 已有证据、建议直接冻结的参数

### 3.1 模型与运行时

| 参数 | 冻结候选 |
|---|---|
| model ID | `rwkv7-g1h-2.9b-20260710` |
| weights revision | `ceb1830a7df8c9a7d9438ec56f308af41f4e3d62` |
| weights SHA-256 | `295595b3b8dbff3f8c2a0585975622ddaba4feea7a377022f0bd75347c90c9b3` |
| weights size | `5,896,273,469` bytes |
| tokenizer revision | `8cc2ab5d72b0a75713a62be5192a6e39b28df0ed` |
| tokenizer SHA-256 | `e6dee3d4e31b4d5c40ac99508ac6c701ceef4bed681bf2167ce9a908552bca89` |
| strategy | `cuda fp16` |
| Python | `>=3.11,<3.13`；已验证 `3.12.3` |
| PyTorch / CUDA | `2.12.0 / 13.2` |
| RWKV | `0.8.32` |
| RWKV 环境变量 | `RWKV_V7_ON=1`、`RWKV_JIT_ON=0`、`RWKV_CUDA_ON=0` |

### 3.2 答案读出

| 参数 | 冻结候选 |
|---|---|
| 答案代码 | `A / B / C / D` |
| 轮换次数 | 每个语义案例完整轮换 4 次 |
| 聚合量 | `candidate_log_score` |
| 聚合方式 | 同一语义选项跨四代码映射取算术平均 |
| 选择方式 | 选择平均分最高的语义选项 |
| assistant prefix | `<think></think` |
| forced answer prefix | `>\n` |
| candidate continuation prefix | 空字符串 |
| 事后校准 | 禁止拟合 A–D 偏移；禁止用真值参与评分 |

该方案不是把原始错误抹掉。原始代码级准确率仍为 `0.90625`；四轮平均是预先对称化答案代码先验的测量设计。

### 3.3 合成标签池

当前 tokenizer、roundtrip 和能力门共同验证过：

- identity pairs：`[baf, zom]`、`[niv, teg]`；
- goal pairs：`[vam, zep]`、`[qir, bok]`；
- answer codes：`A/B/C/D` 均为等长单 token。

建议把这四组 pair 作为 Track S 的冻结标签池，而不是继续搜索更有利的新词。

### 3.4 标准 delay

开发阶段按 tokenizer-only 规则，从 1–32 个 filler units 中选择最接近 128 tokens 的候选：

- `standard_delay_units=11`；
- `standard_delay_token_count=131`；
- 与目标 128 tokens 相差 3，满足不超过 16 tokens 的预设规则。

模型迁移没有更换 tokenizer，因此建议冻结 11 units / 131 tokens，不再按任务表现重新选择。

### 3.5 状态工程协议

| 项目 | 冻结候选 |
|---|---|
| state 组件数 | 96 |
| 跨进程恢复次数 | 100 |
| reset / random 续算正式重复 | 10 |
| 同形状预热 | 每种首次 suffix 形状 1 次，明确排除计分 |
| logits 最大绝对误差 | `0.0625` |
| state 最大绝对误差 | `0.125` |
| random scale 最大相对 L2 误差 | `0.01` |
| random base seed | `314159265` |
| random alternate seed | `271828182` |
| 来源状态 | 所有操作必须保持 immutable |

首次预热是预先由 Impl-3n-a 证据决定的工程规则，不得根据某个正式样本是否通过来选择性执行。

### 3.6 样本量下限

评价草案已规定：

- Core Set 至少 `320` 个独立 factorial groups；
- 不能因为开发结果看起来稳定而下调；
- 正式冻结前仍需运行一次 320-group 模拟功效复核。

## 4. 建议默认值，但需要共同确认

项目负责人已于 2026-07-31 确认按以下建议执行。这些决定进入冻结候选，
但仍需与 Impl-3p 结果和最终预注册包一起计算 digest。

### D1 state-only 测试 Prompt 是否保留通用组合规则

建议：**保留规则，只隐藏当前 identity/goal 的具体值。**

理由：核心问题是 recurrent state 是否保留变量，不是模型能否同时记住题目规则。规则在所有条件中保持相同，不会泄漏当前状态答案。

### D2 Track S 是否作为首个确认实验

建议：**先运行 Track S，Track N 延后。**

理由：合成标签已经通过 tokenizer、能力和答案偏差控制；自然语义会额外引入词频、常识和价值先验。

### D3 确认性条件范围

建议首轮主要条件只包含：

- `continuous`；
- `restored`；
- `reset`；
- `random_matched`；
- `swapped_I`；
- `swapped_G`；
- `swapped_both`；
- `prompt_visible` 作为能力上限。

`interpolated`、layer/channel `ablated`、Probe 和 Track N 放入后续批次，不参与首轮 Go/Revise/Stop。

确认结果：

- D1：接受；
- D2：接受；
- D3：接受。

## 5. B1–B5 已形成冻结候选；B6 等待云端证据与人工 checksum

### B1 历史写入协议

Impl-3p 已在 G1h 2.9B 上完成比较：

- 单次绑定声明；
- 声明后立即验证一次；
- 多次一致绑定事件。

三种模式使用相同语义案例、标签、131-token delay、state-only 查询和四代码
轮换，只改变历史写入方式。选择规则已在运行前固定：

1. 按 `single_statement → statement_plus_verification → repeated_consistent`
   的复杂度顺序检查；
2. 每个模式使用 32 个语义案例，每个案例完整轮换 A–D，共 128 条读出；
3. 标签边际化准确率至少为 `0.80`；
4. 四轮必须完整，来源 recurrent state 必须保持不变；
5. 选择第一个通过的模式，不选择分数最高的模式；
6. 三种都失败则 Revise，不降低阈值。

配置：
[`impl3p_g1h_2.9b_history_binding.dev.json`](../configs/gates/impl3p_g1h_2.9b_history_binding.dev.json)。

实际结果：

| 模式 | 代码级准确率 | 标签边际化准确率 | 完整四轮案例 | 来源 state 不变 | 通过 |
|---|---:|---:|---:|---|---|
| `single_statement` | 0.9296875 | 0.96875（31/32） | 32/32 | 是 | 是 |
| `statement_plus_verification` | 0.9765625 | 1.0（32/32） | 32/32 | 是 | 是 |
| `repeated_consistent` | 0.9453125 | 1.0（32/32） | 32/32 | 是 | 是 |

按照运行前固定的复杂度顺序，冻结候选为 `single_statement`。后两个模式
虽然分数更高，但不能取代已经达到门槛的更简单模式。运行还确认：

- 384/384 条受控读出完成；
- 三种模式各 32 个语义案例；
- 三种 token 形状各执行一次不计分预热；
- forced prefix 贪心一致率为 1.0；
- 峰值显存约 6.23 GB；
- 总 `valid=true`，路线为 `freeze_single_statement`。

### B2 正式历史模板、测试模板与 filler 清单

正式候选已经写入
[`exp001_track_s.formal_v1.json`](../configs/preregistration/exp001_track_s.formal_v1.json)：

- 4 个单次声明历史模板；
- 4 个 state-only 查询模板；
- 16 个模板组合在 320 groups 中各出现 20 次；
- 4 个由 tokenizer 确认恰好 131 tokens 的中性 filler；
- I/G、答案代码和模板组合成组平衡；
- 只允许用 prompt-visible 结果做资格审查。

### B3 正式 generator seeds

正式 seed 已用命名空间
`PSA|EXP-001|formal-v1|<purpose>` 的 SHA-256 前 32 bits 固定，并与
`20260729`、`20260730`、`20260731` 等开发 seed 隔离。具体数值见 D7；
运行时会重新推导并逐项核对，任何手工改值都会使 Impl-3q 失败。

### B4 通用能力控制任务

已固定 96 条控制：32 条答案代码复制、32 条无关单字段词法匹配、32 条
无关双字段符号匹配。每类为 8 个语义案例×4代码轮换。Impl-3q 先验证
prompt-visible 基线；正式实验以后必须在每个 state 条件同步运行同一控制集。

### B5 统计实现闭环

SESOI、10,000次cluster bootstrap、至少100,000次置换、Holm校正和五个
正式seed均已写入配置。Impl-3q 会先从prompt-visible正式模板资格记录的
32个factorial groups估计E1–E3开发期nuisance SD，再同时运行经验代理和
\(d_z=0.20\) 保守标准化两套10,000次功效模拟；任一套、任一主要终点低于
90%，路线都只能提高 N，不能降低门槛或生成 Core Set。

### B6 不可变预注册包

Impl-3q 会锁定：

- frozen task config；
- 最终 tokenizer 后的模板/filler manifest；
- 资格审查与控制的原始记录；
- 功效报告；
- 所有影响结果的源码、配置和 schema digest；
- payload root 与 candidate checksum。

它不会生成 Core Set，也不会启用确认性运行入口。只有候选包
`eligible_for_human_freeze=true`，且项目负责人再次人工确认
`candidate_digest_sha256` 后，才可把候选升级为最终预注册包。

## 6. Batch 2 的完成顺序

```text
记录 Impl-3o 通过
  → D1–D3 已确认
  → Impl-3p 选择 single_statement
  → 冻结历史/测试模板和 filler
  → 固定正式 seeds 与通用能力控制
  → 运行 320-group 模拟功效复核
  → 生成 frozen config + digest
  → 共同审阅预注册文本
  → 生成/解封 Core Set
```

## 7. 当前 Go / Hold 判断

- **Go：运行独立 Impl-3r v2 prompt-visible 冻结候选门。**
- **Hold：不得进入确认性 Batch 4。**
- **原因：Impl-3q-a 已确认首版模板与双字段控制都有真实语义失败；第二版仍须先取得资格。安全边界正常，未读取确认结果、未生成 Core Set。**

已收到的细分证据：

- 格式有效率为100%，排除回答边界故障；
- 模板标签边际化joint accuracy为83.59%，95% BCa区间为75%–88.28%；
- identity与goal点估计分别为88.28%和92.97%，其区间下界均未过门；
- `formal-history-01` 为78.125%，`formal-query-03` 为75%；
- 原始A–D中D仍最弱，但四代码平均后仍有21/128语义错误，不能只归因于字母；
- 复制与单字段控制均100%，无关双字段控制代码级50%。

Impl-3q-a 的只读审计进一步确认：

- 双字段控制四轮平均后仅2/8正确（25%）；
- 错误不是跟随A–D字母，而是几乎固定猜测 `cinder + trace`；
- 总路线为 `revise_formal_and_control_two_field_prompt_families`。

因此修订范围只包括证据指向的正式措辞、控制字段和语义读出；不更换模型、
不改变历史模式、delay、标签池、seed、样本量、SESOI、功效目标或安全边界。

这一区分保证我们不会把“工程工具可用”误写成“研究假设已经成立”。

## 8. 下一轮共同确认的推荐方案

项目负责人已于 2026-07-31 确认 D4–D8，以下方案已经写入 Impl-3q
配置与代码。它们在云端资格门通过且 checksum 再次人工确认前仍只是冻结候选。

### D4 正式模板族

冻结候选：

- 采用 4 个语义等价的 `single_statement` 历史模板；
- 采用 4 个 state-only 查询模板；
- 16 种历史×查询组合在 320 个 factorial groups 中各出现 20 次；
- 每个 group 的四种 I×G 状态、I/G 出现顺序和 A–D 代码轮换必须成组平衡；
- 准备 4 个与任务标签无关的 filler 版本，均固定为 11 units / 131 tokens；
- 正式措辞、filler 和 seed 与开发集隔离；
- 模板只能通过 prompt-visible 能力门和 tokenizer/leakage 检查取得资格，
  不能查看 state-only 结果后淘汰“表现不佳”的模板。

通俗地说：正式考试用四套等价卷面，避免结论只对某一句话有效；但只能在
“答案写在题面里”的练习模式下检查卷面是否清楚，不能先偷看正式答案再换卷。

### D5 通用能力控制集

冻结候选为每个 state 条件同时运行 96 条短控制读出：

- 32 条固定答案代码复制；
- 32 条与 I/G 标签无关的单字段词法匹配；
- 32 条与 I/G 标签无关的四选一符号映射。

每类由 8 个语义案例×4 个 A–D 轮换组成。控制集使用独立 seed，并与主任务
共享答案接口和同形状预热规则。以下任一项触发“可能是整体损伤”警报：

- 相对 `continuous`，准确率下降超过 5 个百分点；
- 格式有效率下降超过 2 个百分点；
- 正确答案平均 log-prob 下降超过 0.25 natural-log units；
- state norm/RMS 超出开发分布的 99.9% 范围。

控制集不会证明“Self”存在；它负责排除一种更简单的解释：干预只是把模型
整体弄坏了。

### D6 SESOI 与主要统计规则

冻结评价协议 v0.1 中在正式结果之前写下的默认值：

- \(D_I,D_G\)：0.50 log-odds；
- 联合 margin：0.50；
- joint accuracy：至少 0.60；
- \(Spec_I,Spec_G\)：0.25；
- prompt-normalized retention：0.20；
- 10,000 次 group-level cluster bootstrap；
- E1–E3 至少 100,000 次配对置换或符号翻转；
- 三个主要终点使用 Holm 校正，family-wise \(\alpha=0.05\)。

这些阈值不是从 Impl-3p 的分数倒推出来的。Impl-3p 只选择写入协议，不为
正式机制效应设门槛。

### D7 正式随机种子

正式seed不手挑“看起来吉利”的数字，而用固定命名空间
`PSA|EXP-001|formal-v1|<purpose>` 的 SHA-256 前 32 bits 生成。这样任何人
都能重新算出相同 seed，也无法在结果出现后悄悄换 seed：

| 用途 | SHA-256 前 8 位 | seed |
|---|---|---:|
| Core Set generator | `0153033a` | `22217530` |
| 控制集 generator | `03b9d9b5` | `62511541` |
| cluster bootstrap | `756305a8` | `1969423784` |
| permutation | `a23c707d` | `2721869949` |
| 功效模拟 | `f1224b58` | `4045556568` |

这些 seed 与开发阶段的 `20260729`、`20260730`、`20260731` 分离。

### D8 样本量与功效复核

冻结 `N=320` 个独立 factorial groups，不下调。实现完整生成器后：

1. 使用prompt-visible正式模板资格记录的开发期 group contrast 估计
   nuisance variance，并同时保留 \(d_z=0.20\) 标准化保守模拟；
2. 分别验证 E1、E2、E3 在各自 SESOI 下的功效；
3. 两套模拟的目标功效都至少为 90%；
4. 如果不足，只允许在解封 Core Set 前提高 N；
5. 不允许因为模拟看起来“很容易”而把 N 降到 320 以下。

### D9 确认口令后的实施边界

D4–D8 确认后的 Impl-3q 只会：

1. 实现并测试正式模板、filler 和控制集生成器；
2. 运行 prompt-visible 模板资格门与 320-group 功效模拟；
3. 生成 frozen config、schema、manifest 和所有 digest；
4. 输出一份可人工审阅的预注册文本。

首版 Impl-3q 已运行并保留为失败候选。独立 Impl-3r v2 继承所有不允许改动
的冻结参数，只做三项受控修订：

1. 四个历史模板都使用稳定的 `CURRENT DOMAIN` / `CURRENT OPERATION`；
2. 查询模板明确逐字段比较同一组字段名；
3. 双字段控制使用常见 `COLOR` / `SHAPE`，语义控制按预声明的四轮平均分数
   判定，同时保留原始代码级准确率供诊断。

Impl-3r 已在云端有效运行：控制基线与功效通过，正式模板资格失败，因此
继续 Hold。Impl-3r-a 显示总体四轮平均为119/128，9个错误中6个集中在
history-v2-03，history-v2-02满分；四个query均超过90%。这支持只修正式
历史模板族，但不允许事后只保留满分模板。下一步先核对冻结BCa区间的精确
失败项；只有后续独立候选的所有资格门通过，且不可变候选checksum再次得到
人工确认，Core Set才可生成或解封。

精确门槛复核显示唯一失败项是goal/OPERATION的BCa下界0.890625，距离冻结
要求0.90差0.009375；identity下界0.91127、joint下界0.859375、格式和所有
单模板点估计均通过。Impl-3s因此不选择观测满分的history-v2-02，而将四个
history作为整体改为平行的FIELD 1 DOMAIN/FIELD 2 OPERATION结构。v2查询、
控制、模型、delay、seed、N和阈值不变。

Impl-3s 已有效运行，但正式模板资格再次失败；控制与功效仍通过，未读取
确认结果或生成Core Set。为限制开发集过拟合，现在暂停v4设计，先只读比较
v2/v3的BCa区间、模板交互与错误集中度。没有新的终止判定前，不再修改模板。

v3精确指标显示goal点估计96.09%、BCa下界0.8984375，较v2改善但仍比0.90
低0.0015625；identity下界0.90888、joint下界0.875、格式和所有单模板点
估计均通过。该结果仍按失败记录，不四舍五入。下一判断只依据已有错误分布，
不再修改措辞追逐这0.0015625。

完整错误审计显示8个错误跨query-02/03/04、filler-01/02/04、两组标签和
全部目标组合。history-03占5个，但其索引在v2/v3沿用同一案例流，模板措辞
与案例难度混杂。因此终止prompt调优。Impl-3t把v3逐字冻结，用新命名空间
seed `3061017642`做一次未观察的同规模留出验证；通过只进入checksum审阅，
失败停止2.9B路线，且禁止再次抽样、增样或修改模板。

Impl-3t 已一次性通过：模板资格、控制基线和功效门均为true，
`freeze_candidate_ready=true`，安全边界正常。不可变候选digest为
`a354b208be0640da7ea70fe070f75bdec69186e496ba1cc14c3157dcd984e6cd`。

人工技术与内容审阅也已完成：

- `self_digest_valid=true`、`payload_root_valid=true`；
- 23项源码检查和10项证据检查全部为true；
- `safety_boundary_valid=true`，确认性结果未读取，Core Set未生成；
- 8个正式条件与D3接受清单完全一致；
- D4–D8、N=320、5个正式seed、统计方案、4×4模板和4个filler均与已接受
  冻结设计一致。

项目负责人已于2026-07-31逐字确认完整candidate checksum，并明确只授权升级
最终预注册包。最终包状态为`final_preregistration_frozen`，digest为
`0daf056dc6b38aa20fa69dd9e8df9b8065876529947cbc01353ffe604933d0c9`。
候选、验证报告和人工确认记录均已锁定，包自校验、payload root与安全边界
通过。

项目负责人随后单独授权生成并冻结Core Set，但仍未授权正式实验。该授权
绑定最终预注册digest、N=320和“只生成不运行”范围。生成工具固定320组×4
状态×4代码轮换，共1,280个语义案例和5,120条试题；真实Core Set digest必须
在云端用冻结tokenizer拟合4个131-token filler后一次性产生。正式模型推理
入口继续关闭。
