# EXP-001 Batch 2 参数冻结审阅单

> 版本：v0.1
> 状态：Review Candidate；不是预注册文件，不得据此运行确认集
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

## 5. 仍缺证据，当前不能冻结

### B1 历史写入协议

尚未在 G1h 2.9B 上比较：

- 单次绑定声明；
- 声明后立即验证一次；
- 多次一致绑定事件。

这是当前最重要的未决项。它直接决定 recurrent state 是如何形成的，不能由 Codex 靠偏好选择，也不能等正式结果出来后再改。

### B2 正式历史模板、测试模板与 filler 清单

开发能力模板已经固定，但正式 state-only 历史和测试模板仍需：

- 与开发模板措辞隔离；
- I/G、顺序、答案代码和选项位置完全平衡；
- 通过泄漏检查；
- 固定后生成 digest。

### B3 正式 generator seeds

当前 `20260729`、`20260730` 等是开发 seed。正式 Core Set seed 必须在代码和模板冻结后另行固定，并与开发 seed 隔离。

### B4 通用能力控制任务

需要冻结一个不依赖 I/G 的小型控制集，用来判断 state 干预是否只是整体损伤模型。

### B5 统计实现闭环

还需要：

- 320-group 模拟功效复核；
- 确认主要 SESOI；
- 固定 bootstrap / permutation seeds；
- 固定统计库及版本；
- 验证完整 primary report 生成。

### B6 不可变预注册包

当前还没有正式的：

- frozen task config；
- Core Set manifest；
- source/config/schema digest；
- preregistration checksum；
- 只读确认性运行入口。

在这些文件生成并共同确认前，不得运行 Batch 4。

## 6. Batch 2 的完成顺序

```text
记录 Impl-3o 通过
  → 共同确认 D1–D3
  → 只在开发集比较历史写入协议 B1
  → 冻结历史/测试模板和 filler
  → 固定正式 seeds 与通用能力控制
  → 运行 320-group 模拟功效复核
  → 生成 frozen config + digest
  → 共同审阅预注册文本
  → 生成/解封 Core Set
```

## 7. 当前 Go / Hold 判断

- **Go：进入 Batch 2 冻结审阅。**
- **Hold：不得进入确认性 Batch 4。**
- **原因：状态工程门已经闭合，但历史写入协议、正式模板、控制任务、统计模拟和不可变预注册包仍未冻结。**

这一区分保证我们不会把“工程工具可用”误写成“研究假设已经成立”。
