# EXP-001 checkpoint 迁移方案

> 状态：接口门已通过；三层能力门进入输出与错误审计
>
> 日期：2026-07-30
>
> 原则：更换模型不等于更换研究问题；旧模型的失败记录永久保留。

## 1. 为什么现在需要换模型

RWKV-7 World 0.4B 已通过 state 保存、恢复、交换和随机对照等工程门，但在
Impl-3b 中表现为：

| 能力层级 | 结果 | 通俗解释 |
|---|---:|---|
| 直接抄写 A/B/C/D | 100% | 模型看得见答案代码，评分接口也没有坏 |
| 单字段查表 | 25% | 只要需要根据一个符号寻找答案，模型就固定选 A |
| 双字段组合 | 25% | 身份与目标组合任务也无法完成 |

因此，现在进入 state-only 实验会把“模型不会做题”误写成“state 没有保存
信息”。正确路线是先找到能通过显式能力门的 RWKV-7 checkpoint，再继续比较
state 条件。

## 2. 候选选择

首选候选固定为：

```text
BlinkDL/rwkv7-g1
rwkv7-g1h-1.5b-20260710-ctx10240.pth
revision: bc3b5c8dae5b09db2445bf4f7589fe800d88688e
SHA-256: 737079d81865801fd85e5459488d89a36d5304a524e890244eb83d44f531c89c
size: 3,055,444,605 bytes
```

选择理由：

1. 它仍是 RWKV-7，保留本项目需要的 recurrent state 接口；
2. 官方说明 G1 系列训练数据包含 instruction、chat 和 reasoning 数据，更接近
   当前任务需要的“读条件并做选择”能力；
3. 1.5B 是官方说明中能够完成结构化 function-call 类任务的最低规模；
4. 约 3.06 GB 的权重远小于云服务器 32 GB 显存，适合作为最低成本升级；
5. 先测 1.5B，可以避免一开始就把模型规模扩大到 2.9B 或 7.2B。

官方来源：

- [RWKV-7 G1 官方模型页](https://huggingface.co/BlinkDL/rwkv7-g1)
- [固定版本的 1.5B 权重页](https://huggingface.co/BlinkDL/rwkv7-g1/blob/bc3b5c8dae5b09db2445bf4f7589fe800d88688e/rwkv7-g1h-1.5b-20260710-ctx10240.pth)
- [RWKV-7 G1 官方提示模板](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/RWKV7-G1x-templates.txt)

## 3. 为什么不先选别的模型

| 候选 | 判断 | 原因 |
|---|---|---|
| G1 0.4B | 暂不选 | 与失败模型处于同一参数规模，节省的成本不足以抵消再次失败风险 |
| G1h 1.5B | 首选 | 能力、显存和迁移成本之间最平衡 |
| G1h 2.9B | 后备 | 如果 1.5B 接口通过但能力门失败，再升级到这一档 |
| G1h 7.2B | 第二后备 | 32 GB 显存可能容纳，但没有必要在小模型尚未验证前先承担更大成本 |
| 13.3B | 暂不考虑 | 权重本身约 26.5 GB，给运行时和后续实验留下的显存余量太小 |

## 4. 迁移门顺序

新 checkpoint 不继承旧 checkpoint 的通过记录。复验顺序是：

```text
固定下载与哈希验证
  → G1h 接口门（加载、tokenizer、state inventory、同进程恢复）
  → 使用官方 G1 提示格式重新跑 copy / single-field / two-field
  → 通过后重跑磁盘恢复、reset/diff/swap、matched random
  → 冻结新模型和任务参数
  → 才能进入正式 state 因果实验
```

每一层都有明确停止条件：

- 接口门失败：先处理运行库或 checkpoint 兼容性，不运行能力门；
- copy 失败：检查官方提示格式和答案接口；
- single-field 失败：1.5B 不适合 EXP-001，评估 2.9B；
- two-field 失败：记录组合能力限制，评估 2.9B；
- 三层通过：才重跑 state 工程门。

## 5. 提示格式约束

G1 系列不应直接沿用 World 0.4B 的提示格式。能力门必须使用官方结构：

```text
User: <任务内容>

Assistant:
```

并遵守两点：

1. 输入末尾不能带空格；
2. 用户内容中的连续空行要清理，避免与对话轮次分隔符冲突。

旧 World 0.4B 的 v0.1/v0.2/Impl-3b 结果不会被重写；G1h 使用新的配置、
输出目录和报告版本。

## 6. Impl-3c 接口门结果

2026-07-30 云端结果：

```text
固定权重大小：3,055,444,605 bytes
权重 SHA-256：有效
tokenizer SHA-256：有效
模型加载：成功
state inventory：已生成
同进程 state roundtrip：valid=true
```

这只回答“新模型能否被当前框架可靠操作”，不回答任务能力或 Self 假设。

## 7. 云端执行：Impl-3d 三层能力门

拉取新提交后运行：

```bash
source .venv/bin/activate
bash scripts/run_impl3d_g1h_capability_ladder_gate.sh
cat results/development/impl3d_g1h_1.5b_capability_ladder/summary.json
```

本门现场运行 96 条平衡诊断，不复用 World 0.4B 的 two-field 结果：

```text
copy_code:    32
single_field: 32
two_field:    32
```

`valid=true` 只代表诊断完整。只有 `capability_gate_passed=true` 才允许进入
G1h 的 state 工程门复验。

## 8. Impl-3d 结果与暂缓决定

云端诊断完整运行，得到：

| 层级 | 候选评分准确率 | 自由生成格式率 | 位置问题 |
|---|---:|---:|---|
| copy | 1.0 | 0.0 | A–D 均为 1.0 |
| single-field | 1.0 | 1.0 | A–D 均为 1.0 |
| two-field | 0.875 | 0.0 | A/B/C 为 1.0，D 为 0.5 |

这说明 G1h 1.5B 已经显示出 0.4B 不具备的查表和组合能力，但当前评价接口仍
混合了两个问题：

1. 候选 continuation scoring 是否选择了正确代码；
2. 模型自由生成的前四个 token 是否恰好只包含 A/B/C/D。

自动路线首先被 copy 的格式项触发，因此
`revise_checkpoint_or_answer_interface` 不是“模型不会照抄”的充分证据。
另一方面，two-field 仍有 4 个真实评分错误和 D 位置偏差，也不能直接宣布
能力门通过。

下一步只审计已有原始报告，不重新采样：

- 查看 copy/two-field 的生成文本；
- 检查 4 个 two-field 错误的标签、映射和分数间隔；
- 冻结候选评分与自由生成各自的研究角色；
- 再决定修订评价接口还是升级 2.9B。

审计已经完成。copy 的 32 条都生成 ` <think>\n`，two-field 的 32 条都生成
` <think>We`；4 个评分错误全部是 D→B，正确答案相对 B 的分差约为
`-0.95` 至 `-1.33`。

## 9. Impl-3e-b：官方 fake-think 单变量复验

官方 G1 模型卡推荐 hard prompt 使用 fake-think 前缀：

```text
User: USER_PROMPT

Assistant: <think></think
```

本项目精确保留这个未闭合前缀，固定补入所有候选共同的 `>`，然后才比较
` A`、` B`、` C`、` D`。同时记录模型在被强制前是否本来就会贪心生成
`>`；若 96 条中有任何一条不一致，`forced_prefix_greedy_exact_rate` 就会
低于 1.0，门不会通过。

与 Impl-3d 相比，以下内容保持完全相同：

- checkpoint 与 tokenizer；
- 96 条样本及其正确答案；
- label pools、answer codes、base seed；
- bootstrap 参数与全部阈值；
- 最大生成 token 数。

因此本轮只检验一个问题：跳过 G1 默认的自由思考开头后，格式失败和 D→B
偏差是否消失。

Impl-3e-b 结果显示，共同的 `>` 在 96/96 条上与模型贪心下一 token 一致，
two-field 自由生成格式率从 0 提升到 1.0；但评分准确率为 0.84375，区间
下界为 0.75，D 位置仍为 0.5。因此格式问题已经排除，而组合准确率和位置
平衡问题没有消失。

## 10. Impl-3f：G1h 2.9B 接口候选

停止继续修改 1.5B Prompt，按原迁移计划升级一档：

```text
rwkv7-g1h-2.9b-20260710-ctx10240.pth
revision: ceb1830a7df8c9a7d9438ec56f308af41f4e3d62
SHA-256: 295595b3b8dbff3f8c2a0585975622ddaba4feea7a377022f0bd75347c90c9b3
size: 5,896,273,469 bytes
```

2.9B 为 32 层、宽度 2560、head size 64。Impl-3f 云端接口门已经通过：

- 固定权重与 tokenizer 校验有效；
- 实际观察到 32 层、96 个 state 组件；
- state 总量为 21,299,200 字节，全部数值有限；
- 峰值显存为 6,232,199,168 字节，加载约 5.93 秒；
- tokenizer roundtrip 与同进程 state 恢复均有效。

这说明 2.9B 与现有实验框架兼容，但不代表它已经通过 EXP-001 的能力门。

## 11. Impl-3g：G1h 2.9B 受控能力复验

Impl-3g 复制已经审计过的 1.5B fake-think 评价接口，只允许改变两项：

1. 模型配置改为 `rwkv7_g1h_2.9b.candidate.json`；
2. 接口证据改为 Impl-3f 的 2.9B 结果。

以下内容保持完全相同：96 条题、label pairs、答案位置、随机种子、
`<think></think` 与强制补入的 `>`、生成长度、bootstrap 参数和全部门槛。
因此 1.5B 与 2.9B 的结果可以直接比较，不能把 Prompt 修改混入模型规模
比较。

```bash
bash scripts/run_impl3g_g1h_2.9b_fake_think_gate.sh
cat results/development/impl3g_g1h_2.9b_fake_think/summary.json
```

实际结果为 `capability_gate_passed=false`：

| 层级 | 评分准确率 | 区间 | 格式有效率 | D 位置准确率 |
|---|---:|---|---:|---:|
| copy | 1.0 | `[1.0, 1.0]` | 0.75 | 1.0 |
| single-field | 0.90625 | `[0.78125, 0.96875]` | 0.875 | 0.625 |
| two-field | 0.875 | `[0.78125, 0.9375]` | 0.90625 | 0.5 |

因此“只把 1.5B 换成 2.9B”没有通过能力门。two-field 比 1.5B 的
0.84375 略高，但区间仍未过线，D 偏差没有消失；同时 single-field 和
输出格式反而出现新的失败，不能把这种波动写成规模提升成功。

## 12. Impl-3h：原始结果只读审计

下一步不重跑模型，也不立即下载更大的 checkpoint。审计命令读取 Impl-3g
已有的 manifest 和 JSONL，生成：

- 每个层级的输出文本与 token 变体计数；
- target→predicted 混淆矩阵；
- 全部评分错误的题目字段、选项映射和分差；
- 格式异常数以及它与评分错误的重叠线索。

```bash
bash scripts/audit_impl3g_g1h_2.9b_results.sh
cat results/development/impl3g_g1h_2.9b_fake_think/audit_report.json
```

若评分正确而格式异常集中在一种稳定续写，才允许设计一次预先声明的答案
接口修订。若评分错误仍集中于 D 或双字段组合，则 checkpoint 能力不足的
解释更强，不再用 Prompt 格式掩盖该结果。

审计结果确认：

- 7 个评分错误的正确答案全部是 D，其中 6 个 D→B、1 个 D→C；
- 4 个错误同时格式失败，另 3 个在正常生成 B/C 时也真实答错；
- copy 的目标 C 会固定进入 Markdown 代码块；
- single/two-field 的部分样本会转而解释 `The current symbol/domain`；
- 最关键的是，模型 96/96 次在 `>` 后自然生成换行，而 Impl-3g 候选评分
  比较的是空格+A–D。

## 13. Impl-3i：自然换行边界对齐

Impl-3i 只改变一个完整回答边界：

```text
Impl-3g: <think></think> A
Impl-3i: <think></think>
A
```

配置层面表现为把共同强制前缀从 `>` 改成 `>\n`，同时把每个候选自身的
前导空格移除。模型、Prompt 正文、96 条题、样本 ID、seed、bootstrap
参数和阈值全部保持不变。

```bash
bash scripts/run_impl3i_g1h_2.9b_newline_aligned_gate.sh
cat results/development/impl3i_g1h_2.9b_newline_aligned/summary.json
```

自由生成的代码块/解释文本不会因为这次评分边界对齐而自动消失，因此本轮
首先检验评分准确率与 D 位置。如果 D 偏差仍存在，就不再继续调整该边界。

实际结果：

| 层级 | Impl-3g 评分 | Impl-3i 评分 | Impl-3g D | Impl-3i D |
|---|---:|---:|---:|---:|
| copy | 1.0 | 1.0 | 1.0 | 1.0 |
| single-field | 0.90625 | 1.0 | 0.625 | 1.0 |
| two-field | 0.875 | 0.875 | 0.5 | 0.5 |

`forced_prefix_greedy_exact_rate=1.0`，说明自然换行对齐成立。它完整修复了
single-field，却没有改变任何 two-field 指标。因此不能再把剩余 4 个
双字段错误归因于空格/换行评分边界。

## 14. Impl-3j：对齐后双字段错误审计

不重新运行模型，直接审计 Impl-3i 的已有记录：

```bash
python -m psa g1-capability-audit \
  --output-dir results/development/impl3i_g1h_2.9b_newline_aligned
cat results/development/impl3i_g1h_2.9b_newline_aligned/audit_report.json
```

重点比较 Impl-3g 与 Impl-3i 的 two-field 错误样本 ID、D→B/C 混淆方向
和分差。若错误集合稳定，停止修改评分边界。

审计确认错误集合完全稳定：

- 同样 4 个 sample ID；
- 同样 3 个 D→B、1 个 D→C；
- 四个 `target_minus_predicted` 在新旧边界下均保持为负；
- 两题的错误选项只匹配 domain，另两题只匹配 operation。

这排除了随机数值翻转，也没有证据表明模型固定忽略某一个字段。剩余问题
可能是答案代码 D 的系统偏差，也可能是字母偏差与组合语义的交互。

## 15. Impl-3k：答案代码轮换诊断

对每个相同的双字段语义案例，保持 domain、operation、选项语义和模型
不变，只循环更换 A/B/C/D 的映射。32 个案例各 4 轮，共 128 条：

```bash
bash scripts/run_impl3k_g1h_2.9b_code_rotation_gate.sh
cat results/development/impl3k_g1h_2.9b_code_rotation/summary.json
```

每个语义案例的正确答案在四轮中恰好分别为 A、B、C、D。报告提供：

- `per_code`：每个答案代码的准确率；
- `all_rotations_correct_case_count`：四轮全对的语义案例数；
- `multi_code_error_case_count`：同一语义案例在多个字母下出错的数量；
- `route_decision`：答案代码偏差、语义组合失败或混合效应。

Impl-3k 是开发诊断，不会把原能力门的失败追溯改写为通过。

实际结果：

| 代码 | 正确数 | 准确率 |
|---|---:|---:|
| A | 30/32 | 0.9375 |
| B | 32/32 | 1.0 |
| C | 32/32 | 1.0 |
| D | 22/32 | 0.6875 |

共 12 个错误，其中 10 个发生在 D、2 个发生在 A。22/32 个语义案例四轮
全对；另外 10 个案例都在 D 下出错，其中 2 个还在 A 下出错。因此原始
`route_decision=semantic_composition_failure` 只捕捉到“存在跨代码错误
案例”，却没有表达 D 的强主效应。研究解释应为强答案代码偏差加少量
语义×代码交互。

## 16. Impl-3l：跨代码标签边际化复核

同一语义选项在四轮中分别映射到 A、B、C、D。将它的四个 log-score
取平均后，每个字母的整体先验对所有语义选项贡献相同，因此能够在配对
设计内抵消，而不需要事后拟合校准参数。

```bash
bash scripts/review_impl3k_g1h_2.9b_code_rotation.sh
cat results/development/impl3k_g1h_2.9b_code_rotation/code_rotation_review.json
```

该命令只读取已有 manifest 与 JSONL，不加载模型。它保留原 summary 的
路线字段，同时用 `route_logic_version=0.2` 输出：

- `label_marginalized_accuracy`
- `label_marginalized_correct_case_count`
- `label_marginalized_error_count`
- 未被代码轮换消除的具体语义案例

只有标签边际化后仍错的案例，才能作为代码偏差之外的语义失败证据。
