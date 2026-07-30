# Persistent Self Architecture 项目进度表

> 最后更新：2026-07-30
> 当前节点：G1h 1.5B 能力门未通过；Impl-3f 正在迁移验证 G1h 2.9B
> 研究状态：尚未进入正式确认性实验，尚未实现显式 Self Model

## 1. 这张表怎么使用

这份文件是项目的长期进度账本，用通俗语言记录：

- 每一步正在做什么；
- 为什么必须做这一步；
- 已经得到什么结果；
- 当前由谁执行；
- 下一步由什么证据决定。

状态含义：

- ✅ **完成**：代码或设计已经完成，并取得所要求的验证证据；
- ⚠️ **Revise**：运行本身有效，但结果表明方案需要修改；
- 🟡 **进行中/等待验证**：实现已完成，正在等待云端结果或共同审阅；
- ⏳ **未开始**：前置条件尚未满足；
- 🛑 **Stop**：证据表明当前路线不应继续。当前没有 Stop 项。

## 2. 总进度

| 步骤 | 状态 | 我们在做什么 | 为什么要做 | 当前结果 | 主要执行者 |
|---:|---|---|---|---|---|
| 1. 理解项目目标 | ✅ 完成 | 阅读项目简介，明确 Persistent Self 的研究问题 | 先弄清楚到底想证明什么，防止最后只做成普通记忆系统 | 核心问题确定：模型内部能否维持一个持续、可干预、能影响行为的“自我状态” | Codex |
| 2. 文献和理论调研 | ✅ 首轮完成 | 梳理记忆、World Model、Persona、Agent state、Self Model 等研究 | 看前人做到了哪里，避免重复造轮子或把旧机制换个名字 | 已形成研究地图、术语边界和首轮研究主张；后续仍会按需要补充文献 | Codex |
| 3. 划清概念边界 | ✅ 完成 | 区分 Prompt、Memory、原生 recurrent state 和显式 Self State | 模型“记住了内容”不等于模型“拥有 Self Model” | 已明确：只有状态能持续、更新、被干预并因果影响行为，才有资格继续讨论 Self | Codex |
| 4. 设计总体架构 | ✅ 初版完成 | 设计 World Model、Memory、原生 state、Self Store、Encoder、注入和更新模块 | 先画清楚未来系统由什么组成，再决定先验证哪一块 | 显式 Self Model 已完成理论设计，但尚未写入模型 | Codex |
| 5. 设计 EXP-001 | ✅ 完成 | 设计“身份约束 × 当前目标”的四状态任务 | 用一个很小、可量化的任务测试状态是否真的影响选择 | 已形成四组合任务、swap/reset/random 对照和评价指标 | Codex |
| 6. 建立代码与实验骨架 | ✅ 完成 | 实现任务生成、泄漏检查、统计方法、配置和报告格式 | 相当于先把实验室的记录表、评分器和质检流程搭好 | 本地纯逻辑测试目前达到 64 项全部通过 | Codex |
| 7. 准备模型和数据下载 | ✅ 完成 | 提供脚本下载固定版本的 RWKV 模型和 tokenizer | 云服务器只需运行脚本，不用手动寻找文件 | 模型约 861 MB、tokenizer 约 1.1 MB，哈希验证通过 | Codex 编写；项目负责人云端执行 |
| 8. 检查云端环境 | ✅ 通过 | 核对 GPU、CUDA、PyTorch、Python、RWKV 和磁盘 | 先确认实验机器不会因为版本问题产生假结果 | RTX 5090 32 GB、CUDA 13.2、PyTorch 2.12、RWKV 0.8.32，环境有效 | 项目负责人运行；Codex 分析 |
| 9. Impl-1：模型接口 | ✅ 通过 | 加载模型、测试 tokenizer、读取 recurrent state | 确认我们真的能够观察和操作模型内部状态 | RWKV-7 0.4B 加载成功；24 层、每层 3 个组件，共 72 个 state tensors | 共同完成 |
| 10. 内存状态恢复 | ✅ 通过 | 复制 state 后继续运行并比较结果 | 检查“保存状态”是否像复制文件一样可靠 | 同进程 logits 和所有 state tensors 均逐位一致 | 共同完成 |
| 11. Impl-2：磁盘 checkpoint | ✅ 通过 | 把 state 保存到磁盘，在新进程加载并继续 | 真正的持久状态必须在程序关闭后还能恢复 | 100/100 次在数值容差内，top-1 选择全部一致，达到 L3 | 共同完成 |
| 12. 处理跨进程数值差异 | ✅ 完成 | 分析为什么跨进程不是逐位完全一致 | FP16/CUDA 可能产生极小数值差，不能误判为保存失败 | 精确一致为 0/100，但误差很小且行为一致；据此冻结开发容差 | Codex 分析修正；项目负责人重跑 |
| 13. Impl-2b：reset/diff/swap | ✅ 通过 | 测试状态重置、比较和完整交换 | 后续要证明“行为跟着状态走”，必须先保证交换工具可靠 | 72/72 组件可区分；reset、swap、来源不变性全部通过 | 共同完成 |
| 14. Impl-2c：matched random | ✅ 通过 | 生成与真实 state 尺度相同的随机状态 | 排除“只要塞入任意同等大小数值就会改变输出”的解释 | 同 seed 可复现、不同 seed 可区分、续算稳定；尺度误差仅约 0.0038% | 共同完成 |
| 15. Impl-3 v0.1：任务能力门 | ⚠️ Revise | 明示身份和目标，检查模型能否完成四选一任务 | 在测试“状态记不记得”之前，先确认模型看着答案条件时会不会做题 | 基础设施全部正常，但模型 32 次都选 A；联合准确率 0.25，等于机会水平 | 项目负责人运行；Codex 诊断 |
| 16. 排查 A 偏置 | ✅ 完成 | 检查答案 token、分数和前两组正确答案映射 | 判断是答案 token 不公平，还是模型没理解题目 | A–D 都是等长单 token；非 A 分数变化也未稳定指向正确项，不采用事后分数校正 | 共同完成 |
| 17. 修复 JSONL 输出 | ✅ 完成 | 修正原始结果文件中的多余空行 | 保证以后每行都是一个可直接读取的 JSON 记录 | 已修复并添加回归测试；旧数据内容没有损坏 | Codex |
| 18. Impl-3 v0.2：新能力模板 | ⚠️ Revise | 直接显示 `CURRENT DOMAIN`、`CURRENT OPERATION` 和结构化选项 | 检查 v0.1 失败是否只是措辞或 filler 造成 | 模型仍在全部 32 条轨迹中选择 A；联合准确率 0.25，两个边际准确率 0.5，格式有效率降至 0.5 | 项目负责人运行；Codex 诊断 |
| 19. Impl-3b：分层能力诊断 | ⚠️ Revise | 分别测试“直接抄代码”和“单字段匹配”，并复用 v0.2 双字段结果 | 不再盲目换措辞，而是定位模型究竟在哪一级能力上失败 | 诊断有效；copy 32/32 正确，但 single-field 仅 8/32，仍固定选 A；路线为 `revise_single_field_matching` | 项目负责人运行；Codex 诊断 |
| 20. Impl-3c：checkpoint 接口迁移 | ✅ 通过 | 固定官方 G1h 1.5B 候选，验证加载、tokenizer 与 recurrent state 接口 | 当前 0.4B 会抄答案但不会查表，不能用来区分“任务能力不足”和“state 失效” | 3,055,444,605 字节权重及 tokenizer 哈希有效；G1h 1.5B 加载、state inventory 与同进程恢复门 `valid=true` | 项目负责人运行；Codex 诊断 |
| 21. Impl-3d：新模型分层能力复验 | ⚠️ Revise | 使用官方 G1 提示格式，现场测试 copy、single-field 和 two-field 共 96 条 | 更换模型后必须取得新的显式任务能力证据，不能复用旧模型结果 | 诊断有效；评分准确率为 copy 1.0、single 1.0、two-field 0.875；但 copy/two-field 自由生成格式率为 0，two-field 的 D 准确率仅 0.5，综合门未通过 | 项目负责人运行；Codex 诊断 |
| 22. Impl-3e-a：输出与错误审计 | ✅ 完成 | 查看自由生成文本，并逐条检查 4 个 two-field 评分错误 | 当前自动路线被 copy 的格式项优先触发，不能把“格式不合规”误写成“不会查表” | copy 全部生成 `<think>`；two-field 全部生成 `<think>We`；4 个评分错误均为正确 D 被选成 B，分差约 0.95–1.33 | 共同完成 |
| 23. Impl-3e-b：官方 fake-think 复验 | ⚠️ Revise | 只把 Assistant 前缀改为官方推荐的 `<think></think`，固定补全共同的 `>` 后重新评分 | 让 reasoning checkpoint 在受控条件下跳过自由思考，同时不事后修改样本、阈值或模型 | `>` 续写一致率 1.0；two-field 格式修复至 1.0，但准确率仅 0.84375、区间下界 0.75，D 仍为 0.5；1.5B 能力门未通过 | 项目负责人运行；Codex 诊断 |
| 24. Impl-3f：G1h 2.9B 接口迁移 | 🟡 等待云端 | 固定官方 G1h 2.9B，先验证加载、tokenizer、state inventory 和同进程恢复 | 1.5B 的格式问题已排除，剩余是组合能力与位置平衡不足；按预定路线只提升模型规模 | 5,896,273,469 字节权重的 revision、SHA-256、配置、下载脚本和独立接口门已固定 | Codex 已完成；项目负责人运行 |
| 25. 2.9B 能力门复验 | ⏳ 未开始 | 使用已审计的 fake-think 接口重跑相同 96 条能力题 | 验证规模提升是否解决 1.5B 的 D→B 与组合能力问题 | 等待 2.9B 接口门通过 | Codex 实现；项目负责人运行 |
| 26. 新模型 state 工程门复验 | ⏳ 未开始 | 重跑磁盘恢复、reset/diff/swap 和 matched random | 新模型的 state 形状和数值尺度不同，旧模型的通过记录不能替代复验 | 等待最终候选通过能力门 | 共同完成 |
| 27. Batch 2：冻结任务参数 | ⏳ 未开始 | 冻结 checkpoint、标签池、模板、delay、答案格式和阈值 | 一旦冻结，后面不能因为结果不好随意改题或换模型 | 必须等待新模型能力门与 state 工程门都通过 | 共同审阅 |
| 28. Impl-4：预注册 | ⏳ 未开始 | 固定代码、配置、样本量、随机种子和判断标准 | 防止看到正式结果后改变成功标准 | 尚未开始 | Codex 整理；项目负责人确认 |
| 29. Phase 2：正式原生 state 实验 | ⏳ 未开始 | 比较 original/reset/random/swap 等条件 | 这一步才真正测试 recurrent state 是否是跨时间因果载体 | 尚无研究结论 | 项目负责人运行；Codex 分析 |
| 30. Phase 3：显式 Self Model | ⏳ 未开始 | 实现 Self Store、Self Encoder 和 gated injection | 只有原生状态基线可靠后，才能判断显式 Self Model 是否带来额外价值 | 目前只有设计，没有加入模型 | 后续由 Codex 实现 |
| 31. Self 更新与演化 | ⏳ 未开始 | 让 Self State 根据经历受控更新、回滚和分化 | 这是“持续自我”真正更深入的部分 | 尚未开始 | 后续阶段 |
| 32. 最终研究结论 | ⏳ 未开始 | 汇总统计结果、失败案例和替代解释 | 最终回答项目假设是否得到支持，而不是只展示几个有趣案例 | 尚未开始 | 共同完成 |

## 3. 当前所在位置

```text
理论设计
   ✅
实验工具
   ✅
模型状态保存 / 恢复 / 交换
   ✅
随机状态对照
   ✅
模型是否看得懂考题
   ⚠️ v0.1/v0.2 均为固定 A 策略
能力失败发生在哪一层
   ⚠️ copy 通过；single-field 失败，定位为 0.4B checkpoint 能力不足
更强 checkpoint 的受控迁移
   ✅ G1h 1.5B 接口门通过
新模型能否看懂考题
   ⚠️ 评分：copy 100% → single 100% → two-field 87.5%
输出格式与剩余错误审计
   ✅ reasoning 前缀与 D→B 错误已定位
官方 fake-think 单变量复验
   ⚠️ 格式已修复，但 two-field 能力与位置平衡仍未达门槛
更大 checkpoint 的受控迁移
   🟡 G1h 2.9B 接口门等待云端
正式 state 因果实验
   ⏳
显式 Self Model
   ⏳
```

当前不能得出的结论：

- 不能说模型已经拥有 Self；
- 不能说 recurrent state 已经具有身份或目标语义；
- 不能把工程门通过当成研究假设通过；
- 不能因为 v0.1/v0.2/Impl-3b 失败就断言 state persistence 不存在，因为当前证据首先指向 0.4B checkpoint 的任务能力不足；
- 不能把 G1h 1.5B 的官方能力描述当成本项目任务已通过，必须实际复验；
- 不能把 World 0.4B 的 state 工程门结果直接移植到 G1h 1.5B；
- 不能仅根据当前 `route_decision` 就说 G1h 不会照抄或不会查表，因为它的选择评分准确率分别是 100% 和 100%；
- 不能在看到 two-field 87.5% 后事后放宽原门槛，必须先解释格式失败与 D 位置偏差。

## 4. 当前下一步

Impl-3b 已完成：模型能 100% 照抄答案代码，说明 tokenizer、候选答案评分和
A–D 接口有效；但单字段查表只有 25%，且逐答案位置准确率为
`A=1, B=C=D=0`。因此当前 World 0.4B checkpoint 不进入 state-only
实验。

Impl-3c 已通过，说明当前实验框架能够可靠加载和复制 G1h 1.5B 的 state。
这仍不代表模型会做 EXP-001，也不代表 checkpoint 已正式替换。

Impl-3d 已完成，不能直接按 `revise_checkpoint_or_answer_interface` 升级模型：

- copy 的候选评分 32/32 正确，只是自由生成格式 0/32；
- single-field 的评分和自由生成均为 32/32；
- two-field 的候选评分为 28/32，但自由生成格式 0/32；
- two-field 的 A/B/C 均为 100%，D 只有 50%。

Impl-3e-a 审计已经确认：

- copy 的 32 条都先生成 ` <think>\n`；
- two-field 的 32 条都先生成 ` <think>We`；
- 4 个评分错误全部是 D→B，并非固定偏向 domain 或 operation；
- 错误分差不小，说明不能只把 87.5% 四舍五入成“通过”。

Impl-3e-b 已证明：

- 模型 96/96 次本来就会补全共同的 `>`；
- two-field 自由生成格式从 0 修复到 1.0；
- two-field 准确率没有改善，为 0.84375，区间下界 0.75；
- D 位置仍为 0.5，能力与位置平衡门都未通过。

因此停止继续调 1.5B Prompt，也不放宽阈值。下一步 Impl-3f 固定官方 G1h
2.9B，先运行接口门：

```bash
git pull --ff-only
source .venv/bin/activate
HF_ENDPOINT=https://hf-mirror.com \
  bash scripts/prepare_g1h_2.9b_candidate.sh
bash scripts/run_g1h_2.9b_interface_gate.sh
cat results/development/impl3f_g1h_2.9b_interface/summary.json
cat results/development/impl3f_g1h_2.9b_interface/model_interface_report.json
cat results/development/impl3f_g1h_2.9b_interface/state_inventory.json
```

这一步只验证 2.9B 是否与现有 state 框架兼容，不直接重跑能力题。

完整选择依据和后续复验顺序见
[checkpoint 迁移方案](docs/checkpoint_migration.md)。

## 5. 分工

| Codex 负责 | 项目负责人负责 |
|---|---|
| 理论分析和实验设计 | 决定项目方向并确认关键设计 |
| 编写代码、配置、测试和文档 | 手动推送 GitHub |
| 根据云端结果诊断问题 | 在云服务器拉取代码并执行脚本 |
| 保证失败结果不被覆盖或美化 | 把云端原始报告返回给 Codex |
| 设计下一轮受控修改 | 确认是否继续使用当前模型和计算预算 |

## 6. 强制更新规则

从本文件建立后，每完成一个实现步骤、云端门禁或研究决策，Codex 必须：

1. 更新本文件顶部的“最后更新”和“当前节点”；
2. 更新总进度表对应行的状态、工作内容和实际结果；
3. 如果出现新步骤，在正确位置新增一行，不偷偷合并或删除失败步骤；
4. 更新“当前所在位置”和“当前下一步”；
5. 在下方追加一条变更记录；
6. 将进度表更新与对应代码或结果记录一起提交，或紧随其后单独提交。

失败、Revise 和 Stop 记录必须保留，不能在后续成功后删除。

## 7. 更新记录

| 日期 | 更新内容 | 证据 |
|---|---|---|
| 2026-07-30 | 建立项目进度表；汇总 Phase 0 至 Impl-3 v0.2 的设计、实现和云端结果 | Git 历史、云端 environment/Impl-1/Impl-2/Impl-2b/Impl-2c/Impl-3 v0.1 summaries |
| 2026-07-30 | Impl-3 v0.2 再次得到固定 A 的机会水平策略；记录为 Revise，并增加不改双字段题的 Impl-3b 分层能力诊断 | `results/development/impl3_development_v02/summary.json` |
| 2026-07-30 | Impl-3b 证明答案接口有效，但 World 0.4B 在单字段查表层失败；停止该 checkpoint 的 state-only 路线，固定 G1h 1.5B 候选和独立接口门 | `results/development/impl3b_capability_ladder/summary.json`、`docs/checkpoint_migration.md` |
| 2026-07-30 | G1h 1.5B 固定资源及 Impl-3c 接口门通过；新增不复用旧模型结果、使用官方 G1 提示格式的 Impl-3d 三层能力门 | `results/development/impl3c_g1h_1.5b_interface/summary.json`、`configs/gates/impl3d_g1h_1.5b_capability_ladder.dev.json` |
| 2026-07-30 | Impl-3d 完整运行但综合门未通过；记录评分能力显著提升、自由生成格式失败和 two-field D 位置偏差，进入不重跑模型的 Impl-3e 原始结果审计 | `results/development/impl3d_g1h_1.5b_capability_ladder/summary.json` |
| 2026-07-30 | 原始输出证明 copy/two-field 先进入 `<think>`；4 个真实错误均为 D→B。新增仅改变官方 fake-think 答案前缀的 Impl-3e-b 受控复验 | `raw_capability_ladder.jsonl` 审计、`configs/gates/impl3e_g1h_1.5b_fake_think.dev.json` |
| 2026-07-30 | fake-think 将 two-field 格式修复为 1.0，但准确率和 D 位置仍未达门槛；停止 1.5B 提示修订，按预定路线固定 G1h 2.9B 接口候选 | `results/development/impl3e_g1h_1.5b_fake_think/summary.json`、`configs/assets/exp001_rwkv7_g1h_2.9b_candidate.json` |
