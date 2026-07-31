# Persistent Self Architecture 项目进度表

> 最后更新：2026-07-31
> 当前节点：Impl-3n 仅 reset 重复性失败；暂停 Impl-3o，等待 reset 详细报告
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
- ⏸️ **暂停**：后续步骤已经准备好，但前置门未通过，暂时不得运行；
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
| 6. 建立代码与实验骨架 | ✅ 完成 | 实现任务生成、泄漏检查、统计方法、配置和报告格式 | 相当于先把实验室的记录表、评分器和质检流程搭好 | 本地纯逻辑测试目前达到 72 项全部通过 | Codex |
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
| 24. Impl-3f：G1h 2.9B 接口迁移 | ✅ 通过 | 验证固定权重、tokenizer、state inventory 和同进程恢复 | 先确认新模型能被现有实验框架可靠读取和保存状态，再谈能力 | 权重校验有效；实际为 32 层、宽度 2560、96 个 state 组件；state 共 21,299,200 字节且全部有限；峰值显存 6,232,199,168 字节；接口与恢复门 `valid=true` | 项目负责人运行；Codex 诊断 |
| 25. Impl-3g：2.9B 能力门复验 | ⚠️ Revise | 使用已审计的 fake-think 接口重跑与 1.5B 完全相同的 96 条能力题 | 公平检验“只增大模型”能否解决 1.5B 的组合能力和答案位置偏差 | 诊断有效但能力门失败：copy 评分 1.0/格式 0.75；single 评分 0.90625/格式 0.875；two-field 评分 0.875/格式 0.90625；D 位置仍为 0.5 | 项目负责人运行；Codex 诊断 |
| 26. Impl-3h：2.9B 原始结果审计 | ✅ 完成 | 不重跑模型，读取已有 JSONL，汇总输出变体、混淆矩阵、格式异常和 7 个评分错误 | 判断失败究竟来自答案接口，还是 checkpoint 的真实组合能力与位置偏差 | 7 个评分错误全部以 D 为正确答案，其中 6 个 D→B、1 个 D→C；4 个同时格式失败、3 个是正常格式下的真实错误；模型 96/96 次在 `>` 后先换行，而旧评分比较的是空格+A–D | 项目负责人运行；Codex 诊断 |
| 27. Impl-3i：自然换行对齐复验 | ⚠️ Revise | 保持相同模型、96 条题、seed 和阈值，只把评分边界从 `> A` 改为模型自然生成的 `>\nA` | 检验 D 偏差是否由候选评分路径与自然回答路径错位造成 | `>\n` 贪心一致率 1.0；single-field 从 0.90625 修复到 1.0，D 从 0.625 修复到 1.0；two-field 完全不变，仍为 0.875、D=0.5；格式率也按预期不变 | 项目负责人运行；Codex 诊断 |
| 28. Impl-3j：换行对齐后的双字段错误审计 | ✅ 完成 | 只读 Impl-3i 的已有 JSONL，核对 4 个双字段错误是否与 Impl-3g 相同 | 区分“稳定的组合能力缺口”与“边界改变后发生的随机换错题” | 错误仍是完全相同的 4 个样本、3 个 D→B 和 1 个 D→C；全部分差仍为负；两题只匹配 domain、两题只匹配 operation，说明没有固定忽略同一个字段 | 项目负责人运行；Codex 诊断 |
| 29. Impl-3k：答案代码轮换诊断 | ⚠️ Revise | 让 32 个相同语义案例各自轮换使用 A/B/C/D，共 128 条配对题 | 判断错误究竟跟随字母 D，还是跟随特定语义组合 | 总准确率 0.90625；B/C 均 1.0、A=0.9375、D=0.6875；22/32 案例四轮全对，10 个案例在 D 下错，其中 2 个在 A 下也错；原自动路线“纯语义失败”过粗，应解释为强代码偏差加少量交互 | 项目负责人运行；Codex 诊断 |
| 30. Impl-3l：标签边际化只读复核 | ✅ 通过 | 将同一语义选项在 A/B/C/D 四轮下的对数分数取平均，再做一次语义选择 | 四轮平均能抵消每个字母的整体先验，判断去掉答案代码干扰后是否仍有语义错误 | 32/32 个语义案例全部正确，准确率 1.0、剩余语义错误 0；路线修订为 `answer_code_bias_controlled_by_rotation`。2.9B 的开发能力前置条件满足，后续固定采用四代码轮换平均读出 | 项目负责人运行；Codex 诊断 |
| 31. Impl-3m：2.9B 磁盘恢复复验 | ✅ 通过 | 把 2.9B 的 recurrent state 保存到磁盘，在独立进程恢复并重复续算 100 次 | 新模型的 state 更大，必须证明程序退出后仍能可靠恢复，不能借用 0.4B 的通过记录 | 达到 L3；100/100 容差通过且 100/100 top-1 一致。状态载荷 21,299,200 字节，保存约 0.169 秒，跨进程复验约 25.81 秒。逐位一致 0/100 属于已预期的 CUDA/FP16 跨进程微差，不影响通过 | 项目负责人运行；Codex 诊断 |
| 32. Impl-3n：2.9B reset/diff/swap 复验 | ⚠️ Revise | 对 96 个 state 组件执行比较、官方 reset 和完整交换 | 正式因果实验会依赖这些操作，必须先证明工具在 2.9B 上仍可靠且不修改来源状态 | 运行有效但总门失败：96/96 组件可区分，tokenizer、diff、swap 和来源不变性均通过；只有 `reset_valid=false`。不改阈值，等待 `reset_validation.json` 确定是数值误差还是行为不一致 | 项目负责人运行；Codex 诊断 |
| 33. Impl-3o：2.9B matched random 复验 | ⏸️ 暂停 | 生成与 2.9B 真实 state 分组件尺度匹配的随机状态，并验证种子复现和稳定续算 | random 对照必须和真状态同形同尺度，才能排除“随便塞噪声”的解释 | 实现和配置已完成，但因 Impl-3n 尚未通过而暂停；不得提前运行 | Codex 已完成；项目负责人待运行 |
| 34. Batch 2：冻结任务参数 | ⏳ 未开始 | 冻结 checkpoint、标签池、模板、delay、答案格式、轮换读出和阈值 | 一旦冻结，后面不能因为结果不好随意改题或换模型 | 必须等待 Impl-3m/3n/3o 都通过 | 共同审阅 |
| 35. Impl-4：预注册 | ⏳ 未开始 | 固定代码、配置、样本量、随机种子和判断标准 | 防止看到正式结果后改变成功标准 | 尚未开始 | Codex 整理；项目负责人确认 |
| 36. Phase 2：正式原生 state 实验 | ⏳ 未开始 | 比较 original/reset/random/swap 等条件 | 这一步才真正测试 recurrent state 是否是跨时间因果载体 | 尚无研究结论 | 项目负责人运行；Codex 分析 |
| 37. Phase 3：显式 Self Model | ⏳ 未开始 | 实现 Self Store、Self Encoder 和 gated injection | 只有原生状态基线可靠后，才能判断显式 Self Model 是否带来额外价值 | 目前只有设计，没有加入模型 | 后续由 Codex 实现 |
| 38. Self 更新与演化 | ⏳ 未开始 | 让 Self State 根据经历受控更新、回滚和分化 | 这是“持续自我”真正更深入的部分 | 尚未开始 | 后续阶段 |
| 39. 最终研究结论 | ⏳ 未开始 | 汇总统计结果、失败案例和替代解释 | 最终回答项目假设是否得到支持，而不是只展示几个有趣案例 | 尚未开始 | 共同完成 |

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
   ✅ G1h 2.9B 接口门通过：32 层、96 个 state 组件、恢复有效
新模型能否看懂相同能力题
   ⚠️ 2.9B 仍未通过：single/two-field 与格式门均有失败，D 位置仍为 50%
失败来自答案接口还是 checkpoint
   ✅ 所有评分错误的正确答案均为 D；同时发现评分使用空格，而自然回答先换行
自然回答边界受控复验
   ⚠️ single-field 已完全修复；two-field 与 D 偏差完全不变
换行对齐后的错误是否稳定
   ✅ 同4个样本、同3次D→B和1次D→C；不是数值随机翻转
错误跟着语义还是答案字母走
   ⚠️ D=68.75%，A=93.75%，B/C=100%；同时有2个案例在A和D下都错
抵消字母先验后语义是否正确
   ✅ Impl-3l 四轮平均后 32/32 正确，答案代码偏差已被配对轮换控制
2.9B state 工程复验
   ✅ Impl-3m 达到 L3，100/100 容差与行为一致
   ⚠️ Impl-3n 的 diff/swap 通过，但 reset 重复性失败，等待详细报告
   ⏸️ Impl-3o 暂停
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

因此停止继续调 1.5B Prompt，也不放宽阈值。Impl-3f 已证明官方 G1h 2.9B
能够被现有框架可靠加载、读取状态并完成同进程恢复。Impl-3g 随后证明，
仅提升到 2.9B 并不能通过能力门：

- copy 评分仍为 1.0，但格式率只有 0.75；
- single-field 评分为 0.90625，区间下界 0.78125，D 位置为 0.625；
- two-field 评分为 0.875，区间下界 0.78125，D 位置仍为 0.5；
- 共有 7 个评分错误和 15 个格式异常，二者可能重叠。

Impl-3h 审计已经完成：

- copy 的 8 个格式异常全部是目标 C 被续写成 Markdown 代码块；
- single-field 的 3 个评分错误全部是 D→B，并同时开始解释
  `The current symbol`；
- two-field 的 4 个评分错误全部以 D 为目标，其中 3 个 D→B、1 个 D→C；
- 总计 7 个评分错误中，4 个同时格式失败，另外 3 个是正常输出 B/C 的
  真实错误；
- 96 条自然生成都在 fake-think 关闭符 `>` 后先产生换行，但 Impl-3g
  比较的候选却是空格+A–D。

因此允许一次有审计证据支持的评分边界修正。Impl-3i 不改变模型、Prompt
正文、题目、样本 ID、seed 或阈值，只把完整候选从 `> A` 改为自然路径
`>\nA`。结果证明：

- `forced_prefix_greedy_exact_rate=1.0`，新边界与模型自然路径完全一致；
- copy 评分仍为 1.0；
- single-field 从 0.90625 提升至 1.0，D 从 0.625 提升至 1.0；
- two-field 完全不变，仍为 0.875、区间下界 0.78125、D=0.5；
- 三层格式率完全不变，符合本轮不修改自由生成格式的预期。

所以评分边界错位确实制造了 Impl-3g 的单字段错误，但不是双字段组合错误的
原因。现在不再继续修改换行或候选评分。先对 Impl-3i 现有结果做只读审计：

```bash
source .venv/bin/activate
python -m psa g1-capability-audit \
  --output-dir results/development/impl3i_g1h_2.9b_newline_aligned
cat results/development/impl3i_g1h_2.9b_newline_aligned/audit_report.json
```

Impl-3j 证明 4 个错误与 Impl-3g 完全相同：相同样本、3 个 D→B、1 个
D→C，而且新旧评分下分差都保持负值。错误分别表现为两次只保留 domain、
两次只保留 operation，并没有固定忽略某一个字段。至此停止修改
fake-think 和空格/换行边界。

剩余歧义是：错误究竟跟随答案字母 D，还是跟随这 4 个语义案例。Impl-3k
用配对轮换回答这个问题：32 个语义案例各运行 4 次，每次只循环替换
A/B/C/D 映射，共 128 条。运行：

```bash
git pull --ff-only
source .venv/bin/activate
bash scripts/run_impl3k_g1h_2.9b_code_rotation_gate.sh
cat results/development/impl3k_g1h_2.9b_code_rotation/summary.json
```

Impl-3k 得到 116/128 正确。错误分布不是纯语义失败：

- D 只有 22/32，贡献 10 个错误；
- A 为 30/32，贡献 2 个错误；
- B/C 均为 32/32；
- 22 个语义案例四轮全对；
- 10 个案例都在 D 下错，其中 2 个在 A 下也错。

因此原报告的 `semantic_composition_failure` 分类过于粗糙。它说明存在跨代码
错误案例，但忽略了 10/12 错误集中在 D 的强主效应。保留原报告不覆盖，
Impl-3l 已将每个语义选项在四种代码映射下的 log-score 取平均。结果为
32/32 全部正确，`label_marginalized_error_count=0`。这说明 2.9B 能完成
双字段语义匹配，之前 12 个代码级错误主要来自答案字母读出偏差，而不是
32 个语义案例本身无法组合。

开发阶段的正式读出规则已写入
`configs/readouts/exp001_g1h_2.9b_code_marginalized.dev.json`：每个案例
预先轮换 A/B/C/D 四种映射，对相同语义选项的四个 log-score 求平均，
不拟合任何事后字母校准参数。

Impl-3m 已证明 2.9B 状态可以跨进程落盘恢复：

- `achieved_level=L3`
- `tolerance_pass_count=100/100`
- `top1_match_count=100/100`
- 状态载荷为 21,299,200 字节
- 跨进程逐位一致为 0/100，但这是开发阶段已经识别并用冻结容差控制的
  CUDA/FP16 数值微差；行为选择全部一致，因此不构成失败

Impl-3n 已经运行。96 个状态组件全部可区分，tokenizer、diff、swap 和来源
状态不变性均通过；只有官方 `state=None` 重置后的重复续算未通过，因此
总 `valid=false`。这是一份有效失败报告，不能删除或用放宽阈值覆盖。

```bash
python -c "import json; p='results/development/impl3n_g1h_2.9b_state_operations/reset_validation.json'; r=json.load(open(p)); print(json.dumps({k:v for k,v in r.items() if k!='trials'}, indent=2)); print(json.dumps(r['trials'], indent=2))"
```

这条命令只读取已有结果，不运行模型、不使用 GPU。需要检查
`tolerance_pass_count`、`top1_match_count`、两个 worst error，以及每次失败
是 logits 超限还是 state 超限。拿到证据前不修改配置，也不运行 Impl-3o。

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
| 2026-07-30 | Impl-3f G1h 2.9B 接口门通过：固定权重、tokenizer、32 层/96 组件 state inventory 与同进程恢复均有效；新增只更换模型证据、保持 96 条题和全部阈值不变的 Impl-3g 能力复验门 | `results/development/impl3f_g1h_2.9b_interface/summary.json`、`configs/gates/impl3g_g1h_2.9b_fake_think.dev.json` |
| 2026-07-30 | Impl-3g 诊断完整但能力门未通过：2.9B 没有消除格式失败，single/two-field 也未达阈值且 D 位置仍弱；新增无需 GPU 的 Impl-3h 原始结果审计 | `results/development/impl3g_g1h_2.9b_fake_think/summary.json`、`scripts/audit_impl3g_g1h_2.9b_results.sh` |
| 2026-07-30 | Impl-3h 发现全部 7 个评分错误都以 D 为目标，同时确认模型 96/96 次在 `>` 后自然先换行，而 Impl-3g 候选使用空格；新增只对齐这一自然回答边界的 Impl-3i 复验 | `results/development/impl3g_g1h_2.9b_fake_think/audit_report.json`、`configs/gates/impl3i_g1h_2.9b_newline_aligned.dev.json` |
| 2026-07-30 | Impl-3i 将 single-field 与 D 位置完全修复到 1.0，证明换行边界修正有效；但 two-field 仍为 0.875、D 仍为 0.5，说明剩余组合错误不是该边界造成；进入 Impl-3j 只读错误审计 | `results/development/impl3i_g1h_2.9b_newline_aligned/summary.json` |
| 2026-07-30 | Impl-3j 确认换行前后是相同 4 个 D-only 双字段错误，且分别保留不同单字段；停止边界调试，新增每个语义案例完整轮换 A–D 的 Impl-3k 配对诊断 | `results/development/impl3i_g1h_2.9b_newline_aligned/audit_report.json`、`configs/gates/impl3k_g1h_2.9b_code_rotation.dev.json` |
| 2026-07-30 | Impl-3k 显示 D=0.6875、A=0.9375、B/C=1.0，且仅 2/10 失败案例跨多个代码；将原“纯语义失败”解释纠正为强代码偏差加少量交互，新增不重跑模型的 Impl-3l 标签边际化复核 | `results/development/impl3k_g1h_2.9b_code_rotation/summary.json`、`scripts/review_impl3k_g1h_2.9b_code_rotation.sh` |
| 2026-07-30 | Impl-3l 标签边际化后 32/32 语义案例正确，确认 2.9B 的开发能力前置条件通过；冻结四代码轮换平均读出，并准备不放宽旧标准的 Impl-3m/3n/3o 状态工程复验 | `results/development/impl3k_g1h_2.9b_code_rotation/code_rotation_review.json`、`configs/readouts/exp001_g1h_2.9b_code_marginalized.dev.json`、`configs/gates/impl3m_g1h_2.9b_checkpoint_roundtrip.dev.json` |
| 2026-07-30 | Impl-3m 达到 L3，2.9B 状态跨进程恢复 100/100 容差通过且 top-1 全部一致；保留 0/100 逐位一致的 CUDA/FP16 诊断记录，进入 Impl-3n 状态操作复验 | `results/development/impl3m_g1h_2.9b_checkpoint_roundtrip/summary.json` |
| 2026-07-31 | Impl-3n 有效运行但仅 reset 重复性失败；96/96 组件 diff、完整 swap、tokenizer 和来源状态不变性均通过。保留 `valid=false`，暂停 Impl-3o，先读取 reset 的逐次误差和行为证据 | `results/development/impl3n_g1h_2.9b_state_operations/summary.json` |
