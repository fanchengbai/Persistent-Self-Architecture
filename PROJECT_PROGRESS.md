# Persistent Self Architecture 项目进度表

> 最后更新：2026-07-31
> 当前节点：Impl-3r-a 审计完成；等待确认模板资格的具体统计阈值阻塞项
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
| 4a. 评审内生驱动扩展 | ✅ 设计完成 | 评审“持续 Self + 世界模型 + 内生驱动”的闭环，并设计零新外部观察下的自主审议层 | 当前架构说明了 Self 如何影响一次决策，但还没有解释系统为什么会因内部冲突主动继续计算 | 已新增 Drive Signal、Deliberation Controller、预算与记忆回放的未来设计；明确 timer/random/外部反思基线和三道 ED 实验门。它属于显式 Self 与受约束更新通过后的阶段，不改变当前 Impl-3o | Codex |
| 5. 设计 EXP-001 | ✅ 完成 | 设计“身份约束 × 当前目标”的四状态任务 | 用一个很小、可量化的任务测试状态是否真的影响选择 | 已形成四组合任务、swap/reset/random 对照和评价指标 | Codex |
| 6. 建立代码与实验骨架 | ✅ 完成 | 实现任务生成、泄漏检查、统计方法、配置和报告格式 | 相当于先把实验室的记录表、评分器和质检流程搭好 | 本地纯逻辑测试目前达到 95 项全部通过 | Codex |
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
| 32. Impl-3n：2.9B reset/diff/swap 复验 | ⚠️ Revise | 对 96 个 state 组件执行比较、官方 reset 和完整交换 | 正式因果实验会依赖这些操作，必须先证明工具在 2.9B 上仍可靠且不修改来源状态 | 运行有效但总门失败：diff/swap 等均通过；reset 的 10/10 top-1 一致，logits 误差 0.03125 低于 0.0625，但 state 误差每次均为 0.155052，高于 0.125，因此容差通过 0/10 | 项目负责人运行；Codex 诊断 |
| 33. Impl-3n-a：reset 首次形状执行诊断 | ✅ 完成 | 复现原门前奏后连续 reset 11 次，分别用第 1 次和第 2 次作参考比较后续调用 | 十次完全相同的超限更像第一次遇到该 token 长度时的 CUDA 首次执行差异；必须直接验证，不能直接丢掉第一次后宣布通过 | 路线为 `first_shape_call_outlier`：第1次参考对后续 0/10 通过，第2次稳定参考对第3–11次 9/9 通过，相邻调用 9/10 通过；异常仅发生在第1→2次 | 项目负责人运行；Codex 诊断 |
| 34. Impl-3n-b：单次预热后完整状态操作复验 | ✅ 通过 | 在评分前执行一次相同 suffix 的 `state=None` 调用，然后原样重跑 diff/reset/swap | 用独立新门检验预先声明的单次预热能否消除已确认的首次形状效应，同时不覆盖原失败 | `reset_shape_warmup_count=1`；96/96 组件可区分，tokenizer、diff、reset、swap、来源状态不变性和总门全部通过。原 Impl-3n 的失败记录保留 | 项目负责人运行；Codex 诊断 |
| 35. Impl-3o：2.9B matched random 复验 | ✅ 通过 | 生成与 2.9B 真实 state 分组件尺度匹配的随机状态，并验证种子复现和稳定续算 | random 对照必须和真状态同形同尺度，才能排除“随便塞噪声”的解释 | 96 个组件；同 seed 逐位复现、异 seed 可区分、尺度和续算均有效；最大相对 L2 误差仅 `2.8688e-05`，远低于 `0.01`；来源不变且总门 `valid=true` | 项目负责人运行；Codex 诊断 |
| 36. Batch 2：冻结任务参数 | 🟡 候选修订中 | 冻结 checkpoint、标签池、模板、delay、答案格式、轮换读出和阈值 | 一旦冻结，后面不能因为结果不好随意改题或换模型 | D1–D8 的模型、协议、delay、seeds、SESOI、N=320与功效目标不变；首版模板和双字段控制未取得资格，现等待独立 Impl-3r 第二版资格证据和最终 checksum 人工确认 | Codex 实现；项目负责人运行与确认 |
| 36a. Impl-3p：历史写入协议比较 | ✅ 已通过 | 在相同案例、delay 和 state-only 查询下比较单次声明、声明后验证、多次一致绑定 | recurrent state 如何形成会直接影响正式实验，必须在确认集前固定，又不能简单挑分数最高的方案 | 384 条比较完成；三种模式标签边际化准确率分别为 96.875%、100%、100%，均超过 80% 门槛且来源 state 不变。按预注册的简洁性优先规则选择首个达标的 `single_statement`；峰值显存约 6.23 GB | Codex 已完成；项目负责人已运行 |
| 36b. Impl-3q：正式冻结候选门 | 🟠 有效 Hold | 只用 prompt-visible 题资格审查4×4正式模板，验证96条通用控制，运行10,000次功效模拟，并锁定源码、配置、原始记录和报告 digest | 在不偷看正式 state-only 结果的情况下，确认“试卷清楚、控制题可做、样本量够用、文件不能悄悄改” | `valid=true`、功效门通过，但模板资格与控制基线均失败，故 `freeze_candidate_ready=false`；608条读出完整，约18.1分钟，峰值显存约6.23GB；确认集未读取、Core Set未生成 | 项目负责人已运行；Codex 审计 |
| 36c. Impl-3q-a：失败细分审计 | ✅ 完成 | 只读取模板、控制和轮换错误分布，不重跑模型、不修改阈值 | 必须先区分模板理解、答案代码偏差和格式失败，才能决定是否修订 | 模板四轮平均后仍为107/128（83.59%）；双字段控制四轮平均仅2/8（25%），且几乎总猜 `cinder+trace`，确认是真实双字段语义失败，不是格式或A–D偏差；路线为 `revise_formal_and_control_two_field_prompt_families` | 项目负责人已运行；Codex 已诊断 |
| 36d. Impl-3r：正式冻结候选 v2 | 🟠 有效 Hold | 保留首版失败记录，统一正式模板的 `CURRENT DOMAIN/OPERATION` 字段，用常见 `COLOR/SHAPE` 重写双字段控制，并预先采用四轮平均语义读出 | 只修订审计证据明确指向的措辞和读出层，避免换模型、降门槛或扩大实验自由度 | 运行有效；控制基线与功效门均通过，但正式模板资格仍失败，故 `freeze_candidate_ready=false`。耗时约19.7分钟、峰值显存约6.23GB；确认结果未读取、Core Set未生成，checksum不得确认 | 项目负责人已运行；Codex 诊断 |
| 36e. Impl-3r-a：第二版模板细分审计 | ✅ 完成 | 读取已有512条模板分数，按历史模板、查询模板、二者交互、filler、标签对和目标组合定位剩余错误 | 控制题已通过，下一次修订只能针对正式模板，必须先知道失败是否集中在少数措辞或贯穿整个任务 | 路线为 `revise_formal_template_family_only`；四轮平均119/128（92.97%），仅9个语义错误。6个错误集中在 history-v2-03（26/32），history-v2-02为32/32；四个query均为90.63%–93.75%，没有单一查询模板崩溃；确认结果未读取、Core Set未生成 | 项目负责人已运行；Codex 已诊断 |
| 36f. 正式模板 v3 设计判定 | ⏸️ 等待精确门槛证据 | 核对 joint/identity/goal 三个 BCa 区间与冻结下界，判断失败是某个能力维度仍不稳，还是只由抽样不确定性造成 | 不能看到 history-v2-02 满分就事后只留它，也不能在不知道具体失败门时继续改措辞 | 已知点估计约为 joint 92.97%、identity 96.88%、goal 95.31%，且所有单模板点估计门均通过；尚需读取 `template_qualification_report.json` 的三个区间后再决定 v3，不降低阈值 | Codex 诊断；项目负责人回传指标 |
| 37. Impl-4：预注册 | ⏳ 未开始 | 固定代码、配置、样本量、随机种子和判断标准 | 防止看到正式结果后改变成功标准 | 尚未开始 | Codex 整理；项目负责人确认 |
| 38. Phase 2：正式原生 state 实验 | ⏳ 未开始 | 比较 original/reset/random/swap 等条件 | 这一步才真正测试 recurrent state 是否是跨时间因果载体 | 尚无研究结论 | 项目负责人运行；Codex 分析 |
| 39. Phase 3：显式 Self Model | ⏳ 未开始 | 实现 Self Store、Self Encoder 和 gated injection | 只有原生状态基线可靠后，才能判断显式 Self Model 是否带来额外价值 | 目前只有设计，没有加入模型 | 后续由 Codex 实现 |
| 40. Self 更新与演化 | ⏳ 未开始 | 让 Self State 根据经历受控更新、回滚和分化 | 这是“持续自我”真正更深入的部分 | 尚未开始 | 后续阶段 |
| 41. 内生调节与自主审议 | ⏳ 未开始 | 让 Self/冲突决定是否检索、回放、模拟或停止，并在零新外部观察条件下受控更新 | 检验系统是否不仅“有状态”，还会因内部状态选择继续计算；同时排除定时器和随机回放解释 | 设计说明已完成；必须等待显式 Self 因果价值和受约束更新两道前置门，不创建空壳代码 | 后续阶段 |
| 42. 最终研究结论 | ⏳ 未开始 | 汇总统计结果、失败案例和替代解释 | 最终回答项目假设是否得到支持，而不是只展示几个有趣案例 | 尚未开始 | 共同完成 |

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
   ⚠️ Impl-3n 的 diff/swap 通过；reset 行为一致但 state 稳定超限
   ✅ Impl-3n-a 确认只有第1→2次异常，之后 9/9 稳定
   ✅ Impl-3n-b 单次预热后 diff/reset/swap 全部通过
   ✅ Impl-3o matched-random 全部通过
Batch 2 参数冻结
   ✅ D1–D3 设计建议已确认
   ✅ Impl-3p 选择 single_statement
   ✅ D4–D8 正式模板、控制、seeds、SESOI 与统计协议已确认
   🟠 Impl-3q 有效 Hold：模板资格与控制基线未通过
   ✅ Impl-3q-a 确认模板和双字段控制均有真实语义失败
   🟠 Impl-3r 有效 Hold：控制和功效通过，仅模板资格失败
   ✅ Impl-3r-a：119/128；9错中6个集中于 history-v2-03
   ⏸️ 先核对 BCa 区间的精确失败项，再设计独立 v3
   ⏳ 后续独立候选全部门通过后才可人工确认新 checksum
正式 state 因果实验
   ⏳
显式 Self Model
   ⏳
受约束 Self 更新
   ⏳
内生调节与自主审议
   ⏳ 仅完成设计，前置门未满足
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

Impl-3n 的详细报告证明，96 个状态组件全部可区分，tokenizer、diff、swap
和来源状态不变性均通过。reset 的 10/10 次 top-1 都一致，logits 最大误差
0.03125 低于 0.0625；唯一失败项是 state 最大误差 0.155052，十次完全
相同且高于冻结上限 0.125。

锁定的 `rwkv==0.8.32` 官方源码显示，`state=None` 每次都通过
`generate_zero_state()` 建立全零状态，并非随机初始化。当前可检验推断是：
第一次遇到该 suffix 形状时产生一次 CUDA 首次执行差异，之后调用稳定。
Impl-3n-a 不覆盖原报告、不放宽阈值，结果确认：

- 第1次参考对后续为 0/10 通过；
- 第2次作为稳定参考后，第3–11次为 9/9 通过；
- 相邻调用为 9/10 通过，唯一失败是第1→2次；
- 路线明确为 `first_shape_call_outlier`。

Impl-3n-b 已在独立目录完成。在评分 baseline 之前只执行一次同 suffix、
同 `state=None` 的预热调用，并把它明确排除在计分外；其余 repeat、阈值、
确定性、diff/reset/swap 逻辑完全不变。结果为：

- `reset_shape_warmup_count=1`
- `component_count=different_component_count=96`
- tokenizer、diff、reset、swap、来源状态不变性全部为 `true`
- 总 `valid=true`

Impl-3o 已完整通过：

- 96 个 state 组件；
- 固定一次不计分形状预热；
- 同 seed 逐位复现，不同 seed 可区分；
- 最大相对 L2 尺度误差为 `2.8687819151988067e-05`，低于 `0.01`；
- tokenizer、续算、来源状态不变性和总门全部为 `true`。

这表示 2.9B 的跨进程恢复、状态操作和 matched-random 三类工程门已经闭合。
项目负责人已经确认：

1. state-only Prompt 保留通用规则，只隐藏当前 I/G；
2. 首轮先做 Track S，把 Track N 延后；
3. 首轮只纳入核心恢复与 swap 条件。

Impl-3p 已完成 384 条开发比较并通过。三种历史写入模式的标签边际化准确率为：

| 模式 | 标签边际化准确率 | 是否达 80% 门槛 | 是否修改来源 state |
|---|---:|---|---|
| `single_statement` | 96.875%（31/32） | 是 | 否 |
| `statement_plus_verification` | 100%（32/32） | 是 | 否 |
| `repeated_consistent` | 100%（32/32） | 是 | 否 |

因此不挑最高分，而是严格执行运行前固定的
`single_statement → statement_plus_verification → repeated_consistent`
顺序，冻结首个达标的 `single_statement` 作为正式候选协议。

Impl-3q 已有效完成：功效门通过，但模板资格与控制基线均未通过，因此路线为
`hold_and_revise_without_confirmatory_results`。安全边界正常：
`confirmatory_results_observed=false`、`core_set_generated=false`。
当前 checksum 只用于标识这次失败候选，不能人工确认为最终预注册版本。

Impl-3q-a 只读审计已经完成。它确认：

- 正式模板的四轮平均准确率为107/128，仍有21个语义错误；
- 双字段控制的代码级准确率为50%，四轮平均后反而只有2/8（25%）；
- 双字段控制几乎总选择 `cinder + trace`，所以不是单纯答案字母偏差；
- 复制与单字段控制均为100%，说明模型和回答接口没有整体失效。

因此新增独立 Impl-3r 第二版候选，而不覆盖或重跑 Impl-3q。它只统一正式
字段措辞、把陌生控制字段换成常见的 COLOR/SHAPE，并把已预先声明的四代码
平均用于语义控制判定；其他冻结参数和安全边界全部不变。下一步在云端运行：

```bash
git pull --ff-only
source .venv/bin/activate
bash scripts/run_impl3r_exp001_formal_freeze_candidate_v2_gate.sh
```

Impl-3r-a 已完成，结果不是“所有正式模板都不行”：

- 总体四轮平均为119/128（92.97%），比v1的107/128明显改善；
- 9个错误中6个集中在 `formal-history-v2-03`；
- `formal-history-v2-02` 为32/32，但不能事后只挑这个满分模板；
- 四个query都在90.63%–93.75%，没有证据支持只删除某个query；
- 控制任务四轮平均全部通过，因此修订范围只剩正式历史模板族。

下一步先读取模板资格报告中的三个 BCa 区间和冻结阈值，确定究竟是哪一项
导致总门失败。确认前不设计v3、不降低阈值、不重跑模型：

```bash
python -c "import json; r=json.load(open('results/development/impl3r_exp001_formal_freeze_candidate_v2/template_qualification_report.json')); print(json.dumps({'metrics':r['metrics'],'thresholds':r['thresholds'],'format_valid_rate':r['format_valid_rate'],'history_template_metrics':r['history_template_metrics'],'query_template_metrics':r['query_template_metrics']},indent=2))"
```

本次“持续 Self + 世界模型 + 内生驱动”理论评审不改变上述下一步。它补充的是
显式 Self 和受约束更新均通过后的未来研究层：系统能否根据内部冲突、
不确定性或未完成目标，选择 `stop / retrieve / replay / simulate / verify`，
并在零新外部观察时产生有依据、有预算、可回滚的状态变化。完整设计见
[`docs/endogenous_deliberation.md`](docs/endogenous_deliberation.md)。

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
| 2026-07-31 | reset 详细报告显示 10/10 top-1 一致、logits 误差通过，但 state 误差固定为 0.155052 并超过 0.125；官方 0.8.32 实现确认 `state=None` 创建全零状态。新增不覆盖失败、不放宽阈值的 Impl-3n-a 首次形状执行诊断 | `results/development/impl3n_g1h_2.9b_state_operations/reset_validation.json`、`configs/gates/impl3na_g1h_2.9b_reset_stability.dev.json` |
| 2026-07-31 | Impl-3n-a 确认 `first_shape_call_outlier`：第1次对后续 0/10，第2次参考后 9/9 稳定，相邻调用仅第1→2次失败。新增独立 Impl-3n-b，在计分前固定一次同形状 reset 预热，其余门槛和操作不变 | `results/development/impl3na_g1h_2.9b_reset_stability/summary.json`、`configs/gates/impl3nb_g1h_2.9b_state_operations_warmed.dev.json` |
| 2026-07-31 | Impl-3n-b 在固定一次不计分预热后完整通过：96/96 组件 diff、官方 reset、完整 swap、tokenizer 和来源不变性全部有效；保留原 Impl-3n 失败，解除 Impl-3o 暂停 | `results/development/impl3nb_g1h_2.9b_state_operations_warmed/summary.json` |
| 2026-07-31 | 在 Impl-3o 首次运行前，将已确认的首次 suffix 形状效应纳入其工程协议：固定一次使用 matched-random 状态副本的不计分续算预热；seed、尺度阈值和正式10次续算判定不变 | `configs/gates/impl3o_g1h_2.9b_random_matched.dev.json`、Impl-3n-a/3n-b 证据 |
| 2026-07-31 | 评审“持续 Self + 世界模型 + 内生驱动”架构：吸收计算路由、记忆回放和零新外部观察实验；将内部张力改为派生控制信号，加入 timer/random/外部反思基线与有意义更新标准；纳入 Phase/Stage 5，但不改变当前 Impl-3o | `docs/endogenous_deliberation.md`、`docs/architecture.md` v0.2、`docs/definitions.md` |
| 2026-07-31 | Impl-3o 完整通过：96 组件 matched-random 同 seed 逐位复现、异 seed 可区分、尺度与续算有效，最大相对 L2 误差 `2.8688e-05`；2.9B 状态工程门全部闭合，进入 Batch 2 冻结审阅而非正式实验 | 云端 `results/development/impl3o_g1h_2.9b_random_matched/summary.json`、`docs/exp001_batch2_freeze_review.md` |
| 2026-07-31 | 项目负责人确认 Batch 2 的 D1–D3 建议；新增 Impl-3p，在相同 Track S 案例、131-token delay 和四代码轮换下比较三种历史写入协议，按预先固定的最简通过规则选择，三种均失败时保持 Revise | `configs/gates/impl3p_g1h_2.9b_history_binding.dev.json`、`scripts/run_impl3p_g1h_2.9b_history_binding_gate.sh`、82 项本地测试 |
| 2026-07-31 | Impl-3p 完整通过：三种模式标签边际化准确率分别为 96.875%、100%、100%，来源 state 均保持不变；按运行前固定的简洁性优先顺序选择首个达标的 `single_statement`，未因另外两种满分而改选 | 云端 `results/development/impl3p_g1h_2.9b_history_binding/summary.json`、`docs/exp001_batch2_freeze_review.md` |
| 2026-07-31 | 项目负责人确认按 D4–D8 推荐冻结；新增 Impl-3q 正式冻结候选门，固定4×4模板、4个131-token filler、96条控制、5个SHA-256种子、原SESOI和N=320双重功效模拟，并硬性禁止生成 Core Set或读取正式state-only结果 | `configs/preregistration/exp001_track_s.formal_v1.json`、`scripts/run_impl3q_exp001_formal_freeze_candidate_gate.sh`、`schemas/exp001_preregistration_candidate.schema.json`、89项本地测试 |
| 2026-07-31 | Impl-3q 诊断完整但冻结候选Hold：功效门通过，模板资格和控制基线失败；候选未就绪，确认集未读取、Core Set未生成。进入只读细分审计，不确认本次checksum | 云端 `results/development/impl3q_exp001_formal_freeze_candidate/summary.json` |
| 2026-07-31 | Impl-3q细分显示格式始终有效，正式模板在四代码边际化后仍有21/128语义错误；复制和单字段控制100%，双字段控制代码级50%。新增不加载模型的模板交互与控制标签边际化审计，先区分答案代码偏差和真实组合失败 | `src/psa/preregistration/formal_review.py`、`scripts/review_impl3q_exp001_formal_freeze_candidate.sh`、90项本地测试 |
| 2026-07-31 | Impl-3q-a 只读审计确认双字段控制四轮平均仅2/8，是真实语义失败而非字母偏差；新增独立 Impl-3r v2，仅统一正式字段措辞、改用常见 COLOR/SHAPE 控制并预先采用四轮平均语义读出，保留模型、delay、seeds、N、阈值及所有安全边界 | 云端 `formal_freeze_review.json`、`configs/preregistration/exp001_track_s.formal_v2.json`、`scripts/run_impl3r_exp001_formal_freeze_candidate_v2_gate.sh` |
| 2026-07-31 | Impl-3r 有效运行但继续Hold：控制基线和功效门已通过，只有正式模板资格失败；候选未就绪、确认结果未读取、Core Set未生成。新增 Impl-3r-a 只读模板细分脚本，修订范围缩小为正式模板族 | 云端 `results/development/impl3r_exp001_formal_freeze_candidate_v2/summary.json`、`scripts/review_impl3r_exp001_formal_freeze_candidate_v2.sh` |
| 2026-07-31 | Impl-3r-a 路线确认只修正式模板：总体119/128，9错中6个集中在history-v2-03，history-v2-02满分，四个query均超过90%；不事后挑选满分模板，先读取joint/identity/goal的BCa区间确定精确失败门 | 云端 `results/development/impl3r_exp001_formal_freeze_candidate_v2/formal_freeze_review.json` |
