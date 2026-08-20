# Persistent Self Architecture 项目进度表

> 最后更新：2026-08-20
> 当前节点：Phase 3 D4B真实2.9B稳态OFF等价门已单次通过；等待D5离线审阅与独立确认
> 研究状态：D4失败保持不变；D4B的21次调用、24项同路和96项跨路比较全部逐元素精确相等，仅形成D5审阅候选，D5/active/Self效果仍未授权

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
| 6. 建立代码与实验骨架 | ✅ 完成 | 实现任务生成、泄漏检查、统计方法、配置和报告格式 | 相当于先把实验室的记录表、评分器和质检流程搭好 | 当前全项目150项本地测试全部通过 | Codex |
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
| 36. Batch 2：冻结任务参数 | ✅ 完成 | 冻结 checkpoint、标签池、模板、delay、答案格式、轮换读出和阈值 | 一旦冻结，后面不能因为结果不好随意改题或换模型 | Impl-3t一次性留出资格门、技术审阅和负责人checksum确认均已完成；冻结参数已经进入最终预注册包 | 共同完成 |
| 36a. Impl-3p：历史写入协议比较 | ✅ 已通过 | 在相同案例、delay 和 state-only 查询下比较单次声明、声明后验证、多次一致绑定 | recurrent state 如何形成会直接影响正式实验，必须在确认集前固定，又不能简单挑分数最高的方案 | 384 条比较完成；三种模式标签边际化准确率分别为 96.875%、100%、100%，均超过 80% 门槛且来源 state 不变。按预注册的简洁性优先规则选择首个达标的 `single_statement`；峰值显存约 6.23 GB | Codex 已完成；项目负责人已运行 |
| 36b. Impl-3q：正式冻结候选门 | 🟠 有效 Hold | 只用 prompt-visible 题资格审查4×4正式模板，验证96条通用控制，运行10,000次功效模拟，并锁定源码、配置、原始记录和报告 digest | 在不偷看正式 state-only 结果的情况下，确认“试卷清楚、控制题可做、样本量够用、文件不能悄悄改” | `valid=true`、功效门通过，但模板资格与控制基线均失败，故 `freeze_candidate_ready=false`；608条读出完整，约18.1分钟，峰值显存约6.23GB；确认集未读取、Core Set未生成 | 项目负责人已运行；Codex 审计 |
| 36c. Impl-3q-a：失败细分审计 | ✅ 完成 | 只读取模板、控制和轮换错误分布，不重跑模型、不修改阈值 | 必须先区分模板理解、答案代码偏差和格式失败，才能决定是否修订 | 模板四轮平均后仍为107/128（83.59%）；双字段控制四轮平均仅2/8（25%），且几乎总猜 `cinder+trace`，确认是真实双字段语义失败，不是格式或A–D偏差；路线为 `revise_formal_and_control_two_field_prompt_families` | 项目负责人已运行；Codex 已诊断 |
| 36d. Impl-3r：正式冻结候选 v2 | 🟠 有效 Hold | 保留首版失败记录，统一正式模板的 `CURRENT DOMAIN/OPERATION` 字段，用常见 `COLOR/SHAPE` 重写双字段控制，并预先采用四轮平均语义读出 | 只修订审计证据明确指向的措辞和读出层，避免换模型、降门槛或扩大实验自由度 | 运行有效；控制基线与功效门均通过，但正式模板资格仍失败，故 `freeze_candidate_ready=false`。耗时约19.7分钟、峰值显存约6.23GB；确认结果未读取、Core Set未生成，checksum不得确认 | 项目负责人已运行；Codex 诊断 |
| 36e. Impl-3r-a：第二版模板细分审计 | ✅ 完成 | 读取已有512条模板分数，按历史模板、查询模板、二者交互、filler、标签对和目标组合定位剩余错误 | 控制题已通过，下一次修订只能针对正式模板，必须先知道失败是否集中在少数措辞或贯穿整个任务 | 路线为 `revise_formal_template_family_only`；四轮平均119/128（92.97%），仅9个语义错误。6个错误集中在 history-v2-03（26/32），history-v2-02为32/32；四个query均为90.63%–93.75%，没有单一查询模板崩溃；确认结果未读取、Core Set未生成 | 项目负责人已运行；Codex 已诊断 |
| 36f. 正式模板 v3 设计判定 | ✅ 完成 | 核对 joint/identity/goal 三个 BCa 区间与冻结下界，判断失败是某个能力维度仍不稳，还是只由抽样不确定性造成 | 不能看到 history-v2-02 满分就事后只留它，也不能在不知道具体失败门时继续改措辞 | 唯一失败项是 goal/OPERATION 的 BCa 下界0.890625，低于0.90要求0.009375；identity下界0.91127、joint下界0.859375、格式、所有history/query点估计均通过。确定只做历史写入的双字段对称强化 | Codex 已诊断；项目负责人已回传指标 |
| 36g. Impl-3s：正式冻结候选 v3 | 🟠 有效 Hold | 将四个历史模板作为整体改为平行的 FIELD 1 DOMAIN / FIELD 2 OPERATION 结构，并让确认语句明确两个字段都已保存 | goal错误几乎全部来自最弱历史模板，但不能事后只挑满分模板；整体重写能针对第二字段脆弱性，同时保留模板族泛化检验 | 运行有效但模板资格仍失败，`freeze_candidate_ready=false`；控制和功效继续通过。耗时约20.3分钟、峰值显存约6.23GB；确认结果未读取、Core Set未生成，候选checksum不得确认 | 项目负责人已运行；Codex 诊断 |
| 36h. Impl-3s-a：v3 终止判定审计 | ✅ 完成 | 读取v3已有分数，比较v2/v3的三个BCa区间、模板交互和错误集中度 | 连续两次受控模板修订仍未取得资格，必须防止无限prompt调参和开发集过拟合 | 8个联合错误中5个在history-03，但该位置v2/v3沿用同一案例流；其余错误跨query-02/03/04、filler-01/02/04、两组标签和全部目标组合。没有可辩护的下一句prompt修订；终止措辞调优 | Codex 已诊断；项目负责人已回传明细 |
| 36i. Impl-3t：v3 一次性留出资格门 | ✅ 通过 | v3模板逐字不变，用从新SHA-256命名空间生成的未观察seed创建一次全新prompt-visible资格集 | 把v1–v3视为开发过程，用未参与模板修改的数据做一次最终验证，比继续调词或直接用近失结果更能控制过拟合 | `valid=true`；模板、控制、功效全部通过，`freeze_candidate_ready=true`，路线为 `review_preregistration_checksum`。耗时约20.25分钟、峰值显存约6.23GB；确认结果未读取、Core Set未生成 | 项目负责人已运行；Codex 核对 |
| 36j. 人工 checksum 技术与内容审阅 | ✅ 通过 | 比较候选自校验、payload root、全部源码与证据digest、安全边界和冻结设计字段 | 自动门通过后仍要确认文件没有缺失、设计没有写错、冻结范围与此前决定一致 | 自校验、payload root、23项源码检查、10项证据检查和安全边界全部通过；8个正式条件与已接受D3一致，D4–D8、N=320、种子、统计方案和模板数量均一致 | Codex 核对 |
| 36k. 项目负责人确认 checksum | ✅ 已确认 | 由项目负责人逐字确认完整candidate digest，并授权把候选升级为最终预注册包 | 技术审阅不能代替研究负责人的最终冻结决定；这一步只确认预注册内容，不自动授权生成Core Set或运行正式实验 | 已明确确认`a354b208be0640da7ea70fe070f75bdec69186e496ba1cc14c3157dcd984e6cd`；授权范围明确排除Core Set生成和正式实验 | 项目负责人确认 |
| 37. Impl-4：最终预注册包 | ✅ 已冻结 | 固定候选、人工验证、负责人确认、样本量、随机种子、统计标准和安全边界，并计算最终包digest | 防止看到正式结果后改变成功标准，同时让任何人都能检查最终包是否被改动 | 状态=`final_preregistration_frozen`；最终digest=`0daf056dc6b38aa20fa69dd9e8df9b8065876529947cbc01353ffe604933d0c9`；包自校验、3个锁定文件、payload root和安全边界全部通过 | Codex 实现与验证 |
| 37a. Core Set 生成授权 | ✅ 已授权 | 单独授权只生成并冻结Core Set，不运行模型或正式实验 | 把“允许出卷”和“允许考试”分开，避免一句继续同时打开两个不可逆阶段 | 授权绑定最终预注册digest `0daf056d…d0c9`和N=320；`generate_and_freeze_core_set=true`、`run_confirmatory_experiment=false` | 项目负责人确认 |
| 37b. Core Set生成与冻结工具 | ✅ 已完成 | 验证最终预注册包和单独授权，用冻结seed、模板、标签、filler与tokenizer生成平衡试题并计算两层digest | 正式试题必须可重建、不可暗改，而且生成过程不能顺便加载模型或查看答案表现 | 固定320组×4状态×4代码轮换=1,280个语义案例、5,120条试题；验证16种模板对各20组，四类模板/filler/标签组合各80组；篡改、越权和重复生成测试通过 | Codex |
| 37c. 云端生成 Core Set v1 | ✅ 已冻结 | 在云端只加载固定tokenizer，拟合4个131-token filler并生成最终Core Set包 | 本机没有冻结tokenizer，不能用假计数器制造正式digest；云端已有经校验的1.1MB tokenizer | `status=core_set_frozen_unrun`；320组、1,280语义案例、5,120试题；Core Set digest=`6ea2b6be…eb9d`，包digest=`9659e286…1642`，安全边界保持关闭 | 项目负责人云端运行；Codex核对 |
| 37d. 持久化冻结 Core Set | ✅ 已完成 | 将云端`core_set_v1`目录原样纳入普通Git | 只留在临时云盘上可能因实例释放而丢失；必须保留生成时的原始冻结文件 | `core_set.json`和整个目录均约11MB，已通过提交`ffd79ae`推送GitHub并快进同步到本机；3个锁定文件的SHA-256均与`manifest.json`一致。无需LFS、压缩或重新生成 | 项目负责人云端提交；Codex同步复核 |
| 37e. Impl-5b-a：非推理预检与授权锁 | ✅ 已完成 | 在加载模型前核对最终预注册包、Core Set、模型/Tokenizer、环境、磁盘、显存和冻结评分源码，并产生稳定预检digest | “文件齐全”不等于“已获准考试”；必须让旧Core Set授权无法越权，并把未来正式授权绑定到一次精确预检 | 新增`confirmatory-preflight`、授权Schema和云端脚本；计划固定8条件×5,120轮换试题=40,960个trial-condition单元；旧授权复用、digest错绑均被拒绝；111项测试通过，未加载模型、未评分Core Set | Codex |
| 37f. 新主机/云端运行非推理预检 | ✅ 已通过 | 在实际实验主机验证CUDA环境、固定版本的软件依赖、模型文件SHA-256、Git干净状态和冻结包完整性 | 正式runner设计必须建立在真实主机和真实5.5GB权重校验通过的基础上，不能仅凭本机逻辑测试 | `valid=true`、无失败检查；digest=`fc6c5ccc…d9a7`；模型未加载、Core Set未评分、正式实验未授权、结果未观察。该digest只代表runner开发前基线，代码变化后必须重跑，不能用于最终授权 | 项目负责人运行；Codex核对 |
| 37g. Impl-5b-b：正式runner本地开发自测 | ✅ 已完成 | 用非Core合成夹具实现8条件状态路由、分组原子写入、断点恢复、完整性账本和默认拒绝正式执行 | 在真正考试前先用假试卷证明执行器不会串state、漏条件、半写结果或在失败后悄悄重跑 | 已实现continuous/restored/reset/random/swap/prompt-visible八条件、逐组原子落盘和仅补缺组的恢复；开发入口拒绝EXP-001/Core身份，不汇报准确率或中间决策；122项本地测试与编译检查通过 | Codex |
| 37h. Impl-5b-b：云端非Core真实模型开发门 | ✅ 已通过 | 在2.9B真实模型上只运行固定的16条非Core开发题，形成128条条件记录；随后重跑非推理preflight | 本地假后端只能证明流程，云端真模型才能证明状态构造、磁盘恢复、matched random和显存路径确实可执行 | runner开发门`valid=true`，128条原始记录完整，运行约32.84秒、加载约6.37秒、峰值显存6,311,951,360字节；不含准确率或中间结论。新版preflight全部检查通过，runner证据有效，digest=`d41d735c…74f5`；模型未在预检中加载、Core Set未评分、正式实验未授权 | 项目负责人云端运行；Codex核对 |
| 37i. Impl-5b-c：正式执行锁与封装 | ✅ 本地完成 | 在不运行Core Set的前提下实现最终授权文件校验、整批320组执行、失败即停、断点恢复、完成前禁止汇总以及最终只读结果封装 | 当前通用runner已通过，但正式启动入口必须先于最终授权冻结，否则后补代码会改变digest并使授权失效 | 启动前会现场重建preflight并逐项验证精确授权；只能完整执行320组，首次中断后必须显式resume，完成后拒绝重跑；runner只写40,960条原始记录和完整性digest，不输出准确率或中间结论。127项测试与编译检查通过，Core Set仍未运行；等待提交和云端新版preflight | Codex |
| 37j. Impl-5b-c：最终云端非推理预检 | ✅ 已通过 | 在正式执行锁源码固定后，再次核对主机、Git、模型、Tokenizer、冻结包、runner证据和全部执行源码 | 最终授权必须绑定不会再因补入口代码而变化的运行计划，不能复用任何开发阶段digest | `valid=true`、runner证据有效、失败检查为空；最终待授权digest=`9a22a0cf7fc89eed51caaed227211608b2e9492fc4db6c20fd2a89351389bd2f`。授权、运行和结果观察字段仍全部为false | 项目负责人云端运行；Codex核对 |
| 37k. 项目负责人正式实验授权 | ✅ 已明确授权 | 负责人逐字确认最终preflight digest，并明确允许运行完整320组、只在全量完成后观察结果、禁止修改冻结设计和完成后自动重跑 | 这是首次允许模型读取冻结Core Set，属于不可由“继续”或旧授权推断的独立决策 | 已逐字确认`9a22a0cf7fc89eed51caaed227211608b2e9492fc4db6c20fd2a89351389bd2f`；授权范围固定320组、5,120试题、8条件、40,960单元，只允许全量完成并验证后观察结果 | 项目负责人 |
| 37l. EXP-001完整确认性运行 | ✅ 原始包完整结束 | 在云端创建不进入Git的授权记录，通过启动锁后运行全部320组，并只监控完成组数 | 将授权文本转成机器可验证记录，同时避免授权文件改变Git commit和preflight digest | `status=confirmatory_raw_complete`、`valid=true`；320组和40,960条原始记录全部写入，payload digest=`db4ba70e…5ba7`，峰值显存6,391,454,720字节。没有派生准确率、中间决策或结果观察；禁止重跑 | 项目负责人云端运行；Codex核对完成摘要 |
| 37m. 原始确认包只读完整性验证 | ✅ 已通过 | 不计算研究指标，只核对Core Set与授权链、320个组文件、每组128条结构、SHA-256账本、总记录数和payload digest | `completion.json`是运行器自报完成；首次观察分数前必须由独立入口证明原始包没有缺失、篡改或半写 | 320/320组、40,960条记录逐项有效，失败检查和失败组均为0；payload digest仍为`db4ba70e…5ba7`，状态为`raw_package_verified_unanalyzed` | 项目负责人云端运行；Codex核对 |
| 37n. EXP-001冻结只读分析 | ✅ 已完成 | 按预注册统计方案分析E1/E2/E3、reset/random、restore、swap、特异性与联合绑定 | 把完整原始分数变成可审计的研究证据，同时禁止换指标、挑样本或自动重跑 | E1=4.5255、E2=3.7965、E3=1.2264，95%区间均超过0且明显超过0.5 SESOI，Holm p均约`3.0e-5`；continuous联合准确率93.83%，所有已测决定通过，Gate 3=`go` | Codex实现；项目负责人云端运行；共同解读 |
| 37o. EXP-001结果报告与证据边界 | ✅ 已完成 | 用通俗和技术两层语言解释结果，并明确哪些门仍不能判断 | 强效结果不能掩盖冻结Core Set缺少matched-context、同步控制和自由生成格式读出的事实 | 结果支持原生recurrent state在本任务中具有强、特异、可恢复且可因果迁移的联合行为作用；Gate 2/Gate 4仍为`not_assessable_no_full_go`，不把它夸大为“模型已经拥有Self” | Codex |
| 38. Phase 2：正式原生 state 实验 | 🟡 主实验完成，等待控制闭合 | EXP-001已完成；下一步只补齐原设计缺失的三类控制，不重跑主要终点 | 只有matched-context、每条件通用能力和正式生成格式也通过，才能完成原生state载体资格判定 | 已取得强正面信号；完整Go仍被控制缺项阻断 | 共同完成 |
| 38a. EXP-001B补充控制设计 | ✅ B1–B7已确认，未冻结 | 固定matched-context 5,120条、正式生成格式5,120条、96条控制×8条件=768条，共11,008条新记录 | 用最小补充实验闭合Gate 2/Gate 4，同时避免重跑EXP-001、修改E1–E3或追逐显著性 | 项目负责人已确认B1–B7；确认只覆盖设计和非Core开发，不是checksum确认，不授权生成补充集或正式运行 | 项目负责人确认；Codex记录 |
| 38b. EXP-001B B-Dev1 | ✅ 云端通过 | 用64个`amber/cobalt × orbit/prism`非Core案例验证4种无绑定历史能否与配对真实历史完全等token，并冻结96个state组件的99.9% RMS阈值 | matched-context必须真正等信息长度；state norm上限必须在正式数据生成前确定，不能看结果后临时设线 | `valid=true`；64/64 matched-context案例有效，64个state校准案例与96个组件阈值全部有效。加载约5.40秒、总运行约43.68秒、峰值显存6,232,199,168字节；Core未访问，补充集未生成，实验未授权/未运行/未观察 | Codex实现；项目负责人云端运行 |
| 38b-a. B-Dev1首次云端启动诊断 | ⚠️ Revise后已修复 | 分析启动时报出的`'filler_protocol'`缺字段 | 必须区分主机/模型问题与配置解析缺陷，并保留首次失败，不能直接反复运行 | 原因是v3 holdout文件是继承v1的差异配置，B-Dev1错误地直接读取overlay，没有调用既有深度合并解析器；失败发生在模型推理和报告生成前，未读取Core、未生成补充集、无结果。现已改用`_load_formal_config`并增加回归测试，等待云端重跑 | 项目负责人回传；Codex修复 |
| 38c. EXP-001B B-Dev2 | ⚠️ 有效Revise | 在非Core夹具上运行8条件128条记录，再运行16条matched-context、16条greedy格式探针和4个state norm检查 | 在冻结候选前证明运行器不会串条件，格式探针和报警路径可用，输出可原子保存和恢复 | 运行完整但`valid=false`：B-Dev1证据、条件别名、128条8条件runner、16条matched-context及`>\n`前缀率1.0均通过；生成格式仅14/16=0.875，state norm探针也失败。峰值显存6,435,030,016字节；未访问Core，未生成/授权/运行补充实验。进入只读错误明细审计，不重跑 | 项目负责人云端运行；Codex诊断 |
| 38c-a. B-Dev2失败只读审计 | ✅ 完成 | 检查两条生成失败和四个state的逐组件越界比例 | 判断是模型/接口真实失败，还是开发夹具与冻结阈值比较对象不一致 | 两条格式失败均在正确`>\n`后生成`<tool_call`；四个state各有66–67/96组件越界，最高约1.209倍，说明不是偶发数值毛刺。审计确认通用runner短历史不属于B-Dev1的131-token正式history校准族，norm资格判定输入错配；16条通用生成题也不能代表四正式模板族 | 项目负责人回传；Codex审计 |
| 38d. B-Dev2 v0.2正式形状非Core复验 | ✅ 云端通过 | 保持8条件runner和matched探针不变，改用64条与正式history/query/filler结构一致的非Core题运行生成与norm探针，并写入新目录 | 修复错误的开发比较对象，同时保留v0.1失败；不放宽0.99格式阈值或state norm阈值 | `valid=true`：条件runner、别名、16条matched、64条正式形状生成和64个norm检查全部通过；前缀与格式率均1.0，清单digest=`4a4a2700…34acf`。加载约5.42秒、总耗时约133.57秒、峰值显存6,435,030,016字节；安全边界全为false | Codex实现；项目负责人云端运行 |
| 38e. EXP-001B预注册候选确认 | ✅ 负责人已确认 | 锁定B1–B7设计、B-Dev1、B-Dev2 v0.1失败与v0.2成功证据，完成技术审阅并由负责人确认完整checksum | 只有逐字确认的候选才能升级，防止把“继续”误当成冻结或运行授权 | 12/12源码、22/22证据及全部digest检查通过；项目负责人确认`c7a69971…a3eb`，授权仅限升级最终预注册包，明确不授权生成补充集或运行实验 | 项目负责人确认；Codex记录 |
| 38f. EXP-001B最终预注册包 | ✅ Git持久化并跨主机复核 | 原样复制候选、验证报告和22份证据，加入负责人确认记录，生成包级manifest并独立复核 | 把自然语言确认变成机器可验证、可提交Git的冻结包，同时继续阻断数据生成和实验运行 | 远程提交`e9c3528`只含冻结包26个文件；本机快进同步后独立复核`valid=true`，25/25锁定文件无失败，final digest=`976cce8c…23be`。数据生成、实验运行、结果观察与自动重跑权限仍全部为false | 项目负责人云端持久化；Codex跨主机复核 |
| 38g. EXP-001B补充集生成门 | ✅ Git持久化并跨主机复核 | 使用绑定加固版预检的负责人授权，确定性生成并冻结11,008条记录；云端提交后在另一主机重新运行独立包验证 | 正式补充数据必须完整、可追溯、不可暗改，同时继续把“生成试卷”和“运行实验”分权 | 远程提交`62dd8b2`严格只有5个冻结包文件；本机复核`valid=true`、`content_valid=true`、锁定文件零失败，两层digest保持`7c3606be…65d33`/`68e9a9a7…d5954`。实验授权/运行/结果观察仍为false | 项目负责人云端持久化；Codex跨主机复核 |
| 38h. EXP-001B正式运行准备 | ✅ 最终预检通过，等待独立授权 | 已实现只读正式预检、三类记录完整runner、320组原子保存/显式续跑、原始包验证与独立运行授权锁；修复后在真实2.9B模型上重跑全部非Core路由并形成最终预检checksum | 已有“试卷”不等于允许“考试”；运行源码、模型、冻结包和资源状态必须形成新的预检checksum | 修复后开发门`valid=true`：16条matched、16条生成和8条控制共40条，三类路由与8条件全部覆盖；加载约7.47秒、运行约18.74秒、峰值显存6,311,951,360字节。最终预检全部检查为true，runner证据有效，digest=`bc91b2b3…13a6`；模型未在预检中加载，冻结补充trial未评分，正式授权/运行/结果观察仍为false | Codex实现；项目负责人云端运行 |
| 38h-a. 首次正式预检运行标志诊断 | ✅ 修复并复验 | 预检唯一失败项为`runtime_flags_frozen`；审计并修复三个GPU入口的自包含RWKV运行模式 | 新终端不应依赖旧shell遗留变量；否则同一脚本在不同会话会得到不同preflight | 首次digest=`3d48b04b…0365`永久作废。三个入口已固定`RWKV_V7_ON=1`、`RWKV_JIT_ON=0`、`RWKV_CUDA_ON=0`及项目`PYTHONPATH`；修复提交`defd6b0`上的非Core门和最终预检均复验通过 | 项目负责人回传；Codex诊断修复 |
| 38i. EXP-001B项目负责人正式运行授权 | ✅ 已明确授权 | 负责人逐字确认最终preflight checksum及完整运行范围，授权记录必须与冻结包、模型、320组和11,008条记录精确绑定 | 生成补充集的旧授权不能自动升级为“允许模型读取并评分试卷”；正式运行是新的不可逆研究边界 | 已确认checksum=`bc91b2b3cd7557cc2000d8990fbc638de38a6d9895891e8b5b944e9572ab13a6`；授权固定2.9B模型、320组和11,008条记录，仅允许全量完成并通过原始包验证后观察结果；禁止修改设计、自动重跑和重跑EXP-001 | 项目负责人 |
| 38j. EXP-001B完整补充运行 | ⏸️ 已授权，尚未启动 | 在云端忽略目录创建机器可验证授权记录，通过现场preflight匹配与执行锁后运行320组；运行期间只保存原始记录 | 将自然语言授权变成可审计的机器记录，并阻止中途看准确率、临时改设计或完成后重跑 | 当前尚未创建云端授权文件、尚未加载冻结补充记录评分。启动后只监控进程与已完成组数；320组结束后先运行独立原始包验证器 | 项目负责人云端运行；Codex核对 |
| 38k. EXP-001C v02 Stage A正控制 | ✅ 已通过并关闭授权 | 使用G1 fake-think/chat prompt-visible协议运行32条非Core code-rotated正控制 | recurrent-state解释前必须先证明同一题在答案信息可见时可做 | 28/32正确，label-marginalized accuracy=0.875，prefix greedy/roundtrip均为1.0；单次授权已关闭，禁止自动重跑 | 项目负责人授权；Codex云端执行与观察 |
| 38l. EXP-001C v02 Stage B离线设计 | ✅ 本轮完成 | 把Stage A作为外部基线，规划continuous/restored/三种swap/reset/random共7条件×32条=224条，并建立确定性manifest、风险审查和未来授权Schema | 防止Stage A隐式重跑、交换状态仍沿用旧target、负控制混入主要端点或“继续”被误当成执行授权 | 5项专项及全项目234项测试通过；交换目标按实际注入状态唯一重映射，reset/random仅作诊断；模型未加载、正式测试集未访问、执行和观察均未授权 | Codex |
| 38m. EXP-001C v02 Stage B纯离线runner/backend | ✅ 本轮完成 | 用只接受未加载fake adapter的backend执行224条路由，原子写入synthetic结果包，并保留无条件关闭的真实模型入口 | 在写真实RWKV路径前证明记录不漏、不重、条件不串，且离线测试输出不会被误当成研究结果 | 7项新增测试、Stage B合计12项专项及全项目241项通过；七条件各32条，缺锁、非fake/已加载adapter、非空输出和模型入口均失败关闭。未加载模型、未访问正式测试集 | Codex |
| 38n. EXP-001C v02 Stage B真实RWKV backend纯代码集成 | ✅ 本轮完成 | 从32条protocol trial构造8个唯一history state，按两个block执行2×2磁盘恢复和matched random，再把continuous/restored/三种swap/reset/random路由到224条评分记录 | 在授权模型执行前证明真实backend的数据依赖、state来源、前缀证据和目标重映射与离线设计一致 | 新增真实结果Schema和5项fake-RWKV专项测试；Stage B合计17项、全项目246项通过。snapshot调用2组×4状态，random seed 8个且唯一；缺少执行授权、错误model路径或digest均在加载前拒绝。本轮未调用模型工厂 | Codex |
| 38o. EXP-001C v02 Stage B执行runner安全外壳 | ✅ 本轮完成 | 在授权验证后独占消费single-use claim，调用backend，原子写入224条原始结果、独立完整性报告和无研究指标摘要；失败后禁止自动重入 | 确保一次授权只能启动一次，半失败或无效输出不能被覆盖重跑，完整性检查也不能提前泄露准确率 | 新增6项runner测试，Stage B合计23项、全项目252项通过；live authority validator默认无条件关闭。成功、重复启动、缺锁、缺授权、无效结果和篡改结果路径均验证；本轮只用fake backend | Codex |
| 38p. EXP-001C v02 Stage B只读live preflight与机器授权锁 | ✅ 云端通过 | 在模型加载前绑定干净main提交、设计/protocol digest、Stage A原始结果、模型配置与资产哈希、主机环境、224条计划和空输出目录；授权只接受固定逐字文本并绑定preflight digest | 防止把“继续”解释成模型执行授权，也防止代码、证据、模型或输出目录变化后复用旧授权 | Stage B 29项远程测试通过，云端只读preflight全部检查为true且失败项为空；`model_loaded=false`、`model_executed=false`、执行/观察均为false。最终digest以本轮最终文档提交后的服务器v02证据为准 | Codex |
| 38q. EXP-001C v02 Stage B项目负责人单次授权 | ✅ 已逐字确认并消费 | 负责人使用冻结原文授权224条recurrent-state非Core pilot及本轮结果观察，同时明确排除Stage A重跑、正式测试集、正式运行、确认性决定和自动重跑 | 模型执行和结果观察是新的不可逆边界，不能由此前“继续”推断 | 授权绑定preflight_v03与Stage A/result digest，机器记录和single-use claim均已消费；224条运行及观察完成后禁止重跑 | 项目负责人；Codex执行 |
| 38r. EXP-001C v02 Stage B冻结只读观察 | ✅ 云端完成 | 对五个状态语义条件按8个语义案例×4代码轮换平均log score，记录联合/字段准确率与margin；reset/random只记录参考匹配率，不定义正确答案 | 原始code top-1容易受A–D先验影响；同时不能把诊断控制事后改成主要端点或临时添加通过阈值 | 五个主要条件均联合7/8、domain 8/8、operation 7/8；continuous/restored预测8/8一致，三种swap均7/8跟随注入state。reset/random参考匹配均2/8；无确认性决定或重跑 | Codex；云端只读分析 |
| 39. Phase 3：显式 Self Model | 🟡 D4B真实稳态OFF等价门已通过，等待D5离线审阅 | 实现静态Self Store、Self Encoder和可关闭/缩放gated injection，并建立字段mask/swap/random/coupling-off消融 | 先证明接口可审计、可干预、失败关闭，再决定真实RWKV注入位置和效果实验 | 干净main提交`949bfa0`上的唯一真实2.9B运行完成：21次调用全部记录，24项同路线和96项跨路线比较的logits及96个state组件均`torch.equal`且误差为0；报告`valid=true`、digest=`8befb5f4…a20`，decision effect仅为`d5_review_candidate_only`。D4原失败不变，D5/active/Self效果均未授权或执行 | 项目负责人授权并运行；Codex独立核验与观察 |
| 40. Self 更新与演化 | ⏳ 未开始 | 让 Self State 根据经历受控更新、回滚和分化 | 这是“持续自我”真正更深入的部分 | 尚未开始 | 后续阶段 |
| 41. 内生调节与自主审议 | ⏳ 未开始 | 让 Self/冲突决定是否检索、回放、模拟或停止，并在零新外部观察条件下受控更新 | 检验系统是否不仅“有状态”，还会因内部状态选择继续计算；同时排除定时器和随机回放解释 | 设计说明已完成；必须等待显式 Self 因果价值和受约束更新两道前置门，不创建空壳代码 | 后续阶段 |
| 42. 最终研究结论 | ⏳ 未开始 | 汇总统计结果、失败案例和替代解释 | 最终回答项目假设是否得到支持，而不是只展示几个有趣案例 | 尚未开始 | 共同完成 |

## 3. 当前所在位置

> 2026-08-20 当前状态：D4真实2.9B失败永久保持；D4A报告digest=`d6b0602a…2e88`、claim=`21055ee6…7754`和`within_route_instability_observed`分类均冻结。D4B已在干净main提交`949bfa0e…54d`上按唯一授权单次完成：authorization digest、授权文件SHA-256、single-use claim SHA-256和最终报告自digest四层链均独立复算一致；21次调用全部记录，阶段数为1次prefix、4次固定预条件和16次拉丁计分，24项同路线及96项跨路线比较的logits和全部96个state组件均逐元素精确相等、最大误差与不等元素数均为0。报告`valid=true`、状态`d4b_real_off_equivalence_passed`、digest=`8befb5f4…a20`，运行约17.18秒、CUDA峰值6,381,519,360字节。该结果支持“固定预条件后的稳态OFF路径等价”，但不改写D4首调用失败，也没有定位底层瞬态机制；决策效应严格只到`d5_review_candidate_only`。D5、active injection、Self projection和Self效果实验均未授权或执行。下一步只允许离线审阅D5范围并等待项目负责人独立确认，不得重跑D4/D4B或自动进入模型实验。以下保留完整历史路径；如与旧阶段描述冲突，以本段和顶部“当前节点”为准。

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
   ✅ 唯一失败项：goal BCa下界0.890625 < 0.90
   🟠 Impl-3s 有效 Hold：模板资格仍失败
   ✅ Impl-3s-a：错误跨多因素，无可辩护的v4措辞修订
   ✅ Impl-3t：一次性未观察留出集全部门通过
    ✅ candidate、verification、设计范围与安全边界人工核对通过
    ✅ 项目负责人明确确认完整 checksum
    ✅ 最终预注册包冻结并自校验通过
    ✅ Core Set生成已单独授权
    ✅ 生成与冻结工具已通过逻辑测试
    ✅ 云端Core Set v1已生成并冻结
    ✅ 冻结目录已通过普通Git持久化并复核
    ✅ 非推理预检与独立授权锁已通过111项测试
    ✅ 实际主机非推理预检完整通过
    ✅ 仅用非Core夹具完成runner本地实现与122项测试
    ✅ 云端非Core真实模型开发门及新版非推理预检通过
    ✅ 正式执行锁与整批原始结果封装已完成本地实现
    ✅ 正式执行锁提交后的云端最终预检通过
    ✅ 项目负责人已逐字确认最终digest并单独授权
    ✅ 云端完整320组原始运行结束，40,960条记录齐全
    ✅ 独立只读完整性验证通过
    ✅ 冻结分析和正式结果报告完成
    ✅ E1–E3、特异性、reset/random、restore与swap均通过
    ⚠️ Gate 2/Gate 4因三类正式控制缺失不可完整判断
EXP-001B补充控制
    ✅ B1–B7已确认；确认不包含冻结或运行授权
    ⚠️ B-Dev1首次启动因overlay未合并失败；失败保留，修复后已通过
    ✅ B-Dev1修复后云端通过：64案例、96组件均有效
    ⚠️ B-Dev2有效Revise：runner/matched通过，生成格式与norm失败
    ✅ 只读审计确认通用夹具与正式形状阈值分布错配
    ✅ B-Dev2 v0.2正式形状非Core复验云端通过
    ✅ 未确认预注册候选已在云端生成：12个源码、22个证据文件
    ✅ self digest、payload root、安全边界及34项文件检查全部通过
    ✅ 项目负责人已确认完整checksum；授权只升级最终预注册包
    ✅ 最终预注册包已云端冻结并自校验通过
    ✅ 最终包已由远程提交`e9c3528`持久化并在本机独立复核
    ✅ 补充集生成前预检、确定性生成器、完整性验证与独立授权锁已实现
    ⚠️ 首次checksum=`1f117d5c…fa65`未覆盖生成器源码，已废止
    ✅ 加固版云端预检通过并绑定12个生成源码/配置/脚本SHA-256
    ✅ 负责人已确认新checksum=`dbb17d97…2ae6`并单独授权生成11,008条记录
    ✅ 云端生成、冻结和独立完整性验证通过；package digest=`68e9a9a7…d5954`
    ✅ 28MB冻结包由提交`62dd8b2`推送并跨主机复核通过
    ✅ 正式runner、原始包验证与独立运行锁已实现
    ✅ 修复后非Core真实模型开发门与最终只读预检均通过
    ✅ 项目负责人已逐字确认最终checksum并单独授权完整正式运行
    ⏸️ 等待云端建立机器授权记录并启动；实验尚未执行、结果尚未观察
正式 state 因果实验
   🟡 EXP-001主实验完成；等待EXP-001B控制闭合后作最终阶段决策
显式 Self Model
   🟡 D4B真实稳态OFF等价门已单次通过；仅进入D5离线审阅候选
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

> 2026-08-20 当前下一步：D4B唯一真实运行已经结束且不得重跑。先在本地进行D5离线审阅，明确D5究竟是“active injection实现前的设计/安全门”还是包含真实模型执行，并冻结最小范围、失败关闭条件、消融矩阵和新的独立授权边界；在项目负责人明确确认前，不实现或运行active injection，不选择性重跑D4/D4B，不加载模型开展Self效果实验，也不把`d5_review_candidate_only`解释为D5已授权。以下保留此前 EXP-001B 轨迹作为历史记录。

截至2026-08-04，项目负责人已经确认EXP-001B设计草案中的B1–B7。
新增范围仍锁在11,008条控制记录，并明确不重跑EXP-001、不重估E1–E3、
不复用旧授权。两个非Core开发门已经本地实现，当前依次执行：

1. B-Dev1已经通过：64个非Core案例全部精确token配对，96组件state norm阈值有效；
2. B-Dev2 v0.1失败已审计：通用runner夹具不属于正式history形状族，不能用于norm资格判定；
3. B-Dev2 v0.2已按原阈值通过，旧v0.1失败仍保留；
4. 未确认候选已在云端生成，digest=`c7a6997179072db22bb518289cd1ab0e2428f8a9eb6ea4dcc50983bbe212a3eb`；
5. `preregistration_verification.json`已核对：self digest、payload root、安全边界、12个源码和22个证据文件全部通过，失败列表为空；
6. 项目负责人已逐字确认完整checksum，授权范围仅为升级最终预注册包；
7. `preregistration/exp001b/final_v1`已在云端冻结并自校验通过，final digest=`976cce8c9e3b53bca2d21ae43f273228c45dfc4607f5b652a3d5b5cdc5d823be`；
8. 最终包已由远程提交`e9c3528`持久化，本机独立复核25/25锁定文件全部有效；
9. 首次云端预检digest=`1f117d5cd0e9d37706c50bc10db37bf826eaa7166a366589e8ab4121499cfa65`因未绑定生成器源码而废止；
10. 加固版预检已纳入12个生成相关源码、配置、Schema和脚本digest，并在云端通过，新digest=`dbb17d975d32956fb92ab39975452f00c6bc1ca8b4114afec84f5a46f8242ae6`；
11. 项目负责人已逐字确认该digest，只授权生成并冻结11,008条补充记录，明确不授权运行正式补充实验；
12. 补充集已在云端生成并由独立验证器复核通过，set digest=`7c3606be819d4e6cc5420f0bf36efd1906f8954d362d83e912785cc943565d33`、package digest=`68e9a9a79fe4e493a0c64ba8c0278c300cc832d940ab902feaceb4ad7f9d5954`；
13. 冻结包已由提交`62dd8b2`持久化并在本机跨主机复核，5个文件、11,008条记录和全部digest一致；
14. 正式运行基础设施已实现：固定320组、11,008条记录，支持原子写入、显式续跑、完成后拒绝重跑，并提供独立原始包验证器；
15. 非Core正式runner开发门首次通过后，首次正式预检发现入口脚本未自行固定三个RWKV运行标志；唯一失败项为`runtime_flags_frozen`，失败digest `3d48b04b…0365`永久作废；
16. 三个GPU入口已补齐自包含运行标志，并在云端提交`defd6b0`上按顺序重跑：非Core开发门`valid=true`，最终只读预检所有检查为true；
17. 最终待授权preflight digest=`bc91b2b3cd7557cc2000d8990fbc638de38a6d9895891e8b5b944e9572ab13a6`；它绑定最终预注册包、父Core Set、补充集、模型、运行源码、开发门证据、主机环境和固定运行标志；
18. 项目负责人已逐字确认该checksum，授权范围固定为2.9B模型、320组、11,008条记录，仅允许在全量完成并通过完整性验证后观察结果；禁止修改冻结设计、自动重跑和重跑EXP-001；
19. 下一步只在当前云端提交`defd6b0`上创建不进入Git的机器授权记录并启动正式runner。运行期间不计算或查看准确率；320组结束后先执行独立原始包验证，再决定是否开启只读冻结分析。

完整设计见[`docs/exp001b_supplemental_design.md`](docs/exp001b_supplemental_design.md)。
下面保留此前各步的诊断依据，作为不可删除的研究轨迹。

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
3. 首轮纳入 `continuous`、`restored`、`reset`、`random_matched`、
   `swapped_I`、`swapped_G`、`swapped_both` 与 `prompt_visible`；
   interpolated、layer/channel ablation、Probe 和 Track N 后移。

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

精确门槛已经确认：

- goal/OPERATION 点估计95.31%，但BCa下界0.890625，低于0.90；
- identity下界0.91127、joint下界0.859375，分别通过0.90和0.80；
- 格式为100%，每个history/query模板点估计也全部通过各自0.80门槛。

因此 Impl-3s 不改query、控制、数据规模或阈值，只把四个history作为一个
模板族进行对称重写：FIELD 1明确保存DOMAIN，FIELD 2明确保存OPERATION，
确认语句同时点名两个字段。

Impl-3s 已有效运行，但模板资格仍未通过；控制和功效继续通过，安全边界
正常。精确指标显示v3确有改善：

- goal点估计从95.31%升到96.09%，BCa下界从0.890625升到0.8984375；
- identity下界0.90888、joint下界0.875，均通过；
- 格式100%，四个history与四个query的点估计全部通过；
- 但goal下界仍比0.90低0.0015625，因此不能确认候选。

v3错误明细已经完成。8个联合错误中5个落在history-03，但这个索引在v2和
v3中沿用同一批随机案例，因此不能把差异纯归因于措辞。错误同时跨越三个
query、三个filler、两组标签和全部目标组合，没有一条可辩护的v4改句方案。

因此永久停止当前模板调优，把v3逐字冻结。Impl-3t使用从公开SHA-256命名空间
推导的新seed `3061017642`，生成一次未观察的同规模资格集。规则在运行前锁定：

- 通过：只进入新candidate checksum人工审阅，仍不生成Core Set；
- 失败：停止G1h 2.9B正式资格路线；
- 无论结果如何：不再改prompt、不重抽样、不增样、不挑选有利模板。

Impl-3t 已按该规则一次性通过：

- 模板资格、控制基线和功效门全部为true；
- `freeze_candidate_ready=true`；
- 确认结果仍未读取，Core Set仍未生成；
- 候选digest为
  `a354b208be0640da7ea70fe070f75bdec69186e496ba1cc14c3157dcd984e6cd`。

人工只读审阅和负责人确认现已完成。候选digest
`a354b208be0640da7ea70fe070f75bdec69186e496ba1cc14c3157dcd984e6cd`
已升级为最终预注册包，最终包digest为
`0daf056dc6b38aa20fa69dd9e8df9b8065876529947cbc01353ffe604933d0c9`。
包内锁定候选、验证报告和人工确认记录；自校验、payload root和安全边界均
通过。项目负责人随后又单独授权生成并冻结Core Set，但仍明确排除正式实验。
Core Set已在云端一次性生成：320组、1,280个语义案例和5,120条完整代码
轮换试题全部吻合，状态为`core_set_frozen_unrun`。Core Set digest为
`6ea2b6be15a7728c96d84dcc8e48da64e740438980f818e78c8ee8570a47eb9d`，
包digest为
`9659e286de4128b43226f2d6df27075eba60bd953c2330ee70c0ec3e677f1642`。
正式实验仍未授权、未运行、未产生结果。`core_set.json`和整个目录均约11MB，
现已通过普通Git提交`ffd79ae`安全保存到GitHub，并快进同步回本机。同步后对
`core_set.json`、`core_set_authorization.json`和
`final_preregistration_manifest.json`重新计算SHA-256，三者均与冻结
`manifest.json`一致。Core Set工程步骤已经闭合。Impl-5b-a非推理预检已在
新主机通过；随后完成Impl-5b-b的本地runner实现：8条件状态路由、真实磁盘
恢复、matched random、分组原子写入、断点恢复与非Core输入锁均已纳入测试，
本地122项测试通过。旧preflight digest `fc6c5ccc…d9a7`只属于runner提交前
基线，现已失效。下一步只能在云端运行固定非Core开发夹具，再重跑不加载模型
的preflight，取得绑定runner源码和开发证据的新digest。Core Set仍未评分，
正式实验仍需项目负责人另行明确授权。

云端非Core开发门现已通过：真实2.9B模型完成1组16题×8条件=128条原始记录，
约32.84秒，峰值显存6,311,951,360字节；没有派生准确率或中间决策。随后新版
preflight所有检查为true，runner证据有效，digest为
`d41d735c5ec7da3462ee1cbc5a6ec400ab3877539b85fa3761aeac9c70aa74f5`。
但正式执行入口与启动锁尚未实现，不能让项目负责人现在绑定这个临时digest；
应先完成Impl-5b-c封装和非Core测试，再重跑preflight取得最终授权digest。

Impl-5b-c现已完成本地实现：任何模型加载之前都会现场重建preflight、验证
持久化preflight与当前主机/源码完全一致，并检查项目负责人授权文件逐字段绑定
新digest、最终预注册包、Core Set和模型。正式入口没有子集参数；固定运行全部
320组、40,960条条件记录。中断后默认拒绝继续，必须人工显式resume；全部完成
后拒绝再次运行。执行器在完整结束前只保存逐组原始分数和SHA-256账本，不计算
准确率或研究结论。当前只完成逻辑测试，Core Set仍未运行。

正式执行锁提交后，云端最终非推理preflight再次通过：`valid=true`、runner
开发证据有效、失败检查为空，最终待授权digest为
`9a22a0cf7fc89eed51caaed227211608b2e9492fc4db6c20fd2a89351389bd2f`。
此时`confirmatory_experiment_authorized=false`、
`confirmatory_experiment_run=false`、`confirmatory_results_observed=false`。
下一步只能由项目负责人作出独立正式授权决定，不能把本次结果或此前的“继续”
解释为授权。

项目负责人现已逐字确认最终digest并明确授权完整确认性实验。授权固定模型
`rwkv7-g1h-2.9b-20260710`、320组、5,120条轮换试题、8条件和40,960个
trial-condition单元；只允许全量完成且完整性验证后观察结果，不允许修改冻结
设计或完成后自动重跑。下一步在云端已忽略的`results/authorizations/`生成机器
可验证授权记录并启动；运行过程中只监控完成组数，不读取分数。

云端正式后台进程现已启动：`bash scripts/run_exp001_confirmatory.sh`与
`python -m psa confirmatory-run`均存活，Python处于高负载。首次看到
`waiting 0/320`是监控终端位于`~`而使用了相对路径造成的观察路径错误；正式
输出仍位于项目绝对路径。没有重新启动进程，也没有打开组级原始分数。

正式runner现已写完全部320组和40,960条原始记录，完成状态为
`confirmatory_raw_complete`，payload digest为
`db4ba70ed521b55f23c4fc0ddafd2fb09af3cbe0132c0f065358a96f858b5ba7`，
峰值显存6,391,454,720字节。完成摘要继续明确
`contains_derived_accuracy=false`、`contains_interim_decision=false`和
`confirmatory_results_observed=false`。下一步先运行独立原始包验证器，不直接
进入统计分析。

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

从本文件建立后，每完成一轮操作（包括实现、云端门禁、实验运行、结果观察、文档同步或研究决策），Codex 必须在结束该轮前：

1. 检查并更新本文件顶部的“最后更新”和“当前节点”，不得只在文末追加记录；
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
| 2026-07-31 | 精确门槛显示唯一失败项为goal BCa下界0.890625<0.90；identity、joint、格式和单模板点估计均通过。新增独立Impl-3s，将全部四个history统一改为FIELD 1 DOMAIN/FIELD 2 OPERATION对称结构，不挑选满分模板且不改query、控制、N、seed或阈值 | `configs/preregistration/exp001_track_s.formal_v3.json`、`scripts/run_impl3s_exp001_formal_freeze_candidate_v3_gate.sh` |
| 2026-07-31 | Impl-3s 有效运行但模板资格再次失败；控制、功效与安全边界通过，候选未就绪。暂停继续设计v4，先只读比较v2/v3区间与错误分布，防止无限prompt调参 | 云端 `results/development/impl3s_exp001_formal_freeze_candidate_v3/summary.json` |
| 2026-07-31 | v3精确指标显示goal点估计96.09%、BCa下界0.8984375，比v2改善但仍严格低于0.90；identity、joint、格式及所有单模板点估计通过。保持Hold且不四舍五入，等待只读错误明细后做终止判定 | 云端 `results/development/impl3s_exp001_formal_freeze_candidate_v3/template_qualification_report.json` |
| 2026-07-31 | v3的8个错误跨query、filler、标签和目标组合；history-03的5个错误与固定案例流混杂，不能继续归因于一句措辞。终止prompt调优，新增一次性未观察Impl-3t留出门：v3逐字不变，新seed 3061017642；通过只审checksum，失败停止2.9B路线，禁止再抽样或改模板 | `configs/preregistration/exp001_track_s.formal_v3_holdout.json`、`scripts/run_impl3t_exp001_formal_v3_holdout_gate.sh` |
| 2026-07-31 | Impl-3t 一次性留出资格门全部通过：模板、控制、功效有效，候选就绪且安全边界正常；停止计算，进入candidate digest `a354b208…e6cd` 的人工核对，尚未确认、未生成Core Set | 云端 `results/development/impl3t_exp001_formal_v3_holdout/summary.json` |
| 2026-07-31 | Impl-3t候选技术与内容审阅通过：self digest、payload root、23项源码、10项证据和安全边界全部有效；8个条件与D3一致，D4–D8、N=320、种子、统计方案和模板数量核对无误。下一步只等待项目负责人明确确认完整checksum；尚未升级预注册包、生成Core Set或运行正式实验 | 云端 `preregistration_verification.manual.json`、`preregistration_candidate.json` |
| 2026-07-31 | 项目负责人逐字确认完整candidate checksum并授权只升级最终预注册包；新增不可越权的finalize/verify入口，冻结候选、验证报告和人工确认记录。最终包digest为`0daf056d…d0c9`，自校验、3个锁定文件、payload root和安全边界全部通过；Core Set和正式实验仍未授权、未执行 | `preregistration/exp001/final_v1/manifest.json`、`scripts/finalize_exp001_preregistration.sh`、新增7项测试 |
| 2026-07-31 | 项目负责人单独授权生成并冻结Core Set、继续禁止正式实验；新增授权锁、Core Set生成/验证入口和云端脚本。设计固定为320组×4状态×4代码轮换=1,280语义案例/5,120试题，16个历史×查询组合各20组，四类模板/filler/标签组合各80组；本地用测试tokenizer验证平衡、幂等、篡改检测和越权拒绝，真实digest等待云端冻结tokenizer生成 | `preregistration/exp001/core_set_authorization.json`、`scripts/generate_exp001_core_set.sh`、`src/psa/preregistration/core_set.py` |
| 2026-07-31 | Core Set v1已在云端用冻结tokenizer一次性生成并自校验：320组、1,280语义案例、5,120试题；状态`core_set_frozen_unrun`，Core Set digest `6ea2b6be…eb9d`、包digest `9659e286…1642`、payload root `1f4bd57f…c02a`。正式实验未授权、未运行、未观察结果；下一步只持久化冻结文件 | 云端`preregistration/exp001/core_set_v1/manifest.json` |
| 2026-07-31 | 冻结Core Set文件大小复核完成：`core_set.json`与整个`core_set_v1`目录均约11MB，适合普通Git且远低于GitHub 100MB单文件限制；不采用LFS、不压缩、不重新生成。为避免桌面文档提交与云端数据提交分叉，固定顺序为桌面先push、云端再pull/add/commit/push | 云端`ls -lh`、`du -sh`、`git status --short` |
| 2026-07-31 | Core Set v1已通过提交`ffd79ae`推送GitHub并快进同步到本机；3个锁定文件的SHA-256全部与冻结manifest一致，持久化步骤闭合。状态继续为`core_set_frozen_unrun`，正式实验仍未授权、未运行、未观察结果 | `preregistration/exp001/core_set_v1/manifest.json`、Git提交`ffd79ae` |
| 2026-08-03 | 完成Impl-5b-a非推理预检和独立正式授权锁：冻结包、资产、环境、源码与资源门统一进入稳定preflight digest；旧Core Set授权不能越权，未来正式授权必须精确绑定该digest。计划规模明确为40,960个trial-condition单元；111项测试通过，未加载模型、未评分Core Set、未观察正式结果 | `src/psa/confirmatory/preflight.py`、`scripts/preflight_exp001_confirmatory_run.sh`、`schemas/exp001_confirmatory_run_authorization.schema.json` |
| 2026-08-03 | 新主机Impl-5b-a预检完整通过：`valid=true`、失败检查为空，冻结资产、环境、Git与源码均吻合；digest为`fc6c5ccc…d9a7`。模型未加载、Core Set未评分、正式实验未授权。该digest只作为runner开发前主机基线，runner代码提交后必须重跑，禁止提前授权 | 云端`results/development/impl5b_confirmatory_preflight/preflight.json` |
| 2026-08-03 | 完成Impl-5b-b本地runner开发：8条件显式路由、每组4个历史state、真实safetensors恢复、确定性matched random、按组原子写入、断点只补缺组和篡改拒绝均已实现；固定开发入口只构造16条非Core题并产生128条原始条件记录，不计算准确率或中间结论。122项本地测试与编译检查通过；Core Set仍未读取，旧preflight digest随源码变化失效 | `src/psa/confirmatory/runner.py`、`src/psa/confirmatory/rwkv_backend.py`、`scripts/run_impl5b_confirmatory_runner_development_gate.sh` |
| 2026-08-03 | 云端Impl-5b-b非Core真实模型开发门通过：1组16题覆盖8条件，共128条原始记录，约32.84秒，峰值显存6,311,951,360字节；无准确率、无中间决策、无Core Set推理。随后新版preflight全部检查通过，runner证据有效，digest=`d41d735c…74f5`，Git=`c1171d2`。正式字段仍全为false；由于正式执行锁尚未实现，该digest不提前用于授权 | 云端`results/development/impl5b_confirmatory_runner_dev/summary.json`、`results/development/impl5b_confirmatory_preflight/preflight.json` |
| 2026-08-03 | 完成Impl-5b-c正式执行锁本地实现：模型加载前现场重建并匹配preflight，授权必须精确绑定全部冻结digest且字段不可扩展；正式入口固定全量320组/40,960条记录，无子集模式。中断必须显式恢复，完整结束后禁止重跑，非空新输出目录和已登记文件篡改均拒绝；完成包只含原始记录账本，不含准确率或中间决策。127项测试与编译检查通过，尚未运行Core Set，需提交后重跑云端preflight | `src/psa/confirmatory/formal.py`、`scripts/run_exp001_confirmatory.sh`、`tests/test_confirmatory_formal.py` |
| 2026-08-03 | 正式执行锁提交后的云端最终preflight通过：`valid=true`、runner证据有效、失败检查为空，最终待授权digest=`9a22a0cf…bd2f`；正式授权、运行和结果观察仍全部为false。进入项目负责人独立授权门，不创建授权记录、不运行Core Set | 云端`results/development/impl5b_confirmatory_preflight/preflight.json` |
| 2026-08-03 | 项目负责人逐字确认最终preflight digest `9a22a0cf…bd2f`并明确授权冻结的EXP-001完整确认性实验：固定2.9B模型、320组、5,120试题、8条件/40,960单元，只在全量完成和完整性验证后观察结果；不修改冻结设计、不在完成后自动重跑。尚未启动模型，下一步只在云端忽略目录生成授权记录并通过正式启动锁 | 项目负责人授权原文 |
| 2026-08-03 | EXP-001完整确认性运行已在云端后台启动；bash和Python runner进程存活，Python高负载。首次`waiting 0/320`由监控终端停在`~`并使用相对路径导致，正式输出目录没有改变。未读取原始分数、未计算中间准确率、未重复启动 | 云端进程检查 |
| 2026-08-03 | EXP-001原始确认性运行完整结束：320/320组、40,960条记录、`valid=true`，payload digest=`db4ba70e…5ba7`，峰值显存6,391,454,720字节；runner未派生准确率、未作中间决策、未标记结果已观察。新增独立只读完整性验证器，在统计分析前复核授权链、文件集合、逐组SHA-256、结构覆盖和总payload digest；全套130项测试通过 | 云端`completion.json`、`src/psa/confirmatory/verification.py`、`scripts/verify_exp001_confirmatory_raw.sh` |
| 2026-08-03 | EXP-001原始包独立完整性验证通过：320/320组、40,960条记录逐项有效，失败检查与失败组均为0，总payload digest仍为`db4ba70e…5ba7`；状态固定为`raw_package_verified_unanalyzed`，确认结果仍未观察。下一步先冻结只读统计分析器并用非Core合成分数测试，再首次读取真实分数 | 云端`results/confirmatory/exp001_v1.raw_verification.json` |
| 2026-08-03 | 冻结EXP-001只读分析计划和唯一入口：主要终点固定为continuous条件的E1/E2/E3，先做四代码语义边缘化，再按320个group运行10,000次BCa、100,000次单侧符号翻转和Holm校正；同时固定reset/random、restore、swap、prompt上限、specificity与单变量0.50上限。分析配置digest=`d97e0132…b8ea`；4项非Core合成测试与全套134项回归测试通过，尚未读取真实分数。审计同时确认最终原始包没有matched-context、每条件同步通用控制和自由生成格式读出，三项必须标记不可评估，不能用开发数据代替或授予完整Gate 2/Gate 4 Go | `configs/analysis/exp001_confirmatory_v1.json`、`docs/exp001_confirmatory_analysis_plan.md`、`src/psa/confirmatory/analysis.py`、`scripts/analyze_exp001_confirmatory.sh` |
| 2026-08-03 | EXP-001冻结只读分析在云端完整结束：320组、40,960条记录、配置与原始payload digest均吻合，分析包digest=`a0032f08…95ff`，正式结果现已观察。所有已测决定均通过：E1–E3、字段特异性、reset/random优势及联合绑定要求为true，Gate 3=`go`；但因正式Core Set未采集matched-context、每条件同步通用控制和自由生成格式读出，Gate 2与Gate 4依法标记`not_assessable_no_full_go`，不自动重跑。当前只确认门状态，下一步读取冻结完整报告中的点估计、区间与p值并形成结果解释 | 云端`results/confirmatory/exp001_v1_analysis/summary.json`，分析包digest `a0032f0813a3f1c524e0743c2a6feb0c7b2f71aa610782697a4a24614da495ff` |
| 2026-08-03 | EXP-001完整数值结果审阅完成并形成正式结果文档：E1=4.5255、E2=3.7965、E3=1.2264，95%区间均远离0且显著超过0.5 SESOI，三项Holm p均约`3.0e-5`；continuous联合准确率93.83%，identity/goal特异性、reset/random优势、restore和三种swap迁移全部通过。结果支持“原生recurrent state在本任务中具有强、特异、可恢复且因果迁移的联合行为作用”，但不改变Gate 2/Gate 4因控制缺项不可完整评估的状态。项目计划更新到Phase 2结果报告与缺口闭合决策 | `docs/exp001_confirmatory_results.md`、云端`confirmatory_report.json` |
| 2026-08-03 | 完成EXP-001B补充控制设计草案：只补matched-context、每条件同步通用能力和正式生成格式三类缺口，固定新增5,120+5,120+768=11,008条记录；原EXP-001仅作不可修改的配对参照，不重跑主要条件、不重估E1–E3、不声称独立复制。配置明确4个无绑定模板、原D5控制manifest、公开派生新seed、非Core norm校准、两级开发门和独立授权边界；7项专项测试及全项目141项测试通过。当前未冻结、未生成补充集、未授权、未运行 | `docs/exp001b_supplemental_design.md`、`configs/preregistration/exp001b_supplemental_controls.draft.json`、`tests/test_exp001b_design.py` |
| 2026-08-03 | 项目负责人确认EXP-001B的B1–B7，确认范围仅为设计及非Core开发门，不是预注册checksum确认，不授权生成补充测试集或正式运行。完成B-Dev1/B-Dev2本地实现：64案例精确token配对、96组件最近秩99.9% RMS开发阈值、8条件非Core runner、matched-context现场评分、greedy格式探针和state norm报警路径均已纳入；随后含overlay修复回归在内累计15项专项测试及全项目149项测试通过。下一步先云端重跑B-Dev1，成功后才运行B-Dev2 | `src/psa/supplemental/development.py`、`scripts/run_exp001b_bdev1_gate.sh`、`scripts/run_exp001b_bdev2_gate.sh`、`tests/test_exp001b_development.py` |
| 2026-08-04 | B-Dev1首次云端启动在读取`filler_protocol`时失败。诊断确认正式v3 holdout是继承v1的overlay，开发门误把差异文件当完整配置读取；失败早于模型推理和任何报告生成，未接触Core或正式补充数据。修复为复用既有`_load_formal_config`深度合并解析器，并新增“原文件无filler、解析后必须有131-token filler协议”的回归测试；保留本次Revise记录，更新后重跑同一B-Dev1入口 | 云端错误`'filler_protocol'`、`src/psa/supplemental/development.py`、`tests/test_exp001b_development.py` |
| 2026-08-04 | B-Dev1修复版云端完整通过：64个非Core matched-context案例全部通过精确token、filler保留和无绑定校验；64个真实非Core state形成96组件最近秩99.9% RMS阈值且全部有效。加载约5.40秒，总耗时约43.68秒，峰值显存6,232,199,168字节；`core_set_accessed=false`，补充集生成、实验授权、运行和结果观察均为false。路线进入B-Dev2非Core runner | 云端`results/development/exp001b_bdev1_non_core_calibration/summary.json`，design SHA-256 `f6a35d7f…f159b` |
| 2026-08-04 | B-Dev2非Core运行完整但综合门`valid=false`：B-Dev1证据链、`prompt_visible_reset→prompt_visible`别名、8条件128条runner、16条matched-context和强制前缀率1.0均通过；生成格式率只有0.875（14/16），state norm探针失败。峰值显存6,435,030,016字节；安全字段全部为false且Core未访问。保留Revise结果，不进入预注册候选、不修改阈值、不自动重跑，先只读审计两个失败报告 | 云端`results/development/exp001b_bdev2_non_core_runner/summary.json` |
| 2026-08-04 | B-Dev2失败明细审计完成：两条格式失败均生成`<tool_call`；四个state分别有66/67/66/67个组件越界，最大约1.209倍，是系统性分布差异而非FP16毛刺。代码审计确认norm阈值由131-token正式history族校准，v0.1却拿通用runner短历史比较；格式探针也未覆盖正式四模板族。新增独立v0.2：保留runner/matched不变，以64条正式形状非Core题平衡4 history、4 query、4 filler和A–D，原0.99格式及norm阈值不变，写入新目录保留v0.1失败；探针清单和B-Dev1证据哈希均持久化，专项16项/全项目150项测试通过 | 云端`generation_probe.json`、`state_norm_probe.json`；`src/psa/supplemental/development.py` |
| 2026-08-04 | B-Dev2 v0.2云端完整通过：B-Dev1三类证据哈希、条件别名、128条8条件runner、16条matched-context、64条正式形状生成和64个state norm检查全部有效；强制前缀率和格式率均为1.0，formal probe manifest digest=`4a4a2700…34acf`。加载约5.42秒、总耗时约133.57秒、峰值显存6,435,030,016字节；Core未访问，补充集未生成，实验授权/运行/结果观察均为false。进入未确认预注册候选构建，不直接生成正式数据 | 云端`results/development/exp001b_bdev2_non_core_runner_v02/summary.json` |
| 2026-08-04 | EXP-001B未确认预注册候选构建器完成：候选同时锁定B1–B7设计、B-Dev1、B-Dev2 v0.1失败和v0.2成功证据，复制自包含证据快照并计算source/evidence payload root与candidate checksum；验证器会拒绝源码、证据和安全字段篡改。候选安全状态固定为未确认、未生成补充集、未授权/未运行/未观察，不能借此启动正式实验；新增4项专项测试后全项目154项通过 | `src/psa/supplemental/freeze.py`、`scripts/build_exp001b_preregistration_candidate.sh`、`tests/test_exp001b_freeze.py` |
| 2026-08-04 | EXP-001B未确认预注册候选已在云端生成：`valid=true`且可进入人工审阅，固定模型2.9B、设计digest=`f6a35d7f…f159b`；锁定12个源码与22个开发证据文件，candidate digest=`c7a69971…a3eb`、payload root=`a99c054d…98a4`。安全边界保持候选未确认、Core未访问、补充集未生成且未获生成授权、正式实验未授权/未运行/未观察。下一步只读核对完整verification，不直接请求确认或生成数据 | 云端`results/development/exp001b_preregistration_candidate_v1/summary.json` |
| 2026-08-04 | EXP-001B候选完整技术审阅通过：candidate self digest、payload root与安全边界均有效，12/12源码和22/22开发证据文件校验通过，失败列表为空；完整candidate digest=`c7a6997179072db22bb518289cd1ab0e2428f8a9eb6ea4dcc50983bbe212a3eb`。候选仍未确认，补充集未生成且未获生成授权，正式实验未授权/未运行/未观察。现可请求项目负责人确认checksum，但确认范围仅限升级最终预注册包 | 云端`preregistration_verification.json` |
| 2026-08-04 | 项目负责人逐字确认EXP-001B候选checksum `c7a6997179072db22bb518289cd1ab0e2428f8a9eb6ea4dcc50983bbe212a3eb`，只授权升级为最终预注册包，并明确暂不授权生成补充测试集、不授权运行正式实验。新增最终化与独立复核工具：原样封装候选、验证、确认及22份证据，精确拒绝扩大授权，检查包清单、payload root、锁定文件库存与安全边界；5项专项及全项目159项测试通过，尚未在云端生成最终包 | 项目负责人授权原文；`src/psa/supplemental/finalize.py`、`scripts/finalize_exp001b_preregistration.sh` |
| 2026-08-04 | EXP-001B最终预注册包已在云端冻结并由脚本内独立验证器自校验通过：状态`final_preregistration_frozen`，candidate digest保持`c7a69971…a3eb`，final digest=`976cce8c…23be`；25个包内文件、12个源码与22份证据数量一致。授权仅有升级最终预注册包=true；补充集生成、实验运行、结果观察及自动重跑均为false。下一步仅Git持久化，不生成数据 | 云端`preregistration/exp001b/final_v1/manifest.json` |
| 2026-08-04 | EXP-001B最终预注册包已通过云端提交`e9c3528`推送到远程仓库；远程差异严格只有`preregistration/exp001b/final_v1`的26个文件，没有`results/`、模型或其他改动。本机快进同步后再次运行独立验证器：manifest、package payload、candidate、stored verification、文件库存、包内容和安全边界全部为true，25/25锁定文件无失败，final digest仍为`976cce8c…23be`。该里程碑完成不授予补充集生成或实验运行权限 | Git提交`e9c3528`；本机独立复核输出 |
| 2026-08-04 | EXP-001B补充集生成基础设施完成：新增非推理预检、确定性11,008条记录构建、冻结包两层digest与独立验证器，并用“绑定云端实时预检digest的负责人授权文件+环境执行锁”双重阻断未授权或过期授权生成。生成计划固定5,120条matched-context、5,120条正式生成读出和96控制题×8条件=768条；精确复用D5控制manifest `30d984fc…e348`，控制源组无放回选择且四状态各24条。新增10项专项测试，全项目169项回归通过；本轮没有运行真实预检、没有读取Core内容来构造正式记录、没有加载模型、没有生成补充集，也没有授权或运行正式补充实验 | `src/psa/supplemental/set_generation.py`、3个云端脚本、授权Schema、`tests/test_exp001b_set_generation.py` |
| 2026-08-04 | EXP-001B首次补充集生成前云端预检通过：最终预注册包与digest、父Core Set包与digest、冻结manifest边界及11,008条记录预算检查均为true；`valid=true`，preflight digest=`1f117d5cd0e9d37706c50bc10db37bf826eaa7166a366589e8ab4121499cfa65`。预检明确记录模型未加载、补充trial未评分、补充集未生成，生成授权、实验授权、运行和结果观察仍全部为false。随后在请求不可逆授权前发现该digest没有覆盖生成器源码，因此本次checksum保留为首次技术记录但不用于负责人授权 | 云端`results/development/exp001b_set_preflight/preflight.json`；授权前安全审计 |
| 2026-08-04 | 加固EXP-001B生成预检：新增12个生成相关源码、配置、模型tokenizer配置、授权Schema和启动脚本的SHA-256清单，并把清单纳入preflight digest；同时把容易误解的“父Core未运行”检查改名为“父冻结manifest边界完整”，明确它检查的是不可变包字段而不是否认EXP-001已经执行。现在任何生成逻辑或关键配置变化都会令旧授权失效；专项测试增至11项、全项目170项通过，等待云端拉取后重新预检 | `src/psa/supplemental/set_generation.py`、`tests/test_exp001b_set_generation.py` |
| 2026-08-04 | EXP-001B加固版生成前云端预检通过：最终预注册包、父Core Set包、固定digest、父冻结manifest边界、11,008条记录预算和12文件生成源码清单共7项检查全部为true；`valid=true`，新preflight digest=`dbb17d975d32956fb92ab39975452f00c6bc1ca8b4114afec84f5a46f8242ae6`。旧`1f117d5c…fa65`不再用于授权。模型未加载、补充trial未评分、补充集未生成，生成授权、实验授权、运行与结果观察仍全部为false；现只进入项目负责人生成授权门 | 云端`results/development/exp001b_set_preflight/preflight.json` |
| 2026-08-04 | 项目负责人逐字确认EXP-001B加固版生成预检checksum `dbb17d975d32956fb92ab39975452f00c6bc1ca8b4114afec84f5a46f8242ae6`及最终预注册checksum `976cce8c…23be`，授权生成并冻结固定11,008条补充记录；授权明确排除正式补充实验运行。该权限只允许创建`supplemental_set_frozen_unrun`包，不允许加载模型评分、观察结果或自动运行；下一步在云端建立机器可验证授权文件后执行一次生成与独立校验 | 项目负责人授权原文 |
| 2026-08-04 | EXP-001B补充集已在云端确定性生成、冻结并通过独立验证：5,120条matched-context、5,120条正式生成读出、96个控制trial×8条件=768条，总计11,008条；4个锁定文件无失败，`content_valid=true`、`valid=true`，set digest=`7c3606be819d4e6cc5420f0bf36efd1906f8954d362d83e912785cc943565d33`、package digest=`68e9a9a79fe4e493a0c64ba8c0278c300cc832d940ab902feaceb4ad7f9d5954`，目录约28MB。状态严格为`supplemental_set_frozen_unrun`，实验授权、运行和结果观察全为false；下一步只Git持久化 | 云端`preregistration/exp001b/supplemental_set_v1/manifest.json`及独立验证输出 |
| 2026-08-04 | EXP-001B补充集冻结包已通过云端提交`62dd8b2`推送到远程仓库，差异严格只有`preregistration/exp001b/supplemental_set_v1`的5个文件；本机快进同步后重新运行独立验证器，11,008条记录、锁定文件、内容结构和两层digest全部通过，set/package digest仍为`7c3606be…65d33`/`68e9a9a7…d5954`。工作树干净，实验授权、运行和结果观察仍为false；下一步进入正式运行基础设施开发，不直接运行 | Git提交`62dd8b2`；本机独立复核输出 |
| 2026-08-04 | EXP-001B正式运行基础设施已完成本地实现：11,008条冻结记录严格映射到320组（224×32、96×40），三类记录分别走matched-context state、正式生成和8条件通用控制路由；新增按组原子写入、显式断点续跑、完成后拒绝重跑、逐record ID原始包验证及独立负责人授权锁。正式预检绑定最终预注册包、父Core Set、补充集、模型资产、运行源码、非Core真实模型门证据和主机环境；运行期间不计算准确率或中间决策。新增5项专项测试，全项目175项通过。当前没有正式授权、没有加载冻结记录评分、没有运行或观察结果；下一步只在云端运行非Core开发门，再运行只读预检 | `src/psa/supplemental/formal_run.py`、`rwkv_run_backend.py`、4个脚本、授权Schema与专项测试 |
| 2026-08-04 | EXP-001B正式runner非Core云端开发门通过：真实2.9B模型加载约7.64秒、40条路由运行约18.66秒、峰值显存6,311,951,360字节；16条matched-context、16条正式生成和8条通用控制完整覆盖三类记录与全部8条件，`valid=true`。运行器保存原始记录但不计算准确率或中间决策；`core_set_accessed=false`、`supplemental_set_accessed=false`、`formal_authorization_used=false`，正式实验未运行、结果未观察。22个运行相关源码/配置/脚本digest已写入证据；下一步只运行不加载模型的正式预检 | 云端`results/development/exp001b_formal_runner_dev/summary.json` |
| 2026-08-04 | EXP-001B首次正式运行预检安全失败：开发门证据仍`valid=true`，但唯一失败检查为`runtime_flags_frozen`，因此总状态`preflight_failed`并停在`hold_and_repair_without_formal_inference`；模型未加载、补充trial未评分、授权/运行/结果观察均为false。审计发现新开发门、预检和正式运行脚本漏掉了其他GPU入口统一使用的三个RWKV环境标志。已补齐自包含标志和`PYTHONPATH`并新增入口回归测试，全项目176项通过；由于脚本在runner哈希清单中，旧开发证据和失败digest `3d48b04b…0365`不得复用，下一步按顺序重跑非Core门与预检 | 云端失败预检；`scripts/*exp001b*`修复 |
| 2026-08-04 | EXP-001B运行标志修复完成云端复验：在固定提交`defd6b0`上，真实2.9B非Core正式runner开发门再次`valid=true`，40条夹具覆盖三类记录与8个控制条件，冻结Core和补充集均未访问；随后最终只读preflight所有检查为true，失败项为空，runner证据有效，待授权digest=`bc91b2b3cd7557cc2000d8990fbc638de38a6d9895891e8b5b944e9572ab13a6`。预检未加载模型、未评分补充trial，正式授权/运行/结果观察仍为false；现在只进入项目负责人独立运行授权门，旧失败digest永久无效 | 云端`results/development/exp001b_formal_runner_dev/summary.json`、`results/development/exp001b_run_preflight/preflight.json` |
| 2026-08-04 | 项目负责人逐字确认EXP-001B最终preflight digest `bc91b2b3cd7557cc2000d8990fbc638de38a6d9895891e8b5b944e9572ab13a6`并授权冻结的完整补充确认性实验：固定`rwkv7-g1h-2.9b-20260710`、320组、11,008条记录（5,120 matched-context、5,120正式生成、96×8控制）；只允许全量完成并通过原始包完整性验证后观察结果，明确禁止修改冻结设计、自动重跑和重跑EXP-001。当前仅完成自然语言授权，尚未创建机器授权文件、启动模型或观察结果 | 项目负责人授权原文 |
| 2026-08-11 | EXP-001C v02 Stage A prompt-visible 非Core正控制在一次性机器授权下完成：执行提交`94847da`，manifest/preflight/authorization digest分别为`6874ea63…0909`、`0a9dfbf2…0a52`、`9f8dd3d9…2e08`；32条记录中28条正确，label-marginalized accuracy=0.875，预测A/B/C/D为9/9/8/6，最大单一代码占比0.28125，prefix greedy与roundtrip均为1.0，所有预设门槛通过。Stage B和正式测试集均未访问，无正式决策且禁止自动重跑；下一步只进入Stage B独立授权审查 | 远程`results/development/exp001c_v02_stage_a_pilot_v01/summary.json`；`docs/exp001c_v02_stage_a_pilot_v01_observation.md` |
| 2026-08-11 | 修正进度表同步遗漏：顶部“最后更新”由2026-08-04更新为2026-08-11，并把“当前节点”“研究状态”“当前所在位置”“当前下一步”同步到EXP-001C v02 Stage A已通过、Stage B待独立设计与授权的真实状态；强化强制规则，明确每轮操作结束前都必须检查并更新这些位置，不得只追加文末记录 | `PROJECT_PROGRESS.md` |
| 2026-08-11 | EXP-001C v02 Stage B第一版离线设计与风险审查完成：Stage A的32条已完成结果只作外部基线，Stage B不重跑Stage A；固定continuous/restored/三种swap/reset/random七条件×32条=224条。交换条件按实际注入状态重新映射唯一正确代码，reset/random不设状态语义正确项。新增draft config、两个Schema、确定性manifest builder/verifier、风险审查和5项专项测试；全项目234项通过。本轮未加载模型、未访问正式测试集，Stage B执行/观察、正式运行与自动重跑均未授权 | `docs/exp001c_v02_stage_b_risk_review.md`；`src/psa/development/exp001c_v02_stage_b_design.py`；`tests/test_exp001c_v02_stage_b_design.py` |
| 2026-08-11 | EXP-001C v02 Stage B纯离线runner/backend契约完成：只接受`offline_fake_adapter=true`且`model_loaded=false`的adapter，完整路由七条件各32条并原子写入224条synthetic结果；所有输出固定`synthetic_output_not_research_evidence=true`、`model_executed=false`。缺少离线锁、非fake或已加载adapter、非空输出目录及真实模型入口均失败关闭；Stage B专项12项、全项目241项通过。本轮未加载模型、未访问正式测试集，也未创建执行授权 | `src/psa/development/exp001c_v02_stage_b_offline.py`；`schemas/exp001c_v02_stage_b_offline_result.schema.json`；`tests/test_exp001c_v02_stage_b_offline.py` |
| 2026-08-11 | EXP-001C v02 Stage B真实RWKV backend纯代码集成完成：从Stage A同源protocol重建32条trial与8个唯一history state，按两个block各自执行4状态disk roundtrip，并为8个源状态生成唯一deterministic matched-random；224条路由覆盖continuous/restored/三种swap/reset/random，交换来源和重映射target逐条一致。新增真实结果Schema与5项fake-RWKV测试，Stage B专项17项、全项目246项通过；缺少执行授权、错误model路径和protocol digest均在评分/加载前拒绝。本轮未调用真实模型工厂、未执行模型或访问正式测试集 | `src/psa/development/exp001c_v02_stage_b_rwkv.py`；`schemas/exp001c_v02_stage_b_result.schema.json`；`tests/test_exp001c_v02_stage_b_rwkv.py` |
| 2026-08-11 | EXP-001C v02 Stage B执行runner安全外壳完成：live authority validator默认关闭；通过验证后先独占创建single-use execution claim，再调用backend并原子写入原始结果、独立verification和summary。无效backend结果仍消费claim，后续自动重入由非空目录拒绝；独立验证逐条核对224条route、scores、prefix和target边界，但固定`contains_derived_accuracy=false`、`contains_research_decision=false`。新增execution claim Schema与6项runner测试，Stage B专项23项、全项目252项通过；本轮只用fake backend，未调用模型工厂或访问正式测试集 | `src/psa/development/exp001c_v02_stage_b_run.py`；`schemas/exp001c_v02_stage_b_execution_claim.schema.json`；`tests/test_exp001c_v02_stage_b_run.py` |
| 2026-08-11 | EXP-001C v02 Stage B只读live preflight与机器授权锁本地完成：预检在模型加载前绑定干净main提交、Stage B设计/protocol、Stage A原始结果与摘要、模型配置和资产哈希、主机环境、224条计划及空输出目录；固定记录`model_loaded=false`、`model_executed=false`及全部未授权边界。未来授权builder只接受一条固定逐字文本，并把授权绑定到design/preflight/Stage A digest；普通“继续”必定失败。新增6项测试，Stage B专项29项、全项目258项通过；当前未创建授权文件、未执行模型或访问正式测试集，下一步只允许云端只读预检 | `src/psa/development/exp001c_v02_stage_b_preflight.py`；`scripts/build_exp001c_v02_stage_b_preflight.py`；`schemas/exp001c_v02_stage_b_preflight.schema.json` |
| 2026-08-11 | EXP-001C v02 Stage B云端只读preflight流程通过：服务器快进到本轮代码提交，29项Stage B测试通过；首次预检所有检查为true且失败项为空，明确`model_loaded=false`、`model_executed=false`、执行/观察授权均为false。因本条进度更新会产生新的Git提交，首次digest只作流程证据；服务器必须在本轮最终文档提交上生成v02证据，后续授权只能绑定v02 digest。当前仍无授权文件、模型推理或Stage A重跑 | 远程`results/development/exp001c_v02_stage_b_preflight_v01/preflight.json`；最终以同路径版本v02证据为准 |
| 2026-08-11 | 项目负责人逐字授权EXP-001C v02 Stage B recurrent-state非Core 224条pilot及本轮结果观察；授权明确排除Stage A重跑、正式测试集、正式运行、确认性决定和自动重跑。新增独占创建机器授权文件并调用single-use runner的云端入口；缺少固定环境锁、已有授权文件或非空输出目录均在模型加载前拒绝。下一步先在最终提交上生成preflight_v03，再单次启动；本条授权不允许失败后自动重试 | 项目负责人授权原文；`scripts/run_exp001c_v02_stage_b.py` |
| 2026-08-11 | EXP-001C v02 Stage B单次原始运行完成：224条、7条件、`valid=true`，recurrent state已访问，single-use claim已消费；结果SHA-256=`e0b871e7…8b655`，Stage A未重跑、正式测试集未访问、无正式运行或确认性决定。冻结后续只读观察口径：五个状态语义条件做四代码轮换边际化，reset/random仅作诊断，不设事后阈值；当前尚未读取派生指标 | 远程`results/development/exp001c_v02_stage_b_pilot_v01/summary.json`；`configs/analysis/exp001c_v02_stage_b_observation_v01.json` |
| 2026-08-11 | EXP-001C v02 Stage B冻结只读观察完成：五个状态语义条件均为轮换边际化联合7/8、domain 8/8、operation 7/8，平均目标margin约1.627；continuous/restored语义预测8/8一致，三种swap均7/8跟随实际注入state。唯一错误始终是`indigo+harbor`读成`indigo+spiral`，并随state交换移动到不同query，非固定题目错误。reset/random与原Stage A参考字段均仅2/8；本轮不设阈值、不作确认性决定、不重跑。该结果支持进入Self Model v0.1工程原型设计，但不能替代EXP-001B控制闭合 | `docs/exp001c_v02_stage_b_pilot_v01_observation.md`；远程`results/development/exp001c_v02_stage_b_observation_v01/observation.json` |
| 2026-08-11 | Phase 3 Self Model v0.1纯离线接口完成：新增六字段静态Self State Schema、typed item/update-class约束、payload checksum和独占不可变SelfStore；fake Encoder支持确定性字段向量与mask，禁止自然语言Prompt序列化；fake gated residual支持off、scale、layer mask，字段swap保持来源不变，encoded random按字段/seed复现且L2 norm匹配。离线manifest锁定配置、Schema、设计、源码和测试；9项专项、全项目270项通过。真实RWKV hook、模型加载、训练和Self效果实验均未实现/未授权 | `docs/self_model_v0_1_design.md`；`src/psa/self_model/`；`tests/test_self_model_v0_1.py` |
| 2026-08-11 | Self Model v0.1纯离线接口完成服务器复验：提交`fdf626a`上9项专项测试通过，服务器确定性重建并验证配置、两个Schema、设计文档、脚本、五个模块和测试的完整source digest清单；manifest `valid=true`、digest=`694d8e2e…84ca6`。证据明确`model_loaded=false`、`model_executed=false`、`real_rwkv_coupling_implemented=false`；下一步只进入真实RWKV接口调查确认门，不直接实现或运行 | 远程`results/development/self_model_v0_1_offline/manifest.json` |
| 2026-08-11 | Self Model v0.1真实RWKV coupling接口静态调查本地完成：固定审计`rwkv==0.8.32`及`model.py` digest，确认项目配置会把公开`RWKV`指向`RWKV_x070`；该类虽在JIT关闭时继承`nn.Module`，但block权重和运算都在映射及`forward_one/forward_seq`循环中，没有可直接使用的逐block模块hook。单token/序列残差shape分别为`[2560]`/`[T,2560]`，传入state会原位更新每层3组件。post-FFN residual被列为最小原型优先接口族，但具体层未选；3项专项及全项目273项测试通过，等待云端只读审计 | `docs/self_model_v0_1_rwkv_coupling_audit.md`；`scripts/audit_self_model_v0_1_rwkv_interface.py` |
| 2026-08-11 | 服务器在提交`3b030db`上完成真实RWKV coupling只读静态审计：先启用`network_turbo`并快进拉取main，随后3项专项测试通过；固定包版本、`model.py` digest、RWKV-7别名分支、单token/序列路径和96个预期state组件的独立断言全部有效。报告`valid=true`、digest=`243b86da…0d29`，安全字段明确`rwkv_model_imported=false`、`torch_imported=false`、`weights_accessed=false`、`model_loaded=false`、`model_executed=false`、`real_hook_implemented=false`、`final_layers_selected=false`。本轮到此停止，下一轮需重新确认 | 远程`results/development/self_model_v0_1_rwkv_interface_audit/audit.json` |
| 2026-08-11 | Self Model v0.1无权重双路径残差回调壳本地完成：新增纯Python3层×4维fake runtime，`forward_one`与`forward_seq`共享`post_ffn_residual`回调协议；off与scale=0完全绕过callback并与无callback基线逐位一致，active路径保留shape/dtype/device，输入state在运行前clone且callback请求不暴露recurrent state。fake `broadcast_all_tokens`和`fake-layer-01`均明确不冻结真实策略/层。8项专项及全项目281项测试通过；没有导入模型、访问权重、修改`site-packages`或运行效果实验，等待云端复验 | `docs/self_model_v0_1_fake_callback_contract.md`；`src/psa/self_model/fake_callback_runtime.py` |
| 2026-08-11 | 无权重双路径残差回调壳在服务器提交`1f3064e`上复验通过：旧离线manifest、RWKV静态审计和fake callback共12项跨阶段测试通过；报告`valid=true`、digest=`42717dac…a456`。独立重算确认16项检查、6个锁定源码digest和`forward_one/forward_seq`两条active路径全部有效；9个安全字段全部为false，包括未导入`rwkv.model`/`torch`、未访问权重、未加载/执行模型、未修改`site-packages`、未实现真实hook或选择真实层、未运行Self效果实验。本轮到此停止 | 远程`results/development/self_model_v0_1_fake_callback/report.json` |
| 2026-08-11 | 真实2.9B coupling-off adapter设计本地完成：固定项目内未来路径并禁止修改`site-packages`，设置OFF-G1直接委托原始forward与OFF-G2双路径instrumented-off两级门；未来模型门要求同进程、同shape预热1次、单token/序列、None/restored state、`full_output=false/true`，logits及全部state必须`torch.equal`，失败即停止且不能换容差/top-1。D1–D5授权分离，当前仅D1；真实adapter目标文件不存在，active、模型执行、真实层和效果实验均为false。8项专项及全项目289项通过，等待云端只读设计复验 | `docs/self_model_v0_1_real_adapter_off_design.md`；`configs/development/self_model_v0_1_real_adapter_off_design.draft.json` |
| 2026-08-11 | 真实2.9B coupling-off adapter设计在服务器提交`f72e0cf`上复验通过：19项Phase 3组合测试通过；报告`valid=true`、digest=`dbd64e9a…f585`。独立重算确认23项设计检查、9个锁定文件digest、OFF-G1/OFF-G2顺序及未来adapter文件缺失全部有效；11个安全字段全为false，包括未导入模型/torch、未访问权重、未加载/执行模型、未修改`site-packages`、未实现adapter/active、未选真实层、未运行效果实验和未授权自动重跑。本轮到此停止 | 远程`results/development/self_model_v0_1_real_adapter_off_design/report.json` |
| 2026-08-11 | D2 off-only adapter本地实现完成：新增项目内`RWKV7CouplingOffAdapter`，固定上游版本/digest后仅执行`base_model.forward(tokens,state,full_output)`；默认或精确off请求可用，非off对象、子类伪装、active方法和source lock错误均在底层调用前失败。fake base验证tokens/state/full-output与返回对象identity、上游原位state变更和异常传播不被改写；静态AST确认无`rwkv`/`torch`/`importlib`、无projection或post-FFN instrumentation。D2报告29项检查与8个源码digest有效，OFF-G1=true、OFF-G2=false；8项专项及全项目297项通过，等待服务器同一fake复跑 | `src/psa/self_model/rwkv7_coupling_adapter.py`；`docs/self_model_v0_1_off_only_adapter.md` |
| 2026-08-11 | D2 off-only adapter在服务器提交`8fbfb8a`上完成纯fake-base复跑：8项专项通过，报告`valid=true`、digest=`527fc6ed…2e6c`；独立重算确认29项检查、8个锁定源码digest和OFF-G1=true/OFF-G2=false。安全记录明确off-only adapter已实现，但`installed_rwkv_source_probed=false`、`rwkv_model_imported=false`、`torch_imported=false`、权重/模型执行/site-packages修改/OFF-G2/active/真实层/效果实验/自动重跑均为false，因此本轮不算D3。本轮到此停止 | 远程`results/development/self_model_v0_1_off_only_adapter/report.json` |
| 2026-08-11 | D3服务器无模型静态复验工具本地完成：冻结`rwkv==0.8.32`、`model.py` digest/大小、D2报告digest和wrapper digest；探针只通过包元数据定位并读取源码字节，报告同时重算D2的29项检查/8个源码digest并执行wrapper AST审计。新增6项专项、全项目303项测试通过，33项报告检查与10个源码digest有效；当前尚未生成服务器D3证据，模型导入/权重/执行、OFF-G2、active和真实层继续关闭 | `src/psa/self_model/d3_static_verification.py`；`docs/self_model_v0_1_d3_static_verification.md` |
| 2026-08-11 | D3在服务器提交`16cb69d`上完成无模型静态复验：6项专项测试通过，已安装`rwkv==0.8.32`的`model.py`大小85,425字节且digest=`75482aee…05e0`；报告`valid=true`、digest=`fcb8dfeb…2918`。独立重算确认33项检查、10个源码digest及报告自digest全部有效；`installed_rwkv_source_probed=true`，但模型/torch导入、权重访问、模型加载/执行、site-packages修改、OFF-G2/active/真实层/效果实验/自动重跑均为false。本轮到此停止 | 远程`results/development/self_model_v0_1_d3_static_verification/report.json` |
| 2026-08-19 | D3B OFF-G2 instrumented-off项目内实现并推送：不复制或修改`site-packages`，而是对固定上游源码AST定位`RWKV_x070`两条forward方法，在每条CMix后的残差加法处要求恰好一个None-guarded post-FFN分支；运行时callback固定为None，临时方法在成功/异常后均恢复，active/非精确off请求失败关闭。13项专项及全项目316项本地测试通过，提交`4733f90`已在main；服务器SSH三次在握手前拒绝，端口诊断确认30587/TCP关闭，故尚未拉取或生成云端静态报告，也未加载/执行模型 | `src/psa/self_model/rwkv7_instrumented_off_runtime.py`；`docs/self_model_v0_1_instrumented_off_runtime.md`；`Test-NetConnection`诊断 |
| 2026-08-19 | D3B首次服务器静态门有效失败：服务器成功快进到`395d5e2`，13项fake专项全部通过；真实源码探针未导入模型，但AST变换器只在模块顶层寻找`RWKV_x070`，未覆盖上游feature guard内的类定义，抛出`RuntimeError`并未生成report。该失败不是模型结果，也没有权重/模型执行 | 项目负责人粘贴的远程终端输出；缺失的`results/development/self_model_v0_1_instrumented_off_runtime/report.json` |
| 2026-08-19 | D3B嵌套类AST修复本地完成：目标类搜索从模块顶层扩展为整棵AST，但仍要求恰好一个`RWKV_x070`；新增feature guard嵌套类回归测试，runtime新digest=`386f2410…1877`同步进冻结配置。14项专项及全项目317项通过，等待推送与服务器复验；首次失败保留 | `src/psa/self_model/rwkv7_instrumented_off_runtime.py`；`tests/test_self_model_instrumented_off_runtime.py` |
| 2026-08-19 | D3B第二次服务器静态门有效失败：嵌套类修复后的14项fake专项全部通过，静态探针也已找到目标类；但真实`forward_one/forward_seq`定义位于类体内部条件分支，v02只查类体第一层，因缺少`forward_one`抛出`RuntimeError`且无report。仍未导入模型、访问权重或执行模型 | 项目负责人粘贴的第二次远程终端输出；缺失的D3B report |
| 2026-08-19 | D3B类内条件方法修复本地完成：方法搜索扩展为`RWKV_x070`子树并继续要求两条方法各自唯一；新增类外与类内双层feature guard回归，runtime新digest=`ca8c1385…7572`同步进冻结配置。15项专项及全项目318项通过，等待推送与服务器第三次复验；两次失败均保留 | `src/psa/self_model/rwkv7_instrumented_off_runtime.py`；`tests/test_self_model_instrumented_off_runtime.py` |
| 2026-08-19 | D3B第三次服务器静态门有效失败并完成精确结构诊断：服务器在最新`d628643`上仍以缺少唯一`forward_one`失败；只读AST显示目标类唯一，但两条forward各有两个定义，分别位于`RWKV_DE_VERSION=="1"`的body与else。源码版本、85,425字节和digest仍匹配，诊断明确未导入模型/torch | 项目负责人粘贴的第三次错误及AST lineage/condition输出；仍缺失D3B report |
| 2026-08-19 | D3B DE双variant v04修复本地完成：两条路径的body/else版本都必须各有一个post-FFN注入点；冻结环境要求`RWKV_DE_VERSION`未设置并只选择else版，非空值在runtime构造阶段拒绝。配置与Schema增加variant证据，runtime新digest=`df4da07c…7cd2`；16项专项及全项目319项通过，等待服务器复验，前三次失败保留 | `src/psa/self_model/rwkv7_instrumented_off_runtime.py`；`schemas/self_model_v0_1_instrumented_off_report.schema.json` |
| 2026-08-19 | D3B第四次服务器静态门有效失败：v04能识别两套DE variant，但每variant注入点仍为0，因为真实函数名不是旧审计标记`RWKV_x070_CMix`。只读上下文确认单token两版均为`RWKV_x070_CMix_one`、序列两版均为`RWKV_x070_CMix_seq`，且四处都紧跟`x=x+xx`；无report、无模型/torch/权重执行 | 项目负责人粘贴的第四次错误及四方法CMix上下文 |
| 2026-08-19 | D3B路径专用CMix v05修复本地完成：`forward_one`只接受`RWKV_x070_CMix_one`，`forward_seq`只接受`RWKV_x070_CMix_seq`，错误交叉名称同样失败关闭；报告增至48项静态检查并记录路径映射，runtime新digest=`ce9862b6…aea5`。16项专项及全项目319项通过，等待服务器复验，前四次失败保留 | `src/psa/self_model/rwkv7_instrumented_off_runtime.py`；冻结配置与report Schema |
| 2026-08-19 | D3B OFF-G2服务器无模型静态门在提交`96b18cd`上通过：16项专项测试成功，报告`valid=true`、digest=`46f05bd8…e57a`；48项检查、10个源码digest和报告自digest独立重算有效。真实源码`forward_one`两版在306/336、`forward_seq`两版在426/458，每版均一个注入点，冻结环境选择else行336/458；`off_g2_implemented=true`但`real_model_equivalence_executed=false`，模型/torch/权重/active/Self projection/效果实验均未发生。本轮停止，四次先前失败永久保留 | 远程`results/development/self_model_v0_1_instrumented_off_runtime/report.json`；项目负责人终端输出 |
| 2026-08-19 | 项目负责人独立授权D4真实2.9B OFF等价门；本地完成single-use runner与冻结配置。矩阵固定原始baseline/OFF-G1/OFF-G2三路、单token/序列、None/克隆恢复态及序列两种`full_output`共6个计分单元，每route/cell预热1次；logits和全部state只接受shape/dtype/device一致及`torch.equal`。入口在模型加载前消费claim，失败同样禁止自动重跑；active/Self projection/效果实验/确认性决定全部关闭。6项新增测试、Phase 3组合24项及全项目325项通过；尚待推送后在服务器单次执行 | `src/psa/self_model/d4_real_off_equivalence.py`；`configs/development/self_model_v0_1_d4_real_off_equivalence.json`；`tests/test_self_model_d4_real_off_equivalence.py` |
| 2026-08-20 | D4一次性runner、冻结配置、说明、测试和进度记录已由提交`36f903d`推送GitHub main；`.env`继续由`.gitignore`排除。服务器尚未拉取或执行，因此single-use claim尚未消费、真实模型等价结论仍为空。下一步固定为服务器拉取最终进度提交、先跑24项无模型组合测试，再单次运行D4 | GitHub main提交`36f903d`；本地`git check-ignore -v .env` |
| 2026-08-20 | D4真实2.9B OFF等价门在服务器提交`a4d110c`上有效失败：24项组合测试通过，模型成功加载并完成6单元；OFF-G1全部逐位一致，OFF-G2仅`forward_one+state=None`失败，logits及92/96个state组件不等，其余5单元逐位一致。报告digest=`39d4611a…721a`在本机从粘贴原文独立重算一致；claim=`2900bf11…9de`已消费，运行约10.65秒、峰值显存6,129,678,336字节。保留失败，不自动重跑或改容差；进入离线调用顺序/预热/绑定边界审计，D5继续暂停 | 远程`results/development/self_model_v0_1_d4_real_off_equivalence_v01/report.json`；`docs/self_model_v0_1_d4_failure_observation.md` |
| 2026-08-20 | D4A失败诊断离线设计完成：审计确认D4固定None→恢复态单元顺序、原始→G1→G2路线顺序、预热输出丢弃且无同路线重复轨迹；OFF-G2同时存在AST重编译、decorator清空、globals复制和双方法临时绑定边界。未来最小诊断只用原失败token/state，比较原始、无注入重编译G0、OFF-G2，按3×3拉丁顺序执行9次且全部记录tensor digest/误差；它只定位原因，不能改写D4或授权D5。本地16项静态检查、4项新增测试和全项目329项通过，design report=`a6eb22d7…6f1c`；模型/torch/权重/runtime实现/执行全为false | `configs/development/self_model_v0_1_d4a_failure_diagnostic_design.json`；`docs/self_model_v0_1_d4a_failure_diagnostic_design.md`；`src/psa/self_model/d4a_failure_diagnostic_design.py` |
| 2026-08-20 | D4A fake-only诊断runtime完成：G0按与OFF-G2相同的variant选择、decorator清空、globals复制和双方法临时绑定执行未注入方法；9次平衡记录器保存每次logits/state digest并生成9个同路线、27个跨路线的精确与误差比较。fake覆盖全等、仅G2扰动分类、绑定恢复及active/source lock拒绝；8项新增测试、D4A组合12项和全项目337项通过。14项实现静态检查全真，digest=`b9f27cd2…4a08`；没有真实入口、installed-source探针、RWKV/Torch导入、权重、模型执行、claim或D5授权 | `src/psa/self_model/d4a_failure_diagnostic_runtime.py`；`configs/development/self_model_v0_1_d4a_failure_diagnostic_runtime.json`；`docs/self_model_v0_1_d4a_failure_diagnostic_runtime.md` |
| 2026-08-20 | D4A服务器无模型静态复验工具本地完成：只通过包元数据读取已安装`rwkv==0.8.32`锁定源码字节，要求G0/G2选择同一DE-unset else variant，核对真实原始`MyFunction` decorator、G0清除decorator及G2每variant单注入点，并记录源码/方法digest。4项新增静态测试、D4A组合16项和全项目341项通过；本机无目标installed-source，尚无真实静态报告。未导入RWKV/Torch，未访问权重、加载或执行模型，未创建真实诊断入口/claim，也未授权D5 | `src/psa/self_model/d4a_cloud_static_verification.py`；`scripts/verify_self_model_v0_1_d4a_cloud_static.py`；`configs/development/self_model_v0_1_d4a_cloud_static_verification.json` |
| 2026-08-20 | D4A服务器无模型静态复验在提交`9203aaf`上通过：16项组合测试成功，真实`rwkv==0.8.32`源码的版本/digest/大小、G0/G2同variant选择、原始`MyFunction` decorator、G0 decorator清除及G2每variant单注入点共25/25检查全真。报告`valid=true`、digest=`f8e74653…e57`，由贴回完整JSON独立复算一致。该结果不改变D4失败；模型/Torch未导入，权重/模型未访问或执行，真实入口/claim/active/D5均不存在 | 远程`results/development/self_model_v0_1_d4a_cloud_static_verification/report.json`；`docs/self_model_v0_1_d4a_cloud_static_observation.md` |
| 2026-08-20 | D4A真实2.9B最小诊断入口本地完成：固定原失败夹具与三路3×3拉丁九调用，新增独立逐字授权、机器授权文件、受控结果路径、模型访问前single-use claim和claim后失败记录。普通“下一轮”不能生成授权；实现时执行明确为false。11项新增测试（含CLI缺锁和越界路径失败关闭）、D4A组合27项及全项目352项通过；入口无模型静态报告24/24检查全真，digest=`c743a9f9…eae1`。未创建机器授权/claim，未导入RWKV/Torch、访问权重、加载或执行模型，D4/active/Self效果/D5均不变 | `src/psa/self_model/d4a_real_diagnostic.py`；`configs/development/self_model_v0_1_d4a_real_diagnostic.json`；`docs/self_model_v0_1_d4a_real_diagnostic_entry.md` |
| 2026-08-20 | D4A真实入口在服务器完成无模型静态复验：27项组合测试通过，入口24/24检查与8个源码digest完整，报告`valid=true`、digest=`c743a9f9…eae1`并由本地同源报告独立复算一致。机器授权/claim未创建，RWKV/Torch未导入，权重/模型未访问或执行；D4、active、Self效果、D5和自动重跑均不变。下一步只等待冻结文本的逐字单次执行及结果观察授权，普通“下一轮”无效 | 远程`results/development/self_model_v0_1_d4a_real_diagnostic_entry/report.json`；`docs/self_model_v0_1_d4a_real_entry_cloud_observation.md` |
| 2026-08-20 | 项目负责人逐字授权执行一次Self Model v0.1 D4A真实2.9B最小诊断，并授权观察本次结果；授权文本与冻结配置完全一致，明确排除D4重跑、自动重跑、D5、active injection和Self效果实验。当前只记录人类授权，服务器机器授权/claim尚未创建，模型尚未执行；下一步在最终干净main上通过唯一入口单次运行，完成或失败均停止 | 项目负责人授权原文；`configs/development/self_model_v0_1_d4a_real_diagnostic.json` |
| 2026-08-20 | D4A真实2.9B最小诊断在提交`63a1878`上单次完成：14/14完整性检查、9次调用、9个同路线与27个跨路线比较完整，报告`valid=true`、digest=`d6b0602a…2e88`独立复算一致，claim=`21055ee6…7754`已消费。original与G0各自第一次调用独立，后续original/G0及全部G2共7次逐位一致；首轮original/G0从`state[4]`到`state[95]`共92组件不同，与D4失败位置签名一致。分类`within_route_instability_observed`，但不改写D4或授权D5；下一步只离线闭合D4完整调用轨迹 | 远程`results/development/self_model_v0_1_d4a_real_diagnostic_v01/report.json`；`docs/self_model_v0_1_d4a_real_diagnostic_observation.md` |
| 2026-08-20 | D4/D4A离线诊断闭环与D4B前瞻设计完成：重建确认D4共37次调用，G2确实有一次预热，但全部预热输出均被丢弃、矩阵没有G0；D4A的9次全记录调用显示original和G0各自首次瞬态后七次共享稳态，却没有复现D4的prefix和完整调度。因此不能逐调用对齐，共享`state[4..95]`位置签名只作为关联证据，不宣称已定位低层缓存机制。D4B只补测原失败单元，复现并记录prefix后固定四路线各预条件一次，再按4×4拉丁顺序计分16次，共21次调用、24个同路线和96个跨路线严格`torch.equal`比较；这是一项前瞻控制而非已证实修复，禁止自适应预热、容差、跨运行digest替代和自动重跑。本地4项新增、37项Phase 3组合及全项目356项测试通过；22项设计静态检查通过，digest=`7f3cfb7f…658d`。当前仅设计，模型/权重/runtime/claim/执行/观察/D5/active/Self效果均未授权 | `configs/development/self_model_v0_1_d4b_steady_state_off_design.json`；`docs/self_model_v0_1_d4b_steady_state_off_design.md`；`src/psa/self_model/d4b_steady_state_off_design.py` |
| 2026-08-20 | D4B服务器无模型静态复验通过：贴回完整报告22/22检查全真，D4 37次与D4A 9次重建轨迹完整，八个冻结源码digest与本地一致；报告digest=`7f3cfb7f…658d`从完整JSON独立复算匹配，所有安全字段为假，`git status --short`为空。贴回片段未包含HEAD与37项测试输出，因此只确认源码级跨主机静态门，不补写服务器提交号或测试计数。D4失败不变；runtime、真实授权/claim、模型执行、D5、active和Self效果均未授权 | 项目负责人贴回的`results/development/self_model_v0_1_d4b_steady_state_off_design/report.json`；`docs/self_model_v0_1_d4b_cloud_static_observation.md` |
| 2026-08-20 | D4B fake-first runtime核心本地完成：复用现有OFF-G1、G0和OFF-G2边界，固定执行并记录1次prefix、4次预条件与16次拉丁计分，共21次调用；只对计分输出生成24个同路线和96个跨路线严格`torch.equal`比较。fake全等只产生`runtime_core_verification_only`，不形成D5候选；单路线扰动固定失败且不加调用，异常只传播一次并恢复临时绑定。9项新增、46项Phase 3组合和全项目365项测试通过；17项静态检查全真，digest=`261325c4…ae46`。runtime源码不导入RWKV/Torch，真实入口、机器授权、claim、权重/模型执行、D5、active和Self效果均不存在或未授权 | `src/psa/self_model/d4b_steady_state_off_runtime.py`；`configs/development/self_model_v0_1_d4b_steady_state_off_runtime.json`；`docs/self_model_v0_1_d4b_steady_state_off_runtime.md` |
| 2026-08-20 | D4B runtime服务器无模型静态复验通过：完整报告17/17检查全真，九个配置/文档/脚本/源码/测试digest与本地最终报告一致，报告digest=`261325c4…ae46`匹配；除runtime核心已实现外，RWKV/Torch导入、installed-source探针、权重、模型加载/执行、真实入口、机器授权、claim、D4变化、D5、active、Self效果和自动重跑全部为假，服务器工作区为空。贴回内容未含HEAD及9项/46项测试输出，因此只确认源码级跨主机静态门。下一步需另行确认真实入口安全外壳，仍不等于模型执行授权 | 项目负责人贴回的`results/development/self_model_v0_1_d4b_steady_state_off_runtime/report.json`；`docs/self_model_v0_1_d4b_runtime_cloud_static_observation.md` |
| 2026-08-20 | D4B真实2.9B稳态OFF入口安全外壳本地完成：冻结唯一逐字授权、机器授权Schema、授权/结果唯一路径、干净main及配置/runtime digest绑定；精确环境锁和授权通过后先核对installed-source，再在模型配置资产验证、加载及任何forward前独占消费claim。claim后成功/失败均消耗机会，异常持久化且不重跑。真实外层报告区分fake模板安全字段，D4B通过只形成D5审阅候选而不授权D5。11项新增、57项Phase 3组合和全项目376项测试通过；27项静态检查全真，digest=`3c03a87e…24d2`。实现时机器授权/claim/RWKV/Torch/权重/模型执行/结果观察均未发生 | `src/psa/self_model/d4b_real_off_equivalence.py`；`configs/development/self_model_v0_1_d4b_real_off_equivalence.json`；`schemas/self_model_v0_1_d4b_real_authorization.schema.json`；`docs/self_model_v0_1_d4b_real_off_equivalence_entry.md` |
| 2026-08-20 | D4B真实入口服务器无模型静态复验通过：完整报告27/27检查全真，12个配置/Schema/文档/脚本/源码/测试digest与本地一致，报告digest=`3c03a87e…24d2`匹配，claim时序严格早于模型配置资产验证、加载和runtime核心；全部模型及研究升级安全字段为假，工作区干净。独立存在性检查输出`machine authorization absent`和`execution claim absent`。贴回未含HEAD及11项/57项测试输出，因此只确认入口源码级跨主机静态门。真实执行仍必须等待逐字单次执行与观察授权，普通确认无效 | 项目负责人贴回的`results/development/self_model_v0_1_d4b_real_off_equivalence_entry/report.json`；`docs/self_model_v0_1_d4b_real_entry_cloud_observation.md` |
| 2026-08-20 | 项目负责人逐字授权执行一次Self Model v0.1 D4B真实2.9B稳态OFF等价门并观察本次结果；授权文本与冻结配置完全一致，范围固定为21次调用和120项严格比较，明确排除D4/D4B重跑、自动重跑、D5、active injection及Self效果实验。当前仅持久化人类授权，服务器机器授权/claim尚未创建，模型尚未加载/执行；下一步只允许最终干净main上的唯一runner单次消费，完成或失败均停止 | 项目负责人授权原文；`docs/self_model_v0_1_d4b_real_execution_authorization.md`；`configs/development/self_model_v0_1_d4b_real_off_equivalence.json` |
| 2026-08-20 | D4B真实2.9B稳态OFF等价门在干净main提交`949bfa0`上单次通过：authorization digest、授权文件SHA-256、claim SHA-256和报告自digest四层完整性链均独立复算一致；21次调用全部记录，1次prefix、4次固定预条件、16次拉丁计分结构正确。24项同路线和96项跨路线比较中，logits及全部96个state组件均`torch.equal`，最大误差和不等元素数均为0；报告`valid=true`、digest=`8befb5f4…a20`，运行约17.18秒、CUDA峰值6,381,519,360字节。结果只支持固定预条件后的稳态OFF等价，D4失败不变；决策仅为`d5_review_candidate_only`，D5、active、Self效果和自动重跑均未授权或执行 | 项目负责人回传的authorization/claim/report；`docs/self_model_v0_1_d4b_real_off_equivalence_observation.md` |
