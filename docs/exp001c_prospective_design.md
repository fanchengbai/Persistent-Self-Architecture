# EXP-001C 前缀语义与状态控制前瞻性设计草案

版本：0.1 Draft  
日期：2026-08-10  
状态：仅离线 instrumentation 开发获批；未冻结；未批准模型试跑、测试集生成、正式运行或结果观察

## 1. 研究目的

EXP-001C 用新的盲测项目回答 EXP-001B 冻结记录无法回答的问题：

1. 自然状态路径是否保持正确答案语义；
2. 严格 `>\n` 前缀偏差的 token 分数和排名结构是什么；
3. 随机状态是否作为 assay sensitivity control 稳定地产生可检测破坏；
4. state-norm 阈值在正式输入形状上是否经过充分覆盖校准。

EXP-001C 不是 EXP-001B 重跑，不复用其确认性题目，也不修改其 Revise/Stop
决定。EXP-001B 仅可用于功效规划与设计风险识别。

## 2. 核心设计修订

### 2.1 分离语义与格式端点

主要语义端点：

- target answer 是否正确；
- target answer log-prob；
- target answer 相对最佳错误答案的 margin。

次要格式端点：

- `>\n` forced-prefix greedy exact；
- tokenizer roundtrip exact；
- 最终答案是否可解析。

两个端点分别冻结、分别判定、分别报告。格式失败不能自动改写成语义失败，语义
成功也不能取消格式端点本身的失败。

### 2.2 重新定义 `random_matched` 的角色

`random_matched` 是 assay sensitivity control，预期方向是相对干净基线出现语义
或 token 稳定性下降。它不再属于“自然状态无损伤”单元，也不能因为成功保持能力
而被默认解释为更好。

自然状态无损伤条件为：

- `continuous`；
- `restored`；
- `swapped_I`；
- `swapped_G`；
- `swapped_both`。

干净基线为 `reset` 与 `prompt_visible_reset`。

### 2.3 保存完整的 token 证据

对 `>` 与换行两个前缀位置，必须保存：

- expected token ID 与 greedy token ID；
- expected 与 greedy token 的 float32 logit、log-prob；
- expected token rank；
- greedy-minus-expected logit margin；
- top-10 token IDs、logits 与 log-probs。

答案边界还必须保存 target 与最佳错误答案的 log-prob 和 margin。任一必填字段缺失
时，正式记录无效且不得插补。

## 3. 假设与判定结构

### H1：自然状态语义保持

自然状态条件相对 `prompt_visible_reset` 的正确率、target log-prob 与 target margin
满足预先冻结的等效或无损伤界限。界限必须在任何正式项目生成前完成选择。

### H2：格式协议独立稳定

严格前缀和最终答案解析率满足单独冻结的格式要求。H2 失败不自动令 H1 失败，
但会阻止声称接口协议已经可靠。

### H3：随机状态检验具有灵敏度

`random_matched` 相对干净基线产生预先定义方向的破坏，用于证明测量管线能检测
状态损伤。它不是自然状态 no-harm Gate 的成员。

### H4：state-norm 阈值覆盖正式形状

阈值必须来自新的 formal-shape 非确认性校准包，按组件冻结，并在正式运行前证明
覆盖充分。只有完成覆盖验证后，零告警要求才可作为正式 Gate。

## 4. 样本与功效规划

当前不冻结正式样本量。冻结前必须完成：

1. 以 source control sample 为功效单位；
2. 四种来源组合等量；
3. 三种任务类型等量；
4. 明确最小可检测效应或等效界限；
5. 明确 cluster/factorial-group 处理；
6. 明确多重性控制；
7. 使用全新盲测项目，排除 EXP-001B 的 96 个控制项目。

EXP-001B 的 7 个格式事件只能辅助保守规划，不能据此选择有利阈值。

## 5. 分析要求

- 主要单位：source control sample；
- 语义与格式决定分开输出；
- 按 factorial group 聚类；
- 所有比例报告区间；
- 所有主要比较预先指定多重性策略；
- 缺失数据不允许静默删除；受影响单元转入 review；
- 后验影子指标不能改变确认性决定；
- `random_matched` 的灵敏度结果与自然状态 no-harm 结果分栏报告。

## 6. 开发门与授权顺序

必须依次完成：

1. instrumentation schema 与 roundtrip 单元测试；
2. synthetic logits fixture 测试；
3. formal-shape 非 Core 前缀探针；
4. formal-shape 非 Core state-norm 校准；
5. 功效与等效设计审查；
6. 源文件清单和 digest 审查；
7. 独立预注册 checksum 确认。

此后仍需三个独立授权边界：

1. 生成并冻结新的盲测集；
2. 正式运行；
3. 观察并分析结果。

任何前一阶段授权都不能自动扩展到后一阶段。

## 7. 当前未决定事项

- 模型与 checkpoint 的最终固定 digest；
- 正式样本量；
- H1 的等效或无损伤界限；
- H2 的格式阈值；
- H3 的最小灵敏度要求；
- state-norm 组件阈值；
- 多重性方案与主分析模型；
- 新盲测项目生成 seed 与 manifest。

这些事项必须在非确认性开发阶段解决并在正式数据生成前冻结。

## 8. 明确禁止

- 自动重跑 EXP-001B；
- 将 EXP-001B Gate 改写为通过；
- 把 EXP-001B 题目作为 EXP-001C 正式题目；
- 在观察 EXP-001C 正式结果后选择阈值；
- 结果观察后合并语义与格式端点；
- 把 `random_matched` 当作自然状态 no-harm 条件；
- 未获得独立授权就生成测试集、运行模型或观察结果。

## 9. 本草案的权限边界

本文件和 `configs/preregistration/exp001c_prefix_semantics.draft.json` 仍不是预注册冻结
包、测试集生成授权或正式运行授权。项目负责人于 2026-08-10 只批准了以下离线
开发范围：token instrumentation、JSON schema、synthetic logits fixtures、单元测试、
development probe manifest/verifier、CLI authority check 与带锁 backend-factory runner
scaffold。项目负责人随后授权进入离线 backend 集成阶段；该追加范围仅包括真实 RWKV
backend 工厂代码、formal-shape 非 Core fixture、带锁 `exp001c-probe-run` CLI，以及
fake-adapter backend 集成测试。

项目负责人随后于 2026-08-10 明确授权进入非 Core 模型开发门。该授权仅允许使用已锁定
formal-shape 非 Core fixture 执行一次 development pilot，并要求 manifest、独立授权包和
环境变量执行锁三者同时匹配；不允许自动重跑。该次 v01 pilot 已完成，并于 2026-08-11
获得只读结果观察授权；观察结论记录在
`docs/exp001c_noncore_pilot_v01_observation.md`。

v01 执行权限现已关闭。当前只授权离线设计 v02 prompt-visible 正控制 manifest，以及
Stage A 的锁定 runner/backend、未来授权与结果 Schema、假适配器测试；v02
执行前还必须通过只读 preflight，并将授权绑定到当前 manifest、服务器环境、Git 提交和
模型资产摘要；
模型执行、recurrent-state 阶段、正式测试集生成、正式运行和正式结果观察均未授权。
