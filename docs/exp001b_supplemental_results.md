# EXP-001B 补充控制分析结果

版本：1.0
结果观察日期：2026-08-10
分析实现：`v0.2-decision-precedence-correction`
实验状态：原始包完整性验证通过；补充数据只读分析完成；父 EXP-001 逐组引用缺失

## 1. 结论

EXP-001B 的正式生成格式与能力门通过，但同步通用能力控制和 matched-context
state-norm 门失败。按照冻结决策规则，任何已测控制失败都优先进入 Revise/Stop，
因此 Gate 2 与 Gate 4 均为
`revise_or_stop_measured_supplemental_control_failure`，当前证据不能闭合 Phase 2
行为控制包，也不能进入 Phase 3 显式 Self Model。

父 EXP-001 的逐组原始包和分析包不在执行主机上，因此 continuous minus
matched-context 的配对主估计量、BCa 区间和符号翻转 p 值不可评估。未使用聚合
均值、代理数据或重跑父实验替代缺失引用。

## 2. 结果包身份

| 项目 | 值 |
|---|---|
| 远程源码提交 | `11a86c69d895b413c33eb85674406a02457fa60f` |
| EXP-001B 原始记录 | 11,008 |
| EXP-001B 原始 payload SHA-256 | `6926a932220f34b37c6b4e86fa65edc230e414726bd2d4d308bf471d1af290f6` |
| 分析配置 SHA-256 | `e7b1aff62004f545636905c399ca3c48f1d31f56a2c59b51d7c4ac66c0120d31` |
| group metrics SHA-256 | `19a9256dce612e27cafee29fbf19aa8c97dcb0f27845d60a8d2f61e885fb99a6` |
| supplemental report SHA-256 | `1f054a8e1d2b9809908de8af6e788d92a5f26a245eaa81158437c85d977ea2b6` |
| 分析包 SHA-256 | `7c666dea178e215c84712d28b907266792e505e33e3d7c38c9fa63b49f32773b` |

两个报告文件的 SHA-256 已根据 summary 独立重算并一致。分析写入新目录
`results/confirmatory/exp001b_v1_analysis_v02`，没有覆盖原始结果或第一版分析包。

## 3. 正式生成读出

正式生成门通过。

| 指标 | 均值 | 95% BCa CI | 冻结要求 |
|---|---:|---:|---:|
| 格式有效率 | 1.0000 | [1.0000, 1.0000] | ≥ 0.99 |
| forced-prefix 有效率 | 1.0000 | [1.0000, 1.0000] | = 1.00 |
| joint 准确率 | 0.8889 | [0.8799, 0.8971] | CI 下界 ≥ 0.80 |
| identity 准确率 | 0.9430 | [0.9344, 0.9502] | CI 下界 ≥ 0.90 |
| goal 准确率 | 0.9377 | [0.9301, 0.9447] | CI 下界 ≥ 0.90 |

答案位置准确率为 A `0.9641`、B `0.9492`、C `0.8977`、D `0.7445`；最大差
`0.2195`，低于冻结上限 `0.25`。

## 4. 同步通用能力控制

24 个 condition × task 单元中有 13 个触发预声明警报：

- `random_matched` 的三个 task 全部同时触发准确率、forced-prefix 格式和目标
  log-prob 警报；准确率相对基线下降 `0.59375` 至 `0.8125`，平均目标
  log-prob 下降 `7.2373` 至 `7.9399`。
- `continuous`、`restored`、`swapped_I`、`swapped_G` 和 `swapped_both` 的
  `single_field_lexical_match` 与 `unrelated_two_field_symbol_match` 均触发
  forced-prefix 格式警报；格式率下降 `0.03125` 至 `0.09375`。
- 上述非 random-matched 单元的准确率和目标 log-prob 没有触发阈值，但格式
  警报本身已经足以使同步控制门失败。

冻结控制输出没有逐条件 state-norm 和来源不变性诊断，因此这两项仍为
`not_recorded`；不能把缺失诊断改写成通过。

## 5. Matched-context

父 EXP-001 逐组 continuous 引用缺失，因此预注册的 continuous minus
matched-context 联合 margin 差不可评估。只允许报告以下描述性结果：

- matched-context 联合 margin 均值 `0.04751`；
- 95% BCa CI `[0.01842, 0.07610]`；
- 冻结 state-norm 组件警报总数 `92,149`。

警报总数是跨 5,120 条 matched-context 记录累计的组件级警报，不等于失败记录
条数。由于冻结规则要求零警报，该项已测门失败。描述性 matched-context margin
不能替代缺失的配对主估计量。

## 6. 决策与下一步

- Gate 2：`revise_or_stop_measured_supplemental_control_failure`；
- Gate 4：`revise_or_stop_measured_supplemental_control_failure`；
- 路由：`review_frozen_failures_without_rerun`；
- 禁止自动重跑或用相同冻结测试集调参后再次作为确认性证据。

后续应先区分控制格式警报来自 forced-prefix 检查、状态条件执行路径还是模型真实
能力退化，并复核 matched-context state-norm 阈值与正式输入分布的适配性。如需
恢复完整 matched-context 主判定，应找回 SHA-256 为
`db4ba70ed521b55f23c4fc0ddafd2fb09af3cbe0132c0f065358a96f858b5ba7`
的父 EXP-001 原始包或其已验证逐组分析包；不得重跑父实验来替代该引用。
