# EXP-001B 冻结只读分析计划

状态：结果观察前冻结版
版本：1.0

本分析只处理已经完整运行并通过独立完整性验证的 EXP-001B 原始包。它不重跑
EXP-001，不重新估计 E1–E3，不修改冻结阈值，也不写回任何原始结果目录。

## 输入绑定

- 父 EXP-001 原始 payload：`db4ba70ed521b55f23c4fc0ddafd2fb09af3cbe0132c0f065358a96f858b5ba7`；
- EXP-001B 原始 payload：`6926a932…90f6`；
- 320 个 factorial group；
- 11,008 条补充记录；
- EXP-001 与 EXP-001B 原始包都必须先通过独立验证。

## Matched-context

每组对完整 A–D 轮换做语义边缘化，计算 continuous 与 matched-context 的联合
margin 差。按 320 个 group 运行 10,000 次 BCa 和 100,000 次单侧符号翻转。
使用预注册种子 `1388088795`（BCa）和 `4175014787`（符号翻转）。
通过要求为：均值至少 0.50、95% 区间下界大于 0、单侧 p 小于 0.05，且
matched-context 记录没有 state-norm 警报。

## 同步通用控制

以 `prompt_visible_reset` 为基线，按 condition × task type 报告：

- 准确率下降不超过 0.05；
- forced-prefix greedy/roundtrip 格式率下降不超过 0.02；
- 平均目标 log-prob 下降不超过 0.25。

冻结原始控制记录没有保存逐条件 state-norm 和来源不变性诊断，因此这两项必须
标记为 `not_recorded`。即使已记录的控制指标全部通过，也不能把缺失诊断改写为通过。

## 正式生成

按 group 聚类区间报告格式、joint、identity 和 goal 准确率；同时报告 A–D 四个
目标位置准确率。通过要求为格式率至少 0.99，joint 区间下界至少 0.80，identity
和 goal 区间下界至少 0.90，四位置最大准确率差不超过 0.25，forced-prefix
greedy exact 率为 1.0。

## 决策边界

EXP-001B 只能补充 EXP-001 的控制证据，不能声称独立复制 E1–E3。任何已测指标
失败都进入 Revise/Stop；若已测指标通过但冻结记录缺少必需诊断，则 Gate 2/Gate 4
保持 `not_assessable_no_full_go`。只有所有预注册控制都可评估且通过，才允许进入
Phase 3 显式 Self Model。
