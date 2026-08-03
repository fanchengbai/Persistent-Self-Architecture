# EXP-001 确认性结果只读分析计划

版本：1.0  
状态：真实分数首次读取前冻结  
适用对象：`EXP-001`、模型 `rwkv7-g1h-2.9b-20260710`

## 1. 安全边界

分析入口只接受已经通过独立完整性验证的全量原始包。它要求320个factorial
group和40,960条原始记录完整存在，且总payload digest与完成报告一致。分析只
读取原始目录，结果写入新的独立目录；非空结果目录不会被覆盖。分析不会重新
加载模型、修改Core Set、修改原始分数或触发实验重跑。

首次生成正式分析报告时，`confirmatory_results_observed`从原始包中的`false`
转为分析报告中的`true`；原始运行manifest和completion保持不变，以保存当时
“尚未观察结果”的事实。

## 2. 标签轮换读出

每个语义案例必须完整包含A、B、C、D四次轮换。先把每次记录的代码分数映射回
四个语义组合，再对同一语义组合的四个log score求均值。所有准确率、margin、
维度log-odds和swap判断都使用这个代码边缘化分数，避免把答案字母偏好误当成
语义能力。

## 3. 主要终点

冻结的主要读出条件为`continuous`：

- E1：每组`identity_transfer`；
- E2：每组`goal_transfer`；
- E3：每组`mean_joint_margin`。

三个终点均以factorial group为统计单位，报告均值、中位数、标准差、IQR、
10,000次BCa 95%区间和至少100,000次单侧配对符号翻转检验。三项原始p值使用
Holm校正，family-wise alpha为0.05。支持一个主要终点必须同时满足：校正p值
小于0.05、点估计达到冻结SESOI、区间下界大于0。

## 4. 预注册的辅助门

- `reset`和`random_matched`：比较continuous相对基线的联合log-margin优势；
- `restored`：比较代码边缘化option score及语义argmax与continuous的一致性；
- `swapped_I`、`swapped_G`：检查来源字段方向迁移；
- `swapped_both`：检查答案是否迁移到完整donor组合；
- `prompt_visible`：给出分数式identity、goal和joint能力上限，并计算跨组均值之
  比的prompt-normalized retention；
- joint accuracy：区间下界至少0.60，并以0.50作为I-only、G-only、最近变量
  及固定答案位置的保守共同上限进行组级比较；
- identity/goal specificity：点估计至少0.25且区间下界大于0。

## 5. 已冻结但本次无法计算的项目

旧评价协议同时要求`matched-context`对照，以及每个state条件同步运行96条通用
能力控制；正式Core Set最终只冻结了8个条件，40,960条记录中没有这两类数据。
此外，正式runner保存的是候选代码log score，不是自由生成文本，因此无法从
正式原始包重新计算generated-format有效率。

这些缺口不得用开发集资格结果、reset或其他条件代替。分析仍可报告E1–E3和已
采集的因果/工程证据，但Gate 2与最终Gate 4必须标记为
`not_assessable_no_full_go`。这是一项设计完整性结论，不授权补跑、追加样本或
修改冻结设计。

## 6. 结果解释顺序

1. restore门失败：停止语义解释，报告基础设施问题，不自动重跑；
2. 任一主要终点不支持：如实报告阴性或不确定结果，不追加样本；
3. E1/E2存在但联合门失败：最多报告部分或weak persistence；
4. 已采集的全部门通过：报告“测得的效应通过”，同时因未采集的预注册控制而
   不授予完整原生state载体Go；
5. 无论哪一种结果，都不把EXP-001写成完整Persistent Self已经成立。
