# EXP-001C v02 Stage B Pilot v01 观察记录

状态：`stage_b_exploratory_observation_complete_no_confirmatory_decision`

观察日期：2026-08-11

本记录只覆盖已授权并完成的一次 recurrent-state、非 Core、224 条 Stage B pilot。
它不授权重跑 Stage A、访问正式测试集、启动正式运行、作确认性决定或自动重跑。

## 1. 完整性与分析边界

- 执行 commit：`30c4607a9f13d4e8daf45cfef542eb7a600a1ebe`
- preflight digest：`a06f73607703d92228028e341d2d7469490cb620245d06dcd9dd03a06cc1151b`
- authorization digest：`a2f6919d3a84992067936ed089f9effa1340c4087f5d0610df2910f4b896559f`
- Stage B result SHA-256：`e0b871e7c12e11819fdb913be8307eace1acbc7e8a52d79f966e50d35da8b655`
- analysis plan digest：`c4393267b3d7ed2fbd1fd60479dcdaecf957d63e033e0c0e42880f40d43eee5a`
- 原始记录：7 条件 × 32 = 224
- 语义读出：每条件 8 个语义案例，每案例先平均 A/B/C/D 四轮对应语义选项的 log score
- `single_use_execution_claim_consumed=true`
- `stage_a_rerun=false`
- `formal_test_set_accessed=false`
- `formal_run=false`
- `contains_confirmatory_decision=false`
- `automatic_rerun_authorized=false`

分析配置在读取派生分数前冻结并绑定原始结果 SHA-256。分析过程没有加载模型，只读取已完成
原始包。五个状态语义条件使用预先定义的注入状态字段作为目标；`reset` 和
`random_matched` 没有“正确答案”，只描述其预测与原 Stage A 参考字段是否一致。

## 2. 描述性结果

| 条件 | 联合准确率 | Domain | Operation | 平均目标 margin |
|---|---:|---:|---:|---:|
| continuous | 7/8 = 0.875 | 8/8 = 1.000 | 7/8 = 0.875 | 1.626872 |
| restored | 7/8 = 0.875 | 8/8 = 1.000 | 7/8 = 0.875 | 1.626971 |
| swapped_I | 7/8 = 0.875 | 8/8 = 1.000 | 7/8 = 0.875 | 1.626971 |
| swapped_G | 7/8 = 0.875 | 8/8 = 1.000 | 7/8 = 0.875 | 1.626971 |
| swapped_both | 7/8 = 0.875 | 8/8 = 1.000 | 7/8 = 0.875 | 1.626971 |

continuous 与 restored 的 8/8 个轮换边际化语义预测完全一致。三种 swap 都有 7/8 个
语义案例跟随实际注入的 state 字段，且 domain 始终为 8/8；共同的单个错误是状态语义
`indigo + harbor` 被读成 `indigo + spiral`。swap 后这个错误移动到承载该注入状态的不同
query case，而不是固定停留在原 query 上，符合行为跟随 state 来源的模式。

五个主要条件的原始代码 top-1 计数完全相同：A=9、B=9、C=8、D=6，raw code accuracy
也均为 28/32=0.875，没有单一代码塌缩。由于每个条件只有8个轮换边际化语义案例，以上
数值只作探索性描述，不计算事后显著性或设立通过阈值。

| 诊断条件 | 与原 Stage A 参考字段匹配 | 平均参考 margin | 描述 |
|---|---:|---:|---|
| reset | 2/8 = 0.250 | -0.750179 | 只预测两种 `spiral` 组合，不支持保留原绑定 |
| random_matched | 2/8 = 0.250 | -0.063293 | 输出较分散，但与原绑定仅机会水平匹配 |

这两个值不是“控制准确率”，因为设计没有为 reset/random 定义状态语义正确项；它们只说明
去除真实 state 或替换为尺度匹配随机 state 后，原绑定不再稳定保留。

## 3. 解释与下一步

本轮为原生 recurrent state 提供了额外的非 Core 支持：磁盘恢复保持预测，身份/目标/双字段
交换后的行为大多跟随被注入的 state，reset 与随机 state 不保留原绑定。该模式与 EXP-001
中已观察到的持久、可恢复、可干预因果信号一致。

边界同样明确：本轮只有8个语义案例、使用开发标签对和受控四选一读出，也没有补齐
EXP-001B 所针对的 matched-context、每条件同步通用能力和正式自由生成控制。因此不能据此
宣称模型“已经拥有 Self”，也不能替代 Phase 2 的正式控制闭合。

工程路线可以把本结果视为进入显式 Self Model v0.1 原型设计的正面依据：先实现静态
Self Store、Self Encoder、可关闭/缩放的 gated injection，以及字段 mask/swap/random/
coupling-off 消融；第一轮只做非 Core A–E 基线比较。任何正式 Self Model 优越性结论仍需
新的预注册、控制和独立授权。
