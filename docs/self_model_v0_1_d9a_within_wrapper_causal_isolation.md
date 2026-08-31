# Self Model v0.1 D9-A within-wrapper causal isolation preregistration

## 为什么必须转向同一路径对照

D8-C 的单次真实结果有效确认：wrapper-zero 与 public 的差异超过两条路径各自的重复性背景。因此，后续不能把 wrapper-active 与 public 的差异直接归因于 Self projection。D9-A 把这个混杂从设计中移除：所有计分条件都使用同一个 persistent wrapper，只比较 wrapper-zero 与未来 active、mask、swap、matched-random 和 synthetic-active。

D8-C 只作为路线依据和非复用边界。D9-A 不复用 D8-C 的 token、fixture、seed、claim 或结果作为新实验数据，也不修改或重跑 D8-C。

## 全新 calibration 与 held-out 数据

D9-A 冻结 32 个 calibration fixture，覆盖 4×4 identity/goal 单元、每单元 2 个独立 replicate。未来 projection 只能读取这 32 个 calibration capture；这些输出不进入 held-out endpoint。

held-out 部分包含 16 个 identity/goal base case，每个使用 4 个代码轮换，共 64 个全新 fixture。calibration 与 held-out 使用不同 namespace、seed 和互斥 token 定义；同一 base case 的四轮故意共享完全相同的内容 token，只使用各自不同的 rotation-code token，从而让轮换只改变代码映射而不改变案例内容。held-out 在 projection artifact、训练超参数和 digest 全部冻结前不得访问。四轮代码的 target margin 必须先做 label marginalization，之后才以 16 个 base case 为统计单位。

## 同一 wrapper 内的因果调度

每个 held-out fixture 固定 7 个成对对照：

- `active_true` 对 `wrapper_zero`；
- `mask_identity` 对 `wrapper_zero`；
- `mask_goal` 对 `wrapper_zero`；
- `swap_identity` 对 `wrapper_zero`；
- `swap_goal` 对 `wrapper_zero`；
- `matched_random` 对 `wrapper_zero`；
- `synthetic_active` 对 `wrapper_zero`。

每个 pair 的两个调用从同一个 fixture prebuilt zero state 的独立克隆开始，且都经过同一个 persistent wrapper。每个 contrast 在 64 个 fixture 中严格平衡为 32 次 zero-first 和 32 次 condition-first；七个调度位置各出现 9 或 10 次。未来计划为 32 次 calibration capture 加 448 个 held-out pair×2，共 928 次 forward。public route 不进入调度、计分或诊断。

`synthetic_active` 只验证注入机制能在目标层改变输出，不是 Self 证据。`matched_random` 排除“任何同尺度扰动都有效”；mask 和 swap 用于检验 identity/goal 字段的选择性与因果方向。

## 冻结端点

主要估计量是 16 个 base case 上 `active_true - wrapper_zero` target-alignment margin 的均值。主要门要求 base-case cluster bootstrap 的单侧 99% 下界大于零，同时至少 13/16 base case 为正，并要求每个 identity 与 goal 水平至少 3/4 为正。

全部支持门同样是必要条件：true active 必须在 99% 下界上优于 matched random；identity/goal mask 各至少 13/16 呈现对应字段下降且另一字段保持；identity/goal swap 各至少 12/16 跟随交换目标；synthetic active 至少改变 60/64 个 held-out 输出并满足目标层单次应用。

只有全部门同时通过，未来结果才可写成 `within_wrapper_causal_specificity_candidate_supported_nonformal_nonself_engineering_only`。这仍只是非正式、非 Self 的工程候选证据，不允许形成 Self Model 效果结论。缺失、非有限、不兼容、不完整或任一门失败都必须停止，不能放宽阈值或重跑。

## 当前权限与后续门

本轮只冻结 D9-A 研究问题、fixture/schedule commitments、确定性策略、端点与 14 个全新 artifact namespace，并用纯 Python 合成数据验证判定逻辑。没有实现 projection、manifest、真实 runner 或执行入口；没有探测 installed source、导入 RWKV/Torch、访问权重、加载或执行模型。

下一步只能在项目负责人另行确认后进入 D9-B deterministic manifests 与 fake endpoint contract。D9-C、D9-D 真实执行、D8-C或历史重跑、D7-D/D7-E、正式测试集、Self 效果、Self Updater、raw-original 和自动重跑继续关闭。
