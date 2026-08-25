# Self Model v0.1 Coupling-D6D 核心趋近非 Core 联合实验设计

日期：2026-08-25  
状态：仅设计与无模型审查；wrapper、真实 Self projection、模型执行和效果结论均未实现或授权

## 结论先行

D6D 不再增加一个独立的纯机制执行轮。未来若获得逐门授权，只允许一次联合的非 Core 实验：在同一个真实模型进程、同一个 project-owned wrapper、同一个 single-use claim 和同一份结果中，交错运行 synthetic positive control 与真实冻结 Self projection 的因果条件。synthetic 只回答注入链路能否工作；identity/goal 的 matched、swap、mask、norm-matched random 和 OFF 才接近 Self 的核心语义问题。

本轮没有导入 RWKV/Torch、读取权重、加载或执行模型，也没有构造 projection。现有 `DeterministicHashFakeSelfEncoder` 和 D5A fake projection 明确不能升级为真实 Self 证据。

## 为什么从 D6C 改为 wrapper ownership

D6C 的唯一 claim 已消费。模型加载后、首个 forward 前，runtime 把 callback 和两个方法写到真实 RWKV 对象，再假设三个名字都直接存在于实例 `__dict__`；真实对象不满足这个存储假设，因此 26 次计划调用实际为零。D6D 不修补或重跑 D6C。

D6D 的设计边界是：真实 RWKV 对象只作为冻结的读取委托目标。`forward`、`forward_one`、`forward_seq`、固定 dispatcher 和 request context 都由外部 wrapper 持有，instrumented 方法只绑定到 wrapper。创建 wrapper 前、每次调用前后和实验结束时，都必须核验真实模型实例字典的键与对象身份没有变化；任何 `setattr`、`delattr` 或字典变动都丢弃未提交输出并停止。

OFF、zero、synthetic 和所有 Self 条件必须走同一个 wrapper 和同一组方法身份。禁止 raw-original 路线，避免再次把 route 与方法生命周期混在一起。

## 真正的冻结 Self projection

未来 projection 不能是 hash fake 或无训练的 synthetic vector。设计冻结为字段分离的 learned projection：输入只能是校验通过的 Self State v0.1 中 `identity_anchors` 与 `active_goals`，两个字段使用独立编码分支，经冻结 gate 合并为 2560 维 post-FFN residual，目标位置固定为 zero-based 第 15 层。Self State 不进入 prompt，基础模型权重不训练，推理中没有 Self Updater 或在线更新。

projection 只能使用与 pilot 隔离的非 Core development 数据训练。两个字段分支以无 bias 的冻结 gate 求和，所以双 mask 必须产生严格零 projection。训练清单、字段编码器参数、projection 参数、优化器/seed/超参数和最终资产都必须在任何 pilot 模型调用前生成 digest 并冻结；pilot manifest 的具体值在 projection digest 冻结前保持盲态。pilot 数据不能回头选择 projection、层、阈值或删除对照。本轮没有构造或训练这些资产；这属于下一独立实现门。

## 同一实验内的 11 个条件

每个 fixture 都运行以下计分条件：

1. `wrapper_off`：禁用且不构造 projection；
2. `wrapper_zero`：同一 wrapper 的零 scale，不构造 projection；
3. `synthetic_positive`：同一固定层的 wrapper-owned synthetic 正控制，不是 Self 证据；
4. `self_matched`：fixture 匹配的 identity 与 goal；
5. `self_identity_swap`：只交换配对 Self State 的 identity；
6. `self_goal_swap`：只交换 goal；
7. `self_identity_goal_swap`：两个字段同时交换；
8. `self_identity_mask`：identity 分支精确置零；
9. `self_goal_mask`：goal 分支精确置零；
10. `self_identity_goal_mask`：两个分支精确置零，输出必须与 zero 条件精确一致；
11. `self_identity_goal_norm_matched_random`：两个编码分支独立做固定 seed、分支 L2 norm 匹配随机化。

synthetic 正控制和八个真实 Self projection 条件在同一 cyclic Latin 调度中交错，不得拆成另一份授权、另一个 claim 或另一次“先验证机制”的执行。

## 非 Core pilot 蓝图和调用数

pilot 使用 12 个冻结 fixture：identity-bound choice、goal-bound choice、identity-goal conflict 各 4 个。每个 fixture 有两份配对静态 Self State、预先冻结的 choice token IDs、目标 margin 和通用能力 sentinel；prompt 本身不出现 Self 内容，每次调用重置 recurrent state。正式测试集完全关闭。

每个 fixture 先做 1 次不计分 OFF 预条件，再按 11×11 cyclic Latin 的一行执行全部条件；第 12 个 fixture 重复第一行。因此总计 144 次模型 forward，其中 132 次计分、12 次 synthetic、96 次真实 Self projection。任何执行都需要未来新的逐字授权、机器授权、唯一目录和 single-use claim。

## 判定边界

D6D 只产生 `noncore engineering pilot` 分类，不产生 Self 效果结论。未来阈值必须在首次模型调用前冻结，至少覆盖：

- 真实模型实例字典完全不变；
- OFF、zero、双 mask 精确一致；
- synthetic 在固定层产生可观察差异；
- 输出有限、调用与层访问计数精确、所有资产 digest 匹配；
- identity family 中 matched margin 高于 identity swap/mask；
- goal family 中 matched margin 高于 goal swap/mask；
- joint family 中 matched 高于双 swap、双 mask 和 norm-random；
- 非目标字段对照不反转预注册的字段特异性；
- 通用能力 sentinel 不越过预冻结退化界限。

数值门在本设计中直接冻结：12/12 OFF-zero-double-mask 精确匹配，12/12 synthetic 输出必须与 OFF 不同；identity、goal、joint 三组的每个预注册方向性比较至少 3/4 fixture 通过；12 个通用能力 sentinel 最多允许 1 个 choice code 相对 OFF 改变；非有限输出上限为 0。这些只是非 Core 工程分类阈值，不是统计确认性 Self 效果阈值。

无论通过或失败，都不能自动打开 D6E、重跑、删除 route 或形成正式 Self 效果结论。

## 当前授权边界与下一门

本轮只允许设计、静态合同和纯 Python 计划审查。wrapper runtime、projection 构建/训练、installed source 探测、RWKV/Torch、权重、模型和真实执行均未授权。D5C/P1/P2/D6C 不得重跑。

下一步若继续，需负责人逐字确认：

> 确认进入 Self Model v0.1 Coupling-D6D-I wrapper-owned真实路径与冻结Self projection构建工具的无模型实现；必须保持D6D单一联合实验、不得修改真实RWKV实例字典，只实现wrapper、projection训练冻结与artifact审计接口及纯Python验收；不探测installed source、不导入RWKV/Torch、不访问权重、不加载或执行模型，不授权D6D真实执行、D6E、正式测试集、Self效果结论、Self Updater、D5C/P1/P2/D6C重跑或自动重跑。
