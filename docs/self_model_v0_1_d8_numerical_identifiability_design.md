# Self Model v0.1 D8-A 数值可识别性与 excess-drift 预注册设计

## 研究问题

D8-A 不再重复 D7-C 的“public 与 wrapper 是否逐项完全相等”问题。它提出一个新的、可识别的问题：在调用顺序平衡且确定性策略预先固定后，public 与 instrumented wrapper 之间的输出差异，是否仍然超过 public 自身和 wrapper 自身的数值重复性包络？

这是工程机制识别，不是 Self 效果实验。即使未来结果为正，也只能说明两条路径存在超过路径内噪声的差异，不能证明 Self Model 有效。

## 与 D7-C 的隔离

D7-C 只作为历史失败边界和禁止复用约束：

- 不复用 D7-C 的 8 个 cell、token、seed、claim 或输出；
- 不把 D7-C 的误差数值并入 D8 的统计数据或阈值；
- D7-C 的失败、claim 消费和禁止重跑结论保持不变；
- D8 使用新的 fixture、schedule、determinism、authorization、claim 和 output 命名空间。

## 全新 fixture

冻结 24 个计分 fixture，分为四层，每层 6 个：

1. `forward_one`、`full_output=false`；
2. `forward_one`、`full_output=true`；
3. 三 token `forward_seq`、`full_output=false`；
4. 五 token `forward_seq`、`full_output=true`。

token 由新的字符串 seed 通过 SHA-256 确定性产生，范围为 1024–60000，全局不重复，并明确排除 D7-C 使用过的 187、931、2764。每次计分调用都从同一 fixture 的全新预构造 zero-state clone 开始，不使用 `state=None`，从而把 state 初始化问题排除在新研究问题之外。

每层另有一个全新 conditioning fixture，固定先 public 后 wrapper zero，各调用一次，共 8 次。conditioning 输出不计分，只让两条路径都完成同形状首次执行；失败则整门无效，不允许临时放宽或重跑。

fixture commitment：`8976ac9f3f0b042e92ba146e58cc1df8c2d05e5a4635ccb0de558fb36161499e`。

## 路径内包络与顺序平衡

每个 fixture 包含四类配对：

- public→public；
- wrapper→wrapper；
- public→wrapper；
- wrapper→public。

每类配对重复 3 次，每对两次 forward，因此每 fixture 有 12 对、24 次计分调用。24 个 fixture 共 288 对、576 次计分调用；加上 8 次 conditioning，总未来调用数固定为 584。

四类配对先由新的 schedule seed 通过 SHA-256 得到固定基础顺序，再采用 4×4 拉丁轮换。跨全部 fixture 与重复轮，每一类配对在第 1、2、3、4 顺序位置都恰好出现 18 次。每一对的两次调用都使用同一基础 state 的独立 clone。

schedule commitment：`a53cf5edc4f132c3fc773d63f7686f91f485b5c08c52f82beec983af36816465`。

## 确定性策略

未来 launcher 必须在启动 Python 进程前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、固定 `PYTHONHASHSEED` 并确保 `RWKV_DE_VERSION` 未设置。进入进程后、加载模型前，再固定 Python、Torch CPU/CUDA seed，启用严格 deterministic algorithms，关闭 warn-only、cuDNN benchmark 和两处 TF32，启用 cuDNN deterministic，并固定 float32 matmul precision 为 highest。

若实际栈不支持该策略，结论只能是“确定性前置门失败并停止”。不得看到错误后改成 warn-only、关闭 deterministic 或继续计分。

## 主要端点

对每个输出定义：

- tensor distance：最大绝对差除以两边最大绝对值与 `1e-12` 的最大者；
- state distance：96 个兼容 state 组件 tensor distance 的最大值；
- output distance：logits distance 与 state distance 的最大值。

对每个 fixture、每个重复轮：

```text
within envelope = max(D(public, public), D(wrapper, wrapper))
cross floor     = min(D(public, wrapper), D(wrapper, public))
excess drift    = cross floor - within envelope
```

使用 cross-route 的较小值和 within-route 的较大值是保守设计：只有两种跨路径顺序都超过最坏的路径内重复性，excess 才为正。每个 fixture 的值是三个重复 excess 的中位数；主要估计量是 24 个 fixture 值的均值。

未来正面判定必须同时满足：

- fixture 聚类 bootstrap 的单侧 99% 下界大于 0，固定 100,000 次和独立 seed；
- 24 个 fixture 中至少 21 个 excess 严格为正；
- 四个层中每层至少 5/6 为正；
- 所有输出完整、有限、结构兼容，调用和确定性审计全部通过。

不满足正面门时统一判为 `inconclusive_no_route_equivalence_claim`，不得宣称两条路径等价。logits/state 分项、顺序交互和两条路径各自重复性不对称只作预注册描述指标，不替换主要端点。

## 当前权限边界

本轮只冻结 D8-A 设计并用纯 Python 验证生成器、平衡和端点逻辑。没有探测 installed source，没有修改 D7-C runner，没有实现真实入口，也没有导入 RWKV/Torch、访问权重、加载或执行模型。

D8-B 的 manifest/fake contract 与 D8-C 的未来真实执行都需要新的独立确认或逐字授权。D7-C修复/重跑、D7-D/D7-E、projection、正式测试集、Self效果、Self Updater、raw-original 和自动重跑继续关闭。
