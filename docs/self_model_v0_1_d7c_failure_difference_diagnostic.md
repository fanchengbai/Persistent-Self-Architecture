# Self Model v0.1 D7-C 失败差异来源纯离线诊断

## 结论

D7-C 的真实失败保持不变：8/8 个 public OFF 与 wrapper zero cell 的 logits 和 state 都没有逐项精确相等。现有结果足以证明“精确等价门失败”，但不足以唯一判断差异来自 instrumented 路径，还是来自固定调用顺序下的同路径数值重复性/首次形状效应。

本轮没有导入 RWKV 或 Torch，没有访问权重、加载或执行模型，没有修改真实 runner，也没有实现修复或授权重跑。

## 已冻结的观察

- 真实 runner 正常完成并生成完整失败报告，不是异常崩溃；single-use claim 已消费。
- 18/18 次 forward 完成：8 次 public OFF、8 次 wrapper zero、2 次 synthetic active。
- active 正控制、callback/目标层计数、wrapper 生命周期和基础实例字典不变性均通过。
- 8 个 cell 都有 96 个形状/类型兼容的 state 组件，但都只有 `state[0..3]` 精确；第一个非精确组件均为 `state[4]`，最大误差组件均为 `state[94]`。
- `state[4]` 对应第 1 层 attention KV，`state[94]` 对应第 31 层 attention KV；这与 recurrent 差异逐层传播相容，但不能单独证明差异源头。
- cell 2/3/4 的三项误差指标完全相同，cell 6/8 也相同，所以 none/prebuilt 初始化方式和 `full_output` 都不能作为唯一解释。
- 确定性算法、cuDNN deterministic 和全局 determinism 均未启用；这使数值重复性成为合理候选，但仍不是因果证明。

## 源码级可识别性审计

真实 runner 在每个 cell 中固定先调用 public 路径，再调用 wrapper 路径。每条路径每个 cell 只有一次调用，没有 public-public、wrapper-wrapper 的同路径重复性基线，也没有 wrapper-first/public-first 的顺序平衡。

因此“路径”与“第几次调用/调用顺序”在 D7-C 中完全混杂。纯 Python 合成 fixture 给出了两个因果机制：

1. 两条路径语义相同，但第二次调用发生 background/order drift；
2. 时间稳定，但 instrumented 路径自身发生 drift。

两者在 D7-C 固定 public→wrapper 观察方式下产生相同摘要指纹：`state[0..3]` 精确、`state[4]` 首次不精确、`state[94]` 误差最大，并且 logits/state 都不精确。由此证明现有摘要不能唯一识别原因。

## 可以与不可以得出的结论

可以得出：

- D7-C 精确等价门真实失败，历史结论不得改变。
- 没有发现 state 形状或组件清单损坏；问题是数值精确性而非结构完整性。
- state 初始化计数、none/prebuilt 和 `full_output` 不是唯一原因。
- instrumented-path 数值效应、同进程重复性/首次形状效应和 recurrent 放大都与结果相容。

不能得出：

- 不能断言 instrumented wrapper 一定改变了模型语义。
- 不能断言差异一定只是 CUDA 非确定性。
- 不能用当前报告选择修复方案，也不能据此重跑 D7-C。

## 科学独立的新路线候选

存在一个设计层面的独立候选：`D8 numerical identifiability and excess drift`。它的新问题不是再次要求 D7-C 的逐项精确相等，而是检验“跨路径差异是否超过各路径自身的数值重复性包络”。最低要求是：

- 使用全新的 token fixture、seed、预注册、授权、claim 和输出命名空间；
- 不复用 D7-C 的 8 个 cell、claim 或结果作为新实验数据；
- 同时测 public-public 与 wrapper-wrapper 重复性；
- 对 public→wrapper 与 wrapper→public 做顺序平衡；
- 观察结果前冻结确定性策略；
- 以 cross-route drift 减去 within-route repeatability envelope 的差异中差异作为主要终点；
- D7-C 失败继续保留，D7-D/D7-E、projection 和 Self 效果门继续关闭。

本轮只建立这一预注册候选，没有实现入口、创建授权或执行任何实验。下一门是由项目负责人审阅是否允许编写 D8 的纯离线预注册设计。
