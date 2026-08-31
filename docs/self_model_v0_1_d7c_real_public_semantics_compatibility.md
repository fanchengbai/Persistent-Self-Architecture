# Self Model v0.1 D7-C 真实 public 语义兼容门

日期：2026-08-28

状态：设计与无模型安全入口已实现；未探测 installed source，未创建机器授权或 claim，未加载或执行模型。

## 研究位置

D7-C 不是 Self 效果实验。它只验证未来 D7 wrapper 是否忠实保留 RWKV public `forward` 的调用语义，尤其是 D6D 失败暴露的 `state=None` 初始化边界。兼容门失败时必须消费唯一 claim 并停止，不能进入 D7-D 或 D7-E。

## 18-call 计划

兼容矩阵由 `forward_one`/`forward_seq`、`state=None`/prebuilt 和 `full_output=false`/`true` 的笛卡尔积形成 8 个 cell。每个 cell 未来各执行一次 base public OFF 和一次 external wrapper zero，总计 16 次；另执行 single 与 sequence 各一次 synthetic active，共 18 次 forward。

所有 token 都来自固定 synthetic protocol，与 calibration、held-out 和 capability payload 无关。D7-C 不读取三类研究 payload，也不使用正式测试集。

## public 语义 contract

新的 wrapper 完全位于基础实例之外，不向真实模型实例字典写入或删除方法。wrapper 在 child dispatch 前执行与 public `forward` 相同的边界：`state=None` 时先调用 `generate_zero_state()`，prebuilt state 时不初始化；单 token 分派到 `forward_one`，序列分派到 `forward_seq`，并把 `full_output` 传给 sequence 路径。

8 个 cell 的 public OFF 与 wrapper zero logits 和全部 state component 在未来真实运行中都必须逐项 `torch.equal`，state inventory 也必须一致。基础实例字典、wrapper 方法身份和 context 都必须稳定。

## 独立目标层规则

D7-C 只为 synthetic 正控制定义规则 `d7_lower_half_terminal_layer_v01`：对 32 层模型取下半区最后一层，即 `32 // 2 - 1 = 15`。该规则由架构层数直接推导，不使用 D6D 的 fixture、seed、claim、authorization、输出或结果，也不是根据效果挑层。

两次 synthetic active 调用各应触发 32 次 callback，并只在 0-based 第 15 层应用一次，总计 64 次 callback、2 次应用。active 输出必须区别于 zero；它仍不是 projection 或 Self 表征。

## 单次执行边界

未来真实执行需要逐字授权、干净的 `main`、固定配置和源码摘要、唯一机器授权文件、唯一输出目录与 single-use claim。授权验证先于 installed source 探测；claim 先于模型配置、权重访问和模型加载。成功或失败都不允许 D7-C 重跑或自动重跑。

机器授权生命周期必须绑定同一个“创建前”静态报告摘要：首次创建仍要求 authorization、claim、report 和 failure 全部缺席；创建后的验证只把这些执行 artifact 的存在性检查规范化为创建前状态，同时继续复算配置、Schema、源码、Git 和其余全部静态证据。这样机器授权本身的出现不会改变它所绑定的摘要，任何其他源码或协议变化仍会失败关闭。

当前静态验证不得调用未来 runner。D7-C 执行、D7-D、D7-E、projection、正式测试集、Self 效果结论、Self Updater 和 raw-original 路线全部保持关闭。
