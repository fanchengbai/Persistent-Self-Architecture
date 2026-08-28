# Self Model v0.1 Coupling-D6D 失败纯离线 state 初始化诊断

日期：2026-08-28  
状态：纯离线根因闭环；未修改真实 runner，未建立独立后继实验

## 诊断范围

本轮只读取已提交的 D6D authorization/claim/failure 观察、冻结项目源码及纯 Python fixture。没有导入 `rwkv.model` 或 Torch，没有访问权重、加载或执行模型，也没有修改 `d6d_ii_joint_runtime.py`、真实 runner、manifest 或已消费的执行边界。

冻结事实保持不变：D6D 唯一 claim 已消费；第一次训练 forward 未完成；0/16 capture、无 projection artifact、0/144 pilot；D6D 与自动重跑均未授权。

## AST 边界证据

源码级审计同时确认：

1. `execute_projection_training()` 的真实调用明确向 wrapper 传入 `state=None`；
2. `D6DIIWrapperOwnedRuntime.forward()` 直接选择 `forward_one()` 或 `forward_seq()`，自身没有调用 `generate_zero_state()`；
3. 冻结 RWKV 接口审计证明上游 public `forward()` 才包含 `state == None` → `generate_zero_state()`，随后再向两个子方法分派；
4. 既有真实入口 wrapper 测试传入 `_state()` 生成的 96 分量预构造 state，没有覆盖真实训练入口的 `state=None` 语义。

这四项把调用栈中的 `NoneType` 下标错误与一个确定的边界缺口连接起来：wrapper 在拥有 instrumented 子方法的同时，没有保留 public forward 的零状态初始化语义。

## 纯 Python 三角复现

同一份 32 层、96 state 分量合成上游执行三种情况：

- public `forward(tokens, None)`：调用一次 `generate_zero_state()` 并成功；
- 当前 wrapper-owned 直达 `forward_seq(tokens, None)`：在 dispatcher 之前精确复现 `TypeError: 'NoneType' object is not subscriptable`；
- 当前 wrapper-owned 直达 `forward_seq(tokens, prebuilt_state)`：成功经过 32 层 dispatcher，且调用方 state 不变。

因此失败不是 AST 注入点、callback、projection 数值或 CUDA 随机性所必需的解释。它发生在这些机制之前，并解释了为什么只使用预构造 state 的无模型测试会通过。

## 路线判定

在修正此边界后，继续使用同一模型、同一 training/pilot manifests、同一 160-call 计划来取得原本预期的 D6D 结果，实质上仍是“修复后重跑 D6D”。已有授权明确禁止该行为，不能通过改名或新目录规避。

D6E 的前置工程证据也没有形成，因此 D6E 继续关闭。本诊断没有建立一个科学上独立的后继实验；单纯修复 wrapper 不是新研究问题。

未来若要提出独立路线，至少必须同时满足：新的研究问题而非仅修 bug、新的预注册阶段身份、全新的授权/claim/输出命名空间、模型前覆盖 `state=None` 的真实协议兼容门，并且绝不复用 D6D claim 或把已声明的 160-call attempt 当作未发生。

当前决定为：关闭 D6D，不实现修复、不重跑，也不自动进入 D6E。下一步只能由项目负责人审阅该停止判定；本轮不授予模型执行、正式测试集、Self 效果结论或 Self Updater 权限。
