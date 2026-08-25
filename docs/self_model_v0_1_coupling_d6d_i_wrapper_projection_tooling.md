# Self Model v0.1 Coupling-D6D-I wrapper 与 projection 工具

日期：2026-08-25  
状态：无模型工程实现；installed source、真实 projection、模型执行和效果结论未授权

## 本轮完成什么

D6D-I 把上一轮设计落实为两个项目内接口：

1. `D6DIWrapperOwnedRuntime`：编译后的 `forward_one`、`forward_seq`、固定 dispatcher 和 request context 全部属于外部 wrapper。wrapper 自己提供 `forward`，基础模型对象只接受只读属性委托；
2. frozen projection tooling：从字段分离的开发训练记录学习 identity/goal 两组权重，生成无 bias、带参数与 artifact digest 的冻结资产，并提供 matched、swap、mask、norm-matched random 投影接口。

本轮纯 Python 验收使用 synthetic teacher 构造 2560 维 fixture artifact。它测试真实 artifact 格式和算法，但明确 `fixture_only=true`、`research_evidence_eligible=false`，不能充当已经构造真实 Self projection 的证据。

## wrapper ownership

旧 D6C 把 callback 和双方法写入真实 RWKV 实例，然后错误依赖这些名字直接存在于实例字典。D6D-I 不对基础实例调用 `setattr`、`delattr` 或直接字典写入：

- constructor 前后记录基础实例字典的精确键和对象身份；
- 每个 forward 前后重复核验；
- instrumented 方法仅通过 `MethodType` 绑定到 wrapper；
- callback 属性只存在于 wrapper；
- OFF、zero、synthetic 和全部 Self 条件使用同一个 wrapper；
- wrapper 的 forward 不调用 `base_model.forward`，而是按单 token/序列直接调用自己的 instrumented 方法；
- 嵌套与并发请求在第二个内部 forward 前拒绝；
- callback 异常恢复 context、丢弃未提交输出，runtime 可继续用于 OFF。

本轮 runtime 构造器只接受 `model_loaded=false`、`model_executed=false` 的纯 Python fixture。它没有开放真实模型执行入口。

## projection artifact

工具使用 `identity_anchors` 和 `active_goals` 两个独立类别分支。每个类别权重由非 Core 训练记录中的目标分支向量算术平均得到，再以无 bias 的字段和形成最终 residual。真实 artifact 规格固定为 2560 维、zero-based 第 15 层、post-FFN：

- training manifest 与 blinded pilot commitment 必须是不同的 SHA-256；
- identity/goal vocabulary、参数、optimizer/seed、parameter digest 和 artifact digest 全部进入资产；
- 双 mask 严格产生零向量；
- random 在两个字段分支上分别固定 seed 并保持各自 L2 norm；
- artifact 不包含基础模型参数，不做 prompt serialization，不提供在线 update；
- 任一参数或元数据被修改而 digest 未同步时，审计失败关闭。

真实训练数据、真实 projection 参数和真实 artifact 本轮均未生成。

## 同一联合验收

纯 Python 验收在同一个 wrapper 上运行全部 11 条 D6D condition，不拆出 synthetic 机制轮。锁定 AST 的 single/sequence 两条方法各编译一个 post-FFN site；fixture 维度为 32 层、2560 hidden、96 state components。

验收覆盖：OFF=zero=双 mask、synthetic 输出变化、八条 Self projection 路线区分、字段 swap/mask/random 语义、random 分支 norm、artifact tamper、wrapper/base 身份、single/sequence、异常恢复、嵌套/并发拒绝以及所有来源输入不变性。

这仍是无模型工程证据：没有探测 installed source，没有导入 RWKV/Torch，没有读取权重、加载模型或执行真实 forward，也没有产生 Self 效果结论。

## 下一门

服务器复验通过后，下一门才可在新确认下探测 installed source、静态编译真实源码，并实现训练/pilot manifest、授权 Schema、唯一输出和 single-use claim 的无模型入口。本轮不提前开放该门。

冻结确认文本：

> 确认进入 Self Model v0.1 Coupling-D6D-II installed source静态兼容、联合训练/试验manifest与单次真实入口的无模型实现；只允许探测并静态编译锁定installed source、冻结projection训练与pilot清单、实现新Schema/唯一目录/single-use claim入口，不访问权重、不加载或执行模型，不构造真实projection，也不授权D6D真实执行、D6E、正式测试集、Self效果结论、Self Updater、任何历史重跑或自动重跑。
