# Self Model v0.1 Coupling-D6D 真实联合实验观察

日期：2026-08-28  
状态：唯一真实 2.9B D6D 联合尝试有效失败；single-use claim 已消费，禁止重跑

## 执行身份与授权边界

本次机器授权绑定 Git commit `563c3144d23f1a10e27c3e4377952f165fd0230f`、D6D-II config digest `f9f53635f64d66a9f8d309f004d0a8acd8f7c68ba47ab223e08d3f2c2db4af1f`、training manifest digest `a0aa594f6b020b03546fa95d2cd135783b97c72436e84d31d27c37f369dab0fd`、pilot manifest digest `4c18addedb26de4fc80438ad687d2606df5a68511cde96661b3b87e4c60023b2` 和 installed source digest `75482aee89a08d2a8c8dbe628110b317fc8d0974ddffbaa52aa19190667305e0`。这三个项目文件 digest 已在本地重新计算并与回传一致。

机器授权内部 digest 为 `fabe99e84af60b7edc193f72de2d6617a3cdeb89569a7fcc8b3148e55c96d5fc`；claim 记录的授权文件 SHA-256 为 `3e514c9a8afa3d2aaf7a49b4b9f80a972160ccccb8a03dbd18a00af211a8a468`。二者用途不同：前者摘要授权 payload，后者绑定实际授权文件字节。

single-use claim 状态为 `d6d_single_use_joint_execution_claim_consumed`，claim 文件 SHA-256 由 failure 绑定为 `421a4c811722e2600abe08bf742b43faa56f2162e9075c1668710187f5fed909`。claim 冻结 16 次训练 forward、144 次 pilot forward、总计 160 次，并明确 `d6d_rerun_authorized=false`、`automatic_rerun_authorized=false`。

## 失败位置

failure 状态为 `d6d_real_joint_attempt_failed_claim_consumed`，类型为 `TypeError`，错误为 `'NoneType' object is not subscriptable`，报告 digest 为 `5fc3570762c4d64e30de328711d1f5b9596876d87adf30898fe89a1e2a7ba65d`。

调用栈显示失败发生在第一个训练 record：

1. `execute_projection_training()` 以 `state=None` 调用 wrapper；
2. `D6DIIWrapperOwnedRuntime.forward()` 根据多 token 输入直接调用编译后的 `forward_seq()`；
3. 真实 `forward_seq()` 在访问 recurrent state 分量时对 `None` 执行下标操作并失败。

冻结 installed source 的 public `forward()` 原本负责在 `state == None` 时调用 `generate_zero_state()`，然后才分派到 `forward_one()` 或 `forward_seq()`。D6D wrapper 为保持方法所有权在外部，直接分派到 instrumented 子方法，因而绕过了这一步初始化。

无模型测试没有发现该问题，是因为真实入口 wrapper 测试向 instrumented 方法传入了预构造的 96 分量 fake state；它验证了绑定、dispatcher、AST 注入和生命周期，却没有覆盖“真实训练入口以 `state=None` 开始，同时 wrapper 必须保留上游 public-forward 初始化语义”的组合边界。

## 可作出的结论

2.9B 模型已经加载，第一次训练 forward 已被调用但没有完成。训练 callback 尚未到达有效 capture，16 次 capture 未完成，真实 projection artifact 未构造或冻结，144 次 pilot 没有开始。因此本次失败不能评价：

- synthetic 正控制是否有效；
- frozen Self projection 是否能改变模型行为；
- identity/goal swap、mask 或 random 对照；
- 通用能力 sentinel；
- Self Model 是否产生类似“自我状态”的因果效果。

失败属于 wrapper-owned dispatch 与 RWKV zero-state 初始化之间的真实接口缺口，不是 projection 数值、pilot 统计或 Self 假设本身的正负证据。

本次 claim 已消费，授权明确禁止 D6D 重跑和自动重跑。当前不得修改代码后重启同一实验，也不得拆出一轮机制运行。后续最多先进行新的纯离线失败闭环与研究路线审查；任何修复实现、真实模型执行、D6E、正式测试集、Self 效果结论或 Self Updater 都需要新的明确边界，本文件不授予这些权限。
