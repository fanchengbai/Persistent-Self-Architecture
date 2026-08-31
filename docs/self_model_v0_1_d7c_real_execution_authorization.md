# Self Model v0.1 D7-C real public semantics compatibility execution authorization

日期：2026-08-31
状态：首次授权因生命周期修复保持未消费并失效；修复远程复验后，项目负责人已在最终修正版上重新逐字授权，等待单次执行

在 D7-C 设计与无模型安全入口服务器复验闭环后，项目负责人给出配置中冻结的精确授权：

> 授权执行 Self Model v0.1 D7-C 真实2.9B public语义兼容门一次（固定8个public OFF/wrapper zero等价cell共16次调用，加2次synthetic active，共18次forward；不访问calibration/held-out payload），并授权观察本次兼容结果；不授权D7-C重跑、自动重跑、D7-D/D7-E、projection实现或构造、D6D重跑、正式测试集、Self效果结论、Self Updater或raw-original路线。

该授权只覆盖一次 D7-C 真实 2.9B public 语义兼容尝试：8 个 cell 各执行一次 public OFF 与 wrapper zero，共 16 次；另执行两次 synthetic active，总计 18 次 forward。它不读取 calibration/held-out payload，不实现或构造 projection，也不形成 Self 效果结论。

执行前只读生命周期审计发现，原入口先创建机器授权，再在验证机器授权时重新构造静态报告；原报告会把新出现的机器授权记为 artifact 状态变化，导致验证摘要与机器授权绑定的创建前摘要不同。若直接启动，runner 会在 installed source 探测、claim、权重和模型之前失败，并遗留不可复用的机器授权。

因此本次授权没有被执行或消费：机器授权和 claim 均未创建，installed source 未探测，权重未访问，模型未加载或执行。项目内现只进行纯无模型生命周期修复，使创建后的验证规范化复现同一创建前 artifact 摘要，同时继续严格复算配置、Schema、源码和 Git 绑定；首次创建仍要求全部执行 artifact 缺席。

修复会改变锁定入口与测试源码摘要。修复提交必须先通过本地专项/全量测试和服务器无模型复验；之后项目负责人必须在最终提交上重新给出同一逐字授权，才能创建机器授权、消费 single-use claim、探测 installed source、访问权重或执行模型。本文件本身不触发任何执行。

## 修复后重新授权

D7-C 授权生命周期修复已在服务器通过 14 项专项测试、14 项入口检查、15 项配置检查和 14 类 synthetic acceptance；修正版静态报告 digest=`f1134f2684f3935a9c1151a8e0b2d263564d9bc36844b0281e3da00b3b56748e`。相关远程观察已持久化到最终 `main` 提交 `d20e42cfa570cda14199f6103b60ed26ec7d94c7`。

在上述修复与远程无模型门闭环后，项目负责人于 2026-08-31 再次给出配置中完全相同的逐字授权：

> 授权执行 Self Model v0.1 D7-C 真实2.9B public语义兼容门一次（固定8个public OFF/wrapper zero等价cell共16次调用，加2次synthetic active，共18次forward；不访问calibration/held-out payload），并授权观察本次兼容结果；不授权D7-C重跑、自动重跑、D7-D/D7-E、projection实现或构造、D6D重跑、正式测试集、Self效果结论、Self Updater或raw-original路线。

这次重新授权只在包含本记录的最终干净 `main` 上有效。当前仅持久化人类授权：机器授权和 execution claim 尚未创建，installed source 尚未探测，权重尚未访问，模型尚未加载或执行。服务器 runner 一旦创建并消费 single-use claim，无论成功或失败都必须停止且不得重跑。
