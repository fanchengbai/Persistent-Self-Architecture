# Self Model v0.1 D7-C real public semantics compatibility execution authorization

日期：2026-08-31
状态：项目负责人已逐字授权；执行前生命周期审计发现无模型缺陷，本次授权保持未消费并暂停

在 D7-C 设计与无模型安全入口服务器复验闭环后，项目负责人给出配置中冻结的精确授权：

> 授权执行 Self Model v0.1 D7-C 真实2.9B public语义兼容门一次（固定8个public OFF/wrapper zero等价cell共16次调用，加2次synthetic active，共18次forward；不访问calibration/held-out payload），并授权观察本次兼容结果；不授权D7-C重跑、自动重跑、D7-D/D7-E、projection实现或构造、D6D重跑、正式测试集、Self效果结论、Self Updater或raw-original路线。

该授权只覆盖一次 D7-C 真实 2.9B public 语义兼容尝试：8 个 cell 各执行一次 public OFF 与 wrapper zero，共 16 次；另执行两次 synthetic active，总计 18 次 forward。它不读取 calibration/held-out payload，不实现或构造 projection，也不形成 Self 效果结论。

执行前只读生命周期审计发现，原入口先创建机器授权，再在验证机器授权时重新构造静态报告；原报告会把新出现的机器授权记为 artifact 状态变化，导致验证摘要与机器授权绑定的创建前摘要不同。若直接启动，runner 会在 installed source 探测、claim、权重和模型之前失败，并遗留不可复用的机器授权。

因此本次授权没有被执行或消费：机器授权和 claim 均未创建，installed source 未探测，权重未访问，模型未加载或执行。项目内现只进行纯无模型生命周期修复，使创建后的验证规范化复现同一创建前 artifact 摘要，同时继续严格复算配置、Schema、源码和 Git 绑定；首次创建仍要求全部执行 artifact 缺席。

修复会改变锁定入口与测试源码摘要。修复提交必须先通过本地专项/全量测试和服务器无模型复验；之后项目负责人必须在最终提交上重新给出同一逐字授权，才能创建机器授权、消费 single-use claim、探测 installed source、访问权重或执行模型。本文件本身不触发任何执行。
