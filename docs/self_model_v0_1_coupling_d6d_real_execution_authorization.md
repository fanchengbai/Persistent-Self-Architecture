# Self Model v0.1 Coupling-D6D real joint execution authorization

日期：2026-08-25  
状态：项目负责人已逐字授权一次真实联合工程实验；机器授权与 single-use claim 尚未创建

在 D6D-II installed source、manifest 与真实入口远程无模型门闭环后，项目负责人给出配置中冻结的精确授权：

> 授权执行 Self Model v0.1 Coupling-D6D 真实2.9B单一联合projection训练与非Core pilot一次（同一进程、同一wrapper、16次只读训练capture后冻结真实projection，再按12个fixture各1次OFF预条件和11条件调度执行144次pilot，共160次forward），并授权观察本次工程结果；不授权重跑D5C/P1/P2/D6C或D6D、自动重跑、D6E、正式测试集、Self效果结论、Self Updater、raw-original路线或任何拆分机制运行。

该授权只允许一次 D6D 真实 2.9B 单一联合工程尝试。服务器 runner 必须在最终干净 `main` 上创建新的机器授权，并把授权绑定到该提交、冻结 config、training/pilot manifests 以及 installed-source 静态报告。随后必须先创建并消费唯一 execution claim，才可访问模型资产、加载模型和执行 forward。

单次联合实验必须使用同一进程和同一 wrapper：先完成 16 次只读 residual capture，构造并持久化冻结真实 projection；只有 artifact digest 已落盘后才可加载 pilot payload，并按冻结计划执行 144 次 blinded non-Core pilot。成功或任何失败都会消费本次机会并停止。

本授权不允许重跑 D5C/P1/P2/D6C 或 D6D，不允许自动重跑、D6E、正式测试集、Self 效果结论、Self Updater、raw-original 路线或拆分出的机制运行。即使工程 pilot 的全部方向性门通过，结果也只能分类为 non-Core engineering evidence，不能直接形成 Self 效果结论。

本文件只持久化项目负责人的人类授权。创建机器授权、execution claim、真实 projection、加载模型或执行 forward 必须发生在服务器冻结 runner 中，不能由本文件本身触发。
