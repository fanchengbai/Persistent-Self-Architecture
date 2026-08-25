# Self Model v0.1 Coupling-D6D-I 首次远程无模型观察

日期：2026-08-25  
状态：功能与安全验收通过；跨平台报告摘要差异已定位，等待修正版远程复验

服务器首次运行 D6D-I 专项测试得到 10/10 `OK`。报告中的 20 类联合验收、14 项总检查和全部配置检查均为真；基础模型 fixture 实例字典未变化，installed source、RWKV/Torch、权重、模型、真实 projection、效果结论和后续实验权限均未触发。

服务器报告 digest 为 `e263b9e5ccd844e22db4f89495277381e2ecac35b68a77f2b65d573d0922351a`，与本地首次报告 `b9267f24b1c37263bd73d001d9fa63530cdce6f63e386d2867872c45796b4290` 不同，因此本轮没有误报为完全闭环。逐字段比较确认：源码 inventory、artifact/parameter digest、所有检查、计数和最终路线输出 digest 均一致；唯一差异是 `self_identity_goal_norm_matched_random` 的中间 projection digest。服务器值为 `a6a08d9453b45e655f58f8dd62ac33eb66036c3dec5833c3c0ec230a3db6b6db`，本地值为 `7fc71465e5d43de429c6ce98d3a211b48b479211e592ae38a2eada5555a28ee1`。

该向量经标准库高斯采样、中心化、L2 norm 和缩放生成；不同平台数学库会在最后数位产生舍入差异。最终 forward 输出摘要仍完全相同，说明差异局限于中间证据的原始浮点序列化，而不是 wrapper 行为或模型路径。

修正版把 projection 路线证据摘要规范为有限浮点数的固定 12 位科学计数法字符串后再执行 canonical JSON SHA-256，并加入相邻 IEEE-754 数值摘要相同、材料级变化摘要不同的回归测试。projection 数值、forward 数值、20 类验收、D6D 设计和全部安全边界均不改变。

在服务器拉取修正版并再次得到跨主机一致报告前，D6D-I 仍标记为“功能通过、摘要闭环待复验”；D6D-II 不提前开放。
