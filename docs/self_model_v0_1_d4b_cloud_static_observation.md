# Self Model v0.1 D4B 服务器无模型静态复验观察

## 1. 观察范围

本记录只观察项目负责人贴回的 D4B 离线设计验证器输出，不运行或重跑模型。
贴回片段包含完整报告和空的 `git status --short`，但没有包含此前命令中的
`git rev-parse HEAD` 与 37 项组合测试输出，因此本记录不补写未观察到的提交号
或服务器测试计数。

## 2. 完整性核对

- 状态：`d4b_steady_state_off_design_static_verified`；
- `valid=true`；
- 22/22 项设计检查全部为真；
- D4 调用轨迹重建为 37 次；
- D4A 调用轨迹重建为 9 次；
- 报告保存 digest 为
  `7f3cfb7fecf6892532f9ecdb27f528716b363077e37e288f8940d8e883ef658d`；
- 从贴回完整 JSON 移除自 digest 字段后按项目 canonical JSON 规则独立复算，
  得到相同 digest；
- 八个冻结源码 digest 与本地最终报告一致；
- 服务器 `git status --short` 无输出。

这些证据足以确认 D4B 设计内容在服务器上完成了源码级跨主机静态复验。它们
不用于推断本次未贴出的准确 Git 提交号或组合测试计数。

## 3. 安全边界

报告中 runtime、机器授权、执行 claim、RWKV/Torch 导入、权重访问、模型加载与
执行、D4 状态变化、D5、active injection、Self 效果实验和自动重跑字段全部为
假。因此本轮不构成 D4B runtime 实现或模型等价实验，也不改变 D4 的失败结论。

## 4. 下一门

D4B 离线设计的本地与服务器静态门已经闭合。下一步只能在项目负责人另行确认
后进入 D4B runtime 的纯离线/fake-first 实现与失败关闭测试；该确认不得解释为
允许创建真实执行 claim、加载2.9B模型、运行D4B、进入D5、active injection或
Self效果实验。
