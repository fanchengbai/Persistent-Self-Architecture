# Self Model v0.1 D4 首次真实 OFF 等价门失败观察

## 结论

2026-08-20，项目负责人在服务器提交
`a4d110c5de5a8e638137c0559a15de8941172eed`上执行了已授权的单次 D4。
运行本身有效，但总门失败：`valid=false`，状态为
`d4_real_off_equivalence_failed`。这份记录不授权重跑，也不把失败改写为通过。

服务器先完成 24 项 D4/OFF-G1/OFF-G2 组合测试，全部通过。真实模型运行加载了
固定 RWKV-7 G1h 2.9B 权重，耗时约 10.65 秒，峰值显存
6,129,678,336 字节。single-use execution claim 已消费，claim SHA-256 为
`2900bf111031f878bf18004d3f7123439fdfea62dabf8da0a67eefb54e7479de`。

报告 SHA-256 为
`39d4611a6d50791f1677f9eb27e6fb2ea702151a26236fa4094699b821ca721a`；
根据项目 canonical JSON 规则在本机从负责人粘贴的完整报告独立重算，结果一致。

## 六个单元

原始 baseline 与 OFF-G1 在 6/6 单元中 logits 和全部 96 个 state tensor 都
`torch.equal`。OFF-G2 在 5/6 单元中也完全逐位一致：

- 单 token + 克隆恢复态；
- 序列 + `state=None` + `full_output=false`；
- 序列 + `state=None` + `full_output=true`；
- 序列 + 克隆恢复态 + `full_output=false`；
- 序列 + 克隆恢复态 + `full_output=true`。

唯一失败单元是
`forward_one__none__full_output_false`。该单元中 OFF-G2 的 logits 与 baseline
不相等；state 路径、shape、dtype 和 device 全部一致，但 `state[4]` 至
`state[95]` 共 92 个组件不是 `torch.equal`，`state[0]` 至 `state[3]` 相等。

其余安全检查全部通过：恢复态来源未改变，OFF-G1/OFF-G2 callback 计数均为
0，Self projection 未构造，OFF-G2 临时方法绑定已恢复，三条路径的预热和计分
调用数符合冻结协议。active injection、Self 效果实验、正式确认性决定和自动
重跑均未发生。

## 可以和不能解释的内容

当前证据确认 OFF-G1 关闭包装对真实模型透明，但 OFF-G2 尚未通过完整关闭态门。
因为 OFF-G2 在相同单 token 的恢复态以及全部序列单元都逐位一致，不能把结果
简化为“instrumented 方法必然改变运算”；但唯一失败又集中在单 token 的
`state=None` 路径，也不能事后把它当作可忽略的偶然误差。

这次报告只保存逐位相等与否，没有保存数值误差幅度或逐次调用轨迹，因此无法仅
凭现有数据区分以下解释：首次/调用顺序相关的 CUDA 数值效应、临时方法绑定后的
路径特定预热效应，或 `state=None` 分支中的真实实现差异。任何区分这些解释的
模型诊断都属于新的执行，必须先冻结设计并获得新的独立授权；不得复用本次已消费
的授权，也不得通过改用容差、top-1 或删除失败单元来宣布 D4 通过。

在新的模型诊断获批之前，只允许离线审计现有代码、报告和调用顺序。D5 active
设计与 Self 效果实验继续暂停。
