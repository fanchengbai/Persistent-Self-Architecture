# Self Model v0.1 D4A：首次 D4 失败后的最小诊断设计

## 1. 边界

D4A 是 D4 失败后的新诊断，不是 D4 重跑，也不是新的 OFF 通过门。当前轮只完成
离线审计与设计；没有实现真实诊断入口，没有导入模型、访问权重或运行推理。
D4 的 `valid=false` 永久保留，D5 active injection 继续暂停。

## 2. 离线审计结果

D4 的六个单元采用固定顺序，单 token 的 `state=None` 永远先于单 token 恢复态。
每个单元内部又固定依次执行原始、OFF-G1、OFF-G2，每条路线先丢弃一次预热输出，
再只保存一个计分输出。于是失败单元同时是：第一个计分单元、OFF-G2 最早的
`forward_one` 调用阶段和唯一尚未经历此前 OFF-G2 单 token 单元的时点。现有报告
没有保存预热输出或同一路线重复性，无法把 state 模式与调用年龄分离。

OFF-G2 与原始方法还存在 callback 分支以外的工程边界：它从 AST 重新编译所选
方法，清空 decorator，复制上游 globals，并在每次调用时把两条新方法临时绑定
到模型实例。D4 没有“执行相同重编译/绑定但不插入 callback 分支”的控制路线，
因此无法判断差异来自重编译/绑定，还是 None-guarded instrumentation 本身。

报告只保存 `torch.equal` 布尔值，没有每次调用的 tensor digest、非相等元素数、
最大/平均绝对误差；确定性算法也未启用。这些都是定位信息缺失，不是放宽 D4
逐位标准的理由。

## 3. 最小未来诊断

未来若另行授权，只使用原失败夹具：token `[2764]`、`state=None`、
`full_output=false`。不再运行已通过的恢复态和序列单元，也省略已在 D4 6/6
通过的 OFF-G1。

新增三条纯关闭路线：

1. `original_baseline`：上游原始方法；
2. `g0_recompiled_unmodified`：与 OFF-G2 使用相同的 AST 选择、decorator 清空、
   globals 复制和临时方法绑定，但不添加 callback 属性或分支；
3. `off_g2_instrumented`：当前 None-guarded OFF-G2。

三轮采用拉丁顺序：

1. original → G0 → G2；
2. G0 → G2 → original；
3. G2 → original → G0。

每条路线恰好在第一、第二、第三位置各出现一次，共 9 次模型调用。所有调用都
保留，不设置或丢弃预热。每次保存 logits 和 96 个 state tensor 的
shape/dtype/device及内容SHA-256；离线比较全部同路线与跨路线组合，记录
`torch.equal`、非相等元素数、最大/平均绝对误差和首个失败组件。

这个设计可区分：同一路线是否随调用稳定、重编译/绑定控制 G0 是否已产生差异，
以及只有 G2 分支是否不同。它仍不能自动把 D4 改成通过。任何未来执行都必须有
新的 single-use claim、新结果目录和项目负责人独立授权；完成或异常后都停止。

## 4. 禁止事项

- 不复用 D4 已消费的 claim 或授权；
- 不用容差或 top-1 作为通过标准；
- 不删除原失败单元或覆盖失败报告；
- 不实现 active callback、Self projection、层选择或效果实验；
- 不因诊断结果自动重跑或进入 D5。
