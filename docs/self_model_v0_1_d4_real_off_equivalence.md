# Self Model v0.1 D4：真实 2.9B OFF 等价门

D4 只回答一个工程问题：在同一个真实 RWKV-7 2.9B 实例中，项目的
OFF-G1 直接包装和 OFF-G2 双路径 instrumented-off 是否与未包装的原始
`forward` 逐位相同。它不是 Self Model 效果实验。

固定矩阵共 6 个计分单元：单 token 使用 `state=None` 与克隆恢复态；序列使用
两种 state 输入，并分别覆盖 `full_output=false/true`。每个单元的原始、OFF-G1、
OFF-G2 三条路径各先执行一次相同形状的不计分预热，然后执行一次计分调用。
logits 的 shape、dtype、device 和内容必须一致；返回 state 的路径、全部组件的
shape、dtype、device 和内容也必须一致。内容判断只使用 `torch.equal`，没有容差、
top-1 或失败后修改标准的后备路径。

runner 只接受干净的 `main`、固定 `rwkv==0.8.32` 源码 digest、固定 2.9B 权重与
tokenizer，以及逐字环境锁。模型加载前会在结果目录独占创建 single-use claim；
运行异常也会保留 claim 和 failure report，自动重跑保持未授权。结果目录非空时
入口直接拒绝。

本门明确排除 active callback、Self projection、真实层选择、Self 效果观察、
确认性结论和自动重跑。D4 通过只意味着“关闭机制没有改变原模型输出”，下一步
D5 仍需新的设计和单独授权。
