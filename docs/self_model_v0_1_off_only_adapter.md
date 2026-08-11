# Self Model v0.1：D2 off-only adapter

日期：2026-08-11

状态：OFF-G1项目内wrapper已实现；OFF-G2、active injection和模型执行均未实现或授权

## 1. 实现范围

`RWKV7CouplingOffAdapter`只实现OFF-G1。它验证固定上游版本和`model.py` digest后，在coupling
关闭时执行：

`base_model.forward(tokens, state, full_output)`

wrapper不复制、不编码也不修改tokens或state，不改变`full_output`，并原样返回底层结果。底层
若按上游语义更新传入state，该更新仍发生在同一个对象上；D2不暗中clone或改变所有权。

## 2. active路径默认拒绝

唯一合法请求是`CouplingOffRequest(mode="off", enabled=false, scale=0)`。任何其他对象、非零
scale、enabled=true或`forward_active`调用都必须在底层`forward`之前抛出`PermissionError`。

该文件没有导入`rwkv`或`torch`，没有callback、Self projection、layer mask、序列注入策略或
instrumented RWKV循环。`callback_call_count`固定为0，`self_projection_constructed`固定为
false。

## 3. 当前验证方式

D2只用纯Python fake base：fake base记录收到的对象identity与`full_output`，返回预先创建的
sentinel对象，并模拟上游原位更新state。测试要求wrapper收到和返回的对象保持相同identity，
异常也原样传播，active拒绝时fake base调用数保持0。

服务器复跑同一组fake-base测试仅证明提交可复现，不构成D3：本轮不读取服务器已安装RWKV
源码，不导入模型，不访问权重，也不执行2.9B。

## 4. 后续门

- D3：独立确认后，服务器只读核对已安装源码与wrapper静态结构，仍不加载模型；
- D4：D3通过并再次授权后，才加载2.9B执行OFF-G1/OFF-G2等价测试；
- active injection必须等待OFF-G2逐位通过后的新设计与新授权。
