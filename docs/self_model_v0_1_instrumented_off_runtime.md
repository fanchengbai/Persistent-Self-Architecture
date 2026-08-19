# Self Model v0.1：OFF-G2 instrumented-off runtime

日期：2026-08-19

状态：项目内OFF-G2代码已实现；真实模型逐位等价尚未执行，active injection未实现或授权

## 1. 为什么需要第二个OFF门

OFF-G1只把调用直接交给原始`base_model.forward`，因此没有经过未来的残差注入代码位置。
OFF-G2必须让同一个调用经过带有instrumentation的`forward_one`或`forward_seq`，但关闭态不能
构造或调用callback，也不能构造Self projection。只有真实模型下OFF-G2与原始路径逐位一致，
才能证明instrumentation本身没有改变基线。

## 2. 项目内实现

本实现不复制整份`rwkv/model.py`，也不修改`site-packages`。它要求已安装源码版本和SHA-256
与冻结值完全一致，然后：

1. 用Python AST找到`RWKV_x070.forward_one`和`forward_seq`；
2. 每条路径必须恰好出现一次`RWKV_x070_CMix(...)`后紧跟`x = x + xx`；
3. 只在该post-FFN残差位置插入一个`callback is not None`分支；
4. OFF-G2运行时把项目命名空间下的callback属性固定为`None`；
5. 临时把两条变换后方法绑定到同一base model实例，让原始公开`forward`继续负责dispatch；
6. 无论成功还是异常，都在`finally`中删除临时方法和callback属性。

任一类名、方法、CMix位置、残差加法、源码摘要或临时属性冲突都会失败关闭。runtime模块自身不
导入`rwkv`或`torch`；当前服务器门只读取源码字节并执行AST变换，不编译或调用真实方法。

## 3. 本轮能证明与不能证明的内容

纯fake测试覆盖单token、序列`full_output=false/true`、state结果、异常恢复、active拒绝和临时
绑定清理。服务器静态门将确认真实安装源码能产生`forward_one=1`、`forward_seq=1`两个注入点。

本轮不能证明真实2.9B的logits和96个state组件逐位一致，因为模型没有加载或执行。因此：

- `off_g2_implemented=true`只表示代码路径已存在；
- `real_model_equivalence_executed=false`仍然成立；
- callback、Self projection、active injection、真实层mask和序列策略都没有构造或冻结；
- 不得把本轮当作添加Self Model后的效果证据。

## 4. 下一道门

下一步是单独授权的D4真实2.9B OFF等价门。它应在同一进程、固定同shape预热后比较原始路径与
OFF-G1/OFF-G2，覆盖单token/序列、None/restored state及`full_output=false/true`；logits与
所有state tensor必须`torch.equal`。任何差异都停止，不得改用容差或top-1掩盖。
