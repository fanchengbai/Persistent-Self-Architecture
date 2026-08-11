# Self Model v0.1：真实RWKV adapter的coupling-off等价设计

日期：2026-08-11

状态：设计草案；真实adapter、active injection和模型执行均未实现或授权

## 1. 目的

无权重fake callback只能证明接口契约自洽，不能证明项目内改造后的RWKV执行路径仍与固定
`rwkv==0.8.32`一致。本设计把“真实模型可以接入callback”和“关闭callback时没有改变模型”
拆成独立门。任何active Self注入之前，必须先通过coupling-off等价验证。

固定上游为`RWKV_x070`，`model.py` SHA-256为
`75482aee89a08d2a8c8dbe628110b317fc8d0974ddffbaa52aa19190667305e0`，运行标志固定为
`RWKV_V7_ON=1`、`RWKV_JIT_ON=0`、`RWKV_CUDA_ON=0`。

## 2. 项目内实现，不修改安装包

未来adapter只能放在项目路径`src/psa/self_model/rwkv7_coupling_adapter.py`。不得改写服务器
`site-packages/rwkv/model.py`，不得在运行时用字符串替换已安装源码，也不得把修改后的文件伪装
成上游`rwkv`包。

项目内文件必须记录上游包版本、完整源码digest和使用的结构锚点。digest不一致时，在导入模型
或读取权重前失败关闭。当前轮不创建该adapter文件。

## 3. 两级coupling-off门

### OFF-G1：passthrough wrapper

关闭状态下，wrapper直接调用原始`model.forward`，不构造callback、Self projection、gate或
layer mask。这一门验证上层adapter API没有偷偷改变token、state、`full_output`或返回值。

### OFF-G2：instrumented runtime

未来项目内instrumented runtime需要同时覆盖`forward_one`和`forward_seq`，但callback关闭时
必须完全绕过注入分支。OFF-G2把该路径与固定上游实现逐位比较。OFF-G2通过前：

- active injection入口必须抛出权限错误；
- 真实layer mask保持空数组；
- 序列注入策略保持`unfrozen`；
- 不得开展层搜索、scale扫描或Self效果测试。

OFF-G1通过不能代替OFF-G2，因为直接委托原始forward没有覆盖未来instrumented代码。

## 4. 未来2.9B等价测试标准

模型测试尚未授权。未来若单独授权，固定比较规则为：

1. 只使用明确的非Core token ID夹具，不读取正式测试集；
2. 同一进程、同一个已加载模型实例；
3. 每种token shape先执行一次相同shape warmup，处理已知首次shape调用效应；
4. 覆盖`forward_one`和`forward_seq`；
5. 输入state覆盖`None`和同一来源的cloned restored snapshot；
6. 序列路径覆盖`full_output=false/true`；
7. baseline与adapter分支各自接收独立clone，来源snapshot保持不变；
8. logits必须`torch.equal`；每个next-state tensor也必须`torch.equal`；
9. shape、dtype和device完全一致；callback调用数为0；Self projection未构造。

任一检查失败即停止。不得在看到失败后改用容差、top-1一致或重新挑选样本来宣布通过；若未来
有充分工程原因需要不同标准，必须先形成新设计并重新授权，不能自动重跑。

## 5. state所有权

上游RWKV会更新传入的state list。为保持上游语义，未来adapter不应暗中改变这个约定。实验
runner负责从只读来源snapshot分别clone baseline和adapter输入。等价报告必须同时证明来源
snapshot未变化，并比较两个分支返回的全部state组件。

## 6. 分阶段授权

- D1：本轮设计和静态校验；
- D2：只实现off-only项目内adapter，不导入/加载模型；
- D3：服务器只运行静态和fake测试，不导入/加载模型；
- D4：新的明确授权才允许加载2.9B并执行off等价测试；
- D5：只有OFF-G1/G2通过后，才能重新设计并授权active injection。

“继续”只进入紧接着的一门，不自动跨越后续门。当前D1之外全部未授权。
