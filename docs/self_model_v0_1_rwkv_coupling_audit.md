# Self Model v0.1：RWKV-7 coupling 接口静态调查

日期：2026-08-11
状态：只读源码调查完成；未导入 `rwkv.model`、未访问权重、未加载或执行模型、未实现真实 hook

## 1. 调查对象与边界

本轮调查服务器环境中固定安装的 `rwkv==0.8.32`。`rwkv/model.py` 的 SHA-256 为
`75482aee89a08d2a8c8dbe628110b317fc8d0974ddffbaa52aa19190667305e0`。项目2.9B配置固定：

- `RWKV_V7_ON=1`；
- `RWKV_JIT_ON=0`；
- `RWKV_CUDA_ON=0`；
- 32层、`n_embd=2560`、`head_size=64`；
- 已有云端接口证据记录96个 recurrent-state组件，即每层3个。

审计器只通过安装包元数据定位并读取源码文本，不导入 `rwkv.model`，也不读取5.9GB模型
权重。版本、源码digest或必要结构标记任一变化时失败关闭。

## 2. 实际执行类与公开接口

源码末尾在 `RWKV_V7_ON=1` 时把公开的 `RWKV` 名称指向 `RWKV_x070`。由于项目固定
`RWKV_JIT_ON=0`，该类继承 `torch.nn.Module`，但内部各层不是独立的PyTorch子模块：
权重存放在 `self.z` 映射中，attention、FFN和残差操作直接写在 `forward_one` 与
`forward_seq` 的循环里。

因此：

- 根模块的 `forward_pre_hook` 只能改 token/state 输入；
- 根模块的 `forward_hook` 只能看到 logits/state 输出；
- 没有可直接注册的“第N个block activation hook”；
- 只使用公开根hook无法实现当前设计的层内 gated residual coupling。

## 3. 残差流和 state 生命周期

单token与序列路径的结构一致：

1. token embedding成为残差流 `x`；
2. 进入逐层循环；
3. layer norm后运行TimeMix，更新该层 `att_x_prev` 与 `att_kv`；
4. `x = x + attention_output`；
5. layer norm后运行ChannelMix，更新该层 `ffn_x_prev`；
6. `x = x + ffn_output`；
7. 全部层结束后执行输出layer norm与词表head。

当传入 `state=None` 时，运行时创建每层3个组件。传入已有list时，各层组件会在调用中被
替换/更新，返回的仍是持续状态。因此任何continuous/restored/swap/off对照都必须在进入
forward前复制来源state，不能让一个条件的运行污染另一个条件。

残差张量形状为：

- 单token：`[2560]`；
- 序列：`[T, 2560]`。

真实Self projection必须在注入点匹配 `x` 的device和dtype。序列路径是对全部token广播
同一个Self residual、只改最后一个token，还是逐token计算，尚未决定；进入实现前必须冻结。

## 4. 候选接口判定

| 候选边界 | 可行性 | 当前判定 |
|---|---|---|
| 修改公开 `state` 输入 | 无需改包，现有adapter已经支持 | 只作为原生state基线/对照；会混淆显式Self与原生recurrent state，不作为主coupling |
| embedding后、第一层前修改 `x` | 工程上可行 | 影响全部层，但不能定位层效应；保留为候选 |
| attention残差后修改 `x` | 工程上可行 | 会继续影响同层FFN；保留为候选 |
| FFN残差后修改 `x` | 工程上可行，边界清楚 | 作为最小真实原型的优先接口族；尚未选择具体层 |
| 输出layer norm/head后修改logits | 易实现 | 属于输出策略控制，不足以证明内部Self coupling，仅可作对照 |

“优先接口族”不是最终层选择，也不是已实现hook。当前没有运行任何比较来选择层。

## 5. 最小未来实现边界

若下一轮获得独立授权，最小真实adapter应在项目代码中维护，不能直接修改服务器
`site-packages`。接口应同时覆盖 `forward_one` 和 `forward_seq`，并提供类似：

`callback(phase, layer_index, residual_x, self_vector) -> residual_x`

的显式回调。第一版只应开放 `post_ffn_residual`，并满足：

- coupling off完全绕过回调，得到原始代码路径；
- layer mask、scale和gate均进入证据记录；
- Self projection输出严格匹配残差shape/device/dtype；
- 来源Self State和来源recurrent state均保持不可变；
- 单token与序列语义一致；
- 包版本、源码digest和模型配置不匹配时拒绝运行；
- 不修改 `v_first` 或三个原生state组件的含义。

在真实模型测试前，先用无权重的fake runtime验证两个forward路径、off逐位等价、shape、
clone隔离和失败关闭。具体层集合、projection参数、gate训练方式和非Core效果实验仍需要后续
独立设计与授权。

## 6. 本轮结论

真实coupling在当前RWKV包上可实现，但不存在开箱即用的逐层hook。最稳妥的下一步是先实现
项目内的“双路径残差回调壳 + fake runtime契约测试”，仍不加载模型；通过后再单独授权
2.9B非Core coupling-off等价性与小规模层候选测试。本轮没有产生Self效果证据。
