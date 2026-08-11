# Self Model v0.1：无权重双路径残差回调契约

日期：2026-08-11

状态：纯Python fake runtime工程验证；没有导入或加载真实RWKV模型

## 1. 目标

上一轮静态调查确认，`rwkv==0.8.32`的RWKV-7实现把逐层运算直接写在
`forward_one`和`forward_seq`中，没有可直接使用的逐block模块hook。本轮先用无权重夹具固定
最小回调形状：

`callback(phase, layer_index, layer_name, execution_path, residual_x, self_vector) -> residual_x`

只有`post_ffn_residual`阶段进入回调。回调请求不携带recurrent state，因此不能直接重写
原生state组件。

## 2. 两条执行路径

`FakeRWKV7ResidualRuntime`提供两个明确入口：

- `forward_one`：残差shape为`[hidden_dimension]`；
- `forward_seq`：残差shape为`[T, hidden_dimension]`。

两个入口共享同一个逐层运行壳和同一个callback类型，但在调用记录中保留各自的
`execution_path`。fake runtime在进入计算前复制每层3组件的输入state；返回的next state可以
因注入影响后续层而变化，但来源state不得被修改。

## 3. off、scale和安全语义

- `enabled=false`时不调用callback；
- `scale=0`时也不调用callback；
- 两种关闭方式的残差和next state都必须与完全没有callback的基线逐位相同；
- active callback必须保持shape、dtype和device元数据；
- Self vector维度、layer mask、phase或执行路径不匹配时失败关闭；
- 所有device名称以`fake-`开头，防止把夹具输出误写成真实GPU证据。

## 4. 序列策略和层选择

夹具暂用`broadcast_all_tokens_fake_only`，用于证明`[T,D]`广播实现和shape保护。该名称明确
标记为fake-only；它不冻结真实RWKV将来是全token广播、只作用最后token还是逐token计算。

层mask只包含`fake-layer-01`，也不对应2.9B的任何真实层。真实序列策略和真实层选择字段均为
false。

## 5. 本轮不包含

- 不导入`rwkv.model`或`torch`；
- 不访问模型权重，不加载或执行模型；
- 不修改服务器`site-packages`；
- 不实现真实hook、projection或可训练gate；
- 不运行2.9B coupling-off等价测试、层候选测试或Self效果实验。

下一轮若继续，应先设计“真实2.9B只验证coupling-off等价性”的最小adapter方案和授权边界，
不能把本轮fake通过直接解释为真实模型兼容或Self有效。
