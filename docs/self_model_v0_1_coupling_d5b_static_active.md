# Self Model v0.1 Coupling-D5B：项目内 Active 路径静态集成

日期：2026-08-21  
状态：无模型、真实隐藏维 shape 的静态夹具通过；D5C 真实模型机制冒烟未授权

## 目标

D5B 新增独立的 project-local active wrapper，复用 D3B/D4B 已验证的锁定源码 AST 注入点，但不修改已经冻结的 OFF runtime。它把 callback 从“永远为 None”扩展为可在无模型夹具中临时绑定，并在成功或异常后恢复全部实例属性。

本门仍不是 Self Model：callback 使用 synthetic probe，不读取 Self Store，不训练 projection，也不对应任何真实层选择。

## 真实 shape、假设备

静态 residual 使用与 2.9B 配置一致的隐藏维 2560：

- `forward_one`：`[2560]`；
- `forward_seq`：`[T, 2560]`；
- dtype 标签：`float16`；
- device 标签：`fake-cuda:0`。

所有对象都是纯 Python fixture。runtime 强制要求 base fixture 明确声明 `offline_static_fixture=true`、`model_loaded=false`、`model_executed=false`；真实对象或缺少标记的对象在编译/调用前失败。

## Active wrapper 边界

- 继续锁定 `rwkv==0.8.32` 与 `model.py` SHA-256；
- 继续只在 post-FFN residual 处插入一个 None-guarded callback；
- 同时覆盖 `forward_one` 与 `forward_seq`；
- OFF 与 zero-scale 都不把 callback 绑定到模型；
- active 只接受明确标记为 synthetic、未加载模型、非真实 projection 的 callback；AST在每层调用，callback对mask外层原样返回，并分别精确记录调用数与实际应用数；
- fake layer mask 只用于夹具覆盖，不是 2.9B 真实层选择；
- 非有限向量、hidden dimension、shape、dtype、device 或路径不匹配均失败关闭；
- temporary methods 与 callback attribute 在成功和异常后都恢复。

## 本轮不包含

未探测服务器 installed source，未导入 RWKV/Torch，未访问权重，未加载或执行模型，未选择真实层，未构造真实 Self projection，未运行 Self 效果实验或 Self Updater。

下一步只能先设计 Coupling-D5C 的真实2.9B非Core机制冒烟、安全入口、synthetic probe、固定非选择性层位置与单次授权文本；D5B 通过不直接授权真实模型执行。

冻结的下一门确认文本为：

> 确认进入 Self Model v0.1 Coupling-D5C 真实2.9B非Core机制冒烟设计与无模型安全入口实现；不授权模型加载或执行、D5D/D5E、正式测试集、Self效果结论、Self Updater或自动重跑。
