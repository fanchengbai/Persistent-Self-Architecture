# Self Model v0.1 离线接口设计

版本：0.1 Draft  
日期：2026-08-11  
状态：纯离线工程接口；真实 RWKV coupling、模型执行和效果实验均未授权

## 1. 本轮目标

本轮把 Phase 3 的抽象架构落实为可测试接口，但不声称已经实现神经 Self Model：

1. 静态、版本化、带 checksum 的 Structured Self Store；
2. 字段级 Self Encoder 接口和 deterministic fake encoder；
3. 可关闭、可缩放、可选择字段和层的 gated residual coupling 接口；
4. Self 字段 swap、encoded-field matched random 和 coupling-off 消融；
5. 所有测试只使用未加载模型的 fake adapter。

本轮明确不包含自动 Self Updater、自然语言 Prompt 注入、真实 RWKV 层 hook、训练、
非 Core 效果实验、正式测试集或正式 Self 实验。

## 2. Self Store

Self State v0.1 保留六类字段：

- `identity_anchors`：受保护；
- `preferences`、`capability_estimate`：慢变量；
- `active_goals`、`confidence`、`uncertainty_conflicts`：快变量。

每个字段项都必须记录 item ID、typed value、confidence、update class、创建/更新时间、
证据引用和状态。顶层记录 agent、trajectory、parent、step、模型/Tokenizer兼容标识以及
payload SHA-256。`SelfStore.save` 使用独占创建，已存在的 state ID 不允许覆盖；读取时重新
验证结构与 checksum。

v0.1 只有静态构造、快照读取和实验性字段 swap，没有通用 `update()`。这防止接口存在后
被误解为已经允许模型自主改写身份或目标。

## 3. Self Encoder

真实目标仍是字段 embedding 加小型 MLP/attention，输出字段级表示和整体表示。本轮只实现
`DeterministicHashFakeSelfEncoder` 来验证数据契约：

- 输入只能是已验证的结构化 Self State；
- field mask 必须显式、非空且无重复；
- 每个字段独立产生固定维度向量，再聚合成整体向量；
- 不序列化为自然语言，不占用 Prompt；
- fake 输出明确标记 `model_loaded=false`，不能作为研究证据。

embedding dimension=16 仅用于离线夹具，不是未来真实 RWKV coupling 的冻结维度。

## 4. Coupling

接口目标为 gated residual：

`residual(layer) = scale × gate(layer) × projection(layer, encoded_self)`

离线 fake adapter 支持：

- `enabled=false` 或 `scale=0` 时完全不调用 injection；
- scale 范围固定为 `[0, 2]`；
- layer mask 只能选择 adapter 声明的层；
- encoder/coupling dimension 必须一致；
- 每层记录 gate、scale、residual norm 和 digest；
- 非 fake adapter 或任何 `model_loaded=true` 对象均失败关闭。

当前 `fake-layer-00/01` 只是接口夹具。真实 RWKV 注入层、projection、gate 参数和训练方式
仍未冻结，也没有实现 layer hook。

## 5. 消融接口

- field mask：只编码指定 Self 字段；
- field swap：交换两份静态 Self 的指定字段，来源对象保持不变；
- encoded random：按字段和 seed 生成确定性、L2 norm 匹配随机向量；
- coupling off：gate 路径完全不产生 adapter 调用；
- scale：检查效应是否能够随注入强度变化；
- layer mask：未来定位作用层，当前只验证接口。

## 6. 下一授权门

本轮完成后仍不能加载模型。下一步若继续，应先做“真实 RWKV coupling 接口调查”：只读
确认可用 hook/activation/state 接口并设计最小 real adapter，同时冻结非 Core A–E 对照和
通用能力副指标。任何真实模型加载、层注入或效果实验需要新的明确授权。
