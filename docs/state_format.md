# PSA 状态与 Checkpoint 格式规范

> 版本：v0.2-dev
> 状态：Impl-1 state 契约与 Impl-2 开发容差已固定；云端 L3/100 次恢复门已通过，格式仍待 official reset 与兼容性门后冻结
> 日期：2026-07-29  
> 依赖：[`architecture.md`](architecture.md)、[`task_design.md`](task_design.md)、[`evaluation_protocol.md`](evaluation_protocol.md)  
> 目标：定义原生 recurrent state、显式 Self State、耦合状态和审计记录如何安全、可验证、可恢复地持久化。

## 1. 设计目标

状态格式必须支持：

1. 同进程和跨进程 save / restore；
2. 不同轨迹之间的完整 state swap；
3. state reset、尺度匹配 random、ablation 和 interpolation；
4. Self State 的字段级交换、版本化更新和 rollback；
5. checkpoint fork 和不可变历史；
6. 模型、tokenizer、dtype、kernel 和代码版本兼容检查；
7. bitwise、numerical 和 behavioral 三层恢复验证；
8. 文件损坏、写入中断和错误版本的明确拒绝；
9. 实验结果与状态来源的完整追踪；
10. 不依赖可执行反序列化格式读取不可信研究资产。

## 2. 状态类型

PSA 禁止用一个模糊的 `state` 文件同时表示所有对象。

| 类型 | 符号 | 内容 | 首次使用阶段 |
|---|---|---|---|
| Native State | \(R_t\) | RWKV 原生 recurrent tensor state | EXP-001 |
| Structured Self State | \(S_t\) | 字段化、版本化 Self 数据 | 显式 PSA |
| Encoded Self | \(z_t\) | Self Encoder 输出 | 显式 PSA |
| Coupling State | \(c_t\) / \(\psi\) | coupling 参数、gate 配置或运行快照 | 显式 PSA |
| World Working State | \(W_t\) | 当前任务与环境状态 | 可选 |
| External Memory | \(M_t\) | 事件、事实与检索记录 | Memory 基线 |
| Evidence | \(e_t\) | Self 更新依据 | Self Evolution |
| Ledger | — | 版本、操作和来源事件 | 全阶段 |

EXP-001 的最小 checkpoint 只要求 `Native State + Manifest + Provenance`。不存在的显式 Self 组件不得用空文本冒充“已实现”。

## 3. Checkpoint 目录布局

```text
checkpoint_<opaque-id>/
├─ manifest.json
├─ native_state/
│  ├─ tensors.safetensors
│  └─ inventory.json
├─ self_state/
│  └─ state.json
├─ coupling/
│  ├─ adapter.safetensors
│  └─ runtime.json
├─ world_state/
│  └─ state.json
├─ provenance/
│  ├─ events.jsonl
│  └─ evidence.jsonl
├─ validation/
│  ├─ restore_probe.json
│  └─ checksums.sha256
└─ metrics/
   └─ state_summary.json
```

规则：

- `manifest.json` 必须存在；
- `native_state/` 在 EXP-001 checkpoint 中必须存在；
- 未使用的可选目录应省略，不创建含糊的空占位；
- checkpoint 目录一旦提交即只读；
- 新更新、新干预和新验证均创建新 checkpoint 或旁路报告。

## 4. 标识符

所有 ID 都是 opaque string，不从路径、实验条件或正确答案推导。

| 字段 | 含义 |
|---|---|
| `checkpoint_id` | 当前 checkpoint |
| `trajectory_id` | 一条连续 Agent 轨迹 |
| `parent_checkpoint_id` | 直接父 checkpoint |
| `fork_root_id` | 分叉共同起点 |
| `run_id` | 产生 checkpoint 的运行 |
| `experiment_id` | 如 EXP-001 |
| `sample_id` | 任务样本 |
| `state_id` | 某个逻辑状态对象 |

禁止在 ID 中编码：

- I/G 真值；
- 答案位置；
- 实验条件；
- state 来源类别；
- 数据切分。

这些信息只存放于 manifest 的受控元数据，避免文件名泄漏。

## 5. Manifest

### 5.1 必需字段

```json
{
  "format_version": "0.1",
  "checkpoint_id": "opaque-id",
  "checkpoint_kind": "native_state",
  "created_at": "RFC3339 timestamp",
  "experiment_id": "EXP-001",
  "run_id": "opaque-id",
  "trajectory_id": "opaque-id",
  "parent_checkpoint_id": null,
  "fork_root_id": "opaque-id",
  "environment_step": 0,
  "model": {},
  "tokenizer": {},
  "runtime": {},
  "state_components": [],
  "input_boundary": {},
  "provenance": {},
  "integrity": {},
  "status": "complete"
}
```

### 5.2 模型标识

```json
{
  "model": {
    "family": "RWKV",
    "architecture": "RWKV-7",
    "model_id": "",
    "revision": "",
    "weight_digest_sha256": "",
    "parameter_count": null,
    "layer_count": null,
    "hidden_size": null,
    "implementation": "",
    "implementation_revision": ""
  }
}
```

模型 ID 或文件名相同不足以证明权重相同，必须记录 revision 或权重 digest。

### 5.3 Tokenizer 标识

```json
{
  "tokenizer": {
    "tokenizer_id": "",
    "revision": "",
    "vocab_digest_sha256": "",
    "special_tokens": {},
    "normalization": "",
    "implementation": ""
  }
}
```

restore 不能只检查 vocabulary size。

### 5.4 Runtime 标识

```json
{
  "runtime": {
    "os": "",
    "python": "",
    "framework": "",
    "framework_version": "",
    "cuda": "",
    "driver": "",
    "gpu_model": "",
    "dtype": "",
    "kernel_family": "",
    "deterministic_mode": false,
    "code_commit": "",
    "config_digest_sha256": ""
  }
}
```

不得把主机名、用户名、密钥路径等敏感信息写入公开 manifest。

### 5.5 输入边界

checkpoint 必须说明 state 对应到哪一个 token 边界：

```json
{
  "input_boundary": {
    "last_token_id": null,
    "tokens_consumed": 0,
    "prefix_digest_sha256": "",
    "conversation_boundary": "",
    "eot_seen": false,
    "next_expected_input_digest_sha256": null
  }
}
```

保存了正确 tensor 却在错误 token 边界恢复，会产生无法解释的结果。

## 6. Native Recurrent State

### 6.1 Tensor 文件

首选不可执行的 tensor 容器，例如 `safetensors`。最终库和版本在实现规范冻结。

禁止把不可信 checkpoint 默认保存在会执行任意对象反序列化的格式中。若官方实现只能输出其他格式，转换过程必须在受控环境完成，并记录原始与转换后 digest。

### 6.2 RWKV-7 World 0.4B 实测状态契约

2026-07-29 的 Impl-1 云端接口调查使用固定权重
`RWKV-x070-World-0.4B-v2.9-20250107-ctx4096`、`rwkv==0.8.32`、
PyTorch 2.12.0 + CUDA 13.2、`cuda fp16` 策略和纯 PyTorch kernel，观察到：

| 项目 | 实测值 |
|---|---:|
| 层数 | 24 |
| 每层组件数 | 3 |
| tensor 总数 | 72 |
| tensor 净载荷 | 6,389,760 bytes（6.09375 MiB） |
| 捕获设备 | `cuda:0` |
| 有限性 | 72/72 全部 finite |

每层严格按以下顺序出现：

| 列表位置 | 稳定名称 | role | shape | dtype | 每层字节数 |
|---|---|---|---|---|---:|
| `state[3L]` | `layers.L.att_x_prev` | attention 前一 token 激活 | `[1024]` | `torch.float16` | 2,048 |
| `state[3L+1]` | `layers.L.att_kv` | attention recurrent KV | `[16,64,64]` | `torch.float32` | 262,144 |
| `state[3L+2]` | `layers.L.ffn_x_prev` | FFN 前一 token 激活 | `[1024]` | `torch.float16` | 2,048 |

其中 \(L\in[0,23]\)。24 个 `att_kv` 共占 6 MiB，48 个 FP16
激活向量共占 96 KiB。因此保存时必须保留混合精度，不能为了格式统一把
`att_kv` 降为 FP16，也不能把两个 FP16 分量无依据地升为 FP32。

接口调查 fixture 的 state digest 为
`9d63d57dac737cb49ee0e95bb10359699b2af23e1d500c4af15d763143514d84`。
该 digest 只标识固定开发前缀形成的那一份具体状态，不是所有合法 state
都必须相同的格式常量。

### 6.3 Inventory

`inventory.json` 为每个 tensor 记录：

```json
{
  "name": "layers.0.component",
  "shape": [],
  "dtype": "",
  "device_at_capture": "",
  "numel": 0,
  "byte_length": 0,
  "sha256": "",
  "statistics": {
    "min": null,
    "max": null,
    "mean": null,
    "std": null,
    "rms": null,
    "l2_norm": null,
    "nan_count": 0,
    "inf_count": 0
  }
}
```

state statistics 用于诊断，不能替代 tensor checksum。

### 6.4 Tensor 容器

Impl-2 固定使用 `safetensors==0.8.0`：

- 文件为 `native_state/tensors.safetensors`；
- key 使用 6.2 节的稳定逻辑名称；
- 保存前转为 contiguous CPU tensor，但不改变 dtype 或数值；
- 加载时先在 CPU 安全读取，完成 checksum、名称、shape 和 dtype 检查后再移动到目标设备；
- 不使用 pickle 或 `torch.load` 读取 PSA native-state checkpoint。

### 6.5 Impl-1 内存恢复基线

同一 prefix snapshot 上两次运行相同 9-token suffix 的结果：

- logits：bitwise exact，最大绝对误差 0；
- 最终 state：72/72 tensor bitwise exact，最大绝对误差 0；
- tokenizer prefix/suffix roundtrip：exact；
- 该基线只证明同进程内存 clone/restore，不替代磁盘和跨进程验证。

### 6.6 Impl-2 跨进程开发容差

第一轮 RTX 5090 跨进程开发运行确认 checkpoint checksum、tensor inventory、
shape、dtype 和内容摘要均精确恢复，但独立进程续算不是 bitwise exact：

| 指标 | 100 次实测最大误差 | Impl-2 开发阈值 |
|---|---:|---:|
| 最终 logits 最大绝对误差 | 0.03125 | 0.0625 |
| 最终 state 最大绝对误差 | 0.05822181701660156 | 0.125 |

阈值采用“大于两倍实测最大值的下一个二进制整齐边界”，只用于非确认性
Impl-2 开发门。重跑同时固定随机种子、cuBLAS workspace、PyTorch
deterministic algorithms、最高 FP32 matmul 精度并关闭 TF32。

L3 开发门要求 100/100 次：

- checkpoint tensor 与 inventory checksum 精确；
- logits/state shape 与 dtype 兼容；
- 误差不超过上述阈值；
- 最终 logits top-1 token 一致。

Bitwise exact 次数和同一子进程内的重复一致性继续记录为诊断指标，但不与
“序列化是否无损”混为同一判断。正式确认批次的最终容差仍需在开发门结果
复核后预注册。

2026-07-29 重跑结果：

| 项目 | 结果 |
|---|---:|
| checkpoint validation | L2，checksum/inventory/model compatible 全部通过 |
| 跨进程恢复 | L3 |
| 容差通过 | 100/100 |
| logits top-1 一致 | 100/100 |
| 跨进程 bitwise exact | 0/100 |
| 子进程内相对首轮 bitwise exact | 1/100 |
| tensor 净载荷 | 6,389,760 bytes |
| SafeTensors 文件 | 6,396,096 bytes |
| 容器开销 | 6,336 bytes（约 0.099%） |
| checkpoint 保存 | 0.088768 s |
| 完整子进程门 | 14.021353 s |

该结果将“无损保存”与“低精度续算的逐位确定性”明确区分：前者已由逐
tensor SHA-256 验证；后者在当前 RWKV/PyTorch/CUDA 栈上不成立，但其数值
漂移稳定落在开发容差内且不改变 top-1。

### 6.4 保存 dtype

默认按捕获时原始 dtype 保存，不擅自转换为 fp32 或更低精度。

若实验需要转换：

- 创建新 checkpoint；
- `operation_type=cast`；
- 记录 source checkpoint；
- 记录转换前后 dtype 和误差；
- 不把转换 checkpoint 当作原始 state。

## 7. Structured Self State

### 7.1 逻辑结构

```json
{
  "schema_version": "0.1",
  "state_id": "opaque-id",
  "agent_instance_id": "opaque-id",
  "parent_state_id": null,
  "trajectory_id": "opaque-id",
  "step": 0,
  "identity_anchors": [],
  "preferences": {},
  "capability_estimate": {},
  "active_goals": [],
  "confidence": {},
  "uncertainty_conflicts": [],
  "provenance_refs": [],
  "integrity": {}
}
```

### 7.2 字段项

每个可变字段项至少包含：

```text
field_item_id
value
value_type
confidence
update_class
created_step
updated_step
source_evidence_ids
status
```

`update_class` 只能来自：

```text
protected
slow
fast
```

### 7.3 不允许的内容

Self State 不得默认包含：

- 完整对话记录；
- 未筛选的 Memory 文本；
- 自由生成的长篇人格描述；
- 模型的全部 chain-of-thought；
- 密钥或凭据；
- 未声明来源的“自我总结”。

## 8. Evidence 与 Ledger

### 8.1 `events.jsonl`

每行一个不可变事件：

```json
{
  "event_id": "opaque-id",
  "event_type": "capture",
  "step": 0,
  "actor": "experiment_controller",
  "source_checkpoint_id": null,
  "target_checkpoint_id": "opaque-id",
  "parameters": {},
  "result": "success",
  "timestamp": "RFC3339 timestamp"
}
```

### 8.2 事件类型

```text
initialize
capture
restore
fork
swap
reset
randomize
ablate
interpolate
cast
self_update_proposed
self_update_rejected
self_update_committed
archive
validate
migrate
```

自然状态更新与实验干预必须使用不同事件类型。

### 8.3 `evidence.jsonl`

只在显式 Self Update 阶段使用，记录：

```text
evidence_id
target_field
observation_refs
source_type
reliability
direction
strength
conflicts
accepted
decision_rule
```

## 9. Coupling State

显式 PSA checkpoint 将 coupling 分为：

1. 参数状态：Self Encoder、projection、gate 参数；
2. 运行状态：字段 mask、layer mask、scale、gate 输出摘要。

参数文件与基础模型权重分离，以便：

- 明确基础模型仍然冻结；
- 单独 checksum；
- coupling-off；
- 比较不同 adapter；
- 避免把 adapter 混进 RWKV checkpoint 后无法区分。

运行时 gate 的每 token 全量输出通常进入 run artifact，不必全部复制到 checkpoint；checkpoint 只记录恢复所需配置和摘要。

## 10. Checksum

### 10.1 算法

统一使用 SHA-256：

- 每个 payload 文件计算 digest；
- `checksums.sha256` 列出 payload 的相对路径和 digest；
- payload 不包含 `manifest.json` 和 `checksums.sha256` 本身，避免循环摘要；
- manifest 的 `integrity.payload_root_digest_sha256` 对排序后的 payload digest 清单再计算一次 digest；
- checkpoint 进入传输包时，由外层 bundle checksum 覆盖 manifest、checksum 清单和全部 payload。

### 10.2 JSON 字节规范

写入 JSON 时固定：

- UTF-8；
- LF 换行；
- key 排序；
- 明确小数格式；
- 禁止 NaN / Infinity；
- 文件末尾一个换行；
- 不写与语义无关的动态空白。

checksum 针对实际文件字节，不针对解析后的对象。

### 10.3 校验顺序

```text
目录完整性
→ manifest schema
→ 单文件 checksum
→ root digest
→ tensor inventory
→ model compatibility
→ restore probe
```

任一步失败都不得静默继续。

## 11. 写入事务

checkpoint 创建采用两阶段提交：

```text
1. 创建 sibling temporary directory
2. 写入全部组件
3. flush / fsync 可用内容
4. 计算 payload checksums 与 payload root digest
5. 写最终 manifest，status=complete
6. 再次 flush / fsync
7. 在 temporary directory 内完成 L0/L1 验证
8. 原子 rename 到最终 opaque directory
```

规则：

- 最终目录已存在时拒绝覆盖；
- temporary 目录不视为 checkpoint，即使其中已有 `status=complete` manifest；
- 清理遗留 temporary 目录必须是明确维护操作；
- 不能先创建最终目录再逐步补文件。

## 12. 兼容性级别

### L0：Readable

- 文件存在；
- JSON 可解析；
- tensor 容器可读取。

只说明可读。

### L1：Structurally Valid

- schema 通过；
- checksum 通过；
- inventory 与 tensor 匹配；
- 无 NaN/Inf（除非明确允许）。

### L2：Model Compatible

- model family/revision 匹配；
- layer/component/shape 匹配；
- tokenizer 和特殊 token 匹配；
- dtype 和 kernel 策略兼容。

### L3：Numerically Restorable

- restore probe 在冻结容差内；
- 候选 logits 和 state 摘要通过；
- 同一输入边界可继续运行。

### L4：Behaviorally Restorable

- 预注册行为分布在容差内恢复；
- 不仅 top-1 偶然一致。

EXP-001 正式 state 至少达到 L3；rollback 结论需要 L4。

## 13. Validation Report

每次验证生成旁路报告，不修改 checkpoint：

```json
{
  "validation_id": "opaque-id",
  "checkpoint_id": "opaque-id",
  "validator_version": "",
  "requested_level": "L3",
  "achieved_level": "L3",
  "checks": [],
  "errors": [],
  "warnings": [],
  "restore_metrics": {},
  "created_at": "RFC3339 timestamp"
}
```

报告本身进入 run artifacts，并计算 checksum。

## 14. State 操作规则

### 14.1 Restore

必须提供：

- checkpoint ref；
- target model spec；
- expected input boundary；
- requested validation level。

默认拒绝：

- model revision 不同；
- tensor shape 不同；
- tokenizer digest 不同；
- checkpoint 非 complete；
- checksum 失败。

### 14.2 Swap

完整 swap 要求：

- 同一 model weight digest；
- 同一 state schema/inventory；
- 同一 dtype；
- 同一任务输入边界；
- 配对轨迹除目标变量外匹配。

swap 创建新运行记录，不修改来源 checkpoint。

### 14.3 Reset

RWKV-7 World 0.4B 的 Impl-2b reset 候选定义为官方包
`rwkv.model.RWKV.forward(tokens, state=None)` 路径。`None` 是 reset
操作的语义标记，不是一个可与普通 72-tensor state 混淆的空 checkpoint。

开发门必须从同一 `None` reset 边界重复运行相同 token 序列，检查 shape、
dtype、数值容差和 top-1 一致性。正式 reset 干预记录
`operation_type=reset`、输入边界与官方实现版本。

不能用“全零 tensor”冒充 reset；只有在另外证明官方初始化确实等于全零后，
全零构造才可作为等价实现。

### 14.4 Randomize

random state 记录：

- random seed；
- 分布；
- 尺度匹配方法；
- 匹配目标统计；
- 是否逐层匹配。

random state 是新 checkpoint，`operation_type=randomize`。

### 14.5 Ablation

记录：

- target component；
- layer/channel mask；
- replacement strategy；
- source state；
- ablation 前后统计。

`zero`、`mean replacement` 和 `matched random` 是不同条件。

### 14.6 Interpolation

\[
R_\alpha=(1-\alpha)R_A+\alpha R_B
\]

仅当两端：

- shape、dtype、模型版本相同；
- component 一一对应；
- 数值有限；
- 使用相同 token 边界；

才允许插值。

插值是数值干预，不假定语义线性。

### 14.7 Fork

fork：

- 新 trajectory ID；
- 共享 parent checkpoint；
- 保持 fork root；
- 不复制或重写父 ledger；
- 后续证据链独立。

### 14.8 Rollback

rollback 创建新 active checkpoint，引用历史 checkpoint 作为恢复来源。它不是删除新历史，也不是移动版本指针后掩盖中间事件。

## 15. State Diff

`state diff` 至少输出：

- component 是否存在；
- shape/dtype 差异；
- checksum 相等性；
- L1/L2/L∞ 数值距离；
- cosine similarity（适用时）；
- RMS ratio；
- 每层差异摘要；
- NaN/Inf；
- Self 字段级变更；
- provenance 分叉点。

Diff 默认不输出完整 tensor 内容，避免日志爆炸。

## 16. 错误分类

| 代码 | 含义 | 行为 |
|---|---|---|
| `E_FORMAT_VERSION` | 格式版本不支持 | 拒绝 |
| `E_INCOMPLETE` | checkpoint 未完成提交 | 拒绝 |
| `E_CHECKSUM` | 完整性失败 | 拒绝 |
| `E_MODEL_MISMATCH` | 权重或架构不匹配 | 拒绝 |
| `E_TOKENIZER_MISMATCH` | tokenizer 不匹配 | 拒绝 |
| `E_SHAPE_MISMATCH` | state shape 不匹配 | 拒绝 |
| `E_DTYPE_MISMATCH` | dtype 不兼容 | 默认拒绝 |
| `E_BOUNDARY_MISMATCH` | token 边界不匹配 | 拒绝 |
| `E_NUMERICAL` | NaN/Inf 或恢复超容差 | 拒绝正式实验 |
| `E_SCHEMA` | Self schema 无效 | 拒绝 |
| `E_PERMISSION` | 受保护字段更新无权限 | 拒绝更新 |

错误不能退化成 warning 后继续确认性实验。

## 17. 格式迁移

迁移规则：

1. 原 checkpoint 永不修改；
2. 迁移生成新 checkpoint；
3. 记录 source format、target format 和 migration tool revision；
4. 迁移前后做 checksum、inventory 和 restore comparison；
5. 不可逆或有损迁移必须显式标记；
6. 正式实验批次内禁止混用不同格式版本。

v0.x 期间不保证向后兼容，但每次破坏性变更必须增加 `format_version`。

## 18. 保留与发布

### 18.1 必须长期保留

- 所有正式实验 checkpoint manifest；
- 关键 original/restored/swapped state；
- checksums；
- validation reports；
- provenance；
- 产生论文核心图表的状态。

### 18.2 可按策略压缩

- 重复的非关键开发 state；
- 可从父 state 和确定性操作重建的中间 artifact；
- 大型逐 token gate trace。

删除或压缩前必须有独立保留策略，且不能破坏复现核心结论。

### 18.3 公开发布

发布前检查：

- 无凭据和主机敏感信息；
- 模型权重许可；
- 数据许可；
- checkpoint 是否意外包含用户文本；
- manifest 是否足以复现但不过度暴露基础设施。

## 19. 最小测试集

实现后至少覆盖：

1. checkpoint roundtrip；
2. 中断写入不产生 complete checkpoint；
3. checksum 篡改被发现；
4. 错误模型 revision 被拒绝；
5. 错误 tokenizer 被拒绝；
6. shape/dtype mismatch 被拒绝；
7. reset 与官方初始化一致；
8. random state 可由 seed 重建；
9. swap 不修改来源；
10. interpolation 端点 \(\alpha=0/1\) 等于来源；
11. fork 父子关系正确；
12. rollback 不删除中间历史；
13. Self protected field 无权限更新被拒绝；
14. L3 restore probe 通过；
15. 错误 token boundary 被拒绝。

## 20. 冻结前待填写

- [x] RWKV-7 官方实现的 state component 清单；
- [x] 精确 tensor 名称、shape 和 dtype；
- [x] tensor 容器库及版本；
- [x] 官方 reset `state=None` 的云端验证；
- [ ] kernel compatibility 规则；
- [x] 开发 restore probe 输入；
- [x] Impl-2 开发数值容差；
- [ ] 确认性实验最终数值容差；
- [x] Impl-2 checkpoint 大小和开发存储预算；
- [x] manifest JSON Schema 文件；
- [x] Self State JSON Schema 文件；
- [ ] 格式测试向量；
- [ ] 将状态改为 Frozen for Implementation。

Impl-2 开发容差现已固定；在 official reset、kernel compatibility 和确认性
实验最终容差完成前，本规范仍为实现候选。
