# 云服务器开发门运行手册

> 状态：v0.1 草案  
> 范围：下载项目、验证纯逻辑骨架、准备远程模型适配。本文不授权跳过预注册直接运行确认集。

## 1. 下载与固定代码版本

```bash
git clone <repository-url>
cd Persistent-Self-Architecture
git rev-parse HEAD
```

把 commit 写入远程 environment manifest。正式批次不使用浮动的 `main` 或未记录的工作区。

## 2. 创建 Python 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

当前资源下载器只使用 Python 标准库，不要求额外安装 Hugging Face SDK。

## 3. 准备模型、Tokenizer 与开发数据

在项目根目录执行：

```bash
bash scripts/prepare_exp001_assets.sh
```

脚本会按固定 revision 下载 RWKV-7 World 0.4B 和 World tokenizer，将文件放到项目内被 Git 忽略的 `.psa-assets/`，生成下载收据，然后创建 EXP-001 合成开发集。

中断后重复运行同一命令即可续传。若需要使用挂载盘：

```bash
PSA_ASSET_ROOT=/mnt/psa-assets \
  bash scripts/prepare_exp001_assets.sh
```

完整说明见 `docs/asset_management.md`。

当前不下载 RWKV 完整预训练语料，因为 EXP-001 是冻结基础模型上的状态实验，不是预训练或全量微调。

## 4. 安装并验证 Impl-1 GPU 环境

资源准备成功后，在已经激活的 `.venv` 中执行：

```bash
bash scripts/install_impl1_gpu.sh
```

脚本固定安装：

```text
PyTorch 2.12.0 + CUDA 13.2 wheel
NumPy 1.26.4
rwkv 0.8.32
safetensors 0.8.0
```

首轮接口门使用：

```text
RWKV_V7_ON=1
RWKV_JIT_ON=0
RWKV_CUDA_ON=0
```

也就是说先运行官方 `rwkv` 包的纯 PyTorch 路径，不编译自定义 CUDA kernel。这样便于检查 state 结构和数值一致性；性能优化留到接口门通过之后。

安装结束会生成：

```text
results/development/environment_manifest.json
```

只有报告中的 `valid` 为 `true` 才进入模型加载门。若 PyTorch 下载受网络影响，可先运行 AutoDL 的 `source /etc/network_turbo`。

## 5. 运行 Impl-1 模型接口门

环境报告通过后执行：

```bash
bash scripts/run_impl1_interface_gate.sh
```

这个开发门只做：

- 再次校验模型和 tokenizer；
- 加载冻结的 RWKV-7 World 0.4B；
- 验证 tokenizer 编码—解码往返；
- 用开发文本形成一份 recurrent state；
- 清点每个 state tensor 的 shape、dtype、device、大小、有限性和 SHA-256；
- 从同一份内存 snapshot 两次运行相同 suffix；
- 比较两次最终 logits 和 state 是否逐元素一致。

输出位于：

```text
results/development/impl1_model_interface/
├─ model_interface_report.json
├─ state_inventory.json
├─ roundtrip_validation.json
└─ summary.json
```

如果模型在生成上述成功报告前加载失败，目录中会保留：

```text
results/development/impl1_model_interface/failure_report.json
```

失败报告记录异常类型、错误信息、配置路径和时间，不包含模型权重或访问凭据。重新运行成功后，旧的失败报告会自动清除。

这是一次非确认性的工程开发门，不使用 EXP-001 身份/目标任务，也不产生项目主张所需的行为证据。

该门只验证内存内 clone/restore。磁盘序列化、跨进程恢复和 100 次重复属于下一批开发门。

### 5.1 运行 Impl-2 Checkpoint 恢复门

Impl-1 的 `summary.json` 为 `valid: true` 后执行：

```bash
bash scripts/run_impl2_checkpoint_gate.sh
```

该开发门会：

- 形成一份新的原生 recurrent state；
- 保留 48 个 FP16 与 24 个 FP32 tensor 的原始 dtype；
- 用 SafeTensors 写入不可执行的 tensor 容器；
- 通过临时目录、完整性校验和原子 rename 提交 checkpoint；
- 校验权重、tokenizer、输入 token 边界、tensor 名称、shape、dtype 和 SHA-256；
- 启动新的 Python 子进程，重新加载模型；
- 固定随机种子、cuBLAS workspace，启用 PyTorch deterministic algorithms，并关闭 TF32；
- 在子进程中从磁盘加载同一 checkpoint 100 次；
- 每次继续相同 suffix，比较 logits 与最终 state；
- 输出 bitwise exact 比例、容差通过率、top-1 一致率、首次/中位数/P95
  保存恢复耗时和最大数值误差。

主要输出：

```text
results/development/impl2_checkpoint_roundtrip/
├─ summary.json
├─ checkpoints/ckpt-<opaque-id>/
│  ├─ manifest.json
│  ├─ native_state/
│  │  ├─ tensors.safetensors
│  │  └─ inventory.json
│  ├─ provenance/events.jsonl
│  └─ validation/checksums.sha256
└─ runs/run-<opaque-id>/
   ├─ probe_config.json
   ├─ reference.safetensors
   └─ cross_process_restore_report.json
```

通过条件是：独立子进程成立、checkpoint tensor 与 inventory 逐位一致并达到
L2；100/100 次续算保持 shape/dtype 和 top-1 一致，logits 最大绝对误差不超过
`0.0625`、state 不超过 `0.125`，最终 `achieved_level` 为 `L3`。
`exact_repeat_count` 继续作为诊断项，但不再错误地充当跨进程 FP16 的唯一通过
条件。若提前失败，查看：

```bash
cat results/development/impl2_checkpoint_roundtrip/failure_report.json
```

### 5.2 运行 Impl-2b State 操作门

Impl-2 的 `summary.json` 为 `valid: true` 后执行：

```bash
bash scripts/run_impl2b_state_operations_gate.sh
```

该门验证：

- 官方 reset 语义为 `rwkv.model.RWKV.forward(..., state=None)`；
- 两条等长 token 边界分支形成兼容但内容不同的 72-tensor state；
- state diff 输出逐组件 L1、L2、L∞、cosine 和 RMS ratio；
- full-state swap 使用 donor 的深拷贝；
- 两个 source state 在 swap 续算前后 digest 不变；
- reset 与双向 swap 的重复续算均满足 Impl-2 开发容差和 top-1 一致性。

输出：

```text
results/development/impl2b_state_operations/
├─ state_diff_report.json
├─ reset_validation.json
├─ swap_validation.json
└─ summary.json
```

通过后，`summary.json` 中的 `reset_valid`、`state_diff_valid`、
`swap_valid`、`source_states_immutable` 和总 `valid` 均应为 `true`。

### 5.3 运行 Impl-2c Matched Random State 门

Impl-2b 通过后执行：

```bash
bash scripts/run_impl2c_random_state_gate.sh
```

该门为 EXP-001 的 `random_matched` 对照验证：

- 每个 tensor 独立生成零均值高斯噪声；
- 保持 72 个组件的 shape、dtype 和 device；
- 逐组件匹配来源 state 的 L2/RMS 尺度；
- seed 固定且与任务标签和答案无关；
- 同 seed 必须逐位可重建，不同 seed 必须产生不同 digest；
- 来源 state 在随机化及续算后保持不变；
- 随机 state 能完成重复续算且不出现超容差漂移。

输出：

```text
results/development/impl2c_random_matched/
├─ random_state_validation.json
└─ summary.json
```

通过时 `same_seed_bitwise_reproducible`、`different_seed_distinct`、
`scale_match_valid`、`continuation_valid` 和总 `valid` 均为 `true`。

2026-07-30 的 RTX 5090 / CUDA 13.2 开发运行已通过：72 个组件全部满足
上述条件，最大逐组件相对 L2 误差为 `3.824590840690877e-05`
（约 `0.0038%`）。该结果只验证 `random_matched` 是可复现且数值稳定的
原生 state 对照，不赋予随机 state 任何 Self 语义。

### 5.4 运行 Impl-3 Batch 1 开发门

确认 Impl-2、Impl-2b 和 Impl-2c 的 `summary.json` 仍在原路径后执行：

```bash
bash scripts/run_impl3_development_gate.sh
```

该命令会：

1. 校验三个 Batch 0 基础设施 summary；
2. 用目标 tokenizer 筛选候选标签与答案代码；
3. 只按 token 数标定约 128-token 标准 delay；
4. 生成并检查 8 个完整 factorial groups；
5. 运行 32 条 Prompt-visible T0 轨迹；
6. 输出 group-cluster BCa 能力报告和 320-group 资源外推。

输出：

```text
results/development/impl3_development/
├─ batch0_evidence.json
├─ label_pool_report.json
├─ delay_calibration.json
├─ development_dataset.json
├─ task_validation.json
├─ raw_prompt_visible.jsonl
├─ prompt_visible_report.json
├─ resource_estimate.json
└─ summary.json
```

若 `summary.json` 的 `decision` 为 `go` 且 `valid=true`，可进入 Batch 2
参数审阅与冻结；若为 `revise`，先查看 `prompt_visible_report.json` 中失败的
具体门槛。Batch 1 只检验模型是否理解任务，不提供 persistence 或 Self 证据。

首次 v0.1 云端运行得到 `decision=revise`：模型在全部 32 条轨迹中固定选择
A，joint accuracy 为 0.25，两个 marginal accuracy 均为 0.5。A–D 已确认
都是独立且等长的单 token，因此 v0.2 不修改答案分数，也不做事后先验校正；
只把 T0 改成更直接的显式精确字段匹配，并将结果写入新目录。

运行 v0.2：

```bash
bash scripts/run_impl3_development_v02_gate.sh
```

查看：

```bash
cat results/development/impl3_development_v02/summary.json
```

不要删除或覆盖 `results/development/impl3_development/`，它是有效的首次
Revise 记录。v0.2 仍使用相同模型、标签筛选规则、任务 seed、答案代码和
验收阈值。

v0.2 云端结果同样为 `decision=revise`：32/32 条仍选择 A，joint accuracy
为 0.25，两个 marginal accuracy 均为 0.5，format-valid rate 为 0.5。
因此不再继续修改双字段措辞，改运行独立的 Impl-3b 能力阶梯。

### 5.5 运行 Impl-3b 分层能力诊断

保留 v0.2 结果原路径，然后执行：

```bash
bash scripts/run_impl3b_capability_ladder_gate.sh
```

查看：

```bash
cat results/development/impl3b_capability_ladder/summary.json
```

`valid=true` 表示诊断完整运行，不等于任务能力通过。真正的路线判断看：

```text
capability_gate_passed
route_decision
copy_code_valid
single_field_valid
two_field_valid
```

`route_decision` 的含义：

- `revise_checkpoint_or_answer_interface`：连直接照抄代码都失败；
- `revise_single_field_matching`：会照抄，但不会单字段查表；
- `revise_compositional_matching`：前两层通过，双字段组合失败；
- `go_batch2`：三层均通过，可以进入参数冻结。

本次云端结果为 `valid=true`、`copy_code_valid=true`、
`single_field_valid=false`、`route_decision=revise_single_field_matching`。
copy accuracy 为 1.0，single-field accuracy 为 0.25，且后者仍只会选 A。
这说明答案接口有效，但 World 0.4B 不具备 EXP-001 所需的最小查表能力。

### 5.6 运行 Impl-3c G1h 1.5B 候选接口门

不要覆盖 World 0.4B 的任何结果。先下载已固定版本的新候选：

```bash
bash scripts/prepare_g1h_1.5b_candidate.sh
```

然后只运行接口兼容门：

```bash
bash scripts/run_g1h_1.5b_interface_gate.sh
```

查看：

```bash
cat results/development/impl3c_g1h_1.5b_interface/summary.json
cat results/development/impl3c_g1h_1.5b_interface/model_interface_report.json
cat results/development/impl3c_g1h_1.5b_interface/state_inventory.json
```

`valid=true` 只表示 G1h 1.5B 能被当前框架加载、读取和复制 state。下一步仍
要用官方 G1 提示格式重新跑三层能力门；在此之前不要运行正式 state 实验。
完整迁移顺序见 `docs/checkpoint_migration.md`。

本次 Impl-3c 云端结果已经通过：固定权重和 tokenizer 哈希有效，模型加载
成功，接口门 `valid=true`。

### 5.7 运行 Impl-3d G1h 三层能力门

拉取包含 Impl-3d 的提交后运行：

```bash
bash scripts/run_impl3d_g1h_capability_ladder_gate.sh
```

查看：

```bash
cat results/development/impl3d_g1h_1.5b_capability_ladder/summary.json
```

本门对 G1h 现场运行 96 条平衡样本，不读取旧模型的 v0.2 能力结果。Prompt
使用官方 G1 对话结构，末尾没有空格。

判断时区分：

- `valid=true`：程序和全部样本运行完整；
- `capability_gate_passed=true`：copy、single-field、two-field 都达到门槛；
- `route_decision`：若失败，指出第一次失败发生在哪一层。

本次 Impl-3d 结果为 `valid=true`、`capability_gate_passed=false`。候选评分
准确率分别为 copy 1.0、single-field 1.0、two-field 0.875。copy 和
two-field 的自由生成格式率为 0；two-field 的 A/B/C 位置准确率为 1.0，
D 为 0.5。

因此当前 `route_decision=revise_checkpoint_or_answer_interface` 不能直接
解释为模型不会照抄：copy 的候选评分实际为 32/32。进入 Impl-3e 原始输出
审计，在审计前不重跑、不改阈值、不升级 2.9B。

Impl-3e-a 审计结果：

- copy 32/32 都以 ` <think>\n` 开始；
- two-field 32/32 都以 ` <think>We` 开始；
- 4 个 two-field 评分错误全部为 D→B；
- D→B 的分差约为 0.95–1.33，不是数值舍入误差。

### 5.8 运行 Impl-3e-b 官方 fake-think 复验

本轮只修改 Assistant 前缀，使用官方推荐的：

```text
Assistant: <think></think
```

随后固定补入共同的 `>`，再评分 A–D。原模型、96 条样本、seed、标签、
答案代码、阈值和生成长度均与 Impl-3d 相同。

运行：

```bash
bash scripts/run_impl3e_g1h_fake_think_gate.sh
```

查看：

```bash
cat results/development/impl3e_g1h_1.5b_fake_think/summary.json
```

除原有字段外，还要看 `forced_prefix_greedy_exact_rate`。只有该值为 1.0，
才说明固定补入的 `>` 与模型自身的下一 token 一致。

本次 Impl-3e-b 得到 `forced_prefix_greedy_exact_rate=1.0`，two-field
格式率也修复到 1.0；但 two-field accuracy 为 0.84375，区间下界 0.75，
D 位置准确率仍为 0.5。因此 1.5B 能力门未通过，不再修改它的 Prompt。

### 5.9 运行 Impl-3f G1h 2.9B 接口门

下载固定的 2.9B 候选。此前服务器需要镜像，因此建议直接运行：

```bash
HF_ENDPOINT=https://hf-mirror.com \
  bash scripts/prepare_g1h_2.9b_candidate.sh
```

下载器保留 `.part` 并支持断点续传。资源准备完成后运行：

```bash
bash scripts/run_g1h_2.9b_interface_gate.sh
```

查看：

```bash
cat results/development/impl3f_g1h_2.9b_interface/summary.json
cat results/development/impl3f_g1h_2.9b_interface/model_interface_report.json
cat results/development/impl3f_g1h_2.9b_interface/state_inventory.json
```

本次云端结果已经通过接口门：模型实际为 32 层、96 个 state 组件，
state 总量 21,299,200 字节且全部有限；峰值显存 6,232,199,168 字节，
加载、tokenizer roundtrip 与同进程恢复均有效。

### 5.10 运行 Impl-3g G1h 2.9B 能力复验

先确保已经进入项目虚拟环境，然后运行：

```bash
source .venv/bin/activate
bash scripts/run_impl3g_g1h_2.9b_fake_think_gate.sh
```

查看总结果：

```bash
cat results/development/impl3g_g1h_2.9b_fake_think/summary.json
```

这一步不会重新下载 2.9B 权重。它使用与 Impl-3e-b 完全相同的 96 条题、
答案前缀、随机种子和阈值，只把模型配置与接口证据换成 2.9B。重点查看：

- `capability_gate_passed`
- `forced_prefix_greedy_exact_rate`
- `copy_code_metrics`
- `single_field_metrics`
- `two_field_metrics`
- `route_decision`

`valid=true` 只表示诊断流程完整；是否通过能力门以
`capability_gate_passed=true` 为准。

本次结果为 `valid=true`、`capability_gate_passed=false`。不要重跑模型，
先执行只读审计：

### 5.11 审计 Impl-3g 原始输出

```bash
bash scripts/audit_impl3g_g1h_2.9b_results.sh
cat results/development/impl3g_g1h_2.9b_fake_think/audit_report.json
```

该命令不加载 RWKV、不使用 GPU，也不改变原始 JSONL；它只新增
`audit_report.json`，汇总输出变体、混淆矩阵和全部评分错误。

审计确认模型所有 96 条回答都在 `>` 后自然先输出换行，而旧候选评分使用
前导空格。运行只对齐这一边界的 Impl-3i：

### 5.12 运行 Impl-3i 自然换行对齐复验

```bash
bash scripts/run_impl3i_g1h_2.9b_newline_aligned_gate.sh
cat results/development/impl3i_g1h_2.9b_newline_aligned/summary.json
```

这仍使用同一个 2.9B 模型和同一批 96 条题，不需要下载新资源。重点比较
single/two-field 的 `accuracy`、区间下界和 D 位置；自由生成格式仍作为
单独问题保留。

本次结果显示 single-field 已修复到 1.0，但 two-field 仍为 0.875、
D 仍为 0.5。无需重跑模型，直接审计：

### 5.13 审计 Impl-3i 的双字段错误

```bash
python -m psa g1-capability-audit \
  --output-dir results/development/impl3i_g1h_2.9b_newline_aligned
cat results/development/impl3i_g1h_2.9b_newline_aligned/audit_report.json
```

审计结果显示换行前后都是相同 4 个 D-only 错误。停止修改答案边界，运行
答案代码轮换诊断：

### 5.14 运行 Impl-3k 答案代码轮换诊断

```bash
bash scripts/run_impl3k_g1h_2.9b_code_rotation_gate.sh
cat results/development/impl3k_g1h_2.9b_code_rotation/summary.json
```

该门运行 128 条双字段配对题，预计耗时会高于此前 96 条混合层级能力门。
它不下载新模型。查看 `per_code`、`multi_code_error_case_count` 和
`route_decision`。

实际结果有 10/12 个错误集中于 D，原自动路线分类过粗。无需重跑模型，
进行标签边际化复核：

### 5.15 运行 Impl-3l 标签边际化复核

```bash
bash scripts/review_impl3k_g1h_2.9b_code_rotation.sh
cat results/development/impl3k_g1h_2.9b_code_rotation/code_rotation_review.json
```

重点查看 `label_marginalized_accuracy`、`label_marginalized_error_count`
和修订后的 `route_decision`。

实际复核为 32/32 全部正确。能力前置条件已满足，下面按顺序复验 2.9B
状态工程。不要一次把三个脚本连在一起运行；每一步先把 summary 返回审阅。

### 5.16 运行 Impl-3m 2.9B 磁盘恢复复验

```bash
bash scripts/run_impl3m_g1h_2.9b_checkpoint_roundtrip_gate.sh
cat results/development/impl3m_g1h_2.9b_checkpoint_roundtrip/summary.json
```

期望 `valid=true`、`achieved_level=L3`、`tolerance_pass_count=100` 和
`top1_match_count=100`。该门包含独立子进程的 100 次恢复续算。

### 5.17 Impl-3m 通过后运行 Impl-3n 状态操作复验

```bash
bash scripts/run_impl3n_g1h_2.9b_state_operations_gate.sh
cat results/development/impl3n_g1h_2.9b_state_operations/summary.json
```

期望 `state_diff_valid`、`reset_valid`、`swap_valid` 和
`source_states_immutable` 全部为 `true`，组件数应为 96。

首次结果只有 reset 失败，且详细报告显示 state 误差固定超限。运行独立的
首次形状执行诊断：

### 5.17a 运行 Impl-3n-a reset 稳定性诊断

```bash
bash scripts/run_impl3na_g1h_2.9b_reset_stability_diagnostic.sh
cat results/development/impl3na_g1h_2.9b_reset_stability/summary.json
```

重点查看 `route_decision`、`first_reference_tolerance_pass_count`、
`stabilized_reference_tolerance_pass_count` 和
`adjacent_tolerance_pass_count`。该诊断不会覆盖 Impl-3n 的失败报告。

实际路线为 `first_shape_call_outlier`。按已确认的单次效应运行新输出门：

### 5.17b 运行 Impl-3n-b 单次预热状态操作复验

```bash
bash scripts/run_impl3nb_g1h_2.9b_state_operations_warmed_gate.sh
cat results/development/impl3nb_g1h_2.9b_state_operations_warmed/summary.json
```

重点确认 `reset_shape_warmup_count=1`，并检查 `state_diff_valid`、
`reset_valid`、`swap_valid`、`source_states_immutable` 和总 `valid`。

实际结果上述字段全部通过，Impl-3o 的暂停已经解除。

### 5.18 Impl-3n-b 通过后运行 Impl-3o 随机状态复验

```bash
bash scripts/run_impl3o_g1h_2.9b_random_state_gate.sh
cat results/development/impl3o_g1h_2.9b_random_matched/summary.json
```

期望同 seed 可逐位复现、不同 seed 可区分、尺度匹配和续算均有效，组件数
应为 96，`continuation_shape_warmup_count=1`，最终 `valid=true`。

实际结果全部通过。最大相对 L2 尺度误差为
`2.8687819151988067e-05`，同 seed 逐位复现、不同 seed 可区分，续算、
tokenizer 和来源状态不变性均有效。至此停止继续运行开发模型门，进入
[`EXP-001 Batch 2 参数冻结审阅`](exp001_batch2_freeze_review.md)；在预注册
包完成前不运行确认集。

### 5.19 运行 Impl-3p 历史写入协议比较门

项目负责人确认 Batch 2 的 D1–D3 建议后，只运行以下开发门：

```bash
git pull --ff-only
source .venv/bin/activate
bash scripts/run_impl3p_g1h_2.9b_history_binding_gate.sh
cat results/development/impl3p_g1h_2.9b_history_binding/summary.json
```

该门比较：

- `single_statement`；
- `statement_plus_verification`；
- `repeated_consistent`。

每种模式使用相同的 32 个语义案例和 131-token delay，每个案例完整轮换
A–D，因此共运行 384 条受控读出。它按固定复杂度顺序选择第一个达到
`0.80` 标签边际化准确率的模式；不是选择分数最高的模式。重点返回：

- `mode_metrics`；
- `selected_mode`；
- `history_binding_gate_passed`；
- `route_decision`；
- `valid`。

若 `valid=true` 但 `history_binding_gate_passed=false`，表示诊断完整但三种
写入方式都没有达到预设门槛，应进入 Revise，不得降低阈值或直接生成确认集。

实际结果为：

- `valid=true`；
- `history_binding_gate_passed=true`；
- `selected_mode=single_statement`；
- `route_decision=freeze_single_statement`；
- 单次声明标签边际化准确率 31/32；
- 声明后验证和重复一致绑定均为 32/32。

根据运行前规则冻结候选必须是单次声明，不能改选开发分数更高的复杂方案。
至此不再运行历史写入开发门，进入正式模板、控制任务、seeds、统计模拟和
预注册包冻结。

### 5.20 运行 Impl-3q 正式冻结候选门

项目负责人确认 D4–D8 后运行：

```bash
git pull --ff-only
source .venv/bin/activate
bash scripts/run_impl3q_exp001_formal_freeze_candidate_gate.sh
cat results/development/impl3q_exp001_formal_freeze_candidate/summary.json
```

该门会：

1. 用固定 tokenizer 将4个中性 filler 各自确定为恰好131 tokens；
2. 运行128个语义案例×4代码轮换，共512条prompt-visible模板资格读出；
3. 运行96条与I/G无关的通用能力控制；
4. 从32个prompt-visible factorial groups估计开发期nuisance SD；
5. 同时运行经验代理和 \(d_z=0.20\) 标准化两套10,000次功效模拟；
6. 锁定配置、源码、schema、原始记录和报告的SHA-256；
7. 生成等待人工确认checksum的预注册候选。

必须检查：

```bash
python -m psa preregistration-verify \
  --candidate results/development/impl3q_exp001_formal_freeze_candidate/preregistration_candidate.json \
  --project-root .
```

只有以下字段同时满足，才把结果交给项目负责人确认：

- `valid=true`；
- `template_qualification_passed=true`；
- `control_baseline_passed=true`；
- `power_gate_passed=true`；
- `freeze_candidate_ready=true`；
- `route_decision=review_preregistration_checksum`；
- `confirmatory_results_observed=false`；
- `core_set_generated=false`。

即使全部满足，也不能继续运行确认集。下一步是人工核对并明确确认
`candidate_digest_sha256`；没有这一步，不得生成或解封 Core Set。

若 Impl-3q 为有效Hold，先运行只读审计：

```bash
git pull --ff-only
source .venv/bin/activate
bash scripts/review_impl3q_exp001_formal_freeze_candidate.sh
```

它只读取现有manifest、原始分数和报告，生成
`formal_freeze_review.json`。它不会加载模型、重跑题目、读取确认结果或
生成Core Set。重点检查 `control_review.route_decision` 和总
`route_decision`，再决定是否只修正控制读出，或同时修订双字段模板族。

## 6. 运行纯逻辑测试

```bash
python -m unittest discover -s tests -v
```

这些测试只验证：

- 四状态任务生成；
- 答案与历史顺序平衡；
- 泄漏检查；
- E1/E2/E3 公式；
- bootstrap、置换和 Holm；
- checksum；
- CLI。

它们不是模型实验。

## 7. 单独生成开发任务样例

```bash
psa task-generate \
  --output results/dev/identity_goal.synthetic.json \
  --config configs/tasks/exp001_identity_goal.dev.json
```

输出只用于检查生成逻辑。当前示例标签尚未经过目标 RWKV tokenizer 审核，不得作为正式测试集。

## 8. 远程环境信息

进入模型适配前记录：

```text
OS
GPU model/count
driver
CUDA
Python
framework
可用显存
可用磁盘
容器能力
RWKV 实现来源与 revision
checkpoint 路径或模型 ID
tokenizer revision
```

不要把访问密钥、用户名或私有路径提交进仓库。

## 9. 开发门顺序

```text
环境检查
→ RWKV adapter 接口调查
→ official initial state
→ state inventory
→ 100 次开发 roundtrip
→ 冻结数值容差
→ tokenizer 标签池
→ Prompt-visible T0
→ 资源估算
→ 最终预注册
→ 确认性 Core Set
```

在 RWKV adapter、正式配置和预注册尚未完成前，不运行 Core Set。

## 10. 需要从云端回传

第一轮只回传非确认性开发信息：

```text
environment_manifest.json
model_interface_report.json
state_inventory.json
roundtrip_validation.json
tokenizer_label_report.json
capability_gate_report.json
resource_estimate.json
.psa-assets/receipts/exp001-rwkv7-world-0.4b.json
```

这些文件用于补齐 `state_format.md` 和 `evaluation_protocol.md` 的空缺参数。
