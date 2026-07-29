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
