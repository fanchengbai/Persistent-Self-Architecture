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
rwkv 0.8.32
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

## 5. 运行纯逻辑测试

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

## 6. 单独生成开发任务样例

```bash
psa task-generate \
  --output results/dev/identity_goal.synthetic.json \
  --config configs/tasks/exp001_identity_goal.dev.json
```

输出只用于检查生成逻辑。当前示例标签尚未经过目标 RWKV tokenizer 审核，不得作为正式测试集。

## 7. 远程环境信息

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

## 8. 开发门顺序

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

## 9. 需要从云端回传

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
