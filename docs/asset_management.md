# 模型与数据资源管理

> 状态：Impl-0.1  
> 目的：让云服务器从仓库中的固定清单准备外部资源，同时避免把大型权重、数据和凭据提交到 Git。

## 1. 基本原则

仓库保存：

- 资源 ID、来源仓库和不可变 revision；
- 目标路径、许可证和已知 checksum；
- 下载、断点续传和验证代码；
- 程序生成数据的配置；
- 下载完成后的本地收据。

仓库不保存：

- 模型权重本体；
- 完整训练语料；
- Hugging Face token；
- 原始实验输出和大型 checkpoint。

默认资源根目录是项目内的 `.psa-assets/`，已被 `.gitignore` 排除。

## 2. EXP-001 当前资源

资源清单位于：

```text
configs/assets/exp001_rwkv7_world_0.4b.json
```

当前包含：

1. RWKV-7 World 0.4B v2.9 基础权重；
2. RWKV World `rwkv_vocab_v20230424` tokenizer；
3. 由本项目生成器创建的 EXP-001 合成开发集。

EXP-001 首轮是冻结模型的状态保存、恢复和交换实验，不需要下载 RWKV 的完整预训练语料。训练数据清单要等到微调目标、数据许可和泄漏控制方案确定后另行冻结。

## 3. 一键准备

在项目根目录运行：

```bash
bash scripts/prepare_exp001_assets.sh
```

脚本依次执行：

```text
展示下载计划
→ 下载或续传模型与 tokenizer
→ 校验模型 SHA-256
→ 生成 EXP-001 开发集
→ 验证资源完整性
```

默认目录结构：

```text
.psa-assets/
├─ models/
│  └─ rwkv7-world-0.4b/
├─ tokenizers/
├─ datasets/
│  └─ exp001/
└─ receipts/
```

若模型文件下载中断，再次执行同一脚本即可从 `.part` 文件续传。

## 4. 自定义存储位置

磁盘空间不足时，可把资源根目录指向挂载盘：

```bash
PSA_ASSET_ROOT=/mnt/psa-assets \
  bash scripts/prepare_exp001_assets.sh
```

这仍然属于项目运行资源，只是物理文件放在更大的云硬盘中。

可以指定 Python：

```bash
PYTHON_BIN=/opt/venvs/psa/bin/python \
  bash scripts/prepare_exp001_assets.sh
```

## 5. 分步执行

只查看计划，不下载：

```bash
python -m psa assets-plan \
  --manifest configs/assets/exp001_rwkv7_world_0.4b.json \
  --root .psa-assets
```

下载：

```bash
python -m psa assets-fetch \
  --manifest configs/assets/exp001_rwkv7_world_0.4b.json \
  --root .psa-assets
```

只下载单个资源：

```bash
python -m psa assets-fetch \
  --manifest configs/assets/exp001_rwkv7_world_0.4b.json \
  --root .psa-assets \
  --only rwkv-world-tokenizer-20230424
```

重新验证：

```bash
python -m psa assets-verify \
  --manifest configs/assets/exp001_rwkv7_world_0.4b.json \
  --root .psa-assets
```

## 6. 网络与凭据

公开资源通常不需要 token。若 Hugging Face 要求身份验证，只在云服务器会话中设置：

```bash
export HF_TOKEN="..."
```

不得把 token 写进资源清单、脚本、日志或 Git。下载程序只读取环境变量，不会把 token 写入收据。

## 7. 完整性与安全边界

- 模型权重固定到具体 revision，并校验发布页给出的 SHA-256；
- tokenizer 固定到具体 revision，下载后把实际 SHA-256 写入本地收据；
- 在最终预注册前，应把 tokenizer 收据中的 digest 回填到资源清单；
- `.pth` 属于 PyTorch 序列化文件，只从已记录的官方来源获取；
- 模型适配器实现时优先使用限制反序列化能力的加载方式，并在加载前再次校验 digest；
- 下载成功不等于模型接口门通过，仍需完成 state inventory 和 roundtrip 验证。

## 8. 以后加入训练数据

训练数据不能只写一个浮动数据集名称。每个数据资产至少要固定：

- 数据集仓库与 revision；
- 具体文件或明确的 snapshot 范围；
- 许可证与允许用途；
- 原始文件 digest；
- 清洗、过滤、去重和切分配置；
- 与确认集的泄漏检查报告；
- 生成后数据的 digest。

这些信息确定后，把数据项加入新的训练资源清单，而不是直接修改已经用于 EXP-001 的冻结清单。
