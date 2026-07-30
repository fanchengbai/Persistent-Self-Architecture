# EXP-001 checkpoint 迁移方案

> 状态：开发候选已固定，等待云端接口门
>
> 日期：2026-07-30
>
> 原则：更换模型不等于更换研究问题；旧模型的失败记录永久保留。

## 1. 为什么现在需要换模型

RWKV-7 World 0.4B 已通过 state 保存、恢复、交换和随机对照等工程门，但在
Impl-3b 中表现为：

| 能力层级 | 结果 | 通俗解释 |
|---|---:|---|
| 直接抄写 A/B/C/D | 100% | 模型看得见答案代码，评分接口也没有坏 |
| 单字段查表 | 25% | 只要需要根据一个符号寻找答案，模型就固定选 A |
| 双字段组合 | 25% | 身份与目标组合任务也无法完成 |

因此，现在进入 state-only 实验会把“模型不会做题”误写成“state 没有保存
信息”。正确路线是先找到能通过显式能力门的 RWKV-7 checkpoint，再继续比较
state 条件。

## 2. 候选选择

首选候选固定为：

```text
BlinkDL/rwkv7-g1
rwkv7-g1h-1.5b-20260710-ctx10240.pth
revision: bc3b5c8dae5b09db2445bf4f7589fe800d88688e
SHA-256: 737079d81865801fd85e5459488d89a36d5304a524e890244eb83d44f531c89c
size: 3,055,444,605 bytes
```

选择理由：

1. 它仍是 RWKV-7，保留本项目需要的 recurrent state 接口；
2. 官方说明 G1 系列训练数据包含 instruction、chat 和 reasoning 数据，更接近
   当前任务需要的“读条件并做选择”能力；
3. 1.5B 是官方说明中能够完成结构化 function-call 类任务的最低规模；
4. 约 3.06 GB 的权重远小于云服务器 32 GB 显存，适合作为最低成本升级；
5. 先测 1.5B，可以避免一开始就把模型规模扩大到 2.9B 或 7.2B。

官方来源：

- [RWKV-7 G1 官方模型页](https://huggingface.co/BlinkDL/rwkv7-g1)
- [固定版本的 1.5B 权重页](https://huggingface.co/BlinkDL/rwkv7-g1/blob/bc3b5c8dae5b09db2445bf4f7589fe800d88688e/rwkv7-g1h-1.5b-20260710-ctx10240.pth)
- [RWKV-7 G1 官方提示模板](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/RWKV7-G1x-templates.txt)

## 3. 为什么不先选别的模型

| 候选 | 判断 | 原因 |
|---|---|---|
| G1 0.4B | 暂不选 | 与失败模型处于同一参数规模，节省的成本不足以抵消再次失败风险 |
| G1h 1.5B | 首选 | 能力、显存和迁移成本之间最平衡 |
| G1h 2.9B | 后备 | 如果 1.5B 接口通过但能力门失败，再升级到这一档 |
| G1h 7.2B | 第二后备 | 32 GB 显存可能容纳，但没有必要在小模型尚未验证前先承担更大成本 |
| 13.3B | 暂不考虑 | 权重本身约 26.5 GB，给运行时和后续实验留下的显存余量太小 |

## 4. 迁移门顺序

新 checkpoint 不继承旧 checkpoint 的通过记录。复验顺序是：

```text
固定下载与哈希验证
  → G1h 接口门（加载、tokenizer、state inventory、同进程恢复）
  → 使用官方 G1 提示格式重新跑 copy / single-field / two-field
  → 通过后重跑磁盘恢复、reset/diff/swap、matched random
  → 冻结新模型和任务参数
  → 才能进入正式 state 因果实验
```

每一层都有明确停止条件：

- 接口门失败：先处理运行库或 checkpoint 兼容性，不运行能力门；
- copy 失败：检查官方提示格式和答案接口；
- single-field 失败：1.5B 不适合 EXP-001，评估 2.9B；
- two-field 失败：记录组合能力限制，评估 2.9B；
- 三层通过：才重跑 state 工程门。

## 5. 提示格式约束

G1 系列不应直接沿用 World 0.4B 的提示格式。能力门必须使用官方结构：

```text
User: <任务内容>

Assistant:
```

并遵守两点：

1. 输入末尾不能带空格；
2. 用户内容中的连续空行要清理，避免与对话轮次分隔符冲突。

旧 World 0.4B 的 v0.1/v0.2/Impl-3b 结果不会被重写；G1h 使用新的配置、
输出目录和报告版本。

## 6. 云端执行：当前只做接口门

拉取本次代码后运行：

```bash
source .venv/bin/activate
bash scripts/prepare_g1h_1.5b_candidate.sh
bash scripts/run_g1h_1.5b_interface_gate.sh
cat results/development/impl3c_g1h_1.5b_interface/summary.json
cat results/development/impl3c_g1h_1.5b_interface/model_interface_report.json
cat results/development/impl3c_g1h_1.5b_interface/state_inventory.json
```

这一步只回答“新模型能否被当前实验框架可靠操作”，不回答 Self 假设，也不
运行正式实验。
