# 语义目标迁移：纯离线数据、评分器和状态接口

日期：2026-10-08。实现版本：`0.1-semantic-goal-offline`。本阶段承接语义目标路线审查，交付工程预览数据和接口；尚未预注册、冻结正式样本数、构造真实目标向量或实现新真实 runner。

## 实现内容

`semantic_goal_offline.py` 使用自然语言描述任务，包含耗时、能耗、A/B 两个选项和回答要求。模型接口的目标来自 Self State 中的 `minimize_energy` 或 `minimize_time`；当前任务题面不包含具体目标或参考答案。目标可见基线使用独立函数添加明确的自然语言目标说明。

评分器按目标对应的属性计算参考答案，并对实际传入的 A/B 响应评分。多字母、解释性文本和其他格式记为错误，不能丢弃。汇总函数要求一份完整且无重复的响应表，按基础场景聚合两个目标和两种 A/B 标签排列，分别报告目标、trade-off 和 dominance 子集，不创建执行声明或科学通过决定。

参考评分器是评估 oracle，不是模型策略。任何后续执行入口都必须让模型输出选择；不得用这个 oracle 替代模型决策。

## 工程预览数据

| split | 基础场景 | trade-off | dominance |
|---|---:|---:|---:|
| calibration | 32 | 24 | 8 |
| development | 64 | 48 | 16 |
| heldout | 128 | 96 | 32 |
| persistence | 32 | 24 | 8 |
| 合计 | 256 | 192 | 64 |

每个场景有两种 A/B 标签排列，生成 512 个公共题面；对两个目标生成独立存放的 1,024 条评估参考。trade-off 场景中两个目标应选择不同方案，dominance 场景中应选择相同方案。各 split、各目标及各场景类型均平衡 A/B 答案。

数据以独立 seed namespace 确定性生成。相同或选项顺序颠倒的数值事实都不能跨 split 复用；场景 ID、模板族、目标表达分别校验。同一场景的目标和标签变化属于一个统计簇。自然语言词表允许共享。

这些数量用于工程接口预览，不是被冻结的真实实验样本数；不存在 tokenizer 绑定或真实模型调用调度。原 D9 fixtures、token 序列、projection、claim 和结果没有复用为新实验数据。

## 目标状态接口

复用项目既有 Self State v0.1 schema、摘要、不可覆盖 SelfStore 和字段交换函数。本阶段只接受一个已知 active goal，其余 Self 字段为空；模型/tokenizer 身份标为 `offline-contract-unbound`。

`GoalRequest` 仅包含 `mode` 和 `goal`，没有题目、答案、分数或 fixture ID。active 请求携带语义目标；zero/mask/random 请求不携带目标。random 目前只是接口模式，没有构造随机向量或任何 projection。显式 clear 操作生成空目标的新快照，保留父快照引用；清除状态只能生成无目标请求，不能被默认为一个 active 目标。

保存恢复比较的是 JSON 状态及请求一致性，不是模型 logits/state 恢复实验。交换、清除和保存均保持源快照不变。

## 样本量审查的实际限度

实现了基础场景级配对均值的正态近似预算，使用单侧 alpha=0.01、目标功效0.8。假定配对差的场景标准差为0.5，检测10/15/20个百分点提升，约需251/112/63个独立场景；标准差为1.0时，约需1004/447/251个。

这些是明确假设下的规划近似，不是对实际数据方差的估计，不验证 cluster-bootstrap 区间覆盖率，也不证明完整联合门或 dominance 子集的功效。工程预览中的128个 heldout 场景只有96个trade-off，不能把全部响应数当作独立样本数。

进入真实协议冻结前，仍需对完整联合判定作前瞻模拟，确定样本数、必要支持门的数值标准、capture/development选择预算、持久性容差和总调用预算。本实现没有把审查稿百分比转成正式执行门，也没有增加授权/claim/schema层级。

## 本地与服务器验证

- 本地专项：16项通过，0.131秒；全量：689项通过，31.794秒。
- 服务器专项：16项通过，0.062秒；全量：689项通过，17.492秒。
- 两端离线包报告摘要一致：`0d6cccc71808c478603bcd293414d1c5c38921adadbbd17b2b2459a92d999bf4`。
- 六项状态接口检查全部通过；`model_accuracy_measured=false`、`actual_model_forward_calls=0`。

验收覆盖有明确答案的trade-off/dominance示例、标签反转、跨split事实泄漏、缺失/重复数据、非法数值/平局、格式错误计入失败、完整表聚类计分、目标交换/显式清除、持久化与摘要篡改、源快照不变性、配置权限升级拒绝和输出不覆盖。

服务器复验在独立目录 `/root/autodl-tmp/psa-semantic-goal-offline-preview-v01` 进行。基础代码克隆自已有 `1004a3d`，再上传本轮配置、模块、脚本、测试四份文件；执行前逐一核对源文件SHA-256。它是基线加已核对新文件的隔离预览，不能冒称服务器运行了尚未提交的新 Git HEAD。原项目目录没有修改。

服务器使用 `/root/autodl-tmp/psa-exp001c-venv/bin/python`，未安装 Torch/RWKV。复验日志和报告已下载到本地忽略目录 `results/development/semantic_goal_remote_v01/`。

- 服务器专项日志SHA-256：`c4d1f5dacec429efb0c5ee14b495d9cb830d2e549a6710cdc9b92af389b3567e`。
- 服务器全量日志SHA-256：`acbcdea9c130ee5cc13bc5e0cc4ebcc198aa7accb456c033900fd8bf1a8b00e7`。

## 使用方式

在项目根目录运行，输出目录必须不存在：

```bash
PYTHONPATH=src python -m unittest tests.test_self_model_semantic_goal_offline
PYTHONPATH=src python scripts/verify_self_model_v0_1_semantic_goal_offline.py \
  --output-dir results/development/semantic_goal_offline_preview
```

输出包括 `public_tasks.json`、`evaluation_references.json`、三个目标状态快照和 `report.json`。报告明确区分参考评分与模型表现，生成后不覆盖旧目录。CLI和评分器都没有模型调用能力。

下一项工作是完成前瞻功效模拟及单项联合真实协议的离线设计，再审查目标编码与推理入口。真实模型依赖部署、tokenizer调用、激活capture、projection构造和模型执行尚未发生，旧D9-D失败和已消费claim不变。
