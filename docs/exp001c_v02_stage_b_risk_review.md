# EXP-001C v02 Stage B 离线设计与风险审查

版本：0.1 Draft  
日期：2026-08-11  
状态：离线设计、真实 backend 与执行 runner 安全外壳完成；live preflight 和模型执行未授权

## 1. 本轮结论

Stage A 的 prompt-visible 非 Core 正控制已经以 28/32、label-marginalized accuracy
0.875 通过。本结论只允许进入 Stage B 的独立设计和授权审查，不自动授权模型执行。

Stage B 使用 Stage A 已完成的 32 条结果作为外部 prompt-visible 基线，不在 Stage B
中重跑 Stage A。Stage B 规划 7 个条件，每个条件映射 Stage A manifest 中相同的 32
条 code-rotated trial，共 224 条：

| 条件 | 角色 | 状态来源 | 语义目标 |
|---|---|---|---|
| `continuous` | 状态语义保持 | 当前 history state | 跟随实际状态字段 |
| `restored` | 磁盘 roundtrip 保持 | 同一状态保存并恢复 | 跟随恢复后的状态字段 |
| `swapped_I` | 身份维度因果跟踪 | 身份互换、目标不变 | 跟随互换后的状态字段 |
| `swapped_G` | 目标维度因果跟踪 | 身份不变、目标互换 | 跟随互换后的状态字段 |
| `swapped_both` | 双维度因果跟踪 | 身份与目标同时互换 | 跟随互换后的状态字段 |
| `reset` | 无状态负控制 | 无 recurrent state | 仅诊断，不设状态语义正确项 |
| `random_matched` | assay sensitivity control | 确定性 shape-matched 随机状态 | 仅诊断，不设状态语义正确项 |

交换条件的正确答案必须根据“实际注入的状态字段”重新映射到当前 trial 的 A–D 选项，
不能沿用 query 原始 target；否则会把正确的因果跟踪误判为错误。离线 manifest builder
已对每个 2×2 factorial group 做唯一来源映射，并要求每个状态语义条件都有 A–D 中
唯一的预期代码。

## 2. 已关闭的设计风险

1. **Stage A 被隐式重跑**：Stage B 条件列表明确不含 `prompt_visible_reset`；基线只引用
   已完成结果的 digest。
2. **交换状态目标错误**：`swapped_I`、`swapped_G`、`swapped_both` 的预期代码由来源
   状态字段和当前选项映射重新计算。
3. **负控制被当成主要语义端点**：`reset` 与 `random_matched` 没有预设状态语义正确项，
   只作为诊断或灵敏度控制。
4. **权限扩张**：设计 manifest 固定 `execution_authorized=false`、
   `formal_test_set_accessed=false`、`formal_run_authorized=false` 和
   `automatic_rerun_authorized=false`。
5. **旧 Stage A 证据漂移**：设计绑定 Stage A manifest、preflight、authorization 和
   result digest；任何执行前仍须在服务器重新核验真实 Stage A result 文件。
6. **源码漂移**：设计 manifest 保存独立 Stage B 源文件清单和 digest，并支持确定性重建。

## 3. 执行授权前仍必须完成

真实执行 runner、结果完整性验证、原子输出和 single-use claim 已完成，但机器授权验证
函数仍默认无条件关闭；测试通过 patch 后只调用 fake backend。claim 在 backend 启动前以
独占创建方式消费，失败结果也不能自动重跑。进入任何 Stage B 模型执行前，仍必须：

1. 实现并运行只读服务器 preflight，核验 Stage A 原始结果、当前 Git commit、模型资产和主机环境；
2. 实现机器授权 builder/validator，并由项目负责人逐字授权，绑定 design manifest 与 live preflight digest；
3. 明确结果观察是否与执行同时授权；未写明时默认不授权观察；
4. 继续禁止访问正式测试集、正式运行、确认性决定和自动重跑。

## 4. 本轮权限声明

本文件、draft config、schema、manifest builder、未调用的真实 backend、默认关闭授权
验证的runner和测试只属于离线开发与风险审查。它们不是 Stage B 执行授权，不创建测试
集，不加载模型，不观察新的模型结果，也不改变 EXP-001B 或任何确认性决定。
