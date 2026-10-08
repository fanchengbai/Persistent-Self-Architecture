# 2026-10-08 云服务器环境重建与无模型复验

负责人明确要求“重新新建环境，进行测试”。本轮恢复已提交的项目代码、创建独立 Python 环境，并在服务器上执行无模型回归和静态验证。

## 环境与代码

- 项目：`/root/autodl-tmp/Persistent-Self-Architecture`。
- 分支：`main`；运行提交：`1004a3d09985c09d8254859a253cef1c3a6e24b2`。
- 系统：Ubuntu 22.04.5 LTS；Python：3.12.3。
- 独立环境：`/root/autodl-tmp/psa-exp001c-venv`，`sys.prefix != sys.base_prefix`。
- 环境未安装 Torch 或 RWKV；纯离线测试通过 `PYTHONPATH=src` 使用项目源码，未安装模型可选依赖。
- 本轮最后服务器 `git status --short` 无输出。

已按既有要求在获取代码前执行 `source /etc/network_turbo`。GitHub HTTPS 克隆返回 HTTP 503，因此改用本地提交生成的 Git bundle：`git bundle verify` 通过、包含完整历史和唯一 `main=1004a3d` 引用，大小 3,432,061 字节。上传后从该包克隆，并将 origin 配置回原 GitHub 仓库。没有覆盖已有项目，也没有把本地未提交文档、`.env` 或结果文件加入包中。

## 服务器实测结果

| 项目 | 结果 |
|---|---|
| D9-A/B/C/D 离线专项 | 49 项通过，6.898 秒 |
| 全量 unittest 回归 | 673 项通过，17.462 秒 |
| D9-C 静态入口 | 14 项检查通过，`valid=true` |
| D9-D 离线诊断器静态检查 | 8 项检查通过，`valid=true`、`real_evidence_loaded=false` |
| 真实模型 forward | 0 次 |
| 原 D9-D authorization/claim/projection/ledger/report/integrity | 全部未找到，未创建替代文件 |

D9-C 报告内部摘要：`e9ad2903a5bf703b0eebcc61cdc8d5afb87f27b7838443a406df84e77fc5cc09`。

D9-D 静态报告内部摘要：`3e32a34d3ee4c53997c4aeb636fc2ceedd71b8388ae9b23ba0a6aeaff8d68260`。

两份摘要均与此前冻结记录一致。本次是已有实现跨服务器的无模型复验，没有产生新的 Self 效果证据。

## 可核验工件

服务器运行目录：`results/development/environment_rebuild_20261008_v01/`。

| 文件 | SHA-256 |
|---|---|
| `d9_tests.log` | `b5ab7542b712e7253218956facc59195865db6cb6eed6f936897d0fac61b8d14` |
| `full_tests.log` | `e4e7e6b3475e77b29460a580506eee74ab7c8d413fdd88129cdbce167ce58373` |
| `summary.json` | `e3139384622203476ef8cd1fec4c3bd0b95d57822bb3a0b4d21812be95607b25` |

静态报告文件摘要分别为：

- `results/development/self_model_v0_1_d9c_projection_entry/report.json`：`0b3e7b7ab3fb18e734d6f84e7229ba3112adebcdda7d50d55e9c0f134b46eb73`。
- `results/development/self_model_v0_1_d9d_offline_causal_diagnostic_static/report.json`：`ea3c30dba93020a146cdaebd6e1c5231690508880ae2029137c006e0f108acde`。

以上五个文件已通过 SFTP 下载到本地被 Git 忽略的 `results/development/server_rebuild_20261008_evidence/`，下载后的摘要与服务器记录一致。

## 当前边界

环境重建与无模型复验已经完成。原 D9-D 的 projection/ledger 因果结构诊断仍需旧实例、数据盘或备份中的原始工件。源码或当前静态报告不能代替这些文件；已消费的 D9-D 不因新环境建立而重新获得执行权限。

本轮没有修改真实 runner、冻结阈值或已有科学结论，没有访问权重、加载或执行模型。当前环境是无模型测试环境；真实模型依赖和权重尚未部署。
