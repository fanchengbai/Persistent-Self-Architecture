# D9-D 离线诊断：当前服务器环境观察

观察日期：2026-10-08（Asia/Shanghai）。本轮依据已有 D9-D 纯离线诊断授权和负责人“继续”指示，仅检查本地项目状态及更新后 `.env` 所配置的服务器。

后续同日更新：负责人要求重新新建环境并测试后，项目与隔离环境已恢复，49 项 D9 专项和 673 项全量回归通过。本文保留恢复前的只读观察；当前状态见 `self_model_v0_1_remote_environment_rebuild_20261008.md`。原 D9-D 工件仍未恢复。

## 已确认事实

- 本地 HEAD 为 `1004a3d`，即已经推送过的 D9-D 离线诊断实现；本轮没有新增推送。
- `.env` 最后修改时间为 2026-10-08 09:33:41，`git check-ignore -- .env` 返回 `.env`，`git ls-files -- .env` 无输出。未输出或提交连接凭据。
- 通过 SSH 成功认证并执行了只读目录检查。新连接地址提供的主机公钥与本机已有信任记录完全匹配，未关闭主机身份验证，也未持久化新的信任记录。
- 服务器默认工作目录为 `/root`。
- `.env` 中配置的 `/root/autodl-tmp/Persistent-Self-Architecture` 不存在。
- 旧运行环境 `/root/autodl-tmp/psa-exp001c-venv` 不存在。
- `/root/autodl-tmp` 存在，目录列表只显示 `.autodl`。
- 对 `/root/autodl-tmp`、`/root/autodl-fs`、`/workspace` 做最大深度 5 的目录查找，未返回 `Persistent-Self-Architecture`、`.git`、`self_model_v0_1_d9_real_v01` 或 `psa-exp001c-venv`。

以上是限定路径与深度的观察，不能据此断言旧服务器的数据已经丢失、其他挂载点没有备份，或历史实验没有执行。

## 诊断状态及证据边界

本次尚未运行服务器无模型测试、静态验证或真实工件离线诊断；模型 forward 次数为 0，未访问权重，未创建机器授权、claim 或新的真实实验输出。现有 D9-D 真实完成与预注册因果门失败结论保持为历史记录，没有新增数值结论。

待诊断的输入包括原机器 authorization，以及原结果目录中的 `execution_claim.json`、`projection.json`、`raw_ledger.jsonl`、`report.json` 和 `integrity.json`。源码提交与聊天中的结果摘要不能替代这些原始工件；重新克隆仓库也不会恢复被 Git 忽略的实验结果。

下一步需要定位旧实例、数据盘或备份中的完整工件包。找到后先只读核对冻结 SHA-256，再使用已提交的诊断工具分析。缺少原始工件时，不能将诊断记为完成，不能由摘要补造 ledger/projection，也不能以补数据为由重跑已消费的 D9-D。

本轮本地 SSH 辅助程序和连接依赖仅位于被 Git 忽略的 `results/development/`，未加入项目依赖或待提交文件。
