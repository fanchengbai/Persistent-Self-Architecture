# Self Model v0.1 D8-C：真实数值可识别性协议与无模型安全入口

## 本轮边界

D8-C 只把 D8-B 已冻结的数值可识别性研究变成未来可审计的真实协议，并实现一个不执行模型的安全入口。它不探测 installed source，不修改真实 runner，不读取权重、calibration 或 held-out payload，不导入 RWKV/Torch，也不创建机器授权或 single-use claim。

## 冻结协议

- 绑定 D8-B contract、fixture、schedule、determinism 和 endpoint 的 SHA-256；任一 digest 或 commitment 改变即停止。
- 使用同一进程、固定 launcher 环境与模型加载前确定性标志，执行 8 次 conditioning（不计分）和 576 次计分调用，共 584 次 forward。
- 计分顺序由 D8-B 的 288 个 pair block 展开为不可重排的 584-call ledger；缺失、重复、重排、异常或调用数不等于 584 都在决策前失败关闭。
- 只记录有限的 logits 与 96 个 state component，使用 D8-A/D8-B 预注册的 distance、excess-drift、bootstrap 和支持门；不改变阈值，不产生 Self 效果结论。
- determinism policy 无法满足时停止，不放宽算法、不重试、不丢弃异常调用。

## 授权生命周期

D8-C 拥有独立的 schema、authorization、claim 和 output 命名空间：

`schemas/self_model_v0_1_d8c_real_authorization.schema.json`、
`results/authorizations/self_model_v0_1_d8c_real_v01.json`、
`results/development/self_model_v0_1_d8c_real_v01/`。

本轮只审计 schema 和路径，authorization/claim/output 均必须不存在。未来真实运行必须由项目负责人逐字授权、绑定所有冻结 digest，并消费 single-use claim；缺少授权不能升级为执行。

未来执行授权文本也已在配置与 Schema 中逐字冻结，范围只覆盖一次 2.9B、584-call 数值可识别性验证及结果观察；任何重跑、历史路线、正式测试集、Self 效果或自动重跑都不在授权范围内。

## 纯 Python 验收

安全入口用 D8-B schedule 生成纯 Python call plan，验收 conditioning/scored 数量、唯一 ID、路由合法性和完整顺序；合成 ledger 的缺失、重复和重排均被拒绝。报告明确标记 `model_executed=false`、`payload_accessed=false`、`execution_claim_created=false`，只证明协议完整性，不证明真实路径等价或 Self 效果。

## 下一道门

先在远程环境运行同一无模型静态验证并回传报告；之后若要真实执行，必须另行确认精确的 D8-C 单次执行授权。D7-C/D6D 重跑、D7-D/D7-E、projection、正式测试集、Self 效果、Self Updater、raw-original 和自动重跑均保持关闭。
