# Self Model v0.1 D8-C-I single-use runner

## 目的

D8-C-I 把已通过远程无模型验证的 D8-C 数值可识别性协议实现为未来可单次执行的 2.9B runner。它回答的是“public 与 wrapper-zero 的跨路径差异是否超过各自路径内重复性背景”这一工程可识别性问题，不测试 projection，也不形成 Self 效果结论。

本轮只实现并静态验证 runner。installed source、RWKV/Torch、权重和模型均不在实现时触发；机器 authorization、execution claim 和 output namespace 也必须保持不存在。

## 冻结执行计划

runner 从 D8-A design 通过已验证的 D8-B 展开函数生成：

- 4 个 conditioning fixture，各执行 public 与 wrapper-zero，共 8 次，不计分；
- 24 个 scored fixture，每个包含 public-public、wrapper-wrapper、public-wrapper、wrapper-public 各三次；
- 288 个 pair block、576 次 scored forward，总计严格 584 次；
- 每个 pair 的两个调用都从同一 fixture prebuilt zero state 的独立克隆开始；
- call-ID digest 固定为 `7004dd99e62d0657be968096f83b4099b6752cb07bf203577c31b487db3190ca`。

缺失、重复、重排、非有限输出、shape/dtype/device 不兼容、state 不是 96 个组件、forward 异常或调用数不等于 584 都会使 attempt 失败；不能丢弃记录、放宽确定性策略或自动重跑。

## 确定性门

launcher 必须在 Python 启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 和 `PYTHONHASHSEED=28083101`，并保持 `RWKV_DE_VERSION` 未设置。claim 消费后、模型加载前，runner 固定 Python/Torch/CUDA seed，启用严格 deterministic algorithms，关闭 cuDNN benchmark 与 TF32，并将 float32 matmul precision 固定为 `highest`。任一项无法满足即持久化失败并停止。

## 授权与 single-use 生命周期

未来授权使用独立的 D8-C-I Schema，并绑定：干净 GitHub `main` 提交、runner config、runner 静态报告、D8-C 远程无模型报告、D8-A/B/C manifests 与 call-ID digest。执行顺序固定为：

1. 检查 launcher lock、确定性环境和干净 `main`；
2. 验证逐字机器 authorization；
3. 核对 installed source；
4. 在导入 Torch、访问权重或加载模型前独占创建 claim；
5. 执行 584-call 并持续写入 ledger；
6. 完整 ledger 后才计算冻结 endpoint；
7. 写入 report 与 integrity，或在异常时写入 failure。

authorization、claim、raw comparisons、report、failure 和 integrity 都使用唯一固定路径，全部采用 exclusive-create，拒绝覆盖或复用。claim 一旦创建，无论成功或失败都消耗本次机会。

## 输出与解释边界

`raw_comparisons.jsonl` 保存 8 条 conditioning 记录和 288 条 scored pair 比较记录；每条 scored 记录包含 logits distance、96 个 state component distance、state/output 最大距离和冻结 schedule 元数据。完整后复用 D8-B endpoint 计算 fixture excess-drift、bootstrap 和支持门。

即使 endpoint 为正，也只能写为 route-specific numerical excess 的非 Self 工程证据；不能据此宣称模型具备 Self Model 效果。D7-D/D7-E、projection、正式测试集、Self Updater、raw-original、历史重跑及自动重跑继续关闭。

## 下一道门

先在服务器进行 D8-C-I 无模型静态复验，确认 runner AST 时序、授权绑定、584-call expansion、source digest 与 artifact 缺席。通过后仍需项目负责人逐字给出冻结的真实单次执行授权，普通“下一步”不能启动模型。
