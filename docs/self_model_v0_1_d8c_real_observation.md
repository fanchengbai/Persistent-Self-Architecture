# Self Model v0.1 D8-C real numerical identifiability observation

## 结论

D8-C 真实 2.9B 数值可识别性验证在单次授权范围内有效完成。runner 在同一进程、同一 wrapper 中完成全部 584 次 forward，写出 8 条 conditioning 和 288 条 scored pair 记录，共 296 条 ledger；报告 `valid=true`，状态为 `d8c_real_numerical_identifiability_completed`。

冻结主要端点为正：24/24 个 fixture 的 excess drift 均为正，四个 stratum 各 6/6；平均 fixture excess drift 为 `0.0032601490100549004`，fixture-cluster bootstrap 99% 下界为 `0.002858416711489245`。预注册决策为 `route_specific_excess_drift_detected_non_self_engineering_evidence_only`。

这个结果证明 public 与 wrapper-zero 的跨路径差异能够从两条路径各自的重复性背景中被识别。它否定“wrapper-zero 可直接视为 public 的数值等价替代”这一工程假设，但不涉及真实 Self projection，也不能形成 Self 效果结论。

## 执行与完整性证据

- runner 观察到干净 `main=e0ab61a58394e6eaef2567aa3a988afa6e47738c`，且 `origin_main` 一致。
- 人类授权被物化为机器 authorization，authorization digest=`d78b5df7…14b1`；机器授权文件 SHA-256=`66c8c67f…6dd8`。
- single-use claim 已消费，claim SHA-256=`85403630…05db`；D8-C 和历史重跑、自动重跑均保持 false。
- installed RWKV 版本为 `0.8.32`，源码 SHA-256=`75482aee…05e0`；2.9B 权重 SHA-256=`295595b3…c9b3`，大小 `5,896,273,469` 字节。
- launcher 与运行期严格确定性检查全部通过；模型为 32 层、宽度 2560、96 个 state 组件，策略为 `cuda fp16`。
- 8 次 conditioning、576 次 scored、288 条 scored pair 和每条 96 个 state component distance 均完整；wrapper bindings/context 稳定且基础实例字典未改变。
- 原始 ledger 共 296 行，SHA-256=`72a3e919…8e73`；完整报告文件 SHA-256=`1ee3b10f…8c08`，报告内 digest=`a0dad92b…2ac5`；integrity digest=`49416072…4300`。
- 真实运行耗时约 `55.71` 秒，未产生 failure artifact。

## 解释边界

三次 replicate 在每个 fixture 内给出相同 excess drift，且 24 个 fixture 与四个 stratum 方向一致，说明观察到的差异不是本协议所估计的路径内随机重复性漂移。当前证据只支持“wrapper 路径本身引入了可识别的数值变化”这一非 Self 工程判断。

因此后续不能把 public 与 wrapper-zero 当作已证明等价的 OFF 基线，也不能把未来 wrapper-active 与 public 的差异直接归因于 Self projection。若继续研究，必须先在新的纯离线路线审查中决定是否能使用同一路径内的 zero/active 因果对照隔离 wrapper 路径效应，并建立不复用本次 claim、fixture 或结果作为新实验数据的独立协议。

本次 D8-C 已关闭且不得重跑。D7-D/D7-E、projection、正式测试集、Self 效果结论、Self Updater、raw-original 和自动重跑均未获授权。
