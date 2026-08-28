# Self Model v0.1 D7-B manifest 与 fake-first runtime contract

日期：2026-08-28

状态：纯离线实现完成；未探测 installed source，未实现 projection，未加载或执行模型。

## 本轮做了什么

D7-B 把已冻结的 D7 预注册转为两个确定性 manifest：一个描述 5×5 identity/goal calibration 网格并展开 25 条未来只读 capture 记录；另一个描述四个 held-out 任务族、每族四个语义案例和四次答案代码轮换并展开 64 条 fixture。

held-out manifest 固定 13 个条件。每条 fixture 先执行一次不计分 OFF 预条件，再按由冻结 seed 决定的循环顺序执行全部 13 个条件，因此未来 held-out 计划为 64×14=896 次 forward；加上 25 次 calibration 后，D7-E 仍是单一联合 921 次计划。D7-B 只生成计划和 commitment，不进行这些 forward。

## 训练与 held-out 分离

calibration、hidden held-out 和 prompt-visible qualification 使用不同模板及不同摘要集合。hidden prompt 不写入 identity/goal key；qualification prompt 只为未来 D7-D 显式呈现相同语义，而且不使用 projection。任何 held-out fixture 都标记为不可用于 projection training。

D7 的 identity、goal、任务族和四个 seed 保持与 D6D 分离。D6D 的 fixture、authorization、claim、输出和定量结果都不是 manifest 输入。

## fake-first runtime 的含义

纯 Python runtime 只解析符号路由：例如 matched、identity swap、goal mask 或 dual random 应对应哪些 key。它不创建张量、数值 projection 或模型输出，也不接收模型对象。错误 phase、未知 condition 或 fixture/call 不匹配会在写入 ledger 前失败；输入对象在全部 896 条符号调用后必须保持不变。

该 ledger 只能证明 manifest 完整性和条件分派契约，不能作为兼容性、模型行为或 Self 效果证据。

## 冻结阈值与后续门

D7-D 的能力阈值及 D7-E 的 matched、swap、mask、random、OFF/zero/synthetic 因果与安全阈值均原样进入 held-out manifest。通过 D7-E 最多构成 non-Core engineering evidence，不能直接形成 Self 效果结论。

下一步只能在本轮远程无模型复验后，由项目负责人另行确认 D7-C 真实协议兼容设计。D7-C、D7-D、D7-E 的实现与执行、D6D 重跑、正式测试集、Self Updater、raw-original 和自动重跑当前全部关闭。
