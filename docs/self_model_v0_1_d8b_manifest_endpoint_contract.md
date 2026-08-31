# Self Model v0.1 D8-B deterministic manifest 与 fake endpoint contract

## 本轮完成内容

D8-B 将 D8-A 的纯离线预注册设计物化为四份独立 manifest：

- fixture manifest：固定 4 个 conditioning fixture、24 个计分 fixture、四层形状、全新 token 生成规则和 D7-C token 排除；
- schedule manifest：固定四类路径配对、每类每 fixture 三次重复、种子化基础顺序和 4×4 拉丁平衡；
- determinism manifest：区分 launcher 启动 Python 前必须设置的环境与模型加载前必须设置的运行期标志；
- endpoint manifest：固定 output distance、路径内包络、双向跨路径下限、fixture 中位数、bootstrap 和支持门。

四份 manifest 均绑定 D8-A config SHA-256，并由 D8-B contract 绑定各自文件 SHA-256。D8-A 的 fixture/schedule commitment 保持为：

- fixture：`8976ac9f3f0b042e92ba146e58cc1df8c2d05e5a4635ccb0de558fb36161499e`；
- schedule：`a53cf5edc4f132c3fc773d63f7686f91f485b5c08c52f82beec983af36816465`。

确定性展开得到 24 个计分 fixture、4 个 conditioning fixture、288 个计分 pair block、8 次 conditioning 调用和 576 次未来计分调用，总计 584 次。这里只生成纯 Python 描述，不生成模型输入张量或运行对象。

## 纯 Python endpoint contract

fake runtime 只接受有限数值组成的 Python 序列。它实现并验收：

1. tensor distance：最大绝对差除以两边最大绝对值与 `1e-12` 的最大者；
2. 96 个 state 组件的最大距离；
3. logits/state 两者最大值作为 output distance；
4. 288 条 pair ledger 的ID、pair type、重复轮和完整性检查；
5. 每个 fixture 三轮 excess 的中位数；
6. 24 fixture 聚类 bootstrap 99% 下界、21/24支持门和每层5/6支持门。

fake acceptance 包含三种冻结情形：

- 两种跨路径顺序都明显超过路径内重复性：判定 route-specific excess；
- 只有 public→wrapper 一个顺序较大：保守 cross floor 将其判为 inconclusive；
- 四类配对都只有相同 background drift：within envelope 抵消后判为 inconclusive。

无论哪种情形，runtime 都不会产生路径等价或 Self 效果结论。记录缺失、重复、pair type 改变、负数/非有限距离、state 不是96组件或形状不兼容都会在决策前失败关闭。

## 权限边界

本轮没有探测 installed source，没有修改 D7-C runner，没有实现真实执行入口，也没有导入 RWKV/Torch、访问权重、加载或执行模型。D7-C 的 cell、token、seed、claim 和结果没有进入 manifest 或 fake ledger。

D8-C 仍未设计、实现或授权。后续如需推进，必须先完成服务器纯离线复验，再由项目负责人单独确认 D8-C 真实协议设计；当前不开放D7-C修复/重跑、D7-D/D7-E、projection、正式测试集、Self效果、Self Updater、raw-original或自动重跑。
