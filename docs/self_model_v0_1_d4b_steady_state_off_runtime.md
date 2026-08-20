# Self Model v0.1 D4B：稳态 OFF 等价门 fake-first runtime

## 1. 当前实现边界

本轮只实现项目内 D4B 核心调度与记录器。它接收调用者提供的四条路线和 tensor
接口，不导入 RWKV 或 Torch，不定位已安装源码，不加载权重，也没有真实模型入口、
机器授权文件或 single-use claim。

四条路线继续固定为 original、OFF-G1、G0 和 OFF-G2。runtime 不新增任何 active
callback、Self projection 或真实层选择；G0 和 G2 继续复用已有的临时方法绑定及
`finally` 恢复边界。

## 2. 固定调用与记录

runtime 先用 original 对 `[187, 931]` 执行一次 prefix，再按 original、OFF-G1、
G0、OFF-G2 各执行一次 `[2764]` 预条件。前五次输出全部记录但不计分。随后严格
执行冻结的四轮 4×4 拉丁顺序，共十六次计分调用。

所有21次输出都记录 logits 及每个 state tensor 的 shape、dtype、device、numel 和
SHA-256。只在同一次 runtime 调用内比较十六个计分输出：四路线各四次产生24个
同路线对，六个路线对各产生16个跨路线对，共96个。120个比较都要求 shape、
dtype、device 相同并满足 `torch.equal`。

## 3. 失败关闭

调用数、顺序、路线位置、输入夹具、tensor inventory、路线计数、临时绑定恢复或
任一精确比较不满足时，报告 `valid=false` 和 `stop_without_rerun`。runtime 不会
观察中间输出后追加预热、重新排序、改容差或自动重跑。

即使 fake runtime 全部通过，也只证明核心实现符合冻结设计，不是2.9B模型的
D4B结果。fake报告的通过效果固定为`runtime_core_verification_only`；只有未来
另行授权的真实D4B通过才可能形成D5审阅候选。当前实现不能改写D4失败或自动
授权真实入口、D5、active injection和Self效果实验。
