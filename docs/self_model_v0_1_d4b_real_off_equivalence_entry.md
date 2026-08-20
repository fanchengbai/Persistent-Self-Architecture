# Self Model v0.1 D4B：真实2.9B稳态OFF等价入口

## 1. 本轮范围

本轮只实现 D4B 真实入口的安全外壳和无模型静态验证，不授权执行。入口复用已
通过本地与服务器静态门的四路线、21次调用、120项严格比较核心；不修改D4B
runtime调度，不实现active injection、Self projection或D5。

## 2. 唯一未来授权

未来执行需要项目负责人逐字发送冻结配置中的完整授权文本。普通“确认”“下一轮”
或此前D4A授权都无效。机器授权必须符合固定Schema，绑定最终干净main提交、入口
配置digest和D4B runtime静态报告digest，并且只能写入固定授权路径。授权同时覆盖
本次结果观察，但不覆盖重跑D4/D4B、D5、active injection或Self效果实验。

## 3. 单次claim和调用时序

入口首先核对精确环境锁、唯一配置/授权/结果路径、干净main、机器授权digest和
已安装RWKV源码锁。随后在空结果目录独占创建single-use claim。claim必须先于
模型配置及资产校验、模型加载和任何forward调用。claim后的成功或失败都会消费
唯一机会；异常保存`failure.json`，不得自动重跑。

真实执行若未来获权，将依次构造original、OFF-G1、G0和OFF-G2，调用冻结核心。
外层报告单独记录真实模型加载/执行上下文，不把fake-first核心的模板安全字段误作
真实运行事实。D4B失败则停止；通过也只形成D5审阅候选，不自动授权D5。

## 4. 当前安全状态

当前`execution_authorized_at_implementation=false`。本轮不会创建机器授权、claim
或结果目录，不导入RWKV/Torch，不访问权重，不加载或执行2.9B模型。入口完成并
通过服务器无模型静态复验后，仍需新的逐字单次执行与观察授权。
