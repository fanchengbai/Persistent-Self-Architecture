# Self Model v0.1 D4A 真实最小诊断结果

## 执行完整性

真实 2.9B D4A 在服务器提交
`63a1878b249af5016c54f527eab170cfb7fbcf7e`上单次完成。报告状态为
`d4a_real_diagnostic_complete`，`valid=true`，14/14 项执行完整性检查通过；
九次调用、九个同路线比较和二十七个跨路线比较全部存在。

- 报告 digest：`d6b0602a85553fddae184e2accb3ef06ed280a925ebc4d90a9e13032726b2e88`
- 独立复算：与报告 digest 完全一致
- execution claim：`21055ee608973bad5db60b75351d42cff0ba69e71e1c7a45303bac6b28677754`
- machine authorization：`bfded08e523063c941aa852fbb9c0408b9c83e101c503cc043536fb3bca85ce7`
- 运行时间：约 14.90 秒
- 峰值显存：6,124,028,416 字节

claim 已消费，本诊断不得重跑。D4 状态未改变，active injection、Self projection、
Self 效果实验、D5、确认性决定和自动重跑均未发生或获授权。

## 三个输出簇

九次调用形成三个逐位不同的 logits/state 簇：

1. 第一轮第一个 `original_baseline` 独占一个簇；
2. 第一轮第一个 `g0_recompiled_unmodified` 独占一个簇；
3. 全部三个 `off_g2_instrumented`、后两个 original 和后两个 G0 共七次调用
   共享同一 logits digest 和同一 96 组件 state digest。

稳定簇的 logits digest 为 `bcd3ec17…32e0`，聚合 state digest 为
`ab59b491…9785`。这七次输出逐位一致，不依赖它们位于第一、第二还是第三顺序位置。

同路线三个两两比较中：

- original：1/3 逐位一致；第一次不同，第二次与第三次一致；
- G0：1/3 逐位一致；第一次不同，第二次与第三次一致；
- OFF-G2：3/3 全部逐位一致。

跨路线比较中，original/G0 为 4/9 一致，original/G2 为 6/9 一致，G0/G2
为 6/9 一致。所有不一致都由 original 或 G0 的第一次调用参与；稳定簇内部没有
任何路线差异。

## 数值和组件签名

original 第一次和 G0 第一次相对稳定簇都各有 95/96 个 state 组件 digest 不同，
范围为 `state[1]` 至 `state[95]`。二者彼此比较时，首个不同组件为 `state[4]`，
不同范围恰为 `state[4]` 至 `state[95]` 共 92 个组件；logits 有 65,526 个元素
不同，最大绝对误差为 0.11328125，state 不同元素总数为 4,421,366，最大绝对
误差为 0.109375。

这个 92 组件位置签名与 D4 唯一失败单元记录的 `state[4]` 至 `state[95]`
一致，是首次方法路线调用效应与 D4 失败相关的强证据；但 D4 旧报告没有数值
digest，且 D4A 没有复现 D4 的完整 prefix snapshot 和六单元调用轨迹，因此只能
视为推断，不能宣称已经证明同一底层原因。

## 结论边界

冻结分类为 `within_route_instability_observed`。证据反对“None-guarded OFF-G2
分支本身在稳态下必然改变输出”：全部 G2 调用与后续 original/G0 逐位一致。
当前最强解释是 original 与无注入重编译方法存在各自的首次调用瞬态，随后三条
路线汇合到同一稳态；具体是 CUDA 首形状、decorator/编译缓存还是方法绑定年龄
效应，现有数据不能进一步区分。

D4 仍保持 `valid=false`。在解释 D4 已有单次预热为何仍留下失败前，不得直接
宣布 OFF-G2 通过或进入 D5。

## 下一步

下一轮只做离线诊断闭环：把 D4 的 prefix snapshot、每单元原始/G1/G2预热和计分
调用逐项映射到 D4A 的三个输出簇，解释两份证据的表面冲突；随后才可提出一个
预先固定、路线平衡且不自适应调参的稳态 OFF 等价门草案。该轮不授权模型执行、
D4 重跑、active injection 或 Self 效果实验。
