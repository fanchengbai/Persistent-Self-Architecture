# D8-C-I 服务器无模型静态复验观察

## 结论

D8-C-I single-use 真实 runner 的服务器无模型静态复验通过：D8-A、D8-B、D8-C 与 D8-C-I 合计 44 项测试全部 `OK`；报告 `valid=true`，状态为 `d8c_i_single_use_real_runner_static_verified`，分类明确为 `execution_not_authorized`。

服务器报告摘要 SHA-256 为 `831e3dc7dba84462a05e4e20a594049cc51cd82ba15729ba93a61bdd8192f1c1`，与最终提交 `9fd2dfc` 在本地重新生成的报告完全一致。提交前暂存报告 `fab21a57…158b` 因随后清理两个纳入 source inventory 文件的末尾空行而失效，不作为跨主机基准。

回传片段未包含 `git rev-parse HEAD` 和最终 `git status --short` 输出，因此本记录不声称直接观察到了服务器 HEAD 或工作树洁净状态；跨主机一致性结论限定为报告所锁定的 source inventory 和纯离线行为。

## 通过证据

- 14 项静态检查、11 项配置检查、9 项 source lock 与 7 类纯 Python 验收全部通过。
- 17 个 source digest 与最终本地报告逐项一致，包括 runner、launcher、验证器、授权 Schema、模型配置、D7-C wrapper、instrumenter 及 D8-A/B/C manifests。
- 纯 Python 展开固定为 8 次 conditioning、288 个 pair block、总计 584 次未来 forward；call-ID digest 为 `7004dd99e62d0657be968096f83b4099b6752cb07bf203577c31b487db3190ca`。
- AST 顺序门确认 launcher 环境预检先于授权创建，runner 内部按 launcher preflight、authorization、installed-source probe、claim、运行期依赖、严格确定性和模型加载的冻结顺序排列。
- authorization 篡改拒绝、调用数/ID/顺序、conditioning/pair 数量和无模型对象创建均通过验收。

## 安全边界

机器 authorization、execution claim、raw comparisons、report、failure 和 integrity 均未创建；未来 execution output namespace 缺席。报告同时确认 installed source 未探测、RWKV/Torch 未导入、权重未访问、模型未加载或执行。

因此本次只证明 D8-C-I runner 的静态时序、授权绑定、584-call 展开与失败关闭机制在服务器环境可复验。它不是 D8-C 真实执行，也不产生数值可识别性或 Self 效果结论。D8-C 真实单次执行仍必须由项目负责人另行逐字授权；D8-C/历史重跑、自动重跑、D7-D/D7-E、projection、正式测试集、Self Updater 和 raw-original 路线继续关闭。
