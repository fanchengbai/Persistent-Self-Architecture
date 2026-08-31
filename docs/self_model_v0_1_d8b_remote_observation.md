# Self Model v0.1 D8-B 服务器纯离线复验观察

## 观察范围

项目负责人在远程服务器拉取 D8-B 后，执行了 D8-A/D8-B 两组专项测试和 D8-B 纯离线验证器，并贴回测试摘要、完整报告和终端末尾状态。本记录只观察既有输出，不触发服务器操作、installed source 探测或模型执行。

## 复验结果

- D8-A + D8-B 共 22 项测试在约 0.033 秒内全部通过，状态为 `OK`。
- D8-B 报告 `valid=true`，状态为 `d8b_deterministic_manifests_and_fake_endpoint_verified`。
- 分类为 `d8b_deterministic_manifests_and_fake_excess_drift_endpoint_verified_no_model`。
- 24 个计分 fixture、288 个 pair block 和 584 次未来 forward 承诺与本地冻结设计一致。
- 总报告 digest 为 `c7456e8705236e578466052ca52db3ad413a3e86b942b88b2dd890fbaa2e27d2`，与本地最终报告一致。

完整报告中的 13 项总检查全部为真：contract、D8-A设计、fixture/schedule/determinism/endpoint四份manifest、四文件哈希、fake acceptance、阈值不变、D7-C非复用、source inventory和无RWKV/Torch导入均通过。

## 跨主机一致性

贴回报告列出的 13 个 source digest 与本地逐项复算完全一致，包括：

- D8-A预注册config、设计代码、测试和设计文档；
- D8-B contract config、四份manifest、runtime、验证脚本、测试和说明；
- fixture/schedule commitments继续为`8976ac9f…499e`和`a53cf5ed…6465`。

因此可以确认：D8-B锁定源码inventory、manifest完整性和纯Python端点行为跨主机一致。贴回内容未包含`git rev-parse HEAD`的输出，所以本记录不独立声明服务器的具体commit；终端末尾`git status --short`未显示条目，未观察到新增工作树差异。

## fake验收观察

- 完全相同输出的distance为0；
- logits变化由logits distance主导；
- state变化正确定位到第94个组件；
- 双顺序route-specific情形得到正面工程判定，24/24 fixture及四层6/6均支持；
- 单顺序情形和共同background drift均为`inconclusive_no_route_equivalence_claim`；
- 缺失、重复、非有限和错误state数量均在决策前拒绝；
- 所有情形都保持`route_equivalence_claim=false`和`self_effect_conclusion=false`。

## 安全边界与下一门

报告明确记录：installed source未探测，真实runner未修改，执行入口未实现，RWKV/Torch未导入，权重未访问，模型未加载或执行；D8-C、D7-C修复/重跑、D7-D/D7-E、projection、正式测试集、Self效果、Self Updater、raw-original和自动重跑全部关闭。

D8-B服务器纯离线复验现已闭环。下一步只能由项目负责人单独确认是否进入D8-C真实数值可识别性协议设计；该确认仍不等于模型执行授权。
