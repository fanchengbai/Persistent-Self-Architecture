# Self Model v0.1 D4B 真实入口服务器无模型静态复验观察

## 1. 观察范围

本记录只观察项目负责人贴回的 D4B 真实入口静态报告和安全文件存在性检查，不
运行真实runner或模型。贴回内容未包含`git rev-parse HEAD`、11项入口测试和57项
Phase 3组合测试输出，因此不补写未观察到的服务器提交号或测试计数。

## 2. 静态报告核对

- 状态：`d4b_real_off_equivalence_entry_static_verified`；
- `valid=true`；
- 27/27项入口、调度、授权和安全检查全部为真；
- claim调用位于模型配置digest、资产验证、模型加载和runtime核心之前；
- 12个配置、Schema、文档、脚本、源码和测试digest与本地最终报告一致；
- 报告digest为
  `3c03a87e7a569d24f51b38713e14c26d2dc338386cca9f4d63c4d6464b8724d2`，
  与本地最终报告匹配；
- 服务器`git status --short`无输出。

## 3. 未执行证据

静态报告中的RWKV/Torch导入、权重访问、模型加载/执行、机器授权、执行claim、
active injection、Self效果实验、D5和自动重跑字段全部为假。项目负责人还分别
执行文件不存在检查并得到：

- `machine authorization absent`；
- `execution claim absent`。

因此本轮只是入口源码级跨主机静态门，不是2.9B D4B执行或结果观察。

## 4. 下一门

真实D4B单次执行与本次结果观察必须等待项目负责人逐字发送冻结配置中的完整授权
文本。普通“确认”“下一轮”或此前D4A授权均无效。授权后也只允许在最终干净main
上创建一次机器授权和claim，执行固定21次调用；成功或失败均停止，D4失败记录、
D5、active injection和Self效果实验不会自动改变或获得授权。
