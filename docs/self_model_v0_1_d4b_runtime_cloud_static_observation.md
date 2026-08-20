# Self Model v0.1 D4B runtime 服务器无模型静态复验观察

## 1. 观察范围

本记录只观察项目负责人贴回的 D4B fake-first runtime 静态报告，不运行模型。
贴回内容包含验证器摘要、完整报告和空的 `git status --short`，但没有包含此前
命令中的 `git rev-parse HEAD`、9项新增测试或46项Phase 3组合测试输出。因此本
记录不补写未观察到的服务器提交号和测试计数。

## 2. 完整性核对

- 状态：`d4b_fake_first_runtime_static_verified`；
- `valid=true`；
- 17/17项runtime设计与权限检查全部为真；
- 报告digest为
  `261325c459c08e5fa2c8d3e9ff08574c36e49ab36eccfabcf89e0045b7abae46`；
- digest与本地最终冻结报告一致；
- 九个配置、文档、脚本、源码和测试digest逐项与本地报告一致；
- 服务器`git status --short`无输出。

这些证据确认 D4B runtime 核心在服务器上完成源码级跨主机静态复验。它们不
证明本次未贴出的测试命令已经运行，也不是2.9B模型结果。

## 3. 安全边界

报告仅将`runtime_core_implemented`标记为真。RWKV/Torch导入、installed-source
探针、权重访问、模型加载与执行、真实入口、机器授权、执行claim、D4状态变化、
D5、active injection、Self效果实验和自动重跑字段全部为假。

因此D4失败保持不变，fake runtime通过只代表核心实现符合冻结设计，不能形成
D5审阅候选或Self效果证据。

## 4. 下一门

下一步只能在项目负责人另行确认后实现 D4B 真实执行入口的安全外壳和无模型静态
验证。该实现必须复用冻结21次计划，在任何模型访问前验证独立逐字授权并消费
single-use claim；实现入口本身不得创建授权、加载或执行模型。真实D4B执行与
结果观察仍需要之后独立、逐字、单次授权。
