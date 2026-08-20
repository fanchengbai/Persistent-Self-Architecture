# Self Model v0.1 D4A：fake-only 诊断 runtime

本轮把已冻结的 D4A 设计实现成项目内纯 Python runtime，但没有提供真实模型加载
或执行入口。runtime 模块不导入 `rwkv` 或 `torch`，只接受调用方显式传入的
base model、上游源码字节、globals 和 tensor API。

## G0 控制

`RWKV7RecompiledUnmodifiedRuntime` 从锁定源码中按与 OFF-G2 相同的规则选择
`RWKV_DE_VERSION` 未设置时的 `forward_one/forward_seq`，复制 AST、清空
decorator、复制 globals 并编译为新函数。每次调用只临时绑定这两条函数，结束或
异常后删除。它不设置 callback 属性，也不插入 None-guarded 分支，因此专门隔离
“重新编译和绑定新函数”这一工程边界。

版本、源码 digest、DE 环境、方法variant或实例属性冲突都会在调用前失败。
active 请求和伪装的 off 子类同样拒绝。

## 九次记录器

`execute_d4a_fake_or_authorized_diagnostic` 本身不加载模型。它固定使用 `[2764]`、
`state=None`、`full_output=false`，按设计中的三轮拉丁顺序调用原始、G0、OFF-G2。
没有丢弃预热；9 次调用全部记录 logits 和每个 state tensor 的 shape、dtype、
device、numel 与 SHA-256。

比较器生成9个同路线组合和27个跨路线组合，记录 `torch.equal`、非等元素数、
最大/平均绝对误差和首个失败state组件。分类只用于区分同路线不稳定、重编译/
绑定边界、None-guarded分支或混合差异；报告始终明确不能修改D4失败状态或授权D5。

## 当前验证和权限

fake tensor/model测试覆盖：G0逐位等价、decorator记录、临时绑定恢复、完整9次
inventory、36个比较、G2单独扰动分类、active/源码锁/冲突拒绝，以及模块不导入
RWKV/Torch。当前没有真实 CLI、single-use claim、模型工厂或服务器运行命令。

下一门若获确认，只能增加无模型的已安装源码静态验证和真实执行入口的默认关闭
外壳；任何真正加载2.9B的执行仍需后续独立授权。
