# Self Model v0.1 D4A：服务器无模型静态复验

本门只在服务器读取已安装 `rwkv==0.8.32` 的包元数据和 `rwkv/model.py` 字节，
不导入 `rwkv.model` 或 `torch`。它不访问权重，不创建执行 claim，也没有模型加载
或推理入口。

静态探针对同一份锁定源码分别执行两种 AST 检查：

1. G0 选择未设置 `RWKV_DE_VERSION` 时的 `forward_one/forward_seq` else variant，
   记录原始 decorator，随后确认编译候选 decorator 已清空且没有 callback 分支；
2. OFF-G2 对两条路径的 body/else variant 都检查唯一 post-FFN 注入点，并记录未
   设置 DE 时选择的源码行。

两条检查必须在每条执行路径上得到相同 candidate 数、条件、selected branch 和
selected source line。真实方法原始 decorator 预期为 `MyFunction`；因为 G0 与
OFF-G2 都清空 decorator，这项证据明确记录重编译边界，而不是假设 decorator
不存在。

服务器执行前先运行 D4A 的12项设计/runtime/manifest fake测试，再运行本静态
探针。报告必须明确模型/Torch未导入、权重未访问、模型未加载/执行、真实入口与
claim未实现。通过只允许进入下一轮“默认关闭的真实入口设计/实现”讨论，不能
直接运行2.9B、改写D4失败或进入D5。
