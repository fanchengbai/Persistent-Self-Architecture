# Self Model v0.1：D3服务器无模型静态复验

日期：2026-08-11

状态：D3工具已实现；只允许读取已安装包元数据和`rwkv/model.py`字节，不允许导入或执行模型

## 1. 复验目标

D3把服务器实际安装的`rwkv==0.8.32`与项目D2 wrapper连接到同一份可审计报告。脚本通过
`importlib.metadata`定位包文件并直接读取`rwkv/model.py`字节，核对固定SHA-256与文件大小；它不
执行`import rwkv.model`或`import torch`。

同时，D3重新核对D2报告自身digest、29项检查、8个源码digest、OFF-G1/OFF-G2状态，并对
项目内wrapper执行固定文件digest和AST审计。任何版本、源码、wrapper或D2证据变化都会让本门
失败关闭。

## 2. 明确不包含的操作

- 不读取checkpoint或其他权重文件；
- 不创建RWKV模型对象，不调用forward；
- 不修改`site-packages`；
- 不实现OFF-G2 instrumented runtime或active injection；
- 不选择真实层，不运行Self效果实验，不自动重跑。

因此，D3通过只证明“服务器已安装源码与D2 off-only wrapper的静态锁一致”，不证明2.9B输出
等价，也不证明添加Self Model后有任何行为效果。

## 3. 后续门

D3通过后仍需独立设计并实现OFF-G2项目内instrumented-off路径。真实2.9B加载与OFF等价执行
必须另行授权；active injection和效果实验还要等待两级OFF门通过以及新的设计与授权。
