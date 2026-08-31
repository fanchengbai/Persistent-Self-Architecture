# Self Model v0.1 D9-B deterministic manifests 与 fake-first causal endpoint contract

## 本轮物化内容

D9-B 把 D9-A 的预注册设计物化为四份确定性 manifest：

- calibration manifest：32 条 identity×goal×replicate capture fixture，只能作为未来 projection 拟合输入，不进入 held-out 端点；
- held-out manifest：16 个 identity×goal base case 各四个 code rotation，共 64 条全新 fixture；同一 base case 共享内容 token，只更换 rotation-code token；
- schedule manifest：七类 condition 均与 wrapper-zero 在同一 persistent wrapper 内成对比较，每类 64 pair 且 zero-first/condition-first 各 32；共 448 pair、896 次 held-out forward，连同 32 次 calibration 共 928 次未来 forward；
- endpoint manifest：逐项复制 D9-A 冻结的 99% bootstrap、13/16方向、字段层3/4、random、mask、swap和60/64 synthetic正控制门。

三份展开 commitment 保持不变：calibration=`2e8d555e…fc39`、held-out=`02d33c92…15e4`、schedule=`a6b34ef7…85b9`。四份 manifest 都绑定 D9-A config SHA-256，D9-B contract 再绑定四份文件 SHA-256。没有创建 projection contract、authorization、claim 或 output。

## fake-first endpoint

纯 Python ledger 含32条 calibration capture记录和448条 held-out pair记录；每条pair包含冻结ID、fixture、base case、contrast、Latin位置、顺序、同源state约束和两次标量margin观察，因此代表928次未来forward。它不构造tensor、projection、模型对象或真实输出。

合成验收冻结三类情形：

1. 字段特异候选：active-zero为正，同时通过random、mask、swap和synthetic门；结论仍只允许非正式、非Self的工程候选；
2. 同wrapper但无因果信号：主要端点失败；
3. active与random同样变化且mask/swap不特异：特异性门失败。

任何缺失、重复、重排、public路由、非有限margin、condition顺序改变或calibration/held-out泄漏都在计算结论前失败关闭。输入ledger在验证和计算后保持不变。

## 权限边界与下一门

本轮没有实现projection contract或真实projection，没有探测installed source、修改真实runner、实现执行入口、导入RWKV/Torch、访问权重、加载或执行模型，也没有创建authorization、claim或output。D8-C及全部历史实验继续禁止重跑。

下一步只允许先在服务器运行同一组纯离线测试和验证器。服务器复验闭环后，D9-C仍需项目负责人另行明确确认；普通“继续/下一步”不构成D9-C或D9-D授权。
