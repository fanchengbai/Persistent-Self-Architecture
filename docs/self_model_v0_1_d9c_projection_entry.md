# Self Model v0.1 D9-C calibration-only projection contract 与 single-use 入口

## 本轮完成内容

D9-C 将已闭环的 D9-A/B 同路径因果隔离设计推进到可执行前的最后一道无模型门。本轮定义并实现：

- 仅使用32条 calibration capture 的字段分离 projection contract；
- 4×4 identity/goal网格每格两个replicate，先格内平均，再做双因素中心化闭式拟合；
- 0-based第15层、post-FFN residual、2560维、无bias、基础模型参数不可训练；
- projection artifact 的内嵌schema、参数digest、artifact digest和exclusive-create冻结规则；
- 在projection通过审计并写入之前不得解析64条held-out fixture或448-pair schedule；
- 全新authorization、single-use claim、projection/raw/report/failure/integrity路径；
- 同一进程、同一persistent wrapper的928-call未来入口与失败即停止生命周期。

本轮只实现这些接口并进行纯Python/AST验证。没有创建真实projection、机器authorization、claim或output，也没有导入RWKV/Torch、访问权重、加载或执行模型。

## Projection 与评分冻结

projection将identity和goal各自表示为四个2560维分支向量，每个分支RMS固定为calibration grand mean RMS的0.005。active使用identity+goal；两个mask分别只保留另一字段；两个swap使用循环后的对应字段；matched-random由artifact digest与fixture ID确定性生成并与active向量RMS匹配；wrapper-zero严格为零。synthetic active由独立正控制callback产生，不读取projection artifact。

D9-A已经冻结mask/swap端点，但真实计算还需要明确替代target code。D9-C在看见任何真实结果前固定A/B/C/D token为66/67/68/69，并沿用D9-A的`(identity×3+goal) mod 4`语义索引和code rotation。identity/goal swap code分别通过对应字段循环加一得到。mask字段特异性、swap跟随和target-alignment margin的逐项计算规则都写入projection contract，禁止事后改变。

## 未来 single-use 生命周期

未来D9-D只有在项目负责人给出配置中冻结的逐字授权后才能启动。入口顺序固定为：

1. 核对launcher确定性环境、干净GitHub main和机器authorization；
2. 探测锁定installed source；
3. 在导入Torch、访问权重或加载模型前独占消费claim；
4. 应用严格确定性策略并加载2.9B模型及单一persistent wrapper；
5. 完成32次calibration capture，拟合、审计并exclusive-create真实projection；
6. 仅在projection文件存在后读取64条held-out fixture和448-pair schedule；
7. 完成896次held-out forward与完整有序ledger后才计算冻结端点；
8. 成功写report/integrity，任何异常写failure；claim一旦创建便禁止重跑。

未来ledger包含32条capture记录和448条pair记录，共480条，代表928次forward。所有计分调用只走persistent wrapper；public不进入计划。D8-C数据、fixture、token、seed、claim和结果不作为D9数据复用。

## 权限边界与下一门

D9-C当前只证明projection contract、artifact审计、授权绑定、call-plan完整性和入口时序在无模型环境下成立。fake artifact明确不可作为研究证据。即使未来D9-D全部门通过，也只能得到非正式、非Self的工程候选结论。

下一步先在服务器运行无模型专项测试和静态验证器。通过后仍需项目负责人逐字授权D9-D真实单次执行；普通“继续/下一步”不构成模型执行授权。D8-C及历史重跑、D7-D/E、正式测试集、Self效果结论、Self Updater、raw-original和自动重跑继续关闭。
