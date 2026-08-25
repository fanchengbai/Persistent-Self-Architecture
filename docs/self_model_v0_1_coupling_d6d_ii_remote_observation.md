# Self Model v0.1 Coupling-D6D-II 远程无模型观察

日期：2026-08-25  
状态：远程 installed source 静态兼容与单次真实入口无模型门通过；等待独立逐字执行授权

服务器运行 D6D-II 专项测试得到 12/12 `OK`。入口报告的 24 项检查、训练与 pilot manifest 报告的 35 项检查全部为真；总报告状态为 `d6d_ii_installed_source_and_real_entry_no_model_verified`，digest 为 `93b0b09dfb303a963eb9fa073b99e71235488ab0a4c26720549cb3a13bf1496f`。

## Installed source 静态兼容

服务器探测到 `rwkv==0.8.32`，锁定源码为 `/root/autodl-tmp/psa-exp001c-venv/lib/python3.12/site-packages/rwkv/model.py`，SHA-256 为 `75482aee89a08d2a8c8dbe628110b317fc8d0974ddffbaa52aa19190667305e0`。该源码只经过读取、AST 变换与 compile-only 检查：9 项兼容检查全部为真，`forward_one` 与 `forward_seq` 均找到且各形成一个冻结 post-FFN 注入点，`exec_called=false`。installed-source probe 报告 digest 为 `300f3f19d1bc937ab2cfbae58abfd9b95189701f086ee77a38ec78e95c0baa0d`。

此检查没有导入 `rwkv.model` 或 Torch，没有访问权重、加载模型或执行 forward。它证明的是服务器已安装源码与 D6D persistent instrumenter 的静态兼容性，不是模型运行证据。

## 联合训练与 pilot 清单

训练 manifest 冻结 4×4 identity/goal 网格的 16 次只读 layer-15 residual capture。pilot manifest 冻结 12 个 non-Core fixture，每个 fixture 包含 1 次 OFF 预条件及 11 条件调度，共 144 次；通用能力 sentinel 与相应条件共享同一次 full-output forward，不增加额外调用。未来单一联合运行总计 160 次 forward。

训练 manifest digest 为 `a0aa594f6b020b03546fa95d2cd135783b97c72436e84d31d27c37f369dab0fd`，pilot manifest digest 为 `4c18addedb26de4fc80438ad687d2606df5a68511cde96661b3b87e4c60023b2`，manifest 报告 digest 为 `bb1855dc905367b408f7dae5eefbb796ab177bdf648f5e09cb955ed27cb6288a`。展开承诺中的 call plan、fixture 与 training digest 分别为 `449e99614d9045b31215ae54ded758063e903f1790f30301016567536fdd7cf0`、`7c8ff5c88d8520845619db386fda25c23c8edb9f929b57b94fadef4bc34f17c3` 和 `e263bf7cc6babe895e76a2816f20bc81b04be62f13a10056760f0570b60a3a38`。

## 安全边界与结论

回传明确显示机器授权、execution claim 和真实 projection artifact 三者均缺席。权重访问、模型加载/执行、projection 训练/构造、pilot、正式测试集、Self 效果结论、Self Updater、raw-original 路线、D6E、历史重跑和自动重跑全部保持关闭。

回传没有包含 `git rev-parse HEAD` 或最终 `git status --short`。因此本结论只覆盖锁定 source inventory、installed source compile-only 兼容性和无模型入口行为，不声称额外验证了服务器 HEAD 或整个工作区清洁度。

D6D-II 无模型准备门现已闭环，但真实 2.9B 联合运行尚未授权。下一步只能等待项目负责人另行给出配置中冻结的逐字单次执行授权；本次回传和一般性的“继续”均不能创建机器授权、claim、projection 或启动模型。
