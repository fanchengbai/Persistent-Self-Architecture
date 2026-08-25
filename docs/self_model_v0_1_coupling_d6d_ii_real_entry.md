# Self Model v0.1 Coupling-D6D-II installed source、manifest 与单次真实入口

日期：2026-08-25
状态：无模型实现；真实 D6D 联合执行尚未授权

## 本轮边界

D6D-II 把 D6D-I 已闭环的 wrapper/projection 工具推进到真实联合实验入口，但本轮只允许：

1. 读取已安装 `rwkv/model.py` 的字节与包版本；
2. 对锁定源码执行 AST 变换和 Python `compile`，不执行编译结果；
3. 冻结 projection 训练 manifest 与独立的 blinded non-Core pilot manifest；
4. 实现未来单次运行所需的新授权 Schema、唯一授权路径、唯一输出目录和 single-use claim；
5. 用纯 Python 测试入口顺序、manifest 展开、wrapper ownership 和失败关闭。

本轮不导入 `rwkv.model` 或 Torch，不读取权重/tokenizer资产，不加载或执行模型，不训练或构造真实 projection，不运行 pilot，不形成 Self 效果结论。

## installed source 静态兼容

服务器无模型验证可读取 `rwkv==0.8.32` 的 `model.py`，要求 SHA-256 精确等于项目锁定值。验证器将：

- 重新计算 source bytes digest，拒绝 provider 自报摘要与字节不一致；
- 定位唯一 `RWKV_x070` 及 `forward_one`/`forward_seq`；
- 在 `RWKV_DE_VERSION` 未设置的分支中为每条路径生成一个 post-FFN callback site；
- 把两个变换方法组成 AST module 并调用内建 `compile`；
- 不调用 `exec`，不导入 `rwkv.model`/Torch，不读取权重，不创建模型对象。

本地没有服务器的 installed package，因此本地报告允许把该项明确标记为 `remote_installed_source_probe_pending`。服务器必须用 `--probe-installed-source` 完成这一项，之后才可能进入独立真实执行授权门。

## projection 训练 manifest

训练清单固定四个 identity key 与四个 goal key 的 4×4 网格，共 16 次只读 capture。所有 capture：

- 使用同一个 wrapper；
- 固定 zero-based 第 15 层 post-FFN residual 的最后序列位置；
- callback 必须返回原 residual 对象，禁止修改模型输出；
- 基础模型权重冻结、真实实例字典不变；
- 训练 prompt 明确排除于 pilot，Self State 对象本身不序列化进 prompt。

闭式 trainer 计算 grand mean、四个 identity 条件均值和四个 goal 条件均值，把 grand mean 等分进两个无 bias 分支，再把各分支缩放到训练 residual RMS 的 0.5%。两个字段相加的名义尺度为 1%。parameter digest 与 artifact digest 必须在 pilot payload runtime load 前持久化。

## blinded non-Core pilot manifest

pilot 固定 12 个 fixture、三个 task family 各四个。每个 fixture 先运行一次不计分 OFF 预条件，再按 11 路 cyclic Latin 行运行一次，共 144 次 forward；其中 96 次使用冻结 Self projection，synthetic 正控制 12 次。

答案边界继续绑定已验证 tokenizer：`>\n` token IDs 为 `[63, 11]`，`A/B/C/D` 分别为 `66/67/68/69`。查询只呈现平衡选项，不写出“当前 Self State 是什么”；matched/paired identity 与 goal 来自 manifest 中独立的结构化 Self State。

每次同一 full-output forward 还在固定 teacher-forced 位置读取一个通用能力 sentinel，因此不增加调用数。sentinel 在三个 family 中完全相同；相对各 fixture 的 scored OFF，所有其他路线合计最多允许一次 sentinel code 变化。

pilot 只产生 non-Core engineering classification。即使全部方向性门通过，也不能升级为 Self 效果结论，不能自动打开 D6E。

## 未来单次入口顺序

真实 runner 的顺序冻结为：精确机器授权 → clean main/source binding → installed source 静态兼容 → claim → 模型配置/权重/tokenizer校验 → 模型加载 → 一个 wrapper → 16 次训练 capture → projection artifact 落盘 → runtime load pilot payload → 144 次 pilot → report 或 failure 后停止。

授权、claim 和输出目录均为全新路径。任一失败都会留下 failure，claim 视为已消费；禁止 D5C/P1/P2/D6C/D6D 重跑、raw-original、拆分纯机制运行和自动重跑。

## 下一门

服务器 `--probe-installed-source` 无模型复验通过后，真实联合执行仍需项目负责人逐字给出：

> 授权执行 Self Model v0.1 Coupling-D6D 真实2.9B单一联合projection训练与非Core pilot一次（同一进程、同一wrapper、16次只读训练capture后冻结真实projection，再按12个fixture各1次OFF预条件和11条件调度执行144次pilot，共160次forward），并授权观察本次工程结果；不授权重跑D5C/P1/P2/D6C或D6D、自动重跑、D6E、正式测试集、Self效果结论、Self Updater、raw-original路线或任何拆分机制运行。
