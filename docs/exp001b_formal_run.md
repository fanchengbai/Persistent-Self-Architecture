# EXP-001B 正式运行准备

## 1. 这一阶段做什么

冻结补充集相当于已经封好的试卷，本阶段只建设考场、监考记录和验卷程序，不代表允许开考。
正式运行器处理三类冻结记录：

| 记录类型 | 数量 | 通俗解释 | 原始输出 |
|---|---:|---|---|
| matched-context | 5,120 | 给模型同样长、也出现同样词，但明确“不构成当前状态”的历史，检查优势是否真来自绑定 | A–D原始分数、state norm报警信息 |
| formal generation | 5,120 | 使用冻结的完整可见提示，让模型真实生成答案，检查格式与能力 | A–D原始分数、最多4个生成token、原始文本 |
| general capability controls | 768 | 96道无关能力题分别经过8种state条件，检查干预是否顺手破坏普通能力 | A–D原始分数与条件路由信息 |

这些记录按父Core Set的320个组运行：224组各32条，96组各40条，总计11,008条。

## 2. 安全与完整性规则

- 非Core开发门不得读取父Core Set或冻结补充集；它只验证真实模型下的三类代码路径。
- 正式预检不加载模型、不评分记录；它把代码、模型、资产、三个冻结包、开发证据和当前主机写入checksum。
- 正式运行必须同时具有项目负责人授权文件和环境执行锁。
- 授权必须明确允许完整补充实验和全量完成后观察结果，同时明确禁止修改设计、自动重跑和重跑EXP-001。
- 每组结果先写临时文件再原子替换；中断后需要显式`--resume`，且已完成组的SHA-256必须保持不变。
- 运行器不输出准确率或中间决策。只有320组和11,008条全部完成并通过独立原始包验证，才允许进入冻结只读分析。

## 3. 云端顺序

拉取包含本阶段代码的提交并激活虚拟环境后，先运行：

```bash
bash scripts/run_exp001b_runner_development_gate.sh
cat results/development/exp001b_formal_runner_dev/summary.json
```

只有当该summary为`valid=true`，才运行：

```bash
bash scripts/preflight_exp001b_supplemental_run.sh
cat results/development/exp001b_run_preflight/preflight.json
```

到这里必须停止。预检产生的新checksum需要项目负责人单独确认；不要创建授权文件，不要设置
`PSA_EXP001B_RUN=AUTHORIZED_EXP001B_SUPPLEMENTAL_RUN`，也不要运行正式脚本。

## 4. 完成后的独立验收

未来获得明确授权并全量运行后，`scripts/verify_exp001b_supplemental_raw.sh`会逐一核对320个组文件、
11,008个冻结record ID、每条原始输出结构、授权链、所有SHA-256账本和总payload digest。
验证器同样不计算研究准确率；通过状态固定为`raw_package_verified_unanalyzed`。
