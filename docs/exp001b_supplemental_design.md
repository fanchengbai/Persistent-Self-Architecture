# EXP-001B 补充控制实验设计草案

版本：0.1 Draft  
日期：2026-08-03  
状态：B1–B7已由项目负责人确认；仅批准非Core开发门，未冻结、未生成测试集、未授权运行

## 1. 为什么需要 EXP-001B

EXP-001的实测核心效应很强，但冻结的8条件Core Set没有包含三项旧协议要求：

1. matched-context表面内容控制；
2. 每个状态条件同步运行的96条通用能力控制；
3. 正式prompt-visible试题的自由生成格式读出。

因此EXP-001B只负责补齐这三项。它不是EXP-001重跑，不修改E1–E3，不追加
样本追逐显著性，也不声称是对E1–E3的独立复制。

## 2. 设计原则

- 模型、tokenizer、运行时、答案边界和原任务模板全部继承；
- EXP-001的320组和continuous原始分数只作为不可修改的配对参照；
- 新指标和阈值在读取任何EXP-001B结果前冻结；
- 新runner必须先通过非Core开发门；
- 生成补充测试集、正式运行和结果观察仍由三个独立边界控制；
- EXP-001B失败时保留EXP-001的正面结果，但不能授予完整载体资格。

## 3. 新增数据规模

| 模块 | 新记录数 | 用途 |
|---|---:|---|
| matched-context | 5,120 | 检查是否只是追随相同标签或最近内容 |
| 正式生成格式 | 5,120 | 补正式format与答案位置能力门 |
| 96条控制 × 8条件 | 768 | 检查state干预是否损伤无关能力 |
| 合计 | **11,008** | 仅补控制，不重复E1–E3 |

样本量不根据已经观察到的EXP-001效应缩减。matched-context继续使用全部320个
factorial groups和完整四代码轮换。

## 4. Matched-context 构造

### 4.1 保持相同的部分

每条matched-context记录复用对应冻结试题的：

- query文本；
- 四个选项及A–D位置；
- identity/goal标签；
- query template、filler variant和factorial group配对；
- 完整四代码轮换。

### 4.2 唯一改变的部分

真实历史中的“当前状态绑定”改成明确无关的设备日志、外部归档、假设例子或
被拒绝的未授权提议。domain和operation标签各出现一次，但文本明确说明它们
没有设置当前状态。

matched历史复制原始131-token filler，并只用预先固定的中性片段补齐，使其
token数与配对真实history prompt完全相同。如果任一模板比原历史更长、无法
只靠增加中性padding精确匹配，开发门失败并在冻结前修订；不得在正式结果后
选择性丢弃。

### 4.3 主要补充终点

每组先完成四代码语义边缘化，再计算：

```text
Delta J_matched = mean(J_continuous - J_matched_context)
```

Go要求：均值至少0.50、95%区间下界大于0、单侧配对符号翻转p<0.05。
continuous来自已验证的EXP-001原始包，不重新运行。

## 5. 同步通用能力控制

复用D5已经冻结的96条控制及原seed，重新生成后manifest digest必须精确等于
`30d984fc3eac987a27f27b8539b96cc7e2fd600ccd9a8c8904c7aa4f67a2e348`。
它包含：

- 32条答案代码复制；
- 32条无关单字段词法匹配；
- 32条无关双字段符号匹配。

96条控制确定性分配给96个不同的冻结group，四种来源组合各24条。每条在
continuous、restored、reset、random-matched、三种swap及无state的
prompt-visible-reset基线下运行。控制目标始终显式可见，与I/G无关。

每个condition × task单元都必须满足：

- 相对基线准确率下降不超过5个百分点；
- 格式有效率下降不超过2个百分点；
- 平均目标log-prob下降不超过0.25；
- 不能触发已冻结的state norm或来源不变性警报。

这些是预声明的损伤警报阈值，不包装成新的主要显著性检验。

## 6. 正式生成格式读出

对全部5,120条冻结prompt-visible试题，只新增确定性greedy格式探针：

1. 使用已经冻结的`<think></think` assistant前缀；
2. 强制自然答案边界`>\n`，同时检查其greedy exact率必须为100%；
3. 最多继续生成4个token；
4. 第一个完成的去空白文本必须严格等于A、B、C或D之一。

Go要求：格式有效率至少99%，joint准确率95%下界至少80%，identity和goal
准确率下界至少90%，四答案位置准确率最大差不超过0.25。该模块只补生成能力
与格式证据，不重新估计EXP-001的score-based E1–E3。

## 7. State norm 开发校准

旧方案写下了99.9%开发分布警报，但没有把2.9B的逐组件数值阈值持久化。
EXP-001B不得在正式数据中临时估门槛，因此先用64个非Core开发group、
`amber/cobalt × orbit/prism`标签和相同长度协议，记录每个组件的RMS分布并冻结
最近秩99.9%分位数（`ceil(q*n)`；64例时等于开发最大值）。该门不读取Core Set，
不计算EXP-001B行为结果。

## 8. 统计与可解释范围

- matched-context：320个group、10,000次BCa、100,000次单侧符号翻转；
- controls：按预声明point alerts逐cell判定，任何一格失败即不能完整Go；
- generation：按完整5,120条、group cluster区间和答案位置分层报告；
- 不重算E1–E3，不改变原Holm结果，不合并两次实验的p值；
- EXP-001B通过时，允许写“EXP-001 + EXP-001B共同闭合Phase 2行为控制包”；
- 不允许写“EXP-001B独立复制了E1–E3”。

## 9. 实施门顺序

```text
B-Dev1 非Core matched-context/token/norm校准
→ B-Dev2 非Core runner与格式探针
→ 生成预注册候选与源码digest
→ 项目负责人确认完整checksum
→ 生成并冻结EXP-001B补充测试集
→ 新主机preflight
→ 项目负责人单独授权正式运行
→ 全量运行（无中间指标）
→ 独立完整性验证
→ 冻结只读分析
→ Phase 2 Go / Revise / Stop
```

任何一步失败都保留记录。实验完成后禁止自动重跑，补充结果观察前禁止报告
中间准确率。

## 10. 当前需要确认的设计决定

项目负责人已于2026-08-03确认以下内容，确认范围仅包括设计与B-Dev1/B-Dev2开发：

- B1：EXP-001B只补控制，不复制E1–E3；
- B2：matched-context使用4种明确无绑定模板并精确token配对；
- B3：复用原D5的96条控制，确定性分配给96个不同group；
- B4：全部5,120条prompt-visible试题增加greedy生成格式探针；
- B5：matched条件使用原0.50 joint-margin SESOI，控制损伤阈值原样继承；
- B6：正式新增记录固定为11,008，不因EXP-001结果缩减；
- B7：先完成两个非Core开发门，再进入checksum冻结，当前不生成测试集、不运行。

该确认不是预注册候选checksum确认，不授权生成EXP-001B补充测试集，也不授权
正式运行。两个开发门的固定云端顺序为：

```bash
bash scripts/run_exp001b_bdev1_gate.sh
bash scripts/run_exp001b_bdev2_gate.sh
```

B-Dev1必须先得到`valid=true`，B-Dev2才会接受其summary、matched-context报告和
96组件RMS阈值。两步都只使用`amber/cobalt × orbit/prism`非Core材料。

首次B-Dev2使用通用runner夹具同时承担格式与norm资格检查，暴露了输入分布错配：
通用夹具历史不是131-token正式history族，导致四个state均有66–67/96组件超过
正式形状阈值；16条通用生成题中两条进入`<tool_call`。该失败保留为v0.1 Revise。

B-Dev2 v0.2不改8条件runner、matched-context、阈值或正式设计，只把资格探针换成
64条与正式history/query/filler结构相同的非Core材料。四个query、四个history、
四个filler、四个答案位置各自平衡为16条；正式格式阈值仍为0.99，因此64条中必须
64条全部格式有效。State norm也只比较同一正式形状族，避免把任意短prompt误当成
正式state异常。v0.2写入新目录，不覆盖v0.1失败报告。

## 11. 生成基础设施状态（2026-08-04）

EXP-001B最终预注册包已经冻结，最终digest为
`976cce8c9e3b53bca2d21ae43f273228c45dfc4607f5b652a3d5b5cdc5d823be`。
当前已实现下一阶段所需工具，但没有借此扩大授权：

- `exp001b-set-preflight`核对最终包、父Core Set包、固定digest、记录预算，以及
  12个生成相关源码、配置、Schema和脚本的SHA-256清单；它不加载模型、不评分
  试题、不生成数据；
- `exp001b-set-generate`固定生成5,120条matched-context、5,120条生成读出和
  768条控制条件记录；
- 生成入口同时要求项目负责人授权文件与
  `PSA_EXP001B_SET_GENERATE=AUTHORIZED_EXP001B_SET_GENERATION`执行锁；
- 授权原文必须同时绑定云端实时预检digest和最终预注册digest；源码、主机或包状态
  改变后，旧预检对应的授权不能复用；
- `exp001b-set-verify`独立检查记录数量、唯一ID、文件SHA-256、payload root、
  set digest、package digest和“实验仍未授权/未运行/未观察”边界；
- 生成成功只会把状态推进到`supplemental_set_frozen_unrun`，不会运行模型。

因此当前准确状态是“工具完成、等待补充集生成授权”。在项目负责人给出新的、
精确绑定最终预注册digest和11,008条记录预算的授权前，不得运行正式生成脚本。
即使以后补充集获准生成并冻结，正式补充实验仍需要另一次独立授权。

首次云端生成前预检于2026-08-04通过，digest为
`1f117d5cd0e9d37706c50bc10db37bf826eaa7166a366589e8ab4121499cfa65`。
但授权前审计发现该版本未把生成器源码纳入digest，因此该checksum不用于授权。
加固版已把完整生成源码清单纳入预检，须在云端更新代码后重新运行；负责人只能
逐字确认加固版的新digest。首次预检本身不构成生成授权。

加固版云端预检已经通过，有效digest为
`dbb17d975d32956fb92ab39975452f00c6bc1ca8b4114afec84f5a46f8242ae6`。
7项检查全部通过且12文件生成源码清单完整；该digest现在可以提交项目负责人做
“只生成并冻结补充集、不运行正式实验”的独立授权确认。

项目负责人随后逐字确认该加固版digest与最终预注册digest，授权生成并冻结固定
11,008条补充记录，并明确不授权运行正式补充实验。当前权限因此只推进到数据包
生成与独立完整性验证；生成完成后状态仍必须是`supplemental_set_frozen_unrun`。

补充集已在云端按该权限生成并通过独立验证：总计11,008条，4个锁定文件零失败，
set digest为`7c3606be819d4e6cc5420f0bf36efd1906f8954d362d83e912785cc943565d33`，
package digest为`68e9a9a79fe4e493a0c64ba8c0278c300cc832d940ab902feaceb4ad7f9d5954`。
包状态为`supplemental_set_frozen_unrun`，正式实验授权、运行和结果观察仍全部为false。

冻结包随后由Git提交`62dd8b2`持久化，提交范围严格只有5个包文件。本机从远程
快进同步后再次运行独立验证器，11,008条记录、锁定文件、内容结构以及set/package
digest全部一致。该里程碑不扩大权限；下一步只能开发正式运行预检、runner、原始包
验证和独立运行授权锁，不能直接运行补充实验。
