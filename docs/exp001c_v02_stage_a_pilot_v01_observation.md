# EXP-001C v02 Stage A Pilot v01 观察记录

状态：`stage_a_positive_control_pass_closed_no_rerun`

观察日期：2026-08-11

本记录仅覆盖负责人明确授权的一次 prompt-visible、非 Core、32 条 Stage A pilot。
未授权 Stage B、正式测试集、正式运行或自动重跑。

## 1. 授权、完整性与边界

- 执行 commit：`94847dad44154a677053fc774bd53701163ef5b3`
- manifest digest：`6874ea634ae6eb55421fe59d12dca0d910ba097eef63e3314db6f4a6ecea0909`
- preflight digest：`0a9dfbf299c96a4f3f121fe94ff0beb8771edde2757ef36541285f977ca10a52`
- authorization digest：`9f8dd3d9f41e744f9eed73298283594b2312f7872a60269ca14e74d1b7c62e08`
- Stage A result SHA-256：`763090bce72837e0957569e9abbeeda0db211c07d6d29e77a6fdf910889f3dcb`
- `record_count=32`
- `prompt_visible_only=true`
- `stage_b_recurrent_state_accessed=false`
- `formal_test_set_accessed=false`
- `formal_run=false`
- `contains_confirmatory_decision=false`
- `automatic_rerun_authorized=false`

模型只执行了授权的 Stage A。运行结束后进程退出，未启动第二个进程；结果文件 SHA-256
与 summary 完全一致，32 个 sample ID 唯一，8 个语义案例均具有完整的四代码轮换。

## 2. 锁定正控制结果

| 指标 | 结果 | 预设门槛 | 判定 |
|---|---:|---:|---|
| label-marginalized top-1 accuracy | 28/32 = 0.875 | ≥ 0.8 | 通过 |
| 最大单一预测代码占比 | 9/32 = 0.28125 | ≤ 0.5 | 通过 |
| forced prefix tokenizer roundtrip | 32/32 = 1.0 | = 1.0 | 通过 |
| forced prefix greedy exact | 32/32 = 1.0 | 次要指标 | 记录 |
| 平均 target margin over best incorrect | +1.503184 | 描述性 | 记录 |

预测代码计数为 A=9、B=9、C=8、D=6，没有复现 v01 的固定 A 策略。

| rotation | 正确数 | accuracy |
|---:|---:|---:|
| 0 | 6/8 | 0.750 |
| 1 | 7/8 | 0.875 |
| 2 | 8/8 | 1.000 |
| 3 | 7/8 | 0.875 |

所有预先锁定的 Stage A 正控制检查均为 true，因此本轮判定为
`stage_a_positive_control_pass`。

## 3. 解释边界与下一步

本结果说明修改后的 G1 fake-think/chat prompt-visible 协议能够承担后续状态比较所需的
基础正控制，并且 A-D 轮换有效抑制了单一代码塌缩。它不包含 recurrent-state 条件，
因此不能证明或否定 recurrent state 的语义保持能力，也不改变 EXP-001B 的确认性结论。

本次授权在单次运行完成后关闭，不允许自动重跑。Stage B 仍需新的、独立的负责人授权；
在获得该授权前，只允许离线审查、设计和代码测试，不得加载模型执行 Stage B。
