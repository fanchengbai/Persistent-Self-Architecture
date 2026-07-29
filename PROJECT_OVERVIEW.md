# Persistent Self Architecture (PSA)

## 持续自我架构研究项目

---

## 1. 项目简介

Persistent Self Architecture (PSA) 是一个探索人工智能智能体内部认知结构的研究项目。

本项目不试图定义或创造“意识”，也不假设机器一定能够产生主观体验。

我们的目标是：

> 探索一种具有持续内部状态（Persistent Internal State）、自我模型（Self Model）以及世界模型（World Model）的人工智能架构，并研究这种架构是否能够产生类似有意识智能体的行为特征。

重点研究：

- 身份连续性（Identity Continuity）
- 元认知能力（Metacognition）
- 自我状态演化（Self State Evolution）
- 长期目标保持（Long-term Goal Persistence）
- 内部状态对行为的因果影响（Causal Influence）

---

# 2. 核心研究假设

当前大语言模型主要解决：

```
输入
 ↓
理解世界
 ↓
推理
 ↓
输出
```

即：

**World Modeling（世界建模）**

但是人类智能不仅包含：

“世界是什么？”

还包含：

“我是谁？”

“我正在做什么？”

“我为什么这样决定？”

“我是否应该改变？”

因此提出：

```
智能体 =
World Model
+
Self Model
```

其中：

## World Model

负责：

- 外部世界理解
- 知识表示
- 因果推理
- 预测未来状态


## Self Model

负责：

- 身份连续
- 当前目标
- 偏好
- 价值倾向
- 能力评估
- 不确定性
- 内部冲突
- 自我更新

---

# 3. 核心理念

## Self ≠ Memory

记忆只是保存信息。

Self 是：

> 一个持续存在，并且能够影响未来行为的内部状态。


例如：

普通 Memory：

```
我曾经喜欢数学
```

Self State：

```
我是一个长期探索数学的人，
这个偏好会影响我的未来选择。
```


---

## Self ≠ Persona

Persona 是外部设定：

```
你是一名医生
```

Self 是内部演化：

```
经过长期经历，
形成稳定行为倾向。
```

---

## Self 必须具有因果作用

如果删除 Self State：

行为应该发生系统性变化。

如果交换两个智能体的 Self State：

行为应该产生对应迁移。

否则 Self 只是描述，不是真正机制。

---

# 4. 总体架构

```
                 Input
                   |
                   |
        +----------+----------+
        |                     |
        ▼                     ▼

  World Model             Self Model

  世界状态 W_t             自我状态 S_t

  -知识                   -身份
  -环境                   -目标
  -预测                   -偏好
  -推理                   -价值

        |                     |
        +----------+----------+

                   |
             Coupling Layer

                   |

              Policy / Action

                   |

             Environment Feedback

                   |

        Update World + Self State

```

---

# 5. 技术路线

## 第一阶段：研究基础状态

目标：

研究现有模型的内部状态是否已经包含类似 Self 的信息。

候选模型：

- RWKV
- Mamba
- State Space Model


实验：

- State 保存与恢复
- State 插值
- State 交换
- State Probe
- State 消融


---

# 第二阶段：加入显式 Self Model

设计：

```
RWKV World State

        +

Self State

        ↓

Coupling Mechanism

        ↓

Output
```


Self State 初始包含：

```
Goal

Preference

Confidence

Curiosity

Identity

Memory Summary

```

---

# 第三阶段：研究 Self Evolution

研究：

Self 是否可以：

- 长期保持身份
- 根据经验改变
- 形成新的偏好
- 产生内部目标
- 进行自我修正


---

# 6. 实验原则

## 不研究：

- 机器是否真的有意识
- 主观体验是否存在

因为目前没有科学统一定义。


## 研究：

可观察、可验证的行为：

### 1. 身份连续性

同一个智能体长期运行后：

是否保持稳定身份。


### 2. Self State 因果性

修改 Self：

是否改变行为。


### 3. Self 分化

相同模型：

不同经历

↓

是否产生不同个体轨迹。


### 4. 元认知

模型是否知道：

- 自己不知道什么
- 自己什么时候不可靠
- 是否需要进一步思考

---

# 7. 初始技术方案

基础模型：

```
RWKV-7 0.4B
```

原因：

- 具有天然 recurrent state
- 状态持续存在
- 适合研究内部状态演化


初始架构：

```
Base RWKV
      |
      |
World State

      +

Self Encoder

      |

Self State

      |

Gated Injection

      |

Output

```

---

# 8. 开发原则

## 原则1：

先验证思想，再扩大模型。

不要一开始训练大模型。


## 原则2：

所有核心假设必须可实验验证。


## 原则3：

Self 必须具有因果影响。

不能只是 Prompt 或描述。


## 原则4：

允许失败。

如果最终证明：

Self Model 不能产生类似意识行为，

这个结果同样具有研究价值。

---

# 9. 项目长期目标

探索一种新的人工智能架构：

不是：

```
更大的模型
+
更多的数据
```

而是：

```
更好的内部结构

World Model
+
Self Model
+
持续状态演化

```

最终希望回答：

> 一个具有持续自我状态的人工智能系统，是否会表现出区别于传统 Agent 的新型智能行为？

---

# 10. 项目状态

Current Phase:

```
Phase 0 - Research Design
```

当前任务：

1. 文献调研
2. 建立研究地图
3. 分析 RWKV State
4. 设计 Self State
5. 构建最小实验


---

# License

MIT License

---

# Disclaimer

本项目不声称创造意识。

本项目探索的是：

计算架构、
持续状态、
自我模型、
智能行为之间的关系。

