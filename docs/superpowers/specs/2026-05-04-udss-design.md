# UDSS - 通用决策支持系统设计文档

**日期**: 2026-05-04
**项目**: Universal Decision Support System (UDSS)
**版本**: 1.0
**状态**: 设计中

---

## 1. 项目概述

### 1.1 项目目标

构建一个**通用决策支持系统**，能够：
- 接收人类任何领域的问题输入
- 通过数学公式计算最优决策
- 结合WebSearch获取最新相关信息
- 输出可执行的动作方案（文本格式）
- 通过强化学习持续优化决策质量

### 1.2 核心公式

```
Decision(E, Q, R) = argmax_a [ α · V(a) + β · Sim(R, a) - γ · C(a) ]
```

| 符号 | 含义 |
|------|------|
| E | 环境向量（时间/空间/社会/信息维度） |
| Q | 问题向量（问题类型/紧急度/复杂度） |
| R | 搜索结果向量（相关性/时效性/权威性） |
| a | 候选动作/方案 |
| V(a) | 动作价值函数（10维Schwartz价值观） |
| Sim(R,a) | 搜索结果与动作的相似度 |
| C(a) | 执行成本（时间/金钱/风险） |
| α, β, γ | 可学习权重 |

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     UDSS 系统架构                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ 用户    │→   │ 问题理解层   │→   │ 数学公式引擎      │   │
│  │ 输入    │    │ QueryParser  │    │ Decision Engine  │   │
│  │ (Q)     │    └──────┬───────┘    └────────┬─────────┘   │
│  └─────────┘           │                     │              │
│         ┌─────────────┴─────────────────────┘              │
│         ↓                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ WebSearch    │→   │ 方案生成器   │→   │ 强化学习    │   │
│  │ 实时信息(R)  │    │ LLM TextGen │    │ RL Optimizer│   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                              ↓                              │
│                    ┌──────────────┐                        │
│                    │ 用户反馈     │                        │
│                    │ (评分/理由)  │                        │
│                    └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
1. 用户输入问题 Q
2. QueryParser 解析 Q → E (环境向量) + K (查询向量)
3. WebSearch(K) → R (搜索结果)
4. Decision Engine: argmax_a [α·V(a) + β·Sim(R,a) - γ·C(a)] → 最优方案 a*
5. TextGen(a*) → 详细文本方案
6. 用户执行方案，反馈评分
7. RL Optimizer 更新 α, β, γ 权重
```

---

## 3. 核心模块设计

### 3.1 问题理解层 (QueryParser)

**职责**：
- 解析用户输入为结构化向量
- 识别问题类型、紧急度、复杂度

**输入**: 用户问题文本 (Q)
**输出**: 环境向量 (E) + 查询向量 (K)

**向量维度**：
- E: 40维 = 时间(10) + 空间(10) + 社会(10) + 信息(10)
- K: 20维 = 问题类型(5) + 关键词(10) + 约束(5)

**问题类型分类**：
| 类型ID | 类型名称 | 描述 |
|--------|----------|------|
| 0 | lifestyle | 生活决策（旅行/购物/餐饮） |
| 1 | professional | 专业决策（商业/投资/技术） |
| 2 | urgent | 紧急决策（危机处理） |
| 3 | personal | 个人发展（职业/学习） |
| 4 | social | 社交决策（人际关系） |

### 3.2 数学公式引擎 (Decision Engine)

**核心公式**：
```
Decision(E, Q, R) = argmax_a [ α · V(a) + β · Sim(R, a) - γ · C(a) ]
```

**动作价值函数 V(a)**：
基于Schwartz 10维价值观：
```
V(a) = Σ_k w_k · v_k(a)
```
- w_k: 可学习权重 (10维)
- v_k(a): 动作a在维度k的价值

**搜索相似度函数 Sim(R, a)**：
```
Sim(R, a) = cosine(embed(R), embed(a))
```

**执行成本函数 C(a)**：
```
C(a) = t_time · c_time + t_money · c_money + t_risk · c_risk
```

**可学习权重**：
- α, β, γ: 标量，通过强化学习更新
- w_k: 10维向量，通过强化学习更新

### 3.3 WebSearch 模块

**职责**：获取最新相关信息

**输入**：查询向量 K
**输出**：搜索结果向量 R

**搜索结果结构**：
```python
{
    "title": str,           # 文章标题
    "url": str,             # URL
    "snippet": str,         # 摘要
    "relevance": float,     # 相关性分数 [0,1]
    "freshness": float,     # 时效性分数 [0,1]
    "authority": float,    # 权威性分数 [0,1]
}
```

**聚合向量 R 计算**：
```
R = Σ_i ω_i · r_i
其中 ω_i = relevance_i · freshness_i · authority_i
```

### 3.4 方案生成器 (TextGenerator)

**职责**：将数学公式输出的方案转为可读文本

**输入**：最优方案 a* (包含动作、参数、约束)
**输出**：详细文本方案

**输出格式**：
```
# 方案标题

## 概述
简要说明方案目标和预期结果。

## 执行步骤
1. [步骤1描述]
   - 理由：[为什么这样做]
   - 注意事项：[如果有的话]

2. [步骤2描述]
   ...

## 时间估计
预计总时长：[时间]

## 风险因素
- [风险1]: [应对策略]
- [风险2]: [应对策略]

## 备选方案
如果主方案不可行，可以考虑：
1. [备选1]
2. [备选2]
```

### 3.5 强化学习优化器 (RL Optimizer)

**职责**：根据用户反馈更新公式权重

**用户反馈格式**：
```python
{
    "score": int,      # 1-5分
    "rationale": str,  # 用户理由
    "accepted": bool   # 是否采纳
}
```

**奖励函数**：
```
R_user = w1 · usefulness + w2 · speed + w3 · satisfaction

其中：
- usefulness = 方案有帮助程度 (0-1)
- speed = 响应速度评分 (0-1)
- satisfaction = 用户满意度 (0-1)
```

**策略更新**：
```
θ_new = θ_old + η · ∇log π(a|S) · R_user

其中：
- θ = (α, β, γ, w_k)
- π(a|S) = Softmax(Decision(E,Q,R))
- η = 学习率
```

---

## 4. 动作空间设计

### 4.1 动作分类

系统支持两类动作：

**原子动作（Primitive Actions）**：
- send_message, wait, search, compare, recommend, reject, accept, modify, cancel, delegate

**复合动作（Composite Actions）**：
- reschedule_meeting, plan_travel, make_decision, solve_problem, negotiate

### 4.2 动作参数化

每个动作 a 包含：
```python
{
    "id": str,              # 动作唯一标识
    "name": str,            # 动作名称
    "type": str,            # primitive/composite
    "parameters": dict,     # 动作参数
    "prerequisites": list,  # 前置条件
    "outcomes": list,       # 可能结果
    "value_vector": np.array(10),  # 10维价值向量
    "cost_vector": np.array(3)     # [时间成本, 金钱成本, 风险]
}
```

---

## 5. 方案生成算法

### 5.1 方案候选生成

1. **动作选择**：从动作空间中选择top-k候选
2. **方案组合**：将候选动作组合成序列
3. **可行性检查**：验证方案满足约束
4. **评分排序**：按公式评分排序

### 5.2 方案验证

```
valid(plan) = ∧_{step ∈ plan} feasible(step) ∧ satisfied(constraints)
```

### 5.3 方案输出

选取评分最高的可行方案，传递给TextGenerator生成文本。

---

## 6. 强化学习详细设计

### 6.1 状态空间 S

```
S = (E, Q, R, a_history)

其中：
- E: 环境向量 (40维)
- Q: 问题向量 (20维)
- R: 搜索结果向量 (30维)
- a_history: 历史动作序列 (可变长度)
```

### 6.2 动作空间 A

```
A = 选择方案 a 的参数 (α, β, γ, w_k)
```

### 6.3 奖励函数

```python
def compute_reward(feedback: UserFeedback, context: dict) -> float:
    """
    R = w1·usefulness + w2·speed + w3·satisfaction + λ·consistency
    """
    base = w1 * feedback.usefulness + w2 * feedback.speed + w3 * feedback.satisfaction

    # 一致性惩罚：如果用户理由与决策不符
    consistency = 1.0 if feedback.rationale_supports_decision else 0.5

    return base * consistency
```

### 6.4 更新算法

使用策略梯度方法（Policy Gradient）：

```python
def update_weights(θ, grad):
    # 克隆当前权重
    θ_new = θ.clone()

    # 更新权重
    for key in grad:
        θ_new[key] = θ[key] + learning_rate * grad[key]

    # 裁剪权重范围
    θ_new = clip_weights(θ_new)

    return θ_new
```

---

## 7. 技术实现

### 7.1 技术栈

| 组件 | 技术选择 |
|------|----------|
| 核心语言 | Python 3.10+ |
| Web搜索 | DuckDuckGo API / SerpAPI |
| LLM文本生成 | Claude API / GPT-4 |
| 向量数据库 | FAISS / ChromaDB |
| 强化学习 | PyTorch + 自定义RL |
| Web框架 | Flask / FastAPI |
| 前端 | React + Ant Design |

### 7.2 项目结构

```
helga/
├── core/                    # 核心模块
│   ├── __init__.py
│   ├── query_parser.py     # 问题理解层
│   ├── decision_engine.py  # 数学公式引擎
│   ├── action_space.py     # 动作空间定义
│   ├── value_functions.py  # 价值函数V(a)
│   ├── cost_functions.py   # 成本函数C(a)
│   └── similarity.py       # 相似度函数
├── search/                  # 搜索模块
│   ├── __init__.py
│   ├── web_search.py       # Web搜索接口
│   └── result_aggregator.py # 结果聚合
├── generation/              # 方案生成模块
│   ├── __init__.py
│   ├── candidate_generator.py # 候选方案生成
│   ├── validator.py        # 方案验证
│   └── text_generator.py   # 文本生成
├── rl/                      # 强化学习模块
│   ├── __init__.py
│   ├── policy.py           # 策略网络
│   ├── reward.py            # 奖励计算
│   └── optimizer.py        # 权重优化
├── api/                     # API接口
│   ├── __init__.py
│   └── endpoints.py        # Flask/FastAPI端点
├── config/                  # 配置文件
│   └── default.yaml
├── tests/                   # 测试
├── docs/                    # 文档
└── main.py                  # 入口
```

### 7.3 API设计

**POST /api/decide**
```json
// Request
{
    "question": "我应该选择哪个offer？公司A给40万但要加班，公司B给30万但work-life balance",
    "context": {
        "deadline": "2024-05-01",
        "location": "北京"
    }
}

// Response
{
    "decision": {
        "recommended": "company_b",
        "confidence": 0.85,
        "reasoning": "基于您对work-life balance的重视..."
    },
    "plan": {
        "title": "选择公司B的决策方案",
        "steps": [...],
        "time_estimate": "2小时内做出决定",
        "risk_factors": [...]
    },
    "search_results": [...],
    "alternatives": [...]
}
```

**POST /api/feedback**
```json
// Request
{
    "session_id": "xxx",
    "decision_id": "yyy",
    "score": 4,
    "rationale": "分析很有道理但缺少行业薪资对比",
    "accepted": false
}
```

---

## 8. 验证标准

| 验证项 | 描述 | 目标 |
|--------|------|------|
| 公式有效性 | 决策公式输出质量 | 用户评分 > 4.0 |
| 搜索准确性 | 搜索结果相关性 | relevance > 0.7 |
| 方案可行性 | 生成方案的可执行性 | 可执行率 > 90% |
| RL收敛性 | 权重更新的收敛速度 | 5轮内收敛 |
| 响应速度 | 系统响应时间 | < 5秒 |

---

## 9. 实施计划

### Phase 1: 核心公式实现
- [ ] 实现QueryParser（问题解析）
- [ ] 实现Decision Engine（公式计算）
- [ ] 定义动作空间

### Phase 2: 搜索集成
- [ ] 集成WebSearch
- [ ] 实现结果聚合

### Phase 3: 方案生成
- [ ] 实现CandidateGenerator
- [ ] 实现TextGenerator

### Phase 4: 强化学习
- [ ] 实现RL Optimizer
- [ ] 集成用户反馈

### Phase 5: API与前端
- [ ] 开发REST API
- [ ] 开发Web界面

---

## 10. 公式汇总

### 核心决策公式
```
Decision(E, Q, R) = argmax_a [ α · V(a) + β · Sim(R, a) - γ · C(a) ]
```

### 价值函数
```
V(a) = Σ_k w_k · v_k(a)
```

### 执行成本
```
C(a) = t_time · c_time + t_money · c_money + t_risk · c_risk
```

### 策略梯度更新
```
θ_new = θ_old + η · ∇log π(a|S) · R_user
```

---

**文档版本**: 1.0
**创建日期**: 2026-05-04
**下次审查**: 设计完成后