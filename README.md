# HELGA - 人文环境学习与生成代理系统

**Humanistic Environmental Learning and Generating Agent**

一个计算模型，模拟人类"观测 → 思考 → 决策 → 行动"闭环。

## 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                     HELGA Agent                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  感知系统   │→ │  认知系统   │→ │  决策系统   │         │
│  │ Perception  │  │ Cognition   │  │  Decision   │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┴────────────────┘                 │
│                          ↓                                  │
│                  ┌─────────────┐                          │
│                  │  验证引擎    │                          │
│                  │ Verification │                          │
│                  └─────────────┘                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  价值观模块 │  │  规范模块   │  │  文化基模   │         │
│  │   Values    │  │   Norms     │  │  Culture    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## 项目结构

```
helga/
├── core/                    # 核心模块
│   ├── types.py           # 共享类型定义
│   ├── culture.py        # 文化参数（Schwartz价值观、规范矩阵、情感基模）
│   ├── perception.py     # 多头注意力感知系统 O_t (dim=32)
│   ├── cognition.py      # 贝叶斯心智推理（粒子滤波/SVI）
│   ├── decision.py       # 期望效用决策 U(a,Z)
│   └── action.py         # 环境闭环 + 在线文化适应
├── validation/             # 六层验证
│   ├── ecological.py     # 生态效度（JS距离 < 0.15）
│   ├── cultural_general.py # 文化泛化（Spearman |ρ| > 0.8）
│   ├── counterfactual.py # 反事实推理（准确率 > 85%）
│   ├── structural.py     # 结构一致性（效应量在CI内）
│   ├── temporal.py       # 时间鲁棒性（ICC > 0.9）
│   └── adversarial.py    # 对抗规范检测（拒绝率 > 95%）
├── memory/                # 记忆系统
│   ├── working.py        # 工作记忆（~10条）
│   └── episodic.py       # 情景记忆（~100条）
├── interfaces/
│   └── cli.py            # 命令行界面
├── config/
│   └── default.yaml      # 默认配置
├── main.py                # 入口
└── requirements.txt      # 依赖
```

## 核心数学公式

### 感知注意力 (`perception.py`)
```
O_t = softmax((E_t + V_t) · W_q (v · W_v)^T / √d_k) · (E_t + V_t + S_t^other)
```

### 贝叶斯信念更新 (`cognition.py`)
```
P(Z_t | O_{1:t}) ∝ P(O_t | Z_t) · Σ_{Z_{t-1}} P(Z_t | Z_{t-1}) · P(Z_{t-1} | O_{1:t-1})
```

### 期望效用决策 (`decision.py`)
```
U(a, Z) = w_v^T · F_value(a) + λ_norm · Comply(a, N) - η · Complexity(a)
```

### 环境更新 (`action.py`)
```
E_{t+1} = f_env(E_t, a_t) + ε,  ε ~ N(0, noise_std²)
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用

### 运行示例场景
```bash
python main.py --scenario colleague_late
```

### 运行交互模式
```bash
python main.py --interactive
```

### 运行全部验证
```bash
python main.py --run-validation
```

### "同事迟到"场景

```
输入: "同事会议迟到"
感知: E_t=[会议时间,当前时间,等待时长,...], V_t="colleague_late"编码
推理: 贝叶斯更新意图/社会关系/因果信念
决策: {等待,发提醒,重新安排,升级} 中选择
输出: 决策 + 完整推理链 + 验证结果
```

## 技术栈

- Python 3.10+
- PyTorch, Pyro (贝叶斯推理)
- NumPy, SciPy
- Pandas, Scikit-learn
- Hydra/OmegaConf (配置管理)
- pytest (测试)

## 验证模块

| 验证 | 描述 | 通过标准 |
|------|------|---------|
| 生态效度 | 行为分布与人类相似度 | JS距离 < 0.15 |
| 文化泛化 | 跨文化行为适应性 | Spearman \|ρ\| > 0.8 |
| 反事实推理 | "如果...会怎样"推理能力 | 准确率 > 85% |
| 结构一致性 | 隐藏状态干预效果 | 效应量在CI内 |
| 时间鲁棒性 | 跨时间一致性 | ICC > 0.9 |
| 对抗规范 | 伦理边界检测 | 拒绝率 > 95% |

## 示例输出

```
============================================================
SCENARIO: Colleague is Late for Meeting
============================================================

[1] Perceiving environment...
    Observation vector (dim=32): [0.12, 0.08, ...]

[2] Reasoning about situation...
    Hidden state (intent): [0.64, 0.56, 0.48]
    Hidden state (social): [0.72, 0.45, 0.68, ...]
    Predicted emotion: interest (valence=0.32, arousal=0.45)
    Motivation: [0.15, 0.22, 0.18, ...]...

[3] Making decision...
    Selected action: send_reminder (id=1)
    Utility: 0.723
    Utility breakdown: value=0.42, norm=0.28, complexity=0.05

[4] Executing action...
    Next environment state: [0.75, 0.28, 0.12, ...]
    Prediction error: 0.0342

[5] Reasoning Chain:
    Step 1: action_generation
             Generated 20 candidate actions based on current state
    Step 2: value_evaluation
             Evaluated each action's contribution to value dimensions
    ...

============================================================
FINAL DECISION
============================================================
Action: send_reminder
Reasoning: Action 1: send_reminder
Alternatives considered: wait, reschedule, ask_reason
```

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/bigdaddyzc/helga.git
cd helga

# 安装依赖
pip install -r requirements.txt

# 运行示例场景
python main.py --scenario colleague_late

# 运行交互模式
python main.py --interactive

# 运行全部验证
python main.py --run-validation
```

## 开发

```bash
# 运行测试
pytest tests/ -v

# 查看代码覆盖率
pytest tests/ --cov
```

## License

MIT