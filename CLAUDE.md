# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

HELGA (Humanistic Environmental Learning and Generating Agent) — 一个计算模型，模拟人类"观测 → 思考 → 决策 → 行动"闭环。

主要组件：
- `core/` — 核心模块（感知、认知、决策、行动、文化）
- `validation/` — 六层验证系统
- `api/` — UDSS 决策引擎 API
- `udss/` — UDSS 系统核心
- `frontend/` — React + Vite 前端
- `rl/` — 强化学习更新

## 常用命令

### Python 后端
```bash
# 运行场景
python main.py --scenario colleague_late

# 交互模式
python main.py --interactive

# 运行全部验证
python main.py --run-validation

# 运行测试
pytest
pytest tests/test_helga.py           # 单个测试文件
pytest tests/ -v                      # 详细输出

# UDSS API（需要同时启动前端）
python udss_api.py                    # 后端 API (http://localhost:5000)
cd frontend && npm run dev            # 前端 (http://localhost:5173)
```

### 前端
```bash
cd frontend
npm run dev              # 开发服务器
npm run build            # 生产构建
```

## 核心架构

### HELGA Agent (`core/`)

| 模块 | 文件 | 功能 |
|------|------|------|
| 感知 | `perception.py` | 多头注意力感知系统 O_t (dim=32) |
| 认知 | `cognition.py` | 贝叶斯心智推理（粒子滤波/SVI） |
| 决策 | `decision.py` | 期望效用决策 U(a,Z) |
| 行动 | `action.py` | 环境闭环 + 在线文化适应 |
| 文化 | `culture.py` | Schwartz价值观、规范矩阵、情感基模 |
| 类型 | `types.py` | 共享类型定义 |

### UDSS 系统 (`api/`)

| 模块 | 文件 | 功能 |
|------|------|------|
| 决策引擎 | `decision_engine.py` | 核心决策公式计算 |
| 动作空间 | `action_space.py` | 动作候选生成与评估 |
| 文本生成 | `text_generator.py` | 决策结果自然语言输出 |
| 网络搜索 | `web_search.py` | 相关信息检索 |

### 验证系统 (`validation/`)

六层验证（单一文件 `__init__.py`）：
- 生态效度（JS距离 < 0.15）
- 文化泛化（Spearman |ρ| > 0.8）
- 反事实推理（准确率 > 85%）
- 结构一致性（效应量在CI内）
- 时间鲁棒性（ICC > 0.9）
- 对抗规范检测（拒绝率 > 95%）

## 技术栈

- Python 3.10+
- PyTorch, Pyro (贝叶斯推理)
- NumPy, SciPy, Pandas, Scikit-learn
- Hydra/OmegaConf (配置管理)
- React 18 + Vite + Ant Design (前端)
- pytest (测试)

## 配置

- `config/default.yaml` — 默认配置
- `interfaces/cli.py` — CLI 入口