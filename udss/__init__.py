"""UDSS - Universal Decision Support System

核心公式: Decision(E, Q, R) = argmax_a [α·V(a) + β·Sim(R,a) + δ·SemSim(Q,a) - γ·C(a)]

其中:
- V(a): 动作价值分数 (value_function)
- Sim(R,a): 搜索结果相似度 (similarity_function)
- SemSim(Q,a): 问题与动作的语义相似度 (新增)
- C(a): 执行成本 (cost_function)
- α, β, δ, γ 为权重参数
"""

from .query_parser import QueryParser, QueryResult
from .decision_engine import DecisionEngine, DecisionResult, ScoredAction
from .action_space import ActionSpace, Action, create_action_space
from .value_functions import ValueFunction, AdaptiveValueFunction
from .cost_functions import CostFunction, DynamicCostFunction, CostVector
from .similarity import SimilarityFunction, WeightedSimilarityFunction
from .text_generator import TextGenerator, ActionPlan, PlanStep

# RL模块
from .rl.optimizer import RLOptimizer, RLPolicy, UserFeedback
from .rl.policy import PolicyNetwork, SoftmaxPolicy

# Search模块
from .web_search import WebSearch, SearchResult, SearchAggregator

# 系统主模块
from .system import UDSS, UDSSConfig, UDSSResult, create_udss

__all__ = [
    # 核心组件
    "QueryParser",
    "QueryResult",
    "DecisionEngine",
    "DecisionResult",
    "ScoredAction",
    "ActionSpace",
    "Action",
    "create_action_space",
    "ValueFunction",
    "AdaptiveValueFunction",
    "CostFunction",
    "DynamicCostFunction",
    "CostVector",
    "SimilarityFunction",
    "WeightedSimilarityFunction",
    "TextGenerator",
    "ActionPlan",
    "PlanStep",

    # RL模块
    "RLOptimizer",
    "RLPolicy",
    "UserFeedback",
    "PolicyNetwork",
    "SoftmaxPolicy",

    # Search模块
    "WebSearch",
    "SearchResult",
    "SearchAggregator",

    # 系统主模块
    "UDSS",
    "UDSSConfig",
    "UDSSResult",
    "create_udss",
]

__version__ = "1.0.0"