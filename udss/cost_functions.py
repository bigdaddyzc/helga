"""成本函数 - Cost Functions

C(a) = t_time · c_time + t_money · c_money + t_risk · c_risk
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from .action_space import Action


@dataclass
class CostVector:
    """成本向量

    Attributes:
        time_cost: 时间成本 [0, 1]
        money_cost: 金钱成本 [0, 1]
        risk_cost: 风险成本 [0, 1]
    """
    time_cost: float
    money_cost: float
    risk_cost: float

    def to_array(self) -> np.ndarray:
        return np.array([self.time_cost, self.money_cost, self.risk_cost], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "CostVector":
        return cls(
            time_cost=float(arr[0]),
            money_cost=float(arr[1]),
            risk_cost=float(arr[2])
        )


class CostFunction:
    """成本函数

    计算动作的执行成本
    """

    def __init__(self, weights: Optional[np.ndarray] = None):
        """初始化成本函数

        Args:
            weights: 3维权重向量 [w_time, w_money, w_risk]
        """
        if weights is None:
            # 默认权重：时间最重要，风险其次，金钱第三
            self.weights = np.array([0.4, 0.2, 0.4], dtype=np.float32)
        else:
            self.weights = weights.astype(np.float32)

    def compute(self, action: Action) -> float:
        """计算动作成本

        C(a) = Σ_i w_i · c_i

        Args:
            action: 动作

        Returns:
            成本分数 (越高成本越高)
        """
        cost_vec = CostVector.from_array(action.cost_vector)
        cost_array = cost_vec.to_array()
        return np.dot(self.weights, cost_array)

    def compute_detailed(self, action: Action) -> CostVector:
        """计算详细成本

        Args:
            action: 动作

        Returns:
            CostVector
        """
        return CostVector.from_array(action.cost_vector)

    def compute_all(self, actions) -> list:
        """计算所有动作的成本

        Args:
            actions: 动作列表

        Returns:
            成本列表
        """
        return [self.compute(a) for a in actions]

    def update_weights(self, time_weight: float, money_weight: float, risk_weight: float):
        """更新权重

        Args:
            time_weight: 时间成本权重
            money_weight: 金钱成本权重
            risk_weight: 风险成本权重
        """
        total = time_weight + money_weight + risk_weight
        self.weights = np.array([
            time_weight / total,
            money_weight / total,
            risk_weight / total
        ], dtype=np.float32)

    def get_cost_breakdown(self, action: Action) -> Dict[str, float]:
        """获取成本分解

        Args:
            action: 动作

        Returns:
            成本分解字典
        """
        cost = self.compute_detailed(action)
        return {
            "time_cost": cost.time_cost,
            "money_cost": cost.money_cost,
            "risk_cost": cost.risk_cost,
            "weighted_total": self.compute(action)
        }


class DynamicCostFunction(CostFunction):
    """动态成本函数

    根据上下文动态调整成本计算
    """

    def __init__(self, base_weights: np.ndarray = None, context_modifiers: Dict = None):
        """初始化

        Args:
            base_weights: 基础权重
            context_modifiers: 上下文修改器
        """
        super().__init__(base_weights)
        self.context_modifiers = context_modifiers or {}

    def compute_with_context(self, action: Action, context: Dict) -> float:
        """根据上下文计算成本

        Args:
            action: 动作
            context: 上下文

        Returns:
            调整后的成本
        """
        base_cost = self.compute(action)

        # 应用上下文修改器
        modifier = 1.0
        if context:
            for key, mod_fn in self.context_modifiers.items():
                if key in context:
                    modifier *= mod_fn(context[key])

        return base_cost * modifier

    def add_modifier(self, context_key: str, modifier_fn):
        """添加上下文修改器

        Args:
            context_key: 上下文键
            modifier_fn: 修改函数
        """
        self.context_modifiers[context_key] = modifier_fn


def create_cost_function(weights: np.ndarray = None) -> CostFunction:
    """工厂函数：创建成本函数"""
    return CostFunction(weights)


def create_dynamic_cost_function(base_weights: np.ndarray = None,
                                 context_modifiers: Dict = None) -> DynamicCostFunction:
    """工厂函数：创建动态成本函数"""
    return DynamicCostFunction(base_weights, context_modifiers)