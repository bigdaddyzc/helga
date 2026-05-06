"""价值函数 - Value Functions

V(a) = Σ_k w_k · v_k(a)

基于Schwartz 10维价值观的动作价值计算
"""

from typing import Dict, List
import numpy as np

from .action_space import Action, ActionSpace


class ValueFunction:
    """价值函数

    基于Schwartz 10维价值观计算动作价值
    """

    # Schwartz 10维价值观名称
    DIMENSION_NAMES = [
        "self_transcendence",  # 0
        "openness",           # 1
        "benevolence",        # 2
        "conformity",         # 3
        "security",           # 4
        "achievement",        # 5
        "hedonism",           # 6
        "stimulation",        # 7
        "self_direction",     # 8
        "universalism"        # 9
    ]

    def __init__(self, weights: np.ndarray = None):
        """初始化价值函数

        Args:
            weights: 10维权重向量，如果为None则使用均匀权重
        """
        if weights is None:
            self.weights = np.ones(10) / 10  # 均匀权重
        else:
            self.weights = weights.astype(np.float32)
            assert len(self.weights) == 10, "权重向量必须是10维"

    def compute(self, action: Action) -> float:
        """计算单个动作的价值

        V(a) = Σ_k w_k · v_k(a)

        Args:
            action: 动作

        Returns:
            动作价值分数 (标量)
        """
        return np.dot(self.weights, action.value_vector)

    def compute_all(self, actions: List[Action]) -> List[float]:
        """计算所有动作的价值

        Args:
            actions: 动作列表

        Returns:
            每个动作的价值分数列表
        """
        return [self.compute(a) for a in actions]

    def compute_vector(self, action: Action) -> np.ndarray:
        """计算动作的价值向量

        Args:
            action: 动作

        Returns:
            加权后的价值向量 (10维)
        """
        return self.weights * action.value_vector

    def update_weights(self, new_weights: np.ndarray):
        """更新权重

        Args:
            new_weights: 新的10维权重向量
        """
        self.weights = new_weights.astype(np.float32)

    def get_dimension_contribution(self, action: Action) -> Dict[str, float]:
        """获取每个维度对动作的贡献

        Args:
            action: 动作

        Returns:
            维度名称 -> 贡献值 的字典
        """
        contributions = {}
        weighted = self.compute_vector(action)
        for i, dim_name in enumerate(self.DIMENSION_NAMES):
            contributions[dim_name] = float(weighted[i])
        return contributions

    def top_dimensions(self, action: Action, k: int = 3) -> List[str]:
        """获取对动作贡献最大的k个维度

        Args:
            action: 动作
            k: 返回前k个

        Returns:
            维度名称列表
        """
        contributions = self.get_dimension_contribution(action)
        sorted_dims = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        return [dim for dim, _ in sorted_dims[:k]]


class AdaptiveValueFunction(ValueFunction):
    """自适应价值函数

    支持根据上下文调整权重
    """

    def __init__(self, base_weights: np.ndarray = None, context_adjustments: Dict = None):
        """初始化

        Args:
            base_weights: 基础权重
            context_adjustments: 上下文调整映射
        """
        super().__init__(base_weights)
        self.context_adjustments = context_adjustments or {}

    def compute_with_context(self, action: Action, context: Dict) -> float:
        """根据上下文计算动作价值

        Args:
            action: 动作
            context: 上下文信息

        Returns:
            调整后的动作价值
        """
        base_value = self.compute(action)

        # 获取上下文调整
        adjustment = 1.0
        if context:
            for key, adj in self.context_adjustments.items():
                if key in context:
                    adjustment *= adj

        return base_value * adjustment

    def add_context_adjustment(self, context_key: str, adjustment_factor: float):
        """添加上下文调整

        Args:
            context_key: 上下文键
            adjustment_factor: 调整因子
        """
        self.context_adjustments[context_key] = adjustment_factor


def create_value_function(weights: np.ndarray = None) -> ValueFunction:
    """工厂函数：创建价值函数"""
    return ValueFunction(weights)


def create_adaptive_value_function(base_weights: np.ndarray = None,
                                   context_adjustments: Dict = None) -> AdaptiveValueFunction:
    """工厂函数：创建自适应价值函数"""
    return AdaptiveValueFunction(base_weights, context_adjustments)