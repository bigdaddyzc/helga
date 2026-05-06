"""强化学习优化器 - RL Optimizer

根据用户反馈更新公式权重，实现策略梯度更新

公式:
    θ_new = θ_old + η · ∇log π(a|S) · R_user

其中:
    π(a|S) = Softmax(Decision(E, Q, R))
    R_user = w1 · usefulness + w2 · speed + w3 · satisfaction
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


@dataclass
class UserFeedback:
    """用户反馈

    Attributes:
        session_id: 会话ID
        decision_id: 决策ID
        score: 评分 (1-5)
        rationale: 用户理由
        accepted: 是否采纳
        timestamp: 时间戳
    """
    session_id: str
    decision_id: str
    score: int
    rationale: str
    accepted: bool = True
    timestamp: float = 0.0


@dataclass
class RLPolicy:
    """RL策略状态

    包含所有可学习的权重参数
    """
    # 公式权重
    alpha: float = 1.0   # 价值权重
    beta: float = 0.5    # 相似度权重
    gamma: float = 0.3   # 成本权重

    # 价值函数权重 (10维)
    value_weights: np.ndarray = None

    # 成本函数权重 (3维)
    cost_weights: np.ndarray = None

    # 相似度函数权重 (3维)
    similarity_weights: np.ndarray = None

    def __post_init__(self):
        if self.value_weights is None:
            self.value_weights = np.ones(10) / 10
        if self.cost_weights is None:
            self.cost_weights = np.array([0.4, 0.2, 0.4])
        if self.similarity_weights is None:
            self.similarity_weights = np.array([0.5, 0.3, 0.2])

    def to_vector(self) -> np.ndarray:
        """转换为单个向量"""
        return np.concatenate([
            [self.alpha, self.beta, self.gamma],
            self.value_weights,
            self.cost_weights,
            self.similarity_weights
        ])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "RLPolicy":
        """从向量创建"""
        policy = cls()
        policy.alpha = float(vec[0])
        policy.beta = float(vec[1])
        policy.gamma = float(vec[2])
        policy.value_weights = vec[3:13]
        policy.cost_weights = vec[13:16]
        policy.similarity_weights = vec[16:19]
        return policy

    def clone(self) -> "RLPolicy":
        """克隆策略"""
        new_policy = RLPolicy()
        new_policy.alpha = self.alpha
        new_policy.beta = self.beta
        new_policy.gamma = self.gamma
        new_policy.value_weights = self.value_weights.copy()
        new_policy.cost_weights = self.cost_weights.copy()
        new_policy.similarity_weights = self.similarity_weights.copy()
        return new_policy


class RLOptimizer:
    """强化学习优化器

    实现策略梯度更新，根据用户反馈调整决策参数
    """

    # 学习率
    LEARNING_RATE = 0.01

    # 权重裁剪范围
    CLIP_MIN = 0.1
    CLIP_MAX = 2.0

    def __init__(self, initial_policy: Optional[RLPolicy] = None):
        """初始化RL优化器

        Args:
            initial_policy: 初始策略，如果为None则使用默认策略
        """
        self.policy = initial_policy or RLPolicy()
        self.history: List[Dict] = []  # 学习历史
        self.episode_count = 0

    def compute_reward(self, feedback: UserFeedback, context: Optional[Dict] = None) -> float:
        """计算奖励

        R_user = w1 · usefulness + w2 · speed + w3 · satisfaction

        Args:
            feedback: 用户反馈
            context: 上下文信息

        Returns:
            奖励分数 [0, 1]
        """
        # 基础分数来自用户评分
        base_score = feedback.score / 5.0

        # 有用性：如果用户采纳，奖励更高
        usefulness = base_score if feedback.accepted else base_score * 0.7

        # 响应速度（从上下文推断，简化处理）
        speed = 0.8  # 默认

        # 满意度
        satisfaction = base_score

        # 综合奖励
        reward = 0.5 * usefulness + 0.2 * speed + 0.3 * satisfaction

        # 一致性惩罚：如果用户理由与决策不符
        # 这里简化处理，实际应该解析理由文本
        if feedback.rationale:
            # 如果理由较短或包含负面词汇，降低奖励
            if len(feedback.rationale) < 10 or any(
                word in feedback.rationale.lower()
                for word in ["不对", "错误", "不好", "没道理"]
            ):
                reward *= 0.8

        return reward

    def update(self, feedback: UserFeedback, context: Optional[Dict] = None) -> Tuple[RLPolicy, float]:
        """更新策略

        使用策略梯度方法:
        θ_new = θ_old + η · ∇log π(a|S) · R

        Args:
            feedback: 用户反馈
            context: 上下文信息

        Returns:
            (new_policy, reward) 元组
        """
        # 计算奖励
        reward = self.compute_reward(feedback, context)

        # 克隆当前策略
        new_policy = self.policy.clone()

        # 计算梯度并更新
        # 简化处理：使用有限差分近似梯度

        # 1. 更新alpha (价值权重)
        delta_alpha = self._estimate_gradient("alpha", reward)
        new_policy.alpha = self._clip_value(new_policy.alpha + self.LEARNING_RATE * delta_alpha)

        # 2. 更新beta (相似度权重)
        delta_beta = self._estimate_gradient("beta", reward)
        new_policy.beta = self._clip_value(new_policy.beta + self.LEARNING_RATE * delta_beta)

        # 3. 更新gamma (成本权重)
        delta_gamma = self._estimate_gradient("gamma", reward)
        new_policy.gamma = self._clip_value(new_policy.gamma + self.LEARNING_RATE * delta_gamma)

        # 4. 更新价值函数权重
        for i in range(10):
            delta = self._estimate_value_weight_gradient(i, reward)
            new_policy.value_weights[i] = self._clip_value(
                new_policy.value_weights[i] + self.LEARNING_RATE * delta
            )

        # 5. 更新成本函数权重
        for i in range(3):
            delta = self._estimate_cost_weight_gradient(i, reward)
            new_policy.cost_weights[i] = self._clip_value(
                new_policy.cost_weights[i] + self.LEARNING_RATE * delta
            )

        # 更新策略
        self.policy = new_policy
        self.episode_count += 1

        # 记录历史
        self.history.append({
            "episode": self.episode_count,
            "reward": reward,
            "feedback": feedback,
            "policy_snapshot": self.policy.to_vector().copy()
        })

        return self.policy, reward

    def _estimate_gradient(self, param_name: str, reward: float) -> float:
        """估计单个参数的梯度

        Args:
            param_name: 参数名称
            reward: 奖励

        Returns:
            梯度估计值
        """
        # 简化的梯度估计：使用随机扰动
        # 实际应用中应该使用更复杂的方法

        # 当前参数值
        current_value = getattr(self.policy, param_name)

        # 随机扰动
        perturbation = np.random.randn() * 0.1

        # 估计梯度（简化）
        gradient = reward * perturbation

        return gradient

    def _estimate_value_weight_gradient(self, index: int, reward: float) -> float:
        """估计价值权重的梯度

        Args:
            index: 权重索引 (0-9)
            reward: 奖励

        Returns:
            梯度估计值
        """
        perturbation = np.random.randn() * 0.1
        return reward * perturbation

    def _estimate_cost_weight_gradient(self, index: int, reward: float) -> float:
        """估计成本权重的梯度

        Args:
            index: 权重索引 (0-2)
            reward: 奖励

        Returns:
            梯度估计值
        """
        perturbation = np.random.randn() * 0.1
        return reward * perturbation

    def _clip_value(self, value: float) -> float:
        """裁剪权重值

        Args:
            value: 输入值

        Returns:
            裁剪后的值
        """
        return max(self.CLIP_MIN, min(self.CLIP_MAX, value))

    def get_policy(self) -> RLPolicy:
        """获取当前策略"""
        return self.policy

    def set_policy(self, policy: RLPolicy):
        """设置策略"""
        self.policy = policy

    def get_learning_summary(self) -> Dict[str, Any]:
        """获取学习总结

        Returns:
            包含学习统计的字典
        """
        if not self.history:
            return {
                "episode_count": 0,
                "avg_reward": 0.0,
                "current_policy": self.policy.to_vector().tolist()
            }

        rewards = [h["reward"] for h in self.history]
        return {
            "episode_count": self.episode_count,
            "avg_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "recent_rewards": rewards[-10:],
            "current_policy": {
                "alpha": self.policy.alpha,
                "beta": self.policy.beta,
                "gamma": self.policy.gamma,
                "value_weights": self.policy.value_weights.tolist(),
                "cost_weights": self.policy.cost_weights.tolist()
            }
        }

    def reset(self):
        """重置优化器到初始状态"""
        self.policy = RLPolicy()
        self.history = []
        self.episode_count = 0


def create_rl_optimizer(initial_policy: Optional[RLPolicy] = None) -> RLOptimizer:
    """工厂函数：创建RL优化器"""
    return RLOptimizer(initial_policy)