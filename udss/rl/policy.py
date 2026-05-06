"""策略网络 - Policy Network

实现决策策略的神经网络表示
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class PolicyNetwork:
    """策略网络

    将决策过程表示为神经网络，用于强化学习
    """

    def __init__(
        self,
        state_dim: int = 90,  # E(40) + Q(20) + R(30)
        action_dim: int = 19,  # alpha, beta, gamma, value_weights(10), cost_weights(3), sim_weights(3)
        hidden_dim: int = 64
    ):
        """初始化策略网络

        Args:
            state_dim: 状态维度
            action_dim: 动作维度（要学习的参数数量）
            hidden_dim: 隐藏层维度
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 初始化网络权重（简化版：使用随机初始化）
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, action_dim) * 0.1
        self.b3 = np.zeros(action_dim)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """前向传播

        Args:
            state: 状态向量 (state_dim,)

        Returns:
            动作参数向量 (action_dim,)
        """
        # 层1
        h1 = np.tanh(np.dot(state, self.W1) + self.b1)

        # 层2
        h2 = np.tanh(np.dot(h1, self.W2) + self.b2)

        # 输出层（无激活，用于回归）
        output = np.dot(h2, self.W3) + self.b3

        return output

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """获取动作（参数更新）

        Args:
            state: 状态向量

        Returns:
            参数更新向量
        """
        return self.forward(state)

    def update_weights(self, gradients: Dict[str, np.ndarray], learning_rate: float = 0.01):
        """更新网络权重

        Args:
            gradients: 梯度字典
            learning_rate: 学习率
        """
        if "W1" in gradients:
            self.W1 -= learning_rate * gradients["W1"]
        if "b1" in gradients:
            self.b1 -= learning_rate * gradients["b1"]
        if "W2" in gradients:
            self.W2 -= learning_rate * gradients["W2"]
        if "b2" in gradients:
            self.b2 -= learning_rate * gradients["b2"]
        if "W3" in gradients:
            self.W3 -= learning_rate * gradients["W3"]
        if "b3" in gradients:
            self.b3 -= learning_rate * gradients["b3"]

    def compute_gradients(self, state: np.ndarray, target: np.ndarray) -> Dict[str, np.ndarray]:
        """计算梯度（简化版）

        Args:
            state: 状态向量
            target: 目标参数向量

        Returns:
            梯度字典
        """
        # 前向传播
        h1 = np.tanh(np.dot(state, self.W1) + self.b1)
        h2 = np.tanh(np.dot(h1, self.W2) + self.b2)
        output = np.dot(h2, self.W3) + self.b3

        # 计算误差
        error = output - target

        # 近似梯度（简化处理）
        gradients = {
            "W3": np.outer(h2, error),
            "b3": error,
            "W2": np.outer(h1, np.dot(self.W3, error) * (1 - h2**2)),
            "b2": np.dot(self.W3, error) * (1 - h2**2),
            "W1": np.outer(state, np.dot(self.W2, np.dot(self.W3, error) * (1 - h2**2)) * (1 - h1**2)),
            "b1": np.dot(self.W2, np.dot(self.W3, error) * (1 - h2**2)) * (1 - h1**2)
        }

        return gradients

    def clone(self) -> "PolicyNetwork":
        """克隆网络"""
        new_net = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        new_net.W1 = self.W1.copy()
        new_net.b1 = self.b1.copy()
        new_net.W2 = self.W2.copy()
        new_net.b2 = self.b2.copy()
        new_net.W3 = self.W3.copy()
        new_net.b3 = self.b3.copy()
        return new_net


class SoftmaxPolicy:
    """Softmax策略

    用于选择动作的概率策略
    """

    def __init__(self, temperature: float = 1.0):
        """初始化

        Args:
            temperature: 温度参数（越高越随机，越低越确定性）
        """
        self.temperature = temperature

    def compute_probabilities(self, scores: np.ndarray) -> np.ndarray:
        """计算动作概率

        π(a|S) = Softmax(score / T)

        Args:
            scores: 动作评分向量

        Returns:
            动作概率向量
        """
        # 归一化评分
        scores = scores / self.temperature

        # 减去最大值以提高数值稳定性
        scores = scores - np.max(scores)

        # Softmax
        exp_scores = np.exp(scores)
        probs = exp_scores / (np.sum(exp_scores) + 1e-8)

        return probs

    def sample_action(self, scores: np.ndarray) -> Tuple[int, float]:
        """采样动作

        Args:
            scores: 动作评分向量

        Returns:
            (动作索引, 动作概率)
        """
        probs = self.compute_probabilities(scores)
        action = np.random.choice(len(probs), p=probs)
        return action, probs[action]


def create_policy_network(
    state_dim: int = 90,
    action_dim: int = 19,
    hidden_dim: int = 64
) -> PolicyNetwork:
    """工厂函数：创建策略网络"""
    return PolicyNetwork(state_dim, action_dim, hidden_dim)


def create_softmax_policy(temperature: float = 1.0) -> SoftmaxPolicy:
    """工厂函数：创建Softmax策略"""
    return SoftmaxPolicy(temperature)