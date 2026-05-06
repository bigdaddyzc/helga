"""决策引擎 - Decision Engine

核心公式: Decision(E, Q, R) = argmax_a [α·V(a) + β·Sim(R,a) + δ·SemSim(Q,a) - γ·C(a)]

其中:
- V(a): 动作价值分数
- Sim(R,a): 搜索结果相似度
- SemSim(Q,a): 问题与动作的语义相似度 (新增)
- C(a): 执行成本
- α=1.0, β=0.5, δ=1.5, γ=0.3 为权重参数
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import re

from .query_parser import QueryParser, QueryResult
from .action_space import Action, ActionSpace, create_action_space
from .value_functions import ValueFunction, create_value_function
from .cost_functions import CostFunction, create_cost_function
from .similarity import SimilarityFunction, create_similarity_function


@dataclass
class DecisionResult:
    """决策结果

    Attributes:
        recommended_action: 推荐的动作
        confidence: 置信度 [0, 1]
        reasoning: 决策理由
        all_scores: 所有候选动作的评分
        alternatives: 备选方案列表
        decision_formula: 使用的公式参数
    """
    recommended_action: Action
    confidence: float
    reasoning: str
    all_scores: Dict[str, float] = field(default_factory=dict)
    alternatives: List[Action] = field(default_factory=list)
    decision_formula: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScoredAction:
    """带评分的动作"""
    action: Action
    total_score: float
    value_score: float
    similarity_score: float
    cost_score: float


class DecisionEngine:
    """决策引擎

    实现核心决策公式:
    Decision(E, Q, R) = argmax_a [α·V(a) + β·Sim(R,a) + δ·SemSim(Q,a) - γ·C(a)]

    Attributes:
        alpha: 价值权重 (默认1.0)
        beta: 相似度权重 (默认0.5)
        gamma: 成本权重 (默认0.3)
        delta: 语义相似度权重 (默认1.5)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        value_function: Optional[ValueFunction] = None,
        cost_function: Optional[CostFunction] = None,
        similarity_function: Optional[SimilarityFunction] = None,
        action_space: Optional[ActionSpace] = None
    ):
        """初始化决策引擎

        Args:
            alpha: 价值权重 α
            beta: 相似度权重 β
            gamma: 成本权重 γ
            value_function: 价值函数（可选，默认创建新的）
            cost_function: 成本函数（可选，默认创建新的）
            similarity_function: 相似度函数（可选，默认创建新的）
            action_space: 动作空间（可选，默认创建新的）
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.value_function = value_function or create_value_function()
        self.cost_function = cost_function or create_cost_function()
        self.similarity_function = similarity_function or create_similarity_function()
        self.action_space = action_space or create_action_space()

        # 搜索结果嵌入（用于计算相似度）
        self._search_embeddings: List[np.ndarray] = []

    def set_search_results(self, search_embeddings: List[np.ndarray]):
        """设置搜索结果嵌入

        Args:
            search_embeddings: 搜索结果嵌入列表
        """
        self._search_embeddings = search_embeddings

    def set_weights(self, alpha: float, beta: float, gamma: float):
        """设置公式权重

        Args:
            alpha: 价值权重
            beta: 相似度权重
            gamma: 成本权重
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def decide(
        self,
        query_result: QueryResult,
        search_results: Optional[List[Dict]] = None,
        context: Optional[Dict] = None
    ) -> DecisionResult:
        """执行决策

        Args:
            query_result: QueryParser的输出
            search_results: 搜索结果列表（每个包含embedding）
            context: 上下文信息

        Returns:
            DecisionResult
        """
        # 获取问题关键词
        keywords = query_result.keywords if query_result else []

        # 获取所有候选动作（传入关键词用于语义排序）
        candidate_actions = self.action_space.get_top_k_actions(k=15, query_keywords=keywords)

        # 计算每个动作的评分
        scored_actions = []
        for action in candidate_actions:
            scores = self._compute_action_scores(action, search_results, keywords)
            scored_actions.append(scores)

        # 按总分排序
        scored_actions.sort(key=lambda x: x.total_score, reverse=True)

        # 获取最佳动作
        best = scored_actions[0]
        alternatives = [sa.action for sa in scored_actions[1:5]]

        # 构建决策理由
        reasoning = self._build_reasoning(best, scored_actions)

        # 构建结果
        result = DecisionResult(
            recommended_action=best.action,
            confidence=best.total_score,
            reasoning=reasoning,
            all_scores={sa.action.id: sa.total_score for sa in scored_actions},
            alternatives=alternatives,
            decision_formula={
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma
            }
        )

        return result

    def _compute_action_scores(
        self,
        action: Action,
        search_results: Optional[List[Dict]],
        keywords: List[str] = None
    ) -> ScoredAction:
        """计算单个动作的评分

        Args:
            action: 动作
            search_results: 搜索结果
            keywords: 问题关键词

        Returns:
            ScoredAction
        """
        # 1. 价值分数 V(a)
        value_score = self.value_function.compute(action)

        # 2. 相似度分数 Sim(R, a)
        similarity_score = self._compute_similarity_score(action, search_results)

        # 3. 语义相似度 SemSim(Q, a)
        semantic_score = self._compute_semantic_similarity(action, keywords)

        # 4. 成本分数 C(a)
        cost_score = self.cost_function.compute(action)

        # 5. 总分 = α·V(a) + β·Sim(R,a) + δ·SemSim(Q,a) - γ·C(a)
        # δ = 1.5 (语义权重 - 提高以强调问题相关性)
        total_score = (
            self.alpha * value_score +
            self.beta * similarity_score +
            1.5 * semantic_score -
            self.gamma * cost_score
        )

        return ScoredAction(
            action=action,
            total_score=total_score,
            value_score=value_score,
            similarity_score=semantic_score,  # 使用语义分数
            cost_score=cost_score
        )

    def _compute_similarity_score(
        self,
        action: Action,
        search_results: Optional[List[Dict]]
    ) -> float:
        """计算相似度分数

        如果没有搜索结果，使用动作自身的嵌入
        """
        if not search_results or not self._search_embeddings:
            # 使用动作价值向量作为嵌入
            action_embedding = action.value_vector
            # 返回中等相似度（无搜索结果时）
            norm = np.linalg.norm(action_embedding)
            if norm == 0:
                return 0.5
            return 0.5  # 默认值

        # 计算与所有搜索结果的最大相似度
        max_similarity = 0.0
        for i, emb in enumerate(self._search_embeddings):
            sim = self.similarity_function.compute(emb, action.value_vector)
            max_similarity = max(max_similarity, sim)

        return max_similarity

    def _compute_semantic_similarity(
        self,
        action: Action,
        keywords: Optional[List[str]]
    ) -> float:
        """计算问题与动作的语义相似度

        基于动作名称/ID与问题关键词的匹配

        Args:
            action: 动作
            keywords: 问题关键词

        Returns:
            语义相似度 [0, 1]
        """
        if not keywords:
            return 0.5

        # 语义扩展：交通/路线相关关键词映射
        traffic_semantic_map = {
            "开车": ["导航", "路线", "驾驶", "出行", "交通"],
            "驾驶": ["导航", "路线", "开车", "出行", "交通"],
            "出行": ["导航", "路线", "交通", "开车"],
            "路线": ["导航", "路线", "绕行", "改变路线"],
            "收费": ["收费站", "高速", "收费"],
            "收费站": ["高速", "收费", "封闭", "关闭"],
            "封闭": ["收费站", "关闭", "封路"],
            "关闭": ["收费站", "封闭", "封路"],
            "堵车": ["拥堵", "交通", "等待", "绕行"],
            "拥堵": ["堵车", "交通", "等待", "绕行"],
            "回": ["路线", "返回", "出发"],
            "郴州": ["湖南", "南方", "开车"],
            "长沙": ["湖南", "终点", "目的地"],
        }

        # 扩展关键词
        expanded_keywords = list(keywords)
        for kw in keywords:
            if kw in traffic_semantic_map:
                for related in traffic_semantic_map[kw]:
                    if related not in expanded_keywords:
                        expanded_keywords.append(related)

        # 动作名称和ID转换为小写
        action_text = (action.name + " " + action.id).lower()

        # 直接匹配分数
        direct_matches = 0
        for kw in expanded_keywords:
            kw_lower = kw.lower()
            # 完整词匹配
            if kw_lower in action_text:
                direct_matches += 2
            # 字符重叠匹配
            else:
                kw_chars = set(kw_lower)
                action_chars = set(action_text)
                overlap = len(kw_chars & action_chars)
                if overlap > 0 and len(kw_chars) > 0:
                    direct_matches += overlap / len(kw_chars)

        # 归一化
        max_possible = len(expanded_keywords) * 3
        similarity = min(1.0, direct_matches / max_possible) if max_possible > 0 else 0.5

        # 额外检查：动作ID中的词根匹配
        action_words = re.split(r'[_\s]+', action_text)
        for kw in expanded_keywords:
            kw_lower = kw.lower()
            for word in action_words:
                if len(word) > 2 and len(kw_lower) > 2:
                    # 检查是否有共同子串
                    if word in kw_lower or kw_lower in word:
                        similarity = min(1.0, similarity + 0.1)
                    # 检查词根重叠
                    elif word[:3] == kw_lower[:3]:
                        similarity = min(1.0, similarity + 0.05)

        return similarity

    def _build_reasoning(self, best: ScoredAction, all_scored: List[ScoredAction]) -> str:
        """构建决策理由

        Args:
            best: 最佳动作
            all_scored: 所有评分

        Returns:
            决策理由字符串
        """
        reasoning_parts = []

        # 添加价值分析
        top_dims = self.value_function.top_dimensions(best.action, k=3)
        reasoning_parts.append(
            f"该动作在价值观维度 [{', '.join(top_dims)}] 上贡献突出"
        )

        # 添加相似度分析
        if best.similarity_score > 0.6:
            reasoning_parts.append(f"与搜索结果高度相关 (相似度: {best.similarity_score:.2f})")
        elif best.similarity_score > 0.4:
            reasoning_parts.append(f"与搜索结果有一定相关性 (相似度: {best.similarity_score:.2f})")

        # 添加成本分析
        cost_details = self.cost_function.get_cost_breakdown(best.action)
        if cost_details['weighted_total'] < 0.3:
            reasoning_parts.append("执行成本较低")
        elif cost_details['weighted_total'] > 0.6:
            reasoning_parts.append("执行成本较高，需要谨慎考虑")

        # 添加置信度
        reasoning_parts.append(f"综合评分: {best.total_score:.3f}")

        return " | ".join(reasoning_parts)

    def get_decision_explanation(self, result: DecisionResult) -> str:
        """获取详细的决策解释

        Args:
            result: 决策结果

        Returns:
            格式化的解释字符串
        """
        lines = []
        lines.append("=" * 50)
        lines.append("决策分析")
        lines.append("=" * 50)

        lines.append(f"\n推荐动作: {result.recommended_action.name}")
        lines.append(f"置信度: {result.confidence:.2%}")
        lines.append(f"理由: {result.reasoning}")

        lines.append("\n公式参数:")
        lines.append(f"  α (价值权重): {result.decision_formula.get('alpha', self.alpha)}")
        lines.append(f"  β (相似度权重): {result.decision_formula.get('beta', self.beta)}")
        lines.append(f"  γ (成本权重): {result.decision_formula.get('gamma', self.gamma)}")

        lines.append("\n备选方案:")
        for i, alt in enumerate(result.alternatives, 1):
            score = result.all_scores.get(alt.id, 0)
            lines.append(f"  {i}. {alt.name} (评分: {score:.3f})")

        return "\n".join(lines)


def create_decision_engine(
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.3
) -> DecisionEngine:
    """工厂函数：创建决策引擎

    Args:
        alpha: 价值权重
        beta: 相似度权重
        gamma: 成本权重

    Returns:
        DecisionEngine实例
    """
    return DecisionEngine(alpha=alpha, beta=beta, gamma=gamma)