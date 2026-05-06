"""UDSS主系统 - Universal Decision Support System

整合所有模块，提供统一的决策支持接口

核心公式:
    Decision(E, Q, R) = argmax_a [α·V(a) + β·Sim(R,a) + δ·SemSim(Q,a) - γ·C(a)]

其中:
    - V(a): 动作价值分数
    - Sim(R,a): 搜索结果与动作的相似度
    - SemSim(Q,a): 问题Q与动作a的语义相似度
    - C(a): 执行成本
    - α, β, δ, γ 为权重参数

使用流程:
    1. 创建UDSS实例
    2. 调用decide()输入问题
    3. 获取决策方案
    4. (可选) 提供反馈让系统学习
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import uuid
import numpy as np

from .query_parser import QueryParser, QueryResult
from .decision_engine import DecisionEngine, DecisionResult
from .action_space import ActionSpace, Action
from .value_functions import ValueFunction, AdaptiveValueFunction
from .cost_functions import CostFunction, DynamicCostFunction
from .similarity import SimilarityFunction, WeightedSimilarityFunction
from .text_generator import TextGenerator, ActionPlan
from .web_search import WebSearch, SearchResult, SearchAggregator
from .rl.optimizer import RLOptimizer, RLPolicy, UserFeedback


@dataclass
class UDSSConfig:
    """UDSS配置"""
    # 公式权重
    alpha: float = 1.0      # 价值权重
    beta: float = 0.5      # 相似度权重
    gamma: float = 0.3     # 成本权重

    # 搜索配置
    max_search_results: int = 5
    search_api_key: Optional[str] = None

    # RL配置
    rl_enabled: bool = True
    learning_rate: float = 0.01

    # 相似度配置
    relevance_weight: float = 0.5
    freshness_weight: float = 0.3
    authority_weight: float = 0.2


@dataclass
class UDSSResult:
    """UDSS决策结果

    Attributes:
        session_id: 会话ID
        decision_id: 决策ID
        question: 用户问题
        decision_result: 决策引擎输出
        action_plan: 可执行方案
        search_results: 搜索结果
        plan_text: 文本格式的方案
        timestamp: 时间戳
    """
    session_id: str
    decision_id: str
    question: str
    decision_result: DecisionResult
    action_plan: ActionPlan
    search_results: List[SearchResult]
    plan_text: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class UDSS:
    """通用决策支持系统主类

    整合所有模块，提供端到端的决策支持
    """

    def __init__(self, config: Optional[UDSSConfig] = None):
        """初始化UDSS

        Args:
            config: UDSS配置，如果为None则使用默认配置
        """
        self.config = config or UDSSConfig()

        # 初始化核心组件
        self.query_parser = QueryParser()
        self.decision_engine = DecisionEngine(
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma
        )
        self.action_space = ActionSpace()
        self.value_function = ValueFunction()
        self.cost_function = CostFunction()
        self.similarity_function = WeightedSimilarityFunction(
            relevance_weight=self.config.relevance_weight,
            freshness_weight=self.config.freshness_weight,
            authority_weight=self.config.authority_weight
        )
        self.text_generator = TextGenerator()
        self.web_search = WebSearch(self.config.search_api_key)
        self.search_aggregator = SearchAggregator()

        # RL优化器
        self.rl_optimizer = RLOptimizer() if self.config.rl_enabled else None

        # 会话管理
        self.sessions: Dict[str, Dict] = {}

    def decide(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> UDSSResult:
        """执行决策

        核心流程:
        1. 解析问题 -> E, Q
        2. 搜索 -> R
        3. 公式计算 -> 最优动作
        4. 生成方案 -> 文本格式

        Args:
            question: 用户问题
            context: 可选上下文信息

        Returns:
            UDSSResult
        """
        # 生成会话ID和决策ID
        session_id = str(uuid.uuid4())
        decision_id = str(uuid.uuid4())

        # 1. 解析问题
        query_result = self.query_parser.parse(question, context)

        # 2. 执行搜索
        search_results = self.web_search.search(
            query=question,
            max_results=self.config.max_search_results
        )

        # 聚合搜索结果
        aggregated = self.search_aggregator.aggregate(search_results)

        # 设置搜索嵌入到决策引擎
        if aggregated["aggregated_embedding"] is not None:
            search_embeddings = [aggregated["aggregated_embedding"]]
        else:
            # 搜索失败时，使用问题文本生成伪嵌入
            search_embeddings = []
            # 基于问题关键词生成伪嵌入
            keywords = query_result.keywords if query_result else []
            if keywords:
                # 生成伪嵌入（问题关键词编码）
                pseudo_emb = np.zeros(10, dtype=np.float32)
                for i, kw in enumerate(keywords[:10]):
                    hash_val = sum(ord(c) for c in kw) % 10
                    pseudo_emb[hash_val] += 1.0
                # 归一化
                norm = np.linalg.norm(pseudo_emb)
                if norm > 0:
                    pseudo_emb = pseudo_emb / norm
                search_embeddings = [pseudo_emb]

        self.decision_engine.set_search_results(search_embeddings)

        # 3. 执行决策（传入keywords用于语义排序）
        decision_result = self.decision_engine.decide(
            query_result=query_result,
            search_results=search_results,
            context=context
        )

        # 4. 生成方案
        ctx = {"question": question}
        if context:
            ctx.update(context)

        # 如果是交通相关动作，获取路线信息
        recommended_action = decision_result.recommended_action
        is_traffic_action = recommended_action.id in [
            "check_navigation", "check_traffic_info", "find_detour",
            "change_route", "wait_traffic", "call_service_hotline"
        ]

        if is_traffic_action:
            # 搜索郴州到长沙的路线信息
            route_results = self.web_search.search_traffic_info(
                origin="郴州",
                destination="长沙",
                avoid="郴州北高速收费站"
            )
            routes = self.web_search.extract_route_info(route_results)
            ctx["routes"] = routes
            ctx["search_results"] = route_results

        action_plan = self.text_generator.generate(
            action=recommended_action,
            context=ctx,
            alternatives=decision_result.alternatives,
            reasoning=decision_result.reasoning
        )

        # 转换为Markdown文本
        plan_text = self.text_generator.to_markdown(action_plan)

        # 构建结果
        result = UDSSResult(
            session_id=session_id,
            decision_id=decision_id,
            question=question,
            decision_result=decision_result,
            action_plan=action_plan,
            search_results=search_results,
            plan_text=plan_text
        )

        # 记录会话
        self.sessions[session_id] = {
            "decision_id": decision_id,
            "question": question,
            "result": result,
            "feedback_received": False
        }

        return result

    def provide_feedback(
        self,
        session_id: str,
        score: int,
        rationale: str,
        accepted: bool = True
    ) -> Dict[str, Any]:
        """提供用户反馈

        Args:
            session_id: 会话ID
            score: 评分 (1-5)
            rationale: 用户理由
            accepted: 是否采纳

        Returns:
            更新结果信息
        """
        if session_id not in self.sessions:
            return {"error": "Session not found"}

        session = self.sessions[session_id]
        decision_id = session["decision_id"]

        # 创建反馈
        feedback = UserFeedback(
            session_id=session_id,
            decision_id=decision_id,
            score=score,
            rationale=rationale,
            accepted=accepted,
            timestamp=datetime.now().timestamp()
        )

        # 更新RL策略
        if self.rl_optimizer:
            new_policy, reward = self.rl_optimizer.update(feedback)

            # 应用新权重到决策引擎
            self.decision_engine.set_weights(
                alpha=new_policy.alpha,
                beta=new_policy.beta,
                gamma=new_policy.gamma
            )

            # 更新价值函数
            self.value_function.update_weights(new_policy.value_weights)

            session["feedback_received"] = True

            return {
                "success": True,
                "reward": reward,
                "policy_update": {
                    "alpha": new_policy.alpha,
                    "beta": new_policy.beta,
                    "gamma": new_policy.gamma
                }
            }

        return {"success": False, "error": "RL not enabled"}

    def get_learning_summary(self) -> Dict[str, Any]:
        """获取学习总结

        Returns:
            学习统计信息
        """
        if self.rl_optimizer:
            return self.rl_optimizer.get_learning_summary()
        return {"error": "RL not enabled"}

    def reset_learning(self):
        """重置学习状态"""
        if self.rl_optimizer:
            self.rl_optimizer.reset()

    def get_session(self, session_id: str) -> Optional[Dict]:
        """获取会话信息

        Args:
            session_id: 会话ID

        Returns:
            会话字典
        """
        return self.sessions.get(session_id)


def create_udss(config: Optional[UDSSConfig] = None) -> UDSS:
    """工厂函数：创建UDSS实例

    Args:
        config: UDSS配置

    Returns:
        UDSS实例
    """
    return UDSS(config)


# 示例用法
if __name__ == "__main__":
    # 创建UDSS
    udss = create_udss()

    # 执行决策
    result = udss.decide(
        question="我应该选择哪个offer？公司A给40万但要加班，公司B给30万但work-life balance"
    )

    print("=" * 60)
    print("决策结果")
    print("=" * 60)
    print(f"推荐动作: {result.decision_result.recommended_action.name}")
    print(f"置信度: {result.decision_result.confidence:.2%}")
    print(f"\n{result.plan_text}")