"""决策迭代过程模拟 - 完整复刻人类思考-决策过程

实现5个阶段的迭代决策，模拟人类在高速公路封闭场景下的思考过程：
1. INFORMATION_GATHERING - 网络搜索封闭信息
2. OPTIONS_GENERATION - 基于搜索结果生成候选方案
3. EVALUATION - 评估每个方案的优劣
4. CONTINGENCY_CHECK - 检查最优方案是否有突发情况（如拥堵）
5. FINAL_DECISION - 综合评估输出最终方案概率分布

每个阶段都会进行真实的网络搜索（如可用），并记录详细的推理过程。
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from .types import (
    DecisionPlan, DecisionIteration, PlanProbabilityDistribution,
    ReasoningChain, ReasoningStep, IterationStage, RouteDetails
)
from .web_search import WebSearchService, get_web_search_service, RouteOption
from . import decision


# 场景类型到方案的映射配置
SCENE_PLAN_CONFIGS = {
    "traffic": {
        "原地等待": {
            "estimated_duration": "未知（取决于封闭解除时间）",
            "target_location": None,
            "risk_factors": {"时间浪费": 0.6, "不确定性": 0.7},
            "overall_risk_level": "medium",
            "description": "在原地等待交通状况改善"
        },
        "绕行路线A": {
            "estimated_duration": "40分钟",
            "target_location": "备选高速",
            "risk_factors": {"路程增加": 0.3, "不熟悉路线": 0.4},
            "overall_risk_level": "low",
            "description": "选择替代路线A绕过拥堵路段"
        },
        "绕行路线B": {
            "estimated_duration": "55分钟",
            "target_location": "国道104",
            "risk_factors": {"路程增加": 0.5, "红绿灯多": 0.5},
            "overall_risk_level": "medium",
            "description": "选择替代路线B绕过拥堵路段"
        },
        "取消行程": {
            "estimated_duration": "0分钟",
            "target_location": None,
            "risk_factors": {"任务失败": 0.8},
            "overall_risk_level": "low",
            "description": "取消当前出行计划"
        }
    },
    "social": {
        "等待观察": {
            "estimated_duration": "待定",
            "target_location": None,
            "risk_factors": {"错过时机": 0.3},
            "overall_risk_level": "low",
            "description": "等待更多信息再做决定"
        },
        "主动帮助": {
            "estimated_duration": "30分钟",
            "target_location": None,
            "risk_factors": {"时间成本": 0.4},
            "overall_risk_level": "low",
            "description": "主动提供帮助或提醒"
        },
        "委托代理": {
            "estimated_duration": "10分钟",
            "target_location": None,
            "risk_factors": {"质量不可控": 0.3},
            "overall_risk_level": "medium",
            "description": "委托他人处理"
        },
        "忽略应对": {
            "estimated_duration": "0分钟",
            "target_location": None,
            "risk_factors": {"关系损失": 0.6},
            "overall_risk_level": "medium",
            "description": "忽略该问题继续其他事情"
        }
    },
    "work": {
        "等待确认": {
            "estimated_duration": "待定",
            "target_location": None,
            "risk_factors": {"进度延误": 0.4},
            "overall_risk_level": "low",
            "description": "等待进一步确认再做决定"
        },
        "批准执行": {
            "estimated_duration": "5分钟",
            "target_location": None,
            "risk_factors": {"执行风险": 0.3},
            "overall_risk_level": "low",
            "description": "批准提案并执行"
        },
        "拒绝否决": {
            "estimated_duration": "5分钟",
            "target_location": None,
            "risk_factors": {"关系影响": 0.5},
            "overall_risk_level": "medium",
            "description": "拒绝或否决该提案"
        },
        "修改调整": {
            "estimated_duration": "1小时",
            "target_location": None,
            "risk_factors": {"迭代成本": 0.5},
            "overall_risk_level": "medium",
            "description": "修改提案后重新提交"
        }
    },
    "emergency": {
        "紧急等待": {
            "estimated_duration": "未知",
            "target_location": None,
            "risk_factors": {"延误救援": 0.8},
            "overall_risk_level": "high",
            "description": "原地等待紧急救援"
        },
        "立即撤离": {
            "estimated_duration": "立即",
            "target_location": "安全区域",
            "risk_factors": {"任务中断": 0.7},
            "overall_risk_level": "medium",
            "description": "立即撤离危险区域"
        },
        "寻求帮助": {
            "estimated_duration": "10分钟",
            "target_location": None,
            "risk_factors": {"沟通成本": 0.3},
            "overall_risk_level": "low",
            "description": "拨打紧急电话或寻求帮助"
        }
    },
    "default": {
        "等待观察": {
            "estimated_duration": "待定",
            "target_location": None,
            "risk_factors": {"不确定性": 0.4},
            "overall_risk_level": "low",
            "description": "等待情况明朗再做决定"
        },
        "主动行动": {
            "estimated_duration": "30分钟",
            "target_location": None,
            "risk_factors": {"执行风险": 0.4},
            "overall_risk_level": "medium",
            "description": "主动采取行动"
        },
        "暂不处理": {
            "estimated_duration": "0分钟",
            "target_location": None,
            "risk_factors": {"潜在风险": 0.5},
            "overall_risk_level": "medium",
            "description": "暂时不处理该事务"
        }
    }
}


@dataclass
class BeliefState:
    """信念状态 - 记录决策过程中的信息更新"""
    beliefs: Dict[str, float] = field(default_factory=dict)
    information_gathered: List[str] = field(default_factory=list)
    closure_info: Optional[Any] = field(default=None)
    route_options: List[RouteOption] = field(default_factory=list)
    traffic_status: Optional[Any] = field(default=None)

    def update(self, key: str, value: float, source: str):
        self.beliefs[key] = value
        self.information_gathered.append(f"{key}={value} (from {source})")

    def get(self, key: str, default: float = 0.0) -> float:
        return self.beliefs.get(key, default)


class DecisionIterationEngine:
    """决策迭代引擎 - 完整复刻人类思考-决策过程

    通过真实的网络搜索（如可用）收集信息，生成具体可执行的方案，
    并在推理链中展示完整的思考过程。
    """

    def __init__(self, context_type: str = "default", use_web_search: bool = True):
        self.context_type = context_type
        self.plan_configs = SCENE_PLAN_CONFIGS.get(
            context_type, SCENE_PLAN_CONFIGS["default"]
        )
        self.belief_state = BeliefState()
        self.iterations: List[DecisionIteration] = []
        self.web_search = get_web_search_service() if use_web_search else None
        self.scenario_text = ""

    def run(
        self,
        scenario_input: Dict[str, Any],
        atomic_probs: Any = None
    ) -> Tuple[List[DecisionPlan], List[float], List[DecisionIteration]]:
        """运行完整的迭代决策过程

        Args:
            scenario_input: 场景输入（包含场景描述、上下文等）
            atomic_probs: 原子行动概率（可选，用于方案排序）

        Returns:
            (plans, probabilities, iterations)
        """
        self.scenario_text = self._extract_scenario_text(scenario_input)
        origin, destination = self._extract_locations(scenario_input)

        # 迭代1: 信息收集
        iter1 = self._information_gathering(origin, destination)
        self.iterations.append(iter1)

        # 迭代2: 选项生成
        iter2 = self._options_generation(origin, destination)
        self.iterations.append(iter2)

        # 迭代3: 方案评估
        iter3 = self._evaluation()
        self.iterations.append(iter3)

        # 迭代4: 突发检查
        iter4 = self._contingency_check()
        self.iterations.append(iter4)

        # 迭代5: 最终决策
        iter5, plans, probs = self._final_decision()
        self.iterations.append(iter5)

        return plans, probs, self.iterations

    def _extract_scenario_text(self, scenario_input: Dict[str, Any]) -> str:
        """提取场景文本"""
        if isinstance(scenario_input, dict):
            return scenario_input.get("text", "") or \
                   scenario_input.get("scenario", "") or \
                   scenario_input.get("description", "")
        return str(scenario_input)

    def _extract_locations(self, scenario_input: Dict[str, Any]) -> Tuple[str, str]:
        """从场景中提取起点和终点"""
        text = self._extract_scenario_text(scenario_input)

        # 尝试从文本中提取地点
        origin = "起点"
        destination = "终点"

        # 常见模式匹配
        if "从" in text and "到" in text:
            import re
            match = re.search(r'从([^到]+)到([^到]+)', text)
            if match:
                origin, destination = match.groups()[:2]
        elif "北京" in text and "上海" in text:
            origin, destination = "北京", "上海"

        return origin, destination

    def _information_gathering(
        self,
        origin: str,
        destination: str
    ) -> DecisionIteration:
        """迭代1: 信息收集 - 搜索封闭信息"""
        search_query = f"{origin}{destination}高速公路封闭信息"
        search_result = ""
        beliefs_updated = {}

        # 尝试网络搜索
        if self.web_search and self.context_type == "traffic":
            try:
                # 搜索封闭信息
                closure_info = self.web_search.search_closure_info(
                    f"{origin}到{destination}",
                    self.scenario_text
                )
                self.belief_state.closure_info = closure_info

                search_result = f"封闭位置:{closure_info.location}, " \
                              f"原因:{closure_info.reason}, " \
                              f"预计解除时间:{closure_info.estimated_duration_hours}小时, " \
                              f"来源:{closure_info.source}"

                beliefs_updated = {
                    "closure_duration": closure_info.estimated_duration_hours,
                    "is_verified": 1.0 if closure_info.is_verified else 0.0,
                    "severity": 1.0 if closure_info.severity == "full" else 0.5
                }
                self.belief_state.update("closure_duration",
                                        closure_info.estimated_duration_hours, "web_search")
                self.belief_state.update("is_verified", 1.0, "web_search")

            except Exception as e:
                search_result = f"搜索失败: {str(e)}，使用场景推断"

        # 如果没有真实搜索结果，从场景推断
        if not search_result:
            search_result, beliefs_updated = self._infer_closure_info()

        return DecisionIteration(
            iteration_id=1,
            stage=IterationStage.INFORMATION_GATHERING.value,
            stage_description="通过网络搜索收集封闭信息",
            search_query=search_query,
            search_result=search_result,
            input_information={"scenario": self.scenario_text},
            reasoning_result=f"信息收集完成：{'已验证' if self.belief_state.get('is_verified') else '已推断'}",
            beliefs_updated=beliefs_updated,
            plans_considered=[],
            action_taken="search_closure_info",
            confidence_delta=0.2,
            timestamp=time.time()
        )

    def _infer_closure_info(self) -> Tuple[str, Dict[str, float]]:
        """从场景上下文推断封闭信息"""
        text = self.scenario_text.lower()
        beliefs_updated = {}

        # 从场景文本中提取信息
        if "2小时" in self.scenario_text:
            duration = 2.0
        elif "3小时" in self.scenario_text:
            duration = 3.0
        elif "30分钟" in self.scenario_text:
            duration = 0.5
        elif "1小时" in self.scenario_text:
            duration = 1.0
        else:
            duration = 1.0  # 默认1小时

        beliefs_updated = {
            "closure_duration": duration,
            "is_verified": 0.0,  # 推断结果
            "severity": 1.0 if duration > 1 else 0.5
        }

        self.belief_state.update("closure_duration", duration, "scenario_inference")
        self.belief_state.update("is_verified", 0.0, "scenario_inference")

        search_result = f"[推断] 封闭时长:{duration}小时, 原因:道路施工, 来源:场景描述"

        return search_result, beliefs_updated

    def _options_generation(
        self,
        origin: str,
        destination: str
    ) -> DecisionIteration:
        """迭代2: 选项生成 - 基于信息生成候选方案"""
        available_plans = []
        search_result = ""

        # 如果是交通场景，搜索替代路线
        if self.web_search and self.context_type == "traffic":
            try:
                routes = self.web_search.search_route_alternatives(
                    origin, destination,
                    self.belief_state.closure_info.location if self.belief_state.closure_info else ""
                )
                self.belief_state.route_options = routes

                if routes:
                    search_result = f"找到{len(routes)}条替代路线:\n"
                    for i, route in enumerate(routes):
                        search_result += f"  {i+1}. {route.route_name}: " \
                                       f"距离{route.distance_km}km, " \
                                       f"预计{route.estimated_time_minutes}分钟, " \
                                       f"拥堵概率{route.congestion_probability*100:.0f}%\n"
            except Exception as e:
                search_result = f"路线搜索失败: {str(e)}"

        # 如果没有真实路线结果，使用配置中的方案
        if not self.belief_state.route_options:
            for plan_name, config in self.plan_configs.items():
                if "绕行" in plan_name:
                    # 生成默认路线详情
                    route = RouteOption(
                        route_name=f"{plan_name}（{origin}→{destination}）",
                        route_type="高速",
                        distance_km=1200.0,
                        estimated_time_minutes=600,
                        toll_cost=0,
                        congestion_probability=0.3,
                        traffic_status="畅通",
                        waypoints=[origin, "收费口", destination]
                    )
                    self.belief_state.route_options.append(route)
                    available_plans.append(plan_name)
                elif "等待" in plan_name or "原地" in plan_name:
                    available_plans.append(plan_name)
                elif "取消" in plan_name:
                    available_plans.append(plan_name)

        if not available_plans:
            available_plans = list(self.plan_configs.keys())

        return DecisionIteration(
            iteration_id=2,
            stage=IterationStage.OPTIONS_GENERATION.value,
            stage_description="基于收集的信息生成候选方案",
            search_query=f"搜索{origin}到{destination}的替代路线",
            search_result=search_result or "使用默认方案配置",
            input_information={
                "closure_duration": self.belief_state.get("closure_duration"),
                "origin": origin,
                "destination": destination
            },
            reasoning_result=f"生成{len(available_plans)}个候选方案: {', '.join(available_plans)}",
            beliefs_updated={},
            plans_considered=available_plans,
            action_taken="generate_options",
            confidence_delta=0.1,
            timestamp=time.time()
        )

    def _evaluation(self) -> DecisionIteration:
        """迭代3: 方案评估 - 评估每个方案的收益与风险"""
        closure_duration = self.belief_state.get("closure_duration", 1.0)

        # 基于时限调整方案评分
        evaluated = []
        for plan_name, config in self.plan_configs.items():
            risk_factors = config.get("risk_factors", {})
            avg_risk = sum(risk_factors.values()) / max(len(risk_factors), 1)

            # 时限影响评分
            if "等待" in plan_name:
                if closure_duration > 2:  # 时限长，等待成本高
                    avg_risk = min(1.0, avg_risk + 0.3)
            elif "绕行" in plan_name:
                if closure_duration <= 0.5:  # 时限短，绕行成本相对高
                    avg_risk = min(1.0, avg_risk + 0.2)

            score = 1.0 - avg_risk
            evaluated.append((plan_name, score, config))

        # 按评估分数排序
        evaluated.sort(key=lambda x: x[1], reverse=True)

        reasoning_result = "方案评估结果:\n"
        for plan_name, score, config in evaluated:
            reasoning_result += f"  - {plan_name}: 评分{score:.2f} (风险:{config.get('overall_risk_level')}, 时长:{config.get('estimated_duration')})\n"

        return DecisionIteration(
            iteration_id=3,
            stage=IterationStage.EVALUATION.value,
            stage_description="评估每个方案的收益与风险",
            search_query="",
            search_result="",
            input_information={"evaluated_count": len(evaluated)},
            reasoning_result=reasoning_result,
            beliefs_updated={},
            plans_considered=[p[0] for p in evaluated],
            action_taken="evaluate_options",
            confidence_delta=0.15,
            timestamp=time.time()
        )

    def _contingency_check(self) -> DecisionIteration:
        """迭代4: 突发检查 - 检查最优方案的潜在风险"""
        adjustments = {}
        reasoning_result = ""

        # 检查是否有绕行方案
        has_detour = any("绕行" in p.route_name for p in self.belief_state.route_options)

        if has_detour and self.web_search:
            try:
                # 对每个路线检查实时路况
                for route in self.belief_state.route_options:
                    status = self.web_search.check_traffic_status(route.route_name)
                    if status.is_congested:
                        adjustments[route.route_name] = {
                            "congestion": True,
                            "delay": status.estimated_delay_minutes,
                            "new_prob": route.congestion_probability * 1.3  # 增加拥堵概率
                        }

                if adjustments:
                    reasoning_result = f"检测到{len(adjustments)}条路线拥堵，调整方案概率\n"
                    for route_name, adj in adjustments.items():
                        reasoning_result += f"  - {route_name}: 拥堵，预计延误{adj['delay']}分钟\n"
                else:
                    reasoning_result = "主要路线暂无拥堵，方案概率保持不变"

            except Exception as e:
                reasoning_result = f"路况查询失败: {str(e)}"
        else:
            reasoning_result = "无需进行突发检查（非交通场景或无可用路线）"

        # 检查场景文本中的拥堵关键词
        text_lower = self.scenario_text.lower()
        if "堵" in text_lower or "拥堵" in text_lower:
            for plan_name in self.plan_configs.keys():
                if "绕行" in plan_name:
                    if plan_name not in adjustments:
                        adjustments[plan_name] = {"congestion": True, "delay": 30, "new_prob": 0.8}
            reasoning_result += "\n[场景推断] 检测到拥堵关键词，调整绕行方案概率"

        return DecisionIteration(
            iteration_id=4,
            stage=IterationStage.CONTINGENCY_CHECK.value,
            stage_description="检查最优方案的潜在风险和突发情况",
            search_query="查询实时路况" if has_detour else "",
            search_result=reasoning_result,
            input_information={"adjustments": adjustments},
            reasoning_result=reasoning_result or "无突发情况",
            beliefs_updated={},
            plans_considered=list(adjustments.keys()) if adjustments else [],
            action_taken="check_contingency",
            confidence_delta=-0.05 if adjustments else 0.0,
            timestamp=time.time()
        )

    def _final_decision(
        self
    ) -> Tuple[DecisionIteration, List[DecisionPlan], List[float]]:
        """迭代5: 最终决策 - 计算方案概率分布"""
        closure_duration = self.belief_state.get("closure_duration", 1.0)
        plans = []
        raw_scores = []

        for plan_name, config in self.plan_configs.items():
            plan_id = f"plan_{plan_name}"

            # 计算原始分数
            risk_factors = config.get("risk_factors", {})
            avg_risk = sum(risk_factors.values()) / max(len(risk_factors), 1)
            risk_score = 1.0 - avg_risk

            # 时长评分
            duration_str = config.get("estimated_duration", "0分钟")
            if "未知" in duration_str or "待定" in duration_str:
                duration_score = 0.3 if closure_duration > 2 else 0.6
            elif "分钟" in duration_str:
                try:
                    minutes = float(duration_str.replace("分钟", ""))
                    duration_score = max(0, 1.0 - minutes / 120)
                except:
                    duration_score = 0.5
            elif "小时" in duration_str:
                try:
                    hours = float(duration_str.replace("小时", ""))
                    duration_score = max(0, 1.0 - hours / 4)
                except:
                    duration_score = 0.5
            elif "立即" in duration_str:
                duration_score = 1.0
            elif "0分钟" in duration_str or "取消" in plan_name:
                duration_score = 0.8 if closure_duration > 1 else 0.4
            else:
                duration_score = 0.5

            # 时限调整因子
            if "等待" in plan_name and closure_duration > 2:
                duration_score *= 0.5  # 时限长，等待成本高
            elif "绕行" in plan_name and closure_duration <= 0.5:
                duration_score *= 0.7  # 时限短，绕行不划算

            total_score = risk_score * 0.5 + duration_score * 0.5
            raw_scores.append((plan_id, plan_name, config, total_score, risk_score))

        # Softmax归一化得到概率
        import numpy as np
        scores = np.array([s[3] for s in raw_scores])
        exp_scores = np.exp(scores - np.max(scores))
        probabilities = (exp_scores / exp_scores.sum()).tolist()

        # 构建方案列表
        for i, (plan_id, plan_name, config, _, risk_score) in enumerate(raw_scores):
            # 获取路线详情（如果有）
            route_details = None
            for route in self.belief_state.route_options:
                if plan_name in route.route_name or "绕行" in plan_name:
                    route_details = RouteDetails(
                        route_name=route.route_name,
                        route_type=route.route_type,
                        waypoints=route.waypoints,
                        distance_km=route.distance_km,
                        estimated_time_minutes=route.estimated_time_minutes,
                        toll_cost=route.toll_cost,
                        has_congestion=route.congestion_probability > 0.5,
                        congestion_probability=route.congestion_probability,
                        traffic_status=route.traffic_status
                    )
                    break

            plan = DecisionPlan(
                plan_id=plan_id,
                name=plan_name,
                description=config.get("description", ""),
                estimated_duration=config.get("estimated_duration", "未知"),
                target_location=config.get("target_location"),
                risk_factors=config.get("risk_factors", {}),
                overall_risk_level=config.get("overall_risk_level", "medium"),
                atomic_actions=[],
                value_orientation={},
                success_probability=probabilities[i],
                iteration_created=2,
                route_details=route_details,
                reasoning=self._generate_plan_reasoning(plan_name, config, closure_duration, risk_score),
                search_queries=[f"搜索{plan_name}相关信息"]
            )
            plans.append(plan)

        # 找到概率最高的方案
        max_idx = probabilities.index(max(probabilities))
        recommended_plan_id = plans[max_idx].plan_id if plans else ""

        reasoning_result = "最终方案概率分布:\n"
        for plan, prob in zip(plans, probabilities):
            reasoning_result += f"  - {plan.name}: {prob*100:.1f}% (理由: {plan.reasoning[:30]}...)\n"
        reasoning_result += f"\n推荐方案: {plans[max_idx].name if plans else '无'}"

        final_iteration = DecisionIteration(
            iteration_id=5,
            stage=IterationStage.FINAL_DECISION.value,
            stage_description="基于所有评估结果计算方案概率分布",
            search_query="",
            search_result="",
            input_information={"num_plans": len(plans)},
            reasoning_result=reasoning_result,
            beliefs_updated={},
            plans_considered=[p.plan_id for p in plans],
            action_taken="final_decision",
            confidence_delta=0.1,
            timestamp=time.time()
        )

        return final_iteration, plans, probabilities

    def _generate_plan_reasoning(self, plan_name: str, config: Dict, closure_duration: float, risk_score: float) -> str:
        """生成方案选择理由"""
        reasoning = ""

        if "等待" in plan_name:
            if closure_duration <= 0.5:
                reasoning = "封闭时限较短（30分钟内），等待成本低，等待概率较高"
            elif closure_duration > 2:
                reasoning = "封闭时限较长（>2小时），等待成本高，建议考虑绕行"
            else:
                reasoning = "封闭时限中等，等待观察等待情况变化"
        elif "绕行" in plan_name:
            if closure_duration > 2:
                reasoning = "封闭时限较长（>2小时），绕行可节省时间成本"
            else:
                reasoning = "绕行可避开封闭路段，但需要额外时间成本"
        elif "取消" in plan_name:
            reasoning = "任务紧急或封闭时间不确定时，取消行程是最稳妥选择"
        else:
            reasoning = f"方案风险等级{config.get('overall_risk_level')}，综合评分{risk_score:.2f}"

        return reasoning


def run_iterative_decision(
    scenario_input: Dict[str, Any],
    context_type: str = "default",
    atomic_probs: Any = None,
    use_web_search: bool = True
) -> Tuple[List[DecisionPlan], List[float], List[DecisionIteration]]:
    """便捷函数：运行迭代决策过程

    Args:
        scenario_input: 场景输入
        context_type: 场景类型 (traffic/social/work/emergency/default)
        atomic_probs: 原子行动概率（可选）
        use_web_search: 是否使用网络搜索（默认True）

    Returns:
        (plans, probabilities, iterations)
    """
    engine = DecisionIterationEngine(context_type, use_web_search)
    return engine.run(scenario_input, atomic_probs)