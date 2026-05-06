"""文本生成器 - Text Generator

将决策结果转换为可读文本方案
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .action_space import Action


@dataclass
class PlanStep:
    """方案步骤

    Attributes:
        step_number: 步骤编号
        description: 步骤描述
        reason: 为什么要这样做
        warning: 注意事项（可选）
        route_name: 可选的路线名称（用于交通场景）
    """
    step_number: int
    description: str
    reason: str
    warning: Optional[str] = None
    route_name: Optional[str] = None


@dataclass
class RouteComparisonRow:
    """路线对比行

    Attributes:
        route_name: 路线名称
        distance_km: 距离（公里）
        estimated_time_minutes: 预计时间（分钟）
        toll_cost: 高速费用（元）
        risk_level: 风险等级（"low", "medium", "high"）
    """
    route_name: str
    distance_km: float
    estimated_time_minutes: int
    toll_cost: float
    risk_level: str


@dataclass
class ActionPlan:
    """可执行的动作方案

    Attributes:
        title: 方案标题
        summary: 方案概述
        steps: 执行步骤列表
        time_estimate: 时间估计
        risk_factors: 风险因素列表
        alternatives: 备选方案列表
        reasoning: 决策理由
        route_comparison: 路线对比表（可选，用于交通场景）
    """
    title: str
    summary: str
    steps: List[PlanStep]
    time_estimate: str
    risk_factors: List[str]
    alternatives: List[str]
    reasoning: str
    route_comparison: Optional[List[RouteComparisonRow]] = None


class TextGenerator:
    """文本生成器

    将数学公式输出的方案转换为可读的文本格式
    """

    # 时间估算映射
    TIME_MAPPING = {
        "low": ("5-15分钟", "快速"),
        "medium": ("30分钟-2小时", "中等"),
        "high": ("2小时以上", "较长")
    }

    # 动作名称映射（中文）
    ACTION_NAMES = {
        "send_message": "发送消息",
        "make_call": "打电话",
        "send_email": "发邮件",
        "search": "搜索信息",
        "compare": "对比选项",
        "read_document": "阅读文档",
        "recommend": "推荐",
        "accept": "接受",
        "reject": "拒绝",
        "modify": "修改",
        "wait": "等待",
        "delay": "延期",
        "cancel": "取消",
        "delegate": "委托",
        "schedule": "安排日程",
        "plan": "制定计划",
        "reschedule_meeting": "重新安排会议",
        "plan_travel": "规划旅行",
        "make_decision": "做出决定",
        "solve_problem": "解决问题",
        "negotiate": "协商谈判",
        # 通用决策动作
        "find_alternative": "寻找替代方案",
        "replan": "重新规划",
        "evaluate_options": "评估选项",
        "explore_options": "探索选项",
        "consult_expert": "咨询专家",
        "make_compromise": "妥协折中",
        # 交通相关动作
        "check_navigation": "查看导航路线",
        "check_traffic_info": "查看路况信息",
        "find_detour": "寻找绕行路线",
        "change_route": "改变路线",
        "wait_traffic": "等待路况好转",
        "call_service_hotline": "拨打服务热线",
        "use_public_transport": "使用公共交通",
    }

    def __init__(self):
        pass

    def generate(
        self,
        action: Action,
        context: Dict[str, Any],
        alternatives: Optional[List[Action]] = None,
        reasoning: str = ""
    ) -> ActionPlan:
        """生成文本方案

        Args:
            action: 推荐的动作
            context: 上下文信息（问题、搜索结果等）
            alternatives: 备选动作列表
            reasoning: 决策理由

        Returns:
            ActionPlan
        """
        # 检测是否交通相关问题
        question = context.get("question", "").lower()
        is_traffic_related = any(kw in question for kw in ["开车", "开车", "高速", "收费", "路线", "堵车", "拥堵", "回长沙", "回北京", "出行"])

        # 确定时间估算（交通问题需要更具体的时间）
        if is_traffic_related and action.id in ["check_navigation", "check_traffic_info", "find_detour", "change_route"]:
            time_estimate = "10-20分钟（导航规划）"
        elif is_traffic_related:
            time_estimate = "30分钟-2小时（含出行时间）"
        else:
            time_estimate = self._estimate_time(action)

        # 生成步骤
        steps = self._generate_steps(action, context)

        # 识别风险因素（交通问题有特定风险）
        risk_factors = self._identify_risks(action, is_traffic_related)

        # 生成备选方案
        alt_texts = self._generate_alternatives(alternatives)

        # 生成路线对比表（如果有路线信息）
        routes = context.get("routes", [])
        route_comparison = self._generate_route_comparison(routes) if routes else None

        # 构建方案
        plan = ActionPlan(
            title=self._generate_title(action, context, is_traffic_related),
            summary=self._generate_summary(action, context, reasoning),
            steps=steps,
            time_estimate=time_estimate,
            risk_factors=risk_factors,
            alternatives=alt_texts,
            reasoning=reasoning,
            route_comparison=route_comparison
        )

        return plan

    def to_markdown(self, plan: ActionPlan) -> str:
        """将方案转换为Markdown格式

        Args:
            plan: ActionPlan

        Returns:
            Markdown字符串
        """
        lines = []

        # 标题
        lines.append(f"# {plan.title}")
        lines.append("")
        lines.append("## 概述")
        lines.append(plan.summary)
        lines.append("")

        # 执行步骤
        lines.append("## 执行步骤")
        for step in plan.steps:
            lines.append(f"### 步骤 {step.step_number}")
            lines.append(step.description)
            lines.append(f"**理由**: {step.reason}")
            if step.warning:
                lines.append(f"**注意**: {step.warning}")
            lines.append("")

        # 时间估计
        lines.append("## 时间估计")
        lines.append(f"预计总时长: {plan.time_estimate}")
        lines.append("")

        # 风险因素
        if plan.risk_factors:
            lines.append("## 风险因素")
            for i, risk in enumerate(plan.risk_factors, 1):
                lines.append(f"{i}. {risk}")
            lines.append("")

        # 路线对比表（如果有）
        if plan.route_comparison:
            lines.append("## 路线对比")
            lines.append("")
            lines.append("| 路线 | 距离 | 预计时间 | 高速费用 | 风险等级 |")
            lines.append("|------|------|----------|----------|----------|")
            for route in plan.route_comparison:
                hours = route.estimated_time_minutes // 60
                mins = route.estimated_time_minutes % 60
                time_str = f"{hours}小时{mins}分钟" if hours > 0 else f"{mins}分钟"
                lines.append(f"| {route.route_name} | {route.distance_km}公里 | {time_str} | {route.toll_cost}元 | {route.risk_level} |")
            lines.append("")

        # 备选方案
        if plan.alternatives:
            lines.append("## 备选方案")
            lines.append("如果主方案不可行，可以考虑:")
            for i, alt in enumerate(plan.alternatives, 1):
                lines.append(f"{i}. {alt}")
            lines.append("")

        # 决策理由
        lines.append("## 决策理由")
        lines.append(plan.reasoning)

        return "\n".join(lines)

    def to_json(self, plan: ActionPlan) -> Dict:
        """将方案转换为JSON格式

        Args:
            plan: ActionPlan

        Returns:
            JSON兼容的字典
        """
        return {
            "title": plan.title,
            "summary": plan.summary,
            "steps": [
                {
                    "step_number": step.step_number,
                    "description": step.description,
                    "reason": step.reason,
                    "warning": step.warning,
                    "route_name": step.route_name
                }
                for step in plan.steps
            ],
            "time_estimate": plan.time_estimate,
            "risk_factors": plan.risk_factors,
            "alternatives": plan.alternatives,
            "reasoning": plan.reasoning,
            "route_comparison": [
                {
                    "route_name": r.route_name,
                    "distance_km": r.distance_km,
                    "estimated_time_minutes": r.estimated_time_minutes,
                    "toll_cost": r.toll_cost,
                    "risk_level": r.risk_level
                }
                for r in plan.route_comparison
            ] if plan.route_comparison else None
        }

    def _generate_title(self, action: Action, context: Dict, is_traffic_related: bool = False) -> str:
        """生成方案标题"""
        action_name = self.ACTION_NAMES.get(action.name, action.name)
        question = context.get("question", "")

        # 交通问题使用更简洁的标题
        if is_traffic_related:
            # 提取关键信息
            if "郴州" in question and "长沙" in question:
                location = "郴州→长沙"
            elif "北京" in question:
                location = "→北京"
            else:
                location = "出行路线"

            if "收费" in question:
                return f"绕行方案：{location}（收费站关闭）"
            else:
                return f"出行方案：{location}"
        else:
            question_short = question[:30] if question else ""
            if question_short:
                return f"关于「{question_short}...」的{action_name}方案"
            return f"{action_name}方案"

    def _generate_summary(self, action: Action, context: Dict, reasoning: str) -> str:
        """生成方案概述"""
        action_name = self.ACTION_NAMES.get(action.name, action.name)
        question = context.get("question", "")

        summary_parts = [
            f"根据您的问题「{question[:50]}...」",
            f"系统推荐采取{action_name}行动。"
        ]

        if reasoning:
            summary_parts.append(f"决策依据: {reasoning[:100]}...")

        return " ".join(summary_parts)

    def _generate_steps(self, action: Action, context: Dict) -> List[PlanStep]:
        """生成执行步骤"""
        steps = []
        action_id = action.id

        # 根据动作类型生成不同步骤
        if action_id in ["send_message", "make_call", "send_email"]:
            # 沟通类动作
            steps.append(PlanStep(
                step_number=1,
                description="准备沟通内容：明确要表达的核心信息和目标",
                reason="清晰的沟通目标可以提高沟通效率",
                warning="避免过长或过于复杂的表达"
            ))
            steps.append(PlanStep(
                step_number=2,
                description=f"通过{self.ACTION_NAMES.get(action_id, action_id)}发送信息",
                reason="选择合适的沟通渠道确保信息传达",
                warning="注意发送时间，避免休息时间打扰"
            ))
            steps.append(PlanStep(
                step_number=3,
                description="等待回复并根据反馈调整后续行动",
                reason="保持沟通的连续性",
                warning="如果长时间未回复，考虑其他沟通方式"
            ))

        elif action_id in ["search", "compare", "read_document"]:
            # 信息类动作
            steps.append(PlanStep(
                step_number=1,
                description="确定需要搜索的关键信息和查询词",
                reason="准确的搜索词可以获得更相关的结果",
                warning="避免过于宽泛或过于狭窄的搜索词"
            ))
            steps.append(PlanStep(
                step_number=2,
                description="执行搜索并收集相关信息",
                reason="获取足够的信息才能做出正确决策",
                warning="注意信息的来源可靠性"
            ))
            steps.append(PlanStep(
                step_number=3,
                description="整理和对比收集到的信息",
                reason="系统化的整理有助于分析",
                warning="记录关键数据和来源"
            ))

        elif action_id in ["accept", "reject", "recommend"]:
            # 决策类动作
            steps.append(PlanStep(
                step_number=1,
                description="确认决策的关键信息和约束条件",
                reason="明确决策的依据",
                warning="检查是否有遗漏的重要信息"
            ))
            steps.append(PlanStep(
                step_number=2,
                description="权衡利弊，分析每个选项的优缺点",
                reason="全面的分析可以降低决策风险",
                warning="避免过度分析导致延误"
            ))
            steps.append(PlanStep(
                step_number=3,
                description="做出决定并明确说明决定内容",
                reason="清晰的决定有助于后续执行",
                warning="记录决定的理由以备复查"
            ))

        elif action_id in ["schedule", "plan"]:
            # 计划类动作
            steps.append(PlanStep(
                step_number=1,
                description="明确目标和时间节点",
                reason="清晰的目标是有效计划的基础",
                warning="目标应该具体、可衡量"
            ))
            steps.append(PlanStep(
                step_number=2,
                description="分解任务，列出具体步骤",
                reason="细化的步骤便于执行和跟踪",
                warning="每个步骤应该有明确的产出"
            ))
            steps.append(PlanStep(
                step_number=3,
                description="预留缓冲时间，考虑潜在风险",
                reason="留有余地可以应对意外情况",
                warning="不要过度乐观估计时间"
            ))

        elif action_id in ["find_alternative", "replan", "evaluate_options", "explore_options"]:
            # 替代方案/重新规划类动作
            steps.append(PlanStep(
                step_number=1,
                description="明确核心问题和约束条件",
                reason="知道要解决什么才能找到合适的替代方案",
                warning="列出所有硬性约束（如时间、预算）"
            ))
            steps.append(PlanStep(
                step_number=2,
                description="列出所有可能的替代方案",
                reason="广泛收集选项，不要提前排除",
                warning="考虑不同的路线、时间、方式"
            ))
            steps.append(PlanStep(
                step_number=3,
                description="评估每个方案的可行性和优劣",
                reason="对比分析才能做出好决策",
                warning="考虑风险、成本、成功率"
            ))
            steps.append(PlanStep(
                step_number=4,
                description="选择最优方案并制定执行计划",
                reason="决策后需要明确执行步骤",
                warning="准备备用方案以防首选失败"
            ))

        elif action_id in ["wait_and_see", "wait", "delay"]:
            # 等待类动作
            steps.append(PlanStep(
                step_number=1,
                description="确定可以等待的时间窗口",
                reason="明确最晚什么时候需要做出决定",
                warning="不要无限期等待"
            ))
            steps.append(PlanStep(
                step_number=2,
                description="监控情况变化，等待更多信息",
                reason="等待期间保持信息更新",
                warning="设置提醒，定期检查状态"
            ))
            steps.append(PlanStep(
                step_number=3,
                description="在时间窗口内做出最终决定",
                reason="等待要有期限，到期必须行动",
                warning="准备好备选方案"
            ))

        elif action_id in ["consult_expert", "ask_advice", "seek_help"]:
            # 咨询类动作
            steps.append(PlanStep(
                step_number=1,
                description="准备问题清单和背景信息",
                reason="有效的咨询需要清晰的问题",
                warning="整理好相关材料和上下文"
            ))
            steps.append(PlanStep(
                step_number=2,
                description="联系相关领域的专家或知情人士",
                reason="专业意见可以帮助做出更好决策",
                warning="选择真正懂行的人"
            ))
            steps.append(PlanStep(
                step_number=3,
                description="综合专家意见，做出最终决策",
                reason="咨询是为了辅助决策，不是替代决策",
                warning="自己判断，不要完全依赖他人"
            ))

        elif action_id in ["check_navigation", "check_traffic_info", "find_detour", "change_route"]:
            # 交通导航类动作 - 检查是否有路线上下文
            routes = context.get("routes", [])

            if routes and len(routes) > 0:
                # 使用具体的路线信息生成步骤
                steps = self._generate_route_steps(routes, action_id)
            else:
                # 使用通用模板步骤
                steps.append(PlanStep(
                    step_number=1,
                    description="打开高德地图/百度地图/腾讯地图",
                    reason="电子地图能显示实时路况和多条路线",
                    warning="确保手机信号良好或离线地图已下载"
                ))
                steps.append(PlanStep(
                    step_number=2,
                    description="输入起点'郴州'和终点'长沙'，查看推荐路线",
                    reason="导航软件会显示预计时间、距离、路况",
                    warning="注意选择'不走高速'选项查看普通公路"
                ))
                steps.append(PlanStep(
                    step_number=3,
                    description="重点关注G4京港澳高速、G107国道、以及其他省级公路",
                    reason="多一个选择就多一条路",
                    warning="注意路线是否有施工或拥堵"
                ))
                steps.append(PlanStep(
                    step_number=4,
                    description="对比各路线的距离、时间、成本，选择最优",
                    reason="选择最符合当前需求的路线",
                    warning="考虑备用路线以防首选拥堵"
                ))

        elif action_id in ["call_service_hotline"]:
            # 拨打服务热线类动作
            steps.append(PlanStep(
                step_number=1,
                description="拨打高速公路服务热线 12122",
                reason="官方渠道获取最准确的收费站状态",
                warning="可能需要等待，保持耐心"
            ))
            steps.append(PlanStep(
                step_number=2,
                description="咨询郴州北收费站预计开放时间和替代路线",
                reason="获取一手信息再做决策",
                warning="记录工作人员提供的信息"
            ))
            steps.append(PlanStep(
                step_number=3,
                description="根据热线信息选择绕行路线或等待",
                reason="官方信息最可靠",
                warning="如果等待时间过长，考虑备用方案"
            ))

        elif action_id in ["wait_traffic"]:
            # 等待路况好转
            steps.append(PlanStep(
                step_number=1,
                description="确定可以等待的时间上限（如1-2小时）",
                reason="等待要有期限，避免耽误重要行程",
                warning="设置手机提醒，定期查看路况"
            ))
            steps.append(PlanStep(
                step_number=2,
                description="利用等待时间查看绕行路线作为备选",
                reason="提前准备，等的同时也在规划",
                warning="不要只被动等待"
            ))
            steps.append(PlanStep(
                step_number=3,
                description="如果等待时间超过上限，选择绕行路线出发",
                reason="时间成本也是成本",
                warning="提前查看备选路线的路况"
            ))

        elif action_id in ["use_public_transport"]:
            # 使用公共交通
            steps.append(PlanStep(
                step_number=1,
                description="查看高铁/城际列车从郴州到长沙的班次",
                reason="郴州西站有高铁直达长沙南，最快35分钟",
                warning="提前买票可能没座位"
            ))
            steps.append(PlanStep(
                step_number=2,
                description="查询大巴/城际巴士从郴州到长沙的班次",
                reason="普通公路客运也是一种选择",
                warning="注意末班车时间"
            ))
            steps.append(PlanStep(
                step_number=3,
                description="比较公共交通vs自驾绕行的时间成本",
                reason="选择最省时的方案",
                warning="考虑出发时间去车站的时间"
            ))

        else:
            # 通用步骤
            steps.append(PlanStep(
                step_number=1,
                description="了解当前情况和可用资源",
                reason="充分的信息是正确行动的基础"
            ))
            steps.append(PlanStep(
                step_number=2,
                description="制定行动方案并确认可行性",
                reason="可行的方案才能顺利执行"
            ))
            steps.append(PlanStep(
                step_number=3,
                description="执行行动并监控结果",
                reason="跟踪执行可以及时发现问题"
            ))

        return steps

    def _generate_route_steps(self, routes: List[Dict[str, Any]], action_id: str) -> List[PlanStep]:
        """生成具体路线的步骤

        Args:
            routes: 路线信息列表
            action_id: 动作ID

        Returns:
            PlanStep列表
        """
        steps = []
        step_num = 1

        for i, route in enumerate(routes[:3]):  # 最多显示3条路线
            route_name = route.get("route_name", "未知路线")
            distance = route.get("distance_km", 0)
            duration = route.get("estimated_time_minutes", 0)
            toll = route.get("toll_cost", 0)
            traffic = route.get("traffic_status", "未知")

            # 路线选择步骤
            steps.append(PlanStep(
                step_number=step_num,
                description=f"路线{i+1}: {route_name}",
                reason=f"距离{distance}公里，预计{duration}分钟，收费{toll}元，路况{traffic}",
                warning="出发前再次确认实时路况",
                route_name=route_name
            ))
            step_num += 1

        # 如果有更多路线，给出对比建议
        if len(routes) > 3:
            steps.append(PlanStep(
                step_number=step_num,
                description=f"还有{len(routes)-3}条备选路线，建议通过导航App对比",
                reason="根据实时路况选择最优路线",
                warning="注意避开施工或拥堵路段"
            ))

        return steps

    def _generate_route_comparison(self, routes: List[Dict[str, Any]]) -> List[RouteComparisonRow]:
        """生成路线对比表

        Args:
            routes: 路线信息列表

        Returns:
            RouteComparisonRow列表
        """
        comparison = []
        for route in routes:
            risk_level = route.get("risk_level", "medium")
            comparison.append(RouteComparisonRow(
                route_name=route.get("route_name", "未知"),
                distance_km=route.get("distance_km", 0),
                estimated_time_minutes=route.get("estimated_time_minutes", 0),
                toll_cost=route.get("toll_cost", 0),
                risk_level=risk_level
            ))
        return comparison

    def _identify_risks(self, action: Action, is_traffic_related: bool = False) -> List[str]:
        """识别风险因素

        Args:
            action: 动作
            is_traffic_related: 是否交通相关问题

        Returns:
            风险因素列表
        """
        risks = []
        action_id = action.id

        # 交通问题特定风险
        if is_traffic_related:
            if action_id in ["check_navigation", "check_traffic_info"]:
                risks.append("导航信息可能与实际路况有差异")
            elif action_id in ["find_detour", "change_route"]:
                risks.append("绕行路线可能增加行程时间30-60分钟")
                risks.append("部分备用路线可能没有完全整修")
            elif action_id in ["wait_traffic"]:
                risks.append("等待时间不确定，可能耽误行程")
            elif action_id in ["use_public_transport"]:
                risks.append("公共交通班次有限，可能需要等待")

        # 成本风险
        cost = action.cost_vector
        if cost[0] > 0.6:  # 时间成本高
            risks.append("执行时间可能较长，需要预留足够时间")
        if cost[1] > 0.6:  # 金钱成本高
            risks.append("可能需要一定的资金投入")
        if cost[2] > 0.6:  # 风险高
            risks.append("存在一定的不确定性和潜在风险")

        # 动作特定风险
        if action_id == "send_message":
            risks.append("消息可能不会被及时看到")
        elif action_id == "make_call":
            risks.append("对方可能不方便接听")
        elif action_id == "accept":
            risks.append("接受后可能无法反悔，需要谨慎")
        elif action_id == "reject":
            risks.append("拒绝可能导致关系紧张")
        elif action_id == "cancel":
            risks.append("取消可能造成资源浪费或机会损失")

        return risks[:3]  # 最多返回3个风险

    def _estimate_time(self, action: Action) -> str:
        """估算时间"""
        cost = action.cost_vector
        avg_cost = (cost[0] + cost[1] + cost[2]) / 3

        if avg_cost < 0.3:
            return self.TIME_MAPPING["low"][0]
        elif avg_cost < 0.6:
            return self.TIME_MAPPING["medium"][0]
        else:
            return self.TIME_MAPPING["high"][0]

    def _generate_alternatives(self, alternatives: Optional[List[Action]]) -> List[str]:
        """生成备选方案描述"""
        if not alternatives:
            return []

        alt_texts = []
        for alt in alternatives[:3]:
            alt_name = self.ACTION_NAMES.get(alt.name, alt.name)
            alt_texts.append(f"{alt_name}（成本: {alt.cost_vector.mean():.1f}）")

        return alt_texts


def create_text_generator() -> TextGenerator:
    """工厂函数：创建文本生成器"""
    return TextGenerator()