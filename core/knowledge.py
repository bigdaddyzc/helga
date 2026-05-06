"""Knowledge base module for HELGA.

Provides domain-specific knowledge for action parameterization.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ActionStep:
    """Single step in an action plan."""
    step_number: int
    description: str
    details: Dict[str, Any]
    warning: Optional[str] = None


@dataclass
class ActionPlan:
    """Detailed action plan with specific steps."""
    title: str
    action_name: str
    steps: List[ActionStep]
    time_estimate: str
    resources: Dict[str, Any]
    alternatives: List[str]
    reasoning: str


class BaseKnowledge(ABC):
    """Base class for domain knowledge."""

    @abstractmethod
    def get_name(self) -> str:
        """Return knowledge domain name."""
        pass

    @abstractmethod
    def supports_action(self, action_name: str) -> bool:
        """Check if this knowledge supports the given action."""
        pass

    @abstractmethod
    def parameterize(
        self,
        action_name: str,
        scenario: str,
        context: Dict[str, Any]
    ) -> Optional[ActionPlan]:
        """Generate detailed action plan for the given action and scenario."""
        pass


class TrafficKnowledge(BaseKnowledge):
    """Traffic/driving domain knowledge."""

    # 路线数据
    ROUTES = {
        ("郴州", "长沙"): {
            "highway_blocked": {
                "detour": "G107国道",
                "via": ["永安", "耒阳", "湘潭"],
                "distance": "约280公里",
                "time": "3.5小时",
                "description": "沿G107国道向东北方向行驶，经永安、耒阳、湘潭到达长沙"
            },
            "check_info": {
                "hotline": "12122",
                "hotline_name": "湖南高速服务热线",
                "apps": ["导航地图", "高速通", "ETC宝"]
            }
        }
    }

    HOTLINES = {
        "湖南高速": {"number": "12122", "service": "路况查询/救援"},
        "交通事故": {"number": "122", "service": "事故报警"},
        "道路救援": {"number": "12122", "service": "拖车服务"}
    }

    def get_name(self) -> str:
        return "traffic"

    def supports_action(self, action_name: str) -> bool:
        traffic_actions = [
            "take_detour", "wait_traffic_clear", "check_info",
            "call_service", "cancel_trip", "continue_anyway"
        ]
        return action_name in traffic_actions

    def parameterize(
        self,
        action_name: str,
        scenario: str,
        context: Dict[str, Any]
    ) -> Optional[ActionPlan]:
        """Generate traffic-specific action plan."""
        scenario_lower = scenario.lower()

        # 提取关键地点
        origin = self._extract_location(scenario, ["郴州", "chenzhou"])
        destination = self._extract_location(scenario, ["长沙", "changsha"])

        if action_name == "take_detour":
            return self._plan_detour(origin, destination, scenario)
        elif action_name == "check_info":
            return self._plan_check_info(origin, destination, scenario)
        elif action_name == "call_service":
            return self._plan_call_service(scenario)
        elif action_name == "cancel_trip":
            return self._plan_cancel_trip(origin, destination, scenario)
        elif action_name == "wait_traffic_clear":
            return self._plan_wait(origin, destination, scenario)

        return None

    def _extract_location(self, text: str, keywords: List[str]) -> Optional[str]:
        """Extract location from text."""
        for kw in keywords:
            if kw.lower() in text.lower():
                return kw
        return None

    def _plan_detour(
        self,
        origin: Optional[str],
        destination: Optional[str],
        scenario: str
    ) -> ActionPlan:
        """Plan detour route."""
        route_key = (origin or "郴州", destination or "长沙")
        route_info = self.ROUTES.get(route_key, {}).get("highway_blocked", {
            "detour": "G107国道",
            "via": ["永安", "耒阳", "湘潭"],
            "distance": "约280公里",
            "time": "3.5小时",
            "description": "沿主要道路向目的地行驶"
        })

        return ActionPlan(
            title=f"{origin or '出发地'}至{destination or '目的地'}绕行方案",
            action_name="take_detour",
            steps=[
                ActionStep(
                    step_number=1,
                    description=f"从{origin or '郴州'}收费站驶出，查看出口指示牌",
                    details={"action": "驶出高速", "direction": "按指示牌行驶"},
                    warning="注意车速降至60km/h以下"
                ),
                ActionStep(
                    step_number=2,
                    description=f"沿{route_info['detour']}国道向{route_info['via'][-1] if route_info.get('via') else '目的地'}方向行驶",
                    details={
                        "route": route_info['detour'],
                        "via": route_info.get('via', []),
                        "distance": route_info.get('distance', "约300公里")
                    },
                    warning="注意沿途交通标志，适时休息"
                ),
                ActionStep(
                    step_number=3,
                    description=f"经{route_info['via'][-1] if route_info.get('via') else '城市'}进入{destination or '长沙'}城区",
                    details={"entering": destination or "长沙城区"},
                    warning="进入城区后注意限速和交通信号"
                )
            ],
            time_estimate=route_info.get('time', '3-4小时'),
            resources={
                "phone": "12122",
                "app": "建议使用导航地图实时规划路线"
            },
            alternatives=[
                f"走省道S312经汝城→宜章→韶关→{destination or '长沙'}（路程更长但风景更好）",
                f"等待高速恢复通行后再出发（建议每30分钟查询一次路况）"
            ],
            reasoning=f"{route_info['detour']}是最短绕行路线，路况良好，预计{route_info.get('time', '3-4小时')}到达"
        )

    def _plan_check_info(
        self,
        origin: Optional[str],
        destination: Optional[str],
        scenario: str
    ) -> ActionPlan:
        """Plan traffic info check."""
        return ActionPlan(
            title="查询实时路况信息",
            action_name="check_info",
            steps=[
                ActionStep(
                    step_number=1,
                    description="拨打湖南高速服务热线12122",
                    details={"phone": "12122", "service": "路况查询"},
                    warning="高峰期可能需要等待"
                ),
                ActionStep(
                    step_number=2,
                    description="使用手机APP查询实时路况",
                    details={
                        "apps": ["高德地图", "百度地图", "腾讯地图"],
                        "features": ["实时路况", "事件播报", "路线规划"]
                    }
                ),
                ActionStep(
                    step_number=3,
                    description="关注可变情报板（CMS）信息",
                    details={"location": "收费站前电子显示屏"},
                    warning="根据情报板提示调整行驶计划"
                )
            ],
            time_estimate="5-10分钟",
            resources={
                "phone": "12122（免费）",
                "data": "需要手机网络连接"
            },
            alternatives=[
                "直接尝试绕行G107国道",
                "关注周边司机群/论坛的真实反馈"
            ],
            reasoning="在做出绕行决定前，先确认路况信息可以避免不必要的麻烦"
        )

    def _plan_call_service(self, scenario: str) -> ActionPlan:
        """Plan calling service hotline."""
        return ActionPlan(
            title="拨打高速服务热线",
            action_name="call_service",
            steps=[
                ActionStep(
                    step_number=1,
                    description="拨打湖南高速服务热线",
                    details={"phone": "12122", "service": "路况咨询"},
                    warning="请记录通话内容以备后续参考"
                ),
                ActionStep(
                    step_number=2,
                    description="咨询收费站关闭原因和预计恢复时间",
                    details={"questions": [
                        "收费站预计何时恢复通行？",
                        "有哪些绕行路线推荐？",
                        "当前路况如何？"
                    ]}
                ),
                ActionStep(
                    step_number=3,
                    description="根据热线指引决定后续行动",
                    details={"options": ["等待恢复", "选择绕行", "咨询救援"]}
                )
            ],
            time_estimate="5-15分钟",
            resources={"phone": "12122"},
            alternatives=["使用手机APP在线咨询"],
            reasoning="人工服务可以提供最准确的信息和建议"
        )

    def _plan_cancel_trip(
        self,
        origin: Optional[str],
        destination: Optional[str],
        scenario: str
    ) -> ActionPlan:
        """Plan trip cancellation."""
        return ActionPlan(
            title="取消/推迟行程",
            action_name="cancel_trip",
            steps=[
                ActionStep(
                    step_number=1,
                    description=f"通知{destination or '目的地'}相关人员行程变更",
                    details={"通知对象": "需要告知的人员"}
                ),
                ActionStep(
                    step_number=2,
                    description="重新安排行程时间",
                    details={"options": ["改期", "更换交通方式", "取消"]}
                ),
                ActionStep(
                    step_number=3,
                    description="如已上高速，驶出最近出口返回",
                    details={"建议": "选择最近出口驶出，不要在高速上停留"}
                )
            ],
            time_estimate="立即执行",
            resources={"通讯": "通知相关方"},
            alternatives=["等待路况好转后出发", "改乘高铁/飞机"],
            reasoning="安全第一，避免在不熟悉的路况下冒险行驶"
        )

    def _plan_wait(
        self,
        origin: Optional[str],
        destination: Optional[str],
        scenario: str
    ) -> ActionPlan:
        """Plan waiting for traffic to clear."""
        return ActionPlan(
            title="等待路况好转",
            action_name="wait_traffic_clear",
            steps=[
                ActionStep(
                    step_number=1,
                    description=f"在{origin or '当前位置'}安全停靠",
                    details={"location": "服务区或安全地带"},
                    warning="确保不占用行车道"
                ),
                ActionStep(
                    step_number=2,
                    description="每15-20分钟查询一次路况",
                    details={"methods": ["APP查询", "热线咨询"]}
                ),
                ActionStep(
                    step_number=3,
                    description="路况好转后立即出发",
                    details={"trigger": "收到恢复通行通知"}
                )
            ],
            time_estimate="不确定（取决于恢复时间）",
            resources={"建议": "保持手机电量充足"},
            alternatives=["选择绕行路线", "取消行程"],
            reasoning="等待是最安全的选择，但可能耗费较多时间"
        )


class WorkplaceKnowledge(BaseKnowledge):
    """Workplace/business domain knowledge."""

    def get_name(self) -> str:
        return "workplace"

    def supports_action(self, action_name: str) -> bool:
        workplace_actions = [
            "send_reminder", "reschedule", "escalate", "ask_reason",
            "provide_help", "delegate", "confirm", "deny", "approve", "reject"
        ]
        return action_name in workplace_actions

    def parameterize(
        self,
        action_name: str,
        scenario: str,
        context: Dict[str, Any]
    ) -> Optional[ActionPlan]:
        """Generate workplace-specific action plan."""
        # 简化实现，待扩展
        return ActionPlan(
            title=f"执行动作: {action_name}",
            action_name=action_name,
            steps=[
                ActionStep(
                    step_number=1,
                    description=f"执行{action_name}动作",
                    details={}
                )
            ],
            time_estimate="待定",
            resources={},
            alternatives=[],
            reasoning="工作场景动作参数化"
        )


def create_knowledge_base() -> List[BaseKnowledge]:
    """Create knowledge base with all domains."""
    return [
        TrafficKnowledge(),
        WorkplaceKnowledge()
    ]


def get_knowledge_for_action(
    action_name: str,
    knowledge_base: List[BaseKnowledge]
) -> Optional[BaseKnowledge]:
    """Find knowledge base that supports the given action."""
    for kb in knowledge_base:
        if kb.supports_action(action_name):
            return kb
    return None
