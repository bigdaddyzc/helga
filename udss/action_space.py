"""动作空间定义 - Action Space

定义原子动作和复合动作，支持参数化
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class Action:
    """动作定义

    Attributes:
        id: 动作唯一标识
        name: 动作名称
        action_type: primitive(原子) / composite(复合)
        parameters: 动作参数
        prerequisites: 前置条件
        outcomes: 可能结果
        value_vector: 10维Schwartz价值向量
        cost_vector: [时间成本, 金钱成本, 风险] 3维
    """
    id: str
    name: str
    action_type: str = "primitive"  # "primitive" or "composite"
    parameters: Dict[str, Any] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)
    value_vector: Optional[np.ndarray] = None
    cost_vector: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.value_vector is None:
            self.value_vector = np.ones(10) * 0.5  # 默认均匀分布
        if self.cost_vector is None:
            self.cost_vector = np.array([0.5, 0.5, 0.5])  # 默认中等成本


# Schwartz 10维价值观索引
VALUE_DIMENSIONS = [
    "self_transcendence",  # 0: 超越自我
    "openness",           # 1: 开放性
    "benevolence",        # 2: 仁爱
    "conformity",         # 3: 顺从
    "security",           # 4: 安全
    "achievement",        # 5: 成就
    "hedonism",           # 6: 享乐
    "stimulation",        # 7: 刺激
    "self_direction",     # 8: 自主
    "universalism"        # 9: 普世
]


class ActionSpace:
    """动作空间管理

    管理所有可用动作，支持动作选择和评分
    """

    # 预定义的原子动作
    PRIMITIVE_ACTIONS = [
        # 沟通类
        Action(id="send_message", name="发消息", action_type="primitive",
               value_vector=np.array([0.3, 0.4, 0.6, 0.5, 0.3, 0.2, 0.2, 0.2, 0.5, 0.4]),
               cost_vector=np.array([0.1, 0.0, 0.1])),
        Action(id="make_call", name="打电话", action_type="primitive",
               value_vector=np.array([0.3, 0.3, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.4, 0.3]),
               cost_vector=np.array([0.2, 0.1, 0.2])),
        Action(id="send_email", name="发邮件", action_type="primitive",
               value_vector=np.array([0.3, 0.4, 0.5, 0.6, 0.4, 0.3, 0.1, 0.1, 0.4, 0.4]),
               cost_vector=np.array([0.2, 0.0, 0.1])),

        # 信息类
        Action(id="search", name="搜索信息", action_type="primitive",
               value_vector=np.array([0.2, 0.8, 0.2, 0.3, 0.4, 0.5, 0.2, 0.5, 0.6, 0.7]),
               cost_vector=np.array([0.1, 0.0, 0.1])),
        Action(id="compare", name="对比选项", action_type="primitive",
               value_vector=np.array([0.3, 0.7, 0.3, 0.4, 0.5, 0.6, 0.2, 0.4, 0.5, 0.6]),
               cost_vector=np.array([0.3, 0.0, 0.1])),
        Action(id="read_document", name="阅读文档", action_type="primitive",
               value_vector=np.array([0.4, 0.6, 0.3, 0.3, 0.4, 0.5, 0.1, 0.3, 0.6, 0.5]),
               cost_vector=np.array([0.5, 0.0, 0.0])),

        # 决策类
        Action(id="recommend", name="推荐", action_type="primitive",
               value_vector=np.array([0.5, 0.4, 0.7, 0.5, 0.5, 0.6, 0.2, 0.2, 0.4, 0.5]),
               cost_vector=np.array([0.2, 0.0, 0.2])),
        Action(id="accept", name="接受", action_type="primitive",
               value_vector=np.array([0.4, 0.3, 0.5, 0.6, 0.5, 0.5, 0.3, 0.2, 0.3, 0.4]),
               cost_vector=np.array([0.1, 0.0, 0.1])),
        Action(id="reject", name="拒绝", action_type="primitive",
               value_vector=np.array([0.2, 0.2, 0.2, 0.4, 0.4, 0.3, 0.1, 0.1, 0.3, 0.2]),
               cost_vector=np.array([0.1, 0.0, 0.3])),
        Action(id="modify", name="修改", action_type="primitive",
               value_vector=np.array([0.3, 0.5, 0.4, 0.4, 0.5, 0.5, 0.2, 0.3, 0.5, 0.4]),
               cost_vector=np.array([0.3, 0.0, 0.2])),

        # 等待类
        Action(id="wait", name="等待", action_type="primitive",
               value_vector=np.array([0.2, 0.2, 0.2, 0.4, 0.7, 0.2, 0.2, 0.1, 0.3, 0.3]),
               cost_vector=np.array([0.5, 0.0, 0.1])),
        Action(id="delay", name="延期", action_type="primitive",
               value_vector=np.array([0.2, 0.3, 0.2, 0.3, 0.5, 0.3, 0.2, 0.2, 0.3, 0.2]),
               cost_vector=np.array([0.4, 0.1, 0.2])),

        # 取消/终止类
        Action(id="cancel", name="取消", action_type="primitive",
               value_vector=np.array([0.1, 0.2, 0.1, 0.3, 0.4, 0.2, 0.1, 0.1, 0.2, 0.2]),
               cost_vector=np.array([0.2, 0.2, 0.4])),
        Action(id="delegate", name="委托", action_type="primitive",
               value_vector=np.array([0.4, 0.5, 0.5, 0.4, 0.4, 0.6, 0.2, 0.3, 0.4, 0.4]),
               cost_vector=np.array([0.2, 0.1, 0.2])),

        # 计划类
        Action(id="schedule", name="安排日程", action_type="primitive",
               value_vector=np.array([0.3, 0.5, 0.4, 0.6, 0.7, 0.5, 0.2, 0.3, 0.5, 0.4]),
               cost_vector=np.array([0.3, 0.0, 0.1])),
        Action(id="plan", name="制定计划", action_type="primitive",
               value_vector=np.array([0.4, 0.6, 0.5, 0.5, 0.6, 0.6, 0.3, 0.4, 0.6, 0.5]),
               cost_vector=np.array([0.4, 0.0, 0.1])),
    ]

    # 预定义的复合动作
    COMPOSITE_ACTIONS = [
        Action(id="reschedule_meeting", name="重新安排会议", action_type="composite",
               value_vector=np.array([0.3, 0.5, 0.4, 0.6, 0.5, 0.4, 0.2, 0.2, 0.4, 0.4]),
               cost_vector=np.array([0.5, 0.1, 0.3])),
        Action(id="plan_travel", name="规划旅行", action_type="composite",
               value_vector=np.array([0.4, 0.8, 0.5, 0.3, 0.5, 0.5, 0.7, 0.8, 0.7, 0.5]),
               cost_vector=np.array([0.6, 0.3, 0.2])),
        Action(id="make_decision", name="做出决定", action_type="composite",
               value_vector=np.array([0.4, 0.5, 0.4, 0.5, 0.6, 0.7, 0.3, 0.4, 0.5, 0.5]),
               cost_vector=np.array([0.3, 0.0, 0.3])),
        Action(id="solve_problem", name="解决问题", action_type="composite",
               value_vector=np.array([0.5, 0.7, 0.6, 0.5, 0.6, 0.7, 0.3, 0.5, 0.6, 0.6]),
               cost_vector=np.array([0.5, 0.1, 0.3])),
        Action(id="negotiate", name="协商谈判", action_type="composite",
               value_vector=np.array([0.5, 0.6, 0.6, 0.5, 0.5, 0.6, 0.3, 0.5, 0.5, 0.5]),
               cost_vector=np.array([0.4, 0.2, 0.4])),
        # 通用决策复合动作
        Action(id="find_alternative", name="寻找替代方案", action_type="composite",
               value_vector=np.array([0.4, 0.8, 0.5, 0.3, 0.6, 0.5, 0.4, 0.6, 0.7, 0.6]),
               cost_vector=np.array([0.4, 0.1, 0.2])),
        Action(id="replan", name="重新规划", action_type="composite",
               value_vector=np.array([0.3, 0.7, 0.4, 0.4, 0.5, 0.5, 0.3, 0.5, 0.6, 0.5]),
               cost_vector=np.array([0.5, 0.0, 0.2])),
        Action(id="evaluate_options", name="评估选项", action_type="composite",
               value_vector=np.array([0.4, 0.7, 0.4, 0.5, 0.6, 0.6, 0.2, 0.5, 0.6, 0.6]),
               cost_vector=np.array([0.4, 0.0, 0.1])),
        Action(id="consult_expert", name="咨询专家", action_type="composite",
               value_vector=np.array([0.5, 0.6, 0.7, 0.5, 0.5, 0.5, 0.2, 0.4, 0.5, 0.6]),
               cost_vector=np.array([0.3, 0.2, 0.2])),
        Action(id="make_compromise", name="妥协折中", action_type="composite",
               value_vector=np.array([0.5, 0.5, 0.6, 0.6, 0.5, 0.4, 0.3, 0.3, 0.4, 0.5]),
               cost_vector=np.array([0.2, 0.1, 0.2])),
    ]

    # 扩展的通用原子动作（补充原有15个，形成30+动作池）
    EXTENDED_ACTIONS = [
        # 探索类
        Action(id="explore_options", name="探索选项", action_type="primitive",
               value_vector=np.array([0.3, 0.9, 0.4, 0.2, 0.4, 0.5, 0.4, 0.7, 0.7, 0.5]),
               cost_vector=np.array([0.3, 0.0, 0.2])),
        Action(id="gather_info", name="收集信息", action_type="primitive",
               value_vector=np.array([0.3, 0.8, 0.3, 0.3, 0.4, 0.5, 0.2, 0.5, 0.6, 0.6]),
               cost_vector=np.array([0.2, 0.0, 0.1])),

        # 适应类
        Action(id="wait_and_see", name="等待观察", action_type="primitive",
               value_vector=np.array([0.2, 0.3, 0.2, 0.4, 0.6, 0.2, 0.2, 0.2, 0.3, 0.3]),
               cost_vector=np.array([0.4, 0.0, 0.1])),
        Action(id="adjust_plan", name="调整计划", action_type="primitive",
               value_vector=np.array([0.3, 0.6, 0.4, 0.4, 0.5, 0.5, 0.2, 0.4, 0.5, 0.4]),
               cost_vector=np.array([0.3, 0.0, 0.2])),
        Action(id="adapt_situation", name="适应情况", action_type="primitive",
               value_vector=np.array([0.3, 0.7, 0.4, 0.4, 0.5, 0.5, 0.3, 0.5, 0.6, 0.5]),
               cost_vector=np.array([0.2, 0.0, 0.2])),

        # 求助类
        Action(id="seek_help", name="寻求帮助", action_type="primitive",
               value_vector=np.array([0.5, 0.5, 0.7, 0.5, 0.4, 0.3, 0.2, 0.3, 0.4, 0.5]),
               cost_vector=np.array([0.2, 0.1, 0.2])),
        Action(id="ask_advice", name="请教他人", action_type="primitive",
               value_vector=np.array([0.4, 0.5, 0.6, 0.5, 0.4, 0.4, 0.2, 0.3, 0.4, 0.5]),
               cost_vector=np.array([0.2, 0.0, 0.1])),
        Action(id="collaborate", name="协作配合", action_type="primitive",
               value_vector=np.array([0.6, 0.5, 0.7, 0.5, 0.4, 0.5, 0.3, 0.4, 0.5, 0.5]),
               cost_vector=np.array([0.3, 0.1, 0.2])),

        # 决策调整类
        Action(id="escalate", name="上报升级", action_type="primitive",
               value_vector=np.array([0.3, 0.3, 0.4, 0.7, 0.5, 0.4, 0.1, 0.1, 0.3, 0.3]),
               cost_vector=np.array([0.2, 0.0, 0.3])),
        Action(id="fallback", name="备用方案", action_type="primitive",
               value_vector=np.array([0.3, 0.5, 0.3, 0.4, 0.6, 0.4, 0.2, 0.3, 0.4, 0.4]),
               cost_vector=np.array([0.3, 0.1, 0.2])),
        Action(id="pivot", name="转变方向", action_type="primitive",
               value_vector=np.array([0.3, 0.8, 0.4, 0.3, 0.4, 0.5, 0.4, 0.6, 0.6, 0.5]),
               cost_vector=np.array([0.4, 0.1, 0.3])),
        Action(id="accept_situation", name="接受现状", action_type="primitive",
               value_vector=np.array([0.2, 0.3, 0.3, 0.5, 0.6, 0.3, 0.2, 0.2, 0.3, 0.4]),
               cost_vector=np.array([0.1, 0.0, 0.1])),
        Action(id="compromise", name="妥协让步", action_type="primitive",
               value_vector=np.array([0.4, 0.4, 0.5, 0.6, 0.5, 0.3, 0.2, 0.2, 0.3, 0.4]),
               cost_vector=np.array([0.2, 0.1, 0.2])),
        Action(id="take_control", name="掌控局面", action_type="primitive",
               value_vector=np.array([0.4, 0.6, 0.5, 0.5, 0.5, 0.7, 0.3, 0.4, 0.5, 0.5]),
               cost_vector=np.array([0.3, 0.0, 0.3])),
    ]

    # 交通/路线专用动作（针对出行、路线规划场景）
    TRAFFIC_ACTIONS = [
        Action(id="check_navigation", name="查看导航路线", action_type="primitive",
               value_vector=np.array([0.2, 0.7, 0.2, 0.2, 0.5, 0.4, 0.2, 0.4, 0.6, 0.5]),
               cost_vector=np.array([0.1, 0.0, 0.1]),
               prerequisites=["手机/车载导航设备"],
               outcomes=["获取可选路线列表", "查看实时路况"]),
        Action(id="call_service_hotline", name="拨打服务热线", action_type="primitive",
               value_vector=np.array([0.3, 0.4, 0.3, 0.4, 0.7, 0.2, 0.1, 0.1, 0.3, 0.4]),
               cost_vector=np.array([0.2, 0.0, 0.1]),
               prerequisites=["高速公路服务电话"],
               outcomes=["获取路况信息", "确认收费站状态"]),
        Action(id="find_detour", name="寻找绕行路线", action_type="primitive",
               value_vector=np.array([0.3, 0.8, 0.3, 0.2, 0.6, 0.5, 0.3, 0.6, 0.7, 0.5]),
               cost_vector=np.array([0.2, 0.0, 0.2]),
               prerequisites=["地图应用"],
               outcomes=["确定绕行路线", "估算额外时间"]),
        Action(id="wait_traffic", name="等待路况好转", action_type="primitive",
               value_vector=np.array([0.2, 0.2, 0.2, 0.3, 0.7, 0.2, 0.2, 0.1, 0.2, 0.2]),
               cost_vector=np.array([0.6, 0.0, 0.1]),
               prerequisites=["时间灵活性"],
               outcomes=["可能需要等待1-2小时"]),
        Action(id="change_route", name="改变路线", action_type="primitive",
               value_vector=np.array([0.3, 0.7, 0.3, 0.3, 0.5, 0.5, 0.3, 0.5, 0.6, 0.4]),
               cost_vector=np.array([0.3, 0.1, 0.3]),
               prerequisites=["多条路线可选"],
               outcomes=["新路线到达目的地"]),
        Action(id="check_traffic_info", name="查看路况信息", action_type="primitive",
               value_vector=np.array([0.2, 0.7, 0.2, 0.2, 0.5, 0.4, 0.2, 0.5, 0.6, 0.5]),
               cost_vector=np.array([0.1, 0.0, 0.1]),
               prerequisites=["网络连接"],
               outcomes=["实时路况", "拥堵情况"]),
        Action(id="use_public_transport", name="使用公共交通", action_type="primitive",
               value_vector=np.array([0.4, 0.5, 0.4, 0.4, 0.6, 0.3, 0.3, 0.3, 0.4, 0.5]),
               cost_vector=np.array([0.3, 0.2, 0.2]),
               prerequisites=["高铁/大巴选项"],
               outcomes=["无需开车", "可能更准时"]),
    ]

    def __init__(self):
        """初始化动作空间"""
        self.actions: Dict[str, Action] = {}
        self._init_actions()

    def _init_actions(self):
        """初始化所有动作"""
        for action in self.PRIMITIVE_ACTIONS + self.COMPOSITE_ACTIONS + self.EXTENDED_ACTIONS + self.TRAFFIC_ACTIONS:
            self.actions[action.id] = action

    def get_action(self, action_id: str) -> Optional[Action]:
        """获取动作"""
        return self.actions.get(action_id)

    def get_all_actions(self) -> List[Action]:
        """获取所有动作"""
        return list(self.actions.values())

    def get_primitive_actions(self) -> List[Action]:
        """获取所有原子动作"""
        return [a for a in self.actions.values() if a.action_type == "primitive"]

    def get_composite_actions(self) -> List[Action]:
        """获取所有复合动作"""
        return [a for a in self.actions.values() if a.action_type == "composite"]

    def get_top_k_actions(self, k: int = 15, query_keywords: List[str] = None) -> List[Action]:
        """获取前k个动作（支持语义排序）

        Args:
            k: 返回动作数量
            query_keywords: 问题关键词列表，用于语义排序

        Returns:
            按相关性排序的动作列表
        """
        all_actions = self.get_all_actions()

        # 如果有关键词，使用语义+价值混合排序
        if query_keywords:
            scored = []
            for a in all_actions:
                # 语义分数：动作名称/ID与关键词的匹配度
                semantic_score = self._compute_keyword_match(a, query_keywords)
                # 价值分数：归一化的value_vector范数
                value_score = np.linalg.norm(a.value_vector) / 3.0  # 归一化
                # 综合分数（语义权重更高）
                combined = semantic_score * 0.6 + value_score * 0.4
                scored.append((a, combined))
            scored.sort(key=lambda x: x[1], reverse=True)
        else:
            # 默认按价值向量范数排序
            scored = [(a, np.linalg.norm(a.value_vector)) for a in all_actions]
            scored.sort(key=lambda x: x[1], reverse=True)

        return [a for a, _ in scored[:k]]

    def _compute_keyword_match(self, action: Action, keywords: List[str]) -> float:
        """计算动作与关键词的匹配度

        Args:
            action: 动作
            keywords: 关键词列表

        Returns:
            匹配分数 [0, 1]
        """
        if not keywords:
            return 0.5

        # 动作名称和ID转换为小写
        action_text = (action.name + " " + action.id).lower()

        # 中文关键词匹配（支持单字符和词组匹配）
        chinese_keywords = [kw for kw in keywords if len(kw) >= 2]
        if not chinese_keywords:
            # 英文或短关键词
            keywords_lower = [kw.lower() for kw in keywords]
            match_count = sum(1 for kw in keywords_lower if kw in action_text)
            return min(1.0, match_count / max(len(keywords_lower), 1))

        # 检查中文词是否出现在动作文本中
        match_count = 0
        for kw in chinese_keywords:
            kw_lower = kw.lower()
            if kw_lower in action_text:
                match_count += 2  # 完整匹配权重更高
            else:
                # 检查字符重叠
                kw_chars = set(kw_lower)
                action_chars = set(action_text)
                overlap = len(kw_chars & action_chars)
                if overlap > 0 and len(kw_chars) > 0:
                    match_count += overlap / len(kw_chars)

        # 归一化
        max_possible = len(chinese_keywords) * 2
        return min(1.0, match_count / max_possible) if max_possible > 0 else 0.5

    def compute_action_similarity(self, action1: Action, action2: Action) -> float:
        """计算两个动作的相似度（余弦相似度）"""
        v1 = action1.value_vector
        v2 = action2.value_vector
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)

    def add_custom_action(self, action: Action):
        """添加自定义动作"""
        self.actions[action.id] = action

    def get_actions_by_type(self, action_type: str) -> List[Action]:
        """按类型获取动作"""
        return [a for a in self.actions.values() if a.action_type == action_type]


def create_action_space() -> ActionSpace:
    """工厂函数：创建动作空间"""
    return ActionSpace()