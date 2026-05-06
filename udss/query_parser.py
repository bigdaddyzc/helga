"""问题理解层 - QueryParser

将用户问题解析为环境向量E和查询向量K

公式:
    E = encode_time(10) + encode_space(10) + encode_social(10) + encode_info(10)
    K = encode_type(5) + encode_keywords(10) + encode_constraints(5)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import re


@dataclass
class QueryResult:
    """QueryParser输出结果"""
    environment_vector: np.ndarray  # 40维，环境向量E
    query_vector: np.ndarray       # 20维，查询向量K
    question_type: int             # 问题类型ID (0-4)
    keywords: List[str]            # 提取的关键词
    constraints: Dict[str, any]    # 约束条件


class QueryParser:
    """问题理解层

    将用户问题解析为结构化向量表示
    """

    # 问题类型定义
    QUESTION_TYPES = {
        0: "lifestyle",      # 生活决策
        1: "professional",   # 专业决策
        2: "urgent",         # 紧急决策
        3: "personal",       # 个人发展
        4: "social"          # 社交决策
    }

    # 关键词类别
    LIFESTYLE_KEYWORDS = ["旅行", "购物", "餐饮", "美食", "酒店", "机票", "餐厅", "旅游"]
    PROFESSIONAL_KEYWORDS = ["商业", "投资", "技术", "战略", "市场", "产品", "客户", "合同"]
    URGENT_KEYWORDS = ["紧急", "危机", "急救", "报警", "急诊", "马上", "立即"]
    PERSONAL_KEYWORDS = ["职业", "学习", "发展", "技能", "工作", "面试", "简历"]
    SOCIAL_KEYWORDS = ["人际", "关系", "朋友", "同事", "家人", "沟通", "矛盾"]

    def __init__(self):
        """初始化QueryParser"""
        self.type_keywords = {
            0: self.LIFESTYLE_KEYWORDS,
            1: self.PROFESSIONAL_KEYWORDS,
            2: self.URGENT_KEYWORDS,
            3: self.PERSONAL_KEYWORDS,
            4: self.SOCIAL_KEYWORDS
        }

    def parse(self, question: str, context: Optional[Dict] = None) -> QueryResult:
        """解析用户问题

        Args:
            question: 用户问题文本
            context: 可选上下文信息

        Returns:
            QueryResult包含环境向量和查询向量
        """
        # 检测问题类型
        question_type = self._detect_question_type(question)

        # 提取关键词
        keywords = self._extract_keywords(question, question_type)

        # 解析约束
        constraints = self._extract_constraints(question, context)

        # 构建环境向量E (40维)
        env_vector = self._build_environment_vector(question, context, question_type)

        # 构建查询向量K (20维)
        query_vector = self._build_query_vector(question, question_type, keywords, constraints)

        return QueryResult(
            environment_vector=env_vector,
            query_vector=query_vector,
            question_type=question_type,
            keywords=keywords,
            constraints=constraints
        )

    def _detect_question_type(self, question: str) -> int:
        """检测问题类型

        Args:
            question: 问题文本

        Returns:
            问题类型ID (0-4)
        """
        question_lower = question.lower()

        # 检查是否紧急
        for kw in self.URGENT_KEYWORDS:
            if kw in question_lower:
                return 2

        # 检查其他类型
        type_scores = {}
        for type_id, keywords in self.type_keywords.items():
            if type_id == 2:  # 紧急类型已处理
                continue
            score = sum(1 for kw in keywords if kw in question_lower)
            type_scores[type_id] = score

        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]

        # 默认返回专业决策
        return 1

    def _extract_keywords(self, question: str, question_type: int) -> List[str]:
        """提取关键词

        Args:
            question: 问题文本
            question_type: 问题类型

        Returns:
            关键词列表
        """
        keywords = []

        # 基于问题类型提取相关关键词
        type_kws = self.type_keywords.get(question_type, [])
        for kw in type_kws:
            if kw in question:
                keywords.append(kw)

        # 预定义的中文词语集合（从常见问题场景中提取）
        predefined_words = {
            # 交通出行
            "开车", "驾驶", "出行", "路线", "导航", "收费", "收费站",
            "封闭", "关闭", "堵车", "拥堵", "绕行", "高速", "国道",
            "回长沙", "回北京", "郴州", "长沙",
            # 生活决策
            "旅行", "酒店", "机票", "餐厅", "购物", "美食",
            # 工作决策
            "工作", "offer", "辞职", "面试", "加薪", "晋升",
            "投资", "商业", "合同", "客户", "产品",
            # 紧急情况
            "紧急", "危机", "急救", "报警", "急诊",
            # 个人发展
            "学习", "技能", "职业", "发展", "培训",
            # 社交
            "人际", "关系", "朋友", "同事", "家人", "沟通", "矛盾",
        }

        # 检查问题中包含哪些预定义词
        for word in predefined_words:
            if word in question:
                if word not in keywords:
                    keywords.append(word)

        # 通用停用词
        stop_words = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都",
                      "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
                      "没有", "看", "好", "自己", "这", "但", "从", "如何", "该", "明天",
                      "想", "可以", "应该", "会", "能", "把", "让", "给", "对", "于", "被"}

        # 使用字符级匹配提取有意义的2字词
        if len(keywords) < 5:
            for i in range(len(question)):
                if i + 2 <= len(question):
                    word = question[i:i+2]
                    if word not in stop_words and word not in keywords:
                        # 跳过全是数字或字母的词
                        if not word.isdigit() and not word.isalpha():
                            keywords.append(word)

        return list(set(keywords))[:15]

    def _extract_constraints(self, question: str, context: Optional[Dict]) -> Dict[str, any]:
        """提取约束条件

        Args:
            question: 问题文本
            context: 上下文信息

        Returns:
            约束字典
        """
        constraints = {}

        # 从上下文中提取
        if context:
            if "deadline" in context:
                constraints["deadline"] = context["deadline"]
            if "budget" in context:
                constraints["budget"] = context["budget"]
            if "location" in context:
                constraints["location"] = context["location"]

        # 从问题中提取
        # 时间约束
        time_indicators = ["明天", "下周", "月底", "尽快", "紧急", "今天"]
        for indicator in time_indicators:
            if indicator in question:
                constraints["time_pressure"] = indicator

        # 预算约束
        if "钱" in question or "预算" in question or "价格" in question:
            constraints["has_budget_constraint"] = True

        return constraints

    def _build_environment_vector(self, question: str, context: Optional[Dict], question_type: int) -> np.ndarray:
        """构建环境向量E (40维)

        维度分配:
            时间维度(10): 紧急度、时间范围、截止压力等
            空间维度(10): 位置类型、物理距离等
            社会维度(10): 权力距离、人际关系等
            信息维度(10): 不确定性、置信度等
        """
        # 时间维度 (10维)
        time_dim = np.zeros(10)
        if question_type == 2:  # 紧急决策
            time_dim[0] = 0.9  # 高紧急度
        else:
            time_dim[0] = 0.3

        time_dim[1] = 0.5  # 默认时间范围
        time_dim[2] = 0.5 if "deadline" in (context or {}) else 0.3

        # 空间维度 (10维)
        space_dim = np.zeros(10)
        if context and "location" in context:
            space_dim[0] = 0.7  # 有位置信息

        # 社会维度 (10维)
        social_dim = np.zeros(10)
        social_dim[0] = 0.5  # 默认人际关系

        # 信息维度 (10维)
        info_dim = np.zeros(10)
        info_dim[0] = 0.6  # 默认不确定性
        info_dim[1] = 0.5  # 默认置信度

        # 合并
        env_vector = np.concatenate([time_dim, space_dim, social_dim, info_dim])
        return env_vector.astype(np.float32)

    def _build_query_vector(self, question: str, question_type: int,
                           keywords: List[str], constraints: Dict) -> np.ndarray:
        """构建查询向量K (20维)

        维度分配:
            问题类型(5): one-hot编码
            关键词(10): 关键词嵌入
            约束(5): 约束编码
        """
        # 问题类型 (5维one-hot)
        type_dim = np.zeros(5)
        if 0 <= question_type < 5:
            type_dim[question_type] = 1.0

        # 关键词 (10维) - 简单哈希编码
        keyword_dim = np.zeros(10)
        for i, kw in enumerate(keywords[:10]):
            hash_val = sum(ord(c) for c in kw) % 10
            keyword_dim[hash_val] += 1.0
        keyword_dim = keyword_dim / (np.linalg.norm(keyword_dim) + 1e-8)

        # 约束 (5维)
        constraint_dim = np.zeros(5)
        if constraints.get("deadline"):
            constraint_dim[0] = 0.8
        if constraints.get("budget"):
            constraint_dim[1] = 0.7
        if constraints.get("location"):
            constraint_dim[2] = 0.6
        if constraints.get("time_pressure"):
            constraint_dim[3] = 0.9
        if constraints.get("has_budget_constraint"):
            constraint_dim[4] = 0.6

        # 合并
        query_vector = np.concatenate([type_dim, keyword_dim, constraint_dim])
        return query_vector.astype(np.float32)

    def get_type_name(self, type_id: int) -> str:
        """获取问题类型名称"""
        return self.QUESTION_TYPES.get(type_id, "unknown")