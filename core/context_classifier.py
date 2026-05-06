"""Context classifier: determines the context type for decision mapping.

This module classifies input scenarios into context types to select
appropriate decision mapping architectures.
"""

from typing import Dict, Any, Optional, List


# Keywords for context type classification
CONTEXT_KEYWORDS = {
    "traffic": [
        "road", "highway", "traffic", " congestion", "堵车", "高速", "路线",
        "driving", "car", "vehicle", "route", "地图", "导航", "拥堵",
        "accident", "closure", "detour", "绕路", "等待路况"
    ],
    "social": [
        "friend", "family", "social", "relationship", "朋友", "家人",
        "聚会", "邀请", "沟通", "交流", "约会", "meeting", "party",
        "conversation", "discussion"
    ],
    "work": [
        "work", "task", "project", "meeting", "deadline", "任务", "项目",
        "会议", "同事", "上司", "下属", "approve", "reject", "proposal",
        "报告", "邮件", "办公室"
    ],
    "emergency": [
        "emergency", "danger", "urgent", "紧急", "危险", "报警",
        "fire", "accident", "医疗", "急救", "事故", "安全"
    ]
}


class ContextClassifier:
    """Classifies input scenarios into context types.

    Uses keyword matching and pattern recognition to determine
    the context type for decision mapping architecture selection.

    Attributes:
        context_types: List of supported context types
        keywords: Keyword mappings per context type
    """

    def __init__(self):
        self.context_types: List[str] = list(CONTEXT_KEYWORDS.keys())
        self.keywords: Dict[str, List[str]] = CONTEXT_KEYWORDS

    def classify(self, scenario_input: Dict[str, Any]) -> str:
        """Classify the context type from scenario input.

        Args:
            scenario_input: Scenario dict that may contain:
                - text: Natural language scenario description
                - scenario_type: Explicit scenario type hint
                - context: Additional context info

        Returns:
            Context type string ("traffic", "social", "work", "emergency", "default")
        """
        # Check for explicit scenario type hint
        if "scenario_type" in scenario_input:
            explicit_type = scenario_input["scenario_type"].lower()
            if explicit_type in self.context_types:
                return explicit_type

        # Check for context hint
        if "context" in scenario_input:
            context = scenario_input["context"]
            if isinstance(context, str):
                context_lower = context.lower()
                for ctx_type in self.context_types:
                    if ctx_type in context_lower:
                        return ctx_type

        # Check text content for keywords
        text = self._extract_text(scenario_input)
        if text:
            return self._classify_by_keywords(text)

        return "default"

    def _extract_text(self, scenario_input: Dict[str, Any]) -> str:
        """Extract text content from scenario input."""
        text_parts = []

        # Natural language description
        if "text" in scenario_input:
            text_parts.append(str(scenario_input["text"]))
        elif "description" in scenario_input:
            text_parts.append(str(scenario_input["description"]))
        elif "scenario" in scenario_input:
            text_parts.append(str(scenario_input["scenario"]))

        # Action hints
        if "action" in scenario_input:
            text_parts.append(str(scenario_input["action"]))

        return " ".join(text_parts).lower()

    def _classify_by_keywords(self, text: str) -> str:
        """Classify based on keyword matching.

        Args:
            text: Lowercase text to search

        Returns:
            Best matching context type
        """
        scores: Dict[str, float] = {ctx: 0.0 for ctx in self.context_types}

        for ctx_type, keywords in self.keywords.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    scores[ctx_type] += 1.0

        # Return type with highest score, or default
        if max(scores.values()) > 0:
            return max(scores.items(), key=lambda x: x[1])[0]

        return "default"

    def get_classification_confidence(
        self,
        scenario_input: Dict[str, Any]
    ) -> float:
        """Get classification confidence score.

        Args:
            scenario_input: Scenario input

        Returns:
            Confidence score 0-1
        """
        text = self._extract_text(scenario_input)
        if not text:
            return 0.5

        scores = {}
        for ctx_type, keywords in self.keywords.items():
            scores[ctx_type] = sum(1.0 for k in keywords if k.lower() in text)

        max_score = max(scores.values()) if scores else 0
        if max_score == 0:
            return 0.5

        # Normalize: higher score = higher confidence
        return min(0.5 + (max_score * 0.1), 1.0)


# Global instance for convenience
_context_classifier_instance: Optional[ContextClassifier] = None


def get_context_classifier() -> ContextClassifier:
    """Get global ContextClassifier instance."""
    global _context_classifier_instance
    if _context_classifier_instance is None:
        _context_classifier_instance = ContextClassifier()
    return _context_classifier_instance
