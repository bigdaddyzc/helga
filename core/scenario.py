"""Scenario encoding module for HELGA.

Maps natural language scenarios to structured environment states.
"""

from typing import Dict, Tuple
import numpy as np


# Environment state dimensions (10,)
# E_t = [urgency, importance, time_constraint, cost_benefit, risk_level,
#        relationship_relevance, authority_level, trust_level, autonomy, familiarity]

# Event stream dimensions (64,)
# Event types: 0=schedule, 1=late, 2=urgent, 3=conflict, 4=approval,
#              5=colleague, 6=traffic, 7=weather, 8=technical, 9=financial...


class ScenarioEncoder:
    """Encode natural language scenarios into environment states."""

    def __init__(self):
        # Scenario templates: maps keywords to (env_state, event_stream) patterns
        self.scenario_templates = {
            # ============ TRAVEL/SCENARIOS ============
            "travel": {
                "keywords": ["开车", "开车去", "驾车", "出行", "回家", "旅程", "高速", "收费站", "堵车", "traffic", "drive", "car", "highway", "toll"],
                "env_state": [0.7, 0.6, 0.8, 0.5, 0.3, 0.0, 0.0, 0.0, 0.6, 0.8],
                "event_stream": {6: 0.9},  # traffic event
            },
            "road_blocked": {
                "keywords": ["关闭", "封闭", "堵塞", "封路", "不通", "blocked", "closed", "closed"],
                "env_state": [0.8, 0.7, 0.9, 0.4, 0.5, 0.0, 0.0, 0.0, 0.8, 0.7],
                "event_stream": {6: 0.9, 7: 0.3},  # traffic + weather
            },
            "waiting": {
                "keywords": ["等待", "等候", "等等", "wait"],
                "env_state": [0.3, 0.4, 0.5, 0.6, 0.2, 0.0, 0.0, 0.0, 0.2, 0.5],
                "event_stream": {},
            },

            # ============ WORKPLACE SCENARIOS ============
            "colleague_late": {
                "keywords": ["迟到", "晚到", "来不及", "late"],
                "env_state": [0.6, 0.7, 0.7, 0.5, 0.4, 0.7, 0.4, 0.7, 0.5, 0.8],
                "event_stream": {1: 0.9, 5: 0.6},  # late event + colleague context
            },
            "urgent_work": {
                "keywords": ["紧急", "急", "迫切", "urgent", "important"],
                "env_state": [0.9, 0.9, 0.9, 0.6, 0.5, 0.0, 0.0, 0.0, 0.7, 0.6],
                "event_stream": {2: 0.9},  # urgent event
            },
            "conflict": {
                "keywords": ["冲突", "矛盾", "争执", "conflict", "dispute"],
                "env_state": [0.5, 0.6, 0.4, 0.3, 0.7, 0.8, 0.5, 0.3, 0.4, 0.6],
                "event_stream": {3: 0.9},  # conflict event
            },
            "meeting": {
                "keywords": ["会议", "开会", "meeting", "schedule"],
                "env_state": [0.5, 0.6, 0.6, 0.5, 0.4, 0.5, 0.4, 0.5, 0.4, 0.7],
                "event_stream": {0: 0.8},  # schedule event
            },

            # ============ FINANCIAL SCENARIOS ============
            "financial": {
                "keywords": ["钱", "费用", "价格", "成本", "经济", "financial", "money", "cost", "price"],
                "env_state": [0.5, 0.7, 0.6, 0.8, 0.4, 0.0, 0.0, 0.0, 0.5, 0.6],
                "event_stream": {9: 0.9},  # financial event
            },

            # ============ TECHNICAL SCENARIOS ============
            "technical": {
                "keywords": ["系统", "软件", "故障", "bug", "技术", "technical", "system", "error"],
                "env_state": [0.7, 0.6, 0.8, 0.4, 0.6, 0.0, 0.0, 0.0, 0.6, 0.5],
                "event_stream": {8: 0.9},  # technical event
            },
        }

    def encode(self, scenario: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode scenario description to states.

        Args:
            scenario: Natural language scenario description

        Returns:
            Tuple of (env_state, event_stream, other_states)
        """
        scenario_lower = scenario.lower()

        # Initialize with defaults (random-like but consistent)
        env_state = np.random.rand(10).astype(np.float32) * 0.3 + 0.35
        event_stream = np.zeros(64, dtype=np.float32)
        other_states = np.random.rand(10).astype(np.float32) * 0.3 + 0.35

        # Match scenario templates
        matched_templates = []
        for template_name, template in self.scenario_templates.items():
            for keyword in template["keywords"]:
                if keyword in scenario_lower:
                    matched_templates.append(template_name)
                    break

        # Apply matched template effects (weighted by number of matches)
        if matched_templates:
            # Average the env_state from all matched templates
            env_sum = np.zeros(10, dtype=np.float32)
            for template_name in matched_templates:
                template = self.scenario_templates[template_name]
                env_sum += np.array(template["env_state"], dtype=np.float32)
            env_state = env_sum / len(matched_templates)

            # Apply event streams from all matched templates
            for template_name in matched_templates:
                template = self.scenario_templates[template_name]
                for event_idx, event_val in template["event_stream"].items():
                    event_stream[event_idx] = max(event_stream[event_idx], event_val)

        # Special handling for specific entities
        # Changsha (长沙) - destination
        if "长沙" in scenario or "changsha" in scenario_lower:
            other_states[0] = 0.8  # destination known

        # Cenzhou (郴州) - origin
        if "郴州" in scenario or "chenzhou" in scenario_lower:
            other_states[1] = 0.8  # origin known

        return env_state, event_stream, other_states


def create_scenario_encoder() -> ScenarioEncoder:
    """Factory function to create scenario encoder."""
    return ScenarioEncoder()
