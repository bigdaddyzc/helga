"""Decision mapper: maps atomic actions to scene-level decisions.

This module implements the mapping from atomic action probabilities to
scene-level decision probabilities, enabling user choice at the decision level.

Scene Decision Probability Formula:
    P(scene_decision_i) = Σ_{a ∈ scene_decision_i.atomic_actions} P(atomic_action_a)
    Normalized: P(scene_decision_i) = P(scene_decision_i) / Σ_j P(scene_decision_j)
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from .types import SceneDecision, SceneDecisionOutput, ReasoningChain, ReasoningStep
from . import decision


# Predefined scene decision mapping architectures
SCENE_DECISION_ARCHITECTURES = {
    "traffic": {
        "原地等待": {
            "atomic_actions": ["wait", "wait_traffic_clear"],
            "value_orientation": {
                "security": 0.9,
                "tradition": 0.8,
                "conformity": 0.3
            },
            "estimated_outcome": {
                "time_impact": "high",
                "cost": 0,
                "risk": "low"
            },
            "description": "在原地等待交通状况改善"
        },
        "绕路行驶": {
            "atomic_actions": ["take_detour", "check_info"],
            "value_orientation": {
                "self_direction": 0.9,
                "achievement": 0.8,
                "openness": 0.7
            },
            "estimated_outcome": {
                "time_impact": "medium",
                "cost": "medium",
                "risk": "medium"
            },
            "description": "选择替代路线绕过拥堵路段"
        },
        "取消行程": {
            "atomic_actions": ["cancel_trip", "call_service"],
            "value_orientation": {
                "security": 0.9,
                "self_preservation": 0.8
            },
            "estimated_outcome": {
                "time_impact": "none",
                "cost": "lost_time",
                "risk": "none"
            },
            "description": "取消当前出行计划"
        },
        "继续前进": {
            "atomic_actions": ["continue_anyway", "check_info"],
            "value_orientation": {
                "risk_taking": 0.9,
                "stimulation": 0.7
            },
            "estimated_outcome": {
                "time_impact": "unknown",
                "cost": "potential_delay",
                "risk": "high"
            },
            "description": "无视拥堵继续按原路线行驶"
        }
    },
    "social": {
        "等待观察": {
            "atomic_actions": ["wait", "ask_reason"],
            "value_orientation": {
                "patience": 0.9,
                "openness": 0.7,
                "universalism": 0.6
            },
            "estimated_outcome": {
                "time_impact": "medium",
                "cost": 0,
                "risk": "low"
            },
            "description": "等待更多信息再做决定"
        },
        "主动帮助": {
            "atomic_actions": ["provide_help", "send_reminder"],
            "value_orientation": {
                "benevolence": 0.9,
                "achievement": 0.6
            },
            "estimated_outcome": {
                "time_impact": "medium",
                "cost": "effort",
                "risk": "low"
            },
            "description": "主动提供帮助或提醒"
        },
        "委托代理": {
            "atomic_actions": ["delegate", "reschedule"],
            "value_orientation": {
                "collaboration": 0.9,
                "flexibility": 0.7
            },
            "estimated_outcome": {
                "time_impact": "low",
                "cost": "delegation_overhead",
                "risk": "medium"
            },
            "description": "委托他人处理"
        },
        "忽略应对": {
            "atomic_actions": ["ignore", "continue_anyway"],
            "value_orientation": {
                "independence": 0.9,
                "risk_taking": 0.6
            },
            "estimated_outcome": {
                "time_impact": "none",
                "cost": 0,
                "risk": "social"
            },
            "description": "忽略该问题继续其他事情"
        }
    },
    "work": {
        "等待确认": {
            "atomic_actions": ["wait", "confirm"],
            "value_orientation": {
                "reliability": 0.9,
                "patience": 0.8
            },
            "estimated_outcome": {
                "time_impact": "medium",
                "cost": 0,
                "risk": "low"
            },
            "description": "等待进一步确认再做决定"
        },
        "批准执行": {
            "atomic_actions": ["approve", "send_reminder"],
            "value_orientation": {
                "achievement": 0.8,
                "conformity": 0.7
            },
            "estimated_outcome": {
                "time_impact": "low",
                "cost": "approval_overhead",
                "risk": "low"
            },
            "description": "批准提案并执行"
        },
        "拒绝否决": {
            "atomic_actions": ["reject", "deny"],
            "value_orientation": {
                "security": 0.8,
                "conformity": 0.7
            },
            "estimated_outcome": {
                "time_impact": "none",
                "cost": "potential_relationship_loss",
                "risk": "medium"
            },
            "description": "拒绝或否决该提案"
        },
        "修改调整": {
            "atomic_actions": ["modify", "reschedule"],
            "value_orientation": {
                "adaptability": 0.9,
                "problem_solving": 0.8
            },
            "estimated_outcome": {
                "time_impact": "high",
                "cost": "iteration",
                "risk": "low"
            },
            "description": "修改提案后重新提交"
        }
    },
    "emergency": {
        "紧急等待": {
            "atomic_actions": ["wait", "call_service"],
            "value_orientation": {
                "caution": 0.9,
                "security": 0.9
            },
            "estimated_outcome": {
                "time_impact": "unknown",
                "cost": "potential_delay",
                "risk": "high"
            },
            "description": "原地等待紧急救援"
        },
        "立即撤离": {
            "atomic_actions": ["cancel_trip", "continue_anyway"],
            "value_orientation": {
                "self_preservation": 0.9,
                "risk_taking": 0.8
            },
            "estimated_outcome": {
                "time_impact": "immediate",
                "cost": "incomplete_task",
                "risk": "physical"
            },
            "description": "立即撤离危险区域"
        },
        "寻求帮助": {
            "atomic_actions": ["call_service", "ask_reason"],
            "value_orientation": {
                "security": 0.9,
                "social_connection": 0.7
            },
            "estimated_outcome": {
                "time_impact": "medium",
                "cost": "time",
                "risk": "medium"
            },
            "description": "拨打紧急电话或寻求帮助"
        }
    },
    "default": {
        "等待观察": {
            "atomic_actions": ["wait"],
            "value_orientation": {
                "patience": 0.9
            },
            "estimated_outcome": {
                "time_impact": "medium",
                "cost": 0,
                "risk": "low"
            },
            "description": "等待情况明朗再做决定"
        },
        "主动行动": {
            "atomic_actions": ["provide_help", "escalate"],
            "value_orientation": {
                "proactivity": 0.9,
                "responsibility": 0.8
            },
            "estimated_outcome": {
                "time_impact": "low",
                "cost": "effort",
                "risk": "medium"
            },
            "description": "主动采取行动"
        },
        "暂不处理": {
            "atomic_actions": ["ignore", "cancel"],
            "value_orientation": {
                "independence": 0.9
            },
            "estimated_outcome": {
                "time_impact": "none",
                "cost": 0,
                "risk": "potential"
            },
            "description": "暂时不处理该事务"
        }
    }
}


class DecisionMapper:
    """Maps atomic actions to scene-level decisions.

    This class implements the hierarchical decision mapping from atomic
    actions (20 predefined) to scene-level decisions, computing probability
    distributions over scene decisions for user choice.

    Attributes:
        architectures: Predefined mapping architectures per context type
        action_name_to_idx: Mapping from action name to index
    """

    def __init__(self):
        self.architectures = SCENE_DECISION_ARCHITECTURES
        self.action_name_to_idx = {
            name: idx for idx, name in enumerate(decision.ACTION_NAMES)
        }

    def _get_architecture(self, context_type: str) -> Dict[str, Dict]:
        """Get mapping architecture for context type."""
        return self.architectures.get(
            context_type,
            self.architectures.get("default", {})
        )

    def _get_default_architecture(self) -> Dict[str, Dict]:
        """Get default mapping architecture."""
        return self.architectures.get("default", {})

    def map_atomic_probs_to_scene_probs(
        self,
        atomic_probs: torch.Tensor,
        context_type: str
    ) -> Tuple[List[str], List[float], List[Dict]]:
        """Map atomic action probabilities to scene decision probabilities.

        Args:
            atomic_probs: Probability distribution over 20 atomic actions
            context_type: Context type for selecting mapping architecture

        Returns:
            Tuple of (scene_names, scene_probs, scene_configs)
        """
        architecture = self._get_architecture(context_type)

        if not architecture:
            architecture = self._get_default_architecture()

        scene_names = []
        scene_probs = []
        scene_configs = []

        for scene_name, config in architecture.items():
            # Sum probabilities of constituent atomic actions
            scene_prob = 0.0
            for atomic_action in config["atomic_actions"]:
                action_idx = self.action_name_to_idx.get(atomic_action)
                if action_idx is not None and action_idx < len(atomic_probs):
                    scene_prob += atomic_probs[action_idx].item()

            scene_names.append(scene_name)
            scene_probs.append(scene_prob)
            scene_configs.append(config)

        # Normalize probabilities
        total_prob = sum(scene_probs)
        if total_prob > 0:
            scene_probs = [p / total_prob for p in scene_probs]
        else:
            # Fallback: uniform distribution
            scene_probs = [1.0 / len(scene_names)] * len(scene_names)

        return scene_names, scene_probs, scene_configs

    def create_scene_decisions(
        self,
        scene_names: List[str],
        scene_probs: List[float],
        scene_configs: List[Dict],
        recommended_action: str
    ) -> List[SceneDecision]:
        """Create SceneDecision objects from mapping results.

        Args:
            scene_names: List of scene decision names
            scene_probs: List of corresponding probabilities
            scene_configs: List of scene configuration dicts
            recommended_action: Recommended atomic action

        Returns:
            List of SceneDecision objects
        """
        scene_decisions = []

        for name, prob, config in zip(scene_names, scene_probs, scene_configs):
            # Find which atomic action is recommended for this scene
            if recommended_action in config.get("atomic_actions", []):
                rec_action = recommended_action
            else:
                rec_action = config["atomic_actions"][0] if config["atomic_actions"] else "wait"

            scene_decision = SceneDecision(
                id=name,
                name=name,
                description=config.get("description", ""),
                atomic_actions=config.get("atomic_actions", []),
                value_orientation=config.get("value_orientation", {}),
                estimated_outcome=config.get("estimated_outcome", {})
            )
            scene_decisions.append(scene_decision)

        return scene_decisions

    def get_scene_decision_output(
        self,
        atomic_probs: torch.Tensor,
        context_type: str,
        reasoning_chain: ReasoningChain,
        recommended_action: str,
        context_summary: Dict[str, Any]
    ) -> SceneDecisionOutput:
        """Generate complete scene decision output.

        Args:
            atomic_probs: Atomic action probabilities (20,)
            context_type: Context type for architecture selection
            reasoning_chain: Existing reasoning chain from decision system
            recommended_action: Recommended atomic action
            context_summary: Summary of decision context

        Returns:
            SceneDecisionOutput with scene decisions and probabilities
        """
        # Map to scene probabilities
        scene_names, scene_probs, scene_configs = self.map_atomic_probs_to_scene_probs(
            atomic_probs, context_type
        )

        # Create scene decisions
        scene_decisions = self.create_scene_decisions(
            scene_names, scene_probs, scene_configs, recommended_action
        )

        return SceneDecisionOutput(
            scene_decisions=scene_decisions,
            probabilities=scene_probs,
            reasoning_chain=reasoning_chain,
            recommended_action=recommended_action,
            context_summary=context_summary
        )

    def get_top_scene_decisions(
        self,
        atomic_probs: torch.Tensor,
        context_type: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top-k scene decisions by probability.

        Args:
            atomic_probs: Atomic action probabilities
            context_type: Context type for architecture
            top_k: Number of top decisions to return

        Returns:
            List of (scene_name, probability) tuples
        """
        scene_names, scene_probs, _ = self.map_atomic_probs_to_scene_probs(
            atomic_probs, context_type
        )

        # Sort by probability descending
        sorted_decisions = sorted(
            zip(scene_names, scene_probs),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_decisions[:top_k]


# Global instance for convenience
_decision_mapper_instance: Optional[DecisionMapper] = None


def get_decision_mapper() -> DecisionMapper:
    """Get global DecisionMapper instance."""
    global _decision_mapper_instance
    if _decision_mapper_instance is None:
        _decision_mapper_instance = DecisionMapper()
    return _decision_mapper_instance
