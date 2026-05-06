"""Action parameterizer for HELGA.

Converts abstract actions into detailed, executable action plans.
"""

from typing import Dict, Any, Optional, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.knowledge import (
    ActionPlan, ActionStep, BaseKnowledge,
    TrafficKnowledge, WorkplaceKnowledge,
    create_knowledge_base, get_knowledge_for_action
)


class ActionParameterizer:
    """Parameterizes abstract actions into detailed action plans.

    Pipeline:
        1. Receive abstract action and scenario context
        2. Find appropriate knowledge base
        3. Generate detailed action plan
        4. Return ActionPlan with specific steps
    """

    def __init__(self):
        """Initialize parameterizer with knowledge bases."""
        self.knowledge_base = create_knowledge_base()

    def parameterize(
        self,
        action_name: str,
        scenario: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ActionPlan]:
        """Convert abstract action to detailed action plan.

        Args:
            action_name: Name of the action (e.g., "take_detour")
            scenario: Scenario description
            context: Additional context information

        Returns:
            Detailed ActionPlan or None if not supported
        """
        context = context or {}

        # Find knowledge base for this action
        knowledge = get_knowledge_for_action(action_name, self.knowledge_base)

        if knowledge is None:
            return self._default_plan(action_name, scenario)

        # Generate plan from knowledge
        plan = knowledge.parameterize(action_name, scenario, context)

        if plan is None:
            return self._default_plan(action_name, scenario)

        return plan

    def _default_plan(self, action_name: str, scenario: str) -> ActionPlan:
        """Create a generic plan when no specialized knowledge exists."""
        return ActionPlan(
            title=f"执行{action_name}",
            action_name=action_name,
            steps=[
                ActionStep(
                    step_number=1,
                    description=f"根据场景'{scenario}'执行{action_name}",
                    details={}
                )
            ],
            time_estimate="待定",
            resources={},
            alternatives=[],
            reasoning=f"通用动作方案，适用于{action_name}"
        )


def create_parameterizer() -> ActionParameterizer:
    """Factory function to create parameterizer."""
    return ActionParameterizer()
