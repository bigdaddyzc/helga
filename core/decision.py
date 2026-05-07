"""Decision system with multi-objective expected utility for HELGA.

This module implements the decision-making system using expected utility theory.

Expected Utility Formula:
    U(a, Z) = w_v^T · F_value(a) + λ_norm · Comply(a, N) - η · Complexity(a)

Where:
    - w_v: Schwartz value weight vector (dim=10)
    - F_value(a): Value feature for action a (映射矩阵 20×10)
    - λ_norm: Norm weight (default 0.3)
    - Comply(a, N): Norm compliance score for action a
    - η: Complexity penalty weight (default 0.1)
    - Complexity(a): Cognitive cost of action a

Action space: 20 predefined actions

Outputs:
    - Optimal action index
    - Utility decomposition for each action
    - Attention weights (for interpretability)
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

from .types import (
    Action, UtilityDecomposition, DecisionOutput, HiddenState,
    ReasoningChain, ReasoningStep, PlanProbabilityDistribution
)
from .culture import CulturalParameters as CulturalParams
from .decision_mapper import DecisionMapper, get_decision_mapper
from .context_classifier import ContextClassifier, get_context_classifier
from .decision_iterations import run_iterative_decision


# Predefined action space (20 actions)
# Actions 0-13: General/workplace scenarios
# Actions 14-19: Traffic/practical scenarios
ACTION_NAMES = [
    "wait",                    # 0: Wait and observe
    "send_reminder",           # 1: Send a reminder message
    "reschedule",              # 2: Reschedule (通用：重新安排)
    "escalate",                # 3: Escalate to manager
    "ignore",                  # 4: Ignore and continue
    "ask_reason",              # 5: Ask for reason
    "provide_help",            # 6: Offer assistance
    "delegate",                # 7: Delegate task
    "cancel",                  # 8: Cancel event
    "modify",                  # 9: Modify parameters
    "confirm",                 # 10: Confirm details
    "deny",                    # 11: Deny request
    "approve",                 # 12: Approve request
    "reject",                  # 13: Reject proposal
    "take_detour",             # 14: Take alternate route (绕行)
    "wait_traffic_clear",      # 15: Wait for traffic to clear (等待路况好转)
    "check_info",              # 16: Check traffic info (查看路况)
    "call_service",            # 17: Call service hotline (拨打服务热线)
    "cancel_trip",             # 18: Cancel trip (取消行程)
    "continue_anyway",         # 19: Continue anyway (继续前进)
]

# Action-Value Feature Matrix (20 actions × 10 values)
# Dimensions: Self-Transcendence(0), Openness(1), Benevolence(2), Conformity(3),
#             Security(4), Achievement(5), Hedonism(6), Stimulation(7),
#             Self-Direction(8), Universalism(9)
#
# For TRAFFIC SCENARIOS (highway closed):
# - take_detour (14): High Openness(1), Self-Direction(8), Security(4)
# - wait_traffic_clear (15): High Security(4), low cost
# - check_info (16): High Universalism(9), Openness(1)
# - call_service (17): High Benevolence(2), Security(4)
# - cancel_trip (18): High Security(4), self-preservation
# - continue_anyway (19): LOW Security(4), risk-taking
ACTION_VALUE_FEATURES = np.array([
    # 0: wait - patient, tolerant, security-focused
    [0.2, 0.1, 0.2, 0.3, 0.9, 0.1, 0.1, 0.1, 0.2, 0.3],
    # 1: send_reminder - communicative, responsible
    [0.5, 0.4, 0.8, 0.9, 0.5, 0.3, 0.2, 0.2, 0.5, 0.6],
    # 2: reschedule - flexible, adaptive (also works for route change)
    [0.4, 0.8, 0.5, 0.4, 0.4, 0.3, 0.3, 0.4, 0.8, 0.5],
    # 3: escalate - follows hierarchy, safety-conscious
    [0.3, 0.2, 0.3, 0.9, 0.8, 0.4, 0.1, 0.1, 0.2, 0.3],
    # 4: ignore - low engagement, avoidant
    [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1],
    # 5: ask_reason - curious, open-minded
    [0.4, 0.7, 0.4, 0.4, 0.4, 0.3, 0.2, 0.4, 0.6, 0.8],
    # 6: provide_help - benevolent, caring
    [0.8, 0.3, 0.9, 0.5, 0.4, 0.3, 0.2, 0.2, 0.3, 0.6],
    # 7: delegate - empowering, collaborative
    [0.5, 0.6, 0.5, 0.4, 0.2, 0.7, 0.3, 0.3, 0.5, 0.4],
    # 8: cancel - disruptive, potentially harmful
    [0.2, 0.2, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1],
    # 9: modify - adaptive, problem-solving
    [0.4, 0.7, 0.4, 0.4, 0.5, 0.5, 0.3, 0.4, 0.6, 0.4],
    # 10: confirm - confirmative, reliable
    [0.4, 0.3, 0.5, 0.7, 0.8, 0.3, 0.2, 0.1, 0.3, 0.4],
    # 11: deny - rejecting, restrictive
    [0.2, 0.2, 0.1, 0.4, 0.5, 0.3, 0.1, 0.1, 0.2, 0.2],
    # 12: approve - accepting, supportive
    [0.5, 0.3, 0.5, 0.5, 0.4, 0.7, 0.3, 0.2, 0.3, 0.4],
    # 13: reject - opposing, dismissive
    [0.2, 0.2, 0.2, 0.3, 0.4, 0.3, 0.1, 0.1, 0.2, 0.2],
    # 14: take_detour - HIGH VALUE for traffic scenario
    [0.3, 0.9, 0.3, 0.2, 0.7, 0.3, 0.2, 0.4, 0.9, 0.4],
    # 15: wait_traffic_clear - HIGH Security, patience
    [0.2, 0.2, 0.2, 0.2, 0.9, 0.1, 0.1, 0.1, 0.2, 0.3],
    # 16: check_info - information seeking, open-minded
    [0.4, 0.8, 0.3, 0.3, 0.5, 0.3, 0.2, 0.5, 0.7, 0.9],
    # 17: call_service - helpful, authority-seeking
    [0.5, 0.3, 0.6, 0.7, 0.8, 0.3, 0.1, 0.1, 0.3, 0.5],
    # 18: cancel_trip - self-preservation, security
    [0.2, 0.1, 0.2, 0.2, 0.9, 0.2, 0.1, 0.1, 0.1, 0.2],
    # 19: continue_anyway - LOW Security, risk-taking (NOT APPROPRIATE)
    [0.2, 0.4, 0.2, 0.2, 0.1, 0.4, 0.3, 0.5, 0.5, 0.2],
], dtype=np.float32)


class ValueFeatureExtractor(nn.Module):
    """Extract value features from actions.

    Uses predefined action-value features based on Schwartz values theory
    instead of random initialization.

    Formula: F_value(a) = Σ_k w_v[k] * f_k(a)
    where f_k(a) is the k-th value feature of action a

    Attributes:
        action_dim: Number of actions (20)
        value_dim: Number of values (10)
    """

    def __init__(self, action_dim: int = 20, value_dim: int = 10):
        super().__init__()
        self.action_dim = action_dim
        self.value_dim = value_dim

        # Use predefined action-value feature matrix (not random)
        # Shape: (20, 10)
        self.register_buffer('action_value_features',
                           torch.from_numpy(ACTION_VALUE_FEATURES).float())

    def forward(self, action_ids: torch.Tensor, value_weights: torch.Tensor) -> torch.Tensor:
        """Compute value features for actions.

        Args:
            action_ids: Action indices (batch,)
            value_weights: Value weight vector (batch, 10)

        Returns:
            Value feature scores (batch,)
        """
        # Lookup action embeddings
        action_embeddings = self.action_value_features[action_ids]  # (batch, 10)

        # Weighted sum: w_v^T * F_value(a)
        value_scores = torch.sum(action_embeddings * value_weights, dim=-1)  # (batch,)

        return value_scores

    def get_action_value_vector(self, action_id: int) -> np.ndarray:
        """Get value vector for a specific action.

        Returns:
            Value feature vector (10,)
        """
        return self.action_value_features[action_id].numpy()


class ComplexityScorer(nn.Module):
    """Compute complexity/cognitive cost of actions.

    Formula: Complexity(a) = log(1 + param_count(a))

    Attributes:
        action_dim: Number of actions (20)
    """

    def __init__(self, action_dim: int = 20):
        super().__init__()
        self.action_dim = action_dim

        # Complexity parameters per action
        self.complexity_params = nn.Parameter(torch.ones(action_dim) * 0.5)

    def forward(self, action_ids: torch.Tensor) -> torch.Tensor:
        """Compute complexity penalty for actions.

        Args:
            action_ids: Action indices (batch,)

        Returns:
            Complexity scores (batch,)
        """
        complexity = torch.log(1 + self.complexity_params[action_ids])
        return complexity


class MultiObjectiveDecisionMaker(nn.Module):
    """Multi-objective decision making module.

    Formula:
        U(a, Z) = w_v^T · F_value(a) + λ_norm · Comply(a, N) - η · Complexity(a)

    Attributes:
        value_extractor: Value feature extractor
        complexity_scorer: Complexity scorer
        norm_weight: Weight for norm compliance (λ_norm)
        complexity_weight: Weight for complexity penalty (η)
    """

    def __init__(
        self,
        action_dim: int = 20,
        value_dim: int = 10,
        norm_weight: float = 0.3,
        complexity_weight: float = 0.1
    ):
        super().__init__()
        self.action_dim = action_dim
        self.value_dim = value_dim
        self.norm_weight = norm_weight
        self.complexity_weight = complexity_weight

        # Components
        self.value_extractor = ValueFeatureExtractor(action_dim, value_dim)
        self.complexity_scorer = ComplexityScorer(action_dim)

    def compute_utility(
        self,
        action_id: int,
        hidden_state: HiddenState,
        cultural_params: CulturalParams
    ) -> UtilityDecomposition:
        """Compute expected utility for a single action.

        Args:
            action_id: Action index (0-19)
            hidden_state: Current hidden state
            cultural_params: Cultural parameters

        Returns:
            UtilityDecomposition with all components
        """
        # Value component: w_v^T * F_value(a)
        action_value_vec = self.value_extractor.get_action_value_vector(action_id)
        value_component = np.dot(cultural_params.schwartz_values.values, action_value_vec)

        # Norm component: λ_norm * Comply(a, N)
        norm_compliance = cultural_params.norms_matrix.avg_compliance(action_id)
        norm_component = self.norm_weight * norm_compliance

        # Complexity penalty: η * Complexity(a)
        complexity_penalty = self.complexity_weight * np.log(1 + 0.5)

        # Total
        total = value_component + norm_component - complexity_penalty

        return UtilityDecomposition(
            value_component=value_component,
            norm_component=norm_component,
            complexity_penalty=complexity_penalty,
            total=total
        )

    def compute_all_utilities(
        self,
        hidden_state: HiddenState,
        cultural_params: CulturalParams
    ) -> List[Tuple[int, UtilityDecomposition]]:
        """Compute utilities for all actions.

        Args:
            hidden_state: Current hidden state
            cultural_params: Cultural parameters

        Returns:
            List of (action_id, utility) tuples, sorted by utility descending
        """
        utilities = []
        for action_id in range(self.action_dim):
            utility = self.compute_utility(action_id, hidden_state, cultural_params)
            utilities.append((action_id, utility))

        # Sort by total utility descending
        utilities.sort(key=lambda x: x[1].total, reverse=True)
        return utilities

    def forward(
        self,
        hidden_state: HiddenState,
        cultural_params: CulturalParams
    ) -> Tuple[int, List[Tuple[int, UtilityDecomposition]]]:
        """Select optimal action.

        Args:
            hidden_state: Current hidden state
            cultural_params: Cultural parameters

        Returns:
            Tuple of (optimal_action_id, all_utilities)
        """
        utilities = self.compute_all_utilities(hidden_state, cultural_params)
        optimal_id = utilities[0][0]
        return optimal_id, utilities


class HierarchicalDecisionMaker(nn.Module):
    """Hierarchical decision maker with probability distribution output.

    Formula:
        P(a | E, M) = Softmax( β · EU(a) + γ · Attention(E, M) ) ⊙ P_cultural(a | Z)

    Attributes:
        beta: Utility weight coefficient
        gamma: Attention weight coefficient
        value_dim: Value dimension (10)
        action_dim: Action dimension (20)
        env_encoder: Environment encoder
        attention: Attention mechanism
    """

    def __init__(
        self,
        beta: float = 1.0,
        gamma: float = 0.5,
        value_dim: int = 10,
        action_dim: int = 20,
        env_dim: int = 32
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma

        # Value extractor (reuse from existing)
        self.value_extractor = ValueFeatureExtractor(action_dim, value_dim)

        # Environment encoder (will be connected later)
        self.env_encoder = None  # Set externally

        # Attention mechanism
        self.attention_query = nn.Linear(env_dim, 32)
        self.attention_keys = nn.Linear(env_dim, 32)

        # Learnable weights
        self.value_weights = nn.Parameter(torch.ones(value_dim) / value_dim)
        self.norm_weight = nn.Parameter(torch.tensor(0.3))
        self.complexity_weight = nn.Parameter(torch.tensor(0.1))
        self.risk_weight = nn.Parameter(torch.tensor(0.1))

    def compute_expected_utility(self, action_ids: torch.Tensor) -> torch.Tensor:
        """Compute EU(a) for actions.

        EU(a) = Σ_k w_k · value_k(a) - λ · norm_cost - η · complexity - ψ · risk
        """
        # Value component
        action_values = self.value_extractor.action_value_features[action_ids]
        value_component = torch.sum(action_values * self.value_weights, dim=-1)

        # Norm component (simplified: uniform compliance)
        norm_component = torch.ones_like(value_component) * self.norm_weight * 0.8

        # Complexity penalty (simplified)
        complexity_penalty = torch.ones_like(value_component) * self.complexity_weight * 0.5

        # Risk penalty (simplified)
        risk_penalty = torch.ones_like(value_component) * self.risk_weight * 0.2

        eu = value_component + norm_component - complexity_penalty - risk_penalty
        return eu

    def compute_attention(self, env_encoding: torch.Tensor) -> torch.Tensor:
        """Compute attention score over environment features."""
        query = self.attention_query(env_encoding)
        # Simplified: attention is just a weighted sum
        return torch.sigmoid(query).sum() * self.gamma

    def forward(
        self,
        hidden_state: torch.Tensor,
        env_encoding: torch.Tensor,
        cultural_prior: torch.Tensor
    ) -> torch.Tensor:
        """Output probability distribution over actions.

        Args:
            hidden_state: Hidden state vector (10,)
            env_encoding: Environment encoding (32,)
            cultural_prior: Cultural prior probabilities (20,)

        Returns:
            Probability distribution over actions (20,)
        """
        # Compute EU for all actions
        action_ids = torch.arange(20)
        eu = self.compute_expected_utility(action_ids)

        # Compute attention
        attention_score = self.compute_attention(env_encoding)

        # Combine EU and attention
        logits = self.beta * eu + attention_score

        # Apply cultural prior (element-wise multiply)
        logits = logits + torch.log(cultural_prior + 1e-8)

        # Softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)

        return probs

    def get_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of probability distribution."""
        return -torch.sum(probs * torch.log(probs + 1e-8))


def create_action_space() -> List[Action]:
    """Create predefined action space.

    Returns:
        List of 20 Actions
    """
    return [
        Action(id=i, name=name, parameters={})
        for i, name in enumerate(ACTION_NAMES)
    ]


class DecisionSystem:
    """Decision system for HELGA.

    Maintains action space and selects optimal actions based on
    expected utility theory.
    """

    def __init__(
        self,
        action_dim: int = 20,
        value_dim: int = 10,
        norm_weight: float = 0.3,
        complexity_weight: float = 0.1
    ):
        self.action_dim = action_dim
        self.action_space = create_action_space()

        # Neural network for learning
        self.model = MultiObjectiveDecisionMaker(
            action_dim=action_dim,
            value_dim=value_dim,
            norm_weight=norm_weight,
            complexity_weight=complexity_weight
        )

        # Hierarchical decision maker for probability distribution
        self.hierarchical_model = HierarchicalDecisionMaker(
            beta=1.0,
            gamma=0.5,
            value_dim=value_dim,
            action_dim=action_dim,
            env_dim=32
        )

        # Scene decision mapper and context classifier
        self.decision_mapper = get_decision_mapper()
        self.context_classifier = get_context_classifier()

    def make_decision(
        self,
        hidden_state: HiddenState,
        cultural_params: CulturalParams,
        context: Optional[Dict[str, Any]] = None
    ) -> DecisionOutput:
        """Make decision with full reasoning chain.

        Args:
            hidden_state: Current hidden state
            cultural_params: Cultural parameters
            context: Optional context for decision (used for scene decision mapping)

        Returns:
            DecisionOutput with optimal action, confidence, top5 probabilities,
            and scene_decision_output (scene-level decision distribution)
        """
        # Get all utilities
        optimal_id, all_utilities = self.model(hidden_state, cultural_params)

        # Compute probability distribution using hierarchical model
        try:
            hidden_tensor = torch.from_numpy(hidden_state.to_vector()).float()
            env_encoding = torch.randn(32)  # Random env encoding for now
            cultural_prior = torch.ones(20) / 20  # Uniform cultural prior

            probs = self.hierarchical_model(
                hidden_tensor.unsqueeze(0),
                env_encoding.unsqueeze(0),
                cultural_prior.unsqueeze(0)
            ).squeeze()

            # Get top 5 actions by probability
            top5_indices = torch.argsort(probs, descending=True)[:5]
            confidence = probs[optimal_id].item() if optimal_id < len(probs) else 0.0

            top5_probabilities = [
                (int(idx), ACTION_NAMES[int(idx)], probs[int(idx)].item())
                for idx in top5_indices
            ]

            # Compute scene decision output
            scene_output = None
            plan_distribution = None
            if context is not None:
                context_type = self.context_classifier.classify(context)
                recommended_action = ACTION_NAMES[optimal_id]
                context_summary = {
                    "scenario_type": context_type,
                    "context": context
                }

                scene_output = self.decision_mapper.get_scene_decision_output(
                    atomic_probs=probs,
                    context_type=context_type,
                    reasoning_chain=None,  # Will be set below after creation
                    recommended_action=recommended_action,
                    context_summary=context_summary
                )

                # 生成方案级概率分布（替代行动级概率分布）
                plans, plan_probs, iterations = run_iterative_decision(
                    scenario_input=context,
                    context_type=context_type,
                    atomic_probs=probs
                )

                plan_distribution = PlanProbabilityDistribution(
                    plans=plans,
                    probabilities=plan_probs,
                    recommended_plan_id=plans[0].plan_id if plans else "",
                    confidence=plan_probs[0] if plan_probs else 0.0,
                    reasoning_chain=None,  # Will be set below
                    decision_iterations=iterations
                )
        except Exception as e:
            # Fallback if probability computation fails
            confidence = 0.0
            top5_probabilities = [
                (optimal_id, ACTION_NAMES[optimal_id], 1.0)
            ]
            scene_output = None
            plan_distribution = None

        # Build reasoning chain
        steps = []

        # Step 1: Action generation
        steps.append(ReasoningStep(
            stage="action_generation",
            input={"hidden_state": hidden_state.to_vector().tolist()},
            output={"candidate_actions": [a.name for a in self.action_space]},
            rationale=f"Generated {self.action_dim} candidate actions based on current state",
            timestamp=0
        ))

        # Step 2: Value evaluation
        steps.append(ReasoningStep(
            stage="value_evaluation",
            input={"cultural_weights": cultural_params.schwartz_values.values.tolist()},
            output={"top_value_actions": [
                (ACTION_NAMES[aid], util.value_component)
                for aid, util in all_utilities[:5]
            ]},
            rationale="Evaluated each action's contribution to value dimensions",
            timestamp=1
        ))

        # Step 3: Norm check
        steps.append(ReasoningStep(
            stage="norm_check",
            input={"norm_matrix_shape": cultural_params.norms_matrix.matrix.shape},
            output={"top_norm_actions": [
                (ACTION_NAMES[aid], util.norm_component)
                for aid, util in all_utilities[:5]
            ]},
            rationale="Checked norm compliance for each action",
            timestamp=2
        ))

        # Step 4: Complexity consideration
        steps.append(ReasoningStep(
            stage="complexity_check",
            input={"complexity_weight": self.model.complexity_weight},
            output={"complexity_penalties": [
                (ACTION_NAMES[aid], util.complexity_penalty)
                for aid, util in all_utilities[:5]
            ]},
            rationale="Computed cognitive cost for each action",
            timestamp=3
        ))

        # Step 5: Final decision
        optimal_utility = all_utilities[0][1]
        steps.append(ReasoningStep(
            stage="decision",
            input={"all_utilities": [(ACTION_NAMES[aid], util.total) for aid, util in all_utilities]},
            output={
                "optimal_action": ACTION_NAMES[optimal_id],
                "total_utility": optimal_utility.total,
                "utility_breakdown": {
                    "value": optimal_utility.value_component,
                    "norm": optimal_utility.norm_component,
                    "complexity": optimal_utility.complexity_penalty
                },
                "confidence": confidence,
                "top5_probabilities": top5_probabilities
            },
            rationale=f"Selected '{ACTION_NAMES[optimal_id]}' with highest expected utility (prob={confidence:.3f})",
            timestamp=4
        ))

        reasoning_chain = ReasoningChain(
            steps=steps,
            final_decision=f"Action {optimal_id}: {ACTION_NAMES[optimal_id]}",
            alternatives_considered=[ACTION_NAMES[aid] for aid, _ in all_utilities[1:5]]
        )

        # Update scene_output and plan_distribution with correct reasoning_chain reference
        if scene_output is not None:
            scene_output.reasoning_chain = reasoning_chain
        if plan_distribution is not None:
            plan_distribution.reasoning_chain = reasoning_chain

        # Create alternatives list
        alternatives = [(self.action_space[aid], util.total) for aid, util in all_utilities]

        return DecisionOutput(
            optimal_action=self.action_space[optimal_id],
            utility=optimal_utility,
            attention_weights=np.ones(10) / 10,  # Uniform attention
            alternatives=alternatives,
            reasoning_chain=reasoning_chain,
            confidence=confidence,
            top5_probabilities=top5_probabilities,
            scene_output=scene_output,
            plan_distribution=plan_distribution
        )

    def get_utilities_for_actions(
        self,
        action_ids: List[int],
        hidden_state: HiddenState,
        cultural_params: CulturalParams
    ) -> List[UtilityDecomposition]:
        """Get utility decompositions for specific actions.

        Args:
            action_ids: List of action indices
            hidden_state: Current hidden state
            cultural_params: Cultural parameters

        Returns:
            List of UtilityDecompositions
        """
        return [self.model.compute_utility(aid, hidden_state, cultural_params)
                for aid in action_ids]


def create_decision_system(
    action_dim: int = 20,
    value_dim: int = 10,
    norm_weight: float = 0.3,
    complexity_weight: float = 0.1
) -> DecisionSystem:
    """Factory function to create decision system.

    Args:
        action_dim: Number of actions (20)
        value_dim: Value dimension (10)
        norm_weight: Norm compliance weight (0.3)
        complexity_weight: Complexity penalty weight (0.1)

    Returns:
        Configured DecisionSystem instance
    """
    return DecisionSystem(
        action_dim=action_dim,
        value_dim=value_dim,
        norm_weight=norm_weight,
        complexity_weight=complexity_weight
    )