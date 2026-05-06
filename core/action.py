"""Action system with environment loop and online cultural adaptation for HELGA.

This module implements the action generation and environment dynamics.

Environment Update Formula:
    E_{t+1} = f_env(E_t, a_t) + ε
    where ε ~ N(0, noise_std²)

Cultural Adaptation Formula:
    params_updated = params - lr * gradient(error)

Attributes:
    noise_std: Environment noise standard deviation (0.01)
    learning_rate: Cultural adaptation learning rate (0.001)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from dataclasses import dataclass

from .types import (
    Action, ActionResult, EnvironmentState,
    ValidationResult, ValidationSeverity, ValidationIssue,
    HiddenState
)
from .culture import CulturalParameters

# Import new decision system components
from .ask_mechanism import AskTrigger, AskDecision, generate_question
from .rl_update import RLUpdate, UserFeedback


class EnvironmentModel(nn.Module):
    """Environment dynamics model.

    Formula: E_{t+1} = f_env(E_t, a_t) + ε
    where f_env is a simple linear transformation with noise.

    Attributes:
        env_dim: Environment dimension (10)
        action_dim: Action embedding dimension
    """

    def __init__(self, env_dim: int = 10, action_dim: int = 5):
        super().__init__()
        self.env_dim = env_dim

        # Environment transition network
        self.transition_net = nn.Sequential(
            nn.Linear(env_dim + action_dim, env_dim * 2),
            nn.ReLU(),
            nn.Linear(env_dim * 2, env_dim),
            nn.Tanh()  # Bounded output
        )

        # Noise standard deviation
        self.noise_std = 0.01

    def forward(
        self,
        current_env: torch.Tensor,
        action_emb: torch.Tensor,
        add_noise: bool = True
    ) -> torch.Tensor:
        """Compute next environment state.

        Args:
            current_env: Current environment state (batch, 10)
            action_emb: Action embedding (batch, action_dim)
            add_noise: Whether to add stochastic noise

        Returns:
            Next environment state (batch, 10)
        """
        # Concatenate current state and action
        inputs = torch.cat([current_env, action_emb], dim=-1)

        # Compute deterministic transition
        next_env = self.transition_net(inputs)

        # Add noise
        if add_noise:
            noise = torch.randn_like(next_env) * self.noise_std
            next_env = next_env + noise

        return next_env

    def predict(
        self,
        current_env: np.ndarray,
        action: Action
    ) -> np.ndarray:
        """Predict next environment state (numpy).

        Args:
            current_env: Current environment (10,)
            action: Action taken

        Returns:
            Predicted next environment (10,)
        """
        self.eval()
        with torch.no_grad():
            env_t = torch.from_numpy(current_env).float().unsqueeze(0)
            action_emb = self._embed_action(action.id)

            next_env = self.forward(env_t, action_emb, add_noise=False)
            return next_env.squeeze(0).numpy()

    def _embed_action(self, action_id: int) -> torch.Tensor:
        """Create action embedding.

        Simple one-hot embedding with learned modifications.
        """
        action_emb = torch.zeros(20)
        action_emb[action_id] = 1.0
        return action_emb.unsqueeze(0)  # (1, 20)


class ActionSystem:
    """Action generation and environment loop.

    Pipeline:
        1. Receive action from decision system
        2. Execute in environment (predict next state)
        3. Compute prediction error for culture adaptation
        4. Update environment state
        5. Trigger online learning if error > threshold

    Attributes:
        env_model: Environment dynamics model
        noise_std: Standard deviation of environment noise
        learning_rate: Learning rate for cultural adaptation
        hierarchical_decision: Hierarchical decision maker for probability output
        env_encoder: Environment encoder for hierarchical decision
        ask_trigger: Ask trigger mechanism
        rl_update: RL update mechanism for user feedback
    """

    def __init__(
        self,
        env_dim: int = 10,
        action_dim: int = 20,
        noise_std: float = 0.01,
        learning_rate: float = 0.001,
        adaptation_threshold: float = 0.05,
        use_hierarchical: bool = True,
        entropy_threshold: float = 1.0
    ):
        self.env_dim = env_dim
        self.noise_std = noise_std
        self.learning_rate = learning_rate
        self.adaptation_threshold = adaptation_threshold

        # Environment model
        self.env_model = EnvironmentModel(env_dim=env_dim, action_dim=action_dim)

        # Current environment state
        self.current_environment: Optional[EnvironmentState] = None

        # Prediction history for adaptation
        self.prediction_errors: list = []

        # New: Hierarchical decision system components
        self.use_hierarchical = use_hierarchical
        if use_hierarchical:
            # Initialize environment encoder
            self.env_encoder = None  # Set when processing

            # Initialize hierarchical decision maker
            from .decision import HierarchicalDecisionMaker
            self.hierarchical_decision = HierarchicalDecisionMaker(
                beta=1.0,
                gamma=0.5,
                value_dim=10,
                action_dim=action_dim,
                env_dim=32
            )

            # Initialize ask trigger and decision
            self.ask_trigger = AskTrigger()
            self.ask_decision = AskDecision(entropy_threshold=entropy_threshold)

            # Initialize RL update mechanism
            self.rl_update = RLUpdate()

            # Store last probability distribution for RL updates
            self.last_action_probs: Optional[torch.Tensor] = None

    def initialize_environment(self, initial_state: Optional[np.ndarray] = None) -> EnvironmentState:
        """Initialize environment state.

        Args:
            initial_state: Optional initial state (10,)
                           If None, use neutral state

        Returns:
            Initial EnvironmentState
        """
        if initial_state is None:
            initial_state = np.ones(self.env_dim, dtype=np.float32) * 0.5

        self.current_environment = EnvironmentState(
            state=initial_state,
            timestamp=0.0
        )
        return self.current_environment

    def execute_action(
        self,
        action: Action,
        cultural_params: Optional[CulturalParameters] = None,
        actual_next_env: Optional[np.ndarray] = None
    ) -> ActionResult:
        """Execute action and update environment.

        Args:
            action: Action to execute
            cultural_params: Cultural parameters (for adaptation)
            actual_next_env: Actual next environment (if known, for computing error)

        Returns:
            ActionResult with new environment and prediction error
        """
        if self.current_environment is None:
            self.initialize_environment()

        # Predict next state
        current_state = self.current_environment.state
        predicted_next = self.env_model.predict(current_state, action)

        # Compute prediction error if actual is provided
        prediction_error = 0.0
        if actual_next_env is not None:
            prediction_error = np.linalg.norm(predicted_next - actual_next_env)
            self.prediction_errors.append(prediction_error)

        # Create new environment state
        new_env_state = EnvironmentState(
            state=predicted_next,
            timestamp=self.current_environment.timestamp + 1.0
        )
        self.current_environment = new_env_state

        # Determine if adaptation should occur
        should_adapt = self.should_adapt_culture(prediction_error)

        return ActionResult(
            new_environment=new_env_state,
            prediction_error=prediction_error,
            success=True  # Simplified - always succeeds for now
        )

    def should_adapt_culture(
        self,
        prediction_error: float,
        threshold: Optional[float] = None
    ) -> bool:
        """Determine if cultural parameters should adapt.

        Args:
            prediction_error: Current prediction error
            threshold: Adaptation threshold (default: self.adaptation_threshold)

        Returns:
            True if adaptation should be triggered
        """
        if threshold is None:
            threshold = self.adaptation_threshold

        return prediction_error > threshold

    def compute_culture_adaptation(
        self,
        prediction_error: float,
        cultural_params: CulturalParameters
    ) -> Tuple[CulturalParameters, float]:
        """Compute cultural parameter adaptation.

        Formula: params_updated = params - lr * gradient(error)

        Args:
            prediction_error: Prediction error signal
            cultural_params: Current cultural parameters

        Returns:
            Tuple of (updated_params, adaptation_magnitude)
        """
        # Simplified gradient descent on Schwartz values
        gradient = prediction_error * self.learning_rate
        new_values = cultural_params.schwartz_values.values - gradient

        # Clamp to valid range
        new_values = np.clip(new_values, 0.0, 1.0)

        # Import and wrap properly
        from .culture import SchwartzValues

        # Recreate with proper classes
        actual_updated_params = CulturalParameters(
            schwartz_values=SchwartzValues(values=new_values),
            norms_matrix=cultural_params.norms_matrix,
            affect_schema=cultural_params.affect_schema
        )

        adaptation_magnitude = np.abs(prediction_error * self.learning_rate)

        return actual_updated_params, adaptation_magnitude

    def update_environment(
        self,
        action: Action,
        observed_next_state: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Update environment with observed state.

        Used for online learning when actual state differs from prediction.

        Args:
            action: Action taken
            observed_next_state: Actually observed next state

        Returns:
            Tuple of (updated_state, prediction_error)
        """
        if self.current_environment is None:
            self.initialize_environment()

        # Predict what we thought would happen
        predicted = self.env_model.predict(self.current_environment.state, action)

        # Compute error
        error = np.linalg.norm(predicted - observed_next_state)

        # Use actual state as new current (with small update toward predicted)
        # This creates a smoothing effect
        alpha = 0.9  # Trust actual observation more
        new_state = alpha * observed_next_state + (1 - alpha) * predicted

        self.current_environment = EnvironmentState(
            state=new_state,
            timestamp=self.current_environment.timestamp + 1
        )

        return new_state, error

    def get_environment_state(self) -> Optional[EnvironmentState]:
        """Get current environment state.

        Returns:
            Current EnvironmentState or None if not initialized
        """
        return self.current_environment

    def reset(self) -> None:
        """Reset action system to initial state."""
        self.current_environment = None
        self.prediction_errors = []

    def compute_probability_distribution(
        self,
        hidden_state: HiddenState,
        env_state: np.ndarray,
        cultural_params: CulturalParameters
    ) -> torch.Tensor:
        """Compute probability distribution over actions.

        Uses hierarchical decision maker to output probability distribution
        instead of single action.

        Args:
            hidden_state: Current hidden state
            env_state: Environment state vector
            cultural_params: Cultural parameters

        Returns:
            Probability distribution over actions (20,)
        """
        if not self.use_hierarchical or self.hierarchical_decision is None:
            # Fallback: uniform distribution
            probs = torch.ones(20) / 20
            self.last_action_probs = probs
            return probs

        # Encode environment if encoder is set
        if self.env_encoder is not None:
            # Split env_state into components (time, space, social, info)
            time = torch.from_numpy(env_state[:4] if len(env_state) >= 4 else np.zeros(4)).float()
            space = torch.from_numpy(env_state[4:7] if len(env_state) >= 7 else np.zeros(3)).float()
            social = torch.from_numpy(env_state[7:11] if len(env_state) >= 11 else np.zeros(4)).float()
            info = torch.from_numpy(env_state[11:15] if len(env_state) >= 15 else np.zeros(4)).float()

            env_encoding = self.env_encoder(time, space, social, info)
        else:
            # Use raw env_state as encoding
            env_encoding = torch.from_numpy(env_state[:32] if len(env_state) >= 32 else np.pad(env_state, (0, 32-len(env_state)))).float()

        # Create cultural prior (from norms matrix)
        cultural_prior = torch.ones(20)
        if cultural_params is not None and cultural_params.norms_matrix is not None:
            for i in range(20):
                cultural_prior[i] = float(cultural_params.norms_matrix.avg_compliance(i))
        cultural_prior = cultural_prior / cultural_prior.sum()

        # Get hidden state as tensor
        hidden_tensor = torch.from_numpy(hidden_state.intent).float()

        # Compute probability distribution
        probs = self.hierarchical_decision(
            hidden_tensor,
            env_encoding,
            cultural_prior
        )

        self.last_action_probs = probs
        return probs

    def get_top_actions(self, probs, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """Get top-k actions with probabilities.

        Args:
            probs: Probability distribution (tensor or numpy array)
            top_k: Number of top actions to return

        Returns:
            List of (action_id, probability, action_name) tuples
        """
        from .decision import ACTION_NAMES

        if not isinstance(probs, torch.Tensor):
            probs = torch.from_numpy(probs)

        values, indices = torch.topk(probs, min(top_k, len(probs)))
        return [(idx.item(), val.item(), ACTION_NAMES[idx.item()]) for idx, val in zip(indices, values)]

    def check_ask_trigger(
        self,
        probs: torch.Tensor,
        missing_key_info: bool = False,
        threshold: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """Check if ask should be triggered based on entropy.

        Args:
            probs: Action probability distribution
            missing_key_info: Whether key info is missing
            threshold: Optional entropy threshold override

        Returns:
            (should_ask, question_or_none)
        """
        if threshold is None:
            threshold = self.ask_decision.entropy_threshold

        return self.ask_decision.decide(probs, missing_key_info)

    def apply_rl_feedback(
        self,
        feedback: UserFeedback,
        hidden_state: Optional[HiddenState] = None,
        pos_embedding: Optional[torch.Tensor] = None,
        neg_embedding: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Apply RL update based on user feedback.

        Args:
            feedback: User feedback with action index and score
            hidden_state: Optional hidden state for supervised loss
            pos_embedding: Optional positive embedding for contrastive
            neg_embedding: Optional negative embedding for contrastive

        Returns:
            (total_loss, loss_breakdown)
        """
        if self.last_action_probs is None:
            raise ValueError("No previous action probabilities for RL update")

        # Parse rationale to get preferred action if mentioned
        preferred_probs = None
        from .decision import ACTION_NAMES
        pos_idx, neg_idx = self.rl_update.parse_rationale(feedback.rationale, ACTION_NAMES)

        if pos_idx is not None:
            # Create preferred distribution with higher probability on preferred action
            preferred_probs = torch.ones(20) * 0.01
            preferred_probs[pos_idx] = 0.95

        # Compute total loss
        total_loss, loss_breakdown = self.rl_update.compute_total_loss(
            self.last_action_probs,
            feedback,
            preferred_probs=preferred_probs,
            pos_embedding=pos_embedding,
            neg_embedding=neg_embedding
        )

        return total_loss, loss_breakdown

    def get_entropy(self, probs: torch.Tensor) -> float:
        """Compute entropy of probability distribution.

        H(P) = -Σ_a P(a) · log P(a)
        """
        return AskTrigger.compute_entropy(probs)


class ActionVerification:
    """Verification for action system."""

    @staticmethod
    def verify_action_safety(
        action: Action,
        cultural_params: CulturalParameters,
        current_state: EnvironmentState
    ) -> ValidationResult:
        """Verify if action is safe given current cultural parameters.

        Args:
            action: Action to verify
            cultural_params: Cultural parameters
            current_state: Current environment state

        Returns:
            ValidationResult with safety assessment
        """
        issues = []
        recommendations = []

        # Check norm compliance
        norm_compliance = cultural_params.norms_matrix.avg_compliance(action.id)
        if norm_compliance < 0.3:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                description=f"Action '{action.name}' has low norm compliance ({norm_compliance:.2f})",
                location="norm_check",
                fix_suggestion="Consider alternative action with higher norm compliance"
            ))
            recommendations.append("Choose action with norm compliance > 0.5")

        # Check value alignment
        # Simplified: just check if action is in acceptable range
        if action.id in [4, 11, 13]:  # ignore, deny, reject - potentially problematic
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                description=f"Action '{action.name}' may require additional justification",
                location="value_check",
                fix_suggestion="Ensure action aligns with core values before execution"
            ))

        passed = len([i for i in issues if i.severity == ValidationSeverity.CRITICAL]) == 0

        return ValidationResult(
            passed=passed,
            layer=None,  # Will be set by caller
            metric_value=norm_compliance,
            threshold=0.3,
            issues=issues,
            recommendations=recommendations,
            explanation=f"Action safety check for '{action.name}': norm compliance = {norm_compliance:.2f}"
        )


def create_action_system(
    env_dim: int = 10,
    action_dim: int = 20,
    noise_std: float = 0.01,
    learning_rate: float = 0.001,
    use_hierarchical: bool = True,
    entropy_threshold: float = 1.0
) -> ActionSystem:
    """Factory function to create action system.

    Args:
        env_dim: Environment dimension (10)
        action_dim: Action dimension (20)
        noise_std: Environment noise (0.01)
        learning_rate: Cultural adaptation learning rate (0.001)
        use_hierarchical: Whether to use hierarchical decision (True)
        entropy_threshold: Threshold for ask trigger (1.0)

    Returns:
        Configured ActionSystem instance
    """
    return ActionSystem(
        env_dim=env_dim,
        action_dim=action_dim,
        noise_std=noise_std,
        learning_rate=learning_rate,
        use_hierarchical=use_hierarchical,
        entropy_threshold=entropy_threshold
    )


def simulate_environment_step(
    current_state: np.ndarray,
    action_id: int,
    noise_std: float = 0.01
) -> np.ndarray:
    """Simple environment simulation without neural network.

    Simplified dynamics: E_{t+1} = 0.9 * E_t + 0.1 * action_effect + noise

    Args:
        current_state: Current environment state (10,)
        action_id: Action taken (0-19)
        noise_std: Noise standard deviation

    Returns:
        Next environment state (10,)
    """
    # Action effect (simplified one-hot with decay)
    action_effect = np.zeros(10)
    action_effect[action_id % 10] = 0.3  # Map to environment dims

    # Simple linear dynamics
    next_state = 0.85 * current_state + 0.15 * action_effect

    # Add noise
    noise = np.random.randn(10) * noise_std
    next_state = next_state + noise

    # Clamp to valid range
    next_state = np.clip(next_state, 0, 1)

    return next_state.astype(np.float32)