"""HELGA - Main entry point.

This module provides the main HELGAAgent class and CLI interface.

Example usage:
    python main.py --scenario colleague_late
    python main.py --run-validation
    python main.py --interactive
"""

import argparse
import sys
import os
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.types import (
    ObservationInput, PerceptionOutput, HiddenState, EmotionState,
    MotivationVector, Action, DecisionOutput, EnvironmentState, ActionResult,
    ReasoningChain
)
from core.culture import CulturalParameters, SchwartzValues, SocialNormsMatrix, AffectSchema
from core.perception import compute_synthetic_observation, create_perception_system
from core.cognition import create_cognition_system
from core.decision import create_decision_system, DecisionSystem
from core.action import create_action_system, ActionSystem, simulate_environment_step
from core.rl_update import UserFeedback
from validation import run_all_validations, print_validation_summary


@dataclass
class HELGAAgent:
    """Main HELGA agent combining all cognitive modules.

    Pipeline:
        1. perception(observation) -> O_t
        2. cognition(O_t, Z_{t-1}) -> Z_t, M_t
        3. decision(Z_t, M_t) -> optimal_action
        4. action(optimal_action) -> result
        5. return decision with reasoning chain

    Attributes:
        perception: Perception system
        cognition: Cognition system
        decision: Decision system
        action: Action system
        cultural_params: Cultural parameters
    """

    perception: Optional[Any] = None  # Can be neural or synthetic
    cognition: Any = None
    decision: DecisionSystem = None
    action: ActionSystem = None
    cultural_params: CulturalParameters = None

    def __init__(
        self,
        config_path: Optional[str] = None,
        use_neural_perception: bool = False
    ):
        """Initialize HELGA agent.

        Args:
            config_path: Path to config YAML file
            use_neural_perception: Whether to use neural perception (requires PyTorch)
        """
        # Load or create cultural parameters
        if config_path:
            self.cultural_params = CulturalParameters.from_config(config_path)
        else:
            self.cultural_params = CulturalParameters.default()

        # Initialize cognition
        self.cognition = create_cognition_system(
            hidden_dim=10,
            obs_dim=32,
            num_particles=100,
            inference_method="particle_filter"
        )

        # Initialize decision system
        self.decision = create_decision_system(
            action_dim=20,
            value_dim=10,
            norm_weight=0.3,
            complexity_weight=0.1
        )

        # Initialize action system
        self.action = create_action_system(
            env_dim=10,
            action_dim=20,
            noise_std=0.01,
            learning_rate=0.001
        )

        # Perception (simplified synthetic version)
        self.use_neural = use_neural_perception
        if use_neural_perception:
            self.perception = create_perception_system()

        # Initialize cognition with cultural parameters
        self.cognition.initialize(self.cultural_params)

    def perceive(
        self,
        env_state: np.ndarray,
        event_stream: np.ndarray,
        other_states: np.ndarray
    ) -> np.ndarray:
        """Process perception.

        Args:
            env_state: Environment vector (10,)
            event_stream: Event stream (64,)
            other_states: Other states (10,)

        Returns:
            Observation vector (32,)
        """
        if self.use_neural and self.perception is not None:
            # Neural perception
            from core.perception import PerceptionSystem
            if isinstance(self.perception, PerceptionSystem):
                result = self.perception.encode_numpy(
                    env_state, event_stream, other_states, self.cultural_params
                )
                return result.observation
        else:
            # Synthetic perception (simplified)
            return compute_synthetic_observation(
                env_state, event_stream, other_states, self.cultural_params
            )

    def reason(
        self,
        observation: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> DecisionOutput:
        """Run reasoning pipeline.

        Args:
            observation: Perception output (32,)
            context: Optional context dict

        Returns:
            DecisionOutput with action and reasoning chain
        """
        # Update cognition with observation
        hidden_state, emotion, motivation = self.cognition.step(
            observation, self.cultural_params
        )

        # Make decision
        decision = self.decision.make_decision(hidden_state, self.cultural_params, context)

        return decision

    def execute(
        self,
        action: Action,
        actual_next_env: Optional[np.ndarray] = None
    ) -> ActionResult:
        """Execute action.

        Args:
            action: Action to execute
            actual_next_env: Actual next environment (for computing error)

        Returns:
            ActionResult
        """
        result = self.action.execute_action(action, self.cultural_params, actual_next_env)

        # Check for cultural adaptation
        if result.prediction_error > self.action.adaptation_threshold:
            updated_params, magnitude = self.action.compute_culture_adaptation(
                result.prediction_error, self.cultural_params
            )
            self.cultural_params = updated_params

        return result

    def step(
        self,
        env_state: np.ndarray,
        event_stream: np.ndarray,
        other_states: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Single agent step.

        Args:
            env_state: Environment vector (10,)
            event_stream: Event stream (64,)
            other_states: Other states (10,)
            context: Optional context

        Returns:
            Dict with observation, hidden_state, emotion, decision, action_result
        """
        # Perception
        observation = self.perceive(env_state, event_stream, other_states)

        # Reasoning
        hidden_state, emotion, motivation = self.cognition.step(
            observation, self.cultural_params
        )

        # Decision
        decision = self.decision.make_decision(hidden_state, self.cultural_params, context)

        # Execution
        action_result = self.action.execute_action(
            decision.optimal_action, self.cultural_params
        )

        return {
            "observation": observation,
            "hidden_state": hidden_state,
            "emotion": emotion,
            "motivation": motivation,
            "decision": decision,
            "action_result": action_result
        }

    def get_reasoning_chain(self, decision: DecisionOutput) -> ReasoningChain:
        """Get reasoning chain for display."""
        return decision.reasoning_chain

    def run_scenario(self, scenario_desc: str) -> Dict[str, Any]:
        """Run custom scenario from description.

        Args:
            scenario_desc: Scenario description text

        Returns:
            Scenario results dict
        """
        # Use ScenarioEncoder for better context understanding
        from core.scenario import create_scenario_encoder
        encoder = create_scenario_encoder()
        env_state, event_stream, other_states = encoder.encode(scenario_desc)

        result = self.step(env_state, event_stream, other_states, context={'text': scenario_desc})

        # Generate detailed action plan
        from core.action_parameterizer import create_parameterizer
        parameterizer = create_parameterizer()
        action_plan = parameterizer.parameterize(
            result['decision'].optimal_action.name,
            scenario_desc,
            {
                'env_state': env_state,
                'event_stream': event_stream,
                'other_states': other_states
            }
        )

        result['action_plan'] = action_plan

        # 更新 final_decision 为详细方案
        if action_plan is not None:
            decision = result['decision']
            decision.reasoning_chain.final_decision = (
                f"Action {decision.optimal_action.id}: {decision.optimal_action.name} | "
                f"方案: {action_plan.title} | "
                f"步骤: {len(action_plan.steps)} 步 | "
                f"预计: {action_plan.time_estimate} | "
                f"理由: {action_plan.reasoning[:80]}..."
            )

        return result

    def process_feedback(
        self,
        action_idx: int,
        score: float,
        rationale: str
    ) -> Dict[str, Any]:
        """Process user feedback and update model.

        Args:
            action_idx: Index of the action feedback is about
            score: User score (1-5)
            rationale: User's reasoning

        Returns:
            Dict with feedback processing results
        """
        feedback = UserFeedback(
            action_idx=action_idx,
            score=score,
            rationale=rationale
        )

        # Get current probability distribution
        if self.action.last_action_probs is None:
            return {"error": "No previous action probabilities to update"}

        # Apply RL update
        try:
            loss, breakdown = self.action.apply_rl_feedback(feedback)
            return {
                "feedback_processed": True,
                "action_idx": action_idx,
                "score": score,
                "loss": loss.item(),
                "loss_breakdown": breakdown
            }
        except Exception as e:
            return {"error": str(e), "feedback_processed": False}

    def get_probability_distribution(
        self,
        hidden_state,
        env_state: np.ndarray
    ) -> np.ndarray:
        """Get probability distribution over actions.

        Args:
            hidden_state: Current hidden state
            env_state: Environment state vector

        Returns:
            Probability distribution (20,)
        """
        probs = self.action.compute_probability_distribution(
            hidden_state, env_state, self.cultural_params
        )
        return probs.detach().numpy()

    def check_ask_needed(
        self,
        probs: Optional[np.ndarray] = None,
        missing_key_info: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """Check if ask should be triggered.

        Args:
            probs: Optional probability distribution (uses last if None)
            missing_key_info: Whether key info is missing

        Returns:
            (should_ask, question_or_none)
        """
        if probs is None:
            if self.action.last_action_probs is not None:
                probs = self.action.last_action_probs
            else:
                return False, None

        import torch
        if not isinstance(probs, torch.Tensor):
            probs = torch.from_numpy(probs)

        return self.action.check_ask_trigger(probs, missing_key_info)


def run_colleague_late_scenario(agent: HELGAAgent) -> Dict[str, Any]:
    """Run 'colleague is late' scenario.

    Scenario:
        - Input: "colleague is late for meeting"
        - Expected behaviors: wait, send_reminder, reschedule, escalate

    Args:
        agent: HELGAAgent instance

    Returns:
        Dict with scenario results
    """
    print("\n" + "=" * 60)
    print("SCENARIO: Colleague is Late for Meeting")
    print("=" * 60)

    # Setup environment state
    # E_t = [meeting_time, current_time, wait_duration, importance, punctuality,
    #        colleague_relation, trust_level, power_distance, task_urgency, context_familiarity]
    env_state = np.array([0.8, 0.3, 0.0, 0.7, 0.5, 0.6, 0.7, 0.4, 0.6, 0.9], dtype=np.float32)

    # Event encoding for "colleague_late"
    # V_t = 64-dim embedding (simplified as one-hot + noise)
    event_stream = np.zeros(64, dtype=np.float32)
    event_stream[1] = 0.9  # "late" event type
    event_stream[5] = 0.6  # workplace context

    # Other states: colleague info
    # S_t_other = [colleague_ontime_rate, relationship_strength, authority, ...]
    other_states = np.array([0.4, 0.7, 0.5, 0.6, 0.5, 0.4, 0.6, 0.5, 0.5, 0.5], dtype=np.float32)

    print("\n[1] Perceiving environment...")
    observation = agent.perceive(env_state, event_stream, other_states)
    print(f"    Observation vector (dim={len(observation)}): {observation[:5].tolist()}...")

    print("\n[2] Reasoning about situation...")
    hidden_state, emotion, motivation = agent.cognition.step(
        observation, agent.cultural_params
    )
    print(f"    Hidden state (intent): {hidden_state.intent.tolist()}")
    print(f"    Hidden state (social): {hidden_state.social.tolist()}")
    print(f"    Predicted emotion: {emotion.dominant_emotion} (valence={emotion.valence:.2f}, arousal={emotion.arousal:.2f})")
    print(f"    Motivation: {motivation.combined[:3].tolist()}...")

    print("\n[3] Making decision...")
    decision = agent.decision.make_decision(hidden_state, agent.cultural_params)
    print(f"    Selected action: {decision.optimal_action.name} (id={decision.optimal_action.id})")
    print(f"    Utility: {decision.utility.total:.3f}")
    print(f"    Utility breakdown: value={decision.utility.value_component:.3f}, "
          f"norm={decision.utility.norm_component:.3f}, "
          f"complexity={decision.utility.complexity_penalty:.3f}")

    print("\n[4] Executing action...")
    action_result = agent.action.execute_action(decision.optimal_action, agent.cultural_params)
    print(f"    Next environment state: {action_result.new_environment.state[:5].tolist()}...")
    print(f"    Prediction error: {action_result.prediction_error:.4f}")

    print("\n[5] Reasoning Chain:")
    for i, step in enumerate(decision.reasoning_chain.steps):
        print(f"    Step {i+1}: {step.stage}")
        print(f"             {step.rationale}")

    print("\n" + "=" * 60)
    print("FINAL DECISION")
    print("=" * 60)
    print(f"Action: {decision.optimal_action.name}")
    print(f"Reasoning: {decision.reasoning_chain.final_decision}")
    print(f"Alternatives considered: {', '.join(decision.reasoning_chain.alternatives_considered[:3])}")

    return {
        "observation": observation,
        "hidden_state": hidden_state,
        "emotion": emotion,
        "motivation": motivation,
        "decision": decision,
        "action_result": action_result
    }


def run_custom_scenario(agent: HELGAAgent, scenario_desc: str) -> Dict[str, Any]:
    """Run custom scenario from description.

    Args:
        agent: HELGAAgent instance
        scenario_desc: Scenario description

    Returns:
        Scenario results
    """
    print(f"\nScenario: {scenario_desc}")

    # Use ScenarioEncoder for better context understanding
    from core.scenario import create_scenario_encoder
    encoder = create_scenario_encoder()
    env_state, event_stream, other_states = encoder.encode(scenario_desc)

    result = agent.step(env_state, event_stream, other_states)

    # Generate detailed action plan
    from core.action_parameterizer import create_parameterizer
    parameterizer = create_parameterizer()
    action_plan = parameterizer.parameterize(
        result['decision'].optimal_action.name,
        scenario_desc,
        {
            'env_state': env_state,
            'event_stream': event_stream,
            'other_states': other_states
        }
    )

    result['action_plan'] = action_plan
    return result


def run_interactive_mode(agent: HELGAAgent) -> None:
    """Run interactive mode.

    Args:
        agent: HELGAAgent instance
    """
    print("\n" + "=" * 60)
    print("HELGA Interactive Mode")
    print("=" * 60)
    print("Enter scenario descriptions or 'quit' to exit")
    print("Example: 'my colleague is late for an important meeting'")
    print()

    while True:
        try:
            user_input = input("> ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not user_input:
                continue

            result = run_custom_scenario(agent, user_input)

            print(f"\n  Decision: {result['decision'].optimal_action.name}")
            print(f"  Emotion: {result['emotion'].dominant_emotion}")
            print(f"  Reasoning chain length: {len(result['decision'].reasoning_chain.steps)} steps")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HELGA - Humanistic Environmental Learning and Generating Agent"
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--scenario",
        choices=["colleague_late", "custom"],
        help="Run predefined scenario"
    )
    parser.add_argument(
        "--custom-scenario",
        type=str,
        help="Run custom scenario description"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive mode"
    )
    parser.add_argument(
        "--run-validation",
        action="store_true",
        help="Run all validation tests"
    )
    parser.add_argument(
        "--neural-perception",
        action="store_true",
        help="Use neural network for perception (requires PyTorch)"
    )

    args = parser.parse_args()

    # Create agent
    try:
        agent = HELGAAgent(
            config_path=args.config if args.config != "config/default.yaml" else None,
            use_neural_perception=args.neural_perception
        )
    except FileNotFoundError:
        print(f"Warning: Config file '{args.config}' not found, using defaults")
        agent = HELGAAgent()

    # Run validation if requested
    if args.run_validation:
        print("\nRunning all validations...")
        results = run_all_validations()
        print_validation_summary(results)
        return

    # Run colleague late scenario
    if args.scenario == "colleague_late":
        run_colleague_late_scenario(agent)
        return

    # Run custom scenario
    if args.custom_scenario:
        result = run_custom_scenario(agent, args.custom_scenario)
        print(f"\nDecision: {result['decision'].optimal_action.name}")
        return

    # Run interactive mode
    if args.interactive:
        run_interactive_mode(agent)
        return

    # Default: show help
    parser.print_help()
    print("\nExample: python main.py --scenario colleague_late")


if __name__ == "__main__":
    main()