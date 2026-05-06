"""Tests for HELGA cognitive agent system."""

import pytest
import numpy as np

from helga.core.culture import (
    SchwartzValues, SocialNormsMatrix, AffectSchema,
    CulturalParameters
)
from helga.core.types import (
    HiddenState, EmotionState, Action, EnvironmentState,
    ValidationResult, ValidationSeverity
)
from helga.core.decision import create_decision_system, ACTION_NAMES
from helga.core.action import create_action_system, simulate_environment_step
from helga.memory.working import WorkingMemory, create_working_memory
from helga.memory.episodic import EpisodicMemory, create_episodic_memory
from helga.core.types import Experience


class TestCultureModule:
    """Test culture module."""

    def test_schwartz_values_default(self):
        """Test default Schwartz values creation."""
        sv = SchwartzValues.default()
        assert sv.values.shape == (10,)
        assert np.all(sv.values >= 0) and np.all(sv.values <= 1)

    def test_schwartz_values_from_list(self):
        """Test creating Schwartz values from list."""
        values = [0.5] * 10
        sv = SchwartzValues.from_list(values)
        assert np.allclose(sv.values, values)

    def test_social_norms_matrix_default(self):
        """Test default social norms matrix."""
        snm = SocialNormsMatrix.default()
        assert snm.matrix.shape == (10, 20)

    def test_affect_schema_default(self):
        """Test default affect schema."""
        af = AffectSchema.default()
        assert af.schema.shape == (10,)

    def test_cultural_parameters_default(self):
        """Test creating default cultural parameters."""
        cp = CulturalParameters.default()
        assert cp.schwartz_values.values.shape == (10,)
        assert cp.norms_matrix.matrix.shape == (10, 20)
        assert cp.affect_schema.schema.shape == (10,)

    def test_valence_computation(self):
        """Test valence computation."""
        af = AffectSchema.default()
        motivation = np.ones(10) * 0.5
        valence = af.compute_valence(motivation)
        assert -1 <= valence <= 1


class TestDecisionModule:
    """Test decision module."""

    def test_create_decision_system(self):
        """Test decision system creation."""
        ds = create_decision_system()
        assert ds.action_dim == 20
        assert len(ds.action_space) == 20

    def test_action_space_names(self):
        """Test action names."""
        assert ACTION_NAMES[0] == "wait"
        assert ACTION_NAMES[1] == "send_reminder"
        assert len(ACTION_NAMES) == 20

    def test_make_decision(self):
        """Test making a decision."""
        ds = create_decision_system()
        cp = CulturalParameters.default()
        hidden = HiddenState(
            intent=np.ones(3) * 0.5,
            social=np.ones(5) * 0.5,
            causal=np.ones(2) * 0.5
        )

        decision = ds.make_decision(hidden, cp)
        assert decision.optimal_action is not None
        assert decision.optimal_action.id >= 0 and decision.optimal_action.id < 20
        assert decision.utility.total is not None
        assert len(decision.reasoning_chain.steps) > 0


class TestActionModule:
    """Test action module."""

    def test_create_action_system(self):
        """Test action system creation."""
        action_sys = create_action_system()
        assert action_sys.env_dim == 10

    def test_initialize_environment(self):
        """Test environment initialization."""
        action_sys = create_action_system()
        env_state = action_sys.initialize_environment()
        assert env_state.state.shape == (10,)
        assert env_state.timestamp == 0.0

    def test_simulate_environment_step(self):
        """Test environment simulation."""
        current_state = np.ones(10) * 0.5
        next_state = simulate_environment_step(current_state, action_id=1)
        assert next_state.shape == (10,)
        assert np.all(next_state >= 0) and np.all(next_state <= 1)


class TestMemoryModule:
    """Test memory module."""

    def test_working_memory_store_retrieve(self):
        """Test working memory store and retrieve."""
        wm = create_working_memory(capacity=5)
        wm.store("key1", "value1")
        assert wm.retrieve("key1") == "value1"

    def test_working_memory_eviction(self):
        """Test working memory eviction when full."""
        wm = create_working_memory(capacity=3)
        wm.store("key1", "value1", importance=0.5)
        wm.store("key2", "value2", importance=0.8)
        wm.store("key3", "value3", importance=0.9)
        wm.store("key4", "value4", importance=0.7)  # Should evict key1

        assert wm.retrieve("key1") is None
        assert wm.retrieve("key2") == "value2"

    def test_episodic_memory_store_sample(self):
        """Test episodic memory store and sample."""
        em = create_episodic_memory(capacity=10)

        exp = Experience(
            state=np.ones(10),
            action=Action(id=1, name="test"),
            reward=0.5,
            next_state=np.ones(10) * 0.8,
            done=False
        )
        em.store(exp)

        samples = em.sample(batch_size=5)
        assert len(samples) <= 5
        assert len(samples) <= em.size()


class TestValidationModule:
    """Test validation module."""

    def test_ecological_validation(self):
        """Test ecological validity validation."""
        from helga.validation import (
            validate_ecological, generate_synthetic_ecological_data
        )

        agent_actions, human_actions = generate_synthetic_ecological_data()
        result = validate_ecological(agent_actions, human_actions)

        assert isinstance(result, ValidationResult)
        assert 0 <= result.metric_value <= 1

    def test_cultural_generalization(self):
        """Test cultural generalization validation."""
        from helga.validation import (
            validate_cultural_generalization, generate_synthetic_cultural_data
        )

        behaviors, ratings = generate_synthetic_cultural_data()
        result = validate_cultural_generalization(behaviors, ratings)

        assert isinstance(result, ValidationResult)
        assert -1 <= result.metric_value <= 1

    def test_run_all_validations(self):
        """Test running all validations."""
        from helga.validation import run_all_validations

        results = run_all_validations()

        assert len(results) == 6
        assert 'ecological' in results
        assert 'cultural_generalization' in results
        assert 'counterfactual' in results
        assert 'structural' in results
        assert 'temporal' in results
        assert 'adversarial' in results


class TestIntegration:
    """Integration tests."""

    def test_full_agent_loop(self):
        """Test complete agent loop."""
        from helga.main import HELGAAgent

        agent = HELGAAgent()

        # Environment state
        env_state = np.array([0.8, 0.3, 0.0, 0.7, 0.5, 0.6, 0.7, 0.4, 0.6, 0.9], dtype=np.float32)
        event_stream = np.zeros(64, dtype=np.float32)
        event_stream[1] = 0.9
        other_states = np.array([0.4, 0.7, 0.5, 0.6, 0.5, 0.4, 0.6, 0.5, 0.5, 0.5], dtype=np.float32)

        # Run agent step
        result = agent.step(env_state, event_stream, other_states)

        assert 'observation' in result
        assert 'hidden_state' in result
        assert 'emotion' in result
        assert 'decision' in result
        assert 'action_result' in result

    def test_colleague_late_scenario(self):
        """Test colleague late scenario."""
        from helga.main import HELGAAgent

        agent = HELGAAgent()
        from helga.main import run_colleague_late_scenario

        result = run_colleague_late_scenario(agent)

        assert 'decision' in result
        assert 'emotion' in result
        assert result['decision'].optimal_action.name in ACTION_NAMES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])