"""Integrated test for full decision loop: encode -> decide -> ask -> update."""

import torch
from core.decision import HierarchicalDecisionMaker
from core.environment import EnvironmentEncoder
from core.ask_mechanism import AskTrigger
from core.rl_update import RLUpdate, UserFeedback


def test_full_decision_loop():
    """Test complete decision loop: encode -> decide -> ask -> update."""
    # Setup
    decision_model = HierarchicalDecisionMaker()
    env_encoder = EnvironmentEncoder()

    # Mock environment input
    time_input = torch.randn(4)
    space_input = torch.randn(3)
    social_input = torch.randn(4)
    info_input = torch.randn(4)

    env_encoding = env_encoder(time_input, space_input, social_input, info_input)
    hidden_state = torch.randn(10)
    cultural_prior = torch.ones(20)

    # Decision
    probs = decision_model(hidden_state, env_encoding, cultural_prior)

    # Check ask trigger
    should_ask, entropy = AskTrigger.should_trigger(probs, threshold=1.0)
    assert isinstance(should_ask, bool)

    # RL update
    updater = RLUpdate()
    feedback = UserFeedback(action_idx=0, score=4, rationale="这个决策比较好")
    loss, breakdown = updater.compute_total_loss(probs, feedback)

    assert loss.item() > 0
    print(f"Test passed! Loss: {loss.item():.4f}, Breakdown: {breakdown}")


if __name__ == "__main__":
    test_full_decision_loop()
