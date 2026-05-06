import torch
from core.rl_update import RLUpdate, UserFeedback

def test_policy_loss_computation():
    """Test policy loss with user score."""
    updater = RLUpdate()
    probs = torch.tensor([0.6, 0.3, 0.1])
    action_idx = 0
    score = 4.0  # 1-5 scale

    loss = updater.compute_policy_loss(probs, action_idx, score)
    assert loss.item() > 0

def test_supervised_loss():
    """Test supervised loss between predicted and preferred."""
    updater = RLUpdate()
    predicted = torch.tensor([0.5, 0.3, 0.2])
    preferred = torch.tensor([0.7, 0.2, 0.1])

    loss = updater.compute_supervised_loss(predicted, preferred)
    assert loss.item() >= 0