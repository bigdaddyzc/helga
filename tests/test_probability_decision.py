import torch
import numpy as np
from core.decision import HierarchicalDecisionMaker

def test_probability_distribution_output():
    """Test decision system outputs probability distribution."""
    model = HierarchicalDecisionMaker()
    # Mock inputs
    hidden_state = torch.randn(10)
    env_encoding = torch.randn(32)
    cultural_prior = torch.ones(20)

    probs = model.forward(hidden_state, env_encoding, cultural_prior)
    # Check probability distribution properties
    assert probs.shape == (20,)
    assert abs(probs.sum() - 1.0) < 1e-5
    assert (probs >= 0).all()