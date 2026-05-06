import torch
from core.ask_mechanism import AskTrigger, generate_question

def test_ask_trigger_condition():
    """Test ask triggered when entropy > threshold."""
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25])  # High entropy (uniform)
    threshold = 1.0  # ~log(4) = 1.386
    should_ask, entropy = AskTrigger.should_trigger(probs, threshold)
    assert should_ask == True

def test_ask_not_triggered_low_entropy():
    """Test ask not triggered when entropy is low."""
    probs = torch.tensor([0.9, 0.05, 0.03, 0.02])  # Low entropy
    threshold = 1.0
    should_ask, entropy = AskTrigger.should_trigger(probs, threshold)
    assert should_ask == False

def test_generate_question():
    """Test question generation for intent."""
    q = generate_question("沟通")
    assert isinstance(q, str)
    assert len(q) > 0