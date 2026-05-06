"""Active ask mechanism for information gathering."""

import torch
from typing import Tuple, Optional

# Intent to question mapping
INTENT_QUESTION_MAP = {
    "沟通": "你想通过什么方式与对方沟通？",
    "等待": "你愿意等待多长时间？",
    "升级": "你想升级给谁？",
    "询问": "你能补充一下情况吗？",
    "放弃": "你想如何处理这个问题？"
}

class AskTrigger:
    """Determine when to trigger ask action based on uncertainty."""

    @staticmethod
    def compute_entropy(probs: torch.Tensor) -> float:
        """Compute entropy of probability distribution.

        H(P) = -Σ_a P(a) · log P(a)
        """
        return -torch.sum(probs * torch.log(probs + 1e-8)).item()

    @staticmethod
    def should_trigger(
        probs: torch.Tensor,
        threshold: float,
        missing_key_info: bool = False
    ) -> Tuple[bool, float]:
        """Determine if ask should be triggered.

        Args:
            probs: Action probability distribution
            threshold: Entropy threshold for triggering
            missing_key_info: Whether key info is missing

        Returns:
            (should_trigger, entropy)
        """
        entropy = AskTrigger.compute_entropy(probs)
        should_trigger = (entropy > threshold) or missing_key_info
        return should_trigger, entropy


def generate_question(intent: str) -> str:
    """Generate clarification question for intent.

    Args:
        intent: The current intent (沟通, 等待, 升级, 询问, 放弃)

    Returns:
        Question string for intent clarification
    """
    return INTENT_QUESTION_MAP.get(intent, "你能补充一下情况吗？")


class AskDecision:
    """Handle ask decision mode."""

    def __init__(self, entropy_threshold: float = 1.0):
        self.entropy_threshold = entropy_threshold

    def decide(
        self,
        action_probs: torch.Tensor,
        missing_key_info: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """Decide whether to ask and generate question.

        Returns:
            (should_ask, question_or_none)
        """
        should_ask, entropy = AskTrigger.should_trigger(
            action_probs,
            self.entropy_threshold,
            missing_key_info
        )

        if should_ask:
            # Default to asking for intent clarification
            return True, "你能补充一下情况吗？"

        return False, None