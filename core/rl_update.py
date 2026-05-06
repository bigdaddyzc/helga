"""Reinforcement learning update mechanism with user feedback."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class UserFeedback:
    """User feedback with score and rationale."""
    action_idx: int
    score: float  # 1-5
    rationale: str

class RLUpdate:
    """Handle RL updates based on user feedback.

    L_total = L_policy + α · L_supervised + β · L_contrastive
    """

    def __init__(
        self,
        policy_weight: float = 1.0,
        supervised_weight: float = 0.5,
        contrastive_weight: float = 0.3,
        margin: float = 1.0
    ):
        self.policy_weight = policy_weight
        self.supervised_weight = supervised_weight
        self.contrastive_weight = contrastive_weight
        self.margin = margin

    def compute_policy_loss(
        self,
        probs: torch.Tensor,
        action_idx: int,
        score: float
    ) -> torch.Tensor:
        """Compute policy loss.

        L_policy = -log P(a_chosen | E, M) · score
        """
        # Normalize score to 0-1 range
        normalized_score = (score - 1) / 4  # 1-5 -> 0-1

        # Negative log prob for chosen action
        neg_log_prob = -torch.log(probs[action_idx] + 1e-8)

        return self.policy_weight * neg_log_prob * normalized_score

    def compute_supervised_loss(
        self,
        predicted: torch.Tensor,
        preferred: torch.Tensor
    ) -> torch.Tensor:
        """Compute supervised loss.

        L_supervised = ||P_predicted - P_preferred||²
        """
        return self.supervised_weight * torch.sum((predicted - preferred) ** 2)

    def compute_contrastive_loss(
        self,
        pos_embedding: torch.Tensor,
        neg_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss.

        L_contrastive = max(0, margin - sim(pos) + sim(neg))
        """
        similarity = torch.nn.functional.cosine_similarity(
            pos_embedding.unsqueeze(0),
            neg_embedding.unsqueeze(0)
        )
        return torch.relu(self.margin - similarity + torch.mean(neg_embedding))

    def compute_total_loss(
        self,
        probs: torch.Tensor,
        feedback: UserFeedback,
        preferred_probs: Optional[torch.Tensor] = None,
        pos_embedding: Optional[torch.Tensor] = None,
        neg_embedding: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Compute total loss from user feedback.

        Returns:
            (total_loss, loss_breakdown)
        """
        # Policy loss
        policy_loss = self.compute_policy_loss(
            probs, feedback.action_idx, feedback.score
        )

        loss_breakdown = {"policy": policy_loss.item()}

        total_loss = policy_loss

        # Supervised loss if preferred provided
        if preferred_probs is not None:
            sup_loss = self.compute_supervised_loss(probs, preferred_probs)
            total_loss = total_loss + sup_loss
            loss_breakdown["supervised"] = sup_loss.item()

        # Contrastive loss if embeddings provided
        if pos_embedding is not None and neg_embedding is not None:
            cont_loss = self.compute_contrastive_loss(pos_embedding, neg_embedding)
            total_loss = total_loss + self.contrastive_weight * cont_loss
            loss_breakdown["contrastive"] = cont_loss.item()

        return total_loss, loss_breakdown

    def parse_rationale(
        self,
        rationale: str,
        action_names: List[str]
    ) -> Tuple[Optional[int], Optional[int]]:
        """Parse rationale to extract positive and negative action references.

        Returns:
            (positive_action_idx, negative_action_idx) or (None, None)
        """
        # Simple keyword-based parsing
        rationale_lower = rationale.lower()

        positive_kw = ["good", "better", "prefer", "like", "应该", "好", "选"]
        negative_kw = ["bad", "worse", "avoid", "dislike", "不应", "差", "不要"]

        pos_idx = None
        neg_idx = None

        for i, name in enumerate(action_names):
            if name.lower() in rationale_lower:
                # Check context
                for kw in positive_kw:
                    if kw in rationale_lower:
                        pos_idx = i
                        break
                for kw in negative_kw:
                    if kw in rationale_lower:
                        neg_idx = i
                        break

        return pos_idx, neg_idx