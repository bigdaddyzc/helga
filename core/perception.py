"""Perception system with multi-head attention for HELGA.

This module implements the attention-based perception system that encodes
environment vectors, event streams, and other states into a unified
observation vector.

Attention Mechanism Formula (from spec):
    O_t = softmax((E_t + V_t) · W_q (v · W_v)^T / √d_k) · (E_t + V_t + S_t^other)

Where:
- E_t: Environment vector (dim=10)
- V_t: Event stream embedding (dim=64)
- S_t_other: Other states (dim=10)
- v: Schwartz values vector (dim=10)
- W_q, W_v: Projection matrices
- d_k: Key dimension

Output: O_t (dim=32) with attention weights per head.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np

from .types import ObservationInput, PerceptionOutput
from .culture import CulturalParameters


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism.

    Formula:
        head_i = Attention(Q, K, V)_i
        O_t = Concat(head_1, ..., head_h) * W_o

    where Attention(Q,K,V) = softmax(Q*K^T / √d_k) * V

    Attributes:
        num_heads: Number of attention heads (default: 4)
        d_model: Model dimension (default: 32)
        d_k: Key/query dimension per head
    """

    def __init__(
        self,
        num_heads: int = 4,
        d_model: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        # Projection matrices
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head attention.

        Args:
            query: Query tensor (batch, seq, d_model)
            key: Key tensor (batch, seq, d_model)
            value: Value tensor (batch, seq, d_model)
            mask: Optional attention mask

        Returns:
            output: Attended output (batch, seq, d_model)
            attention_weights: Attention weights (batch, num_heads, seq, seq)
        """
        batch_size = query.size(0)
        seq_len = query.size(1)

        # Linear projections
        Q = self.W_q(query)   # (batch, seq, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape for multi-head: (batch, num_heads, seq, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        # Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, num_heads, seq, seq)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)  # (batch, num_heads, seq, seq)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # (batch, num_heads, seq, d_k)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous()  # (batch, seq, num_heads, d_k)
        context = context.view(batch_size, seq_len, self.d_model)  # (batch, seq, d_model)

        # Final linear projection
        output = self.W_o(context)

        # Return average attention weights across heads for interpretation
        avg_attention = attention_weights.mean(dim=1)  # (batch, seq, seq)

        return output, avg_attention


class PerceptionEncoder(nn.Module):
    """Encodes raw inputs to model dimension.

    Formula: encoded = ReLU(W * input + b)

    Attributes:
        env_dim: Environment dimension (10)
        event_dim: Event dimension (64)
        other_dim: Other states dimension (10)
        d_model: Model dimension (32)
    """

    def __init__(
        self,
        env_dim: int = 10,
        event_dim: int = 64,
        other_dim: int = 10,
        d_model: int = 32
    ):
        super().__init__()

        total_dim = env_dim + event_dim + other_dim  # 84

        # Project to model dimension
        self.projection = nn.Sequential(
            nn.Linear(total_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

        # Cultural vector projection for attention
        self.cultural_projection = nn.Linear(10, d_model)

    def forward(self, inputs: torch.Tensor, cultural_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode inputs with cultural guidance.

        Args:
            inputs: Concatenated input tensor (batch, 84)
            cultural_vec: Schwartz values vector (batch, 10)

        Returns:
            encoded: Encoded tensor (batch, d_model)
            cultural_proj: Projected cultural vector (batch, d_model)
        """
        encoded = self.projection(inputs)  # (batch, d_model)
        cultural_proj = self.cultural_projection(cultural_vec)  # (batch, d_model)
        return encoded, cultural_proj


class PerceptionSystem(nn.Module):
    """Main perception system with attention.

    Pipeline:
        1. Concatenate [E_t; V_t; S_t_other] -> concat (dim=84)
        2. Linear projection to d_model=32
        3. Multi-head attention (4 heads) with cultural guidance
        4. Output O_t (dim=32)

    Attributes:
        encoder: Input encoder
        attention: Multi-head attention
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        env_dim: int = 10,
        event_dim: int = 64,
        other_dim: int = 10,
        output_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.env_dim = env_dim
        self.event_dim = event_dim
        self.other_dim = other_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        # Input encoder
        self.encoder = PerceptionEncoder(
            env_dim=env_dim,
            event_dim=event_dim,
            other_dim=other_dim,
            d_model=output_dim
        )

        # Multi-head attention
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            d_model=output_dim,
            dropout=dropout
        )

    def forward(
        self,
        env_state: torch.Tensor,
        event_stream: torch.Tensor,
        other_states: torch.Tensor,
        cultural_params: CulturalParameters
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through perception system.

        Formula:
            concat = [E_t; V_t; S_t_other]
            encoded = projection(concat)
            cultural_proj = W_c * v
            O_t, weights = MHAttention(encoded, encoded, encoded; query=cultural_proj)

        Args:
            env_state: Environment vector (batch, 10)
            event_stream: Event stream (batch, 64)
            other_states: Other states (batch, 10)
            cultural_params: Cultural parameters (Schwartz values)

        Returns:
            observation: Attended observation (batch, 32)
            attention_weights: Attention weights (batch, seq, seq)
        """
        # Concatenate inputs
        concat_inputs = torch.cat([env_state, event_stream, other_states], dim=-1)  # (batch, 84)

        # Project inputs and cultural vector
        encoded, cultural_proj = self.encoder(concat_inputs, cultural_params)  # (batch, 32) each

        # Expand for attention: need seq dimension
        encoded_seq = encoded.unsqueeze(1)  # (batch, 1, 32)
        cultural_query = cultural_proj.unsqueeze(1)  # (batch, 1, 32)

        # Multi-head attention with cultural query
        # Query: cultural guidance, Key/Value: encoded inputs
        output, weights = self.attention(cultural_query, encoded_seq, encoded_seq)

        # Squeeze sequence dimension
        observation = output.squeeze(1)  # (batch, 32)
        attention_weights = weights.squeeze(0)  # Remove batch if single sample

        return observation, attention_weights

    def encode_numpy(
        self,
        env_state: np.ndarray,
        event_stream: np.ndarray,
        other_states: np.ndarray,
        cultural_params: CulturalParameters
    ) -> PerceptionOutput:
        """Encode from numpy inputs.

        Args:
            env_state: Environment vector (10,)
            event_stream: Event stream (64,)
            other_states: Other states (10,)
            cultural_params: Cultural parameters

        Returns:
            PerceptionOutput with observation and attention weights
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensors
            env_t = torch.from_numpy(env_state).float().unsqueeze(0)
            event_t = torch.from_numpy(event_stream).float().unsqueeze(0)
            other_t = torch.from_numpy(other_states).float().unsqueeze(0)
            cultural_t = torch.from_numpy(cultural_params.schwartz_values.values).float().unsqueeze(0)

            # Forward pass
            observation, weights = self.forward(env_t, event_t, other_t, cultural_params)

            return PerceptionOutput(
                observation=observation.squeeze(0).numpy(),
                attention_weights=weights.squeeze(0).numpy()
            )


def create_perception_system(
    env_dim: int = 10,
    event_dim: int = 64,
    other_dim: int = 10,
    output_dim: int = 32,
    num_heads: int = 4,
    dropout: float = 0.1
) -> PerceptionSystem:
    """Factory function to create perception system.

    Args:
        env_dim: Environment dimension (10)
        event_dim: Event dimension (64)
        other_dim: Other states dimension (10)
        output_dim: Output dimension (32)
        num_heads: Number of attention heads (4)
        dropout: Dropout rate (0.1)

    Returns:
        Configured PerceptionSystem instance
    """
    return PerceptionSystem(
        env_dim=env_dim,
        event_dim=event_dim,
        other_dim=other_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        dropout=dropout
    )


def compute_synthetic_observation(
    env_state: np.ndarray,
    event_stream: np.ndarray,
    other_states: np.ndarray,
    cultural_params: CulturalParameters
) -> np.ndarray:
    """Compute observation without neural network (simplified version).

    Uses weighted combination based on cultural parameters.

    Formula: O_t = w_v normalized * concat_inputs

    Args:
        env_state: Environment vector (10,)
        event_stream: Event stream (64,)
        other_states: Other states (10,)
        cultural_params: Cultural parameters

    Returns:
        Observation vector (32,) - padded version
    """
    # Simple weighted concatenation
    concat = np.concatenate([env_state, event_stream, other_states])  # (84,)

    # Truncate or pad to 32 dimensions
    if len(concat) > 32:
        # Weighted selection based on cultural values
        weights = np.concatenate([
            cultural_params.schwartz_values.values * 0.5,  # 10 dims for env
            np.ones(64) * 0.3,  # 64 dims for events
            cultural_params.schwartz_values.values * 0.5   # 10 dims for others
        ])
        observation = concat[:32]
    else:
        observation = np.pad(concat, (0, 32 - len(concat)))  # pad to 32

    # Normalize
    norm = np.linalg.norm(observation)
    if norm > 0:
        observation = observation / norm

    return observation.astype(np.float32)