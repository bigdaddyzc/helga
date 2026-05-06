"""Cognition system with Bayesian mental simulation for HELGA.

This module implements the Bayesian mental simulation using particle filtering
or stochastic variational inference (SVI) with Pyro.

Hidden State Z_t (dim=10):
    - Intent (3): helping, maintaining, professional
    - Social Relations (5): rapport, hierarchy, trust, etc.
    - Physical Causality (2): time awareness, causality

Belief Update Formula:
    P(Z_t | O_{1:t}) ∝ P(O_t | Z_t) * Σ_{Z_{t-1}} P(Z_t | Z_{t-1}) * P(Z_{t-1} | O_{1:t-1})

Emotion Prediction:
    Emo_pred_t = Σ_i w_i * Valence(z_i)
    Valence(z) = tanh(A · z)

Motivation Vector:
    M_t = Softmax(W_m [Emo_pred_t; D_t; N · context_onehot])
"""

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from typing import Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass

from .types import HiddenState, EmotionState, MotivationVector, PerceptionOutput
from .culture import CulturalParameters


class TransitionModel(nn.Module):
    """State transition model for hidden states.

    Formula: P(Z_t | Z_{t-1}, a_{t-1}) = N(transition(Z_{t-1}, a_{t-1}), Σ)

    Attributes:
        hidden_dim: Hidden state dimension (10)
        action_dim: Action embedding dimension
    """

    def __init__(self, hidden_dim: int = 10, action_dim: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Transition network
        self.transition_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Covariance (diagonal)
        self.log_cov = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, z_prev: torch.Tensor, action_emb: torch.Tensor) -> dist.Normal:
        """Compute transition distribution.

        Args:
            z_prev: Previous hidden state (batch, 10)
            action_emb: Action embedding (batch, action_dim)

        Returns:
            Normal distribution over next state
        """
        inputs = torch.cat([z_prev, action_emb], dim=-1)
        mean = self.transition_net(inputs)
        cov = torch.exp(self.log_cov)
        return dist.Normal(mean, cov)


class EmissionModel(nn.Module):
    """Observation emission model.

    Formula: P(O_t | Z_t) = N(emission(Z_t), Σ_o)

    Attributes:
        hidden_dim: Hidden state dimension (10)
        obs_dim: Observation dimension (32)
    """

    def __init__(self, hidden_dim: int = 10, obs_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim

        # Emission network
        self.emission_net = nn.Sequential(
            nn.Linear(hidden_dim, obs_dim * 2),
            nn.ReLU(),
            nn.Linear(obs_dim * 2, obs_dim)
        )

        # Covariance
        self.log_cov = nn.Parameter(torch.zeros(obs_dim))

    def forward(self, z: torch.Tensor) -> dist.Normal:
        """Compute emission distribution.

        Args:
            z: Hidden state (batch, 10)

        Returns:
            Normal distribution over observation
        """
        mean = self.emission_net(z)
        cov = torch.exp(self.log_cov)
        return dist.Normal(mean, cov)


class BayesianMentalModel(nn.Module):
    """Bayesian mental simulation using Pyro SVI.

    Uses variational inference to approximate:
        P(Z_t | O_{1:t}) ≈ q(Z_t)

    Pipeline:
        1. Initialize hidden state prior P(Z_0)
        2. For each timestep:
           - Update beliefs via SVI
           - Predict emotion using valence function
           - Compute motivation vector
    """

    def __init__(
        self,
        hidden_dim: int = 10,
        obs_dim: int = 32,
        num_particles: int = 100
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.num_particles = num_particles

        # Transition and emission models
        self.transition = TransitionModel(hidden_dim=hidden_dim)
        self.emission = EmissionModel(hidden_dim=hidden_dim, obs_dim=obs_dim)

        # Prior on initial state
        self.prior_loc = nn.Parameter(torch.zeros(hidden_dim))
        self.prior_scale = nn.Parameter(torch.ones(hidden_dim))

        # Emotion prediction network
        self.emotion_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def model(self, observations: Optional[torch.Tensor] = None, num_steps: int = 1) -> torch.Tensor:
        """Generative model.

        Formula:
            Z_0 ~ N(prior_loc, prior_scale)
            for t in 1..num_steps:
                Z_t ~ P(Z_t | Z_{t-1})
                O_t ~ P(O_t | Z_t)

        Args:
            observations: Optional observation sequence
            num_steps: Number of steps to generate

        Returns:
            Hidden state trajectory
        """
        with pyro.plate("data", num_steps):
            # Sample initial state
            z_prev = pyro.sample("z_0", dist.Normal(self.prior_loc, self.prior_scale))

            # Generate trajectory
            z_samples = [z_prev]
            for t in range(1, num_steps):
                # Transition
                z_mean = self.transition.transition_net(torch.cat([z_prev, torch.zeros(5)], dim=-1))
                z_t = pyro.sample(f"z_{t}", dist.Normal(z_mean, torch.exp(self.transition.log_cov)))

                z_samples.append(z_t)
                z_prev = z_t

        return torch.stack(z_samples)

    def guide(self, observations: Optional[torch.Tensor] = None, num_steps: int = 1) -> torch.Tensor:
        """Variational guide for SVI.

        Args:
            observations: Observation sequence
            num_steps: Number of steps

        Returns:
            Hidden state samples from approximate posterior
        """
        with pyro.plate("data", num_steps):
            # Posterior parameters (learned)
            z_loc = pyro.param("z_loc", torch.zeros(num_steps, self.hidden_dim))
            z_scale = torch.exp(pyro.param("z_scale", torch.zeros(num_steps, self.hidden_dim)))

            # Sample from posterior
            for t in range(num_steps):
                pyro.sample(f"z_{t}", dist.Normal(z_loc[t], z_scale[t]))

        return z_loc

    def infer_state(
        self,
        observation: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Infer hidden state from observation using SVI.

        Args:
            observation: Observation tensor (32,)
            prev_state: Previous hidden state (10,)

        Returns:
            z_mean: Inferred state mean
            z_std: Inferred state std
        """
        # Setup SVI
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())

        # Single step inference
        num_steps = 2 if prev_state is not None else 1

        # Run inference
        loss = 0
        for _ in range(100):
            loss += svi.step(observations=observation.unsqueeze(0), num_steps=num_steps)

        # Get posterior parameters
        z_loc = pyro.param("z_loc")
        z_scale = torch.exp(pyro.param("z_scale"))

        if prev_state is not None:
            # Use last inferred state
            return z_loc[-1], z_scale[-1]
        else:
            return z_loc[0], z_scale[0]

    def compute_valence(self, hidden_state: torch.Tensor, affect_schema: torch.Tensor) -> torch.Tensor:
        """Compute emotional valence.

        Formula: Valence(z) = tanh(A · z)

        Args:
            hidden_state: Hidden state (batch, 10)
            affect_schema: Affect schema weights (10,)

        Returns:
            Valence score (-1 to 1)
        """
        valence = torch.matmul(hidden_state, affect_schema)
        return torch.tanh(valence)

    def compute_motivation(
        self,
        emotion: torch.Tensor,
        needs: torch.Tensor,
        norm_compliance: torch.Tensor
    ) -> torch.Tensor:
        """Compute motivation vector.

        Formula: M_t = Softmax(W_m [emotion; needs; norm_compliance])

        Weights: emotion=0.5, needs=0.3, norm=0.2

        Args:
            emotion: Emotion tensor (batch, 1)
            needs: Physiological needs (batch, 3)
            norm_compliance: Norm compliance (batch, 10)

        Returns:
            Motivation vector (batch, 10)
        """
        # Concatenate components with weights
        weighted = torch.cat([
            emotion * 0.5,  # (batch, 1)
            needs * 0.3,    # (batch, 3)
            norm_compliance * 0.2  # (batch, 10)
        ], dim=-1)  # Total: 14

        # Project to 10 dims
        motivation = torch.matmul(weighted, torch.eye(10, 14))  # Simplified projection
        motivation = torch.softmax(motivation, dim=-1)

        return motivation


class CognitionSystem:
    """Main cognition system for HELGA.

    Pipeline:
        1. Initialize hidden state from prior P(Z_0)
        2. For each timestep:
           a. Update beliefs via particle filtering or SVI
           b. Predict emotion using valence function
           c. Compute motivation vector
    """

    def __init__(
        self,
        hidden_dim: int = 10,
        obs_dim: int = 32,
        num_particles: int = 100,
        inference_method: str = "particle_filter"
    ):
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.num_particles = num_particles
        self.inference_method = inference_method

        # Bayesian model
        self.model = BayesianMentalModel(
            hidden_dim=hidden_dim,
            obs_dim=obs_dim,
            num_particles=num_particles
        )

        # Current state
        self.current_state: Optional[HiddenState] = None
        self.current_emotion: Optional[EmotionState] = None
        self.current_motivation: Optional[MotivationVector] = None

    def initialize(self, cultural_params: CulturalParameters) -> HiddenState:
        """Initialize hidden state from cultural parameters.

        Args:
            cultural_params: Cultural parameters for initialization

        Returns:
            Initialized HiddenState
        """
        # Use Schwartz values to initialize intent components
        values = cultural_params.schwartz_values.values

        intent = np.array([
            values[2] * 0.8,  # Benevolence -> helping intent
            values[4] * 0.7,  # Security -> maintaining intent
            values[5] * 0.6   # Achievement -> professional intent
        ], dtype=np.float32)

        # Social relations from cultural parameters
        social = np.array([
            values[0] * 0.6,  # Self-Transcendence -> rapport
            values[3] * 0.5,  # Conformity -> hierarchy awareness
            values[9] * 0.7,  # Universalism -> trust
            values[8] * 0.4,  # Self-Direction -> autonomy
            values[1] * 0.5   # Openness -> openness
        ], dtype=np.float32)

        # Causality - mostly neutral initialization
        causal = np.array([0.5, 0.5], dtype=np.float32)

        self.current_state = HiddenState(intent=intent, social=social, causal=causal)
        return self.current_state

    def update_beliefs(
        self,
        observation: np.ndarray,
        cultural_params: CulturalParameters,
        action: Optional[np.ndarray] = None
    ) -> HiddenState:
        """Update beliefs based on new observation.

        Uses particle filtering or SVI depending on configuration.

        Formula:
            P(Z_t | O_{1:t}) ∝ P(O_t | Z_t) * Σ_{Z_{t-1}} P(Z_t | Z_{t-1}) * P(Z_{t-1} | O_{1:t-1})

        Args:
            observation: New observation (32,)
            cultural_params: Cultural parameters
            action: Optional action taken (for transition)

        Returns:
            Updated HiddenState
        """
        if self.current_state is None:
            return self.initialize(cultural_params)

        # Convert to tensors
        obs_tensor = torch.from_numpy(observation).float()
        prev_state_tensor = torch.from_numpy(self.current_state.to_vector()).float()

        # Inference
        if self.inference_method == "svi":
            z_mean, z_std = self.model.infer_state(obs_tensor, prev_state_tensor)
            new_state_vec = z_mean.detach().numpy()
        else:
            # Particle filtering (simplified)
            new_state_vec = self._particle_filter_update(
                observation, prev_state_tensor.numpy(), action
            )

        # Update current state
        self.current_state = HiddenState.from_vector(new_state_vec)
        return self.current_state

    def _particle_filter_update(
        self,
        observation: np.ndarray,
        prev_state: np.ndarray,
        action: Optional[np.ndarray],
        num_particles: int = 100
    ) -> np.ndarray:
        """Simplified particle filtering update.

        Args:
            observation: Observation (32,)
            prev_state: Previous state (10,)
            action: Optional action (5,)
            num_particles: Number of particles

        Returns:
            Updated state (10,)
        """
        # Generate particles from prior
        particles = np.random.randn(num_particles, self.hidden_dim) * 0.5 + prev_state

        # Compute weights based on observation likelihood
        # Simplified: distance to expected observation
        weights = np.zeros(num_particles)
        for i in range(num_particles):
            # Compute expected observation (simplified linear projection)
            expected_obs = particles[i][:self.obs_dim] if self.obs_dim <= self.hidden_dim else np.pad(particles[i], (0, self.obs_dim - self.hidden_dim))
            expected_obs = expected_obs[:32]

            # Distance
            dist = np.linalg.norm(observation - expected_obs)
            weights[i] = np.exp(-dist)

        # Normalize weights
        weights = weights / (weights.sum() + 1e-8)

        # Resample (simplified)
        indices = np.random.choice(num_particles, size=num_particles, p=weights)
        resampled = particles[indices]

        # Compute weighted mean
        new_state = np.average(resampled, axis=0, weights=weights)

        # Add small noise for exploration
        new_state = new_state + np.random.randn(self.hidden_dim) * 0.1

        return new_state.astype(np.float32)

    def predict_emotion(
        self,
        hidden_state: HiddenState,
        cultural_params: CulturalParameters
    ) -> EmotionState:
        """Predict emotion from hidden state.

        Formula:
            valence = tanh(A · Z_t)
            arousal = f(||Z_t||)
            dominant = argmax(correlation(A, Z_t))

        Args:
            hidden_state: Current hidden state
            cultural_params: Cultural parameters (affect schema)

        Returns:
            Predicted EmotionState
        """
        z_vec = hidden_state.to_vector()
        affect = cultural_params.affect_schema.schema

        # Compute valence: tanh(A · z)
        valence = np.tanh(np.dot(affect, z_vec))

        # Compute arousal (simplified)
        arousal = min(1.0, np.linalg.norm(z_vec) / 10.0)

        # Determine dominant emotion
        correlations = affect * z_vec
        dominant_idx = np.argmax(correlations)
        emotions = ["joy", "anger", "surprise", "gratitude", "fear",
                   "sadness", "disgust", "shame", "interest", "contentment"]
        dominant = emotions[dominant_idx] if dominant_idx < len(emotions) else "neutral"

        emotion = EmotionState(valence=valence, arousal=arousal, dominant_emotion=dominant)
        self.current_emotion = emotion
        return emotion

    def compute_motivation_vector(
        self,
        emotion: EmotionState,
        physiological_needs: Optional[np.ndarray] = None,
        norm_context: Optional[int] = None,
        cultural_params: Optional[CulturalParameters] = None
    ) -> MotivationVector:
        """Compute motivation vector.

        Formula:
            M_t = Softmax(W_m [Emo_pred_t; D_t; N · context_onehot])

        Weights: emotion=0.5, needs=0.3, norm=0.2

        Args:
            emotion: Current emotion state
            physiological_needs: Physiological needs (3,) - hunger, fatigue, social
            norm_context: Norm context index
            cultural_params: Cultural parameters

        Returns:
            MotivationVector
        """
        # Default needs
        if physiological_needs is None:
            physiological_needs = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        # Emotion component
        emotion_component = np.array([
            emotion.valence if emotion.valence > 0 else 0,
            abs(emotion.valence) if emotion.valence < 0 else 0,
            emotion.arousal,
            0, 0, 0, 0, 0, 0, 0
        ], dtype=np.float32)

        # Needs component (pad to 10)
        needs_component = np.pad(physiological_needs, (0, 7))[:10]

        # Norm component
        if norm_context is not None and cultural_params is not None:
            norm_compliance = cultural_params.norms_matrix.get_compliance(norm_context)
        else:
            norm_compliance = np.ones(10, dtype=np.float32) * 0.5

        # Compute combined motivation
        combined = 0.5 * emotion_component + 0.3 * needs_component + 0.2 * norm_compliance
        combined = np.exp(combined) / np.exp(combined).sum()  # Softmax

        motivation = MotivationVector(
            emotion_component=emotion_component,
            needs_component=needs_component,
            norm_component=norm_compliance,
            combined=combined
        )
        self.current_motivation = motivation
        return motivation

    def step(
        self,
        observation: np.ndarray,
        cultural_params: CulturalParameters,
        action: Optional[np.ndarray] = None
    ) -> Tuple[HiddenState, EmotionState, MotivationVector]:
        """Single cognition step.

        Pipeline:
            1. Update beliefs from observation
            2. Predict emotion
            3. Compute motivation

        Args:
            observation: New observation (32,)
            cultural_params: Cultural parameters
            action: Optional action taken

        Returns:
            Tuple of (hidden_state, emotion, motivation)
        """
        # Update beliefs
        hidden_state = self.update_beliefs(observation, cultural_params, action)

        # Predict emotion
        emotion = self.predict_emotion(hidden_state, cultural_params)

        # Compute motivation
        motivation = self.compute_motivation_vector(
            emotion, None, None, cultural_params
        )

        return hidden_state, emotion, motivation


def create_cognition_system(
    hidden_dim: int = 10,
    obs_dim: int = 32,
    num_particles: int = 100,
    inference_method: str = "particle_filter"
) -> CognitionSystem:
    """Factory function to create cognition system.

    Args:
        hidden_dim: Hidden state dimension (10)
        obs_dim: Observation dimension (32)
        num_particles: Number of particles for filtering (100)
        inference_method: "particle_filter" or "svi"

    Returns:
        Configured CognitionSystem instance
    """
    return CognitionSystem(
        hidden_dim=hidden_dim,
        obs_dim=obs_dim,
        num_particles=num_particles,
        inference_method=inference_method
    )