"""Cultural parameters module for HELGA.

This module implements Schwartz's theory of basic human values,
social norms matrix, and affect schemas for cultural cognition.

Schwartz Values (dim=10):
1. Self-Transcendence - Universalism, Benevolence
2. Openness to Change - Self-Direction, Stimulation
3. Benevolence - Social, warm
4. Conformity - Rules, norms
5. Security - Safety, harmony
6. Achievement - Success, competence
7. Hedonism - Pleasure, enjoyment
8. Stimulation - Excitement, novelty
9. Self-Direction - Independence, freedom
10. Universalism - Understanding, tolerance

Reference: Schwartz, S. H. (2012). Overview of the EVP.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import yaml
from pathlib import Path


@dataclass
class SchwartzValues:
    """Schwartz values vector (dim=10).

    Attributes:
        values: Value importance weights (0-1)

    Formula: v = [v1, v2, ..., v10]

    Dimensions:
    - v[0]: Self-Transcendence (universalism + benevolence)
    - v[1]: Openness to Change (self-direction + stimulation)
    - v[2]: Benevolence (social, warm)
    - v[3]: Conformity (rules, norms)
    - v[4]: Security (safety, harmony)
    - v[5]: Achievement (success, competence)
    - v[6]: Hedonism (pleasure, enjoyment)
    - v[7]: Stimulation (excitement, novelty)
    - v[8]: Self-Direction (independence, freedom)
    - v[9]: Universalism (understanding, tolerance)
    """
    values: np.ndarray  # shape: (10,)

    def __post_init__(self):
        if len(self.values) != 10:
            raise ValueError(f"SchwartzValues must have 10 dimensions, got {len(self.values)}")
        if np.any(self.values < 0) or np.any(self.values > 1):
            raise ValueError("Values must be in range [0, 1]")

    def get_value(self, dimension: int) -> float:
        """Get value for a specific dimension."""
        return self.values[dimension]

    def weighted_sum(self, features: np.ndarray) -> float:
        """Compute weighted sum of features.

        Formula: result = Σ_i v[i] * features[i]
        """
        return np.dot(self.values, features)

    @classmethod
    def from_list(cls, values: List[float]) -> "SchwartzValues":
        """Create from list."""
        return cls(values=np.array(values, dtype=np.float32))

    @classmethod
    def default(cls) -> "SchwartzValues":
        """Create with default values."""
        return cls(values=np.array([0.8, 0.6, 0.7, 0.5, 0.9, 0.4, 0.6, 0.8, 0.7, 0.5], dtype=np.float32))


@dataclass
class SocialNormsMatrix:
    """Social norms compliance matrix (dim=10x20).

    Attributes:
        matrix: Norm compliance scores

    Formula: N[i,j] = compliance_norm(i,j)
    - Row i: norm i
    - Column j: action j
    - Value: 0.0 (forbidden) to 1.0 (completely acceptable)

    Norms (rows):
    0: No harm
    1: Respect property
    2: Transparent actions
    3: Accountable
    4: Fairness
    5: Loyalty
    6: Intimacy maintenance
    7: Role fulfillment
    8: Health maintenance
    9: Resource preservation
    """
    matrix: np.ndarray  # shape: (10, 20)

    def __post_init__(self):
        if self.matrix.shape != (10, 20):
            raise ValueError(f"SocialNormsMatrix must be (10, 20), got {self.matrix.shape}")
        if np.any(self.matrix < 0) or np.any(self.matrix > 1):
            raise ValueError("Matrix values must be in range [0, 1]")

    def get_compliance(self, action_id: int) -> np.ndarray:
        """Get compliance scores for an action across all norms.

        Formula: get_compliance(j) = N[:, j]
        """
        return self.matrix[:, action_id]

    def avg_compliance(self, action_id: int) -> float:
        """Compute average compliance for an action.

        Formula: avg_compliance(j) = Σ_i N[i,j] / 10
        """
        return np.mean(self.matrix[:, action_id])

    @classmethod
    def from_matrix(cls, matrix: List[List[float]]) -> "SocialNormsMatrix":
        """Create from 2D list."""
        return cls(matrix=np.array(matrix, dtype=np.float32))

    @classmethod
    def default(cls) -> "SocialNormsMatrix":
        """Create with default values."""
        # Extended matrix (10 norms × 20 actions)
        # Actions 0-13: General/workplace actions
        # Actions 14-19: Traffic/practical actions
        #   14: take_detour, 15: wait_traffic_clear, 16: check_info
        #   17: call_service, 18: cancel_trip, 19: continue_anyway
        default_values = [
            # Norm 0-9: No-harm, Property, Transparency, Accountability, Fairness,
            #           Loyalty, Intimacy, Role, Health, Resource
            [1.0, 0.9, 0.8, 0.7, 0.3, 0.5, 0.6, 0.8, 0.9, 1.0, 0.7, 0.8, 0.6, 0.4, 0.7, 0.5, 0.8, 0.8, 0.4, 0.2],  # 0-13: standard, 14-19: traffic actions
            [0.9, 1.0, 0.9, 0.8, 0.4, 0.6, 0.5, 0.7, 0.8, 0.9, 0.6, 0.7, 0.5, 0.3, 0.7, 0.5, 0.8, 0.8, 0.4, 0.2],
            [0.8, 0.9, 1.0, 0.9, 0.5, 0.7, 0.6, 0.8, 0.7, 0.8, 0.5, 0.6, 0.4, 0.2, 0.6, 0.4, 0.7, 0.7, 0.3, 0.2],
            [0.7, 0.8, 0.9, 1.0, 0.6, 0.8, 0.7, 0.9, 0.6, 0.7, 0.4, 0.5, 0.3, 0.1, 0.5, 0.3, 0.6, 0.6, 0.3, 0.1],
            [0.3, 0.4, 0.5, 0.6, 1.0, 0.8, 0.7, 0.5, 0.4, 0.3, 0.8, 0.9, 0.7, 0.5, 0.8, 0.9, 0.7, 0.8, 0.9, 0.3],  # Security: high for detour/wait/cancel
            [0.5, 0.6, 0.7, 0.8, 0.8, 1.0, 0.9, 0.7, 0.6, 0.5, 0.9, 0.8, 0.6, 0.4, 0.7, 0.6, 0.8, 0.7, 0.4, 0.3],
            [0.6, 0.5, 0.6, 0.7, 0.7, 0.9, 1.0, 0.8, 0.7, 0.6, 0.7, 0.6, 0.5, 0.3, 0.6, 0.5, 0.6, 0.6, 0.3, 0.2],
            [0.8, 0.7, 0.8, 0.9, 0.5, 0.7, 0.8, 1.0, 0.9, 0.8, 0.7, 0.6, 0.4, 0.2, 0.6, 0.4, 0.7, 0.7, 0.3, 0.2],
            [0.9, 0.8, 0.7, 0.6, 0.4, 0.6, 0.7, 0.9, 1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.7, 0.6, 0.8, 0.8, 0.4, 0.2],
            [1.0, 0.9, 0.8, 0.7, 0.3, 0.5, 0.6, 0.8, 0.9, 1.0, 0.7, 0.8, 0.6, 0.4, 0.7, 0.6, 0.8, 0.8, 0.4, 0.2],
        ]
        return cls(matrix=np.array(default_values, dtype=np.float32))


@dataclass
class AffectSchema:
    """Affect schema for emotion encoding (dim=10).

    Attributes:
        schema: Affect weights

    Formula: A = [a1, a2, ..., a10]

    Positive emotions (positive values):
    - a[0]: Joy
    - a[2]: Surprise (positive)
    - a[3]: Gratitude
    - a[4]: Hope
    - a[8]: Interest

    Negative emotions (negative values):
    - a[1]: Anger
    - a[5]: Fear
    - a[6]: Sadness
    - a[7]: Disgust
    - a[9]: Shame
    """
    schema: np.ndarray  # shape: (10,)

    def __post_init__(self):
        if len(self.schema) != 10:
            raise ValueError(f"AffectSchema must have 10 dimensions, got {len(self.schema)}")

    def compute_valence(self, motivation: np.ndarray) -> float:
        """Compute emotional valence from motivation vector.

        Formula: valence = (A · M) / (||A|| * ||M||)

        Args:
            motivation: Motivation vector (dim=10)

        Returns:
            Valence score (-1 to 1)
        """
        norm_a = np.linalg.norm(self.schema)
        norm_m = np.linalg.norm(motivation)
        if norm_a == 0 or norm_m == 0:
            return 0.0
        return np.dot(self.schema, motivation) / (norm_a * norm_m)

    def dominant_emotion(self, motivation: np.ndarray) -> str:
        """Determine dominant emotion based on motivation.

        Returns:
            Name of dominant emotion
        """
        correlations = self.schema * motivation
        dominant_idx = np.argmax(correlations)
        emotions = ["joy", "anger", "surprise", "gratitude", "fear",
                   "sadness", "disgust", "shame", "interest", "contentment"]
        return emotions[dominant_idx]

    @classmethod
    def from_list(cls, schema: List[float]) -> "AffectSchema":
        """Create from list."""
        return cls(schema=np.array(schema, dtype=np.float32))

    @classmethod
    def default(cls) -> "AffectSchema":
        """Create with default values."""
        return cls(schema=np.array([0.5, -0.3, 0.7, 0.4, -0.6, -0.2, -0.5, -0.4, 0.6, 0.3], dtype=np.float32))


@dataclass
class CulturalParameters:
    """Container for all cultural parameters.

    Attributes:
        schwartz_values: Schwartz value vector
        norms_matrix: Social norms compliance matrix
        affect_schema: Affect encoding schema
    """
    schwartz_values: SchwartzValues
    norms_matrix: SocialNormsMatrix
    affect_schema: AffectSchema

    @classmethod
    def from_config(cls, config_path: str) -> "CulturalParameters":
        """Load from YAML config file.

        Args:
            config_path: Path to config YAML file

        Returns:
            CulturalParameters instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Load Schwartz values
        sv_values = config['schwartz_values']['default']
        schwartz_values = SchwartzValues.from_list(sv_values)

        # Load social norms matrix
        norm_matrix = config['social_norms']['matrix']
        norms_matrix = SocialNormsMatrix.from_matrix(norm_matrix)

        # Load affect schema
        affect_values = config['affect_schema']['default']
        affect_schema = AffectSchema.from_list(affect_values)

        return cls(
            schwartz_values=schwartz_values,
            norms_matrix=norms_matrix,
            affect_schema=affect_schema
        )

    @classmethod
    def default(cls) -> "CulturalParameters":
        """Create with default values."""
        return cls(
            schwartz_values=SchwartzValues.default(),
            norms_matrix=SocialNormsMatrix.default(),
            affect_schema=AffectSchema.default()
        )

    def validate(self) -> bool:
        """Validate parameter bounds and consistency.

        Returns:
            True if valid, raises ValueError otherwise
        """
        self.schwartz_values.validate()
        self.norms_matrix.validate()
        self.affect_schema.validate()
        return True

    def update_schwartz_value(self, dimension: int, value: float) -> None:
        """Update a single Schwartz value dimension.

        Args:
            dimension: Dimension index (0-9)
            value: New value (0-1)
        """
        if dimension < 0 or dimension > 9:
            raise ValueError(f"Dimension must be 0-9, got {dimension}")
        if value < 0 or value > 1:
            raise ValueError(f"Value must be 0-1, got {value}")
        self.schwartz_values.values[dimension] = value


@dataclass
class CulturalUpdateResult:
    """Result of cultural parameter update.

    Attributes:
        updated_params: New cultural parameters
        update_magnitude: Magnitude of change
        adaptation_triggered: Whether adaptation was triggered
    """
    updated_params: CulturalParameters
    update_magnitude: float
    adaptation_triggered: bool


def update_cultural_params(
    params: CulturalParameters,
    error_signal: float,
    learning_rate: float = 0.001
) -> CulturalUpdateResult:
    """Update cultural parameters based on prediction error.

    Formula: params_updated = params - lr * gradient(error)

    Args:
        params: Current cultural parameters
        error_signal: Prediction error signal
        learning_rate: Learning rate for adaptation

    Returns:
        Updated parameters and metadata
    """
    # Simple gradient descent on Schwartz values based on error
    gradient = error_signal * learning_rate
    new_values = params.schwartz_values.values - gradient

    # Clamp to valid range
    new_values = np.clip(new_values, 0.0, 1.0)

    new_params = CulturalParameters(
        schwartz_values=SchwartzValues(values=new_values),
        norms_matrix=params.norms_matrix,
        affect_schema=params.affect_schema
    )

    update_magnitude = np.abs(error_signal * learning_rate)
    adaptation_triggered = update_magnitude > 0.001

    return CulturalUpdateResult(
        updated_params=new_params,
        update_magnitude=update_magnitude,
        adaptation_triggered=adaptation_triggered
    )