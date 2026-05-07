"""HELGA - Humanistic Environmental Learning and Generating Agent

A cognitive agent system that simulates human "observation → thinking → decision → action"闭环.
"""

__version__ = "0.1.0"

from .core import (
    # Types
    ObservationInput,
    PerceptionOutput,
    HiddenState,
    EmotionState,
    MotivationVector,
    Action,
    UtilityDecomposition,
    DecisionOutput,
    EnvironmentState,
    ActionResult,
    ReasoningChain,
    ReasoningStep,
    ValidationResult,
    # Culture
    CulturalParameters,
    SchwartzValues,
    SocialNormsMatrix,
    AffectSchema,
    # Systems
    PerceptionSystem,
    CognitionSystem,
    ActionSystem,
)

__all__ = [
    # Types
    "ObservationInput",
    "PerceptionOutput",
    "HiddenState",
    "EmotionState",
    "MotivationVector",
    "Action",
    "UtilityDecomposition",
    "DecisionOutput",
    "EnvironmentState",
    "ActionResult",
    "ReasoningChain",
    "ReasoningStep",
    "ValidationResult",
    # Culture
    "CulturalParameters",
    "SchwartzValues",
    "SocialNormsMatrix",
    "AffectSchema",
    # Systems
    "PerceptionSystem",
    "CognitionSystem",
    "ActionSystem",
    # Validation
    "run_all_validations",
    "print_validation_summary",
]

# Import validation functions
from .validation import run_all_validations, print_validation_summary