"""Shared type definitions for HELGA core modules.

This module contains all dataclasses and enums used across the HELGA system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np


class MemoryDepth(Enum):
    """Memory retrieval depth levels."""
    WORKING = "working"       # Current context only
    EPISODIC = "episodic"     # Recent experiences
    SEMANTIC = "semantic"     # Long-term knowledge


class VerificationLayer(Enum):
    """Verification layer types."""
    PRE_ACTION = "pre_action"           # Single step validation
    PLAN_EXECUTION = "plan_execution"   # Plan execution validation
    GOAL_ALIGNMENT = "goal_alignment"   # Goal progress validation
    VALUE_CONSISTENCY = "value_consistency"  # Value norm validation


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ObservationInput:
    """Input to perception system.

    Attributes:
        environment: Environment state vector E_t (dim=10)
        event_stream: Event stream embedding V_t (dim=64)
        other_states: Other agent/object states S_t_other (dim=10)

    Formula: Input = [E_t; V_t; S_t_other] concatenated
    """
    environment: np.ndarray      # shape: (10,)
    event_stream: np.ndarray    # shape: (64,)
    other_states: np.ndarray   # shape: (10,)


@dataclass
class PerceptionOutput:
    """Output from perception system.

    Attributes:
        observation: Attended observation vector O_t (dim=32)
        attention_weights: Attention weights per head

    Formula: O_t = MHAttention([E_t; V_t; S_t_other])
    """
    observation: np.ndarray       # shape: (32,)
    attention_weights: np.ndarray  # shape: (num_heads, seq_len)


@dataclass
class HiddenState:
    """Hidden mental state Z_t (dim=10).

    Attributes:
        intent: Intent components (dim=3)
        social: Social relation components (dim=5)
        causal: Physical causality understanding (dim=2)

    Total dim = 3 + 5 + 2 = 10

    Formula: Z_t = [z_intent(3); z_social(5); z_causal(2)]
    """
    intent: np.ndarray           # shape: (3,)
    social: np.ndarray           # shape: (5,)
    causal: np.ndarray          # shape: (2,)

    def to_vector(self) -> np.ndarray:
        """Convert to single vector."""
        return np.concatenate([self.intent, self.social, self.causal])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "HiddenState":
        """Create from vector."""
        return cls(
            intent=vec[:3],
            social=vec[3:8],
            causal=vec[8:10]
        )


@dataclass
class EmotionState:
    """Emotion state from valence function.

    Attributes:
        valence: Emotional valence (-1 to 1)
        arousal: Activation level (0 to 1)
        dominant_emotion: Name of dominant emotion
    """
    valence: float
    arousal: float
    dominant_emotion: str


@dataclass
class MotivationVector:
    """Motivation vector combining emotion, needs, and norms.

    Attributes:
        emotion_component: Emotion contribution
        needs_component: Physiological needs contribution
        norm_component: Norm compliance contribution
        combined: Final motivation vector (dim=10)
    """
    emotion_component: np.ndarray
    needs_component: np.ndarray
    norm_component: np.ndarray
    combined: np.ndarray  # shape: (10,)


@dataclass
class Action:
    """Action in the action space.

    Attributes:
        id: Action index (0-19)
        name: Action description
        parameters: Action-specific parameters
    """
    id: int
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UtilityDecomposition:
    """Decomposition of expected utility.

    Formula: U(a, Z) = w_v^T * F_value(a) + λ_norm * Comply(a, N) - η * Complexity(a)

    Attributes:
        value_component: Value achievement score
        norm_component: Norm compliance score
        complexity_penalty: Cognitive cost
        total: Total expected utility
    """
    value_component: float
    norm_component: float
    complexity_penalty: float
    total: float


@dataclass
class ReasoningStep:
    """Single step in reasoning chain.

    Attributes:
        stage: Stage name (e.g., "action_generation", "value_check")
        input: Stage input data
        output: Stage output data
        rationale: Reasoning rationale
        timestamp: Step timestamp
    """
    stage: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    rationale: str
    timestamp: float


@dataclass
class ReasoningChain:
    """Complete reasoning chain for a decision.

    Attributes:
        steps: List of reasoning steps
        final_decision: Final decision description
        alternatives_considered: Alternative actions considered
    """
    steps: List[ReasoningStep]
    final_decision: str
    alternatives_considered: List[str]


class IterationStage(Enum):
    """决策迭代阶段类型"""
    INFORMATION_GATHERING = "information_gathering"  # 信息收集
    OPTIONS_GENERATION = "options_generation"        # 选项生成
    EVALUATION = "evaluation"                        # 方案评估
    CONTINGENCY_CHECK = "contingency_check"          # 突发情况检查
    FINAL_DECISION = "final_decision"                # 最终决策


@dataclass
class RouteDetails:
    """路线详情 - 具体可执行的路线信息

    Attributes:
        route_name: 路线名称（如"G2京沪高速"）
        route_type: 路线类型（"高速" / "国道" / "省道" / "混合"）
        waypoints: 途经点列表
        distance_km: 距离（公里）
        estimated_time_minutes: 预计时间（分钟）
        toll_cost: 通行费（元）
        has_congestion: 是否拥堵
        congestion_probability: 拥堵概率 0-1
        traffic_status: 交通状态（"畅通" / "缓慢" / "拥堵"）
    """
    route_name: str
    route_type: str = "高速"
    waypoints: List[str] = field(default_factory=list)
    distance_km: float = 0.0
    estimated_time_minutes: int = 0
    toll_cost: float = 0.0
    has_congestion: bool = False
    congestion_probability: float = 0.0
    traffic_status: str = "畅通"


@dataclass
class DecisionPlan:
    """可执行的完整方案 - 替代 SceneDecision

    Attributes:
        plan_id: 方案唯一标识
        name: 方案名称（如"绕行G2京沪高速"）
        description: 方案描述
        route_details: 路线详情（可选，用于交通场景）
        estimated_duration: 预计时长（如"40分钟"）
        target_location: 目标地点（路线相关场景用）
        risk_factors: 风险因素及程度 {具体风险: 0.0-1.0}
        overall_risk_level: 综合风险等级 (low/medium/high)
        atomic_actions: 包含的原子行动序列
        value_orientation: 价值取向向量
        success_probability: 方案成功率估算
        iteration_created: 方案在哪次迭代生成
        reasoning: 本方案的选择理由
        search_queries: 本方案涉及的搜索查询列表
    """
    plan_id: str
    name: str
    description: str
    estimated_duration: str
    target_location: Optional[str]
    risk_factors: Dict[str, float]
    overall_risk_level: str
    atomic_actions: List[str]
    value_orientation: Dict[str, float]
    success_probability: float
    iteration_created: int
    route_details: Optional[RouteDetails] = None
    reasoning: str = ""
    search_queries: List[str] = field(default_factory=list)


@dataclass
class DecisionIteration:
    """单轮决策迭代 - 模拟人类思考-决策过程

    Attributes:
        iteration_id: 迭代序号 (1, 2, 3...)
        stage: 阶段类型
        stage_description: 阶段描述
        search_query: 本轮搜索查询（如适用）
        search_result: 搜索结果摘要
        input_information: 输入信息
        reasoning_result: 推理结果
        beliefs_updated: 迭代后更新的信念
        plans_considered: 本轮考虑的方案ID列表
        action_taken: 本轮采取的行动
        confidence_delta: 置信度变化
        timestamp: 时间戳
    """
    iteration_id: int
    stage: str
    stage_description: str
    search_query: str = ""
    search_result: str = ""
    input_information: Dict[str, Any] = field(default_factory=dict)
    reasoning_result: str = ""
    beliefs_updated: Dict[str, float] = field(default_factory=dict)
    plans_considered: List[str] = field(default_factory=list)
    action_taken: str = ""
    confidence_delta: float = 0.0
    timestamp: float = 0.0


@dataclass
class PlanProbabilityDistribution:
    """方案级决策输出 - 替代 DecisionOutput 的顶层结构

    Attributes:
        plans: 所有候选方案
        probabilities: 对应概率（和为1.0）
        recommended_plan_id: 推荐方案ID
        confidence: 推荐置信度
        reasoning_chain: 完整推理链
        decision_iterations: 迭代决策过程
    """
    plans: List[DecisionPlan]
    probabilities: List[float]
    recommended_plan_id: str
    confidence: float
    reasoning_chain: ReasoningChain
    decision_iterations: List[DecisionIteration]


# Legacy types - kept for backward compatibility with existing modules
@dataclass
class SceneDecision:
    """Scene-level decision aggregating atomic actions.

    Deprecated: Use DecisionPlan instead.
    """
    id: str
    name: str
    description: str
    atomic_actions: List[str]
    value_orientation: Dict[str, float]
    estimated_outcome: Dict[str, Any]


@dataclass
class SceneDecisionOutput:
    """Output from decision system - scene-level decisions with probabilities.

    Deprecated: Use PlanProbabilityDistribution instead.
    """
    scene_decisions: List[SceneDecision]
    probabilities: List[float]
    reasoning_chain: ReasoningChain
    recommended_action: str
    context_summary: Dict[str, Any]


@dataclass
class DecisionOutput:
    """Output from decision system.

    Attributes:
        optimal_action: Selected action
        utility: Utility decomposition
        attention_weights: Attention on hidden state components
        alternatives: Other considered actions with utilities
        reasoning_chain: Full reasoning chain
        confidence: Probability of optimal action (0-1)
        top5_probabilities: Top 5 actions with probabilities [(action_id, name, prob), ...]
        scene_output: Scene-level decision output (if available)
        plan_distribution: Plan-level decision output (if available) - NEW
    """
    optimal_action: Action
    utility: UtilityDecomposition
    attention_weights: np.ndarray
    alternatives: List[Tuple[Action, float]]
    reasoning_chain: ReasoningChain
    confidence: float = 0.0
    top5_probabilities: List[Tuple[int, str, float]] = field(default_factory=list)
    scene_output: Optional[SceneDecisionOutput] = None
    plan_distribution: Optional[PlanProbabilityDistribution] = None


@dataclass
class EnvironmentState:
    """Environment state.

    Formula: E_{t+1} = f_env(E_t, a_t) + ε

    Attributes:
        state: Environment vector
        timestamp: State timestamp
    """
    state: np.ndarray  # shape: (10,)
    timestamp: float


@dataclass
class ActionResult:
    """Result of action execution.

    Attributes:
        new_environment: Updated environment state
        prediction_error: Error in predicting next state
        success: Whether action succeeded
    """
    new_environment: EnvironmentState
    prediction_error: float
    success: bool


@dataclass
class ValidationIssue:
    """Issue found during validation.

    Attributes:
        severity: Issue severity
        description: Issue description
        location: Where the issue occurred
        fix_suggestion: Suggested fix
    """
    severity: ValidationSeverity
    description: str
    location: str
    fix_suggestion: str


@dataclass
class ValidationResult:
    """Result from validation.

    Attributes:
        passed: Whether validation passed
        layer: Which verification layer
        metric_value: Computed metric value
        threshold: Threshold for passing
        issues: Issues found
        recommendations: Suggested fixes
        explanation: Validation explanation
    """
    passed: bool
    layer: VerificationLayer
    metric_value: float
    threshold: float
    issues: List[ValidationIssue]
    recommendations: List[str]
    explanation: str


@dataclass
class Experience:
    """Experience tuple for episodic memory.

    Attributes:
        state: Environment state
        action: Action taken
        reward: Reward received
        next_state: Next state
        done: Whether episode ended
    """
    state: np.ndarray
    action: Action
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class AgentConfig:
    """Configuration for HELGA agent.

    Attributes:
        config_path: Path to config file
        device: Device for computation (cpu/cuda)
    """
    config_path: str = "config/default.yaml"
    device: str = "cpu"


@dataclass
class TimeVariables:
    """Time dimension of environment."""
    current_time: float
    deadline_pressure: float
    time_horizon: float
    duration: float


@dataclass
class SpaceVariables:
    """Space dimension of environment."""
    location_type: float
    physical_context: float
    proximity: float


@dataclass
class SocialVariables:
    """Social dimension of environment."""
    relationship_dynamics: float
    power_distance: float
    group_norm: float
    cultural_context: float


@dataclass
class InfoVariables:
    """Information dimension of environment."""
    uncertainty: float
    confidence: float
    information_quality: float
    missing_key_info: bool


@dataclass
class EnvironmentVariables:
    """Nested environment variables (time/space/social/info)."""
    time: TimeVariables
    space: SpaceVariables
    social: SocialVariables
    info: InfoVariables