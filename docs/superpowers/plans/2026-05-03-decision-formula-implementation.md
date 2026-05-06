# 通用决策公式实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 HELGA 的通用决策数学公式，输出层次动作概率分布，支持多轮对话和强化学习更新

**Architecture:** 基于期望效用 + 注意力机制 + 分层文化先验的混合概率模型。核心公式可解释，部分参数可学习。决策分两层：意图层（intent）和实现层（action）。

**Tech Stack:** Python 3.10+, PyTorch, NumPy

---

## 文件结构

```
core/
├── decision.py              # 现有决策系统 → 扩展为概率分布输出
├── types.py                 # 现有类型定义 → 新增环境变量类型
├── culture.py               # 现有文化参数 → 扩展分层先验
└── NEW: environment.py      # 环境变量编码器（嵌套结构）
└── NEW: ask_mechanism.py    # 主动询问机制
└── NEW: rl_update.py        # 强化学习更新

memory/
├── working.py               # 现有工作记忆 → 扩展为决策记忆
└── episodic.py              # 现有情景记忆

tests/
└── test_decision_formula.py # 新测试文件
```

---

## Task 1: 环境变量类型定义

**Files:**
- Modify: `core/types.py:1-50`
- Create: `tests/test_environment_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_environment_types.py
import numpy as np
from core.types import EnvironmentVariables, TimeVariables, SpaceVariables, SocialVariables, InfoVariables

def test_environment_variables_creation():
    """Test nested environment variables structure."""
    env = EnvironmentVariables(
        time=TimeVariables(current_time=0.5, deadline_pressure=0.3, time_horizon=0.7, duration=0.4),
        space=SpaceVariables(location_type=0.2, physical_context=0.1, proximity=0.3),
        social=SocialVariables(relationship_dynamics=0.5, power_distance=0.6, group_norm=0.4, cultural_context=0.3),
        info=InfoVariables(uncertainty=0.5, confidence=0.7, information_quality=0.6, missing_key_info=False)
    )
    assert env.time.current_time == 0.5
    assert env.info.missing_key_info == False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_environment_types.py -v`
Expected: FAIL - ImportError or TypeError

- [ ] **Step 3: Write minimal implementation**

```python
# core/types.py - Add new types after line 299

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_environment_types.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/types.py tests/test_environment_types.py
git commit -m "feat: add nested environment variables types"
```

---

## Task 2: 环境变量编码器

**Files:**
- Create: `core/environment.py`
- Test: `tests/test_environment.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_environment.py
import torch
from core.environment import EnvironmentEncoder

def test_environment_encoder_forward():
    """Test environment encoding produces correct shape."""
    encoder = EnvironmentEncoder()
    # Create mock input
    time_tensor = torch.randn(4)
    space_tensor = torch.randn(4)
    social_tensor = torch.randn(4)
    info_tensor = torch.randn(4)
    encoded = encoder(time_tensor, space_tensor, social_tensor, info_tensor)
    assert encoded.shape == (encoder.output_dim,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_environment.py -v`
Expected: FAIL - module not found

- [ ] **Step 3: Write minimal implementation**

```python
# core/environment.py
"""Environment variable encoder for nested time/space/social/info dimensions."""

import torch
import torch.nn as nn
from typing import Tuple

class EnvironmentEncoder(nn.Module):
    """Encode nested environment variables into a unified vector.

    Architecture:
        E = [ Enc(E_time); Enc(E_space); Enc(E_social); Enc(E_info) ] -> R^d

    Attributes:
        time_dim: Input dim for time variables (4)
        space_dim: Input dim for space variables (3)
        social_dim: Input dim for social variables (4)
        info_dim: Input dim for info variables (4)
        hidden_dim: Hidden layer dim
        output_dim: Final encoded vector dim
    """

    def __init__(
        self,
        time_dim: int = 4,
        space_dim: int = 3,
        social_dim: int = 4,
        info_dim: int = 4,
        hidden_dim: int = 32,
        output_dim: int = 32
    ):
        super().__init__()
        self.time_dim = time_dim
        self.space_dim = space_dim
        self.social_dim = social_dim
        self.info_dim = info_dim
        self.output_dim = output_dim

        # Separate encoders per dimension
        self.time_encoder = nn.Linear(time_dim, hidden_dim)
        self.space_encoder = nn.Linear(space_dim, hidden_dim)
        self.social_encoder = nn.Linear(social_dim, hidden_dim)
        self.info_encoder = nn.Linear(info_dim, hidden_dim)

        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 4, output_dim)

    def forward(
        self,
        time: torch.Tensor,
        space: torch.Tensor,
        social: torch.Tensor,
        info: torch.Tensor
    ) -> torch.Tensor:
        """Encode environment variables.

        Args:
            time: Time variables (batch, 4)
            space: Space variables (batch, 3)
            social: Social variables (batch, 4)
            info: Info variables (batch, 4)

        Returns:
            Encoded environment vector (batch, output_dim)
        """
        time_enc = torch.relu(self.time_encoder(time))
        space_enc = torch.relu(self.space_encoder(space))
        social_enc = torch.relu(self.social_encoder(social))
        info_enc = torch.relu(self.info_encoder(info))

        # Concatenate and fuse
        combined = torch.cat([time_enc, space_enc, social_enc, info_enc], dim=-1)
        output = self.fusion(combined)
        return output
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_environment.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/environment.py tests/test_environment.py
git commit -m "feat: add environment variable encoder"
```

---

## Task 3: 扩展决策系统支持概率分布输出

**Files:**
- Modify: `core/decision.py:206-313` (MultiObjectiveDecisionMaker class)
- Create: `tests/test_probability_decision.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_probability_decision.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_probability_decision.py -v`
Expected: FAIL - HierarchicalDecisionMaker not defined

- [ ] **Step 3: Write minimal implementation**

```python
# core/decision.py - Add new class after MultiObjectiveDecisionMaker (around line 313)

class HierarchicalDecisionMaker(nn.Module):
    """Hierarchical decision maker with probability distribution output.

    Formula:
        P(a | E, M) = Softmax( β · EU(a) + γ · Attention(E, M) ) ⊙ P_cultural(a | Z)

    Attributes:
        beta: Utility weight coefficient
        gamma: Attention weight coefficient
        value_dim: Value dimension (10)
        action_dim: Action dimension (20)
        env_encoder: Environment encoder
        attention: Attention mechanism
    """

    def __init__(
        self,
        beta: float = 1.0,
        gamma: float = 0.5,
        value_dim: int = 10,
        action_dim: int = 20,
        env_dim: int = 32
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma

        # Value extractor (reuse from existing)
        self.value_extractor = ValueFeatureExtractor(action_dim, value_dim)

        # Environment encoder (will be connected later)
        self.env_encoder = None  # Set externally

        # Attention mechanism
        self.attention_query = nn.Linear(env_dim, 32)
        self.attention_keys = nn.Linear(env_dim, 32)

        # Learnable weights
        self.value_weights = nn.Parameter(torch.ones(value_dim) / value_dim)
        self.norm_weight = nn.Parameter(torch.tensor(0.3))
        self.complexity_weight = nn.Parameter(torch.tensor(0.1))
        self.risk_weight = nn.Parameter(torch.tensor(0.1))

    def compute_expected_utility(self, action_ids: torch.Tensor) -> torch.Tensor:
        """Compute EU(a) for actions.

        EU(a) = Σ_k w_k · value_k(a) - λ · norm_cost - η · complexity - ψ · risk
        """
        # Value component
        action_values = self.value_extractor.action_value_features[action_ids]
        value_component = torch.sum(action_values * self.value_weights, dim=-1)

        # Norm component (simplified: uniform compliance)
        norm_component = torch.ones_like(value_component) * self.norm_weight * 0.8

        # Complexity penalty (simplified)
        complexity_penalty = torch.ones_like(value_component) * self.complexity_weight * 0.5

        # Risk penalty (simplified)
        risk_penalty = torch.ones_like(value_component) * self.risk_weight * 0.2

        eu = value_component + norm_component - complexity_penalty - risk_penalty
        return eu

    def compute_attention(self, env_encoding: torch.Tensor) -> torch.Tensor:
        """Compute attention score over environment features."""
        query = self.attention_query(env_encoding)
        # Simplified: attention is just a weighted sum
        return torch.sigmoid(query).sum() * self.gamma

    def forward(
        self,
        hidden_state: torch.Tensor,
        env_encoding: torch.Tensor,
        cultural_prior: torch.Tensor
    ) -> torch.Tensor:
        """Output probability distribution over actions.

        Args:
            hidden_state: Hidden state vector (10,)
            env_encoding: Environment encoding (32,)
            cultural_prior: Cultural prior probabilities (20,)

        Returns:
            Probability distribution over actions (20,)
        """
        # Compute EU for all actions
        action_ids = torch.arange(20)
        eu = self.compute_expected_utility(action_ids)

        # Compute attention
        attention_score = self.compute_attention(env_encoding)

        # Combine EU and attention
        logits = self.beta * eu + attention_score

        # Apply cultural prior (element-wise multiply)
        logits = logits + torch.log(cultural_prior + 1e-8)

        # Softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)

        return probs

    def get_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of probability distribution."""
        return -torch.sum(probs * torch.log(probs + 1e-8))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_probability_decision.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/decision.py tests/test_probability_decision.py
git commit -m "feat: add hierarchical decision maker with probability output"
```

---

## Task 4: 主动询问机制

**Files:**
- Create: `core/ask_mechanism.py`
- Test: `tests/test_ask_mechanism.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ask_mechanism.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ask_mechanism.py -v`
Expected: FAIL - module not found

- [ ] **Step 3: Write minimal implementation**

```python
# core/ask_mechanism.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ask_mechanism.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/ask_mechanism.py tests/test_ask_mechanism.py
git commit -m "feat: add active ask mechanism for info gathering"
```

---

## Task 5: 强化学习更新机制

**Files:**
- Create: `core/rl_update.py`
- Test: `tests/test_rl_update.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rl_update.py
import torch
from core.rl_update importRLUpdate, UserFeedback

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rl_update.py -v`
Expected: FAIL - module not found

- [ ] **Step 3: Write minimal implementation**

```python
# core/rl_update.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rl_update.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/rl_update.py tests/test_rl_update.py
git commit -m "feat: add RL update mechanism with user feedback"
```

---

## Task 6: 集成测试 - 完整决策流程

**Files:**
- Create: `tests/test_integrated_decision.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_integrated_decision.py
import torch
from core.decision import HierarchicalDecisionMaker
from core.environment import EnvironmentEncoder
from core.ask_mechanism import AskTrigger
from core.rl_update import RLUpdate, UserFeedback

def test_full_decision_loop():
    """Test complete decision loop: encode -> decide -> ask -> update."""
    # Setup
    decision_model = HierarchicalDecisionMaker()
    env_encoder = EnvironmentEncoder()

    # Mock environment input
    time_input = torch.randn(4)
    space_input = torch.randn(3)
    social_input = torch.randn(4)
    info_input = torch.randn(4)

    env_encoding = env_encoder(time_input, space_input, social_input, info_input)
    hidden_state = torch.randn(10)
    cultural_prior = torch.ones(20)

    # Decision
    probs = decision_model(hidden_state, env_encoding, cultural_prior)

    # Check ask trigger
    should_ask, entropy = AskTrigger.should_trigger(probs, threshold=1.0)
    assert isinstance(should_ask, bool)

    # RL update
    updater = RLUpdate()
    feedback = UserFeedback(action_idx=0, score=4, rationale="这个决策比较好")
    loss, breakdown = updater.compute_total_loss(probs, feedback)

    assert loss.item() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_integrated_decision.py -v`
Expected: FAIL

- [ ] **Step 3: Fix integration issues (expected - iterative fix)**

Run tests, fix import/signature issues until passing.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_integrated_decision.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_integrated_decision.py
git commit -m "test: add integrated decision flow test"
```

---

## Task 7: 更新 HELGA Agent 入口集成

**Files:**
- Modify: `main.py`
- Modify: `core/action.py`

- [ ] **Step 1: Update make_decision interface**

```python
# In core/action.py, add integration point:
# Replace single action output with probability distribution
# Connect AskTrigger to decision loop
# Connect RLUpdate for user feedback handling
```

- [ ] **Step 2: Add user feedback handler**

```python
# In main.py or core/action.py:
def handle_user_feedback(feedback: UserFeedback, decision_model, env_encoder, cultural_prior):
    """Handle user feedback and update model."""
    probs = decision_model(hidden_state, env_encoding, cultural_prior)
    updater = RLUpdate()
    loss, _ = updater.compute_total_loss(probs, feedback)
    # Backprop and update
    loss.backward()
```

- [ ] **Step 3: Test the integrated flow**

Run: `python main.py --scenario colleague_late`

---

## 执行选项

**Plan complete and saved to `docs/superpowers/plans/2026-05-03-decision-formula-implementation.md`.**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**