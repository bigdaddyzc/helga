"""Episodic memory module for HELGA.

Episodic memory stores experience tuples with larger capacity (~100 items).
Used for reflection and learning from past experiences.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import time
from collections import deque

from .types import Experience


@dataclass
class EpisodicEntry:
    """Single episodic memory entry."""
    experience: Experience
    timestamp: float
    context: Dict[str, Any]  # Additional context
    emotional_tags: List[str]  # Emotional labels
    importance: float = 0.5


class EpisodicMemory:
    """Episodic memory for storing experience sequences.

    Maintains ~100 recent experiences in FIFO manner.
    Supports retrieval by similarity and temporal ordering.

    Formula: EM stores (state, action, reward, next_state) tuples
    """

    def __init__(self, capacity: int = 100):
        """Initialize episodic memory.

        Args:
            capacity: Maximum number of experiences (default: 100)
        """
        self.capacity = capacity
        self.entries: deque = deque(maxlen=capacity)

    def store(
        self,
        experience: Experience,
        context: Optional[Dict[str, Any]] = None,
        emotional_tags: Optional[List[str]] = None,
        importance: float = 0.5
    ) -> None:
        """Store experience in episodic memory.

        Args:
            experience: Experience tuple
            context: Optional context dict
            emotional_tags: Optional emotional labels
            importance: Importance score (0-1)
        """
        entry = EpisodicEntry(
            experience=experience,
            timestamp=time.time(),
            context=context or {},
            emotional_tags=emotional_tags or [],
            importance=importance
        )
        self.entries.append(entry)

    def sample(self, batch_size: int = 10) -> List[Experience]:
        """Sample random experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of Experience objects
        """
        if len(self.entries) == 0:
            return []

        # Sample without replacement if possible
        num_samples = min(batch_size, len(self.entries))
        indices = np.random.choice(len(self.entries), size=num_samples, replace=False)

        return [self.entries[i].experience for i in indices]

    def get_recent(self, n: int = 10) -> List[Experience]:
        """Get n most recent experiences.

        Args:
            n: Number of experiences to return

        Returns:
            List of most recent Experience objects
        """
        recent_entries = list(self.entries)[-n:]
        return [entry.experience for entry in recent_entries]

    def get_by_context(self, context_key: str, context_value: Any) -> List[Experience]:
        """Get experiences matching context criteria.

        Args:
            context_key: Context key to match
            context_value: Expected context value

        Returns:
            List of matching Experiences
        """
        results = []
        for entry in self.entries:
            if entry.context.get(context_key) == context_value:
                results.append(entry.experience)
        return results

    def get_by_emotional_tag(self, tag: str) -> List[Experience]:
        """Get experiences with specific emotional tag.

        Args:
            tag: Emotional tag to match

        Returns:
            List of matching Experiences
        """
        results = []
        for entry in self.entries:
            if tag in entry.emotional_tags:
                results.append(entry.experience)
        return results

    def get_temporal_range(
        self,
        start_time: float,
        end_time: float
    ) -> List[Experience]:
        """Get experiences within time range.

        Args:
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            List of Experiences in time range
        """
        results = []
        for entry in self.entries:
            if start_time <= entry.timestamp <= end_time:
                results.append(entry.experience)
        return results

    def get_by_similarity(
        self,
        state: np.ndarray,
        threshold: float = 0.8,
        max_results: int = 10
    ) -> List[Tuple[Experience, float]]:
        """Get experiences similar to given state.

        Args:
            state: State vector to compare
            threshold: Similarity threshold (0-1)
            max_results: Maximum number of results

        Returns:
            List of (Experience, similarity_score) tuples
        """
        results = []
        for entry in self.entries:
            exp_state = entry.experience.state
            # Compute cosine similarity
            similarity = self._cosine_similarity(state, exp_state)
            if similarity >= threshold:
                results.append((entry.experience, similarity))

        # Sort by similarity and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Returns:
            Similarity score (0-1)
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def clear(self) -> None:
        """Clear all entries from episodic memory."""
        self.entries.clear()

    def size(self) -> int:
        """Get current size of episodic memory."""
        return len(self.entries)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of episodic memory.

        Returns:
            Dict with memory statistics
        """
        if not self.entries:
            return {
                "size": 0,
                "oldest_timestamp": None,
                "newest_timestamp": None,
                "avg_importance": 0.0,
                "emotional_tags": []
            }

        timestamps = [e.timestamp for e in self.entries]
        importances = [e.importance for e in self.entries]
        all_tags = [tag for e in self.entries for tag in e.emotional_tags]

        return {
            "size": len(self.entries),
            "capacity": self.capacity,
            "oldest_timestamp": min(timestamps),
            "newest_timestamp": max(timestamps),
            "avg_importance": np.mean(importances),
            "emotional_tags": list(set(all_tags))
        }

    def __repr__(self) -> str:
        return f"EpisodicMemory(capacity={self.capacity}, size={len(self.entries)})"


def create_episodic_memory(capacity: int = 100) -> EpisodicMemory:
    """Factory function to create episodic memory.

    Args:
        capacity: Maximum experiences (default: 100)

    Returns:
        EpisodicMemory instance
    """
    return EpisodicMemory(capacity=capacity)