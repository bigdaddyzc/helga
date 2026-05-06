"""Working memory module for HELGA.

Working memory maintains current context with limited capacity (~10 items).
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import time


@dataclass
class MemoryItem:
    """Single item in working memory."""
    key: str
    value: Any
    timestamp: float
    importance: float = 0.5  # 0-1 importance score


class WorkingMemory:
    """Working memory with limited capacity.

    Maintains ~10 items relevant to current task context.
    Uses importance-based eviction when capacity is exceeded.

    Formula: WM_t = [O_t, Z_t, M_t, A_t] concatenated
    """

    def __init__(self, capacity: int = 10):
        """Initialize working memory.

        Args:
            capacity: Maximum number of items (default: 10)
        """
        self.capacity = capacity
        self.items: List[MemoryItem] = []
        self.access_counts: Dict[str, int] = {}  # Track access frequency

    def store(self, key: str, value: Any, importance: float = 0.5) -> None:
        """Store item in working memory.

        Args:
            key: Item key
            value: Item value
            importance: Importance score (0-1)
        """
        # Check if key already exists
        existing_idx = None
        for i, item in enumerate(self.items):
            if item.key == key:
                existing_idx = i
                break

        # Update existing or add new
        if existing_idx is not None:
            self.items[existing_idx].value = value
            self.items[existing_idx].importance = importance
            self.items[existing_idx].timestamp = time.time()
        else:
            # Add new item
            self.items.append(MemoryItem(
                key=key,
                value=value,
                timestamp=time.time(),
                importance=importance
            ))

            # Evict if over capacity
            if len(self.items) > self.capacity:
                self._evict()

        # Update access count
        self.access_counts[key] = self.access_counts.get(key, 0) + 1

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve item from working memory.

        Args:
            key: Item key

        Returns:
            Item value or None if not found
        """
        for item in self.items:
            if item.key == key:
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                return item.value
        return None

    def retrieve_by_pattern(self, pattern: str) -> List[Tuple[str, Any]]:
        """Retrieve items matching key pattern.

        Args:
            pattern: Key pattern to match

        Returns:
            List of (key, value) tuples
        """
        results = []
        for item in self.items:
            if pattern.lower() in item.key.lower():
                results.append((item.key, item.value))
        return results

    def _evict(self) -> None:
        """Evict least important item.

        Uses combination of importance and access frequency.
        """
        if not self.items:
            return

        # Compute scores: importance * log(access_count + 1)
        scores = []
        for item in self.items:
            access_count = self.access_counts.get(item.key, 1)
            score = item.importance * np.log(access_count + 1)
            scores.append(score)

        # Evict lowest scoring item
        evict_idx = np.argmin(scores)
        evicted_key = self.items[evict_idx].key
        self.items.pop(evict_idx)

        # Clean up access count
        if evicted_key in self.access_counts:
            del self.access_counts[evicted_key]

    def clear(self) -> None:
        """Clear all items from working memory."""
        self.items = []
        self.access_counts = {}

    def get_all(self) -> List[Tuple[str, Any]]:
        """Get all items as list of (key, value) tuples."""
        return [(item.key, item.value) for item in self.items]

    def get_recent(self, n: int = 5) -> List[Tuple[str, Any]]:
        """Get n most recent items.

        Args:
            n: Number of items to return

        Returns:
            List of (key, value) tuples
        """
        sorted_items = sorted(self.items, key=lambda x: x.timestamp, reverse=True)
        return [(item.key, item.value) for item in sorted_items[:n]]

    def size(self) -> int:
        """Get current size of working memory."""
        return len(self.items)

    def __repr__(self) -> str:
        return f"WorkingMemory(capacity={self.capacity}, size={len(self.items)})"


def create_working_memory(capacity: int = 10) -> WorkingMemory:
    """Factory function to create working memory.

    Args:
        capacity: Maximum items (default: 10)

    Returns:
        WorkingMemory instance
    """
    return WorkingMemory(capacity=capacity)