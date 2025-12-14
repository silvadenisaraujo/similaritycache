from typing import Any, Optional

class Node:
    """Node for doubly linked list representing a cache entry."""

    def __init__(
        self, key: Any, value: Any, embedding: Any, expires_at: Optional[float] = None
    ):
        self.key = key
        self.value = value
        self.embedding = embedding
        self.expires_at = expires_at
        self.prev: Optional["Node"] = None
        self.next: Optional["Node"] = None